"""
Imaginary time evolution solver.

Both bosonic and fermionic species are evolved with a pseudo-GPE:

    -ℏ ∂ψ/∂τ = [ -ℏ²/(2m) ∇² + V_ext + V_int[n] - μ ] ψ

where τ = i·t is imaginary time.  After each step the wavefunction is
renormalised so that ∫|ψ|² dV = N, which implicitly determines the
chemical potential μ (Lagrange multiplier for the norm constraint).

Nonlinear interaction terms
───────────────────────────
Bosons  (GPE):
    V_int = g_BB · n_B  +  g_BF · n_F

    where g_BB = 4π ℏ² a_BB / m_B  (3-D),  with appropriate
    dimensional reduction for lower geometries.

Fermions (pseudo-GPE, LDA Thomas-Fermi kinetic pressure):
    V_int = E_F(n_F)  +  g_BF · n_B

    The local Fermi energy in D effective dimensions:

      3D:       E_F = (ℏ²/2m) (6π²/g_s)^{2/3} · n_F^{2/3}
      2D:       E_F = (ℏ²/2m) (4π/g_s)         · n_F
      1D:       E_F = (ℏ²/2m) (π²/3 g_s²)      · n_F²

    For 2D_cyl the geometry is 2-D so we use the 2-D expression.

All quantities in SI.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from ..geometry import Geometry
from ..atoms import HBAR


# ─────────────────────────────────────────────────────────────────────────────
# Local Fermi energy prefactors
# ─────────────────────────────────────────────────────────────────────────────

def fermi_energy_term(
    n_F: np.ndarray,
    mass: float,
    spin_degeneracy: int,
    ndim: int,
) -> np.ndarray:
    """
    Local Fermi energy E_F(n_F) in SI (Joules).

    Parameters
    ----------
    n_F : ndarray
        Local fermion number density  (m^{-D}).
    mass : float
        Fermion mass (kg).
    spin_degeneracy : int
        Number of accessible spin states g_s.
    ndim : int
        Effective dimensionality of the system (1, 2, or 3).
    """
    n = np.maximum(n_F, 0.0)   # guard against tiny negative noise
    prefac = HBAR**2 / (2.0 * mass)

    if ndim == 3:
        # E_F = (ℏ²/2m) (6π²/g_s)^{2/3} n^{2/3}
        coeff = (6.0 * np.pi**2 / spin_degeneracy) ** (2.0 / 3.0)
        return prefac * coeff * n ** (2.0 / 3.0)

    elif ndim == 2:
        # E_F = (ℏ²/2m) (4π/g_s) n
        coeff = 4.0 * np.pi / spin_degeneracy
        return prefac * coeff * n

    elif ndim == 1:
        # E_F = (ℏ²/2m) (π²/3 g_s²) n²   [spin-degenerate 1-D TF]
        coeff = np.pi**2 / (3.0 * spin_degeneracy**2)
        return prefac * coeff * n**2

    else:
        raise ValueError(f"ndim must be 1, 2, or 3; got {ndim}")


# ─────────────────────────────────────────────────────────────────────────────
# Interaction coupling constants with dimensional reduction
# ─────────────────────────────────────────────────────────────────────────────

def coupling_constant(
    a_scatt: float,
    mass: float,
    ndim: int,
    a_ho_transverse: float | None = None,
) -> float:
    """
    Effective coupling constant g in D dimensions (SI).

    3D : g = 4π ℏ² a / m
    2D : g = sqrt(8π) ℏ² a / (m a_⊥)   [CIR reduction, a_⊥ = transverse HO length]
             Falls back to g_3D if a_ho_transverse is None.
    1D : g = -2 ℏ² / (m a_1D)           where a_1D = -(a_⊥² / 2a)(1 - C a/a_⊥)
             For simplicity we use g_1D ≈ 2 ℏ² a / (m a_⊥²) when a_⊥ is given,
             otherwise fall back to the 3-D value scaled by grid spacing.
    """
    g3d = 4.0 * np.pi * HBAR**2 * a_scatt / mass

    if ndim == 3:
        return g3d

    elif ndim == 2:
        if a_ho_transverse is not None and a_ho_transverse > 0:
            return np.sqrt(8.0 * np.pi) * HBAR**2 * a_scatt / (mass * a_ho_transverse)
        return g3d   # fallback

    elif ndim == 1:
        if a_ho_transverse is not None and a_ho_transverse > 0:
            return 2.0 * HBAR**2 * a_scatt / (mass * a_ho_transverse**2)
        return g3d   # fallback

    else:
        raise ValueError(f"ndim must be 1, 2, or 3; got {ndim}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-species imaginary time step
# ─────────────────────────────────────────────────────────────────────────────

def _ite_step(
    psi: np.ndarray,
    H_psi: np.ndarray,
    dt: float,
    N_target: float,
    geo: Geometry,
) -> tuple[np.ndarray, float]:
    """
    One forward-Euler imaginary-time step followed by norm renormalisation.

    Returns (psi_new, mu_eff) where mu_eff is the effective chemical
    potential estimated from the current state.
    """
    psi_new = psi - dt * H_psi

    # Renormalise to enforce ∫|ψ|² dV = N
    norm_sq = geo.norm(psi_new)
    if norm_sq <= 0:
        norm_sq = 1e-30
    psi_new = psi_new * np.sqrt(N_target / norm_sq)

    # Estimate μ via virial:  μ ≈ <H> / N  (H|ψ⟩ / |ψ⟩ averaged)
    mu_eff = geo.integrate(np.conj(psi) * H_psi) / N_target

    return psi_new, float(np.real(mu_eff))


# ─────────────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ITESolver:
    """
    Imaginary-time evolution solver for a coupled Bose-Fermi mixture.

    Parameters
    ----------
    geo : Geometry
    boson_mass, fermion_mass : float   (kg)
    boson_N, fermion_N : float         (target particle numbers)
    V_boson, V_fermion : ndarray       (external potentials, SI)
    g_BB : float                       (boson-boson coupling, SI)
    g_BF : float                       (boson-fermion coupling, SI)
    fermion_spin_degen : int
    dt : float                         (imaginary time step, s)
    max_iter : int
    tol : float                        (relative convergence on densities)
    mixing : float                     (density mixing fraction in [0,1])
    """
    geo: Geometry
    boson_mass: float
    fermion_mass: float
    boson_N: float
    fermion_N: float
    V_boson: np.ndarray
    V_fermion: np.ndarray
    g_BB: float
    g_BF: float
    fermion_spin_degen: int = 1
    dt: float = 1e-5
    max_iter: int = 5000
    tol: float = 1e-6
    mixing: float = 0.5

    # results (filled after solve())
    psi_B: np.ndarray = field(init=False, repr=False)
    psi_F: np.ndarray = field(init=False, repr=False)
    mu_B: float = field(init=False)
    mu_F: float = field(init=False)
    converged: bool = field(init=False)
    residuals: list[float] = field(init=False, default_factory=list)
    iterations: int = field(init=False, default=0)

    def __post_init__(self):
        self._ndim = self.geo.ndim
        self._initialise_wavefunctions()

    # ------------------------------------------------------------------ #

    def _initialise_wavefunctions(self):
        """Seed both wavefunctions as Gaussians centred on the trap."""
        shape = self.geo.shape

        def _gaussian(*grids):
            r2 = sum(g**2 for g in grids)
            # use 1/4 of the box half-extent as width
            w2 = (max(g.max() for g in grids) / 4.0)**2
            return np.exp(-r2 / (2.0 * w2))

        psi0_B = _gaussian(*self.geo.grids)
        psi0_F = _gaussian(*self.geo.grids)

        norm_B = self.geo.norm(psi0_B)
        norm_F = self.geo.norm(psi0_F)

        self.psi_B = psi0_B * np.sqrt(self.boson_N  / norm_B)
        self.psi_F = psi0_F * np.sqrt(self.fermion_N / norm_F)

    # ------------------------------------------------------------------ #

    def _kinetic(self, psi: np.ndarray, mass: float) -> np.ndarray:
        """Kinetic energy operator: -ℏ²/(2m) ∇² ψ"""
        return -(HBAR**2 / (2.0 * mass)) * self.geo.laplacian(psi)

    def _H_boson(self, psi_B: np.ndarray, n_F: np.ndarray) -> np.ndarray:
        """Full GPE Hamiltonian applied to boson wavefunction."""
        n_B = np.abs(psi_B)**2
        T   = self._kinetic(psi_B, self.boson_mass)
        V   = (self.V_boson
               + self.g_BB * n_B
               + self.g_BF * n_F) * psi_B
        return T + V

    def _H_fermion(self, psi_F: np.ndarray, n_B: np.ndarray) -> np.ndarray:
        """Pseudo-GPE Hamiltonian applied to fermion wavefunction."""
        n_F  = np.abs(psi_F)**2
        T    = self._kinetic(psi_F, self.fermion_mass)
        E_F  = fermi_energy_term(n_F, self.fermion_mass,
                                  self.fermion_spin_degen, self._ndim)
        V    = (self.V_fermion
                + E_F
                + self.g_BF * n_B) * psi_F
        return T + V

    # ------------------------------------------------------------------ #

    def solve(
        self,
        callback: Callable[[int, float], None] | None = None,
    ) -> "ITESolver":
        """
        Run imaginary-time evolution until convergence or max_iter.

        Parameters
        ----------
        callback : callable(iter, residual) -> None, optional
            Called every 100 iterations; useful for progress reporting.

        Returns
        -------
        self  (for chaining)
        """
        self.residuals = []
        self.converged = False

        psi_B = self.psi_B.copy()
        psi_F = self.psi_F.copy()

        n_B_old = np.abs(psi_B)**2
        n_F_old = np.abs(psi_F)**2

        for it in range(1, self.max_iter + 1):

            n_B = np.abs(psi_B)**2
            n_F = np.abs(psi_F)**2

            # ── boson step ──────────────────────────────────────────
            H_B       = self._H_boson(psi_B, n_F)
            psi_B, mu_B = _ite_step(psi_B, H_B, self.dt, self.boson_N, self.geo)

            # ── fermion step ─────────────────────────────────────────
            H_F       = self._H_fermion(psi_F, n_B)
            psi_F, mu_F = _ite_step(psi_F, H_F, self.dt, self.fermion_N, self.geo)

            # ── convergence check every 50 steps ────────────────────
            if it % 50 == 0:
                n_B_new = np.abs(psi_B)**2
                n_F_new = np.abs(psi_F)**2

                dB = np.max(np.abs(n_B_new - n_B_old)) / (np.max(n_B_new) + 1e-30)
                dF = np.max(np.abs(n_F_new - n_F_old)) / (np.max(n_F_new) + 1e-30)
                res = max(dB, dF)
                self.residuals.append(res)

                if callback is not None:
                    callback(it, res)

                if res < self.tol:
                    self.converged = True
                    self.iterations = it
                    break

                # density mixing for stability
                n_B_old = self.mixing * n_B_new + (1 - self.mixing) * n_B_old
                n_F_old = self.mixing * n_F_new + (1 - self.mixing) * n_F_old
        else:
            self.iterations = self.max_iter

        self.psi_B = psi_B
        self.psi_F = psi_F
        self.mu_B  = mu_B
        self.mu_F  = mu_F

        return self

    # ------------------------------------------------------------------ #
    @property
    def density_boson(self) -> np.ndarray:
        return np.abs(self.psi_B)**2

    @property
    def density_fermion(self) -> np.ndarray:
        return np.abs(self.psi_F)**2

    @property
    def total_boson_number(self) -> float:
        return self.geo.norm(self.psi_B)

    @property
    def total_fermion_number(self) -> float:
        return self.geo.norm(self.psi_F)
