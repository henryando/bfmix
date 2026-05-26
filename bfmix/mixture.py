"""
High-level interface: BoseFermiMixture
=======================================

Typical usage
-------------
>>> from bfmix import BoseFermiMixture, ATOMS, Geometry
>>> from bfmix.potentials import harmonic_trap
>>>
>>> geo = Geometry("3D", points=64, extent=20e-6)
>>>
>>> mix = BoseFermiMixture(
...     boson=ATOMS["87Rb"],
...     fermion=ATOMS["40K"],
...     geometry=geo,
...     N_boson=1e5,
...     N_fermion=5e4,
...     V_boson=harmonic_trap(ATOMS["87Rb"].mass, omega=(2*np.pi*50,)*3),
...     V_fermion=harmonic_trap(ATOMS["40K"].mass, omega=(2*np.pi*60,)*3),
...     a_BB=100.4,          # Bohr radii  (overrides AtomSpec default)
...     a_BF=-185.0,         # Bohr radii
... )
>>>
>>> result = mix.solve(max_iter=3000, tol=1e-6)
>>> print(result.summary())
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .atoms import AtomSpec, HBAR, A0
from .geometry import Geometry
from .potentials import PotentialFn
from .solvers.ite import ITESolver, coupling_constant


@dataclass
class MixtureResult:
    """Container for the output of a completed simulation."""
    geo: Geometry
    density_boson: np.ndarray
    density_fermion: np.ndarray
    mu_boson: float
    mu_fermion: float
    N_boson: float
    N_fermion: float
    converged: bool
    iterations: int
    residuals: list[float]
    # metadata
    boson_name: str
    fermion_name: str
    g_BB: float
    g_BF: float

    def summary(self) -> str:
        lines = [
            "═" * 52,
            "  Bose-Fermi Mixture — Ground State Result",
            "═" * 52,
            f"  Boson species    : {self.boson_name}",
            f"  Fermion species  : {self.fermion_name}",
            f"  Geometry         : {self.geo.geometry}  {self.geo.shape}",
            f"  N_boson          : {self.N_boson:.4g}",
            f"  N_fermion        : {self.N_fermion:.4g}",
            f"  μ_boson          : {self.mu_boson:.4e} J  "
            f"({self.mu_boson / (1.380649e-23 * 1e-9):.3g} nK·k_B)",
            f"  μ_fermion        : {self.mu_fermion:.4e} J  "
            f"({self.mu_fermion / (1.380649e-23 * 1e-9):.3g} nK·k_B)",
            f"  g_BB             : {self.g_BB:.4e} J·m^D",
            f"  g_BF             : {self.g_BF:.4e} J·m^D",
            f"  Converged        : {self.converged}",
            f"  Iterations       : {self.iterations}",
            f"  Final residual   : {self.residuals[-1]:.2e}" if self.residuals else "",
            "═" * 52,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Convenience density accessors                                        #
    # ------------------------------------------------------------------ #

    @property
    def peak_density_boson(self) -> float:
        return float(np.max(self.density_boson))

    @property
    def peak_density_fermion(self) -> float:
        return float(np.max(self.density_fermion))

    def column_density(self, species: str = "boson", axis: int = -1) -> np.ndarray:
        """
        Integrate density along `axis` to produce a column-density projection.
        For 1-D geometries this just returns the density.
        """
        rho = self.density_boson if species == "boson" else self.density_fermion
        if rho.ndim == 1:
            return rho
        # approximate integration: sum * grid spacing
        geo = self.geo
        if geo.geometry == "2D_cyl" and axis == 0:
            # integrate over r with 2π r dr weight → line-of-sight along r
            r   = geo.axes[0]
            dr  = r[1] - r[0]
            return 2 * np.pi * np.sum(rho * r[:, None], axis=0) * dr
        ax_spacing = geo.axes[axis][1] - geo.axes[axis][0]
        return np.sum(rho, axis=axis) * ax_spacing


class BoseFermiMixture:
    """
    High-level interface for setting up and solving a Bose-Fermi mixture.

    Parameters
    ----------
    boson : AtomSpec
        Bosonic species.
    fermion : AtomSpec
        Fermionic species.
    geometry : Geometry
        Spatial grid and geometry type.
    N_boson : float
        Target boson number.
    N_fermion : float
        Target fermion number.
    V_boson : callable(*grids) -> ndarray
        External potential for bosons.
    V_fermion : callable(*grids) -> ndarray
        External potential for fermions.
    a_BB : float, optional
        Boson-boson s-wave scattering length (Bohr radii).
        Defaults to boson.a_scatt_bohr.
    a_BF : float, optional
        Boson-fermion s-wave scattering length (Bohr radii).
        Default 0 (non-interacting mixture).
    a_ho_transverse : float, optional
        Transverse harmonic oscillator length (metres) for dimensional
        reduction of coupling constants in 1-D / 2-D geometries.
    """

    def __init__(
        self,
        boson: AtomSpec,
        fermion: AtomSpec,
        geometry: Geometry,
        N_boson: float,
        N_fermion: float,
        V_boson: PotentialFn,
        V_fermion: PotentialFn,
        a_BB: float | None = None,
        a_BF: float = 0.0,
        a_ho_transverse: float | None = None,
    ):
        if boson.statistics != "boson":
            raise ValueError(f"{boson.name} is not a boson.")
        if fermion.statistics != "fermion":
            raise ValueError(f"{fermion.name} is not a fermion.")

        self.boson   = boson
        self.fermion = fermion
        self.geo     = geometry
        self.N_B     = float(N_boson)
        self.N_F     = float(N_fermion)

        # Evaluate external potentials on the grid
        self._V_B_fn = V_boson
        self._V_F_fn = V_fermion
        self.V_B = V_boson(*geometry.grids)
        self.V_F = V_fermion(*geometry.grids)

        # Scattering lengths → SI
        a_BB_m = (a_BB if a_BB is not None else boson.a_scatt_bohr) * A0
        a_BF_m = a_BF * A0

        ndim = geometry.ndim
        self.g_BB = coupling_constant(a_BB_m, boson.mass,   ndim, a_ho_transverse)
        self.g_BF = coupling_constant(a_BF_m, boson.mass,   ndim, a_ho_transverse)
        # Note: g_BF for fermion equation uses same value (symmetric coupling)

    # ------------------------------------------------------------------ #

    def solve(
        self,
        dt: float | None = None,
        max_iter: int = 5000,
        tol: float = 1e-6,
        mixing: float = 0.5,
        callback: Callable[[int, float], None] | None = None,
    ) -> MixtureResult:
        """
        Run imaginary-time evolution to find the ground state.

        Parameters
        ----------
        dt : float, optional
            Imaginary-time step (s).  Auto-selected if None.
        max_iter : int
            Maximum number of ITE iterations.
        tol : float
            Convergence tolerance on the relative density change.
        mixing : float
            Density mixing fraction for stability (0 < mixing ≤ 1).
        callback : callable(iter, residual), optional
            Progress callback invoked every 50 iterations.

        Returns
        -------
        MixtureResult
        """
        if dt is None:
            dt = self._auto_dt()

        solver = ITESolver(
            geo             = self.geo,
            boson_mass      = self.boson.mass,
            fermion_mass    = self.fermion.mass,
            boson_N         = self.N_B,
            fermion_N       = self.N_F,
            V_boson         = self.V_B,
            V_fermion       = self.V_F,
            g_BB            = self.g_BB,
            g_BF            = self.g_BF,
            fermion_spin_degen = self.fermion.spin_degeneracy,
            dt              = dt,
            max_iter        = max_iter,
            tol             = tol,
            mixing          = mixing,
        )
        solver.solve(callback=callback)

        return MixtureResult(
            geo             = self.geo,
            density_boson   = solver.density_boson,
            density_fermion = solver.density_fermion,
            mu_boson        = solver.mu_B,
            mu_fermion      = solver.mu_F,
            N_boson         = solver.total_boson_number,
            N_fermion       = solver.total_fermion_number,
            converged       = solver.converged,
            iterations      = solver.iterations,
            residuals       = solver.residuals,
            boson_name      = self.boson.name,
            fermion_name    = self.fermion.name,
            g_BB            = self.g_BB,
            g_BF            = self.g_BF,
        )

    # ------------------------------------------------------------------ #

    def _auto_dt(self) -> float:
        """
        Heuristic time-step: dt = 0.1 · ℏ / E_scale
        where E_scale = max kinetic energy ~ ℏ²/(2m·dx²).
        """
        dx   = min(ax[1] - ax[0] for ax in self.geo.axes)
        m    = min(self.boson.mass, self.fermion.mass)
        E_kin = HBAR**2 / (2.0 * m * dx**2)
        V_max = max(float(np.max(np.abs(self.V_B))),
                    float(np.max(np.abs(self.V_F))),
                    1e-30)
        E_scale = max(E_kin, V_max)
        return 0.05 * HBAR / E_scale
