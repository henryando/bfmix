"""
External potential functions.

Each potential is a callable that accepts the grid coordinate arrays
from a Geometry object and returns an ndarray of the same shape.

Convention: potentials are in Joules (SI).
"""

from __future__ import annotations
import numpy as np
from typing import Callable

PotentialFn = Callable[..., np.ndarray]


# ------------------------------------------------------------------ #
#  Harmonic trap                                                       #
# ------------------------------------------------------------------ #

def harmonic_trap(
    mass: float,
    omega: float | tuple[float, ...],
) -> PotentialFn:
    """
    Isotropic or anisotropic harmonic trap.

    V = ½ m (ωx² x² + ωy² y² + ωz² z²)

    Parameters
    ----------
    mass : float
        Atomic mass in kg.
    omega : float or tuple of float
        Angular frequencies (rad/s).  Pass a scalar for an isotropic trap.
        Order matches the geometry axes: (ωx,) for 1D; (ωx, ωy) for 2D;
        (ωr, ωz) for cylindrical; (ωx, ωy, ωz) for 3D.
    """
    def _V(*grids: np.ndarray) -> np.ndarray:
        omegas = np.atleast_1d(omega)
        if len(omegas) == 1:
            omegas = np.repeat(omegas, len(grids))
        V = np.zeros_like(grids[0])
        for g, w in zip(grids, omegas):
            V = V + 0.5 * mass * w**2 * g**2
        return V
    return _V


# ------------------------------------------------------------------ #
#  Hard-wall box                                                       #
# ------------------------------------------------------------------ #

def box_trap(
    walls: float | tuple[float, ...] | None = None,
) -> PotentialFn:
    """
    Infinite potential outside the box defined by `walls`.

    Parameters
    ----------
    walls : float or tuple of float
        Half-widths of the box along each axis (metres).
        Pass None to let the grid boundary act as the wall.
    """
    def _V(*grids: np.ndarray) -> np.ndarray:
        V = np.zeros_like(grids[0])
        ws = np.atleast_1d(walls) if walls is not None else None
        for i, g in enumerate(grids):
            if ws is not None:
                w = ws[min(i, len(ws)-1)]
                V = V + np.where(np.abs(g) > w, 1e40, 0.0)
        return V
    return _V


# ------------------------------------------------------------------ #
#  Gaussian dimple / optical tweezer                                   #
# ------------------------------------------------------------------ #

def gaussian_dimple(
    depth: float,
    waist: float | tuple[float, ...],
    center: float | tuple[float, ...] = 0.0,
) -> PotentialFn:
    """
    Attractive Gaussian dimple: V = -depth · exp(-2 Σ (x_i - c_i)²/w_i²).

    Parameters
    ----------
    depth : float
        Potential depth (J), positive means attractive.
    waist : float or tuple of float
        1/e² beam waist(s) in metres.
    center : float or tuple of float
        Centre of the Gaussian.
    """
    def _V(*grids: np.ndarray) -> np.ndarray:
        ws = np.atleast_1d(waist)
        cs = np.atleast_1d(center)
        exponent = np.zeros_like(grids[0])
        for i, g in enumerate(grids):
            w = ws[min(i, len(ws)-1)]
            c = cs[min(i, len(cs)-1)]
            exponent = exponent + 2.0 * (g - c)**2 / w**2
        return -depth * np.exp(-exponent)
    return _V


# ------------------------------------------------------------------ #
#  1-D optical lattice                                                 #
# ------------------------------------------------------------------ #

def optical_lattice_1d(
    depth: float,
    spacing: float,
    axis: int = 0,
) -> PotentialFn:
    """
    Sinusoidal optical lattice along one axis.

    V = depth · sin²(π x / d)

    Parameters
    ----------
    depth : float
        Lattice depth (J).
    spacing : float
        Lattice spacing d (m).
    axis : int
        Which grid axis the lattice runs along.
    """
    def _V(*grids: np.ndarray) -> np.ndarray:
        g = grids[axis]
        return depth * np.sin(np.pi * g / spacing)**2
    return _V


# ------------------------------------------------------------------ #
#  Combined / sum of potentials                                        #
# ------------------------------------------------------------------ #

def combined(*potentials: PotentialFn) -> PotentialFn:
    """Sum any number of potential functions."""
    def _V(*grids: np.ndarray) -> np.ndarray:
        return sum(p(*grids) for p in potentials)
    return _V


# ------------------------------------------------------------------ #
#  Custom array potential                                              #
# ------------------------------------------------------------------ #

def from_array(V_array: np.ndarray) -> PotentialFn:
    """
    Wrap a pre-computed potential array as a callable.
    The array must match the geometry grid shape.
    """
    def _V(*grids: np.ndarray) -> np.ndarray:
        return V_array
    return _V
