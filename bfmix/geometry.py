"""
Grid and geometry definitions.

Four simulation geometries are supported:

  3D          — Cartesian (x, y, z), full 3-D Laplacian.
                In practice we exploit spherical / cylindrical symmetry
                and work on a reduced radial grid when the potential
                permits it.  Here we keep a general 3-D Cartesian grid
                so any potential shape is handled automatically.

  2D_planar   — (x, y) plane, 2-D Laplacian.

  2D_cyl      — Cylindrical symmetry, coordinates (r, z).
                The kinetic energy operator is the cylindrical Laplacian
                ∂²/∂r² + (1/r)∂/∂r + ∂²/∂z².

  1D          — Single spatial coordinate x, 1-D Laplacian.

All grids use uniform spacing.  The Laplacian is evaluated with
second-order finite differences.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np

GeometryType = Literal["3D", "2D_planar", "2D_cyl", "1D"]


@dataclass
class Geometry:
    """
    Simulation geometry and spatial grid.

    Parameters
    ----------
    geometry : GeometryType
    points : int or tuple of int
        Number of grid points along each axis.
        - 1D         : single int or (Nx,)
        - 2D_planar  : (Nx, Ny)
        - 2D_cyl     : (Nr, Nz)  — r ∈ [dr/2, R], z ∈ [-Z, Z]
        - 3D         : (Nx, Ny, Nz)
    extent : float or tuple of float
        Physical half-extent (metres) of the simulation box along each axis.
        For 2D_cyl the order is (R, Z_half).
    """
    geometry: GeometryType
    points: int | tuple[int, ...]
    extent: float | tuple[float, ...]

    # populated by __post_init__
    axes: list[np.ndarray] = field(init=False, repr=False)
    grids: list[np.ndarray] = field(init=False, repr=False)
    dV: float = field(init=False)          # volume element (m^D)
    shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        pts = self._broadcast(self.points)
        ext = self._broadcast(self.extent)

        if self.geometry == "1D":
            pts, ext = pts[:1], ext[:1]
            x = np.linspace(-ext[0], ext[0], pts[0])
            self.axes = [x]
            self.grids = [x]
            self.dV = x[1] - x[0]
            self.shape = (pts[0],)

        elif self.geometry == "2D_planar":
            pts, ext = pts[:2], ext[:2]
            x = np.linspace(-ext[0], ext[0], pts[0])
            y = np.linspace(-ext[1], ext[1], pts[1])
            self.axes = [x, y]
            X, Y = np.meshgrid(x, y, indexing="ij")
            self.grids = [X, Y]
            self.dV = (x[1]-x[0]) * (y[1]-y[0])
            self.shape = (pts[0], pts[1])

        elif self.geometry == "2D_cyl":
            pts, ext = pts[:2], ext[:2]
            Nr, Nz = pts
            R, Z = ext
            dr = R / Nr
            r = np.linspace(dr/2, R - dr/2, Nr)  # cell-centred to avoid r=0
            z = np.linspace(-Z, Z, Nz)
            self.axes = [r, z]
            rg, zg = np.meshgrid(r, z, indexing="ij")
            self.grids = [rg, zg]
            self.dV = dr * (z[1]-z[0]) * 2 * np.pi  # azimuthal integral
            self.shape = (Nr, Nz)

        elif self.geometry == "3D":
            pts, ext = pts[:3], ext[:3]
            x = np.linspace(-ext[0], ext[0], pts[0])
            y = np.linspace(-ext[1], ext[1], pts[1])
            z = np.linspace(-ext[2], ext[2], pts[2])
            self.axes = [x, y, z]
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            self.grids = [X, Y, Z]
            self.dV = (x[1]-x[0]) * (y[1]-y[0]) * (z[1]-z[0])
            self.shape = tuple(pts)

        else:
            raise ValueError(f"Unknown geometry: {self.geometry!r}")

    # ------------------------------------------------------------------ #
    def laplacian(self, psi: np.ndarray) -> np.ndarray:
        """
        Apply the appropriate Laplacian operator to field psi.
        Uses second-order central finite differences.
        """
        if self.geometry == "1D":
            return self._fd2_1d(psi, self.axes[0])

        elif self.geometry == "2D_planar":
            d2x = self._fd2_1d(psi, self.axes[0], axis=0)
            d2y = self._fd2_1d(psi, self.axes[1], axis=1)
            return d2x + d2y

        elif self.geometry == "2D_cyl":
            r = self.grids[0]
            dr = self.axes[0][1] - self.axes[0][0]
            dz = self.axes[1][1] - self.axes[1][0]
            # ∂²/∂r²  +  (1/r)∂/∂r  +  ∂²/∂z²
            d2r   = self._fd2_1d(psi, self.axes[0], axis=0)
            # first derivative in r via central differences
            dpsi_dr         = np.zeros_like(psi)
            dpsi_dr[1:-1,:] = (psi[2:,:] - psi[:-2,:]) / (2*dr)
            dpsi_dr[0,:]    = (psi[1,:]  - psi[0,:])   / dr
            dpsi_dr[-1,:]   = (psi[-1,:] - psi[-2,:])  / dr
            d2z   = self._fd2_1d(psi, self.axes[1], axis=1)
            return d2r + dpsi_dr / r + d2z

        elif self.geometry == "3D":
            d2x = self._fd2_1d(psi, self.axes[0], axis=0)
            d2y = self._fd2_1d(psi, self.axes[1], axis=1)
            d2z = self._fd2_1d(psi, self.axes[2], axis=2)
            return d2x + d2y + d2z

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fd2_1d(f: np.ndarray, ax: np.ndarray, axis: int = 0) -> np.ndarray:
        """Second derivative along `axis` with Dirichlet BC (psi=0 at walls)."""
        d = ax[1] - ax[0]
        out = np.zeros_like(f)
        slc = [slice(None)] * f.ndim
        # interior
        slc[axis] = slice(1, -1)
        sp = list(slc); sp[axis] = slice(2, None)
        sm = list(slc); sm[axis] = slice(None, -2)
        out[tuple(slc)] = (f[tuple(sp)] - 2*f[tuple(slc)] + f[tuple(sm)]) / d**2
        # boundaries: ghost point = 0 (Dirichlet)  → ( 0 - 2f[0] + f[1] ) / d²
        slc0 = [slice(None)] * f.ndim; slc0[axis] = 0
        slc1 = [slice(None)] * f.ndim; slc1[axis] = 1
        slcN = [slice(None)] * f.ndim; slcN[axis] = -1
        slcN1= [slice(None)] * f.ndim; slcN1[axis] = -2
        out[tuple(slc0)] = (-2*f[tuple(slc0)] + f[tuple(slc1)]) / d**2
        out[tuple(slcN)] = (f[tuple(slcN1)] - 2*f[tuple(slcN)]) / d**2
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _broadcast(v) -> list:
        if np.isscalar(v):
            return [v, v, v]
        return list(v)

    # ------------------------------------------------------------------ #
    def integrate(self, f: np.ndarray) -> float:
        """Integrate a scalar field over the simulation volume."""
        if self.geometry == "2D_cyl":
            # dV already includes the 2π r factor
            return float(np.sum(f * self.grids[0]) * (self.axes[0][1]-self.axes[0][0])
                         * (self.axes[1][1]-self.axes[1][0]) * 2 * np.pi)
        return float(np.sum(f) * self.dV)

    def norm(self, psi: np.ndarray) -> float:
        """L² norm of a wavefunction on this grid."""
        return self.integrate(np.abs(psi)**2)

    @property
    def ndim(self) -> int:
        return {"1D": 1, "2D_planar": 2, "2D_cyl": 2, "3D": 3}[self.geometry]

    def __repr__(self) -> str:
        return f"Geometry({self.geometry!r}, shape={self.shape})"
