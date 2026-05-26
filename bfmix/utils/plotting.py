"""
Plotting utilities for bfmix simulation results.

All functions return matplotlib Figure objects so they can be
embedded in the Streamlit app or saved to disk.
"""

from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mixture import MixtureResult

# Use a non-interactive backend when running headlessly
matplotlib.use("Agg")

# ── Colour palette ──────────────────────────────────────────────────
BOSON_CMAP   = "Blues"
FERMION_CMAP = "Oranges"
BOSON_COLOR  = "#2B7BBA"
FERMION_COLOR= "#E07B2A"


def _si_prefix(val: float):
    """Return (scaled_val, prefix_string) for a density value."""
    if val == 0:
        return val, ""
    exp = np.floor(np.log10(abs(val)))
    if exp >= 18:
        return val * 1e-18, "×10¹⁸ "
    elif exp >= 15:
        return val * 1e-15, "×10¹⁵ "
    elif exp >= 12:
        return val * 1e-12, "×10¹² "
    elif exp >= 6:
        return val * 1e-6, "×10⁶ "
    return val, ""


def plot_1d(result: "MixtureResult") -> plt.Figure:
    """Line plot of boson and fermion densities for 1-D simulations."""
    x  = result.geo.axes[0] * 1e6    # → μm
    nB = result.density_boson
    nF = result.density_fermion

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, nB, color=BOSON_COLOR,   lw=2, label=f"Bosons ({result.boson_name})")
    ax.plot(x, nF, color=FERMION_COLOR, lw=2, label=f"Fermions ({result.fermion_name})")
    ax.set_xlabel("x  (μm)")
    ax.set_ylabel("Linear density  (m⁻¹)")
    ax.set_title("1-D density profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_2d_planar(result: "MixtureResult") -> plt.Figure:
    """Side-by-side 2-D colour maps for planar geometry."""
    x  = result.geo.axes[0] * 1e6
    y  = result.geo.axes[1] * 1e6
    nB = result.density_boson
    nF = result.density_fermion

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, rho, cmap, title in zip(
        axes,
        [nB, nF],
        [BOSON_CMAP, FERMION_CMAP],
        [f"Bosons ({result.boson_name})", f"Fermions ({result.fermion_name})"],
    ):
        im = ax.pcolormesh(x, y, rho.T, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, label="n  (m⁻²)")
        ax.set_xlabel("x  (μm)")
        ax.set_ylabel("y  (μm)")
        ax.set_title(title)
        ax.set_aspect("equal")

    fig.suptitle("2-D planar density", fontsize=13)
    fig.tight_layout()
    return fig


def plot_2d_cyl(result: "MixtureResult") -> plt.Figure:
    """
    Cylindrical geometry: show the (r, z) density map and the
    axial line-of-sight column density.
    """
    r  = result.geo.axes[0] * 1e6   # μm
    z  = result.geo.axes[1] * 1e6
    nB = result.density_boson
    nF = result.density_fermion

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = [(nB, BOSON_CMAP,   f"Bosons ({result.boson_name})"),
             (nF, FERMION_CMAP, f"Fermions ({result.fermion_name})")]

    for col, (rho, cmap, title) in enumerate(pairs):
        # (r, z) map
        ax = axes[0, col]
        im = ax.pcolormesh(z, r, rho, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, label="n  (m⁻²)")
        ax.set_xlabel("z  (μm)")
        ax.set_ylabel("r  (μm)")
        ax.set_title(title + "  [r-z plane]")

        # axial profile n(z) = ∫ n 2πr dr
        dr = (result.geo.axes[0][1] - result.geo.axes[0][0]) * 1e6  # μm→μm ok; use SI below
        dr_si = result.geo.axes[0][1] - result.geo.axes[0][0]
        col_den = 2 * np.pi * np.sum(rho * result.geo.grids[0], axis=0) * dr_si
        ax2 = axes[1, col]
        ax2.plot(z, col_den, color=BOSON_COLOR if col == 0 else FERMION_COLOR, lw=2)
        ax2.set_xlabel("z  (μm)")
        ax2.set_ylabel("Column density  (m⁻¹)")
        ax2.set_title(title + "  [axial profile]")
        ax2.grid(True, alpha=0.3)

    fig.suptitle("Cylindrical geometry", fontsize=13)
    fig.tight_layout()
    return fig


def plot_3d_slices(result: "MixtureResult") -> plt.Figure:
    """
    3-D geometry: show three orthogonal mid-plane slices for each species.
    """
    shape = result.geo.shape
    ix = shape[0] // 2
    iy = shape[1] // 2
    iz = shape[2] // 2

    axes_um = [ax * 1e6 for ax in result.geo.axes]
    x, y, z = axes_um

    nB = result.density_boson
    nF = result.density_fermion

    slice_labels = [
        ("xy", (slice(None), slice(None), iz), x, y, "x (μm)", "y (μm)"),
        ("xz", (slice(None), iy, slice(None)), x, z, "x (μm)", "z (μm)"),
        ("yz", (ix, slice(None), slice(None)), y, z, "y (μm)", "z (μm)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    cmaps = [BOSON_CMAP, FERMION_CMAP]
    titles = [f"Bosons ({result.boson_name})", f"Fermions ({result.fermion_name})"]

    for row, (rho, cmap, title) in enumerate(zip([nB, nF], cmaps, titles)):
        for col, (plane, slc, ax1, ax2, xl, yl) in enumerate(slice_labels):
            ax = axes[row, col]
            data = rho[slc]
            im = ax.pcolormesh(ax1, ax2, data.T, cmap=cmap, shading="auto")
            plt.colorbar(im, ax=ax, label="n  (m⁻³)")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{title}  [{plane}-plane]")
            ax.set_aspect("equal")

    fig.suptitle("3-D density slices", fontsize=13)
    fig.tight_layout()
    return fig


def plot_convergence(result: "MixtureResult") -> plt.Figure:
    """Plot the convergence residual vs. iteration."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    iters = np.arange(1, len(result.residuals) + 1) * 50
    ax.semilogy(iters, result.residuals, color="#444", lw=1.5, marker="o",
                markersize=3)
    if result.converged:
        ax.axhline(result.residuals[-1], color="green", ls="--", alpha=0.6,
                   label=f"Converged ({result.iterations} iter)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max relative density change")
    ax.set_title("ITE convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def auto_plot(result: "MixtureResult") -> plt.Figure:
    """Choose the right plot function for the geometry automatically."""
    g = result.geo.geometry
    if g == "1D":
        return plot_1d(result)
    elif g == "2D_planar":
        return plot_2d_planar(result)
    elif g == "2D_cyl":
        return plot_2d_cyl(result)
    elif g == "3D":
        return plot_3d_slices(result)
    raise ValueError(f"Unknown geometry {g!r}")
