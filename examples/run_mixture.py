"""
examples/run_mixture.py
────────────────────────────────────────────────────────────────────────────
Quick demonstration of bfmix for four geometries.

Usage:
    python examples/run_mixture.py --geo 1D
    python examples/run_mixture.py --geo 2D_cyl
    python examples/run_mixture.py --geo 2D_planar
    python examples/run_mixture.py --geo 3D
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from bfmix import BoseFermiMixture, ATOMS, Geometry
from bfmix.potentials import harmonic_trap
from bfmix.utils.plotting import auto_plot, plot_convergence


# ─── geometry presets ────────────────────────────────────────────────────────
PRESETS = {
    "1D": dict(
        points=128, extent=50e-6,
        omega_B=(2*np.pi*50,),
        omega_F=(2*np.pi*60,),
    ),
    "2D_planar": dict(
        points=(64, 64), extent=(40e-6, 40e-6),
        omega_B=(2*np.pi*50, 2*np.pi*50),
        omega_F=(2*np.pi*60, 2*np.pi*60),
    ),
    "2D_cyl": dict(
        points=(64, 64), extent=(30e-6, 60e-6),
        omega_B=(2*np.pi*80, 2*np.pi*20),   # (ωr, ωz)
        omega_F=(2*np.pi*90, 2*np.pi*25),
    ),
    "3D": dict(
        points=(32, 32, 32), extent=(25e-6, 25e-6, 25e-6),
        omega_B=(2*np.pi*50,)*3,
        omega_F=(2*np.pi*60,)*3,
    ),
}


def run(geo_type: str):
    p = PRESETS[geo_type]

    print(f"\n{'='*60}")
    print(f"  bfmix example  —  geometry: {geo_type}")
    print(f"{'='*60}")

    geo = Geometry(geo_type, points=p["points"], extent=p["extent"])
    print(f"  Grid: {geo.shape}")

    boson   = ATOMS["87Rb"]
    fermion = ATOMS["40K"]

    mix = BoseFermiMixture(
        boson     = boson,
        fermion   = fermion,
        geometry  = geo,
        N_boson   = 30_000,
        N_fermion = 10_000,
        V_boson   = harmonic_trap(boson.mass,   p["omega_B"]),
        V_fermion = harmonic_trap(fermion.mass, p["omega_F"]),
        a_BB      = 100.4,
        a_BF      = -185.0,
    )

    def _cb(it, res):
        if it % 500 == 0:
            print(f"  iter {it:5d}  residual = {res:.2e}")

    result = mix.solve(max_iter=3000, tol=1e-5, callback=_cb)
    print(result.summary())

    # Density plot
    fig_d = auto_plot(result)
    fig_d.savefig(f"density_{geo_type}.png", dpi=150)
    print(f"  Saved: density_{geo_type}.png")

    # Convergence plot
    fig_c = plot_convergence(result)
    fig_c.savefig(f"convergence_{geo_type}.png", dpi=150)
    print(f"  Saved: convergence_{geo_type}.png")

    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geo", default="1D",
        choices=list(PRESETS.keys()),
        help="Geometry type",
    )
    args = parser.parse_args()
    run(args.geo)
