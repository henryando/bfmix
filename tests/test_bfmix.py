"""
Unit and integration tests for bfmix.
Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bfmix.atoms import ATOMS, AtomSpec, list_bosons, list_fermions
from bfmix.geometry import Geometry
from bfmix.potentials import harmonic_trap, box_trap
from bfmix.solvers.ite import fermi_energy_term, coupling_constant, ITESolver
from bfmix.mixture import BoseFermiMixture

# ─────────────────────────────────────────────────────────────────────────────
# Atom database
# ─────────────────────────────────────────────────────────────────────────────

def test_atom_database_not_empty():
    assert len(ATOMS) > 0

def test_bosons_and_fermions_present():
    assert len(list_bosons())   > 0
    assert len(list_fermions()) > 0

def test_atom_mass_positive():
    for atom in ATOMS.values():
        assert atom.mass > 0

def test_statistics_valid():
    for atom in ATOMS.values():
        assert atom.statistics in ("boson", "fermion")

def test_spin_degeneracy_positive():
    for atom in ATOMS.values():
        assert atom.spin_degeneracy >= 1

# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("geo_type,pts,ext", [
    ("1D",       64,         50e-6),
    ("2D_planar",(32,32),   (30e-6, 30e-6)),
    ("2D_cyl",  (32,32),   (30e-6, 50e-6)),
    ("3D",      (16,16,16),(20e-6, 20e-6, 20e-6)),
])
def test_geometry_creation(geo_type, pts, ext):
    geo = Geometry(geo_type, points=pts, extent=ext)
    assert geo.geometry == geo_type
    assert len(geo.axes) == geo.ndim

def test_geometry_1d_shape():
    geo = Geometry("1D", points=50, extent=10e-6)
    assert geo.shape == (50,)

def test_geometry_2d_planar_shape():
    geo = Geometry("2D_planar", points=(30, 40), extent=(10e-6, 15e-6))
    assert geo.shape == (30, 40)

def test_geometry_2d_cyl_shape():
    geo = Geometry("2D_cyl", points=(24, 36), extent=(15e-6, 40e-6))
    assert geo.shape == (24, 36)
    # r-axis should be positive
    assert np.all(geo.axes[0] > 0)

def test_geometry_3d_shape():
    geo = Geometry("3D", points=(10,12,14), extent=(5e-6,6e-6,7e-6))
    assert geo.shape == (10, 12, 14)

def test_laplacian_1d_gaussian():
    """∇² of Gaussian should be negative at centre."""
    geo = Geometry("1D", points=128, extent=30e-6)
    x   = geo.axes[0]
    sig = 5e-6
    psi = np.exp(-x**2 / (2 * sig**2))
    lap = geo.laplacian(psi)
    # at x=0, second derivative of Gaussian is negative
    assert lap[len(x)//2] < 0

def test_norm_1d():
    """Norm of a manually normalised Gaussian should return N."""
    geo = Geometry("1D", points=256, extent=50e-6)
    x   = geo.axes[0]
    psi = np.exp(-x**2 / (2 * (5e-6)**2))
    N   = 1000.0
    psi = psi * np.sqrt(N / geo.norm(psi))
    assert abs(geo.norm(psi) - N) / N < 1e-3

def test_integrate_2d_cyl():
    """Integral of 1 over a cylinder should equal π R² * 2Z * 2π... wait—
    The dV for cylindrical includes 2πr, so ∫ dV = 2πR²/2 * 2Z = πR²·2Z."""
    geo = Geometry("2D_cyl", points=(64, 32), extent=(10e-6, 10e-6))
    ones = np.ones(geo.shape)
    val  = geo.integrate(ones)
    # Should equal π R² · 2Z  (volume of cylinder)
    R = geo.axes[0].max()
    Z = geo.axes[1].max()
    expected = np.pi * R**2 * 2 * Z
    assert abs(val - expected) / expected < 0.05   # 5% tolerance (grid)

# ─────────────────────────────────────────────────────────────────────────────
# Physics: Fermi energy
# ─────────────────────────────────────────────────────────────────────────────

HBAR = 1.054571817e-34
M_K  = ATOMS["40K"].mass

def test_fermi_energy_3d_zero_density():
    assert fermi_energy_term(np.array([0.0]), M_K, 1, 3)[0] == 0.0

def test_fermi_energy_3d_positive():
    n = np.array([1e19])  # m^{-3}
    E = fermi_energy_term(n, M_K, 1, 3)
    assert E[0] > 0

def test_fermi_energy_scaling_3d():
    """E_F ∝ n^{2/3} in 3-D."""
    n1 = np.array([1e18])
    n2 = np.array([8e18])
    E1 = fermi_energy_term(n1, M_K, 1, 3)
    E2 = fermi_energy_term(n2, M_K, 1, 3)
    ratio = E2[0] / E1[0]
    assert abs(ratio - 4.0) < 0.01   # (8)^{2/3} = 4

def test_fermi_energy_scaling_2d():
    """E_F ∝ n in 2-D."""
    n1 = np.array([1e14])
    n2 = np.array([3e14])
    E1 = fermi_energy_term(n1, M_K, 1, 2)
    E2 = fermi_energy_term(n2, M_K, 1, 2)
    assert abs(E2[0] / E1[0] - 3.0) < 0.01

def test_fermi_energy_scaling_1d():
    """E_F ∝ n² in 1-D."""
    n1 = np.array([1e8])
    n2 = np.array([2e8])
    E1 = fermi_energy_term(n1, M_K, 1, 1)
    E2 = fermi_energy_term(n2, M_K, 1, 1)
    assert abs(E2[0] / E1[0] - 4.0) < 0.01

# ─────────────────────────────────────────────────────────────────────────────
# Coupling constants
# ─────────────────────────────────────────────────────────────────────────────

def test_coupling_3d_positive_scattering():
    A0 = 5.29177e-11
    g  = coupling_constant(100.4 * A0, ATOMS["87Rb"].mass, 3)
    assert g > 0

def test_coupling_3d_negative_scattering():
    A0 = 5.29177e-11
    g  = coupling_constant(-27.6 * A0, ATOMS["7Li"].mass, 3)
    assert g < 0

# ─────────────────────────────────────────────────────────────────────────────
# Integration: 1-D mixture solve
# ─────────────────────────────────────────────────────────────────────────────

def _make_1d_mix(N_B=5000, N_F=2000):
    geo = Geometry("1D", points=64, extent=40e-6)
    mix = BoseFermiMixture(
        boson    = ATOMS["87Rb"],
        fermion  = ATOMS["40K"],
        geometry = geo,
        N_boson  = N_B,
        N_fermion= N_F,
        V_boson  = harmonic_trap(ATOMS["87Rb"].mass, (2*np.pi*50,)),
        V_fermion= harmonic_trap(ATOMS["40K"].mass,  (2*np.pi*60,)),
        a_BB     = 100.4,
        a_BF     = 0.0,
    )
    return mix

def test_1d_solve_runs():
    mix    = _make_1d_mix()
    result = mix.solve(max_iter=200, tol=1e-4)
    assert result.density_boson  is not None
    assert result.density_fermion is not None

def test_1d_particle_number_conserved():
    mix    = _make_1d_mix(N_B=5000, N_F=2000)
    result = mix.solve(max_iter=300, tol=1e-5)
    assert abs(result.N_boson   - 5000) / 5000 < 0.02
    assert abs(result.N_fermion - 2000) / 2000 < 0.02

def test_1d_densities_nonnegative():
    mix    = _make_1d_mix()
    result = mix.solve(max_iter=200, tol=1e-4)
    assert np.all(result.density_boson   >= -1e-6 * result.density_boson.max())
    assert np.all(result.density_fermion >= -1e-6 * result.density_fermion.max())

def test_1d_chemical_potentials_finite():
    mix    = _make_1d_mix()
    result = mix.solve(max_iter=200, tol=1e-4)
    assert np.isfinite(result.mu_boson)
    assert np.isfinite(result.mu_fermion)

def test_result_summary_runs():
    mix    = _make_1d_mix()
    result = mix.solve(max_iter=100, tol=1e-3)
    s = result.summary()
    assert "Boson" in s
    assert "Fermion" in s

# ─────────────────────────────────────────────────────────────────────────────
# Species validation
# ─────────────────────────────────────────────────────────────────────────────

def test_reject_fermion_as_boson():
    geo = Geometry("1D", points=32, extent=20e-6)
    with pytest.raises(ValueError):
        BoseFermiMixture(
            boson    = ATOMS["40K"],   # fermion used as boson — should raise
            fermion  = ATOMS["87Rb"],
            geometry = geo,
            N_boson=1000, N_fermion=1000,
            V_boson=harmonic_trap(ATOMS["40K"].mass,  (2*np.pi*50,)),
            V_fermion=harmonic_trap(ATOMS["87Rb"].mass,(2*np.pi*50,)),
        )
