"""
Atomic species definitions.

Each entry in ATOMS contains the physical properties needed for the solver:
  - mass in atomic mass units (u)
  - statistics: 'boson' or 'fermion'
  - common scattering length (a_bg) in Bohr radii, used as default
  - spin degeneracy g_s (relevant for fermions)
"""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np

# Physical constants (SI)
HBAR = 1.054571817e-34   # J·s
AMU  = 1.66053906660e-27 # kg
A0   = 5.29177210903e-11 # Bohr radius, m
KB   = 1.380649e-23      # J/K


@dataclass
class AtomSpec:
    """
    Specification of an atomic species for a mixture simulation.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. '87Rb', '40K'.
    mass_amu : float
        Mass in atomic mass units.
    statistics : 'boson' or 'fermion'
    spin_degeneracy : int
        Number of accessible spin states (g_s).  For a spin-polarised
        fermion gas set g_s = 1.
    a_scatt_bohr : float, optional
        Default s-wave scattering length in Bohr radii.  Used to
        pre-fill the UI; the user can override it at runtime.
    """
    name: str
    mass_amu: float
    statistics: Literal["boson", "fermion"]
    spin_degeneracy: int = 1
    a_scatt_bohr: float = 0.0

    # ------------------------------------------------------------------ #
    #  Derived properties                                                  #
    # ------------------------------------------------------------------ #
    @property
    def mass(self) -> float:
        """Mass in kg."""
        return self.mass_amu * AMU

    @property
    def a_scatt(self) -> float:
        """Default scattering length in metres."""
        return self.a_scatt_bohr * A0

    def g_3d(self, a_scatt_m: float | None = None) -> float:
        """3-D contact interaction coupling constant g = 4π ℏ² a / m (SI)."""
        a = a_scatt_m if a_scatt_m is not None else self.a_scatt
        return 4.0 * np.pi * HBAR**2 * a / self.mass

    def __repr__(self) -> str:
        return (f"AtomSpec({self.name!r}, {self.statistics}, "
                f"m={self.mass_amu:.2f} u, g_s={self.spin_degeneracy})")


# ------------------------------------------------------------------ #
#  Built-in atom library                                               #
# ------------------------------------------------------------------ #
ATOMS: dict[str, AtomSpec] = {
    # ── Bosons ──────────────────────────────────────────────────────
    "87Rb": AtomSpec(
        name="87Rb", mass_amu=86.909, statistics="boson",
        spin_degeneracy=1, a_scatt_bohr=100.4,
    ),
    "23Na": AtomSpec(
        name="23Na", mass_amu=22.990, statistics="boson",
        spin_degeneracy=1, a_scatt_bohr=52.9,
    ),
    "7Li": AtomSpec(
        name="7Li", mass_amu=7.016, statistics="boson",
        spin_degeneracy=1, a_scatt_bohr=-27.6,
    ),
    "133Cs": AtomSpec(
        name="133Cs", mass_amu=132.905, statistics="boson",
        spin_degeneracy=1, a_scatt_bohr=280.0,
    ),
    "41K": AtomSpec(
        name="41K", mass_amu=40.962, statistics="boson",
        spin_degeneracy=1, a_scatt_bohr=60.0,
    ),
    "4He*": AtomSpec(
        name="4He*", mass_amu=4.003, statistics="boson",
        spin_degeneracy=1, a_scatt_bohr=7.5,
    ),
    "174Yb": AtomSpec(
        name="174Yb", mass_amu=173.938, statistics="boson",
        spin_degeneracy=1, a_scatt_bohr=105.0,
    ),
    "84Sr": AtomSpec(
        name="84Sr", mass_amu=83.913, statistics="boson",
        spin_degeneracy=1, a_scatt_bohr=123.0,
    ),
    # ── Fermions ────────────────────────────────────────────────────
    "40K": AtomSpec(
        name="40K", mass_amu=39.964, statistics="fermion",
        spin_degeneracy=1, a_scatt_bohr=-174.0,
    ),
    "6Li": AtomSpec(
        name="6Li", mass_amu=6.015, statistics="fermion",
        spin_degeneracy=1, a_scatt_bohr=-2160.0,
    ),
    "3He*": AtomSpec(
        name="3He*", mass_amu=3.016, statistics="fermion",
        spin_degeneracy=1, a_scatt_bohr=0.0,
    ),
    "173Yb": AtomSpec(
        name="173Yb", mass_amu=172.938, statistics="fermion",
        spin_degeneracy=6, a_scatt_bohr=199.0,
    ),
    "171Yb": AtomSpec(
        name="171Yb", mass_amu=170.936, statistics="fermion",
        spin_degeneracy=2, a_scatt_bohr=200.0,
    ),
    "87Sr": AtomSpec(
        name="87Sr", mass_amu=86.909, statistics="fermion",
        spin_degeneracy=10, a_scatt_bohr=96.2,
    ),
}


def list_bosons() -> list[str]:
    return [k for k, v in ATOMS.items() if v.statistics == "boson"]


def list_fermions() -> list[str]:
    return [k for k, v in ATOMS.items() if v.statistics == "fermion"]
