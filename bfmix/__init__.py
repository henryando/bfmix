"""
bfmix — Ultracold Bose-Fermi Mixture Simulator
Phase 1: Thomas-Fermi / pseudo-GPE ground states via imaginary time evolution
"""

from .mixture import BoseFermiMixture
from .atoms import AtomSpec, ATOMS
from .geometry import Geometry

__version__ = "0.1.0"
__all__ = ["BoseFermiMixture", "AtomSpec", "ATOMS", "Geometry"]
