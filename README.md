# ❄️ bfmix

**Ultracold Bose-Fermi Mixture Simulator**

A Python package for computing Thomas-Fermi / pseudo-GPE ground states of ultracold Bose-Fermi mixtures using imaginary-time evolution within the local density approximation.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bfmix.streamlit.app)

---

## Overview

`bfmix` solves for the ground-state density profiles of a coupled **boson + fermion** ultracold gas. Both species are treated with a unified imaginary-time evolution (ITE) framework:

- **Bosons** — Gross-Pitaevskii equation (GPE) with contact interactions
- **Fermions** — Pseudo-GPE where the interaction nonlinearity is replaced by the local Thomas-Fermi Fermi energy $E_F[n_F]$, capturing the kinetic pressure of a degenerate Fermi gas in the LDA

After each ITE step the wavefunction of each species is renormalised to enforce the target particle number, which determines the chemical potential implicitly.

### Supported geometries

| Code | Description |
|------|-------------|
| `"1D"` | One spatial dimension, Cartesian |
| `"2D_planar"` | Two-dimensional x-y plane |
| `"2D_cyl"` | Cylindrical symmetry, full (r, z) grid |
| `"3D"` | Three-dimensional Cartesian |

---

## Physical Model

### Bosons (GPE)

$$-\hbar\frac{\partial\psi_B}{\partial\tau} = \left[-\frac{\hbar^2}{2m_B}\nabla^2 + V_B^{\rm ext} + g_{BB}|\psi_B|^2 + g_{BF}|\psi_F|^2\right]\psi_B$$

### Fermions (pseudo-GPE / LDA-TF)

$$-\hbar\frac{\partial\psi_F}{\partial\tau} = \left[-\frac{\hbar^2}{2m_F}\nabla^2 + V_F^{\rm ext} + E_F[n_F] + g_{BF}|\psi_B|^2\right]\psi_F$$

**Local Fermi energy by geometry:**

| Geometry | $E_F(n_F)$ |
|---|---|
| 3-D | $\frac{\hbar^2}{2m_F}\left(\frac{6\pi^2}{g_s}\right)^{2/3} n_F^{2/3}$ |
| 2-D (planar & cylindrical) | $\frac{\hbar^2}{2m_F}\frac{4\pi}{g_s}\, n_F$ |
| 1-D | $\frac{\hbar^2}{2m_F}\frac{\pi^2}{3g_s^2}\, n_F^2$ |

---

## Installation

```bash
pip install bfmix
# or, with the webapp dependencies:
pip install "bfmix[webapp]"
```

### From source

```bash
git clone https://github.com/henryando/bfmix.git
cd bfmix
pip install -e ".[webapp,dev]"
```

---

## Quick Start

```python
import numpy as np
from bfmix import BoseFermiMixture, ATOMS, Geometry
from bfmix.potentials import harmonic_trap
from bfmix.utils.plotting import auto_plot

# Define the spatial grid
geo = Geometry("1D", points=256, extent=50e-6)

# Build the mixture
mix = BoseFermiMixture(
    boson     = ATOMS["87Rb"],
    fermion   = ATOMS["40K"],
    geometry  = geo,
    N_boson   = 50_000,
    N_fermion = 20_000,
    V_boson   = harmonic_trap(ATOMS["87Rb"].mass, omega=(2*np.pi*50,)),
    V_fermion = harmonic_trap(ATOMS["40K"].mass,  omega=(2*np.pi*60,)),
    a_BB      = 100.4,   # Bohr radii
    a_BF      = -185.0,  # Bohr radii
)

# Solve
result = mix.solve(max_iter=5000, tol=1e-6)
print(result.summary())

# Plot density profiles
fig = auto_plot(result)
fig.savefig("density.png", dpi=150)
```

### Cylindrical geometry example

```python
geo = Geometry("2D_cyl", points=(64, 128), extent=(30e-6, 80e-6))

mix = BoseFermiMixture(
    boson     = ATOMS["87Rb"],
    fermion   = ATOMS["40K"],
    geometry  = geo,
    N_boson   = 1e5,
    N_fermion = 5e4,
    V_boson   = harmonic_trap(ATOMS["87Rb"].mass, omega=(2*np.pi*80, 2*np.pi*15)),
    V_fermion = harmonic_trap(ATOMS["40K"].mass,  omega=(2*np.pi*90, 2*np.pi*20)),
    a_BB=100.4, a_BF=-185.0,
)
result = mix.solve()
```

---

## Built-in Atom Library

**Bosons:** `87Rb`, `23Na`, `7Li`, `133Cs`, `41K`, `4He*`, `174Yb`, `84Sr`

**Fermions:** `40K`, `6Li`, `3He*`, `173Yb`, `171Yb`, `87Sr`

Custom species can be created:

```python
from bfmix.atoms import AtomSpec
my_atom = AtomSpec(
    name="MyAtom", mass_amu=40.0,
    statistics="boson", spin_degeneracy=1, a_scatt_bohr=80.0,
)
```

---

## Potential Library

```python
from bfmix.potentials import (
    harmonic_trap,       # ½mω²x²
    box_trap,            # infinite walls
    gaussian_dimple,     # attractive Gaussian
    optical_lattice_1d,  # sin² lattice
    combined,            # sum of potentials
)
```

---

## Running the Webapp Locally

```bash
cd webapp
streamlit run app.py
```

Then open http://localhost:8501.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Deploying to Streamlit Cloud

1. Fork / push this repo to `github.com/henryando/bfmix`
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. Set **Main file path** to `webapp/app.py`
4. Set **Requirements file** to `webapp/requirements.txt`
5. Click **Deploy**

---

## Project Structure

```
bfmix/
├── bfmix/                  # Python package
│   ├── __init__.py
│   ├── atoms.py            # Atom database & AtomSpec
│   ├── geometry.py         # Grid construction & Laplacian
│   ├── mixture.py          # BoseFermiMixture high-level API
│   ├── potentials.py       # External potential callables
│   ├── solvers/
│   │   └── ite.py          # Imaginary-time evolution engine
│   └── utils/
│       └── plotting.py     # Density & convergence plots
├── webapp/
│   ├── app.py              # Streamlit interface
│   └── requirements.txt
├── tests/
│   └── test_bfmix.py
├── examples/
│   └── run_mixture.py
├── pyproject.toml
└── README.md
```

---

## Roadmap

- **Phase 2:** Multi-component mixtures (multiple boson / fermion species), finite-temperature LDA
- **Phase 3:** Time-dependent GPE, excitation spectra, collective modes
- **Phase 4:** Beyond-mean-field corrections (Lee-Huang-Yang term)

---

## License

MIT © Henry Ando
