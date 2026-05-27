---
title: bfmix
emoji: ❄️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: Ultracold Bose-Fermi Mixture Simulator
---

# ❄️ bfmix

**Ultracold Bose-Fermi Mixture Simulator**

A Python package for computing Thomas-Fermi / pseudo-GPE ground states of ultracold Bose-Fermi mixtures using imaginary-time evolution within the local density approximation.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/henryando/bfmix)

---

## Overview

`bfmix` solves for the ground-state density profiles of a coupled **boson + fermion** ultracold gas. Both species are treated with a unified imaginary-time evolution (ITE) framework:

- **Bosons** — Gross-Pitaevskii equation (GPE) with contact interactions
- **Fermions** — Pseudo-GPE where the nonlinearity is the local Thomas-Fermi Fermi energy, capturing the kinetic pressure of a degenerate Fermi gas in the LDA

### Supported geometries

| Code | Description |
|------|-------------|
| `"1D"` | One spatial dimension |
| `"2D_planar"` | Two-dimensional x-y plane |
| `"2D_cyl"` | Cylindrical symmetry, full (r, z) grid |
| `"3D"` | Three-dimensional Cartesian |

---

## Physical Model

### Bosons (GPE)

$$-\hbar\frac{\partial\psi_B}{\partial\tau} = \left[-\frac{\hbar^2}{2m_B}\nabla^2 + V_B^{\rm ext} + g_{BB}|\psi_B|^2 + g_{BF}|\psi_F|^2\right]\psi_B$$

### Fermions (pseudo-GPE / LDA-TF)

$$-\hbar\frac{\partial\psi_F}{\partial\tau} = \left[-\frac{\hbar^2}{2m_F}\nabla^2 + V_F^{\rm ext} + E_F[n_F] + g_{BF}|\psi_B|^2\right]\psi_F$$

| Geometry | $E_F(n_F)$ |
|---|---|
| 3-D | $\frac{\hbar^2}{2m_F}\left(\frac{6\pi^2}{g_s}\right)^{2/3} n_F^{2/3}$ |
| 2-D | $\frac{\hbar^2}{2m_F}\frac{4\pi}{g_s} n_F$ |
| 1-D | $\frac{\hbar^2}{2m_F}\frac{\pi^2}{3g_s^2} n_F^2$ |

---

## Installation

```bash
pip install bfmix
# with webapp:
pip install "bfmix[webapp]"
```

## Quick Start

```python
import numpy as np
from bfmix import BoseFermiMixture, ATOMS, Geometry
from bfmix.potentials import harmonic_trap

geo = Geometry("1D", points=256, extent=50e-6)
mix = BoseFermiMixture(
    boson=ATOMS["87Rb"], fermion=ATOMS["40K"],
    geometry=geo,
    N_boson=50_000, N_fermion=20_000,
    V_boson=harmonic_trap(ATOMS["87Rb"].mass, (2*np.pi*50,)),
    V_fermion=harmonic_trap(ATOMS["40K"].mass, (2*np.pi*60,)),
    a_BB=100.4, a_BF=-185.0,
)
result = mix.solve()
print(result.summary())
```

## Running the Webapp Locally

```bash
pip install "bfmix[webapp]"
streamlit run webapp/app.py
```

## Source Code

[github.com/henryando/bfmix](https://github.com/henryando/bfmix)

## License

MIT © Henry Ando
