"""
bfmix  ·  Bose-Fermi Mixture Simulator
Streamlit web interface
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io, sys, time
from pathlib import Path

# ── make the package importable when running from the webapp/ directory ──────
sys.path.insert(0, str(Path(__file__).parent.parent))

from bfmix import BoseFermiMixture, ATOMS, Geometry
from bfmix.atoms import list_bosons, list_fermions, A0, HBAR
from bfmix.potentials import harmonic_trap, box_trap, gaussian_dimple, combined
from bfmix.utils.plotting import auto_plot, plot_convergence

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="bfmix · Bose-Fermi Mixture Simulator",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.03em;
}
.stButton > button {
    background: #0f3460;
    color: white;
    border: none;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.55rem 1.6rem;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #16213e;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
}
.metric-card {
    background: #f8f9fb;
    border-left: 3px solid #0f3460;
    padding: 0.6rem 1rem;
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}
.converged-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
}
.badge-yes { background: #d4edda; color: #155724; }
.badge-no  { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# ❄️ bfmix")
st.markdown("**Ultracold Bose-Fermi Mixture Simulator** · Thomas-Fermi / pseudo-GPE ground states via imaginary-time evolution")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: all input parameters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Parameters")

    # ── Species ─────────────────────────────────────────────────────────
    st.markdown("### Species")
    boson_key   = st.selectbox("Boson",   list_bosons(),   index=0)
    fermion_key = st.selectbox("Fermion", list_fermions(), index=0)

    boson_atom   = ATOMS[boson_key]
    fermion_atom = ATOMS[fermion_key]

    col1, col2 = st.columns(2)
    with col1:
        N_boson = st.number_input(
            "N_boson", min_value=100, max_value=10_000_000,
            value=50_000, step=1000, format="%d",
        )
    with col2:
        N_fermion = st.number_input(
            "N_fermion", min_value=100, max_value=10_000_000,
            value=20_000, step=1000, format="%d",
        )

    spin_degen = st.number_input(
        f"Fermion spin degeneracy g_s  (default for {fermion_key}: {fermion_atom.spin_degeneracy})",
        min_value=1, max_value=20,
        value=fermion_atom.spin_degeneracy, step=1,
    )

    # ── Interactions ────────────────────────────────────────────────────
    st.markdown("### Interactions")
    a_BB = st.number_input(
        f"a_BB  (Bohr radii)  [default: {boson_atom.a_scatt_bohr:.1f}]",
        value=float(boson_atom.a_scatt_bohr),
        format="%.1f",
    )
    a_BF = st.number_input(
        "a_BF  (Bohr radii)",
        value=-185.0, format="%.1f",
    )

    # ── Geometry ────────────────────────────────────────────────────────
    st.markdown("### Geometry")
    geo_type = st.selectbox(
        "Geometry",
        ["1D", "2D_planar", "2D_cyl", "3D"],
        index=0,
    )

    if geo_type == "1D":
        Nx     = st.slider("Grid points", 64, 512, 128, step=64)
        Lx_um  = st.slider("Box half-extent  (μm)", 5, 200, 50)
        points = Nx
        extent = Lx_um * 1e-6

    elif geo_type == "2D_planar":
        Nx    = st.slider("Grid points (x)", 32, 256, 64, step=32)
        Ny    = st.slider("Grid points (y)", 32, 256, 64, step=32)
        Lx_um = st.slider("Half-extent x  (μm)", 5, 200, 40)
        Ly_um = st.slider("Half-extent y  (μm)", 5, 200, 40)
        points = (Nx, Ny)
        extent = (Lx_um * 1e-6, Ly_um * 1e-6)

    elif geo_type == "2D_cyl":
        Nr    = st.slider("Grid points (r)", 32, 256, 64, step=32)
        Nz    = st.slider("Grid points (z)", 32, 256, 64, step=32)
        R_um  = st.slider("Radial extent R  (μm)", 5, 200, 40)
        Z_um  = st.slider("Axial half-extent Z  (μm)", 5, 200, 60)
        points = (Nr, Nz)
        extent = (R_um * 1e-6, Z_um * 1e-6)

    elif geo_type == "3D":
        Nx    = st.slider("Grid points (x)", 16, 128, 32, step=16)
        Ny    = st.slider("Grid points (y)", 16, 128, 32, step=16)
        Nz    = st.slider("Grid points (z)", 16, 128, 32, step=16)
        Lx_um = st.slider("Half-extent x  (μm)", 5, 100, 30)
        Ly_um = st.slider("Half-extent y  (μm)", 5, 100, 30)
        Lz_um = st.slider("Half-extent z  (μm)", 5, 100, 30)
        points = (Nx, Ny, Nz)
        extent = (Lx_um * 1e-6, Ly_um * 1e-6, Lz_um * 1e-6)

    # ── External Potentials ─────────────────────────────────────────────
    st.markdown("### External Potentials")

    def _potential_ui(label: str, mass: float, geo: str) -> dict:
        """Render potential controls and return parameter dict."""
        pot_type = st.selectbox(
            f"{label} potential type",
            ["Harmonic trap", "Box trap", "Harmonic + dimple"],
            key=f"pot_type_{label}",
        )
        params = {"type": pot_type}

        if pot_type in ("Harmonic trap", "Harmonic + dimple"):
            if geo in ("2D_cyl",):
                wr = st.number_input(f"{label} ωᵣ / 2π  (Hz)", value=50.0,
                                     min_value=0.1, key=f"wr_{label}")
                wz = st.number_input(f"{label} ωz / 2π  (Hz)", value=20.0,
                                     min_value=0.1, key=f"wz_{label}")
                params["omega"] = (2*np.pi*wr, 2*np.pi*wz)
            elif geo == "1D":
                wx = st.number_input(f"{label} ωx / 2π  (Hz)", value=50.0,
                                     min_value=0.1, key=f"wx_{label}")
                params["omega"] = (2*np.pi*wx,)
            elif geo == "2D_planar":
                wx = st.number_input(f"{label} ωx / 2π  (Hz)", value=50.0,
                                     min_value=0.1, key=f"wx_{label}")
                wy = st.number_input(f"{label} ωy / 2π  (Hz)", value=50.0,
                                     min_value=0.1, key=f"wy_{label}")
                params["omega"] = (2*np.pi*wx, 2*np.pi*wy)
            else:
                wx = st.number_input(f"{label} ωx / 2π  (Hz)", value=50.0,
                                     min_value=0.1, key=f"wx_{label}")
                wy = st.number_input(f"{label} ωy / 2π  (Hz)", value=50.0,
                                     min_value=0.1, key=f"wy_{label}")
                wz_v = st.number_input(f"{label} ωz / 2π  (Hz)", value=50.0,
                                       min_value=0.1, key=f"wz_{label}")
                params["omega"] = (2*np.pi*wx, 2*np.pi*wy, 2*np.pi*wz_v)
            params["mass"] = mass

        if pot_type == "Harmonic + dimple":
            depth_nK = st.number_input(
                f"{label} dimple depth  (nK)",
                value=100.0, min_value=0.0, key=f"dimple_depth_{label}",
            )
            waist_um = st.number_input(
                f"{label} dimple waist  (μm)",
                value=5.0, min_value=0.1, key=f"dimple_waist_{label}",
            )
            params["dimple_depth"] = depth_nK * 1.380649e-23 * 1e-9
            params["dimple_waist"] = waist_um * 1e-6

        if pot_type == "Box trap":
            walls_um = st.number_input(
                f"{label} wall half-width  (μm)",
                value=30.0, min_value=1.0, key=f"wall_{label}",
            )
            params["wall"] = walls_um * 1e-6

        return params

    with st.expander("Boson potential", expanded=True):
        boson_pot_params = _potential_ui("Boson", boson_atom.mass, geo_type)

    with st.expander("Fermion potential", expanded=True):
        fermion_pot_params = _potential_ui("Fermion", fermion_atom.mass, geo_type)

    # ── Solver settings ─────────────────────────────────────────────────
    st.markdown("### Solver")
    max_iter = st.slider("Max iterations", 500, 20000, 3000, step=500)
    tol      = st.select_slider(
        "Convergence tolerance",
        options=[1e-4, 1e-5, 1e-6, 1e-7],
        value=1e-5,
        format_func=lambda x: f"{x:.0e}",
    )
    mixing   = st.slider("Density mixing", 0.1, 1.0, 0.5, step=0.05)

    run_btn = st.button("▶  Run Simulation", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build potential callables from UI params
# ─────────────────────────────────────────────────────────────────────────────
def _build_potential(params: dict):
    t = params["type"]
    if t == "Harmonic trap":
        return harmonic_trap(params["mass"], params["omega"])
    elif t == "Box trap":
        return box_trap(params.get("wall"))
    elif t == "Harmonic + dimple":
        harm = harmonic_trap(params["mass"], params["omega"])
        dimple = gaussian_dimple(
            depth=params["dimple_depth"],
            waist=params["dimple_waist"],
        )
        return combined(harm, dimple)
    raise ValueError(f"Unknown potential type: {t}")

# ─────────────────────────────────────────────────────────────────────────────
# Main content area
# ─────────────────────────────────────────────────────────────────────────────
tab_sim, tab_theory, tab_about = st.tabs(
    ["🔬 Simulation", "📐 Theory", "ℹ️ About"]
)

with tab_theory:
    st.markdown("""
## Physical Model

### Imaginary-Time Evolution

Both species are evolved under a pseudo Gross-Pitaevskii equation in imaginary time τ = it:

$$-\\hbar \\frac{\\partial \\psi_\\sigma}{\\partial \\tau} = \\hat{H}_\\sigma \\psi_\\sigma$$

After each step the wavefunction is renormalised:

$$\\psi_\\sigma \\to \\psi_\\sigma \\sqrt{\\frac{N_\\sigma}{\\int |\\psi_\\sigma|^2 \\, dV}}$$

which enforces the particle-number constraint and implicitly determines the chemical potential μ.

---

### Bosonic Species (GPE)

$$\\hat{H}_B = -\\frac{\\hbar^2}{2m_B}\\nabla^2 + V_B^{\\rm ext} + g_{BB}|\\psi_B|^2 + g_{BF}|\\psi_F|^2$$

with $g_{BB} = 4\\pi\\hbar^2 a_{BB}/m_B$ in 3-D.

---

### Fermionic Species (pseudo-GPE / LDA-TF)

The kinetic pressure of a spin-polarised ideal Fermi gas is captured through a density-dependent nonlinear term:

$$\\hat{H}_F = -\\frac{\\hbar^2}{2m_F}\\nabla^2 + V_F^{\\rm ext} + E_F[n_F] + g_{BF}|\\psi_B|^2$$

**Local Fermi energy $E_F(n_F)$ by geometry:**

| Geometry | $E_F$ |
|---|---|
| 3-D | $\\dfrac{\\hbar^2}{2m_F}\\left(\\dfrac{6\\pi^2}{g_s}\\right)^{2/3} n_F^{2/3}$ |
| 2-D (planar & cylindrical) | $\\dfrac{\\hbar^2}{2m_F}\\dfrac{4\\pi}{g_s}\\, n_F$ |
| 1-D | $\\dfrac{\\hbar^2}{2m_F}\\dfrac{\\pi^2}{3 g_s^2}\\, n_F^2$ |

---

### Numerical Scheme

- Second-order central finite differences for the Laplacian
- Dirichlet boundary conditions (ψ = 0 at walls)
- Forward-Euler imaginary-time step with automatic step-size selection:
  $d\\tau = 0.05\\,\\hbar / E_{\\rm scale}$, where $E_{\\rm scale} = \\hbar^2 / (2m\\,dx^2)$
- Density mixing between iterations for stability
""")

with tab_about:
    st.markdown("""
## About bfmix

**bfmix** is an open-source Python package for simulating ultracold Bose-Fermi mixtures
in the Thomas-Fermi / local density approximation.

**Phase 1** (this release) solves for ground-state density profiles of a single boson
species and a single fermion species in four simulation geometries:

- 1-D (single axis)
- 2-D planar (x-y plane)
- 2-D cylindrical (r-z, with full azimuthal symmetry)
- 3-D Cartesian

**Source code:** [github.com/henryando/bfmix](https://github.com/henryando/bfmix)

**Installation:**
```bash
pip install bfmix
```

**Quick start:**
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
""")

# ─────────────────────────────────────────────────────────────────────────────
# Simulation tab
# ─────────────────────────────────────────────────────────────────────────────
with tab_sim:
    if not run_btn:
        st.info("Configure your simulation in the sidebar and click **▶ Run Simulation**.")

        # Show a quick preview of the current setup
        with st.expander("Current setup preview", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**Boson:** {boson_key}")
                st.markdown(f"m = {boson_atom.mass_amu:.3f} u")
                st.markdown(f"N = {N_boson:,}")
            with c2:
                st.markdown(f"**Fermion:** {fermion_key}")
                st.markdown(f"m = {fermion_atom.mass_amu:.3f} u")
                st.markdown(f"N = {N_fermion:,}")
                st.markdown(f"g_s = {spin_degen}")
            with c3:
                st.markdown(f"**Geometry:** {geo_type}")
                st.markdown(f"a_BB = {a_BB:.1f} a₀")
                st.markdown(f"a_BF = {a_BF:.1f} a₀")

    else:
        # ── Build objects ────────────────────────────────────────────────
        try:
            geo = Geometry(geo_type, points=points, extent=extent)
        except Exception as e:
            st.error(f"Geometry error: {e}")
            st.stop()

        V_B_fn = _build_potential(boson_pot_params)
        V_F_fn = _build_potential(fermion_pot_params)

        # ── Run ─────────────────────────────────────────────────────────
        progress_bar  = st.progress(0)
        status_text   = st.empty()
        residual_text = st.empty()

        residuals_live = []

        def _cb(it, res):
            residuals_live.append(res)
            frac = min(it / max_iter, 1.0)
            progress_bar.progress(frac)
            status_text.markdown(
                f"**Iteration {it} / {max_iter}** &nbsp;|&nbsp; "
                f"residual = `{res:.2e}`"
            )

        try:
            mix = BoseFermiMixture(
                boson         = boson_atom,
                fermion       = fermion_atom,
                geometry      = geo,
                N_boson       = float(N_boson),
                N_fermion     = float(N_fermion),
                V_boson       = V_B_fn,
                V_fermion     = V_F_fn,
                a_BB          = a_BB,
                a_BF          = a_BF,
            )
            # patch spin degeneracy
            mix.fermion = mix.fermion.__class__(
                name              = mix.fermion.name,
                mass_amu          = mix.fermion.mass_amu,
                statistics        = mix.fermion.statistics,
                spin_degeneracy   = spin_degen,
                a_scatt_bohr      = mix.fermion.a_scatt_bohr,
            )

            t0     = time.time()
            result = mix.solve(
                max_iter = max_iter,
                tol      = float(tol),
                mixing   = mixing,
                callback = _cb,
            )
            elapsed = time.time() - t0

        except Exception as e:
            st.error(f"Simulation error: {e}")
            import traceback; st.code(traceback.format_exc())
            st.stop()

        progress_bar.progress(1.0)
        conv_badge = (
            '<span class="converged-badge badge-yes">✓ Converged</span>'
            if result.converged else
            '<span class="converged-badge badge-no">✗ Not converged</span>'
        )
        status_text.markdown(
            f"Finished in **{elapsed:.1f} s** ({result.iterations} iterations) &nbsp; {conv_badge}",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Results metrics ──────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        kB = 1.380649e-23
        with m1:
            st.metric("N_boson (actual)",   f"{result.N_boson:.4g}")
        with m2:
            st.metric("N_fermion (actual)", f"{result.N_fermion:.4g}")
        with m3:
            mu_B_nK = result.mu_boson / (kB * 1e-9)
            st.metric("μ_boson", f"{mu_B_nK:.2f} nK·k_B")
        with m4:
            mu_F_nK = result.mu_fermion / (kB * 1e-9)
            st.metric("μ_fermion", f"{mu_F_nK:.2f} nK·k_B")

        st.markdown("---")

        # ── Density plots ────────────────────────────────────────────────
        col_plot, col_conv = st.columns([3, 1])

        with col_plot:
            st.markdown("### Density profiles")
            fig_density = auto_plot(result)
            st.pyplot(fig_density, use_container_width=True)
            plt.close(fig_density)

        with col_conv:
            st.markdown("### Convergence")
            if result.residuals:
                fig_conv = plot_convergence(result)
                st.pyplot(fig_conv, use_container_width=True)
                plt.close(fig_conv)
            else:
                st.info("No convergence data.")

        # ── Summary text ─────────────────────────────────────────────────
        with st.expander("Full result summary"):
            st.code(result.summary(), language=None)

        # ── Download density arrays ──────────────────────────────────────
        st.markdown("### Download results")
        dl1, dl2 = st.columns(2)
        with dl1:
            buf = io.BytesIO()
            np.save(buf, result.density_boson)
            st.download_button(
                "⬇️  Boson density (.npy)",
                buf.getvalue(),
                file_name="density_boson.npy",
                mime="application/octet-stream",
            )
        with dl2:
            buf2 = io.BytesIO()
            np.save(buf2, result.density_fermion)
            st.download_button(
                "⬇️  Fermion density (.npy)",
                buf2.getvalue(),
                file_name="density_fermion.npy",
                mime="application/octet-stream",
            )
