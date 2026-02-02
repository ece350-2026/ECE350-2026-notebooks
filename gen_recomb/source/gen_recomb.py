# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "matplotlib==3.10.8",
#     "scipy==1.17.0",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
async def _():
    # Install plotly in WASM environment (Pyodide)
    import sys as _sys
    if "pyodide" in _sys.modules:
        import micropip
        _ = await micropip.install("plotly")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import scipy.constants as const
    import sys

    q = 1.6e-19  # C
    # WASM-compatible ASSET_DIR
    if "pyodide" in sys.modules:
        ASSET_DIR = Path(".")  # WASM: images in same directory
    else:
        ASSET_DIR = Path(__file__).parent  # Local: images in script directory

    mo.md(
        r"""
        # Carrier Transport: Generation and Recombination, Quasi-Fermi Levels

        This interactive notebook covers:

        1. Generation and recombination 
        2. Excess carriers
        3. Quasi-Fermi levels


        ---
        """
    )
    return ASSET_DIR, mo, np, plt


@app.cell
def _(ASSET_DIR, mo):
    _md = mo.md(r"""
    ## 1. Generation and Recombination of Carriers

    **Generation**: creates electron-hole pairs, increasing the carrier concentration.
    - Given as rate per unit volume per unit time, $G$ [cm⁻³s⁻¹]

    - Example causes: absorption of light (especially in direct bandgap semiconductors), thermal generation, etc.

    **Recombination**: destroys electron-hole pairs, decreasing the carrier concentration.
    - Given as rate per unit volume per unit time, $R$ [cm⁻³s⁻¹]

    - Example causes: direct recombination (radiative), indirect recombination (non-radiative), surface recombination, defect states (Schockley-Read-Hall), Auger recombination, etc.
    """)
    _img = mo.hstack([mo.image(src=str(ASSET_DIR / "recombination.png"), width=600)], justify="center")
    mo.vstack([_md, _img])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Excess Carriers

    Let:
    - $n_0$, $p_0$ = **equilibrium** electron and hole densities
    - $n'$, $p'$ = **excess** carriers (e.g., generated optically or thermally)

    Then the total electron and hole densities are:

    $$\boxed{n = n_0 + n' \qquad p = p_0 + p'}$$

    If charge neutrality is maintained, then $n' = p'$.

    ### Carrier Lifetime

    After the excitation is turned off, the excess carriers recombine to return the semiconductor to equilibrium.

    Assuming simple **exponential decay** (low-level injection):

    $$\boxed{\frac{dn'}{dt} = -\frac{n'}{\tau_n}} \qquad \boxed{\frac{dp'}{dt} = -\frac{p'}{\tau_p}}$$

    where $\tau_n$ and $\tau_p$ are the **carrier lifetimes** for electrons and holes.

    Recombination rate:

    $$R_n = n'/\tau_n \qquad R_p = p'/\tau_p$$

    Including generation:

    $$\frac{dn'}{dt} = -\frac{n'}{\tau_n} + G_n  \qquad \frac{dp'}{dt} = -\frac{p'}{\tau_p} + G_p$$

    At steady-state:

    $$n' = G_n\tau_n \qquad p' = G_p\tau_p$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Quasi-Fermi Levels

    - At thermal equilibrium, $n_0p_0 = n_i^2$
    - If $np > n_i^2$: System has **excess carriers** → $R > G$ to return to equilibrium
    - If $np < n_i^2$: System is in **deficit of carriers** → $G > R$ to return to equilibrium

    - Define **quasi-Fermi levels** $E_{Fn}$ and $E_{Fp}$ for electrons and hole concentrations:

    $$\boxed{n = N_c \exp\left(-\frac{E_c - E_{Fn}}{k_B T}\right) = n_i \exp\left(\frac{E_{Fn} - E_{Fi}}{k_B T}\right)}$$

    $$\boxed{p = N_v \exp\left(-\frac{E_{Fp} - E_v}{k_B T}\right) = n_i \exp\left(\frac{E_{Fi} - E_{Fp}}{k_B T}\right)}$$

    $$\boxed{np = n_i^2 \exp\left(\frac{E_{Fn} - E_{Fp}}{k_B T}\right)}$$

    ### Physical significance
    -   If you excite the semiconductor, the electrons and holes are no longer at equilibrium with each other
    - But electrons are at equilibrium with themselves, and holes are at equilibrium with themselves
        - Reasonable for > ps time-scales, because it takes < 1 ps for the electrons and holes to settle amongst themselves (through scattering mechanisms)
    - At thermal equilibrium, $E_{Fn} = E_{Fp} = E_F$
    - The greater $(E_{Fn} - E_{Fp})$ is, the further the semiconductor is from equilibrium
    """)
    return


@app.cell
def _(mo):
    # Sliders for generation rate and carrier lifetime
    generation_slider = mo.ui.slider(0, 21, value=10, step=0.5, label="log(G) [cm⁻³·s⁻¹] - Generation rate")
    tau_qf_slider = mo.ui.slider(-7, -4, value=-6, step=0.5, label="log(τ) [s] - Carrier lifetime")
    return generation_slider, tau_qf_slider


@app.cell
def _(generation_slider, mo, np, plt, tau_qf_slider):
    # Quasi-Fermi level splitting under optical excitation (uniform semiconductor)
    G_qf = 10**generation_slider.value  # Generation rate [cm⁻³·s⁻¹]
    tau_qf = 10**tau_qf_slider.value  # Carrier lifetime [s]
    tau_qf_us = tau_qf * 1e6  # Carrier lifetime in microseconds

    # Excess carrier concentration at steady state: Δn = Δp = G·τ
    delta_n_qf = G_qf * tau_qf

    # For n-type Si with N_D = 10^16 cm⁻³
    N_D_qf = 1e16
    n_i_qf = 1e10
    n_0_qf = N_D_qf
    p_0_qf = n_i_qf**2 / N_D_qf

    # Total concentrations
    n_total_qf = n_0_qf + delta_n_qf
    p_total_qf = p_0_qf + delta_n_qf

    # Quasi-Fermi level splitting
    # np = n_i² exp((E_Fn - E_Fp)/kT)
    # E_Fn - E_Fp = kT ln(np/n_i²)
    kT_qf = 0.026  # eV at 300K
    splitting_qf = kT_qf * np.log(n_total_qf * p_total_qf / n_i_qf**2)

    # Individual quasi-Fermi levels relative to E_i
    # n = n_i exp((E_Fn - E_i)/kT) → E_Fn - E_i = kT ln(n/n_i)
    E_Fn_minus_Ei = kT_qf * np.log(n_total_qf / n_i_qf)
    E_Fp_minus_Ei = -kT_qf * np.log(p_total_qf / n_i_qf)

    # Injection level
    if delta_n_qf < 0.1 * n_0_qf:
        injection = "Low-level injection (Δn ≪ n₀)"

    elif delta_n_qf < 10 * n_0_qf:
        injection = "Moderate injection"

    else:
        injection = "High-level injection (Δn ≫ n₀)"


    # Energy band diagram
    # Energy levels (eV) - using Si values
    E_g = 1.12  # Bandgap
    E_midgap = 0  # Reference point at midgap
    E_c = E_midgap + E_g / 2  # Conduction band edge
    E_v = E_midgap - E_g / 2  # Valence band edge

    # Intrinsic Fermi level: E_Fi = (E_c + E_v)/2 + (kT/2)*ln(N_v/N_c)
    # For Si at 300K: N_c = 2.8e19, N_v = 1.04e19
    N_c_dos = 2.8e19
    N_v_dos = 1.04e19
    E_Fi = (E_c + E_v) / 2 + (kT_qf / 2) * np.log(N_v_dos / N_c_dos)  # Slightly below midgap

    # Equilibrium Fermi level for n-type
    E_F_eq = E_Fi + kT_qf * np.log(n_0_qf / n_i_qf)

    # Quasi-Fermi levels (relative to E_Fi)
    E_Fn = E_Fi + E_Fn_minus_Ei
    E_Fp = E_Fi + E_Fp_minus_Ei

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0, 1, 100)

    # Draw bands
    ax.axhline(E_c, color='blue', linewidth=2, label=r'$E_c$')
    ax.axhline(E_v, color='red', linewidth=2, label=r'$E_v$')
    ax.axhline(E_Fi, color='green', linewidth=1, linestyle=':', label=r'$E_{Fi}$')

    # Fill bands
    ax.fill_between([0, 1], E_c, E_c + 0.2, alpha=0.1, color='blue')
    ax.fill_between([0, 1], E_v - 0.2, E_v, alpha=0.1, color='red')

    # Draw Fermi levels
    if delta_n_qf == 0:  # No generation
        # Show only equilibrium Fermi level
        ax.axhline(E_F_eq, color='black', linewidth=2, linestyle='--', label=r'$E_F$')
        ax.text(1.02, E_F_eq, r'$E_F$', fontsize=16, va='center', fontweight='bold')
        title_text = "Energy Band Diagram (Equilibrium)"
    else:
        # Show quasi-Fermi levels
        ax.axhline(E_Fn, color='blue', linewidth=2, linestyle='--', label=r'$E_{Fn}$')
        ax.axhline(E_Fp, color='red', linewidth=2, linestyle='--', label=r'$E_{Fp}$')
        ax.text(1.02, E_Fn, r'$E_{Fn}$', fontsize=16, va='center', color='blue', fontweight='bold')
        ax.text(1.02, E_Fp, r'$E_{Fp}$', fontsize=16, va='center', color='red', fontweight='bold')

        # Draw splitting arrow
        ax.annotate('', xy=(0.5, E_Fp), xytext=(0.5, E_Fn),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax.text(0.52, (E_Fn + E_Fp)/2, f'{splitting_qf*1000:.0f} meV', 
                fontsize=16, color='purple', va='center')
        title_text = "Energy Band Diagram (Under Illumination)"

    # Labels
    ax.text(1.02, E_c, r'$E_c$', fontsize=16, va='center', color='blue')
    ax.text(1.02, E_v, r'$E_v$', fontsize=16, va='center', color='red')
    ax.text(1.02, E_Fi, r'$E_{Fi}$', fontsize=16, va='center', color='green')

    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(E_v - 0.3, E_c + 0.3)
    ax.set_ylabel('Energy (eV)', fontsize=16)
    ax.set_title(title_text, fontsize=16, fontweight='bold')
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    mo.vstack([
        mo.md("## Interactive: Quasi-Fermi Level Calculator"),
        generation_slider,
        tau_qf_slider,
        mo.md(f"$\\tau = {tau_qf_us:.1f}$ μs"),
        mo.md(f"**n-type Si under uniform illumination** <br> ($N_D = 10^{{16}}$ cm⁻³, $n_i = 10^{{10}}$ cm⁻³, T = 300K, $G_n=G_p=G$, $\\tau_n=\\tau_p=\\tau$):"),
        mo.hstack([
            mo.md(
                f"""
    | Quantity | Equation | Value |
    |:---------|:---------|:---------------:|
    | Equilibrium electrons | $n_0 = N_D$ (n-type) | {n_0_qf:.1e} cm⁻³ |
    | Equilibrium holes | $p_0 = n_i^2 / N_D$ | {p_0_qf:.1e} cm⁻³ |
    | Excess carriers | $ n' = p' = G \\cdot \\tau$ | {delta_n_qf:.1e} cm⁻³ |
    | Total electrons | $n = n_0 + n'$ | {n_total_qf:.1e} cm⁻³ |
    | Total holes | $p = p_0 + p'$ | {p_total_qf:.1e} cm⁻³ |
    | Electron quasi-Fermi | $E_{{Fn}} - E_{{Fi}} = k_BT \\ln(n/n_i)$ | {E_Fn_minus_Ei*1000:.0f} meV |
    | Hole quasi-Fermi | $E_{{Fp}} - E_{{Fi}} = -k_BT \\ln(p/n_i)$ | {E_Fp_minus_Ei*1000:.0f} meV |
    | **Quasi-Fermi level splitting** | $E_{{Fn}} - E_{{Fp}} = k_BT \\ln(np/n_i^2)$ | **{splitting_qf*1000:.0f} meV** |
                """
            ),
            plt.gca()
        ], justify="space-around"),
        mo.md(
            f"""

    **Injection Level:** {injection}

    Ratio: $n' / n_0$ = {delta_n_qf/n_0_qf:.1e}

    | Injection Level | Condition | Meaning |
    |:---------|:-----------:|:------------------|
    | Low-level | $n' \\ll n_0$ | Majority carriers unchanged, minority carriers increase significantly |
    | Moderate | $n' \\sim n_0$ | Both carrier types significantly perturbed |
    | High-level | $n' \\gg n_0$ | Excess carriers dominate, $n \\approx p \\approx n'$ |
            """
        )
    ])
    return


if __name__ == "__main__":
    app.run()
