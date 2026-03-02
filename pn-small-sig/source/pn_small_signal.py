# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "matplotlib==3.10.8",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/pn_small_signal.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    try:
        _test = Path(__file__).parent / "images"
        if _test.exists():
            ASSET_DIR = Path(__file__).parent
        else:
            raise FileNotFoundError
    except Exception:
        ASSET_DIR = None

    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/pn-small-sig/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    q = 1.6e-19  # C
    kT = 0.02585  # eV at 300K
    ni_Si = 1.1e10  # cm^-3 for Si at 300K
    eps_s = 11.7 * 8.854e-14  # F/cm (Si permittivity)

    mo.md(
        r"""
        # PN Junction: Small-Signal Models

        ECE350, Lecture 22

        This interactive notebook covers the small-signal AC behaviour of a PN junction diode:

        1. Small-signal analysis setup
        2. Reverse bias small-signal model
        3. Forward bias small-signal model
        4. Interactive: capacitance and impedance vs. voltage
        """
    )
    return ASSET_DIR, eps_s, kT, mo, ni_Si, np, plt, q


@app.cell
def _(mo):
    mo.md(r"""
    ## Small-Signal Analysis Setup

    We apply a total voltage consisting of a DC bias plus a small AC perturbation:

    $$V = V_{DC} + v(t)$$

    where $|v| \ll |V_{DC}|$ and the AC signal is sinusoidal:

    $$v(t) = v_{ac} \sin(\omega t)$$

    This allows us to characterize the **frequency response** of the diode near its operating point.

    ### Phasor (Complex) Notation

    Using phasors ($v(t) = \hat{V}e^{j\omega t}$, $i(t) = \hat{I}e^{j\omega t}$), we define:

    - **Impedance:** $Z = \dfrac{\hat{V}}{\hat{I}}$

    - **Admittance:** $Y = \dfrac{1}{Z} = G + j\omega C$

    where:

    - $G$ is the **small-signal conductance**
    - $C$ is the capacitance

    Both $G$ and $C$ are evaluated at the DC operating point $V_{DC}$.

    Since $\hat{I} = Y \hat{V}$, $Y$ is the transfer function from the small-signal voltage amplitude to the current.
    """)
    return


@app.cell
def _(ASSET_DIR, mo):
    _header = mo.md(r"""## Reverse Bias Small-Signal Model""")
    _text = mo.md(r"""
    Under reverse bias, the small-signal model simplifies considerably:

    **Capacitance:** Only the depletion capacitance remains, since there is negligible minority carrier injection:

    $$C = C_{dep} = A\,\frac{\varepsilon_s}{W_{dep}}$$

    **Conductance:** Starting from the diode equation $I = I_0(e^{qV/k_BT} - 1)$:

    $$G = \frac{dI}{dV}\bigg|_{V_{DC}} = \frac{q}{k_BT}(I_{DC} + I_0)$$

    For reverse bias, $I_{DC} \approx -I_0$, so:

    $$G \approx 0$$



    """)
    _img = mo.image(src=str(ASSET_DIR / "images/smallsig_revbias.png"), width="100%")
    mo.vstack([
        _header,
        mo.hstack([_text, _img], widths=[0.7, 0.3], align="start"),
    ])
    return


@app.cell
def _(ASSET_DIR, mo):
    _header = mo.md(r"""## Forward Bias Small-Signal Model""")
    _text = mo.md(r"""
    In addition to depletion capacitance, we must account for **diffusion capacitance** due to minority carriers injected into the N and P regions.

    **Diffusion capacitance:**

    $$C_{\text{diff}} = \frac{dQ}{dV}\bigg|_{V_{DC}} = \tau_s \frac{dI}{dV}\bigg|_{V_{DC}} \approx \tau_s \frac{qI_{DC}}{kT}$$

    where $\tau_s$ is the **recombination lifetime** (charge storage time). For sufficiently high $I_{DC}$, $C_{\text{diff}}$ dominates over the depletion capacitance.

    The current is related to the stored charge by:

    $$I = \frac{Q_P}{\tau_p} + \frac{|Q_N|}{\tau_n} \sim \frac{Q}{\tau_s}$$

    i.e., the rate of charge injection into the P and N sides is the current.

    **Conductance** ($I \gg I_0$ in forward bias):

    $$G = \frac{q}{kT}(I_{DC} + I_0) \approx \frac{q}{kT}\,I_{DC}$$
    """)
    _img = mo.image(src=f"{IMAGE_BASE}/smallsig_forwardbias.png", width="100%")
    mo.vstack([
        _header,
        mo.hstack([_text, _img], widths=[0.6, 0.4], align="start"),
    ])
    return


@app.cell
def _(mo):
    V_DC_slider = mo.ui.slider(
        start=-1.0,
        stop=0.7,
        step=0.05,
        value=0.0,
        label=r"$V_{DC}$ (V)",
        show_value=True,
    )
    log_Na_slider = mo.ui.slider(
        start=15,
        stop=19,
        step=0.5,
        value=17,
        label=r"log($N_a$) [cm⁻³]",
        show_value=True,
    )
    log_Nd_slider = mo.ui.slider(
        start=15,
        stop=19,
        step=0.5,
        value=16,
        label=r"log($N_d$) [cm⁻³]",
        show_value=True,
    )
    log_Is_slider = mo.ui.slider(
        start=-12,
        stop=-8,
        step=0.5,
        value=-10,
        label=r"log₁₀($I_0$) [A]",
        show_value=True,
    )
    tau_s_slider = mo.ui.slider(
        start=1,
        stop=100,
        step=1,
        value=10,
        label=r"$\tau_s$ (ns)",
        show_value=True,
    )
    controls = mo.vstack([
        mo.md("### Interactive Controls"),
        mo.hstack([V_DC_slider, log_Na_slider, log_Nd_slider]),
        mo.hstack([log_Is_slider, tau_s_slider]),
    ])
    return (
        V_DC_slider,
        controls,
        log_Is_slider,
        log_Na_slider,
        log_Nd_slider,
        tau_s_slider,
    )


@app.cell
def _(
    V_DC_slider,
    controls,
    eps_s,
    kT,
    log_Is_slider,
    log_Na_slider,
    log_Nd_slider,
    mo,
    ni_Si,
    np,
    plt,
    q,
    tau_s_slider,
):
    _V_DC = V_DC_slider.value
    _Na = 10 ** log_Na_slider.value
    _Nd = 10 ** log_Nd_slider.value
    _Is = 10 ** log_Is_slider.value
    _tau_s = tau_s_slider.value * 1e-9  # convert ns to s

    # Built-in potential
    _phi_bi = kT * np.log(_Na * _Nd / ni_Si**2)

    # Voltage array for plotting (reverse to near forward)
    _V_arr = np.linspace(-1, min(0.95 * _phi_bi, 0.7), 500)

    # Junction capacitance per unit area vs voltage
    _CJ_arr = np.sqrt(q * eps_s * _Na * _Nd / (2.0 * (_Na + _Nd) * (_phi_bi - _V_arr)))

    # Convert I_0 to J_0 using cross-section area of 10 um x 1 um
    _A_cross = 10e-4 * 1e-4  # 10 um x 1 um in cm^2
    _Js_eff = _Is / _A_cross  # A/cm^2

    _exp_term = np.exp(np.clip(_V_arr / kT, None, 500))
    _G_per_area = (1.0 / kT) * _Js_eff * _exp_term  # S/cm^2
    _Cd_arr2 = _tau_s * _G_per_area  # F/cm^2

    # Total capacitance
    _Ctotal_arr = _CJ_arr + _Cd_arr2

    # Create the figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.semilogy(_V_arr, _CJ_arr * 1e9, "b-", linewidth=2, label=r"$C_J$ (junction)")
    ax1.semilogy(_V_arr, _Cd_arr2 * 1e9, "r-", linewidth=2, label=r"$C_d$ (diffusion)")
    ax1.semilogy(_V_arr, _Ctotal_arr * 1e9, "k--", linewidth=2, label=r"$C_{total} = C_J + C_d$")

    # Mark the operating point
    if -1 <= _V_DC <= min(0.95 * _phi_bi, 0.7):
        _CJ_op = np.sqrt(q * eps_s * _Na * _Nd / (2.0 * (_Na + _Nd) * (_phi_bi - _V_DC)))
        _Cd_op = _tau_s * (1.0 / kT) * _Js_eff * np.exp(np.clip(_V_DC / kT, None, 500))
        ax1.axvline(_V_DC, color="gray", linestyle=":", alpha=0.5)
        ax1.plot(_V_DC, _CJ_op * 1e9, "bo", markersize=10, zorder=5)
        ax1.plot(_V_DC, _Cd_op * 1e9, "ro", markersize=10, zorder=5)


    ax1.set_xlabel(r"Applied Voltage $V_A$ (V)", fontsize=14)
    ax1.set_ylabel(r"Capacitance per unit area (nF/cm$^2$)", fontsize=14)
    ax1.set_title(
        f"Junction and Diffusion Capacitance vs. Voltage\n"
        f"($N_A = {_Na:.1e}$, $N_D = {_Nd:.1e}$, "
        rf"$\phi_{{bi}} = {_phi_bi:.3f}$ V, "
        rf"$\tau_s = {_tau_s*1e9:.0f}$ ns)",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(fontsize=12, loc="upper left")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(-1, min(0.95 * _phi_bi, 0.7))
    plt.tight_layout()

    _info = mo.md(
        f"""
        **At $V_{{DC}} = {_V_DC:.2f}$ V:**
        $\\phi_{{bi}}$ = {_phi_bi:.3f} V,
        $C_J$ = {_CJ_op*1e9:.3f} nF/cm$^2$,
        $C_d$ = {_Cd_op*1e9:.3e} nF/cm$^2$
        """
        if -1 <= _V_DC <= min(0.95 * _phi_bi, 0.7)
        else f"**$V_{{DC}} = {_V_DC:.2f}$ V is outside valid range.**"
    )

    _conversion_note = mo.md(
        f"Cross-section A = 10 μm × 1 μm = {_A_cross:.0e} cm²"
        f" → J₀ = I₀ / A = {_Js_eff:.2e} A/cm²"
    )

    mo.vstack([
        mo.md("## Interactive: Capacitance vs. Voltage"),
        controls,
        _conversion_note,
        plt.gca(),
        _info,
    ])
    return


if __name__ == "__main__":
    app.run()
