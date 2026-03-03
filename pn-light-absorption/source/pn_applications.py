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
app = marimo.App(width="medium")


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

    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/pn-light-absorption/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    # Physical constants
    q = 1.6e-19       # C
    k = 1.381e-23     # J/K
    h = 6.626e-34     # J·s
    c = 3.0e8         # m/s
    kT_300 = 0.02585  # eV at 300K

    mo.md(
        r"""
        # PN Junction: Light absorption

        **ECE350, Lecture 23**

        This interactive notebook explores how optical absorption changes the IV relation of the PN junction. This has applications in photodetectors and solar cells.
        """
    )
    return IMAGE_BASE, k, mo, np, plt, q


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(r"""### Light Absorption in a PN Junction""")
    _text = mo.md(r"""
    - Absorbed photons with $E_{ph} = hc/\lambda > E_g$ create electron-hole pairs

    - If the electrons and holes are created in and near the depletion layer, the built-in $\mathcal{E}$-field sweeps out the generated carriers

    - This creates a current, called **short circuit current**, $I_{SC}$
        - $I_{SC}$ is when load has zero impedance

    $$I = I_0(e^{qV/kT} - 1) - I_{SC}$$
    """)
    _img = mo.image(src=f"{IMAGE_BASE}/light-abs-PN.png", width="100%")
    mo.vstack([
        _header,
        mo.hstack([_text, _img], widths=[0.5, 0.5], align="start"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Derivation: I-V with Optical Generation

    We modify the continuity equation by adding a uniform generation rate $G$ [s$^{-1}$cm$^{-3}$] for both electrons and holes:

    $$\frac{dJ_p}{dx} = -q\frac{p'}{\tau_p} + qG$$

    For diffusion current $J_p = -qD_p \dfrac{dp}{dx}$, the minority carrier (holes on the N-side) continuity equation becomes:

    $$\boxed{\frac{d^2 p'_N}{dx^2} = \frac{p'_N}{L_p^2} - \frac{G}{D_p}}$$

    Similarly for electrons on the P-side:

    $$\boxed{\frac{d^2 n'_P}{dx^2} = \frac{n'_P}{L_n^2} - \frac{G}{D_n}}$$

    ---

    **Finding $I_{SC}$** (set $V = 0$ across the diode):

    $I_{SC}$ is the short circuit current.

    Solve for holes on the N-side ($x_n < x < \infty$) with boundary conditions:

    $$p'_N(x_n) = p_{N0}(e^{qV/kT} - 1) = 0 \quad \text{(at } V = 0\text{)}$$

    $$p'_N(\infty) = \frac{G}{D_p}L_p^2 = \tau_p G \quad \left(\text{set } \frac{d^2 p'}{dx^2} = 0\right)$$

    The solution is:

    $$p'_N(x) = \tau_p G\!\left(1 - e^{-(x - x_n)/L_p}\right)$$

    The hole diffusion current at $x = x_n$:

    $$J_{p,\text{diff}}(x_n) = -qD_p \frac{dp'_N}{dx}\bigg|_{x_n} = q\frac{D_p}{L_p}\tau_p G = qL_p G$$

    Therefore:

    $$\boxed{I_{p,SC} = AqL_p G}$$

    Similarly for electrons: $I_{n,SC} = AqL_n G$. The total short-circuit current is:

    $$\boxed{I_{SC} = Aq(L_p + L_n)G}$$

    ---

    **Full I-V relation under illumination:**

    By superposition of the dark diode current and the photocurrent:

    $$\boxed{I = I_0\!\left(e^{qV/kT} - 1\right) - Aq(L_p + L_n)G}$$

    where the second term is $I_{SC}$.

    **Open-circuit voltage** (set $I = 0$):

    $$\boxed{V_{OC} \approx \frac{kT}{q}\ln\!\left[\frac{AqG}{I_0}(L_p + L_n)\right]}$$

    where $I_0 = qA\!\left(\dfrac{D_n n_i^2}{L_n N_a} + \dfrac{D_p n_i^2}{L_p N_d}\right)$.
    """)
    return


@app.cell
def _(mo):
    intensity_slider = mo.ui.slider(
        start=0.1, stop=5.0, step=0.1, value=1.0,
        label=r"Intensity (mW/mm²)", show_value=True,
    )
    wavelength_slider = mo.ui.slider(
        start=400, stop=1100, step=10, value=800,
        label=r"$\lambda$ (nm)", show_value=True,
    )
    absorption_slider = mo.ui.slider(
        start=5, stop=100, step=5, value=80,
        label=r"Absorption (%)", show_value=True,
    )
    Ldiff_slider = mo.ui.slider(
        start=10, stop=500, step=10, value=200,
        label=r"$L_p = L_n$ (μm)", show_value=True,
    )
    I0_slider = mo.ui.slider(
        start=-12, stop=-8, step=0.5, value=-10,
        label=r"log₁₀($I_0$) [A]", show_value=True,
    )
    T_slider = mo.ui.slider(
        start=250, stop=400, value=300, step=5,
        label=r"$T$ (K)", show_value=True,
    )
    solar_controls = mo.vstack([
        mo.md("### Interactive Controls"),
        mo.hstack([intensity_slider, wavelength_slider, absorption_slider]),
        mo.hstack([Ldiff_slider, I0_slider, T_slider]),
    ])
    return (
        I0_slider,
        Ldiff_slider,
        T_slider,
        absorption_slider,
        intensity_slider,
        solar_controls,
        wavelength_slider,
    )


@app.cell
def _(
    I0_slider,
    Ldiff_slider,
    T_slider,
    absorption_slider,
    intensity_slider,
    k,
    mo,
    np,
    plt,
    q,
    solar_controls,
    wavelength_slider,
):
    _lam_nm = wavelength_slider.value
    _lam_m = _lam_nm * 1e-9
    _intensity_mW_mm2 = intensity_slider.value
    _f_abs = absorption_slider.value / 100.0
    _L_diff_cm = Ldiff_slider.value * 1e-4
    _I0 = 10 ** I0_slider.value
    _T = T_slider.value
    _Vt = k * _T / q

    _h = 6.626e-34
    _c = 3.0e8
    _E_ph_J = _h * _c / _lam_m
    _E_ph_eV = 1240.0 / _lam_nm

    # Si absorption coefficient: lookup table with interpolation
    _alpha_lam = np.array([350, 400, 450, 500, 550, 600, 650, 700,
                           750, 800, 850, 900, 950, 1000, 1050, 1100])
    _alpha_val = np.array([1e6, 1e5, 3e4, 1e4, 6e3, 4e3, 3e3, 2e3,
                           1e3, 800, 400, 150, 50, 20, 5, 1])
    _log_alpha = np.interp(_lam_nm, _alpha_lam, np.log10(_alpha_val))
    _alpha = 10 ** _log_alpha  # cm⁻¹
    _L_abs_cm = 1.0 / _alpha

    # Intensity: 1 mW/mm² = 0.1 W/cm²
    _P_opt_W_cm2 = _intensity_mW_mm2 * 0.1

    # Photon flux: Φ = P_opt / E_ph
    _Phi = _P_opt_W_cm2 / _E_ph_J  # photons/cm²/s

    # Average generation rate: G = (f_abs × Φ / L)(1 - e^{-αL}), L = 1/α
    _L = _L_abs_cm  # absorption length = 1/α
    _G = (_f_abs * _Phi / _L) * (1 - np.exp(-_alpha * _L))  # cm⁻³ s⁻¹

    # Device area = 1 mm² = 1e-2 cm²
    _A = 1e-2  # cm²

    # I_SC = A q (L_p + L_n) G, with L_p = L_n = L_diff
    _Isc = _A * q * (2 * _L_diff_cm) * _G

    _Voc = _Vt * np.log(_Isc / _I0 + 1) if _Isc > 0 else 0

    # --- I-V curve ---
    _V = np.linspace(-0.5, 0.8, 1000)
    _exp_term = np.exp(np.clip(_V / _Vt, -40, 40)) - 1
    _I_dark = _I0 * _exp_term
    _I_illum = _I0 * _exp_term - _Isc

    _V_pos = np.linspace(0, 0.8, 1000)
    _exp_pos = np.exp(np.clip(_V_pos / _Vt, 0, 40)) - 1
    _I_illum_pos = _I0 * _exp_pos - _Isc
    _P_curve = -_I_illum_pos * _V_pos
    _idx_mpp = np.argmax(_P_curve)
    _V_mpp = _V_pos[_idx_mpp]
    _I_mpp = _I_illum_pos[_idx_mpp]
    _P_max = _P_curve[_idx_mpp]
    _FF = _P_max / (_Voc * _Isc) if (_Voc * _Isc) > 0 else 0

    _y_upper = max(_Isc * 1e3 * 0.3, 5)
    _fig, _ax = plt.subplots(figsize=(10, 6))

    _I_dark_display = np.clip(_I_dark * 1e3, -100, _y_upper * 2)
    _ax.plot(_V * 1e3, _I_dark_display, "b--", linewidth=1.5, alpha=0.6, label="Dark I–V")
    _ax.plot(_V * 1e3, _I_illum * 1e3, "b-", linewidth=2.5, label="Illuminated I–V")

    _ax.plot(_Voc * 1e3, 0, "ro", markersize=10, zorder=5)
    _ax.plot(0, -_Isc * 1e3, "go", markersize=10, zorder=5)

    if _P_max > 0:
        _ax.plot(_V_mpp * 1e3, _I_mpp * 1e3, "m*", markersize=15, zorder=5, label="Maximum power")
        _rect = plt.Rectangle(
            (0, _I_mpp * 1e3), _V_mpp * 1e3, -_I_mpp * 1e3,
            alpha=0.15, color="orange", label=f"$P_{{max}}$ = {_P_max * 1e3:.2e} mW",
        )
        _ax.add_patch(_rect)

    _ax.axhline(0, color="gray", linewidth=0.5)
    _ax.axvline(0, color="gray", linewidth=0.5)

    _y_min = min(min(_I_illum * 1e3) * 1.15, -1)
    _ax.set_xlim(-500, 800)
    _ax.set_ylim(_y_min, _y_upper)

    _ax.fill_between([0, 800], _y_min, 0, alpha=0.06, color="green", zorder=0)
    _ax.text(
        _Voc * 1e3 * 0.45 if _Voc > 0 else 200, _y_min * 0.55,
        "Photovoltaic\n(power generation)",
        fontsize=13, fontweight="bold", color="green",
        ha="center", va="center", alpha=0.7,
    )
    _ax.fill_between([-500, 0], _y_min, 0, alpha=0.06, color="purple", zorder=0)
    _ax.text(
        -250, _y_min * 0.55,
        "Photodetector\n(reverse bias)",
        fontsize=13, fontweight="bold", color="purple",
        ha="center", va="center", alpha=0.7,
    )

    _ax.set_xlabel("Voltage (mV)", fontsize=16)
    _ax.set_ylabel("Current (mA)", fontsize=16)
    _ax.set_title(
        r"Illuminated Si Diode: $I = I_0(e^{qV/kT} - 1) - I_{SC}$",
        fontsize=14, fontweight="bold",
    )
    _ax.text(
        0.98, 0.12,
        f"● $V_{{OC}}$ = {_Voc * 1e3:.1f} mV",
        transform=_ax.transAxes, fontsize=12, fontweight="bold",
        ha="right", va="bottom", color="red",
    )
    _ax.text(
        0.98, 0.05,
        f"● $I_{{SC}}$ = {_Isc * 1e3:.2f} mA",
        transform=_ax.transAxes, fontsize=12, fontweight="bold",
        ha="right", va="bottom", color="green",
    )
    _ax.legend(fontsize=11, loc="lower left")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()

    _below_gap = " **(below Si bandgap!)**" if _E_ph_eV < 1.12 else ""
    _beer_lambert_factor = 1 - np.exp(-_alpha * _L)

    _info = mo.md(
        f"""
        **Assumed parameters:** Device area $A$ = 1 mm² = 10⁻² cm², $L_p = L_n$ = {Ldiff_slider.value} μm

        **Equations used:**

        $$\\Phi = \\frac{{P_{{opt}}}}{{E_{{ph}}}} = \\frac{{P_{{opt}}}}{{hc/\\lambda}}$$

        $$G_{{avg}} = \\frac{{f_{{abs}} \\cdot \\Phi}}{{L}}\\left(1 - e^{{-\\alpha L}}\\right), \\quad L = 1/\\alpha$$

        $$I_{{SC}} = Aq(L_p + L_n)\\,G_{{avg}}, \\quad V_{{OC}} = \\frac{{kT}}{{q}}\\ln\\!\\left(\\frac{{I_{{SC}}}}{{I_0}} + 1\\right)$$

        | Quantity | Expression | Value |
        |:---------|:-----------|------:|
        | Photon energy $E_{{ph}}$ | $hc/\\lambda$ | {_E_ph_eV:.3f} eV{_below_gap} |
        | Si absorption coeff. $\\alpha$ | lookup at $\\lambda$ = {_lam_nm} nm | {_alpha:.2e} cm⁻¹ |
        | Absorption length $L = 1/\\alpha$ | | {_L_abs_cm*1e4:.1f} μm |
        | Photon flux $\\Phi$ | $P_{{opt}} / E_{{ph}}$ | {_Phi:.2e} cm⁻² s⁻¹ |
        | Beer–Lambert factor | $1 - e^{{-\\alpha L}}$ | {_beer_lambert_factor:.4f} |
        | Generation rate $G_{{avg}}$ | $(f_{{abs}} \\cdot \\Phi / L)(1 - e^{{-\\alpha L}})$ | {_G:.2e} cm⁻³ s⁻¹ |
        | $I_{{SC}}$ | $Aq(L_p + L_n)G$ | {_Isc*1e3:.3f} mA |
        | $V_{{OC}}$ | $(kT/q)\\ln(I_{{SC}}/I_0 + 1)$ | {_Voc*1e3:.1f} mV |
        | $P_{{max}}$ | | {_P_max*1e3:.3e} mW |
        | Fill factor $FF$ | $P_{{max}}/(V_{{OC}} \\cdot I_{{SC}})$ | {_FF:.3f} |
        """
    )

    plt.close(_fig)
    mo.vstack([solar_controls, mo.as_html(_fig), _info])
    return


if __name__ == "__main__":
    app.run()
