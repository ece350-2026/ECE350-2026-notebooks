# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "matplotlib==3.10.8",
# ]
# ///

import marimo

__generated_with = "0.19.11"
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
    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/pn-revbias/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    q = 1.6e-19  # C
    kT = 0.02585  # eV at 300K
    ni_Si = 1.1e10  # cm^-3 for Si at 300K
    eps_r = 11.7  # relative permittivity of Si
    eps_0 = 8.854e-14  # F/cm (vacuum permittivity in CGS)
    eps_s = eps_r * eps_0  # F/cm
    Eg_Si = 1.12  # eV

    mo.md(
        r"""
        # PN Junction Under Reverse Bias

        ECE350, Lecture 18

        This notebook covers the electrostatics and capacitance of a PN junction under reverse bias:

        1. Applied voltage conventions
        2. Reverse bias: effect on the depletion region
        3. Electrostatics under bias: $\rho(x)$, $\mathcal{E}(x)$, $V(x)$, and energy bands
        4. Junction capacitance and C-V characteristics
        5. Breakdown mechanisms
        """
    )
    return ASSET_DIR, Eg_Si, IMAGE_BASE, eps_s, kT, mo, ni_Si, np, plt, q


@app.cell
def _(IMAGE_BASE, mo):
    _text1 = mo.md(r"""
    ## Applied Voltage $V_A$

    We apply a voltage $V_A$ across the PN junction, referenced to the **p-side relative to the n-side**:

    - **Forward bias**: $V_A > 0$ (p-side is at higher potential)
    - **Reverse bias**: $V_A < 0$ (p-side is at lower potential)
    """)
    _img = mo.hstack([mo.image(src=f"{IMAGE_BASE}/lec15-04.png", width="30%")], justify="center")
    _text2 = mo.md(r"""

    We assume that the applied voltage is dropped **entirely across the depletion region**. This is valid when:

    1. The **low-level injection** condition holds (excess minority carriers $\ll$ majority carrier concentration)
    2. There is **no significant series resistance** (contact resistance, bulk resistance)
    3. For forward bias, $V_A < \phi_{bi}$ (otherwise we leave the low-level injection regime)

    Under these assumptions, the voltage across the depletion region is $(\phi_{bi} - V_A)$.
    """)
    mo.vstack([_text1, _img, _text2])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Electrostatics Under Reverse Bias

    All equilibrium expressions remain valid with the substitution $\phi_{bi} \to (\phi_{bi} - V_A)$:

    ### Depletion Widths

    $$\boxed{W = \sqrt{\frac{2\varepsilon_s}{q}\cdot\frac{N_A + N_D}{N_A N_D}\cdot(\phi_{bi} - V_A)}}$$

    $$\boxed{x_n = \sqrt{\frac{2\varepsilon_s}{q}\cdot\frac{N_A}{N_D(N_A + N_D)}\cdot(\phi_{bi} - V_A)}}$$

    $$\boxed{x_p = \sqrt{\frac{2\varepsilon_s}{q}\cdot\frac{N_D}{N_A(N_A + N_D)}\cdot(\phi_{bi} - V_A)}}$$

    ### Maximum Electric Field (at $x = 0$)

    $$\boxed{|\mathcal{E}_{max}| = \frac{qN_A x_p}{\varepsilon_s} = \frac{qN_D x_n}{\varepsilon_s} = \sqrt{\frac{2q}{\varepsilon_s}\cdot\frac{N_A N_D}{N_A + N_D}\cdot(\phi_{bi} - V_A)}}$$

    Note that for forward bias ($V_A > 0$), $(\phi_{bi} - V_A)$ decreases, so $W$, $x_n$, $x_p$, and $|\mathcal{E}_{max}|$ all decrease. For reverse bias ($V_A < 0$), they all increase.
    """)
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(r"""
    ## Quasi-Fermi Levels
    """)
    _img = mo.hstack([mo.image(src=f"{IMAGE_BASE}/reverse_bias_quasiFermi.png", width="50%")], justify="center")
    _text = mo.md(r"""
    Under an applied voltage $V_A$, the quasi-Fermi levels split: $\boxed{E_{Fn} - E_{Fp} = qV_A}$
    """)
    mo.vstack([_header, _img, _text])
    return


@app.cell
def _(mo):
    Na_slider = mo.ui.slider(
        14, 18, value=17, step=0.5,
        label=r"log$_{10}$($N_A$) [cm⁻³]", show_value=True,
    )
    Nd_slider = mo.ui.slider(
        14, 18, value=16, step=0.5,
        label=r"log$_{10}$($N_D$) [cm⁻³]", show_value=True,
    )
    Va_slider = mo.ui.slider(
        -4, 0.5, value=0, step=0.1,
        label=r"$V_A$ [V]", show_value=True,
    )
    return Na_slider, Nd_slider, Va_slider


@app.cell
def _(
    Eg_Si,
    Na_slider,
    Nd_slider,
    Va_slider,
    eps_s,
    kT,
    mo,
    ni_Si,
    np,
    plt,
    q,
):
    # Read slider values
    _Na = 10 ** Na_slider.value
    _Nd = 10 ** Nd_slider.value
    _Va = Va_slider.value

    # Built-in potential
    _phi_bi = kT * np.log(_Na * _Nd / ni_Si**2)

    # Clamp Va to ensure (phi_bi - Va) > 0
    _V_eff = _phi_bi - _Va
    if _V_eff <= 0:
        _V_eff = 0.001

    # Equilibrium depletion widths (for reference dashed lines)
    _xn_eq = np.sqrt(2 * eps_s / q * _Na / (_Nd * (_Na + _Nd)) * _phi_bi)
    _xp_eq = np.sqrt(2 * eps_s / q * _Nd / (_Na * (_Na + _Nd)) * _phi_bi)

    # Biased depletion widths
    _xn = np.sqrt(2 * eps_s / q * _Na / (_Nd * (_Na + _Nd)) * _V_eff)
    _xp = np.sqrt(2 * eps_s / q * _Nd / (_Na * (_Na + _Nd)) * _V_eff)
    _W = _xn + _xp

    # Maximum electric field magnitude (V/cm)
    _E_max = q * _Na * _xp / eps_s

    # Spatial grid (cm)
    _margin = 2.5
    _x_extent = max(_xn_eq, _xp_eq, _xn, _xp) * _margin
    _x = np.linspace(-_x_extent, _x_extent, 1000)

    # --- Equilibrium curves ---
    _rho_eq = np.piecewise(
        _x,
        [_x < -_xp_eq, (_x >= -_xp_eq) & (_x <= 0), (_x > 0) & (_x <= _xn_eq), _x > _xn_eq],
        [0, -q * _Na, q * _Nd, 0],
    )
    _E_eq = np.zeros_like(_x)
    _mask_p_eq = (_x >= -_xp_eq) & (_x <= 0)
    _mask_n_eq = (_x > 0) & (_x <= _xn_eq)
    _E_eq[_mask_p_eq] = -q * _Na / eps_s * (_x[_mask_p_eq] + _xp_eq)
    _E_eq[_mask_n_eq] = q * _Nd / eps_s * (_x[_mask_n_eq] - _xn_eq)
    _V_eq = np.zeros_like(_x)
    _V_eq[_x <= -_xp_eq] = 0.0
    _V_eq[_mask_p_eq] = q * _Na / (2 * eps_s) * (_x[_mask_p_eq] + _xp_eq) ** 2
    _V_eq[_mask_n_eq] = _phi_bi - q * _Nd / (2 * eps_s) * (_x[_mask_n_eq] - _xn_eq) ** 2
    _V_eq[_x > _xn_eq] = _phi_bi

    # --- Biased curves ---
    _rho = np.piecewise(
        _x,
        [_x < -_xp, (_x >= -_xp) & (_x <= 0), (_x > 0) & (_x <= _xn), _x > _xn],
        [0, -q * _Na, q * _Nd, 0],
    )
    _E_field = np.zeros_like(_x)
    _mask_p = (_x >= -_xp) & (_x <= 0)
    _mask_n = (_x > 0) & (_x <= _xn)
    _E_field[_mask_p] = -q * _Na / eps_s * (_x[_mask_p] + _xp)
    _E_field[_mask_n] = q * _Nd / eps_s * (_x[_mask_n] - _xn)
    _V = np.zeros_like(_x)
    _V[_x <= -_xp] = 0.0
    _V[_mask_p] = q * _Na / (2 * eps_s) * (_x[_mask_p] + _xp) ** 2
    _V[_mask_n] = _V_eff - q * _Nd / (2 * eps_s) * (_x[_mask_n] - _xn) ** 2
    _V[_x > _xn] = _V_eff

    # Convert x-axis to micrometers
    _x_um = _x * 1e4
    _xp_um = _xp * 1e4
    _xn_um = _xn * 1e4

    # --- Four-panel plot ---
    _fig, _axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # (a) Charge density
    _axes[0].plot(_x_um, _rho_eq / q * 1e-15, color="gray", linestyle="--",
                  linewidth=1.5, alpha=0.5, label="Equilibrium")
    _axes[0].plot(_x_um, _rho / q * 1e-15, "b-", linewidth=2,
                  label=f"$V_A$ = {_Va:.1f} V")
    _axes[0].set_ylabel(r"$\rho\,/\,q$ ($\times 10^{15}$ cm$^{-3}$)", fontsize=16)
    _axes[0].set_title(
        r"$N_A$ = " + f"{_Na:.1e}" + r" cm$^{-3}$, $N_D$ = " + f"{_Nd:.1e}"
        + r" cm$^{-3}$, $V_A$ = " + f"{_Va:.1f} V",
        fontsize=16,
    )
    _axes[0].axhline(0, color="gray", linewidth=0.5)
    _axes[0].axvline(-_xp_um, color="red", linestyle="--", alpha=0.6,
                     label=f"$-x_p$ = {-_xp_um:.2f} \u00b5m")
    _axes[0].axvline(_xn_um, color="blue", linestyle="--", alpha=0.6,
                     label=f"$x_n$ = {_xn_um:.2f} \u00b5m")
    _axes[0].fill_between(_x_um, _rho / q * 1e-15, 0, alpha=0.15, color="blue")
    _axes[0].legend(fontsize=16)
    _axes[0].axvline(0, color="black", linestyle=":", alpha=0.7)
    _axes[0].grid(True, alpha=0.3)

    # (b) Electric field
    _axes[1].plot(_x_um, _E_eq, color="gray", linestyle="--",
                  linewidth=1.5, alpha=0.5, label="Equilibrium")
    _axes[1].plot(_x_um, _E_field, "r-", linewidth=2,
                  label=f"$V_A$ = {_Va:.1f} V")
    _axes[1].set_ylabel(r"$\mathcal{E}$ (V/cm)", fontsize=16)
    _axes[1].axhline(0, color="gray", linewidth=0.5)
    _axes[1].axvline(-_xp_um, color="red", linestyle="--", alpha=0.6)
    _axes[1].axvline(_xn_um, color="blue", linestyle="--", alpha=0.6)
    _axes[1].fill_between(_x_um, _E_field, 0, alpha=0.15, color="red")
    _axes[1].annotate(
        f"$|\\mathcal{{E}}_{{\\max}}|$ = {_E_max:.2e} V/cm",
        xy=(0, np.min(_E_field)),
        xytext=(0.3 * _xn_um + 0.5 * _xp_um, np.min(_E_field) * 0.5),
        fontsize=16, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )
    _axes[1].axvline(0, color="black", linestyle=":", alpha=0.7)
    _axes[1].grid(True, alpha=0.3)

    # (c) Electrostatic potential
    _axes[2].plot(_x_um, _V_eq, color="gray", linestyle="--",
                  linewidth=1.5, alpha=0.5, label="Equilibrium")
    _axes[2].plot(_x_um, _V, "g-", linewidth=2,
                  label=f"$V_A$ = {_Va:.1f} V")
    _axes[2].set_ylabel(r"$V$ (V)", fontsize=16)
    _axes[2].axhline(0, color="gray", linewidth=0.5)
    _axes[2].axhline(_V_eff, color="green", linestyle="--", alpha=0.6,
                     label=rf"$\phi_{{bi}} - V_A$ = {_V_eff:.2f} V")
    _axes[2].axvline(-_xp_um, color="red", linestyle="--", alpha=0.6)
    _axes[2].axvline(_xn_um, color="blue", linestyle="--", alpha=0.6)
    _axes[2].fill_between(_x_um, _V, 0, alpha=0.10, color="green")
    _axes[2].axvline(0, color="black", linestyle=":", alpha=0.7)
    _axes[2].legend(fontsize=16)
    _axes[2].grid(True, alpha=0.3)

    # (d) Energy band diagram from exact V(x)
    _EF_bd = 0.0  # E_Fp = 0 reference
    _Ec_offset = Eg_Si / 2 + kT * np.log(_Na / ni_Si)  # (E_c - E_Fp) on P-side
    _Ec = _Ec_offset - _V
    _Ev = _Ec - Eg_Si
    _Ei = (_Ec + _Ev) / 2
    _EFp = 0.0
    _EFn = _Va  # quasi-Fermi level split: E_Fn - E_Fp = qV_A

    _axes[3].plot(_x_um, _Ec, "b-", linewidth=2.5, label=r"$E_c$")
    _axes[3].plot(_x_um, _Ev, "b-", linewidth=2.5, label=r"$E_v$")
    _axes[3].plot(_x_um, _Ei, "g:", linewidth=1.5, label=r"$E_{Fi}$")
    _axes[3].axvspan(-_xp_um, _xn_um, alpha=0.08, color="gray")
    _axes[3].axvline(0, color="black", linestyle=":", alpha=0.7)
    _axes[3].axvline(-_xp_um, color="red", linestyle="--", alpha=0.6)
    _axes[3].axvline(_xn_um, color="blue", linestyle="--", alpha=0.6)

    if abs(_Va) < 0.001:
        _axes[3].axhline(_EFp, color="red", linestyle="--", linewidth=2, label=r"$E_F$")
    else:
        # Quasi-Fermi levels: E_Fp on P-side, E_Fn on N-side
        _EFp_line = np.where(_x <= _xn, _EFp, np.nan)
        _EFn_line = np.where(_x >= -_xp, _EFn, np.nan)
        _axes[3].plot(_x_um, _EFp_line, "m--", linewidth=2, label=r"$E_{Fp}$")
        _axes[3].plot(_x_um, _EFn_line, "r--", linewidth=2, label=r"$E_{Fn}$")
        if abs(_Va) > 0.05:
            _x_arrow = _x_um[-1] * 0.4
            _axes[3].annotate(
                "", xy=(_x_arrow, min(_EFp, _EFn)),
                xytext=(_x_arrow, max(_EFp, _EFn)),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=2),
            )
            _axes[3].text(
                _x_arrow * 1.15, (_EFp + _EFn) / 2,
                f"$qV_A$ = {_Va:.2f} eV",
                fontsize=16, color="purple", va="center",
            )

    # Barrier annotation
    _Ec_P_far = _Ec[0]
    _Ec_N_far = _Ec[-1]
    _x_arrow_barrier = _x_um[-1] * 0.85
    _axes[3].annotate(
        "", xy=(_x_arrow_barrier, _Ec_N_far),
        xytext=(_x_arrow_barrier, _Ec_P_far),
        arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=2),
    )
    _axes[3].text(
        _x_arrow_barrier * 1.05, (_Ec_P_far + _Ec_N_far) / 2,
        f"$q(\\phi_{{bi}} - V_A)$\n= {_V_eff:.2f} eV",
        fontsize=16, color="darkgreen", va="center",
    )

    _axes[3].set_ylabel("Energy (eV)", fontsize=16)
    _axes[3].set_xlabel("$x$ (µm)", fontsize=16)
    _axes[3].legend(fontsize=16, loc="lower left")
    _axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    # --- Computed values summary ---
    _info = mo.md(
        f"""
        **Computed Values:**

        | Quantity | Symbol | Value |
        |:---------|:------:|------:|
        | Built-in potential | $\\phi_{{bi}}$ | {_phi_bi:.2f} V |
        | Applied voltage | $V_A$ | {_Va:.1f} V |
        | Barrier | $\\phi_{{bi}} - V_A$ | {_V_eff:.2f} V |
        | N-side depletion width | $x_n$ | {_xn * 1e4:.2f} \u00b5m |
        | P-side depletion width | $x_p$ | {_xp * 1e4:.2f} \u00b5m |
        | Total depletion width | $W_{{dep}}$ | {_W * 1e4:.2f} \u00b5m |
        | Maximum electric field | $\\lvert\\mathcal{{E}}_{{\\max}}\\rvert$ | {_E_max:.2e} V/cm |
        """
    )

    _title = mo.md(r"""## Interactive Calculator of $\rho$, $\mathcal{E}$, $V$, and Energy Bands under Bias""")
    _controls = mo.hstack([Na_slider, Nd_slider, Va_slider], justify="center")

    mo.vstack([_title, _controls, plt.gca(), _info], align="center")
    return


@app.cell
def _(IMAGE_BASE, mo):
    _text = mo.md(r"""
    ## Depletion Capacitance

    Where there is a separation of positive and negative charges, there is a **capacitance**. Due to the separation of charges in the depletion region, there is a capacitance. This is called **depletion capacitance** or **junction capacitance**.
    """)
    _img = mo.hstack([mo.image(src=f"{IMAGE_BASE}/depletion_cap.png", width="70%")], justify="center")
    mo.vstack([_text, _img])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Depletion Capacitance Per Unit Area

    $$\boxed{C_{dep} = \frac{dQ}{dV_A}}$$

    $$Q = qN_D x_n = qN_A x_p \quad \text{charge per area}$$

    $$C_{dep} = qAN_d \frac{dx_n}{dV_A} = qAN_A \frac{dx_p}{dV_A}$$

    Substituting $x_n$ and $x_p$ from the depletion width equations:

    $$\boxed{C_{dep} = A \frac{\varepsilon_s}{W_{dep}}}$$

    As if the the junction is a **parallel plate capacitor with a thickness of $W_{dep}$**

    Re-arranging the equation:

    $$\boxed{\frac{1}{C_{dep}^2} = \frac{W_{dep}^2}{A^2 \varepsilon_s^2}=\frac{2(\phi_{bi} - V_A)}{qA^2 \varepsilon_s}\cdot\frac{N_A + N_D}{N_A N_D}}$$

    Application: A diode is voltage-controlled capacitor (varactor).
    """)
    return


@app.cell
def _(Na_slider, Nd_slider, eps_s, kT, mo, ni_Si, np, plt, q):
    # --- Read slider values ---
    _Na = 10 ** Na_slider.value
    _Nd = 10 ** Nd_slider.value

    # --- Built-in potential ---
    _phi_bi = kT * np.log(_Na * _Nd / ni_Si**2)

    # --- Voltage range for C-V plot ---
    _Va_arr = np.linspace(-5, _phi_bi - 0.05, 500)

    # --- Junction capacitance per unit area ---
    _V_eff_arr = _phi_bi - _Va_arr
    _CJ = np.sqrt(q * eps_s / 2 * _Na * _Nd / (_Na + _Nd) / _V_eff_arr)

    # --- 1/CJ^2 ---
    _inv_CJ2 = 1.0 / _CJ**2

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: C_J vs V_A
    ax1.plot(_Va_arr, _CJ * 1e9, "b-", linewidth=2)  # convert F/cm^2 to nF/cm^2
    ax1.set_xlabel(r"$V_A$ (V)", fontsize=16)
    ax1.set_ylabel(r"$C_{{dep}}$ (nF/cm$^2$)", fontsize=16)
    ax1.set_title(r"Depletion Capacitance vs. $V_A$", fontsize=16, fontweight="bold")
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, _phi_bi)

    # Mark equilibrium capacitance
    _CJ_eq = np.sqrt(q * eps_s / 2 * _Na * _Nd / (_Na + _Nd) / _phi_bi)
    ax1.plot(0, _CJ_eq * 1e9, "ro", markersize=8, label=f"$C_{{{{dep0}}}}$ = {_CJ_eq*1e9:.2f} nF/cm$^2$")
    ax1.legend(fontsize=16)

    # Right: 1/C_J^2 vs V_A
    ax2.plot(_Va_arr, _inv_CJ2 * 1e-14, "r-", linewidth=2)  # scale for readability
    ax2.set_xlabel(r"$V_A$ (V)", fontsize=16)
    ax2.set_ylabel(r"$1/C_{{dep}}^2$ ($\times 10^{14}$ cm$^4$/F$^2$)", fontsize=16)
    ax2.set_title(r"$1/C_{{dep}}^2$ vs. $V_A$", fontsize=16, fontweight="bold")
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, _phi_bi + 0.5)

    # Show x-intercept at phi_bi
    ax2.axvline(_phi_bi, color="green", linestyle="--", alpha=0.7,
                label=f"$\\phi_{{{{bi}}}}$ = {_phi_bi:.3f} V")
    ax2.plot(_phi_bi, 0, "go", markersize=8)

    # Draw extension line to x-axis
    _slope_inv = 2 / (q * eps_s) * (_Na + _Nd) / (_Na * _Nd)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.legend(fontsize=16)

    plt.tight_layout()

    _info = mo.md(
        f"""
        **C-V parameters:**
        $N_A$ = {_Na:.1e} cm$^{{{{-3}}}}$ |
        $N_D$ = {_Nd:.1e} cm$^{{{{-3}}}}$ |
        $\\phi_{{{{bi}}}}$ = {_phi_bi:.2f} V |
        $C_{{dep}}$ = {_CJ_eq*1e9:.1f} nF/cm$^2$ |
        Slope of $1/C_{{dep}}^2$: {_slope_inv*1e-14:.2f} $\\times 10^{{{{14}}}}$ cm$^4$/(F$^2\\cdot$V)
        """
    )

    _title = mo.md(r"""
    ## Interactive: Depletion Capacitance
    """)
    _controls = mo.hstack([Na_slider, Nd_slider])
    mo.vstack([_title, _controls, plt.gca(), _info])
    return


@app.cell
def _(IMAGE_BASE, mo):
    _text = mo.md(r"""
    ## Junction Breakdown

    At sufficiently large reverse bias, the electric field in the depletion region can become very large, leading to a dramatic increase in reverse current. This is called **breakdown**.
    """)
    _img = mo.hstack([mo.image(src=f"{IMAGE_BASE}/Breakdown.png", width="50%")], justify="center")
    mo.vstack([_text, _img])
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(r"""
    ### Zener Breakdown
    """)
    _left = mo.md(r"""
    - Occurs in **heavily doped** junctions with narrow depletion regions
    - The electric field is so strong that electrons can **tunnel** directly from the valence band on the P-side to the conduction band on the N-side
    - This is quantum mechanical tunneling through the narrow barrier
    - Typically occurs at $|V_{BR}| < 5$ V for very heavily doped junctions
    """)
    _right = mo.image(src=f"{IMAGE_BASE}/zener.png", width="100%")
    _cols = mo.hstack([_left, _right], widths=[2, 1], align="center")
    mo.vstack([_header, _cols])
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(r"""
    ### Avalanche Breakdown
    """)
    _left = mo.md(r"""
    - Occurs in **lightly doped** junctions with wide depletion regions
    - High-energy carriers (accelerated by the strong field) collide with lattice atoms and generate new electron-hole pairs via **impact ionization**
    - These new carriers are also accelerated and create more pairs, leading to a **cascade** (avalanche)
    - Breakdown voltage $V_{BR}$ increases with decreasing doping (wider depletion region means the field is spread over a larger distance)
    """)
    _right = mo.image(src=f"{IMAGE_BASE}/avalanche.png", width="100%")
    _cols = mo.hstack([_left, _right], widths=[2, 1], align="center")
    mo.vstack([_header, _cols])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### High Voltage Devices

    - Typical integrated circuit devices have $V_B \sim 10-20$ V
    - Power electronics (for high-voltage DC converters, e.g., for batteries, electric vehicles) need to operate at 10-1000 V

    Reverse breakdown voltage is given by:

    $$V_B = \frac{\varepsilon_s \mathcal{E}_{crit}^2}{2qN} - \phi_{bi} \qquad \frac{1}{N} = \frac{1}{N_A} + \frac{1}{N_D}$$

    where $\mathcal{E}_{crit}$ is the critical field ($\sim 10^6$ V/cm for Si).

    **To increase $V_B$, use:**

    1. **Lighter doping** (lower $N$)
    2. **A semiconductor with a higher dielectric constant** (larger $\varepsilon_s$)
    3. **A larger bandgap material**
        - $\mathcal{E}_{crit}$ is increased $\rightarrow$ more energy required for impact ionization
        - $N_A$ and $N_D$ are in fact $p_0$ and $n_0$; a larger bandgap material has lower carrier concentrations
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### GaN and SiC for Power Electronics

    SiC and GaN can be grown on Si substrates, making them attractive for power electronics.

    | | **Silicon** | **Gallium nitride (GaN)** | **Silicon carbide (SiC)** |
    |:---|:---:|:---:|:---:|
    | **Bandgap (eV)** | 1.1 | 3.45 | 2.4 - 3.3 (depending on crystal structure) |
    | **Dielectric constant** | 11.7 | 10 | 9.7 - 10 |
    | **Electron mobility (cm$^2$/Vs)** | 1400 | ~2000 | ~900 |

    GaN and SiC have **much larger bandgaps** than Si, leading to:
    - Higher critical electric fields (higher breakdown voltages)
    - Lower intrinsic carrier concentrations (better high-temperature operation)
    - Higher electron mobility (GaN) enables faster switching
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary of Equations

    $$\boxed{\phi_{bi} = \frac{kT}{q}\ln\left(\frac{N_A N_D}{n_i^2}\right)}$$

    $$\boxed{W = \sqrt{\frac{2\varepsilon_s}{q}\cdot\frac{N_A + N_D}{N_A N_D}\cdot(\phi_{bi} - V_A)}}$$

    $$\boxed{x_n = \sqrt{\frac{2\varepsilon_s}{q}\cdot\frac{N_A}{N_D(N_A + N_D)}\cdot(\phi_{bi} - V_A)} \qquad x_p = \sqrt{\frac{2\varepsilon_s}{q}\cdot\frac{N_D}{N_A(N_A + N_D)}\cdot(\phi_{bi} - V_A)}}$$

    $$\boxed{|\mathcal{E}_{max}| = \frac{qN_A x_p}{\varepsilon_s} = \frac{qN_D x_n}{\varepsilon_s}}$$

    $$\boxed{C_{dep} = \frac{\varepsilon_s}{W} = \sqrt{\frac{q\varepsilon_s}{2}\cdot\frac{N_A N_D}{(N_A + N_D)(\phi_{bi} - V_A)}}}$$

    $$\boxed{\frac{1}{C_{dep}^2} = \frac{2(\phi_{bi} - V_A)}{q\varepsilon_s}\cdot\frac{N_A + N_D}{N_A N_D}} \quad \text{(linear in } V_A \text{, x-intercept at } \phi_{bi}\text{)}$$

    | Property | Reverse Bias ($V_A < 0$) |
    |:---|:---|
    | Barrier | $q(\phi_{bi} + \vert V_A \vert)$ (increased) |
    | Depletion width | Wider |
    | $\vert \mathcal{E}_{max} \vert$ | Larger |
    | Capacitance $C_{dep}$ | Smaller |
    | Quasi-Fermi splitting | $E_{Fn} - E_{Fp} = qV_A$ |
    """)
    return


if __name__ == "__main__":
    app.run()
