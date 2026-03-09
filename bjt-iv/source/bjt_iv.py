# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "matplotlib==3.10.8",
# ]
# ///

import marimo

__generated_with = "0.20.4"
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
    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/bjt-iv/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    # Physical constants
    q = 1.6e-19       # C
    kT_eV = 0.02585   # eV at 300K
    kT_q = 0.02585    # V at 300K (thermal voltage)
    ni_Si = 1.1e10    # cm^-3 for Si at 300K
    Eg_Si = 1.12      # eV

    # Typical mobility values for Si at 300K
    mu_n = 1350.0  # cm^2/V-s
    mu_p = 480.0   # cm^2/V-s
    D_n = kT_q * mu_n  # cm^2/s
    D_p = kT_q * mu_p  # cm^2/s

    mo.md(
        r"""
        # BJT Current-Voltage Relations

        ECE350, Lectures 25-26

        Hu, Chapter 8

        This notebook derives the current-voltage characteristics of the BJT:

        1. Interactive: NPN Energy Band Diagram
        2. Derivation of IV (Forward Active): Overview
        3. Coordinate System
        4. Interactive: Minority Carrier Profiles 
        5. Base Minority Carrier Profile
        6. Collector, Emitter, and Base Currents
        7. Interactive: $\beta_F$ vs. Doping + Quick Check
        8. Base Width Modulation: Early Effect
        9. Gummel Number Formulation
        10. Interactive: Gummel Plot
        11. Ebers-Moll Model *(optional)*
        """
    )
    return Eg_Si, IMAGE_BASE, kT_q, mo, ni_Si, np, plt


@app.cell(hide_code=True)
def _(mo):
    VBE_band_slider = mo.ui.slider(
        start=0.0, stop=0.7, value=0.5, step=0.01,
        label=r"$V_{BE}$ (V)"
    )
    VBC_band_slider = mo.ui.slider(
        start=-5.0, stop=0.5, value=-2.0, step=0.1,
        label=r"$V_{BC}$ (V)"
    )
    NE_band_slider = mo.ui.slider(
        start=17, stop=20, value=18, step=0.5,
        label=r"log$_{10}$($N_E$) [cm$^{-3}$]"
    )
    NB_band_slider = mo.ui.slider(
        start=14, stop=18, value=16, step=0.5,
        label=r"log$_{10}$($N_B$) [cm$^{-3}$]"
    )
    NC_band_slider = mo.ui.slider(
        start=14, stop=18, value=16, step=0.5,
        label=r"log$_{10}$($N_C$) [cm$^{-3}$]"
    )
    WB_band_slider = mo.ui.slider(
        start=0.1, stop=3.0, value=0.5, step=0.1,
        label=r"$W_B$ ($\mu$m)"
    )
    band_controls = mo.vstack([
        mo.md("## 1. Interactive: NPN Energy Band Diagram"),
        mo.md("Visualize how bias and device parameters change the energy barriers and quasi-Fermi levels."),
        mo.hstack([VBE_band_slider, VBC_band_slider], justify="start"),
        mo.hstack([NE_band_slider, NB_band_slider, NC_band_slider, WB_band_slider], justify="start"),
    ])
    return (
        NB_band_slider,
        NC_band_slider,
        NE_band_slider,
        VBC_band_slider,
        VBE_band_slider,
        WB_band_slider,
        band_controls,
    )


@app.cell(hide_code=True)
def _(
    Eg_Si,
    NB_band_slider,
    NC_band_slider,
    NE_band_slider,
    VBC_band_slider,
    VBE_band_slider,
    WB_band_slider,
    band_controls,
    kT_q,
    mo,
    ni_Si,
    np,
    plt,
):
    _VBE = VBE_band_slider.value
    _VBC = VBC_band_slider.value

    _NE = 10 ** NE_band_slider.value
    _NB = 10 ** NB_band_slider.value
    _NC = 10 ** NC_band_slider.value
    _WB = WB_band_slider.value * 1e-4  # cm
    _eps_Si = 11.7 * 8.854e-14  # F/cm
    _q = 1.6e-19

    _Vbi_BE = kT_q * np.log(_NE * _NB / ni_Si**2)
    _Vbi_BC = kT_q * np.log(_NB * _NC / ni_Si**2)

    _V_BE_eff = max(_Vbi_BE - _VBE, 0.02)
    _V_BC_eff = max(_Vbi_BC - _VBC, 0.02)

    _W_BE = np.sqrt(2 * _eps_Si * _V_BE_eff / _q * (_NE + _NB) / (_NE * _NB))
    _xnBE = _W_BE * _NB / (_NE + _NB)
    _xpBE = _W_BE * _NE / (_NE + _NB)

    _W_BC = np.sqrt(2 * _eps_Si * _V_BC_eff / _q * (_NB + _NC) / (_NB * _NC))
    _xpBC = _W_BC * _NC / (_NB + _NC)
    _xnBC = _W_BC * _NB / (_NB + _NC)

    _WB_um = _WB * 1e4
    _xnBE_um = _xnBE * 1e4
    _xpBE_um = _xpBE * 1e4
    _xpBC_um = _xpBC * 1e4
    _xnBC_um = _xnBC * 1e4

    _Ec_E = 0.0
    _Ec_B = _V_BE_eff
    _Ec_C = _Ec_B - _V_BC_eff

    _Ev_E = _Ec_E - Eg_Si
    _Ev_B = _Ec_B - Eg_Si
    _Ev_C = _Ec_C - Eg_Si

    _Ei_E = _Ec_E - Eg_Si / 2
    _Ei_B = _Ec_B - Eg_Si / 2
    _Ei_C = _Ec_C - Eg_Si / 2

    _EFn_E = _Ei_E + kT_q * np.log(_NE / ni_Si)
    _EFp_B = _Ei_B - kT_q * np.log(_NB / ni_Si)
    _EFn_C = _Ei_C + kT_q * np.log(_NC / ni_Si)

    _dV_E = _V_BE_eff * _NB / (_NE + _NB)
    _dV_B_BE = _V_BE_eff * _NE / (_NE + _NB)
    _dV_B_BC = _V_BC_eff * _NC / (_NB + _NC)
    _dV_C = _V_BC_eff * _NB / (_NB + _NC)

    _emitter_end = -_xpBE_um - _xnBE_um - 2.0
    _collector_end = _WB_um + _xpBC_um + _xnBC_um + 2.0

    _x1 = np.linspace(_emitter_end, -_xpBE_um - _xnBE_um, 80)
    _Ec1 = np.full_like(_x1, _Ec_E)

    _x2 = np.linspace(-_xpBE_um - _xnBE_um, -_xpBE_um, 80)
    _t2 = (_x2 - (-_xpBE_um - _xnBE_um)) / max(_xnBE_um, 1e-6)
    _Ec2 = _Ec_E + _dV_E * _t2**2

    _x3 = np.linspace(-_xpBE_um, 0, 80)
    _t3 = -_x3 / max(_xpBE_um, 1e-6)
    _Ec3 = _Ec_B - _dV_B_BE * _t3**2

    _x4 = np.linspace(0, _WB_um, 80)
    _Ec4 = np.full_like(_x4, _Ec_B)

    _x5 = np.linspace(_WB_um, _WB_um + _xpBC_um, 80)
    _t5 = (_x5 - _WB_um) / max(_xpBC_um, 1e-6)
    _Ec5 = _Ec_B - _dV_B_BC * _t5**2

    _x6 = np.linspace(_WB_um + _xpBC_um, _WB_um + _xpBC_um + _xnBC_um, 80)
    _t6 = (_WB_um + _xpBC_um + _xnBC_um - _x6) / max(_xnBC_um, 1e-6)
    _Ec6 = _Ec_C + _dV_C * _t6**2

    _x7 = np.linspace(_WB_um + _xpBC_um + _xnBC_um, _collector_end, 80)
    _Ec7 = np.full_like(_x7, _Ec_C)

    _x_all = np.concatenate([_x1, _x2, _x3, _x4, _x5, _x6, _x7])
    _Ec_all = np.concatenate([_Ec1, _Ec2, _Ec3, _Ec4, _Ec5, _Ec6, _Ec7])
    _Ev_all = _Ec_all - Eg_Si
    _Ei_all = _Ec_all - Eg_Si / 2

    _fig_band, _ax_band = plt.subplots(figsize=(10, 7))

    _ax_band.plot(_x_all, _Ec_all, 'b-', linewidth=2.5, label='$E_c$')
    _ax_band.plot(_x_all, _Ev_all, 'r-', linewidth=2.5, label='$E_v$')
    _ax_band.plot(_x_all, _Ei_all, 'k--', linewidth=1, alpha=0.4, label='$E_i$')

    _x_EFn = np.concatenate([_x1, _x2, _x3])
    _EFn_left = np.full_like(_x_EFn, _EFn_E)
    _x_EFp = np.concatenate([_x3, _x4, _x5, _x6])
    _EFp_mid = np.full_like(_x_EFp, _EFp_B)
    _x_EFn_r = np.concatenate([_x5, _x6, _x7])
    _EFn_right = np.full_like(_x_EFn_r, _EFn_C)

    _ax_band.plot(_x_EFn, _EFn_left, 'b--', linewidth=1.5, alpha=0.7, label='$E_{Fn}$')
    _ax_band.plot(_x_EFp, _EFp_mid, 'r--', linewidth=1.5, alpha=0.7, label='$E_{Fp}$')
    _ax_band.plot(_x_EFn_r, _EFn_right, 'b--', linewidth=1.5, alpha=0.7)

    _y_top = max(_Ec_all) + 0.3
    _y_bot = min(_Ev_all) - 0.3
    _ax_band.axvspan(_emitter_end, -_xpBE_um - _xnBE_um, alpha=0.04, color='blue')
    _ax_band.axvspan(0, _WB_um, alpha=0.06, color='red')
    _ax_band.axvspan(_WB_um + _xpBC_um + _xnBC_um, _collector_end, alpha=0.04, color='green')

    _ax_band.text((_emitter_end + (-_xpBE_um - _xnBE_um)) / 2, _y_top - 0.1, 'Emitter (N$^+$)',
                  fontsize=14, ha='center', color='blue')
    _ax_band.text(_WB_um / 2, _y_top - 0.1, 'Base (P)',
                  fontsize=14, ha='center', color='red')
    _ax_band.text((_WB_um + _xpBC_um + _xnBC_um + _collector_end) / 2, _y_top - 0.1, 'Collector (N)',
                  fontsize=14, ha='center', color='green')

    _ax_band.axvline(-_xpBE_um - _xnBE_um, color='gray', linestyle=':', alpha=0.4)
    _ax_band.axvline(0, color='gray', linestyle=':', alpha=0.4)
    _ax_band.axvline(_WB_um, color='gray', linestyle=':', alpha=0.4)
    _ax_band.axvline(_WB_um + _xpBC_um + _xnBC_um, color='gray', linestyle=':', alpha=0.4)

    _barrier_BE = _Ec_B - _Ec_E
    _arrow_x = -_xpBE_um - _xnBE_um * 0.5
    _ax_band.annotate('', xy=(_arrow_x, _Ec_E), xytext=(_arrow_x, _Ec_B),
                      arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    _ax_band.text(_arrow_x - 0.3, (_Ec_E + _Ec_B) / 2,
                  f'{_barrier_BE:.2f} eV', fontsize=12, color='purple', ha='right', va='center')

    _ax_band.set_xlabel(r'Position ($\mu$m)', fontsize=16)
    _ax_band.set_ylabel(r'Energy (eV)', fontsize=16)
    _ax_band.set_title(f'NPN Energy Band Diagram ($V_{{BE}}$ = {_VBE:.2f} V, $V_{{BC}}$ = {_VBC:.1f} V)',
                       fontsize=16, fontweight='bold')
    _ax_band.legend(fontsize=13, loc='lower left', ncol=2)
    _ax_band.tick_params(labelsize=14)
    _ax_band.grid(True, alpha=0.2)
    _ax_band.set_xlim(_emitter_end, _collector_end)
    _ax_band.set_ylim(_y_bot, _y_top + 0.2)
    plt.tight_layout()

    _mode = "Forward Active" if _VBE > 0.1 and _VBC < 0 else \
            "Saturation" if _VBE > 0.1 and _VBC > 0 else \
            "Cut-off" if _VBE < 0.1 and _VBC < 0 else "Inverted"

    _info_band = mo.md(
        f"""
        **Mode: {_mode}** | $N_E$ = {_NE:.0e}, $N_B$ = {_NB:.0e}, $N_C$ = {_NC:.0e} cm$^{{-3}}$ | $W_B$ = {_WB*1e4:.1f} $\\mu$m

        $V_{{bi,BE}}$ = {_Vbi_BE:.3f} V, $V_{{bi,BC}}$ = {_Vbi_BC:.3f} V |
        B-E barrier: **{_barrier_BE:.3f} eV** | B-C barrier: **{_Ec_B - _Ec_C:.3f} eV**

        Quasi-Fermi level splitting: $E_{{Fn,E}} - E_{{Fp,B}}$ = {_EFn_E - _EFp_B:.3f} eV $\\approx qV_{{BE}}$ = {_VBE:.2f} eV
        """
    )

    mo.vstack([band_controls, _fig_band, _info_band])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Derivation of IV (Forward Active): Overview


    **Assumptions:**
    - Emitter and collector are semi-infinite
    - Ideal diode approximations
    - Steady-state

    **Procedure:**
    1. Solve for $I_C$ as a function of $V_{BE}$ and $V_{BC}$
    2. Solve for $I_E$
    3. Then solve $I_B = I_E - I_C$, and determine the metrics ($\alpha_F$, $\beta_F$)

    Currents are solved like the PN diode:
    - Find minority carrier distributions in the E, B, C
    - Find electron and hole diffusion currents at edges of depletion regions
    """)
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _img = mo.image(src=f"{IMAGE_BASE}/bjt-structure.png", width="25%")
    _img2 = mo.image(src=f"{IMAGE_BASE}/bjt-minority-carriers.png", width="50%")
    mo.vstack([
        mo.md(r"""
        ## 3. Coordinate System and Minority Carriers

        Define coordinates:
        - **Base:** $x = 0$ at B-E depletion edge, $x = W_B$ at B-C depletion edge
        - **Emitter:** $x'' = 0$ at B-E depletion edge, increasing into emitter
        - **Collector:** $x' = 0$ at B-C depletion edge, increasing into collector

        - $W_B$ is the quasi-neutral base width (distance between depletion edges).
        """),
        mo.hstack([_img, _img2], justify="center"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    VBE_carrier_slider = mo.ui.slider(
        start=0.0, stop=0.7, value=0.5, step=0.01,
        label=r"$V_{BE}$ (V)"
    )
    carrier_controls = mo.vstack([
        mo.md("## 4. Interactive: Minority Carrier Profiles"),
        mo.md("See how forward bias injects minority carriers across the entire NPN structure."),
        VBE_carrier_slider,
    ])
    return VBE_carrier_slider, carrier_controls


@app.cell(hide_code=True)
def _(VBE_carrier_slider, carrier_controls, kT_q, mo, ni_Si, np, plt):
    _VBE = VBE_carrier_slider.value
    _VBC = -2.0

    _NE = 1e18
    _NB = 1e16
    _NC = 1e16
    _WB = 0.5e-4   # cm (0.5 um)

    _mu_nB = 1350.0 / (1 + (_NB / 1e17)**0.7)
    _mu_pE = 480.0 / (1 + (_NE / 1e17)**0.7)
    _mu_pC = 480.0 / (1 + (_NC / 1e17)**0.7)
    _DnB = kT_q * _mu_nB     # electrons in base
    _DpE = kT_q * _mu_pE     # holes in emitter
    _DpC = kT_q * _mu_pC     # holes in collector

    _tau_nB = 5e-7   # s
    _tau_pE = 1e-7   # s
    _tau_pC = 5e-7   # s
    _LnB = np.sqrt(_DnB * _tau_nB)   # cm
    _LpE = np.sqrt(_DpE * _tau_pE)   # cm
    _LpC = np.sqrt(_DpC * _tau_pC)   # cm

    _nB0 = ni_Si**2 / _NB
    _pE0 = ni_Si**2 / _NE
    _pC0 = ni_Si**2 / _NC

    _dn_BE = _nB0 * (np.exp(_VBE / kT_q) - 1)
    _dn_BC = _nB0 * (np.exp(_VBC / kT_q) - 1)
    _dp_E = _pE0 * (np.exp(_VBE / kT_q) - 1)
    _dp_C = _pC0 * (np.exp(_VBC / kT_q) - 1)

    _eps_Si = 11.7 * 8.854e-14
    _q = 1.6e-19
    _Vbi_BE = kT_q * np.log(_NE * _NB / ni_Si**2)
    _Vbi_BC = kT_q * np.log(_NB * _NC / ni_Si**2)
    _V_BE_eff = max(_Vbi_BE - _VBE, 0.02)
    _V_BC_eff = max(_Vbi_BC - _VBC, 0.02)

    _W_BE = np.sqrt(2 * _eps_Si * _V_BE_eff / _q * (_NE + _NB) / (_NE * _NB))
    _xnBE_um = _W_BE * _NB / (_NE + _NB) * 1e4
    _xpBE_um = _W_BE * _NE / (_NE + _NB) * 1e4

    _W_BC = np.sqrt(2 * _eps_Si * _V_BC_eff / _q * (_NB + _NC) / (_NB * _NC))
    _xpBC_um = _W_BC * _NC / (_NB + _NC) * 1e4
    _xnBC_um = _W_BC * _NB / (_NB + _NC) * 1e4

    _WB_um = _WB * 1e4
    _E_edge = -_xpBE_um - _xnBE_um
    _C_edge = _WB_um + _xpBC_um + _xnBC_um

    _x_E = np.linspace(_E_edge - 3, _E_edge, 200)
    _x_B = np.linspace(0, _WB_um, 200)
    _x_C = np.linspace(_C_edge, _C_edge + 3, 200)

    _pE_excess = _dp_E * np.exp(-(_E_edge - _x_E) / (_LpE * 1e4))
    _pE_total = _pE0 + _pE_excess

    _x_B_cm = _x_B * 1e-4
    _nB_excess = _dn_BE * (1 - _x_B_cm / _WB) + _dn_BC * (_x_B_cm / _WB)
    _nB_total = _nB0 + _nB_excess

    _pC_excess = _dp_C * np.exp(-(_x_C - _C_edge) / (_LpC * 1e4))
    _pC_total = _pC0 + _pC_excess

    _x_left = _E_edge - 3
    _x_right = _C_edge + 3

    _fig_c, (_ax_c, _ax_lin) = plt.subplots(1, 2, figsize=(16, 6))

    for _ax in [_ax_c, _ax_lin]:
        _ax.axvspan(_x_left, _E_edge, alpha=0.04, color='blue')
        _ax.axvspan(0, _WB_um, alpha=0.06, color='red')
        _ax.axvspan(_C_edge, _x_right, alpha=0.04, color='green')

        _ax.axvline(_E_edge, color='gray', linestyle=':', alpha=0.4)
        _ax.axvline(0, color='gray', linestyle=':', alpha=0.4)
        _ax.axvline(_WB_um, color='gray', linestyle=':', alpha=0.4)
        _ax.axvline(_C_edge, color='gray', linestyle=':', alpha=0.4)

        _ax.set_xlabel(r'Position ($\mu$m)', fontsize=16)
        _ax.tick_params(labelsize=14)
        _ax.set_xlim(_x_left, _x_right)

    # Left: log scale
    _ax_c.semilogy(_x_E, np.maximum(_pE_total, 1e0), 'r-', linewidth=2.5, label='$p_E(x)$ (holes in emitter)')
    _ax_c.semilogy(_x_B, np.maximum(_nB_total, 1e0), 'b-', linewidth=2.5, label='$n_B(x)$ (electrons in base)')
    _ax_c.semilogy(_x_C, np.maximum(_pC_total, 1e0), 'r--', linewidth=2.5, label='$p_C(x)$ (holes in collector)')

    _ax_c.axhline(_pE0, color='r', linestyle=':', alpha=0.4, linewidth=1)
    _ax_c.axhline(_nB0, color='b', linestyle=':', alpha=0.4, linewidth=1)
    _ax_c.axhline(_pC0, color='r', linestyle=':', alpha=0.4, linewidth=1)

    _ylim_lo = _ax_c.get_ylim()[0]
    _label_y = _ylim_lo * 30
    _ax_c.text((_x_left + _E_edge) / 2, _label_y, 'Emitter\n(N$^+$)', fontsize=14, ha='center', va='bottom', color='blue')
    _ax_c.text(_WB_um / 2, _label_y, 'Base\n(P)', fontsize=14, ha='center', va='bottom', color='red')
    _ax_c.text((_C_edge + _x_right) / 2, _label_y, 'Collector\n(N)', fontsize=14, ha='center', va='bottom', color='green')

    _ax_c.set_ylabel(r'Minority carrier concentration (cm$^{-3}$)', fontsize=16)
    _ax_c.set_title(f'Log scale ($V_{{BE}}$ = {_VBE:.2f} V, $V_{{BC}}$ = {_VBC:.0f} V)', fontsize=16, fontweight='bold')
    _ax_c.legend(fontsize=12, loc='upper right')
    _ax_c.grid(True, alpha=0.3, which='both')

    # Right: linear scale
    _ax_lin.plot(_x_E, _pE_total, 'r-', linewidth=2.5, label='$p_E(x)$')
    _ax_lin.plot(_x_B, _nB_total, 'b-', linewidth=2.5, label='$n_B(x)$')
    _ax_lin.plot(_x_C, _pC_total, 'r--', linewidth=2.5, label='$p_C(x)$')

    _ax_lin.axhline(_nB0, color='b', linestyle=':', alpha=0.4, linewidth=1)

    _ylim_top_lin = _ax_lin.get_ylim()[1] if _ax_lin.get_ylim()[1] > 0 else 1e10
    _label_y_lin = _ylim_top_lin * 0.02
    _ax_lin.text((_x_left + _E_edge) / 2, _label_y_lin, 'Emitter\n(N$^+$)', fontsize=14, ha='center', va='bottom', color='blue')
    _ax_lin.text(_WB_um / 2, _label_y_lin, 'Base\n(P)', fontsize=14, ha='center', va='bottom', color='red')
    _ax_lin.text((_C_edge + _x_right) / 2, _label_y_lin, 'Collector\n(N)', fontsize=14, ha='center', va='bottom', color='green')

    _ax_lin.set_ylabel(r'Minority carrier concentration (cm$^{-3}$)', fontsize=16)
    _ax_lin.set_title(f'Linear scale ($V_{{BE}}$ = {_VBE:.2f} V, $V_{{BC}}$ = {_VBC:.0f} V)', fontsize=16, fontweight='bold')
    _ax_lin.legend(fontsize=12, loc='upper right')
    _ax_lin.grid(True, alpha=0.3)
    _ax_lin.set_ylim(bottom=0)

    plt.tight_layout()

    _info_c = mo.md(
        f"""
        **Device parameters:**

        |  | Emitter (N$^+$) | Base (P) | Collector (N) |
        |:--|:--:|:--:|:--:|
        | Doping | $N_E$ = {_NE:.0e} cm$^{{-3}}$ | $N_B$ = {_NB:.0e} cm$^{{-3}}$ | $N_C$ = {_NC:.0e} cm$^{{-3}}$ |
        | Diffusion coeff. | $D_E$ = {_DpE:.1f} cm$^2$/s | $D_B$ = {_DnB:.1f} cm$^2$/s | $D_C$ = {_DpC:.1f} cm$^2$/s |
        | Lifetime | $\\tau_E$ = {_tau_pE:.0e} s | $\\tau_B$ = {_tau_nB:.0e} s | $\\tau_C$ = {_tau_pC:.0e} s |
        | Diffusion length | $L_E$ = {_LpE*1e4:.1f} $\\mu$m | $L_B$ = {_LnB*1e4:.1f} $\\mu$m | $L_C$ = {_LpC*1e4:.1f} $\\mu$m |

        $W_B$ = {_WB*1e4:.1f} $\\mu$m

        ---

        **At $V_{{BE}}$ = {_VBE:.2f} V:**
        Excess electrons at base edge: $n'_B(0)$ = {_dn_BE:.2e} cm$^{{-3}}$ |
        Equilibrium: $n_{{B0}}$ = {_nB0:.2e} cm$^{{-3}}$ 

        The base profile is approximately **linear** because $W_B$ ({_WB*1e4:.1f} $\\mu$m) $\\ll$ $L_B$ ({_LnB*1e4:.1f} $\\mu$m).
        The slope of $n_B(x)$ determines $I_C$ (steeper slope $\\to$ more diffusion current).
        """
    )

    mo.vstack([carrier_controls, _fig_c, _info_c])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Base Minority Carrier Profile

    The continuity equation for electrons in the P-type base:

    $$D_B\frac{d^2 n'_B}{dx^2} - \frac{n'_B}{\tau_B} = 0$$

    with boundary conditions:
    - At $x = 0$: $n'_B(0) = n_{B0}\left(e^{qV_{BE}/kT} - 1\right)$
    - At $x = W_B$: $n'_B(W_B) = n_{B0}\left(e^{qV_{BC}/kT} - 1\right)$

    where $n_{B0} = n_i^2 / N_B$, $n'_B = n_B - n_{B0}$ is the excess electron concentration, and $L_B = \sqrt{D_B \tau_B}$.

    **General solution (sinh form):**

    $$\boxed{n'_B(x) = n'_B(0)\,\frac{\sinh\!\left[\frac{W_B - x}{L_B}\right]}{\sinh\!\left(\frac{W_B}{L_B}\right)} + n'_B(W_B)\,\frac{\sinh\!\left(\frac{x}{L_B}\right)}{\sinh\!\left(\frac{W_B}{L_B}\right)}}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Collector, Emitter, and Base Currents

    ### Collector current

    Same procedure as the PN diode. The total collector current is the sum of hole and electron diffusion currents at the edges of the B-C junction:

    $$I_C = I_{Cp} + I_{Cn}$$

    $$I_{Cp} = -qA D_C \left.\frac{dp'_C}{dx'}\right|_{x'=0}, \qquad I_{Cn} = qA D_B \left.\frac{dn'_B}{dx}\right|_{x=W_B}$$

    where $A$ is the cross-section area, $D_C$ is the minority carrier (hole) diffusion coefficient in the collector, and $D_B$ is the minority carrier (electron) diffusion coefficient in the base.

    #### $I_{Cp}$: Hole current in the collector

    From continuity + diffusion in the semi-infinite collector ($p'_C = p_C - p_{C0}$, $L_C = \sqrt{\tau_C D_C}$):

    $$\frac{d^2 p'_C}{dx'^2} = \frac{p'_C}{L_C^2}$$

    Boundary conditions: $p'_C(0) = p_{C0}(e^{qV_{BC}/kT} - 1)$, $\;p'_C(\infty) = 0$

    **Solution:** $\;p'_C(x') = p_{C0}(e^{qV_{BC}/kT} - 1)\,e^{-x'/L_C}$

    $$I_{Cp} = -qA D_C \left.\frac{dp'_C}{dx'}\right|_{x'=0} = qA\frac{D_C}{L_C}\,p_{C0}\left(e^{qV_{BC}/kT} - 1\right)$$

    #### $I_{Cn}$: Electron current at collector edge of base

    From the base minority carrier profile $n'_B(x)$ derived in Section 5, evaluating the derivative at $x = W_B$:

    $$I_{Cn} = qA D_B \left.\frac{dn'_B}{dx}\right|_{x=W_B} = -qA\frac{D_B}{L_B}\,n_{B0}\left[\frac{1}{\sinh(W_B/L_B)}\left(e^{qV_{BE}/kT}-1\right) - \frac{1}{\tanh(W_B/L_B)}\left(e^{qV_{BC}/kT}-1\right)\right]$$

    #### Total collector current

    Combining $I_C = I_{Cn} + I_{Cp}$ and noting that $I_C$ is defined flowing *into* the collector (multiply by $-1$):

    $$\boxed{I_C = \underbrace{qA\left[\frac{D_B\,n_{B0}}{L_B\,\sinh(W_B/L_B)}\right]\!\left(e^{qV_{BE}/kT}-1\right)}_{\text{Forward-bias diode (BE junction)}} - \underbrace{qA\left[\frac{D_B\,n_{B0}}{L_B\,\tanh(W_B/L_B)} + \frac{D_C}{L_C}\,p_{C0}\right]\!\left(e^{qV_{BC}/kT}-1\right)}_{\text{Reverse-bias diode (BC junction)}}}$$

    The collector current is the superposition of **two diode terms**: a forward-biased B-E junction that injects carriers across the base, and a reverse-biased B-C junction that collects them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simplification 1: Short-base approximation

    If $W_B \ll L_B$, Taylor series expansions give:

    $$\sinh\!\left(\frac{W_B}{L_B}\right) \approx \frac{W_B}{L_B}, \qquad \cosh\!\left(\frac{W_B}{L_B}\right) \approx 1 + \left(\frac{W_B}{L_B}\right)^2, \qquad \tanh\!\left(\frac{W_B}{L_B}\right) \approx \frac{W_B}{L_B}$$

    Applying to the total collector current:

    $$\boxed{I_C = qA\left[\frac{D_B\,n_{B0}}{W_B}\right]\!\left(e^{qV_{BE}/kT}-1\right) - qA\left[\frac{D_B\,n_{B0}}{W_B} + \frac{D_C}{L_C}\,p_{C0}\right]\!\left(e^{qV_{BC}/kT}-1\right)}$$

    Physically, we have **linearized the minority carrier distribution** in the base:

    $$n'_B(x) \approx n'_B(0)\left(1 - \frac{x}{W_B}\right) + n'_B(W_B)\,\frac{x}{W_B}$$

    Carriers diffuse across the base before recombining — the profile is a straight line connecting the boundary values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    VBE_slider = mo.ui.slider(
        start=0.0, stop=0.7, value=0.5, step=0.01,
        label=r"$V_{BE}$ (V)"
    )
    VBC_slider = mo.ui.slider(
        start=-5.0, stop=0.5, value=-2.0, step=0.1,
        label=r"$V_{BC}$ (V)"
    )
    WB_slider = mo.ui.slider(
        start=0.1, stop=5.0, value=0.5, step=0.1,
        label=r"$W_B$ ($\mu$m)"
    )
    LB_slider = mo.ui.slider(
        start=0.5, stop=10.0, value=2.0, step=0.5,
        label=r"$L_B$ ($\mu$m)"
    )
    profile_controls = mo.vstack([
        mo.md("### Interactive: Base Carrier Profiles and the Short Base Approximation"),
        mo.md("Compare the full sinh solution with the short-base (linear) approximation."),
        mo.hstack([VBE_slider, VBC_slider], justify="start"),
        mo.hstack([WB_slider, LB_slider], justify="start"),
    ])
    return LB_slider, VBC_slider, VBE_slider, WB_slider, profile_controls


@app.cell(hide_code=True)
def _(
    LB_slider,
    VBC_slider,
    VBE_slider,
    WB_slider,
    kT_q,
    mo,
    ni_Si,
    np,
    plt,
    profile_controls,
):
    _VBE = VBE_slider.value
    _VBC = VBC_slider.value
    _WB = WB_slider.value * 1e-4   # cm
    _LnB = LB_slider.value * 1e-4  # cm
    _NB = 1e16

    _nB0 = ni_Si**2 / _NB

    _x = np.linspace(0, _WB, 300)

    # Full sinh solution
    _sinh_WB = np.sinh(_WB / _LnB)
    if abs(_sinh_WB) < 1e-30:
        _sinh_WB = 1e-30
    _dn_sinh = _nB0 * (
        (np.exp(_VBE / kT_q) - 1) * np.sinh((_WB - _x) / _LnB) / _sinh_WB
        + (np.exp(_VBC / kT_q) - 1) * np.sinh(_x / _LnB) / _sinh_WB
    )

    # Short base (linear) approximation
    _dn_linear = _nB0 * (
        (np.exp(_VBE / kT_q) - 1) * (1 - _x / _WB)
        + (np.exp(_VBC / kT_q) - 1) * (_x / _WB)
    )

    _x_um = _x * 1e4  # convert to micrometers

    _fig_p, _ax_p = plt.subplots(figsize=(10, 6))

    _nB_full = _nB0 + _dn_sinh
    _nB_lin = _nB0 + _dn_linear

    _ax_p.plot(_x_um, _nB_full, 'b-', linewidth=2.5, label='Full (sinh) solution')
    _ax_p.plot(_x_um, _nB_lin, 'r--', linewidth=2, label='Short-base (linear) approx.')
    _ax_p.axhline(_nB0, color='gray', linestyle=':', alpha=0.5, label=f'$n_{{B0}}$ = {_nB0:.2e} cm$^{{-3}}$')

    _ax_p.set_xlabel(r'Position in base, $x$ ($\mu$m)', fontsize=16)
    _ax_p.set_ylabel(r'$n_B(x)$ (cm$^{-3}$)', fontsize=16)
    _ax_p.set_title(f'Minority Electrons in Base ($V_{{BE}}$ = {_VBE:.2f} V, $V_{{BC}}$ = {_VBC:.1f} V)', fontsize=16, fontweight='bold')
    _ax_p.legend(fontsize=14)
    _ax_p.tick_params(labelsize=14)
    _ax_p.grid(True, alpha=0.3, which='both')
    _ax_p.set_xlim(0, _WB * 1e4)
    plt.tight_layout()

    _ratio = _WB / _LnB
    _info_p = mo.md(
        f"""
        **$W_B / L_B$ = {_ratio:.3f}** {'$\\ll 1$ (short base: linear approx. is excellent)' if _ratio < 0.1 else '(short base approx. may have noticeable error)' if _ratio < 0.5 else '$\\sim 1$ or larger: significant recombination in base, sinh solution needed'}

        Boundary values: $n_B(0)$ = {_nB0 + _nB0*(np.exp(_VBE/kT_q)-1):.2e} cm$^{{-3}}$,
        $n_B(W_B)$ = {_nB0 + _nB0*(np.exp(_VBC/kT_q)-1):.2e} cm$^{{-3}}$
        """
    )

    mo.vstack([profile_controls, _fig_p, _info_p])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Simplification 2: Neglect reverse current

    In **forward active** mode ($V_{BE} > 0$, $V_{BC} < 0$), the first term is much bigger than the second term (since $V_{BC} < 0$ makes $e^{qV_{BC}/kT} - 1 \approx -1$, a small reverse-bias current). Dropping the second term and substituting $n_{B0} = n_i^2/N_B$:

    $$\boxed{I_C = qA\left[\frac{D_B\,n_{B0}}{W_B}\right]\!\left(e^{qV_{BE}/kT}-1\right) = \frac{qA\,D_B\,n_i^2}{W_B\,N_B}\left(e^{qV_{BE}/kT} - 1\right) = I_S\left(e^{qV_{BE}/kT} - 1\right)}$$

    where $I_S = qA D_B n_i^2 / (W_B N_B)$ is the **saturation current**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Emitter current

    The total emitter current is the sum of diffusion currents at the edges of the B-E junction:

    $$I_E = I_{Ep} + I_{En}$$

    $$I_{Ep} = qA D_E \left.\frac{dp'_E}{dx''}\right|_{x''=0}, \qquad I_{En} = qA D_B \left.\frac{dn'_B}{dx}\right|_{x=0}$$

    (No minus sign on $I_{Ep}$ since $x''$ points in the opposite direction to conventional current flow.)

    #### $I_{En}$: Electron current at base edge

    From the base carrier profile (Section 5), evaluating the derivative at $x = 0$:

    $$I_{En} = qA D_B \left.\frac{dn'_B}{dx}\right|_{x=0} = -qA\frac{D_B}{L_B}\,n_{B0}\left[\frac{1}{\tanh(W_B/L_B)}\left(e^{qV_{BE}/kT}-1\right) - \frac{1}{\sinh(W_B/L_B)}\left(e^{qV_{BC}/kT}-1\right)\right]$$

    #### $I_{Ep}$: Hole current in the emitter

    From continuity + diffusion in the semi-infinite emitter ($p'_E = p_E - p_{E0}$, $L_E = \sqrt{\tau_E D_E}$):

    $$\frac{d^2 p'_E}{dx''^2} = \frac{p'_E}{L_E^2}$$

    Boundary conditions: $p'_E(0) = p_{E0}(e^{qV_{BE}/kT} - 1)$, $\;p'_E(\infty) = 0$

    **Solution:** $\;p'_E(x'') = p_{E0}(e^{qV_{BE}/kT} - 1)\,e^{-x''/L_E}$

    $$I_{Ep} = qA D_E \left.\frac{dp'_E}{dx''}\right|_{x''=0} = -qA\frac{D_E}{L_E}\,p_{E0}\left(e^{qV_{BE}/kT} - 1\right)$$

    #### Total emitter current

    Since $I_E$ is defined flowing in the negative direction (out of the emitter), multiply by $-1$:

    $$\boxed{I_E = \underbrace{qA\left[\frac{D_B\,n_{B0}}{L_B\,\tanh(W_B/L_B)} + \frac{D_E}{L_E}\,p_{E0}\right]\!\left(e^{qV_{BE}/kT}-1\right)}_{\text{Forward-bias diode (BE junction)}} - \underbrace{qA\left[\frac{D_B\,n_{B0}}{L_B\,\sinh(W_B/L_B)}\right]\!\left(e^{qV_{BC}/kT}-1\right)}_{\text{Reverse-bias diode (BC junction)}}}$$

    #### Short-base approximation

    $$I_E = qA\left[\frac{D_B\,n_{B0}}{W_B} + \frac{D_E\,p_{E0}}{L_E}\right]\!\left(e^{qV_{BE}/kT}-1\right) - qA\left[\frac{D_B\,n_{B0}}{W_B}\right]\!\left(e^{qV_{BC}/kT}-1\right)$$

    #### Neglect reverse current ($V_{BC} < 0$, $V_{BE} > 0$)

    $$\boxed{I_E = qA\left[\frac{D_B\,n_{B0}}{W_B} + \frac{D_E\,p_{E0}}{L_E}\right]\!\left(e^{qV_{BE}/kT}-1\right)}$$

    where $n_{B0} = n_i^2/N_B$ and $p_{E0} = n_i^2/N_E$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Base current

    With short-base approximation and neglecting reverse current:

    $$I_B = I_E - I_C = qA\left[\frac{D_B\,n_{B0}}{W_B} + \frac{D_E\,p_{E0}}{L_E}\right]\!\left(e^{qV_{BE}/kT}-1\right) - qA\left[\frac{D_B\,n_{B0}}{W_B}\right]\!\left(e^{qV_{BE}/kT}-1\right)$$

    The $D_B n_{B0}/W_B$ terms cancel:

    $$\boxed{I_B = qA\left[\frac{D_E\,p_{E0}}{L_E}\right]\!\left(e^{qV_{BE}/kT}-1\right) = \frac{qA\,D_E\,n_i^2}{L_E\,N_E}\left(e^{qV_{BE}/kT}-1\right)}$$

    (Additional components not captured here: recombination in the base quasi-neutral region and in the depletion regions. These become important at low $V_{BE}$ and in non-ideal devices.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary of BJT currents

    With infinitely long E and C, short-base approximation, and neglecting reverse saturation current:

    $$I_E = qA\left[\underbrace{\frac{D_B\,n_{B0}}{W_B}}_{\Rightarrow\,I_{En}} + \underbrace{\frac{D_E\,p_{E0}}{L_E}}_{\Rightarrow\,I_{Ep}}\right]\!\left(e^{qV_{BE}/kT}-1\right)$$

    $$I_C = qA\left[\frac{D_B\,n_{B0}}{W_B}\right]\!\left(e^{qV_{BE}/kT}-1\right) \qquad (I_C \approx I_{Cn} = I_{En})$$

    $$I_B = qA\left[\frac{D_E\,p_{E0}}{L_E}\right]\!\left(e^{qV_{BE}/kT}-1\right)$$

    $$I_E = I_B + I_C$$

    ### Finite emitter length

    If the emitter has finite length $W_E$ (with $W_E \ll L_E$, i.e. short emitter), $L_E$ is replaced by $W_E$ in $I_{Ep}$:

    $$I_E = qA\left[\frac{D_B\,n_{B0}}{W_B} + \frac{D_E\,p_{E0}}{W_E}\right]\!\left(e^{qV_{BE}/kT}-1\right)$$

    Compared to the infinite-emitter result where $D_E p_{E0}/L_E$ appears, the short-emitter approximation uses $D_E p_{E0}/W_E$ — a larger hole current into the emitter (steeper gradient over a shorter distance). The same substitution applies to $I_B$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BJT metrics

    **Emitter efficiency** — fraction of emitter current due to electrons (the useful carriers that cross the base):

    $$\gamma_E = \frac{I_{En}}{I_E} = \frac{1}{1 + I_{Ep}/I_{En}} = \frac{1}{1 + \dfrac{D_E\,N_B\,W_B}{D_B\,N_E\,W_E}}$$

    $\gamma_E \to 1$ when $N_E \gg N_B$ (heavily doped emitter, lightly doped base) and $W_B \ll W_E$ (short base).

    **Common-base current gain:**

    $$\alpha_F = \frac{I_C}{I_E} = \frac{1}{1 + \dfrac{D_E\,N_B\,W_B}{D_B\,N_E\,W_E}} = \gamma_E$$

    (Since $I_C = I_{En}$ under our approximations.)

    **Common-emitter current gain:**

    $$\beta_F = \frac{I_C}{I_B} = \frac{D_B\,N_E\,W_E}{D_E\,N_B\,W_B}$$

    As $\alpha_F \to 1$, $\beta_F \to \infty$.

    > **Note:** All expressions above assume the **short-emitter approximation** ($W_E \ll L_E$). For an infinitely long emitter, replace $W_E \to L_E$ everywhere.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    NE_slider = mo.ui.slider(
        start=17, stop=20, value=18, step=0.1,
        label=r"log$_{10}$($N_E$) [cm$^{-3}$]"
    )
    NB_slider = mo.ui.slider(
        start=15, stop=18, value=16, step=0.1,
        label=r"log$_{10}$($N_B$) [cm$^{-3}$]"
    )
    gain_controls = mo.vstack([
        mo.md(r"## 7. Interactive: $\beta_F$ vs. Doping"),
        mo.md("Adjust emitter and base doping to see the effect on $\\beta_F$ and $\\alpha_F = \\gamma_E$. (Short-emitter approximation: $W_E$ = 2 $\\mu$m)"),
        mo.hstack([NE_slider, NB_slider], justify="start"),
    ])
    return NB_slider, NE_slider, gain_controls


@app.cell(hide_code=True)
def _(NB_slider, NE_slider, gain_controls, kT_q, mo, np, plt):
    _NE = 10 ** NE_slider.value
    _NB = 10 ** NB_slider.value
    _WB = 0.5e-4    # 0.5 um
    _WE = 2.0e-4    # 2 um (short emitter)

    _mu_nB = 1350.0 / (1 + (_NB / 1e17)**0.7)
    _mu_pE = 480.0 / (1 + (_NE / 1e17)**0.7)
    _DnB = kT_q * max(_mu_nB, 50)
    _DpE = kT_q * max(_mu_pE, 10)

    _beta = (_DnB / _DpE) * (_NE / _NB) * (_WE / _WB)
    _alpha = _beta / (1 + _beta)

    _ratio = np.logspace(-1, 5, 500)
    _beta_vs_ratio = (_DnB / _DpE) * _ratio * (_WE / _WB)
    _alpha_vs_ratio = _beta_vs_ratio / (1 + _beta_vs_ratio)

    _fig_gain, (_ax_beta, _ax_alpha) = plt.subplots(1, 2, figsize=(14, 6))

    _ax_beta.loglog(_ratio, _beta_vs_ratio, 'b-', linewidth=2.5)
    _ax_beta.axvline(_NE / _NB, color='red', linestyle='--', linewidth=2, label=f'$N_E/N_B$ = {_NE/_NB:.0e}')
    _ax_beta.axhline(_beta, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    _ax_beta.plot(_NE / _NB, _beta, 'ro', markersize=12, zorder=5)

    _ax_beta.annotate(
        f'$\\beta_F$ = {_beta:.0f}',
        xy=(_NE / _NB, _beta),
        xytext=(_NE / _NB * 3, _beta * 0.3),
        fontsize=16, fontweight='bold', color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=2),
    )

    _ax_beta.axhspan(50, 300, alpha=0.08, color='green', label='Typical $\\beta_F$ range (50-300)')

    _ax_beta.set_xlabel(r'$N_E / N_B$', fontsize=16)
    _ax_beta.set_ylabel(r'$\beta_F$', fontsize=16)
    _ax_beta.set_title(r'$\beta_F = \frac{D_B}{D_E} \cdot \frac{N_E}{N_B} \cdot \frac{W_E}{W_B}$', fontsize=16, fontweight='bold')
    _ax_beta.legend(fontsize=14)
    _ax_beta.tick_params(labelsize=14)
    _ax_beta.grid(True, alpha=0.3, which='both')
    _ax_beta.set_xlim(0.1, 1e5)
    _ax_beta.set_ylim(1, 1e5)

    _ax_alpha.semilogx(_ratio, _alpha_vs_ratio, 'b-', linewidth=2.5)
    _ax_alpha.axvline(_NE / _NB, color='red', linestyle='--', linewidth=2, label=f'$N_E/N_B$ = {_NE/_NB:.0e}')
    _ax_alpha.axhline(_alpha, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    _ax_alpha.plot(_NE / _NB, _alpha, 'ro', markersize=12, zorder=5)

    _ax_alpha.annotate(
        f'$\\alpha_F$ = {_alpha:.4f}',
        xy=(_NE / _NB, _alpha),
        xytext=(_NE / _NB * 3, _alpha - 0.15),
        fontsize=16, fontweight='bold', color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=2),
    )

    _ax_alpha.axhspan(0.98, 1.0, alpha=0.08, color='green', label=r'Typical $\alpha_F$ range (0.98-1.0)')

    _ax_alpha.set_xlabel(r'$N_E / N_B$', fontsize=16)
    _ax_alpha.set_ylabel(r'$\alpha_F = \gamma_E$', fontsize=16)
    _ax_alpha.set_title(r'$\alpha_F = \gamma_E = \frac{\beta_F}{1 + \beta_F}$', fontsize=16, fontweight='bold')
    _ax_alpha.legend(fontsize=14)
    _ax_alpha.tick_params(labelsize=14)
    _ax_alpha.grid(True, alpha=0.3, which='both')
    _ax_alpha.set_xlim(0.1, 1e5)
    _ax_alpha.set_ylim(0, 1.05)

    plt.tight_layout()

    _info_gain = mo.md(
        f"""
        **Parameters (short-emitter approximation):**
        $N_E$ = {_NE:.1e} cm$^{{-3}}$, $N_B$ = {_NB:.1e} cm$^{{-3}}$,
        $W_B$ = {_WB*1e4:.1f} $\\mu$m, $W_E$ = {_WE*1e4:.0f} $\\mu$m |
        $D_B$ = {_DnB:.1f} cm$^2$/s, $D_E$ = {_DpE:.1f} cm$^2$/s

        $$\\beta_F = \\frac{{{_DnB:.1f}}}{{{_DpE:.1f}}} \\times \\frac{{{_NE:.1e}}}{{{_NB:.1e}}} \\times \\frac{{{_WE*1e4:.0f}\\ \\mu m}}{{{_WB*1e4:.1f}\\ \\mu m}} = \\mathbf{{{_beta:.0f}}}$$

        $$\\alpha_F = \\gamma_E = \\frac{{\\beta_F}}{{1 + \\beta_F}} = \\frac{{{_beta:.0f}}}{{1 + {_beta:.0f}}} = \\mathbf{{{_alpha:.4f}}}$$
        """
    )

    mo.vstack([gain_controls, _fig_gain, _info_gain])
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _img_checkpoint = mo.image(src=f"{IMAGE_BASE}/checkpoint-bjt-comparison.png", width="100%")
    quiz=mo.vstack([mo.md(r"""
    ### Quick Check

    **Q1:** If you want to increase $\beta_F$ by a factor of 10, which is most effective:
    - **(A)** Increase $N_E$ by 10×
    - **(B)** Decrease $N_B$ by 10×
    - **(C)** Decrease $W_B$ by 10×

    <!--
    **A:** All three work! From $\beta_F = \frac{D_B\,N_E\,W_E}{D_E\,N_B\,W_B}$:
    - (A) increases $N_E/N_B$ by 10× ✓
    - (B) increases $N_E/N_B$ by 10× ✓
    - (C) increases $W_E/W_B$ by 10× ✓

    But **(C) is most practical in fabrication** — modern BJTs use very thin bases (< 100 nm). Verify with the sliders above!
    -->

    **Bonus:** Why not just make $W_B$ arbitrarily small? (Hint: punch-through, base resistance, Early effect...)

    ---

    **Q2:** Three NPN BJTs have identical $N_E$, $D_B$, $D_E$, and $W_E$, but differ in base width $W$ and base doping $N_B$:
    """),
    mo.hstack([_img_checkpoint], justify="center"),
    mo.md(r"""
    **(a)** Which has the highest emitter efficiency ($\gamma_E$) and common-base current gain ($\alpha_F$)?

    **(b)** Which has the highest common-emitter current gain ($\beta_F$)?

    <!---
    **A:**

    Recall: $\gamma_E = \alpha_F = \dfrac{1}{1 + \dfrac{D_E\,N_B\,W_B}{D_B\,N_E\,W_E}}$ and $\beta_F = \dfrac{D_B\,N_E\,W_E}{D_E\,N_B\,W_B}$

    The figure of merit is the product $N_B \cdot W_B$:
    - **BJT 1:** $N_B \cdot W$
    - **BJT 2:** $N_B \cdot 2W$ = $2\,N_B W$
    - **BJT 3:** $2N_B \cdot W$ = $2\,N_B W$

    **(a)** **BJT 1** has the smallest $N_B W_B$ product, so it has the highest $\gamma_E$ and $\alpha_F$.

    **(b)** **BJT 1** also has the highest $\beta_F$, since $\beta_F \propto 1/(N_B W_B)$.

    BJT 2 and BJT 3 have the **same** $\gamma_E$, $\alpha_F$, and $\beta_F$ — doubling the base width has the same effect as doubling the base doping.
    -->
    """)])
    return (quiz,)


@app.cell
def _(mo, quiz):
    mo.vstack([quiz])
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _img_early = mo.image(src=f"{IMAGE_BASE}/early-effect.png", width="80%")
    mo.vstack([
        mo.md(r"""
        ## 8. Base Width Modulation: Early Effect

        As the reverse bias on the B-C junction increases, the B-C depletion region widens, reducing the quasi-neutral base width $W_B$. This is called the **Early effect** (or **base width modulation**), named after James Early of Bell Labs.

        Since $I_C \propto 1/W_B$ and $\beta_F \propto 1/W_B$, a smaller $W_B$ means:
        - $\alpha_F$ and $\beta_F$ increase
        - $I_C$ increases with $V_{CE}$ even at constant $I_B$

        This gives the output characteristics a finite slope in the active region rather than perfectly flat curves.
        """),
        mo.hstack([_img_early], justify="center"),
        mo.md(r"""
        All curves extrapolate back to $V_{CE} = -V_A$, where $V_A$ is the **Early voltage**:

        $$I_C = \beta_F I_B \left(1 + \frac{V_{CE}}{V_A}\right)$$

        **To reduce the Early effect:** choose $N_B > N_C$ so that most of the change in the depletion layer width is in the collector (not the base).
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    IB_base_slider = mo.ui.slider(
        start=1, stop=100, value=10, step=1,
        label=r"$I_B$ step ($\mu$A)"
    )
    VA_slider = mo.ui.slider(
        start=20, stop=200, value=100, step=10,
        label=r"Early voltage $V_A$ (V)"
    )
    output_controls = mo.vstack([
        mo.md("### Interactive: Output Characteristics"),
        mo.md("The output characteristics show $I_C$ vs $V_{CE}$ for different base currents."),
        mo.hstack([IB_base_slider, VA_slider], justify="start"),
    ])
    return IB_base_slider, VA_slider, output_controls


@app.cell(hide_code=True)
def _(IB_base_slider, VA_slider, mo, np, output_controls, plt):
    _IB_step = IB_base_slider.value * 1e-6  # A
    _VA = VA_slider.value  # V
    _beta0 = 100

    _VCE = np.linspace(0, 10, 500)

    _fig_o, _ax_o = plt.subplots(figsize=(10, 7))

    _colors = plt.cm.viridis(np.linspace(0.2, 0.9, 5))

    for _i in range(5):
        _IB = (_i + 1) * _IB_step
        _VBE_sat = 0.2
        _IC_active = _beta0 * _IB * (1 + _VCE / _VA)
        # Phenomenological saturation model (smooth transition from linear to active)
        _IC_sat = _beta0 * _IB * (_VCE / _VBE_sat) * np.tanh(_VCE / _VBE_sat * 3)
        _IC = np.where(_VCE < _VBE_sat, _IC_sat, _IC_active)
        _IC = np.minimum(_IC, _IC_active)

        _ax_o.plot(_VCE, _IC * 1e3, '-', linewidth=2.5, color=_colors[_i],
                   label=f'$I_B$ = {_IB*1e6:.0f} $\\mu$A')

    _ax_o.axvline(0.2, color='gray', linestyle=':', alpha=0.5)
    _ax_o.text(0.05, _ax_o.get_ylim()[1] if _ax_o.get_ylim()[1] > 0 else 5, 'Sat.',
               fontsize=14, color='gray', ha='center', va='top')
    _ax_o.text(3, _ax_o.get_ylim()[1] if _ax_o.get_ylim()[1] > 0 else 5, 'Active region',
               fontsize=14, color='gray', ha='center', va='top')

    _IB_mid = 3 * _IB_step
    _IC_mid_0 = _beta0 * _IB_mid
    _ax_o.plot([-_VA, 0], [0, _IC_mid_0 * 1e3], 'k:', linewidth=1, alpha=0.4)
    _ax_o.annotate(f'$-V_A$ = {-_VA} V', xy=(-_VA * 0.1, 0), fontsize=12, color='gray')

    _ax_o.set_xlabel(r'$V_{CE}$ (V)', fontsize=16)
    _ax_o.set_ylabel(r'$I_C$ (mA)', fontsize=16)
    _ax_o.set_title('BJT Output Characteristics (Common-Emitter)', fontsize=16, fontweight='bold')
    _ax_o.legend(fontsize=14, loc='upper left')
    _ax_o.tick_params(labelsize=14)
    _ax_o.grid(True, alpha=0.3)
    _ax_o.set_xlim(-0.5, 10)
    _ax_o.set_ylim(0, None)
    plt.tight_layout()

    _ro = _VA / (_beta0 * 3 * _IB_step) if _beta0 * 3 * _IB_step > 0 else float('inf')
    _info_o = mo.md(
        f"""
        **Output resistance:** $r_o = V_A / I_C$.
        At $I_B$ = {3*_IB_step*1e6:.0f} $\\mu$A: $r_o \\approx$ {_ro/1e3:.1f} k$\\Omega$

        All curves extrapolate back to $V_{{CE}} = -V_A = -{_VA}$ V.
        """
    )

    mo.vstack([output_controls, _fig_o, _info_o])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Gummel Number Formulation

    For **non-uniform doping** (common in real BJTs), the collector current is expressed using the **Gummel number** $G_B$:

    $$I_C = \frac{qA_E n_i^2}{G_B}\left(e^{qV_{BE}/kT} - 1\right)$$

    where the base Gummel number is:

    $$G_B = \int_0^{W_B} \frac{N_B(x)}{D_B(x)}\,dx$$

    **Physical meaning:** $G_B$ represents the integrated "resistance" to minority carrier diffusion through the base. Higher doping or lower diffusivity increases $G_B$ and reduces $I_C$.

    Similarly, the base current uses the emitter Gummel number:

    $$I_B = \frac{qA_E n_i^2}{G_E}\left(e^{qV_{BE}/kT} - 1\right), \qquad G_E = \int_0^{W_E} \frac{N_E(x'')}{D_E(x'')}\,dx''$$

    **Current gain:**

    $$\beta_F = \frac{G_E}{G_B}$$

    The Gummel number naturally accounts for non-uniform doping profiles, bandgap narrowing, and position-dependent diffusivity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    IS_slider = mo.ui.slider(
        start=-16, stop=-10, value=-14, step=0.5,
        label=r"log$_{10}$($I_S$) [A]"
    )
    beta_slider = mo.ui.slider(
        start=10, stop=500, value=100, step=10,
        label=r"$\beta_F$"
    )
    n_slider = mo.ui.slider(
        start=1.0, stop=2.0, value=1.0, step=0.05,
        label=r"Ideality factor $n$"
    )
    gummel_controls = mo.vstack([
        mo.md("## 10. Interactive: Gummel Plot"),
        mo.md(r"""
        The **Gummel plot** ($\log I_C$ and $\log I_B$ vs $V_{BE}$) is the standard method for extracting BJT parameters:

        - The ideal $I_C$ line has a slope of **60 mV/decade** (at 300 K, for $n = 1$).
        - Extrapolating this straight line to $V_{BE} = 0$ gives the **saturation current** $I_S$.
        - The base Gummel number is then $G_B = A_E q n_i^2 / I_S$.
        - The vertical separation between $I_C$ and $I_B$ gives $\beta_F$.
        """),
        mo.hstack([IS_slider, beta_slider, n_slider], justify="start"),
    ])
    return IS_slider, beta_slider, gummel_controls, n_slider


@app.cell(hide_code=True)
def _(IS_slider, beta_slider, gummel_controls, kT_q, mo, n_slider, np, plt):
    _IS = 10 ** IS_slider.value
    _beta = beta_slider.value
    _n = n_slider.value

    _VBE = np.linspace(0.0, 0.8, 500)

    _IC_n = _IS * (np.exp(_VBE / (_n * kT_q)) - 1)
    _IB_n = _IC_n / _beta

    _IB_rec = (_IS * 1e3) * (np.exp(_VBE / (2 * kT_q)) - 1)
    _IB_total = _IB_n + _IB_rec

    _fig_g, _ax_g = plt.subplots(figsize=(10, 7))

    _ax_g.semilogy(_VBE, np.maximum(_IC_n, 1e-30), 'b-', linewidth=2.5, label=f'$I_C$ ($n$ = {_n:.2f})')
    _ax_g.semilogy(_VBE, np.maximum(_IB_total, 1e-30), 'r-', linewidth=2.5, label='$I_B$ (total)')
    _ax_g.semilogy(_VBE, np.maximum(_IB_n, 1e-30), 'r--', linewidth=1.5, alpha=0.5, label=f'$I_B$ (ideal, $n$ = {_n:.2f})')
    _ax_g.semilogy(_VBE, np.maximum(_IB_rec, 1e-30), 'r:', linewidth=1.5, alpha=0.5, label='$I_B$ (recomb., $n$ = 2)')

    _v_annotate = 0.55
    _ic_at_v = _IS * (np.exp(_v_annotate / (_n * kT_q)) - 1)
    _ib_at_v = _ic_at_v / _beta + (_IS * 1e3) * (np.exp(_v_annotate / (2 * kT_q)) - 1)
    _beta_actual = _ic_at_v / _ib_at_v if _ib_at_v > 0 else _beta
    _ax_g.annotate(
        f'$\\beta$ = {_beta_actual:.0f}\nat $V_{{BE}}$ = {_v_annotate} V',
        xy=(_v_annotate, _ic_at_v), fontsize=14,
        xytext=(_v_annotate - 0.2, _ic_at_v * 10),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
    )

    # I_S intercept line
    _ax_g.axhline(_IS, color='blue', linestyle=':', linewidth=1.5, alpha=0.6)
    _ax_g.annotate(f'$I_S$ = {_IS:.1e} A',
                   xy=(0.02, _IS), fontsize=13, color='blue',
                   xytext=(0.08, _IS * 5),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.2))

    _ax_g.annotate('', xy=(0.46, 1e-6), xytext=(0.4, 1e-6),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    _ax_g.text(0.35, 3e-7, '60 mV/dec\n($n$ = 1)', fontsize=12, color='green', ha='center')

    _ax_g.set_xlabel(r'$V_{BE}$ (V)', fontsize=16)
    _ax_g.set_ylabel(r'Current (A)', fontsize=16)
    _ax_g.set_title('Gummel Plot', fontsize=16, fontweight='bold')
    _ax_g.legend(fontsize=14, loc='upper left')
    _ax_g.tick_params(labelsize=14)
    _ax_g.grid(True, alpha=0.3, which='both')
    _ax_g.set_xlim(0, 0.8)
    _ax_g.set_ylim(1e-18, 1e0)
    plt.tight_layout()

    _slope = 1 / (_n * kT_q * np.log(10))
    _info_g = mo.md(
        f"""
        **$I_S$ = {_IS:.1e} A, $\\beta_F$ = {_beta:.0f}, $n$ = {_n:.2f}**

        Semi-log slope of $I_C$: **{_slope:.1f} decades/V** ({_n * 60:.0f} mV/decade)

        **Parameter extraction:** Extrapolate the straight-line portion of $\\log I_C$ to $V_{{BE}} = 0$ $\\Rightarrow$ intercept = $I_S$.
        Then $G_B = A_E q n_i^2 / I_S$.

        At low $V_{{BE}}$, the base current is dominated by **recombination** in the SCR ($n = 2$).
        At higher $V_{{BE}}$, the ideal diffusion component ($n = {_n:.2f}$) takes over.
        """
    )

    mo.vstack([gummel_controls, _fig_g, _info_g])
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _img_em = mo.image(src=f"{IMAGE_BASE}/ebers-moll.png", width="70%")
    mo.vstack([
        mo.md(r"""
        ## 11. Ebers-Moll Model *(optional)*

        The general BJT model uses two coupled diode equations. Define the forward and reverse diode currents:

        $$I_F = I_{ES}\left(e^{qV_{BE}/kT} - 1\right), \qquad I_R = I_{CS}\left(e^{qV_{BC}/kT} - 1\right)$$

        The terminal currents are:

        $$I_C = \alpha_F\,I_F - I_R$$

        $$I_E = -I_F + \alpha_R\,I_R$$

        **Reciprocity relation:** $\alpha_F I_{ES} = \alpha_R I_{CS}$

        The circuit representation consists of two diodes (B-E and B-C) with dependent current sources $\alpha_F I_F$ and $\alpha_R I_R$:
        """),
        mo.hstack([_img_em], justify="center"),
        mo.md(r"""
        | Parameter | Meaning | Typical (Si) |
        |:---|:---|:---|
        | $I_{ES}$ | B-E saturation current (A) | $10^{-15}$ – $10^{-12}$ |
        | $I_{CS}$ | B-C saturation current (A) | $10^{-13}$ – $10^{-10}$ |
        | $\alpha_F$ | Forward current gain | 0.98 – 0.998 |
        | $\alpha_R$ | Reverse current gain | 0.1 – 0.5 |
        | $\beta_F$ | Forward CE gain | 50 – 500 |
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Short-base ($W_B \ll L_B$) and short-emitter ($W_E \ll L_E$) approximations:

    | Quantity | Expression |
    |:---|:---|
    | $I_C$ | $\dfrac{qA\,D_B\,n_i^2}{W_B\,N_B}\left(e^{qV_{BE}/kT}-1\right)$ |
    | $I_B$ | $\dfrac{qA\,D_E\,n_i^2}{W_E\,N_E}\left(e^{qV_{BE}/kT}-1\right)$ |
    | $\gamma_E$ (emitter efficiency) | $\left[1 + \dfrac{D_E\,N_B\,W_B}{D_B\,N_E\,W_E}\right]^{-1}$ |
    | $\alpha_F$ (common-base gain) | $\gamma_E$ |
    | $\beta_F$ (common-emitter gain) | $\dfrac{D_B\,N_E\,W_E}{D_E\,N_B\,W_B}$ |
    | $I_S$ | $qA\,D_B\,n_i^2 / (W_B\,N_B)$ |

    > For infinitely long emitter, replace $W_E \to L_E$.

    **Key ideas:**
    - $I_C$ is set by electron diffusion across the base
    - $I_B$ is set by hole injection into the emitter
    - Gummel plot ($\log I$ vs $V_{BE}$) is the standard characterization method
    - Early effect gives finite output resistance $r_o = V_A / I_C$
    """)
    return


if __name__ == "__main__":
    app.run()
