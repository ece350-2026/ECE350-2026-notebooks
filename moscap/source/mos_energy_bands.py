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

    q = 1.6e-19
    k = 1.381e-23
    kT_300 = 0.02585
    eps_0 = 8.854e-14   # F/cm
    eps_Si = 11.7
    eps_ox = 3.9
    ni_Si = 1.1e10
    Eg_Si = 1.12        # eV
    chi_Si = 4.05        # eV
    Nc_Si = 2.86e19
    Nv_Si = 1.08e19

    # Intrinsic level position (NOT at mid-gap)
    # E_c - E_Fi = Eg/2 + (kT/2)*ln(Nc/Nv)
    # E_Fi - E_v = Eg/2 + (kT/2)*ln(Nv/Nc)
    Ec_minus_Ei = Eg_Si / 2 + kT_300 / 2 * np.log(Nc_Si / Nv_Si)
    Ei_minus_Ev = Eg_Si / 2 + kT_300 / 2 * np.log(Nv_Si / Nc_Si)

    # SiO2 band offsets
    Eg_ox = 9.0          # eV
    chi_ox = 0.95        # eV
    dEc_ox = chi_Si - chi_ox  # conduction band offset Si-SiO2 ~3.1 eV
    dEv_ox = Eg_ox - Eg_Si - dEc_ox  # valence band offset ~4.8 eV

    try:
        _test = Path(__file__).parent / "images"
        if _test.exists():
            ASSET_DIR = Path(__file__).parent
        else:
            raise FileNotFoundError
    except Exception:
        ASSET_DIR = None
    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/mos-energy-bands/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    mo.md(r"""
    # MOS Capacitor: Energy Bands and Electrostatics
    **ECE350 Lectures 29-30**

    Hu, Chapter 5

    This notebook develops the energy band structure and electrostatics of the Metal-Oxide-Semiconductor (MOS) capacitor — the heart of the MOSFET.

    **Contents:**

    1. **MOS Structure Overview** — ideal MOS assumptions, flat-band energy diagram
    2. **Applying a Gate Voltage** — regimes of operation (accumulation, depletion, threshold, inversion), energy band diagrams in each regime, interactive P-type energy bands
    3. **Threshold Condition** — derivation of why $\phi_s = 2\phi_B$ at threshold
    4. **Electrostatics of the MOS Capacitor** — charge density, electric field, and potential from Poisson's equation; interactive P-type electrostatics
    5. **Threshold Voltage Derivation** — $W_{dep,max}$, $V_T$ expression, numerical example
    6. **N-Body vs. P-Body MOS Capacitor** — comparison table, interactive N-type energy bands and electrostatics
    7. **Summary** — key equations and concepts
    """)
    return (
        Ec_minus_Ei,
        Ei_minus_Ev,
        chi_Si,
        dEc_ox,
        dEv_ox,
        eps_0,
        eps_Si,
        eps_ox,
        kT_300,
        mo,
        ni_Si,
        np,
        plt,
        q,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MOS Structure Overview

    The **MOS capacitor** consists of three layers:

    1. **Metal gate** — a conductor (e.g., aluminum, polysilicon)
    2. **Oxide** — a thin insulating layer of SiO$_2$ (typical thickness 2–20 nm)
    3. **Semiconductor body** — assume p-type silicon first (the "substrate" or "body")

    The MOS capacitor is the heart of the MOSFET, the most important device in modern electronics.

    ### Ideal MOS Assumptions

    - The entire gate is at the same potential (metal has zero resistance)
    - The oxide is a perfect insulator (no current flows through it)
    - No charges in the oxide or at the oxide-semiconductor interface
    - The semiconductor is uniformly doped and thick enough for bulk properties
    - $\Psi_M = \Psi_S$ (gate and semiconductor work functions are identical)

    Under these assumptions, the bands in the semiconductor are **flat** at $V_G = 0$.
    """)
    return


@app.cell(hide_code=True)
def _(
    Ec_minus_Ei,
    Ei_minus_Ev,
    chi_Si,
    dEc_ox,
    dEv_ox,
    kT_300,
    mo,
    ni_Si,
    np,
    plt,
):
    _Na = 1e16
    _phi_B = kT_300 * np.log(_Na / ni_Si)

    _EF = 0.0
    _Ei_bulk = _phi_B
    _Ec_bulk = _Ei_bulk + Ec_minus_Ei
    _Ev_bulk = _Ei_bulk - Ei_minus_Ev

    _Evac = _Ec_bulk + chi_Si

    _Psi_S = chi_Si + Ec_minus_Ei + _phi_B
    _Psi_M = _Psi_S

    _EFm = _EF

    _fig, _ax = plt.subplots(figsize=(8.8, 4.8))

    _xM = [-1.8, -1.2]
    _xO = [-1.2, 0.0]
    _xS = [0.0, 2.0]

    # Metal
    _ax.fill_between(_xM, _EFm - 2.5, _EFm, color='#6baed6', alpha=0.35)
    _ax.plot(_xM, [_EFm, _EFm], 'g-', linewidth=2)
    _ax.text(_xM[0] + 0.05, _EFm + 0.12, '$E_F$', fontsize=14, color='green')
    _ax.plot(_xM, [_Evac, _Evac], 'k-', linewidth=1.5)
    _ax.text(_xM[0] + 0.05, _Evac + 0.12, '$E_{vac}$', fontsize=14)

    # Work function annotation (metal)
    _cap = 0.06
    _xwf_m = -1.5
    _ax.plot([_xwf_m, _xwf_m], [_EFm, _Evac], color='purple', lw=1.5)
    _ax.plot([_xwf_m - _cap, _xwf_m + _cap], [_EFm, _EFm], color='purple', lw=1.5)
    _ax.plot([_xwf_m - _cap, _xwf_m + _cap], [_Evac, _Evac], color='purple', lw=1.5)
    _ax.text(_xwf_m + 0.1, (_EFm + _Evac) / 2, r'$q\Psi_M$', fontsize=14, color='purple', va='center')

    # Oxide — wide-gap insulator with Ec,ox and Ev,ox
    _Ec_ox_semi = _Ec_bulk + dEc_ox
    _Ev_ox_semi = _Ev_bulk - dEv_ox
    _Ec_ox_metal = _Ec_ox_semi  # flat-band: no E-field in oxide
    _Ev_ox_metal = _Ev_ox_semi

    _ax.fill_between(_xO, [_Ev_ox_metal, _Ev_ox_semi], [_Ec_ox_metal, _Ec_ox_semi],
                     color='#ffffcc', alpha=0.5, edgecolor='none')
    _ax.plot(_xO, [_Ec_ox_metal, _Ec_ox_semi], 'k-', linewidth=2)
    _ax.plot(_xO, [_Ev_ox_metal, _Ev_ox_semi], 'k-', linewidth=2)
    _ax.text(-0.6, _Ec_ox_semi + 0.15, '$E_{c,ox}$', fontsize=14, ha='center')
    _ax.text(-0.6, _Ev_ox_semi - 0.3, '$E_{v,ox}$', fontsize=14, ha='center')

    # Semiconductor
    _ax.plot(_xS, [_Ec_bulk, _Ec_bulk], 'b-', linewidth=2.5)
    _ax.plot(_xS, [_Ev_bulk, _Ev_bulk], 'r-', linewidth=2.5)
    _ax.plot(_xS, [_Ei_bulk, _Ei_bulk], 'k--', linewidth=1, alpha=0.5)
    _ax.plot(_xS, [_EF, _EF], 'g-', linewidth=2)
    _ax.plot(_xS, [_Evac, _Evac], 'k-', linewidth=1.5)

    _ax.text(2.1, _Ec_bulk, '$E_c$', fontsize=14, va='center', color='blue')
    _ax.text(2.1, _Ev_bulk, '$E_v$', fontsize=14, va='center', color='red')
    _ax.text(2.1, _Ei_bulk, '$E_{Fi}$', fontsize=14, va='center', alpha=0.6)
    _ax.text(2.1, _EF, '$E_{Fs}$', fontsize=14, va='center', color='green')
    _ax.text(2.1, _Evac, '$E_{vac}$', fontsize=14, va='center')

    # Electron affinity annotation
    _xchi = 1.2
    _ax.plot([_xchi, _xchi], [_Ec_bulk, _Evac], color='darkorange', lw=1.5)
    _ax.plot([_xchi - _cap, _xchi + _cap], [_Ec_bulk, _Ec_bulk], color='darkorange', lw=1.5)
    _ax.plot([_xchi - _cap, _xchi + _cap], [_Evac, _Evac], color='darkorange', lw=1.5)
    _ax.text(_xchi + 0.12, (_Ec_bulk + _Evac) / 2, r'$q\chi$', fontsize=14, color='darkorange', va='center')

    # Semiconductor work function annotation
    _xwf_s = 1.7
    _ax.plot([_xwf_s, _xwf_s], [_EF, _Evac], color='purple', lw=1.5)
    _ax.plot([_xwf_s - _cap, _xwf_s + _cap], [_EF, _EF], color='purple', lw=1.5)
    _ax.plot([_xwf_s - _cap, _xwf_s + _cap], [_Evac, _Evac], color='purple', lw=1.5)
    _ax.text(_xwf_s + 0.12, (_EF + _Evac) / 2, r'$q\Psi_S$', fontsize=14, color='purple', va='center')

    # phi_B annotation
    _xphi = 0.7
    _ax.plot([_xphi, _xphi], [_EF, _Ei_bulk], color='purple', lw=1.5)
    _ax.plot([_xphi - _cap, _xphi + _cap], [_EF, _EF], color='purple', lw=1.5)
    _ax.plot([_xphi - _cap, _xphi + _cap], [_Ei_bulk, _Ei_bulk], color='purple', lw=1.5)
    _ax.text(_xphi + 0.12, (_EF + _Ei_bulk) / 2, r'$q\phi_B$', fontsize=14, color='purple', va='center')

    # Interface lines (from Ev,ox to Ec,ox)
    _ax.plot([-1.2, -1.2], [_Ev_ox_metal, _Ec_ox_metal], 'k-', lw=1.5)
    _ax.plot([0, 0], [_Ev_ox_semi, _Ec_ox_semi], 'k-', lw=1.5)

    # Region labels
    _ax.text(-1.5, _Ev_ox_semi - 0.5, 'Metal\n(Gate)', fontsize=14, ha='center', fontweight='bold')
    _ax.text(2.0, _Ev_ox_semi - 0.5, 'P-type Si', fontsize=14, ha='center', fontweight='bold')

    _ax.set_xlim(-2.1, 3.0)
    _ax.set_ylim(_Ev_ox_semi - 1.0, _Evac + 0.8)
    _ax.set_ylabel('Energy (eV)', fontsize=16)
    _ax.set_title(r'Flat-Band Energy Diagram (Ideal MOS, $V_G = 0$, $\Psi_M = \Psi_S$)',
                  fontsize=16, fontweight='bold')
    _ax.set_xticks([])
    _ax.tick_params(labelsize=14)
    for _sp in ['top', 'right', 'bottom']:
        _ax.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig)

    _caption = mo.md(rf"""
    **Flat-band diagram** for an ideal MOS capacitor with p-type Si body ($N_A$ = {_Na:.0e} cm$^{{-3}}$).

    - $\phi_B = E_{{Fi}} - E_{{Fs, bulk}} = (kT/q)\ln(N_A/n_i)$ = {_phi_B:.3f} eV. 
    - $\Psi_S = \chi + (E_c - E_{{Fs, bulk}})$ = {_Psi_S:.3f} eV
    - In the ideal MOS, $\Psi_M = \Psi_S$, so  bands are flat at $V_G = 0$.
    - The oxide has a large band gap ($E_g \approx 9$ eV).
    """)

    mo.vstack([mo.as_html(_fig), _caption])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Applying a Gate Voltage

    When a voltage $V_G$ is applied to the metal gate relative to the semiconductor body:

    - The metal Fermi level shifts relative to the semiconductor Fermi level: $E_{F,M} - E_{F,S} = -qV_G$
    - The voltage drops partly across the oxide ($V_{ox}$) and partly across the semiconductor surface ($\phi_s$):

    $$\boxed{V_G = \phi_s + V_{ox}}$$

    - $\phi_s$ is the **surface potential**: voltage dropped in the semiconductor
    - $V_{{ox}}$ is the voltage dropped in the oxide

    **Sign convention:** $\phi_s > 0$ means the bands bend **downward** at the surface (toward inversion for p-type).

    ### Regimes of Operation (P-type body)

    | Regime | Condition | Surface | Band Bending |
    |:-------|:----------|:--------|:-------------|
    | **Accumulation** | $V_G < 0$ | Holes accumulate | Bands bend **up** ($\phi_s < 0$) |
    | **Flat band** | $V_G = 0$ | No excess charge | No bending ($\phi_s = 0$) |
    | **Depletion** | $0 < V_G < V_T$ | Depletion region forms | Bands bend **down** ($0 < \phi_s < 2\phi_B$) |
    | **Threshold** | $V_G = V_T$ | $n_s = N_A$ | $\phi_s = 2\phi_B$ |
    | **Inversion** | $V_G > V_T$ | Electron inversion layer | $\phi_s \approx 2\phi_B$ (pinned) |
    """)
    return


@app.cell(hide_code=True)
def _(
    Ec_minus_Ei,
    Ei_minus_Ev,
    dEc_ox,
    dEv_ox,
    eps_0,
    eps_Si,
    kT_300,
    mo,
    ni_Si,
    np,
    plt,
    q,
):
    _Na = 1e16
    _phi_B = kT_300 * np.log(_Na / ni_Si)
    _EF = 0.0
    _Ei_bulk = _phi_B
    _Ec_bulk = _Ei_bulk + Ec_minus_Ei
    _Ev_bulk = _Ei_bulk - Ei_minus_Ev

    _phi_st = 2 * _phi_B
    _eps_s = eps_Si * eps_0
    _Wdep_max_cm = np.sqrt(2 * _eps_s * _phi_st / (q * _Na))
    _Cox = eps_0 * 3.9 / (5e-7)
    _Vt = _phi_st + q * _Na * _Wdep_max_cm / _Cox
    _gamma = np.sqrt(2 * q * _eps_s * _Na) / _Cox

    _regimes = [
        ("Accumulation ($V_G < 0$)", -1.5, '#d62728'),
        ("Depletion ($0 < V_G < V_T$)", _Vt * 0.5, '#2ca02c'),
        ("Threshold ($V_G = V_T$)", _Vt, '#ff7f0e'),
        ("Inversion ($V_G > V_T$)", _Vt + 1.5, '#1f77b4'),
    ]

    _fig, _axes = plt.subplots(1, 4, figsize=(18, 7.5), sharey=True)

    for _idx, (_label, _Vg, _color) in enumerate(_regimes):
        _ax = _axes[_idx]

        # Solve for surface potential from gate voltage
        if _Vg >= 0:
            _phi_s = 0.1
            for _ in range(100):
                if _phi_s < 1e-6:
                    _phi_s = 1e-6
                _f = _phi_s + _gamma * np.sqrt(_phi_s) - _Vg
                _df = 1 + _gamma / (2 * np.sqrt(_phi_s))
                _phi_s -= _f / _df
                if abs(_f) < 1e-8:
                    break
            _phi_s = min(_phi_s, _phi_st)
            _Wdep = np.sqrt(2 * _eps_s * _phi_s / (q * _Na))
        else:
            _phi_s = _Vg * 0.25
            _Wdep = 0

        _EFm = _EF - _Vg

        # Spatial coordinates
        _xM = [-1.8, -1.2]
        _xO = [-1.2, 0.0]
        _x_sc_end = 2.5

        # Semiconductor band profile
        if _phi_s > 0.001:
            _Wdep_plot = min(2.5, max(0.3, _Wdep * 1e4 * 2.5))
            _x_dep = np.linspace(0, _Wdep_plot, 200)
            _x_flat = np.linspace(_Wdep_plot, _x_sc_end, 50)
            _bending = _phi_s * (1 - _x_dep / _Wdep_plot) ** 2
            _Ec_dep = _Ec_bulk - _bending
            _Ev_dep = _Ev_bulk - _bending
            _Ei_dep = _Ei_bulk - _bending
            _x_sc = np.concatenate([_x_dep, _x_flat])
            _Ec_sc = np.concatenate([_Ec_dep, np.full_like(_x_flat, _Ec_bulk)])
            _Ev_sc = np.concatenate([_Ev_dep, np.full_like(_x_flat, _Ev_bulk)])
            _Ei_sc = np.concatenate([_Ei_dep, np.full_like(_x_flat, _Ei_bulk)])
        elif _phi_s < -0.001:
            _bend_len = 0.5
            _x_acc = np.linspace(0, _bend_len, 100)
            _x_flat = np.linspace(_bend_len, _x_sc_end, 50)
            _bending = _phi_s * (1 - _x_acc / _bend_len) ** 2
            _Ec_acc = _Ec_bulk - _bending
            _Ev_acc = _Ev_bulk - _bending
            _Ei_acc = _Ei_bulk - _bending
            _x_sc = np.concatenate([_x_acc, _x_flat])
            _Ec_sc = np.concatenate([_Ec_acc, np.full_like(_x_flat, _Ec_bulk)])
            _Ev_sc = np.concatenate([_Ev_acc, np.full_like(_x_flat, _Ev_bulk)])
            _Ei_sc = np.concatenate([_Ei_acc, np.full_like(_x_flat, _Ei_bulk)])
        else:
            _x_sc = np.array([0, _x_sc_end])
            _Ec_sc = np.array([_Ec_bulk, _Ec_bulk])
            _Ev_sc = np.array([_Ev_bulk, _Ev_bulk])
            _Ei_sc = np.array([_Ei_bulk, _Ei_bulk])

        # Metal
        _ax.fill_between(_xM, _EFm - 2.0, _EFm, color='#6baed6', alpha=0.3)
        _ax.plot(_xM, [_EFm, _EFm], 'g-', linewidth=2)
        _ax.text(_xM[0] + 0.05, _EFm + 0.1, '$E_{F,M}$', fontsize=11, color='green')

        # Oxide bands (Ec,ox and Ev,ox)
        _Ec_ox_s = _Ec_sc[0] + dEc_ox
        _Ev_ox_s = _Ev_sc[0] - dEv_ox
        _V_ox_tilt = _Vg - _phi_s
        _Ec_ox_m = _Ec_ox_s - _V_ox_tilt
        _Ev_ox_m = _Ev_ox_s - _V_ox_tilt

        _ax.fill_between(_xO, [_Ev_ox_m, _Ev_ox_s], [_Ec_ox_m, _Ec_ox_s],
                         color='#ffffcc', alpha=0.4, edgecolor='none')
        _ax.plot(_xO, [_Ec_ox_m, _Ec_ox_s], 'k-', linewidth=2)
        _ax.plot(_xO, [_Ev_ox_m, _Ev_ox_s], 'k-', linewidth=2)

        # Semiconductor bands
        _ax.plot(_x_sc, _Ec_sc, 'b-', linewidth=2.5)
        _ax.plot(_x_sc, _Ev_sc, 'r-', linewidth=2.5)
        _ax.plot(_x_sc, _Ei_sc, 'k--', linewidth=1, alpha=0.5)
        _ax.plot([0, _x_sc_end], [_EF, _EF], 'g-', linewidth=2)

        # Interface lines (from Ev,ox to Ec,ox)
        _ax.plot([-1.2, -1.2], [_Ev_ox_m, _Ec_ox_m], 'k-', lw=1.5)
        _ax.plot([0, 0], [_Ev_ox_s, _Ec_ox_s], 'k-', lw=1.5)

        # Labels at right
        if _idx == 3:
            _ax.text(2.6, _Ec_bulk, '$E_c$', fontsize=12, va='center', color='blue')
            _ax.text(2.6, _Ev_bulk, '$E_v$', fontsize=12, va='center', color='red')
            _ax.text(2.6, _Ei_bulk, '$E_{Fi}$', fontsize=12, va='center', alpha=0.6)
            _ax.text(2.6, _EF, '$E_{Fs}$', fontsize=12, va='center', color='green')

        # phi_s annotation
        if abs(_phi_s) > 0.05:
            _cap = 0.06
            _xann = 0.15
            _ax.plot([_xann, _xann], [_Ec_sc[0], _Ec_bulk], color='darkgreen', lw=1.5)
            _ax.plot([_xann - _cap, _xann + _cap], [_Ec_sc[0], _Ec_sc[0]], color='darkgreen', lw=1.5)
            _ax.plot([_xann - _cap, _xann + _cap], [_Ec_bulk, _Ec_bulk], color='darkgreen', lw=1.5)
            _ax.text(_xann + 0.15, (_Ec_sc[0] + _Ec_bulk) / 2, r'$q\phi_s$',
                     fontsize=12, color='darkgreen', va='center')

        # phi_B annotation (between E_Fi,bulk and E_F in bulk)
        _cap = 0.06
        _xpB = 2.0
        _ax.plot([_xpB, _xpB], [_EF, _Ei_bulk], color='purple', lw=1.2)
        _ax.plot([_xpB - _cap, _xpB + _cap], [_EF, _EF], color='purple', lw=1.2)
        _ax.plot([_xpB - _cap, _xpB + _cap], [_Ei_bulk, _Ei_bulk], color='purple', lw=1.2)
        _ax.text(_xpB + 0.12, (_EF + _Ei_bulk) / 2, r'$q\phi_B$',
                 fontsize=11, color='purple', va='center')

        # V_ox annotation at oxide-metal interface
        if abs(_V_ox_tilt) > 0.01:
            _cap = 0.06
            _xvox = -1.05
            _bracket_top = _Ec_ox_s
            _bracket_bot = _Ec_ox_m
            _min_visual = 0.4
            if abs(_bracket_top - _bracket_bot) < _min_visual:
                _mid = (_bracket_top + _bracket_bot) / 2
                _sign = 1 if _bracket_top >= _bracket_bot else -1
                _bracket_top = _mid + _sign * _min_visual / 2
                _bracket_bot = _mid - _sign * _min_visual / 2
            _ax.plot([_xvox, _xvox], [_bracket_bot, _bracket_top], color='darkorange', lw=1.5)
            _ax.plot([_xvox - _cap, _xvox + _cap], [_bracket_bot, _bracket_bot], color='darkorange', lw=1.5)
            _ax.plot([_xvox - _cap, _xvox + _cap], [_bracket_top, _bracket_top], color='darkorange', lw=1.5)
            _ax.text(_xvox + 0.12, (_bracket_bot + _bracket_top) / 2, r'$qV_{ox}$',
                     fontsize=11, color='darkorange', va='center', ha='left')

        # Title with regime
        _ax.set_title(_label, fontsize=13, fontweight='bold', color=_color)
        _ax.set_xticks([])
        _ax.tick_params(labelsize=12)
        _ax.set_xlim(-2.0, 3.0)
        if _idx == 0:
            _ax.set_ylabel('Energy (eV)', fontsize=14)
        for _sp in ['top', 'right', 'bottom']:
            _ax.spines[_sp].set_visible(False)

    _axes[0].set_ylim(_Ev_bulk - dEv_ox - 1.0, _Ec_bulk + dEc_ox + 1.5)
    _fig.suptitle('MOS Energy Bands in Each Regime (P-type body)', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.close(_fig)

    _caption = mo.md(r"""
    **Four regimes of the MOS capacitor** (p-type body, ideal $\Psi_M = \Psi_S$).

    - **Accumulation:** $V_G < 0$, bands bend up, holes pile up at the surface.
    - **Depletion:** $V_G > 0$, bands bend down, depletion region forms.
    - **Threshold:** $\phi_s = 2\phi_B$, surface electron concentration equals bulk hole concentration.
    - **Inversion:** $V_G > V_T$, strong electron inversion layer at surface; $\phi_s$ pinned at $\approx 2\phi_B$.

    The oxide conduction and valence band edges tilt under applied voltage due to the electric field across the oxide.
    """)

    mo.vstack([mo.as_html(_fig), _caption])
    return


@app.cell(hide_code=True)
def _(Ec_minus_Ei, Ei_minus_Ev, kT_300, mo, ni_Si, np):
    _Na = 1e16
    _phi_B = kT_300 * np.log(_Na / ni_Si)

    mo.md(rf"""
    ## Threshold Condition: Why $\phi_s = 2\phi_B$

    **Definition of threshold:** the gate voltage $V_T$ at which the surface electron
    concentration equals the bulk hole concentration, i.e., $n_s = N_A$.

    ### Step 1: Define $\phi_B$

    The **bulk potential** $\phi_B$ measures the separation between the intrinsic level
    and the Fermi level in the bulk:

    $$q\phi_B \equiv E_{{Fi,\text{{bulk}}}} - E_{{F_s}}$$

    From the carrier concentration $p = n_i \exp\!\bigl((E_{{Fi}} - E_F)/kT\bigr) = N_A$:

    $$\boxed{{q\phi_B = kT \ln\!\left(\frac{{N_A}}{{n_i}}\right)}}$$

    For ${{N_A = }}$ {_Na:.0e} cm$^{{-3}}$ at 300 K: $\phi_B$ = {_phi_B:.4f} V.

    ### Step 2: Surface electron concentration at threshold

    At the surface ($x = 0$), the electron concentration is:

    $$n_s = N_c \, e^{{-(E_{{c,\text{{surf}}}} - E_{{F_s}})/kT}}$$

    where ${{E_{{c,\text{{surf}}}}}}$ is the conduction band energy at the surface.

    In the bulk, the hole concentration satisfies:

    $$N_A = N_v \, e^{{-(E_{{F_s}} - E_{{v,\text{{bulk}}}})/kT}}$$

    At **threshold**, we require $n_s = N_A$. Setting these equal:

    $$N_c \, e^{{-(E_{{c,\text{{surf}}}} - E_{{F_s}})/kT}} = N_v \, e^{{-(E_{{F_s}} - E_{{v,\text{{bulk}}}})/kT}}$$

    Taking the logarithm and rearranging:

    $$(E_{{c,\text{{surf}}}} - E_{{F_s}}) - (E_{{F_s}} - E_{{v,\text{{bulk}}}}) = -kT \ln\!\left(\frac{{N_v}}{{N_c}}\right)$$

    ### Step 3: Connecting to band bending

    Recall the intrinsic level satisfies $E_{{Fi}} - E_v = \frac{{E_g}}{{2}} + \frac{{kT}}{{2}}\ln\!\left(\frac{{N_v}}{{N_c}}\right)$, so:

    $$E_c - E_{{Fi}} = \frac{{E_g}}{{2}} - \frac{{kT}}{{2}}\ln\!\left(\frac{{N_v}}{{N_c}}\right) = E_g - (E_{{Fi}} - E_v)$$

    Substituting $-kT\ln(N_v/N_c) = (E_c - E_{{Fi}}) - (E_{{Fi}} - E_v)$ into the threshold equation:

    $$(E_{{c,\text{{surf}}}} - E_{{F_s}}) - (E_{{F_s}} - E_{{v,\text{{bulk}}}}) = (E_{{c,\text{{bulk}}}} - E_{{Fi,\text{{bulk}}}}) - (E_{{Fi,\text{{bulk}}}} - E_{{v,\text{{bulk}}}})$$

    Since $E_{{c,\text{{bulk}}}} - E_{{v,\text{{bulk}}}} = E_g$ and $(E_{{c,\text{{surf}}}} - E_{{F_s}}) = (E_{{c,\text{{bulk}}}} - E_{{F_s}}) - q\phi_s$, this simplifies to:

    $$\boxed{{q\phi_s = 2(E_{{Fi,\text{{bulk}}}} - E_{{F_s}}) = 2 q\phi_B}}$$

    Therefore the **threshold surface potential** is:

    $$\boxed{{\phi_{{s,T}} = 2\phi_B = \frac{{2kT}}{{q}}\ln\!\left(\frac{{N_A}}{{n_i}}\right)}}$$

    For ${{N_A = }}$ {_Na:.0e} cm$^{{-3}}$: $\phi_{{s,T}} = 2\phi_B$ = {2*_phi_B:.4f} V.

    ### Physical meaning

    At flat band, the surface is p-type with $p_s = N_A$ and $E_{{Fi}}$ is above $E_F$ by $q\phi_B$.

    - After bending the bands down by $q\phi_B$, we reach $E_{{Fi}} = E_F$ at the surface — the surface is **intrinsic** ($n_s = p_s = n_i$).
    - After bending by another $q\phi_B$ (total $2q\phi_B$), the surface is now as **n-type** as the bulk was p-type: $n_s = N_A$.

    This is the **threshold condition**: the surface has been "inverted" from p-type to n-type.

    <!--
    **Note:** $E_c - E_{{Fi}}$ = {Ec_minus_Ei:.4f} eV and $E_{{Fi}} - E_v$ = {Ei_minus_Ev:.4f} eV
    ($E_{{Fi}}$ is {1000*(Ec_minus_Ei - Ei_minus_Ev)/2:.1f} meV below mid-gap because $N_c > N_v$).
    The derivation above uses the exact density-of-states expressions, not the mid-gap approximation.
    -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    Vg_slider = mo.ui.slider(
        start=-3.0, stop=3.0, step=0.05, value=1.0,
        label=r"$V_G$ (V)", full_width=False
    )
    Na_slider = mo.ui.slider(
        start=15, stop=18, step=0.1, value=16.0,
        label=r"$\log_{10}(N_A)$ (cm$^{-3}$)", full_width=False
    )
    tox_slider = mo.ui.slider(
        start=2.0, stop=20.0, step=0.5, value=5.0,
        label=r"$t_{ox}$ (nm)", full_width=False
    )
    return Na_slider, Vg_slider, tox_slider


@app.cell(hide_code=True)
def _(
    Ec_minus_Ei,
    Ei_minus_Ev,
    Na_slider,
    Vg_slider,
    dEc_ox,
    dEv_ox,
    eps_0,
    eps_Si,
    eps_ox,
    kT_300,
    mo,
    ni_Si,
    np,
    plt,
    q,
    tox_slider,
):
    _Vg = Vg_slider.value
    _Na = 10 ** Na_slider.value
    _tox_nm = tox_slider.value
    _tox_cm = _tox_nm * 1e-7

    _phi_B = kT_300 * np.log(_Na / ni_Si)
    _Cox = eps_ox * eps_0 / _tox_cm

    _phi_st = 2 * _phi_B
    _Wdep_max = np.sqrt(2 * eps_Si * eps_0 * _phi_st / (q * _Na))
    _Qd_max = q * _Na * _Wdep_max
    _Vt = _phi_st + _Qd_max / _Cox

    _eps_s = eps_Si * eps_0

    def _solve_phi_s(Vg, Na, Cox, phi_B):
        if abs(Vg) < 1e-6:
            return 0.0
        phi_s_max = 2 * phi_B + 6 * kT_300
        gamma = np.sqrt(2 * q * _eps_s * Na) / Cox

        if Vg > 0:
            phi_s = 0.1
            for _ in range(100):
                if phi_s <= 0:
                    phi_s = 0.01
                if phi_s > phi_s_max:
                    phi_s = phi_s_max
                    break
                f = phi_s + gamma * np.sqrt(phi_s) - Vg
                df = 1 + gamma / (2 * np.sqrt(phi_s))
                delta = f / df
                phi_s -= delta
                if abs(delta) < 1e-8:
                    break
            return min(phi_s, phi_s_max)
        else:
            phi_s = -0.01
            for _ in range(100):
                if phi_s >= 0:
                    phi_s = -0.001
                u_s = min(-phi_s / kT_300, 500)
                arg = max(np.exp(u_s) - u_s - 1, 0)
                Q_s = np.sqrt(2 * q * _eps_s * Na * kT_300 * arg)
                f = phi_s - Q_s / Cox - Vg
                if Q_s > 1e-20:
                    dQ_s = -q * _eps_s * Na * (np.exp(u_s) - 1) / Q_s
                else:
                    dQ_s = 0
                df = 1 - dQ_s / Cox
                if abs(df) < 1e-12:
                    break
                delta = f / df
                phi_s -= delta
                if abs(delta) < 1e-8:
                    break
            return max(phi_s, -1.0)

    _phi_s = _solve_phi_s(_Vg, _Na, _Cox, _phi_B)

    if _phi_s < -0.01:
        _regime = "Accumulation"
        _regime_color = '#d62728'
    elif abs(_phi_s) <= 0.01:
        _regime = "Flat Band"
        _regime_color = '#7f7f7f'
    elif _phi_s < 2 * _phi_B - 0.05:
        _regime = "Depletion"
        _regime_color = '#2ca02c'
    elif abs(_phi_s - 2 * _phi_B) < 0.15:
        _regime = "Threshold"
        _regime_color = '#ff7f0e'
    else:
        _regime = "Strong Inversion"
        _regime_color = '#1f77b4'

    if _phi_s > 0:
        _Wdep = np.sqrt(2 * eps_Si * eps_0 * min(_phi_s, _phi_st) / (q * _Na))
        _Wdep_um = _Wdep * 1e4
    else:
        _Wdep = 0.0
        _Wdep_um = 0.0

    # === Energy band diagram ===
    _EF = 0.0
    _Ei_bulk = _phi_B
    _Ec_bulk = _Ei_bulk + Ec_minus_Ei
    _Ev_bulk = _Ei_bulk - Ei_minus_Ev
    _EFm = _EF - _Vg  # EFm - EFs = -qVg (in eV units)

    _fig, _ax = plt.subplots(figsize=(8.8, 4.8))

    _xM = [-1.8, -1.2]
    _xO = [-1.2, 0.0]
    _x_sc_end = 3.0

    # Semiconductor band profile
    if _phi_s > 0.001:
        _Wdep_plot = min(2.5, max(0.4, _Wdep * 1e4 * 2.5))
        _x_dep = np.linspace(0, _Wdep_plot, 300)
        _x_flat = np.linspace(_Wdep_plot, _x_sc_end, 50)
        _bending = _phi_s * (1 - _x_dep / _Wdep_plot) ** 2
        _Ec_dep = _Ec_bulk - _bending
        _Ev_dep = _Ev_bulk - _bending
        _Ei_dep = _Ei_bulk - _bending
        _x_sc = np.concatenate([_x_dep, _x_flat])
        _Ec_sc = np.concatenate([_Ec_dep, np.full_like(_x_flat, _Ec_bulk)])
        _Ev_sc = np.concatenate([_Ev_dep, np.full_like(_x_flat, _Ev_bulk)])
        _Ei_sc = np.concatenate([_Ei_dep, np.full_like(_x_flat, _Ei_bulk)])
    elif _phi_s < -0.001:
        _bend_len = 0.5
        _x_acc = np.linspace(0, _bend_len, 150)
        _x_flat = np.linspace(_bend_len, _x_sc_end, 50)
        _bending = _phi_s * (1 - _x_acc / _bend_len) ** 2
        _x_sc = np.concatenate([_x_acc, _x_flat])
        _Ec_sc = np.concatenate([_Ec_bulk - _bending, np.full_like(_x_flat, _Ec_bulk)])
        _Ev_sc = np.concatenate([_Ev_bulk - _bending, np.full_like(_x_flat, _Ev_bulk)])
        _Ei_sc = np.concatenate([_Ei_bulk - _bending, np.full_like(_x_flat, _Ei_bulk)])
    else:
        _x_sc = np.array([0, _x_sc_end])
        _Ec_sc = np.array([_Ec_bulk, _Ec_bulk])
        _Ev_sc = np.array([_Ev_bulk, _Ev_bulk])
        _Ei_sc = np.array([_Ei_bulk, _Ei_bulk])

    # Metal
    _ax.fill_between(_xM, _EFm - 3.0, _EFm, color='#6baed6', alpha=0.3)
    _ax.plot(_xM, [_EFm, _EFm], 'g-', linewidth=2)
    _ax.text(_xM[0] + 0.05, _EFm + 0.1, '$E_{F,M}$', fontsize=14, color='green')

    # Oxide bands (Ec,ox and Ev,ox)
    _Ec_ox_s = _Ec_sc[0] + dEc_ox
    _Ev_ox_s = _Ev_sc[0] - dEv_ox
    _V_ox = _Vg - _phi_s
    _Ec_ox_m = _Ec_ox_s - _V_ox
    _Ev_ox_m = _Ev_ox_s - _V_ox

    _ax.fill_between(_xO, [_Ev_ox_m, _Ev_ox_s], [_Ec_ox_m, _Ec_ox_s],
                     color='#ffffcc', alpha=0.4, edgecolor='none')
    _ax.plot(_xO, [_Ec_ox_m, _Ec_ox_s], 'k-', linewidth=2)
    _ax.plot(_xO, [_Ev_ox_m, _Ev_ox_s], 'k-', linewidth=2)

    # Semiconductor bands
    _ax.plot(_x_sc, _Ec_sc, 'b-', linewidth=2.5, label='$E_c$')
    _ax.plot(_x_sc, _Ev_sc, 'r-', linewidth=2.5, label='$E_v$')
    _ax.plot(_x_sc, _Ei_sc, 'k--', linewidth=1, alpha=0.5, label='$E_{Fi}$')
    _ax.plot([0, _x_sc_end], [_EF, _EF], 'g-', linewidth=2, label='$E_{Fs}$')

    # Labels
    _ax.text(_x_sc_end + 0.1, _Ec_bulk, '$E_c$', fontsize=14, va='center', color='blue')
    _ax.text(_x_sc_end + 0.1, _Ev_bulk, '$E_v$', fontsize=14, va='center', color='red')
    _ax.text(_x_sc_end + 0.1, _Ei_bulk, '$E_{Fi}$', fontsize=14, va='center', alpha=0.6)
    _ax.text(_x_sc_end + 0.1, _EF, '$E_{Fs}$', fontsize=14, va='center', color='green')

    # Interface lines (from Ev,ox to Ec,ox)
    _ax.plot([-1.2, -1.2], [_Ev_ox_m, _Ec_ox_m], 'k-', lw=1.5)
    _ax.plot([0, 0], [_Ev_ox_s, _Ec_ox_s], 'k-', lw=1.5)

    # phi_s annotation
    if abs(_phi_s) > 0.05:
        _cap = 0.06
        _xann = 0.2
        _ax.plot([_xann, _xann], [_Ec_sc[0], _Ec_bulk], color='darkgreen', lw=1.5)
        _ax.plot([_xann - _cap, _xann + _cap], [_Ec_sc[0], _Ec_sc[0]], color='darkgreen', lw=1.5)
        _ax.plot([_xann - _cap, _xann + _cap], [_Ec_bulk, _Ec_bulk], color='darkgreen', lw=1.5)
        _ax.text(_xann + 0.15, (_Ec_sc[0] + _Ec_bulk) / 2, r'$q\phi_s$',
                 fontsize=14, color='darkgreen', va='center')

    # phi_B annotation (between E_Fi,bulk and E_F in bulk)
    _cap = 0.06
    _xpB = _x_sc_end - 0.6
    _ax.plot([_xpB, _xpB], [_EF, _Ei_bulk], color='purple', lw=1.5)
    _ax.plot([_xpB - _cap, _xpB + _cap], [_EF, _EF], color='purple', lw=1.5)
    _ax.plot([_xpB - _cap, _xpB + _cap], [_Ei_bulk, _Ei_bulk], color='purple', lw=1.5)
    _ax.text(_xpB + 0.12, (_EF + _Ei_bulk) / 2, r'$q\phi_B$',
             fontsize=14, color='purple', va='center')

    # V_ox annotation at oxide-metal interface
    if abs(_V_ox) > 0.05:
        _cap = 0.06
        _xvox = -1.15
        _ax.plot([_xvox, _xvox], [_Ec_ox_m, _Ec_ox_s], color='darkorange', lw=1.5)
        _ax.plot([_xvox - _cap, _xvox + _cap], [_Ec_ox_m, _Ec_ox_m], color='darkorange', lw=1.5)
        _ax.plot([_xvox - _cap, _xvox + _cap], [_Ec_ox_s, _Ec_ox_s], color='darkorange', lw=1.5)
        _ax.text(_xvox - 0.12, (_Ec_ox_m + _Ec_ox_s) / 2, r'$qV_{ox}$',
                 fontsize=14, color='darkorange', va='center', ha='right')

    # qVG annotation
    if abs(_Vg) > 0.1:
        _xvg = -1.35
        _ax.plot([_xvg, _xvg], [_EFm, _EF], color='darkred', lw=1.5)
        _cap = 0.06
        _ax.plot([_xvg - _cap, _xvg + _cap], [_EFm, _EFm], color='darkred', lw=1.5)
        _ax.plot([_xvg - _cap, _xvg + _cap], [_EF, _EF], color='darkred', lw=1.5)
        _ax.text(_xvg - 0.15, (_EFm + _EF) / 2, r'$qV_G$',
                 fontsize=14, color='darkred', va='center', ha='right')

    # Depletion width marker
    if _phi_s > 0.1 and _Wdep > 0:
        _Wdep_plot_val = min(2.5, max(0.4, _Wdep * 1e4 * 2.5))
        _ax.axvline(_Wdep_plot_val, color='gray', ls=':', lw=1, alpha=0.5)
        _ydep = _Ev_bulk - 0.5
        _ax.plot([0, _Wdep_plot_val], [_ydep, _ydep], color='gray', lw=1.5)
        _ax.plot([0, 0], [_ydep - 0.06, _ydep + 0.06], color='gray', lw=1.5)
        _ax.plot([_Wdep_plot_val, _Wdep_plot_val], [_ydep - 0.06, _ydep + 0.06], color='gray', lw=1.5)
        _ax.text(_Wdep_plot_val / 2, _ydep - 0.2, '$W_{dep}$', fontsize=14, ha='center', color='gray')

    # Regime label
    _ax.text((_x_sc_end + 0) / 2, _Ec_bulk + dEc_ox + 0.05, _regime,
             fontsize=18, ha='center', fontweight='bold', color=_regime_color,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=_regime_color, alpha=0.9))

    # Oxide band labels
    _ax.text(-0.6, max(_Ec_ox_m, _Ec_ox_s) + 0.15, '$E_{c,ox}$', fontsize=13, ha='center')
    _ax.text(-0.6, min(_Ev_ox_m, _Ev_ox_s) - 0.3, '$E_{v,ox}$', fontsize=13, ha='center')

    # Region labels
    _ax.text(-1.5, min(_Ev_ox_m, _Ev_ox_s) - 0.8, 'Metal\n(Gate)', fontsize=14, ha='center', fontweight='bold')
    _ax.text(0.7 * _x_sc_end / 2 + 0.5, min(_Ev_ox_s, _Ev_ox_m) - 0.8, 'P-type Si', fontsize=14, ha='center', fontweight='bold')

    _y_lo = min(_Ev_ox_s, _Ev_ox_m, _EFm) - 1.5
    _y_hi = max(_Ec_bulk + dEc_ox + 0.8, _Ec_ox_m + 0.5, _EFm + 1.0)
    _ax.set_xlim(-2.1, _x_sc_end + 0.8)
    _ax.set_ylim(_y_lo, _y_hi)
    _ax.set_ylabel('Energy (eV)', fontsize=16)
    _ax.set_title('MOS Energy Band Diagram', fontsize=16, fontweight='bold')
    _ax.set_xticks([])
    _ax.tick_params(labelsize=14)
    for _sp in ['top', 'right', 'bottom']:
        _ax.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig)

    _info = mo.md(rf"""
    | Quantity | Value |
    |:---------|:------|
    | **Regime** | {_regime} |
    | $V_G$ | {_Vg:.2f} V |
    | $\phi_B$ | {_phi_B:.4f} eV |
    | $\phi_s$ (surface potential) | {_phi_s:.4f} eV |
    | $V_{{ox}} = V_G - \phi_s$ | {_V_ox:.4f} V |
    | $2\phi_B$ (threshold condition) | {2*_phi_B:.4f} eV |
    | $W_{{dep}}$ | {_Wdep:.2e} cm = {_Wdep_um:.3f} $\mu$m |
    | $W_{{dep,max}}$ | {_Wdep_max:.2e} cm |
    | $C_{{ox}}$ | {_Cox:.2e} F/cm$^2$ |
    | $V_T$ (threshold voltage) | {_Vt:.3f} V |
    | $N_A$ | {_Na:.2e} cm$^{{-3}}$ |
    | $t_{{ox}}$ | {_tox_nm:.1f} nm |
    """)

    _controls = mo.vstack([Vg_slider, Na_slider, tox_slider])

    mo.vstack([
        mo.md("### Interactive Energy Band Diagram (P-type)"),
        _controls,
        mo.as_html(_fig),
        _info,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Electrostatics of the MOS Capacitor

    We now need to find $V_{ox}$ to relate $V_G$ to the surface potential $\phi_s$, since $V_G = \phi_s + V_{ox}$.

    We work in the **depletion approximation**: mobile carriers are swept away, leaving only the fixed
    ionized acceptors $-qN_A$ in a depletion region of width $W_{dep}$.

    ---

    ### Step 1: Charge density in depletion

    In the semiconductor, we define $x = 0$ at the oxide-semiconductor interface and $x$ increases into the bulk:

    $$\rho(x) = \begin{cases} -q N_A & 0 \le x \le W_{dep} \\ 0 & x > W_{dep} \end{cases}$$

    ---

    ### Step 2: Electric field (Gauss's law)

    Poisson's equation in the semiconductor gives:

    $$\frac{d\mathcal{E}}{dx} = \frac{\rho(x)}{\varepsilon_s}$$

    With the boundary condition $\mathcal{E}(W_{dep}) = 0$, integrating from $x$ to $W_{dep}$:

    $$\boxed{\mathcal{E}(x) = \frac{q N_A}{\varepsilon_s}(W_{dep} - x), \quad 0 \le x \le W_{dep}}$$

    The maximum electric field magnitude is at the surface ($x = 0$):

    $$\mathcal{E}_{max} = \mathcal{E}(0) = \frac{q N_A W_{dep}}{\varepsilon_s}$$

    ---

    ### Step 3: Electrostatic potential

    The potential is related to the electric field by $-\dfrac{dV}{dx} = \mathcal{E}$. With the boundary condition $V(W_{dep}) = 0$,
    integrating from $x$ to $W_{dep}$:

    $$\boxed{V(x) = \frac{q N_A}{2\varepsilon_s}(W_{dep} - x)^2, \quad 0 \le x \le W_{dep}}$$

    At the surface ($x = 0$), the surface potential equals:

    $$V(0) = \phi_s = \frac{q N_A}{2\varepsilon_s} W_{dep}^2$$

    Solving for the depletion width:

    $$\boxed{W_{dep} = \sqrt{\frac{2\varepsilon_s \phi_s}{q N_A}}}$$

    ---

    ### Step 4: Oxide electric field and voltage

    The oxide has uniform electric field (no free charge in the oxide):

    $$\mathcal{E}_{ox} = \frac{V_{ox}}{t_{ox}} \tag{1}$$

    At $x = 0$, the **normal component of the displacement field** $\mathbf{D}$ must be continuous across the
    oxide–semiconductor interface:

    $$D_{ox} = D_s \quad\Rightarrow\quad \varepsilon_{ox}\,\mathcal{E}_{ox} = \varepsilon_s\,\mathcal{E}(0) = q N_A W_{dep} \tag{2}$$

    From (1) and (2):

    $$V_{ox} = \mathcal{E}_{ox}\, t_{ox} = \frac{q N_A}{\varepsilon_{ox}} t_{ox}\, W_{dep} = \frac{q N_A W_{dep}}{C_{ox}}$$

    where the **oxide capacitance per unit area** is:

    $$\boxed{C_{ox} \equiv \frac{\varepsilon_{ox}}{t_{ox}}} \quad [\text{F/cm}^2]$$

    and the **depletion charge per unit area** is:

    $$Q_{dep} = q N_A W_{dep}$$

    ---

    ### Step 5: Gate voltage equation

    Substituting $V_{ox} = Q_{dep}/C_{ox}$ into $V_G = \phi_s + V_{ox}$:

    $$\boxed{V_G = \phi_s + \frac{q N_A W_{dep}}{C_{ox}} = \frac{q N_A W_{dep}^2}{2\varepsilon_s} + \frac{q N_A W_{dep}}{C_{ox}}}$$

    Or equivalently, expressing everything in terms of $\phi_s$:

    $$\boxed{V_G = \phi_s + \frac{\sqrt{2 q \varepsilon_s N_A \phi_s}}{C_{ox}}}$$

    Given $V_G$, we can solve for $W_{dep}$ (and hence $\phi_s$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    Vg_es_slider = mo.ui.slider(
        start=-3.0, stop=3.0, step=0.05, value=1.0,
        label=r"$V_G$ (V)", full_width=False
    )
    Na_es_slider = mo.ui.slider(
        start=15, stop=18, step=0.1, value=16.0,
        label=r"$\log_{10}(N_A)$ (cm$^{-3}$)", full_width=False
    )
    tox_es_slider = mo.ui.slider(
        start=2.0, stop=20.0, step=0.5, value=5.0,
        label=r"$t_{ox}$ (nm)", full_width=False
    )
    return Na_es_slider, Vg_es_slider, tox_es_slider


@app.cell(hide_code=True)
def _(
    Na_es_slider,
    Vg_es_slider,
    eps_0,
    eps_Si,
    eps_ox,
    kT_300,
    mo,
    ni_Si,
    np,
    plt,
    q,
    tox_es_slider,
):
    _Vg = Vg_es_slider.value
    _Na = 10 ** Na_es_slider.value
    _tox_cm = tox_es_slider.value * 1e-7
    _eps_s = eps_Si * eps_0
    _Cox = eps_ox * eps_0 / _tox_cm
    _phi_B = kT_300 * np.log(_Na / ni_Si)
    _phi_st = 2 * _phi_B

    _Wdep_max = np.sqrt(2 * _eps_s * _phi_st / (q * _Na))
    _Qd_max = q * _Na * _Wdep_max
    _Vt = _phi_st + _Qd_max / _Cox

    def _solve_phi_s_p(Vg):
        if abs(Vg) < 1e-6:
            return 0.0
        gamma = np.sqrt(2 * q * _eps_s * _Na) / _Cox
        if Vg > 0:
            phi_s_max = _phi_st + 6 * kT_300
            phi_s = 0.1
            for _ in range(100):
                if phi_s <= 0:
                    phi_s = 0.01
                if phi_s > phi_s_max:
                    phi_s = phi_s_max
                    break
                f = phi_s + gamma * np.sqrt(phi_s) - Vg
                df = 1 + gamma / (2 * np.sqrt(phi_s))
                delta = f / df
                phi_s -= delta
                if abs(delta) < 1e-8:
                    break
            return min(phi_s, phi_s_max)
        else:
            phi_s = -0.01
            for _ in range(100):
                if phi_s >= 0:
                    phi_s = -0.001
                u_s = min(-phi_s / kT_300, 500)
                arg = max(np.exp(u_s) - u_s - 1, 0)
                Q_s = np.sqrt(2 * q * _eps_s * _Na * kT_300 * arg)
                f = phi_s - Q_s / _Cox - Vg
                if Q_s > 1e-20:
                    dQ_s = -q * _eps_s * _Na * (np.exp(u_s) - 1) / Q_s
                else:
                    dQ_s = 0
                df = 1 - dQ_s / _Cox
                if abs(df) < 1e-12:
                    break
                delta = f / df
                phi_s -= delta
                if abs(delta) < 1e-8:
                    break
            return max(phi_s, -1.0)

    _phi_s = _solve_phi_s_p(_Vg)

    if _phi_s < -0.01:
        _regime = "Accumulation"
        _regime_color = '#d62728'
    elif abs(_phi_s) <= 0.01:
        _regime = "Flat Band"
        _regime_color = '#7f7f7f'
    elif _phi_s < _phi_st - 0.05:
        _regime = "Depletion"
        _regime_color = '#2ca02c'
    elif abs(_phi_s - _phi_st) < 0.15:
        _regime = "Threshold"
        _regime_color = '#ff7f0e'
    else:
        _regime = "Strong Inversion"
        _regime_color = '#1f77b4'

    if _phi_s > 0.001:
        _Wdep = np.sqrt(2 * _eps_s * min(_phi_s, _phi_st) / (q * _Na))
        _Wdep_um = _Wdep * 1e4
    else:
        _Wdep = 0.0
        _Wdep_um = 0.0

    # ──── Compute electrostatics ────
    if _phi_s > 0.01:
        _x_es = np.linspace(0, _Wdep, 500)
        _x_es_um = _x_es * 1e4
        _x_ext = np.linspace(_Wdep, _Wdep * 1.5, 50)
        _x_ext_um = _x_ext * 1e4

        _rho = np.full_like(_x_es, -q * _Na)
        _E_field = q * _Na / _eps_s * (_Wdep - _x_es)
        _E_max = q * _Na * _Wdep / _eps_s
        _V_es = _phi_s * (1 - _x_es / _Wdep) ** 2

    elif _phi_s < -0.01:
        _L_D = np.sqrt(_eps_s * kT_300 / (q * _Na))
        _N_pts = 600
        _V_cutoff = -kT_300 * 0.1
        _V_arr = np.linspace(_phi_s, _V_cutoff, _N_pts)
        _u_arr = -_V_arr / kT_300
        _E_sq = 2 * q * _Na * kT_300 / _eps_s * (np.exp(_u_arr) - _u_arr - 1)
        _E_abs = np.sqrt(np.maximum(_E_sq, 1e-10))
        _E_arr = -_E_abs

        _dV = np.diff(_V_arr)
        _E_abs_mid = 0.5 * (_E_abs[:-1] + _E_abs[1:])
        _E_abs_mid = np.maximum(_E_abs_mid, 1.0)
        _dx_arr = _dV / _E_abs_mid
        _x_arr = np.concatenate([[0], np.cumsum(_dx_arr)])

        _x_es = _x_arr
        _x_es_um = _x_es * 1e4
        _x_end_cm = _x_arr[-1]
        _x_ext = np.linspace(_x_end_cm, _x_end_cm + 4 * _L_D, 50)
        _x_ext_um = _x_ext * 1e4

        _rho = q * _Na * (np.exp(_u_arr) - 1)
        _E_field = _E_arr
        _E_max = _E_arr[0]
        _V_es = _V_arr

    else:
        _x_es = np.array([0])
        _x_es_um = np.array([0])
        _x_ext = np.linspace(0, 0.5e-4, 50)
        _x_ext_um = _x_ext * 1e4
        _rho = np.array([0])
        _E_field = np.array([0])
        _E_max = 0
        _V_es = np.array([0])

    _rho_ext = np.zeros_like(_x_ext)
    _E_ext = np.zeros_like(_x_ext)
    _V_ext = np.zeros_like(_x_ext)

    _fig, (_ax_rho, _ax_E, _ax_V) = plt.subplots(1, 3, figsize=(16, 4.5))

    _ax_rho.plot(_x_es_um, _rho / q * 1e-15, 'b-', linewidth=2.5)
    _ax_rho.plot(_x_ext_um, _rho_ext, 'b-', linewidth=2.5)
    _ax_rho.fill_between(_x_es_um, 0, _rho / q * 1e-15, alpha=0.2, color='blue')
    _ax_rho.axhline(0, color='k', linewidth=0.5)
    _ax_rho.set_ylabel(r'$\rho / q$ ($\times 10^{15}$ cm$^{-3}$)', fontsize=16)
    _ax_rho.set_xlabel(r'$x$ ($\mu$m)', fontsize=16)
    _ax_rho.set_title(r'Charge density $\rho(x)$', fontsize=16, fontweight='bold')
    _ax_rho.tick_params(labelsize=14)
    _ax_rho.grid(True, alpha=0.3)

    _Qm = _Cox * (_Vg - _phi_s)
    _Qdep_total = -q * _Na * _Wdep if _phi_s > 0.01 else 0.0
    _Qinv_p = 0.0
    if _phi_s >= _phi_st - 0.05:
        _Qinv_p = -(_Qm + _Qdep_total)

    if abs(_Qm) > 1e-12:
        _x_end = max(_x_ext_um[-1], 0.02) if len(_x_ext_um) > 1 else 0.1
        _gap = max(0.12 * _x_end, 0.008)
        _bw = _gap * 0.35
        _rho_sc = _rho / q * 1e-15
        _pk = max(np.max(np.abs(_rho_sc)), 0.1) if len(_rho_sc) > 1 else 1.0
        _Qdep_ref = _Qd_max if _Qd_max > 1e-20 else abs(_Qm)
        _bh = (_Qm / _Qdep_ref) * _pk
        _ax_rho.bar(-_gap - _bw, _bh, width=_bw, color='darkorange', alpha=0.7,
                    edgecolor='k', linewidth=1, zorder=5)
        _sgn = '+' if _Qm > 0 else '\u2212'
        _ax_rho.text(-_gap - _bw / 2, _bh + np.sign(_bh) * _pk * 0.12,
                     f'{_sgn}$Q_M$', fontsize=13, ha='center',
                     fontweight='bold', color='darkorange')
        _ax_rho.axvspan(-_gap, 0, color='#ffffcc', alpha=0.15, zorder=0)
        _ax_rho.text(-_gap / 2, _pk * 0.03, 'oxide', fontsize=10,
                     ha='center', va='bottom', alpha=0.4, style='italic')

        if abs(_Qinv_p) > 1e-12:
            _bh_inv = (_Qinv_p / _Qdep_ref) * _pk
            _ax_rho.bar(0, _bh_inv, width=_bw, color='purple', alpha=0.7,
                        edgecolor='k', linewidth=1, zorder=5)
            _sgn_inv = '\u2212' if _Qinv_p < 0 else '+'
            _ax_rho.text(_bw / 2 + 0.002 * _x_end, _bh_inv + np.sign(_bh_inv) * _pk * 0.12,
                         f'{_sgn_inv}$Q_{{inv}}$', fontsize=12, ha='left',
                         fontweight='bold', color='purple')

        _ax_rho.set_xlim(-_gap - _bw * 3, _x_end * 1.05)

    _ax_E.plot(_x_es_um, _E_field, 'r-', linewidth=2.5)
    _ax_E.plot(_x_ext_um, _E_ext, 'r-', linewidth=2.5)
    _ax_E.fill_between(_x_es_um, 0, _E_field, alpha=0.2, color='red')
    _ax_E.axhline(0, color='k', linewidth=0.5)
    _ax_E.set_ylabel(r'$\mathcal{E}$ (V/cm)', fontsize=16)
    _ax_E.set_xlabel(r'$x$ ($\mu$m)', fontsize=16)
    _ax_E.set_title(r'Electric field $\mathcal{E}(x)$', fontsize=16, fontweight='bold')
    _ax_E.tick_params(labelsize=14)
    _ax_E.grid(True, alpha=0.3)
    if abs(_E_max) > 0:
        _ax_E.text(0.95, 0.95, f'$|\\mathcal{{E}}_{{max}}|$ = {abs(_E_max):.2e} V/cm',
                   fontsize=13, transform=_ax_E.transAxes, va='top', ha='right',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    _ax_V.plot(_x_es_um, _V_es, 'g-', linewidth=2.5)
    _ax_V.plot(_x_ext_um, _V_ext, 'g-', linewidth=2.5)
    _ax_V.fill_between(_x_es_um, 0, _V_es, alpha=0.2, color='green')
    _ax_V.axhline(0, color='k', linewidth=0.5)
    if abs(_phi_s) > 0.01:
        _ax_V.axhline(_phi_s, color='gray', ls=':', lw=1, alpha=0.6)
        _y_off = 0.01 if _phi_s > 0 else -0.03
        _ax_V.text(0.01, _phi_s + _y_off, f'$\\phi_s$ = {_phi_s:.3f} V', fontsize=13, color='gray')
    _ax_V.set_ylabel(r'$V(x)$ (V)', fontsize=16)
    _ax_V.set_xlabel(r'$x$ ($\mu$m)', fontsize=16)
    _ax_V.set_title(r'Potential $V(x)$', fontsize=16, fontweight='bold')
    _ax_V.tick_params(labelsize=14)
    _ax_V.grid(True, alpha=0.3)

    _fig.suptitle(f'P-type MOS Electrostatics — {_regime}',
                  fontsize=16, fontweight='bold', color=_regime_color, y=1.02)
    plt.tight_layout()
    plt.close(_fig)

    _info = mo.md(rf"""
    | Quantity | Value |
    |:---------|:------|
    | **Regime** | {_regime} |
    | $V_G$ | {_Vg:.2f} V |
    | $\phi_s$ (surface potential) | {_phi_s:.4f} V |
    | $\phi_B$ | {_phi_B:.4f} V |
    | $2\phi_B$ (threshold condition) | {_phi_st:.4f} V |
    | $W_{{dep}}$ | {_Wdep:.2e} cm = {_Wdep_um:.3f} $\mu$m |
    | $W_{{dep,max}}$ | {_Wdep_max:.2e} cm |
    | $V_T$ (threshold voltage) | {_Vt:.3f} V |
    | $\mathcal{{E}}_{{max}}$ | {_E_max:.2e} V/cm |
    | $Q_M$ (gate charge/area) | {_Qm:.2e} C/cm² |
    | $Q_{{inv}}$ (inversion charge/area) | {_Qinv_p:.2e} C/cm² |
    | $N_A$ | {_Na:.2e} cm$^{{-3}}$ |
    | $t_{{ox}}$ | {tox_es_slider.value:.1f} nm |
    """)

    _controls = mo.vstack([
        Vg_es_slider, Na_es_slider, tox_es_slider])

    mo.vstack([
        mo.md("### Interactive Electrostatics (P-type)"),
        _controls,
        mo.as_html(_fig),
        _info,
    ])
    return


@app.cell(hide_code=True)
def _(eps_0, eps_Si, eps_ox, kT_300, mo, ni_Si, np, q):
    _Na_ex = 1e16
    _tox_ex = 5e-7  # 5 nm

    _phi_B_ex = kT_300 * np.log(_Na_ex / ni_Si)
    _phi_st_ex = 2 * _phi_B_ex
    _Cox_ex = eps_ox * eps_0 / _tox_ex
    _eps_s = eps_Si * eps_0
    _Wdep_max_ex = np.sqrt(2 * _eps_s * _phi_st_ex / (q * _Na_ex))
    _Qd_max_ex = q * _Na_ex * _Wdep_max_ex
    _Vt_ex = _phi_st_ex + _Qd_max_ex / _Cox_ex

    mo.md(rf"""
    ## Threshold Voltage Derivation

    At threshold, $\phi_s = \phi_{{s,T}} = 2\phi_B$ and the depletion layer reaches its maximum width.
    We substitute $\phi_s = 2\phi_B$ into the gate voltage equation derived above to find $V_T$.

    ### Step 1: Maximum depletion width at threshold

    From $W_{{dep}} = \sqrt{{2\varepsilon_s \phi_s / (qN_A)}}$, substituting $\phi_s = 2\phi_B$:

    $$\boxed{{W_{{dep,max}} = \sqrt{{\frac{{4\varepsilon_s \phi_B}}{{q N_A}}}}}}$$

    ### Step 2: Depletion charge at threshold

    $$Q_{{dep,max}} = q N_A W_{{dep,max}} = \sqrt{{2 q \varepsilon_s N_A \cdot 2\phi_B}}$$

    ### Step 3: Threshold voltage

    Substituting into $V_G = \phi_s + Q_{{dep}}/C_{{ox}}$:

    $$\boxed{{V_T = 2\phi_B + \frac{{Q_{{dep,max}}}}{{C_{{ox}}}} = 2\phi_B + \frac{2}{{C_{{ox}}}}\sqrt{{q N_A \varepsilon_s \phi_B}}}}$$

    where $C_{{ox}} = \varepsilon_{{ox}} / t_{{ox}}$ is the oxide capacitance per unit area.

    - The $2\phi_B$ term is the surface potential needed to invert the surface.
    - The $Q_{{dep,max}}/C_{{ox}}$ term is the voltage drop across the oxide to support the depletion charge.

    ### Numerical Example (${{N_A = }}$ {_Na_ex:.0e} cm$^{{-3}}$, $t_{{ox}}$ = 5 nm)

    | Quantity | Expression | Value |
    |:---------|:-----------|:------|
    | $\phi_B$ | $(kT/q)\ln(N_A/n_i)$ | {_phi_B_ex:.4f} V |
    | $2\phi_B$ | | {_phi_st_ex:.4f} V |
    | $W_{{dep,max}}$ | $\sqrt{{4\varepsilon_s \phi_B / (qN_A)}}$ | {_Wdep_max_ex:.2e} cm = {_Wdep_max_ex*1e4:.3f} $\mu$m |
    | $C_{{ox}}$ | $\varepsilon_{{ox}} / t_{{ox}}$ | {_Cox_ex:.2e} F/cm$^2$ |
    | $Q_{{dep,max}}$ | $qN_A W_{{dep,max}}$ | {_Qd_max_ex:.2e} C/cm$^2$ |
    | $V_T$ | $2\phi_B + Q_{{dep,max}}/C_{{ox}}$ | **{_Vt_ex:.3f} V** |

    $V_T$ increases with higher $N_A$ (more charge to deplete) and larger $t_{{ox}}$ (less oxide capacitance).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## N-Body vs. P-Body MOS Capacitor

    The same physics applies to an N-type semiconductor body, with reversed signs.

    | Property | P-body (NMOS) | N-body (PMOS) |
    |:---------|:--------------|:--------------|
    | **Substrate** | $N_A$ (acceptors) | $N_D$ (donors) |
    | **Majority carriers** | Holes | Electrons |
    | **Inversion carriers** | Electrons | Holes |
    | **Accumulation** | $V_G < 0$ | $V_G > 0$ |
    | **Inversion** | $V_G > V_T > 0$ | $V_G < V_T < 0$ |
    | **$\phi_B$** | $(kT/q)\ln(N_A/n_i) > 0$ | $(kT/q)\ln(N_D/n_i) > 0$ |
    | **$V_T$** | $2\phi_B + Q_{dep,max}/C_{ox} > 0$ | $-2\phi_B - Q_{dep,max}/C_{ox} < 0$ |
    | **Bands at inversion** | Bend **down** | Bend **up** |

    For the N-body:

    - **Inversion** (holes at surface) occurs for $V_G < V_T$, where $V_T < 0$
    - The depletion region contains uncompensated *positive* donor charges ($\rho = +qN_D$)
    - All expressions are the same with $N_A \to N_D$ and sign reversals on $V_G$, $V_T$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    Vg_n_slider = mo.ui.slider(
        start=-3.0, stop=3.0, step=0.05, value=0.0,
        label=r"$V_G$ (V)", full_width=False
    )
    Nd_slider = mo.ui.slider(
        start=15, stop=18, step=0.1, value=16.0,
        label=r"$\log_{10}(N_D)$ (cm$^{-3}$)", full_width=False
    )
    tox_n_slider = mo.ui.slider(
        start=2.0, stop=20.0, step=0.5, value=5.0,
        label=r"$t_{ox}$ (nm)", full_width=False
    )
    return Nd_slider, Vg_n_slider, tox_n_slider


@app.cell(hide_code=True)
def _(
    Ec_minus_Ei,
    Ei_minus_Ev,
    Nd_slider,
    Vg_n_slider,
    dEc_ox,
    dEv_ox,
    eps_0,
    eps_Si,
    eps_ox,
    kT_300,
    mo,
    ni_Si,
    np,
    plt,
    q,
    tox_n_slider,
):
    _Vg = Vg_n_slider.value
    _Nd = 10 ** Nd_slider.value
    _tox_nm = tox_n_slider.value
    _tox_cm = _tox_nm * 1e-7

    _phi_B = kT_300 * np.log(_Nd / ni_Si)
    _Cox = eps_ox * eps_0 / _tox_cm

    _phi_st = 2 * _phi_B
    _Wdep_max = np.sqrt(2 * eps_Si * eps_0 * _phi_st / (q * _Nd))
    _Qd_max = q * _Nd * _Wdep_max
    _Vt = -_phi_st - _Qd_max / _Cox

    _eps_s = eps_Si * eps_0

    def _solve_phi_s_n(Vg, Nd, Cox, phi_B):
        if abs(Vg) < 1e-6:
            return 0.0
        phi_s_min = -2 * phi_B - 6 * kT_300
        gamma = np.sqrt(2 * q * _eps_s * Nd) / Cox

        if Vg < 0:
            phi_s = -0.1
            for _ in range(100):
                if phi_s >= 0:
                    phi_s = -0.01
                abs_phi_s = abs(phi_s)
                if abs_phi_s < 1e-10:
                    abs_phi_s = 1e-10
                if phi_s < phi_s_min:
                    phi_s = phi_s_min
                    break
                f = phi_s - gamma * np.sqrt(abs_phi_s) - Vg
                df = 1 - gamma / (2 * np.sqrt(abs_phi_s))
                if abs(df) < 1e-12:
                    break
                delta = f / df
                phi_s -= delta
                if abs(delta) < 1e-8:
                    break
            return max(phi_s, phi_s_min)
        else:
            phi_s = 0.01
            for _ in range(100):
                if phi_s <= 0:
                    phi_s = 0.001
                u_s = min(phi_s / kT_300, 500)
                arg = max(np.exp(u_s) - u_s - 1, 0)
                Q_s = np.sqrt(2 * q * _eps_s * Nd * kT_300 * arg)
                f = phi_s + Q_s / Cox - Vg
                if Q_s > 1e-20:
                    dQ_s = q * _eps_s * Nd * (np.exp(u_s) - 1) / Q_s
                else:
                    dQ_s = 0
                df = 1 + dQ_s / Cox
                if abs(df) < 1e-12:
                    break
                delta = f / df
                phi_s -= delta
                if abs(delta) < 1e-8:
                    break
            return min(phi_s, 1.0)

    _phi_s = _solve_phi_s_n(_Vg, _Nd, _Cox, _phi_B)

    if _phi_s > 0.01:
        _regime = "Accumulation"
        _regime_color = '#d62728'
    elif abs(_phi_s) <= 0.01:
        _regime = "Flat Band"
        _regime_color = '#7f7f7f'
    elif _phi_s > -(2 * _phi_B - 0.05):
        _regime = "Depletion"
        _regime_color = '#2ca02c'
    elif abs(_phi_s + 2 * _phi_B) < 0.15:
        _regime = "Threshold"
        _regime_color = '#ff7f0e'
    else:
        _regime = "Strong Inversion"
        _regime_color = '#1f77b4'

    if _phi_s < 0:
        _Wdep = np.sqrt(2 * eps_Si * eps_0 * min(abs(_phi_s), _phi_st) / (q * _Nd))
        _Wdep_um = _Wdep * 1e4
    else:
        _Wdep = 0.0
        _Wdep_um = 0.0

    _EF = 0.0
    _Ei_bulk = -_phi_B  # E_F - E_Fi = phi_B for n-type, so E_Fi = -phi_B
    _Ec_bulk = _Ei_bulk + Ec_minus_Ei
    _Ev_bulk = _Ei_bulk - Ei_minus_Ev
    _EFm = _EF - _Vg

    _fig_band, _ax_band = plt.subplots(1, 1, figsize=(8.8, 4.8))

    # ──── Energy band diagram (top row) ────
    _xM = [-1.8, -1.2]
    _xO = [-1.2, 0.0]
    _x_sc_end = 3.0

    if _phi_s < -0.001:
        _Wdep_plot = min(2.5, max(0.4, _Wdep * 1e4 * 2.5))
        _x_dep = np.linspace(0, _Wdep_plot, 300)
        _x_flat = np.linspace(_Wdep_plot, _x_sc_end, 50)
        _bending = _phi_s * (1 - _x_dep / _Wdep_plot) ** 2
        _x_sc = np.concatenate([_x_dep, _x_flat])
        _Ec_sc = np.concatenate([_Ec_bulk - _bending, np.full_like(_x_flat, _Ec_bulk)])
        _Ev_sc = np.concatenate([_Ev_bulk - _bending, np.full_like(_x_flat, _Ev_bulk)])
        _Ei_sc = np.concatenate([_Ei_bulk - _bending, np.full_like(_x_flat, _Ei_bulk)])
    elif _phi_s > 0.001:
        _bend_len = 0.5
        _x_acc = np.linspace(0, _bend_len, 150)
        _x_flat = np.linspace(_bend_len, _x_sc_end, 50)
        _bending = _phi_s * (1 - _x_acc / _bend_len) ** 2
        _x_sc = np.concatenate([_x_acc, _x_flat])
        _Ec_sc = np.concatenate([_Ec_bulk - _bending, np.full_like(_x_flat, _Ec_bulk)])
        _Ev_sc = np.concatenate([_Ev_bulk - _bending, np.full_like(_x_flat, _Ev_bulk)])
        _Ei_sc = np.concatenate([_Ei_bulk - _bending, np.full_like(_x_flat, _Ei_bulk)])
    else:
        _x_sc = np.array([0, _x_sc_end])
        _Ec_sc = np.array([_Ec_bulk, _Ec_bulk])
        _Ev_sc = np.array([_Ev_bulk, _Ev_bulk])
        _Ei_sc = np.array([_Ei_bulk, _Ei_bulk])

    _ax_band.fill_between(_xM, _EFm - 3.0, _EFm, color='#6baed6', alpha=0.3)
    _ax_band.plot(_xM, [_EFm, _EFm], 'g-', linewidth=2)
    _ax_band.text(_xM[0] + 0.05, _EFm + 0.1, '$E_{F,M}$', fontsize=14, color='green')

    _Ec_ox_s = _Ec_sc[0] + dEc_ox
    _Ev_ox_s = _Ev_sc[0] - dEv_ox
    _V_ox = _Vg - _phi_s
    _Ec_ox_m = _Ec_ox_s - _V_ox
    _Ev_ox_m = _Ev_ox_s - _V_ox

    _ax_band.fill_between(_xO, [_Ev_ox_m, _Ev_ox_s], [_Ec_ox_m, _Ec_ox_s],
                          color='#ffffcc', alpha=0.4, edgecolor='none')
    _ax_band.plot(_xO, [_Ec_ox_m, _Ec_ox_s], 'k-', linewidth=2)
    _ax_band.plot(_xO, [_Ev_ox_m, _Ev_ox_s], 'k-', linewidth=2)

    _ax_band.plot(_x_sc, _Ec_sc, 'b-', linewidth=2.5)
    _ax_band.plot(_x_sc, _Ev_sc, 'r-', linewidth=2.5)
    _ax_band.plot(_x_sc, _Ei_sc, 'k--', linewidth=1, alpha=0.5)
    _ax_band.plot([0, _x_sc_end], [_EF, _EF], 'g-', linewidth=2)

    _ax_band.text(_x_sc_end + 0.1, _Ec_bulk, '$E_c$', fontsize=14, va='center', color='blue')
    _ax_band.text(_x_sc_end + 0.1, _Ev_bulk, '$E_v$', fontsize=14, va='center', color='red')
    _ax_band.text(_x_sc_end + 0.1, _Ei_bulk, '$E_{Fi}$', fontsize=14, va='center', alpha=0.6)
    _ax_band.text(_x_sc_end + 0.1, _EF, '$E_{Fs}$', fontsize=14, va='center', color='green')

    # Interface lines (from Ev,ox to Ec,ox)
    _ax_band.plot([-1.2, -1.2], [_Ev_ox_m, _Ec_ox_m], 'k-', lw=1.5)
    _ax_band.plot([0, 0], [_Ev_ox_s, _Ec_ox_s], 'k-', lw=1.5)

    if abs(_phi_s) > 0.05:
        _cap = 0.06
        _xann = 0.2
        _ax_band.plot([_xann, _xann], [_Ec_sc[0], _Ec_bulk], color='darkgreen', lw=1.5)
        _ax_band.plot([_xann - _cap, _xann + _cap], [_Ec_sc[0], _Ec_sc[0]], color='darkgreen', lw=1.5)
        _ax_band.plot([_xann - _cap, _xann + _cap], [_Ec_bulk, _Ec_bulk], color='darkgreen', lw=1.5)
        _ax_band.text(_xann + 0.15, (_Ec_sc[0] + _Ec_bulk) / 2, r'$q\phi_s$',
                      fontsize=14, color='darkgreen', va='center')

    # phi_B annotation (between E_F and E_Fi,bulk — for n-type, E_F > E_Fi)
    _cap = 0.06
    _xpB = _x_sc_end - 0.6
    _ax_band.plot([_xpB, _xpB], [_Ei_bulk, _EF], color='purple', lw=1.5)
    _ax_band.plot([_xpB - _cap, _xpB + _cap], [_Ei_bulk, _Ei_bulk], color='purple', lw=1.5)
    _ax_band.plot([_xpB - _cap, _xpB + _cap], [_EF, _EF], color='purple', lw=1.5)
    _ax_band.text(_xpB + 0.12, (_Ei_bulk + _EF) / 2, r'$q\phi_B$',
                  fontsize=14, color='purple', va='center')

    # V_ox annotation at oxide-metal interface
    if abs(_V_ox) > 0.05:
        _cap = 0.06
        _xvox = -1.15
        _ax_band.plot([_xvox, _xvox], [_Ec_ox_m, _Ec_ox_s], color='darkorange', lw=1.5)
        _ax_band.plot([_xvox - _cap, _xvox + _cap], [_Ec_ox_m, _Ec_ox_m], color='darkorange', lw=1.5)
        _ax_band.plot([_xvox - _cap, _xvox + _cap], [_Ec_ox_s, _Ec_ox_s], color='darkorange', lw=1.5)
        _ax_band.text(_xvox - 0.12, (_Ec_ox_m + _Ec_ox_s) / 2, r'$qV_{ox}$',
                      fontsize=14, color='darkorange', va='center', ha='right')

    if abs(_Vg) > 0.1:
        _xvg = -1.35
        _ax_band.plot([_xvg, _xvg], [_EFm, _EF], color='darkred', lw=1.5)
        _cap = 0.06
        _ax_band.plot([_xvg - _cap, _xvg + _cap], [_EFm, _EFm], color='darkred', lw=1.5)
        _ax_band.plot([_xvg - _cap, _xvg + _cap], [_EF, _EF], color='darkred', lw=1.5)
        _ax_band.text(_xvg - 0.15, (_EFm + _EF) / 2, r'$qV_G$',
                      fontsize=14, color='darkred', va='center', ha='right')

    # Oxide band labels
    _ax_band.text(-0.6, max(_Ec_ox_m, _Ec_ox_s) + 0.15, '$E_{c,ox}$', fontsize=13, ha='center')
    _ax_band.text(-0.6, min(_Ev_ox_m, _Ev_ox_s) - 0.3, '$E_{v,ox}$', fontsize=13, ha='center')

    _ax_band.text((_x_sc_end) / 2, _Ec_bulk + dEc_ox + 0.05, _regime,
                  fontsize=16, ha='center', fontweight='bold', color=_regime_color,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=_regime_color, alpha=0.9))
    _ax_band.text(-1.5, min(_Ev_ox_m, _Ev_ox_s) - 0.8, 'Metal', fontsize=13, ha='center', fontweight='bold')
    _ax_band.text(1.5, min(_Ev_ox_s, _Ev_ox_m) - 0.8, 'N-type Si', fontsize=13, ha='center', fontweight='bold')

    _y_lo = min(_Ev_ox_s, _Ev_ox_m, _EFm) - 1.5
    _y_hi = max(_Ec_bulk + dEc_ox + 0.8, _Ec_ox_m + 0.5, _EFm + 1.0)
    _ax_band.set_xlim(-2.1, _x_sc_end + 0.8)
    _ax_band.set_ylim(_y_lo, _y_hi)
    _ax_band.set_ylabel('Energy (eV)', fontsize=16)
    _ax_band.set_title('Energy Bands (N-type)', fontsize=16, fontweight='bold')
    _ax_band.set_xticks([])
    _ax_band.tick_params(labelsize=14)
    for _sp in ['top', 'right', 'bottom']:
        _ax_band.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig_band)

    # ──── Electrostatics (bottom row, 3 panels) ────
    _fig_es, (_ax_rho, _ax_E, _ax_V) = plt.subplots(1, 3, figsize=(16, 5))
    _eps_s = eps_Si * eps_0
    if _phi_s < -0.001:
        _Wdep_es = np.sqrt(2 * _eps_s * min(abs(_phi_s), _phi_st) / (q * _Nd))
        _x_es = np.linspace(0, _Wdep_es, 500)
        _x_es_um = _x_es * 1e4
        _x_ext = np.linspace(_Wdep_es, _Wdep_es * 1.5, 50)
        _x_ext_um = _x_ext * 1e4

        _rho = np.full_like(_x_es, q * _Nd)
        _E_field = -q * _Nd / _eps_s * (_Wdep_es - _x_es)
        _E_max = -q * _Nd * _Wdep_es / _eps_s
        _V_es = _phi_s * (1 - _x_es / _Wdep_es) ** 2
    elif _phi_s > 0.01:
        _L_D = np.sqrt(_eps_s * kT_300 / (q * _Nd))

        _N_pts = 600
        _V_cutoff = kT_300 * 0.1
        _V_arr = np.linspace(_phi_s, _V_cutoff, _N_pts)
        _u_arr = _V_arr / kT_300
        _E_sq = 2 * q * _Nd * kT_300 / _eps_s * (np.exp(_u_arr) - _u_arr - 1)
        _E_arr = np.sqrt(np.maximum(_E_sq, 1e-10))

        _dV = np.diff(_V_arr)
        _E_mid = 0.5 * (_E_arr[:-1] + _E_arr[1:])
        _E_mid = np.maximum(_E_mid, 1.0)
        _dx_arr = -_dV / _E_mid
        _x_arr = np.concatenate([[0], np.cumsum(_dx_arr)])

        _x_es = _x_arr
        _x_es_um = _x_es * 1e4
        _x_end_cm = _x_arr[-1]
        _x_ext = np.linspace(_x_end_cm, _x_end_cm + 4 * _L_D, 50)
        _x_ext_um = _x_ext * 1e4

        _rho = q * _Nd * (1 - np.exp(_u_arr))
        _E_field = _E_arr
        _E_max = _E_arr[0]
        _V_es = _V_arr
    else:
        _x_es = np.array([0])
        _x_es_um = np.array([0])
        _x_ext = np.linspace(0, 0.5e-4, 50)
        _x_ext_um = _x_ext * 1e4
        _rho = np.array([0])
        _E_field = np.array([0])
        _E_max = 0
        _V_es = np.array([0])

    _rho_ext = np.zeros_like(_x_ext)
    _E_ext = np.zeros_like(_x_ext)
    _V_ext = np.zeros_like(_x_ext)

    _ax_rho.plot(_x_es_um, _rho / q * 1e-15, 'b-', linewidth=2.5)
    _ax_rho.plot(_x_ext_um, _rho_ext, 'b-', linewidth=2.5)
    _ax_rho.fill_between(_x_es_um, 0, _rho / q * 1e-15, alpha=0.2, color='blue')
    _ax_rho.axhline(0, color='k', linewidth=0.5)
    _ax_rho.set_ylabel(r'$\rho / q$ ($\times 10^{15}$ cm$^{-3}$)', fontsize=16)
    _ax_rho.set_xlabel(r'$x$ ($\mu$m)', fontsize=16)
    _ax_rho.set_title(r'$\rho(x)$', fontsize=16, fontweight='bold')
    _ax_rho.tick_params(labelsize=14)
    _ax_rho.grid(True, alpha=0.3)

    _Qm = _Cox * (_Vg - _phi_s)
    _Qdep_total = q * _Nd * _Wdep if _phi_s < -0.01 else 0.0
    _Qinv_n = 0.0
    if abs(_phi_s) >= _phi_st - 0.05:
        _Qinv_n = -(_Qm + _Qdep_total)

    if abs(_Qm) > 1e-12:
        _x_end = max(_x_ext_um[-1], 0.02) if len(_x_ext_um) > 1 else 0.1
        _gap = max(0.12 * _x_end, 0.008)
        _bw = _gap * 0.35
        _rho_sc = _rho / q * 1e-15
        _pk = max(np.max(np.abs(_rho_sc)), 0.1) if len(_rho_sc) > 1 else 1.0
        _Qdep_ref = _Qd_max if _Qd_max > 1e-20 else abs(_Qm)
        _bh = (_Qm / _Qdep_ref) * _pk
        _ax_rho.bar(-_gap - _bw, _bh, width=_bw, color='darkorange', alpha=0.7,
                    edgecolor='k', linewidth=1, zorder=5)
        _sgn = '+' if _Qm > 0 else '\u2212'
        _ax_rho.text(-_gap - _bw / 2, _bh + np.sign(_bh) * _pk * 0.12,
                     f'{_sgn}$Q_M$', fontsize=13, ha='center',
                     fontweight='bold', color='darkorange')
        _ax_rho.axvspan(-_gap, 0, color='#ffffcc', alpha=0.15, zorder=0)
        _ax_rho.text(-_gap / 2, _pk * 0.03, 'oxide', fontsize=10,
                     ha='center', va='bottom', alpha=0.4, style='italic')

        if abs(_Qinv_n) > 1e-12:
            _bh_inv = (_Qinv_n / _Qdep_ref) * _pk
            _ax_rho.bar(0, _bh_inv, width=_bw, color='purple', alpha=0.7,
                        edgecolor='k', linewidth=1, zorder=5)
            _sgn_inv = '+' if _Qinv_n > 0 else '\u2212'
            _ax_rho.text(_bw / 2 + 0.002 * _x_end, _bh_inv + np.sign(_bh_inv) * _pk * 0.12,
                         f'{_sgn_inv}$Q_{{inv}}$', fontsize=12, ha='left',
                         fontweight='bold', color='purple')

        _ax_rho.set_xlim(-_gap - _bw * 3, _x_end * 1.05)

    _ax_E.plot(_x_es_um, _E_field, 'r-', linewidth=2.5)
    _ax_E.plot(_x_ext_um, _E_ext, 'r-', linewidth=2.5)
    _ax_E.fill_between(_x_es_um, 0, _E_field, alpha=0.2, color='red')
    _ax_E.axhline(0, color='k', linewidth=0.5)
    _ax_E.set_ylabel(r'$\mathcal{E}$ (V/cm)', fontsize=16)
    _ax_E.set_xlabel(r'$x$ ($\mu$m)', fontsize=16)
    _ax_E.set_title(r'$\mathcal{E}(x)$', fontsize=16, fontweight='bold')
    _ax_E.tick_params(labelsize=14)
    _ax_E.grid(True, alpha=0.3)
    if abs(_E_max) > 0:
        _ax_E.text(0.95, 0.95, f'$|\\mathcal{{E}}_{{max}}|$ = {abs(_E_max):.2e} V/cm',
                   fontsize=13, transform=_ax_E.transAxes, va='top', ha='right',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    _ax_V.plot(_x_es_um, _V_es, 'g-', linewidth=2.5)
    _ax_V.plot(_x_ext_um, _V_ext, 'g-', linewidth=2.5)
    _ax_V.fill_between(_x_es_um, 0, _V_es, alpha=0.2, color='green')
    _ax_V.axhline(0, color='k', linewidth=0.5)
    if abs(_phi_s) > 0.01:
        _ax_V.axhline(_phi_s, color='gray', ls=':', lw=1, alpha=0.6)
        _y_off = 0.01 if _phi_s < 0 else -0.03
        _ax_V.text(0.01, _phi_s + _y_off, f'$\\phi_s$ = {_phi_s:.3f} V', fontsize=13, color='gray')
    _ax_V.set_ylabel(r'$V(x)$ (V)', fontsize=16)
    _ax_V.set_xlabel(r'$x$ ($\mu$m)', fontsize=16)
    _ax_V.set_title(r'$V(x)$', fontsize=16, fontweight='bold')
    _ax_V.tick_params(labelsize=14)
    _ax_V.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.close(_fig_es)

    _info = mo.md(rf"""
    | Quantity | Value |
    |:---------|:------|
    | **Regime** | {_regime} |
    | $V_G$ | {_Vg:.2f} V |
    | $\phi_B$ | {_phi_B:.4f} eV |
    | $\phi_s$ (surface potential) | {_phi_s:.4f} eV |
    | $-2\phi_B$ (threshold condition) | {-2*_phi_B:.4f} eV |
    | $W_{{dep}}$ | {_Wdep:.2e} cm = {_Wdep_um:.3f} $\mu$m |
    | $W_{{dep,max}}$ | {_Wdep_max:.2e} cm |
    | $V_T$ (threshold voltage) | {_Vt:.3f} V |
    | $Q_M$ (gate charge/area) | {_Qm:.2e} C/cm² |
    | $Q_{{inv}}$ (inversion charge/area) | {_Qinv_n:.2e} C/cm² |
    | $N_D$ | {_Nd:.2e} cm$^{{-3}}$ |
    | $t_{{ox}}$ | {_tox_nm:.1f} nm |
    """)

    _controls = mo.hstack([
        mo.vstack([Vg_n_slider, Nd_slider, tox_n_slider]),
    ], widths=[0.7])

    mo.vstack([
        mo.md("### Interactive N-type MOS (Energy Bands + Electrostatics)"),
        _controls,
        mo.as_html(_fig_band),
        mo.as_html(_fig_es),
        _info,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    ### Sign conventions and definitions

    | Quantity | Definition | P-body | | N-body | |
    |:---------|:-----------|:------:|:--:|:------:|:--:|
    | $V_G$ | $V_{ox} + \phi_s$ | > 0 | | < 0 | |
    | $qV_G$ | $E_{Fs} - E_{Fm}$ | > 0 | | < 0 | |
    | $q\phi_s$ | $E_{c,\text{bulk}} - E_{c,\text{surf}}$ | > 0 | | < 0 | |
    | $q\phi_B$ | $E_{i,\text{bulk}} - E_{Fs}$ | > 0 | | < 0 | |
    | $\phi_B$ | $\frac{kT}{q}\ln\!\left(\frac{N_A}{n_i}\right)$ | > 0 | | $-\frac{kT}{q}\ln\!\left(\frac{N_D}{n_i}\right)$ | < 0 |
    | $\phi_{st}$ | $2\phi_B$ | > 0 | | $2\phi_B$ | < 0 |

    ### Key equations

    | Quantity | P-body | | N-body | |
    |:---------|:------:|:--:|:------:|:--:|
    | $\phi_s$ | $\dfrac{qN_A W_{dep}^2}{2\varepsilon_s}$ | > 0 | $-\dfrac{qN_D W_{dep}^2}{2\varepsilon_s}$ | < 0 |
    | $V_{ox}$ | $\dfrac{qN_A W_{dep}}{C_{ox}}$ | > 0 | $-\dfrac{qN_D W_{dep}}{C_{ox}}$ | < 0 |
    | $W_{dep}$ | $\sqrt{\dfrac{2\varepsilon_s \phi_s}{qN_A}}$ | > 0 | $\sqrt{\dfrac{2\varepsilon_s \lvert\phi_s\rvert}{qN_D}}$ | > 0 |
    | $V_T$ | $\phi_{st} + \dfrac{\sqrt{qN_A 2\varepsilon_s \phi_{st}}}{C_{ox}}$ | > 0 | $\phi_{st} - \dfrac{\sqrt{qN_D 2\varepsilon_s \lvert\phi_{st}\rvert}}{C_{ox}}$ | < 0 |

    ### Main concepts

    - $V_G$ is dropped across oxide and semiconductor: $V_G = V_{ox} + \phi_s$.
    - The **surface potential** $\phi_s$ is the amount of band bending at the oxide-semiconductor interface. It is the voltage dropped in the semiconductor.
    - At **threshold**, the surface is as strongly inverted as the bulk is doped: $\phi_s = 2\phi_B$.
    - Beyond threshold, $\phi_s$ is approximately pinned at $2\phi_B$ because the increasing inversion charge screens additional gate voltage.
    - The **depletion width** reaches its maximum $W_{dep,max}$ at threshold and does not grow further in inversion.
    - Regimes of operation: accumulation, depletion, threshold, and inversion.
    - P-body MOS: positive $V_G$ drives toward inversion; N-body MOS: negative $V_G$ drives toward inversion. All equations are related by swapping $N_A \leftrightarrow N_D$ and reversing signs.
    """)
    return


if __name__ == "__main__":
    app.run()
