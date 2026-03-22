# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "matplotlib==3.10.8",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    q = 1.6e-19
    kT_300 = 0.02585
    eps_0 = 8.854e-14       # F/cm
    eps_Si = 11.7
    eps_ox = 3.9
    ni_Si = 1.1e10
    Eg_Si = 1.12             # eV
    chi_Si = 4.05             # eV
    Nc_Si = 2.86e19
    Nv_Si = 1.08e19

    Ec_minus_Ei = Eg_Si / 2 + kT_300 / 2 * np.log(Nc_Si / Nv_Si)
    Ei_minus_Ev = Eg_Si / 2 + kT_300 / 2 * np.log(Nv_Si / Nc_Si)

    Eg_ox = 9.0
    chi_ox = 0.95
    dEc_ox = chi_Si - chi_ox   # ~3.1 eV
    dEv_ox = Eg_ox - Eg_Si - dEc_ox  # ~4.8 eV

    try:
        _test = Path(__file__).parent / "images"
        if _test.exists():
            ASSET_DIR = Path(__file__).parent
        else:
            raise FileNotFoundError
    except Exception:
        ASSET_DIR = None
    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/mos-nonidealities/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    mo.md(r"""
    # MOS Capacitor: Non-Idealities
    **ECE350 Lecture 31-32**

    This notebook covers deviations from the ideal MOS capacitor:

    1. **Work function mismatch** ($\Psi_M \neq \Psi_S$) and the flatband voltage
    2. **Oxide charges** ($Q_{ox}$)
    3. **Enhancement vs. depletion MOS**
    4. **Poly-Si gate depletion**
    5. **Inversion and accumulation charge thickness**
    6. **Interactive C-V** — combined non-idealities
    7. **Extracting non-idealities from C-V**
    """)
    return (
        Ec_minus_Ei,
        Eg_Si,
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
def _(
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
):
    _wf_md = mo.md(r"""
    ## 1. Work Function Mismatch

    If the work functions of the gate and semiconductor are not identical ($\Psi_M \neq \Psi_S$), the energy bands are **not flat** at equilibrium.

    We apply $V_G = V_{FB}$ (**flatband voltage**) if we want the energy bands to be flat again:

    $$\boxed{V_{FB} = \Psi_M - \Psi_S}$$

    $V_{FB}$ counteracts the built-in potential.

    The expressions we used for $V_G$ in our first pass assuming $V_{FB} = 0$ are now shifted by $V_{FB}$:

    $$\boxed{V_G = V_{FB} + \phi_s + V_{ox}} \qquad V_T = V_{FB} + 2\phi_B + \frac{2}{C_{ox}}\sqrt{q N_A \varepsilon_s \phi_B}$$
    """)

    _Na = 1e17
    _phi_B = kT_300 * np.log(_Na / ni_Si)
    _EF = 0.0
    _Ei_bulk = _phi_B
    _Ec_bulk = _Ei_bulk + Ec_minus_Ei
    _Ev_bulk = _Ei_bulk - Ei_minus_Ev

    _Psi_S = chi_Si + Ec_minus_Ei + _phi_B
    _Psi_M_nplus = chi_Si
    _Vfb = _Psi_M_nplus - _Psi_S

    _eps_s = eps_Si * eps_0
    _tox = 5e-7
    _Cox = eps_ox * eps_0 / _tox
    _gamma = np.sqrt(2 * q * _eps_s * _Na) / _Cox

    _Vg_eff_0 = 0 - _Vfb
    _phi_s_0 = 0.1
    for _ in range(100):
        if _phi_s_0 <= 0:
            _phi_s_0 = 0.01
        _f = _phi_s_0 + _gamma * np.sqrt(_phi_s_0) - _Vg_eff_0
        _df = 1 + _gamma / (2 * np.sqrt(_phi_s_0))
        _phi_s_0 -= _f / _df
        if abs(_f) < 1e-8:
            break
    _phi_s_0 = max(_phi_s_0, 0.01)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for _ax, _title, _phi_s, _Vg_applied in [
        (_ax1, r'$V_G = 0$, $\Psi_M < \Psi_S$: bands NOT flat', _phi_s_0, 0.0),
        (_ax2, r'$V_G = V_{FB}$: flat bands restored', 0.0, _Vfb),
    ]:
        _EFm = _EF - _Vg_applied

        _xM = [-1.8, -1.2]
        _xO = [-1.2, 0.0]
        _x_sc_end = 3.0

        if _phi_s > 0.001:
            _Wdep = np.sqrt(2 * _eps_s * _phi_s / (q * _Na))
            _Wdep_plot = min(2.5, max(0.4, _Wdep * 1e4 * 2.5))
            _x_dep = np.linspace(0, _Wdep_plot, 200)
            _x_flat = np.linspace(_Wdep_plot, _x_sc_end, 50)
            _bending = _phi_s * (1 - _x_dep / _Wdep_plot) ** 2
            _x_sc = np.concatenate([_x_dep, _x_flat])
            _Ec_sc = np.concatenate([_Ec_bulk - _bending, np.full_like(_x_flat, _Ec_bulk)])
            _Ev_sc = np.concatenate([_Ev_bulk - _bending, np.full_like(_x_flat, _Ev_bulk)])
            _Ei_sc = np.concatenate([_Ei_bulk - _bending, np.full_like(_x_flat, _Ei_bulk)])
        else:
            _x_sc = np.array([0, _x_sc_end])
            _Ec_sc = np.array([_Ec_bulk, _Ec_bulk])
            _Ev_sc = np.array([_Ev_bulk, _Ev_bulk])
            _Ei_sc = np.array([_Ei_bulk, _Ei_bulk])

        _ax.fill_between(_xM, _EFm - 2.5, _EFm, color='#6baed6', alpha=0.3)
        _ax.plot(_xM, [_EFm, _EFm], 'g-', linewidth=2)
        _ax.text(_xM[0] + 0.05, _EFm + 0.12, '$E_{F,M}$', fontsize=14, color='green')

        _V_ox = _Vg_applied - _Vfb - _phi_s
        _Ec_ox_s = _Ec_sc[0] + dEc_ox
        _Ev_ox_s = _Ev_sc[0] - dEv_ox
        _Ec_ox_m = _Ec_ox_s - _V_ox
        _Ev_ox_m = _Ev_ox_s - _V_ox

        _ax.fill_between(_xO, [_Ev_ox_m, _Ev_ox_s], [_Ec_ox_m, _Ec_ox_s],
                         color='#ffffcc', alpha=0.4, edgecolor='none')
        _ax.plot(_xO, [_Ec_ox_m, _Ec_ox_s], 'k-', linewidth=2)
        _ax.plot(_xO, [_Ev_ox_m, _Ev_ox_s], 'k-', linewidth=2)

        _ax.plot(_x_sc, _Ec_sc, 'b-', linewidth=2.5)
        _ax.plot(_x_sc, _Ev_sc, 'r-', linewidth=2.5)
        _ax.plot(_x_sc, _Ei_sc, 'k--', linewidth=1, alpha=0.5)
        _ax.plot([0, _x_sc_end], [_EF, _EF], 'g-', linewidth=2)

        _ax.text(_x_sc_end + 0.1, _Ec_bulk, '$E_c$', fontsize=14, va='center', color='blue')
        _ax.text(_x_sc_end + 0.1, _Ev_bulk, '$E_v$', fontsize=14, va='center', color='red')
        _ax.text(_x_sc_end + 0.1, _EF, '$E_F$', fontsize=14, va='center', color='green')
        _ax.text(_x_sc_end + 0.1, _Ei_bulk, '$E_{Fi}$', fontsize=14, va='center', alpha=0.6)

        _ax.plot([-1.2, -1.2], [_Ev_ox_m, _Ec_ox_m], 'k-', lw=1.5)
        _ax.plot([0, 0], [_Ev_ox_s, _Ec_ox_s], 'k-', lw=1.5)

        _cap = 0.06

        if abs(_phi_s) > 0.05:
            _xann = 0.2
            _ax.plot([_xann, _xann], [_Ec_sc[0], _Ec_bulk], color='darkgreen', lw=1.5)
            _ax.plot([_xann - _cap, _xann + _cap], [_Ec_sc[0], _Ec_sc[0]], color='darkgreen', lw=1.5)
            _ax.plot([_xann - _cap, _xann + _cap], [_Ec_bulk, _Ec_bulk], color='darkgreen', lw=1.5)
            _ax.text(_xann + 0.15, (_Ec_sc[0] + _Ec_bulk) / 2, r'$q\phi_s$',
                     fontsize=14, color='darkgreen', va='center')

        if abs(_phi_s) > 0.05 and abs(_Vg_applied) < 0.01:
            _xarr = -0.6
            _ax.annotate(
                '', xy=(-0.15, (_Ec_ox_m + _Ec_ox_s) / 2),
                xytext=(-1.05, (_Ec_ox_m + _Ec_ox_s) / 2),
                arrowprops=dict(arrowstyle='->', color='brown', lw=2),
            )
            _ax.text(_xarr, (_Ec_ox_m + _Ec_ox_s) / 2 + 0.2,
                     r'$\mathcal{E}_{bi} > 0$', fontsize=14, color='brown', ha='center')

        if abs(_Vg_applied) > 0.01:
            _xann = -1.35
            _ax.plot([_xann, _xann], [_EFm, _EF], color='darkred', lw=1.5)
            _ax.plot([_xann - _cap, _xann + _cap], [_EFm, _EFm], color='darkred', lw=1.5)
            _ax.plot([_xann - _cap, _xann + _cap], [_EF, _EF], color='darkred', lw=1.5)
            _ax.text(_xann - 0.15, (_EFm + _EF) / 2, r'$qV_{FB}$',
                     fontsize=14, color='darkred', va='center', ha='right')

        _ax.text(-1.5, _Ev_bulk - 1.8, 'N$^+$\npoly-Si', fontsize=14, ha='center', fontweight='bold')
        _ax.text(1.5, _Ev_bulk - 1.8, 'P-type Si', fontsize=14, ha='center', fontweight='bold')
        _ax.set_title(_title, fontsize=14, fontweight='bold')
        _ax.set_xlim(-2.0, _x_sc_end + 0.6)
        _ax.set_ylim(_Ev_bulk - 2.3, max(_Ec_ox_m, _Ec_ox_s) + 0.5)
        _ax.set_xticks([])
        _ax.tick_params(labelsize=14)
        if _ax == _ax1:
            _ax.set_ylabel('Energy (eV)', fontsize=16)
        for _sp in ['top', 'right', 'bottom']:
            _ax.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig)

    _caption = mo.md(rf"""
    **N$^+$ poly-Si gate on P-body** ($N_A$ = {_Na:.0e} cm$^{{-3}}$).
    $\Psi_M = \chi_{{Si}}$ = {chi_Si} eV, $\Psi_S$ = {_Psi_S:.3f} eV.
    Since $\Psi_M < \Psi_S$, the bands bend **downward** at equilibrium ($V_G = 0$) — the MOS is
    partially depleted. To restore flat bands, apply $V_{{FB}} = \Psi_M - \Psi_S$ = {_Vfb:.3f} V.
    """)

    mo.vstack([_wf_md, mo.as_html(_fig), _caption])
    return


@app.cell(hide_code=True)
def _(
    Ec_minus_Ei,
    Eg_Si,
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
):
    _Na_ex = 1e17
    _tox_ex = 5.7e-7
    _eps_s = eps_Si * eps_0
    _phi_B_ex = kT_300 * np.log(_Na_ex / ni_Si)
    _Cox_ex = eps_ox * eps_0 / _tox_ex
    _Wdep_max_ex = np.sqrt(2 * _eps_s * 2 * _phi_B_ex / (q * _Na_ex))
    _Qdep_max_ex = q * _Na_ex * _Wdep_max_ex
    _Vt_ideal_ex = 2 * _phi_B_ex + _Qdep_max_ex / _Cox_ex

    _Psi_S_ex = chi_Si + Ec_minus_Ei + _phi_B_ex
    _Psi_M_nplus = chi_Si
    _Vfb_nplus = _Psi_M_nplus - _Psi_S_ex
    _Vt_nplus = _Vfb_nplus + _Vt_ideal_ex

    _vt_md = mo.md(rf"""
    ### $V_T$ Modification

    We found that if $\Psi_M < \Psi_S$ and for a P-type body, $V_T$ is **reduced** (see above).

    ### Example

    | Parameter | Ideal ($\Psi_M = \Psi_S$) | N$^+$ poly-Si gate |
    |:---|:---|:---|
    | $N_A$ | {_Na_ex:.0e} cm$^{{-3}}$ | {_Na_ex:.0e} cm$^{{-3}}$ |
    | $t_{{ox}}$ | {_tox_ex*1e7:.1f} nm | {_tox_ex*1e7:.1f} nm |
    | $\phi_B$ | {_phi_B_ex:.3f} V | {_phi_B_ex:.3f} V |
    | $V_{{FB}}$ | 0 | {_Vfb_nplus:.2f} V |
    | $V_T$ | **{_Vt_ideal_ex:.2f} V** | **{_Vt_nplus:.2f} V** |

    Replacing the ideal gate ($\Psi_M = \Psi_S$) with N$^+$ poly-Si shifts $V_T$ by $V_{{FB}}$ = {_Vfb_nplus:.2f} V,
    dramatically reducing $V_T$ from {_Vt_ideal_ex:.2f} V to {_Vt_nplus:.2f} V.

    ---

    > **CHECKPOINT:** Sketch the energy diagram for $\Psi_M > \Psi_S$ with a P-type body.
    > What happens to $V_T$?
    """)

    # ── Accordion answer: energy diagram for Psi_M > Psi_S ──
    _Na = 1e17
    _phi_B = kT_300 * np.log(_Na / ni_Si)
    _EF = 0.0
    _Ei_bulk = _phi_B
    _Ec_bulk = _Ei_bulk + Ec_minus_Ei
    _Ev_bulk = _Ei_bulk - Ei_minus_Ev

    _Psi_S = chi_Si + Ec_minus_Ei + _phi_B
    _Psi_M_pplus = chi_Si + Eg_Si
    _Vfb_pp = _Psi_M_pplus - _Psi_S

    _tox = 5.7e-7
    _Cox = eps_ox * eps_0 / _tox
    _phi_s_acc = -0.05

    _Vfb_pp_val = _Vfb_pp
    _Wdep_max = np.sqrt(2 * _eps_s * 2 * _phi_B / (q * _Na))
    _Qdep_max = q * _Na * _Wdep_max
    _Vt_ideal = 2 * _phi_B + _Qdep_max / _Cox
    _Vt_pplus = _Vfb_pp + _Vt_ideal

    _EFm = _EF
    _fig, _ax = plt.subplots(figsize=(8.5, 5))

    _xM = [-1.8, -1.2]
    _xO = [-1.2, 0.0]
    _x_sc_end = 3.0

    _bend_len = 0.5
    _x_acc = np.linspace(0, _bend_len, 150)
    _x_flat = np.linspace(_bend_len, _x_sc_end, 50)
    _bending = _phi_s_acc * (1 - _x_acc / _bend_len) ** 2
    _x_sc = np.concatenate([_x_acc, _x_flat])
    _Ec_sc = np.concatenate([_Ec_bulk - _bending, np.full_like(_x_flat, _Ec_bulk)])
    _Ev_sc = np.concatenate([_Ev_bulk - _bending, np.full_like(_x_flat, _Ev_bulk)])
    _Ei_sc = np.concatenate([_Ei_bulk - _bending, np.full_like(_x_flat, _Ei_bulk)])

    _ax.fill_between(_xM, _EFm - 2.5, _EFm, color='#6baed6', alpha=0.3)
    _ax.plot(_xM, [_EFm, _EFm], 'g-', linewidth=2)
    _ax.text(_xM[0] + 0.05, _EFm + 0.12, '$E_{F,M}$', fontsize=14, color='green')

    _V_ox = 0.0 - _Vfb_pp - _phi_s_acc
    _Ec_ox_s = _Ec_sc[0] + dEc_ox
    _Ev_ox_s = _Ev_sc[0] - dEv_ox
    _Ec_ox_m = _Ec_ox_s - _V_ox
    _Ev_ox_m = _Ev_ox_s - _V_ox

    _ax.fill_between(_xO, [_Ev_ox_m, _Ev_ox_s], [_Ec_ox_m, _Ec_ox_s],
                     color='#ffffcc', alpha=0.4, edgecolor='none')
    _ax.plot(_xO, [_Ec_ox_m, _Ec_ox_s], 'k-', linewidth=2)
    _ax.plot(_xO, [_Ev_ox_m, _Ev_ox_s], 'k-', linewidth=2)

    _ax.plot(_x_sc, _Ec_sc, 'b-', linewidth=2.5)
    _ax.plot(_x_sc, _Ev_sc, 'r-', linewidth=2.5)
    _ax.plot(_x_sc, _Ei_sc, 'k--', linewidth=1, alpha=0.5)
    _ax.plot([0, _x_sc_end], [_EF, _EF], 'g-', linewidth=2)

    _ax.text(_x_sc_end + 0.1, _Ec_bulk, '$E_c$', fontsize=14, va='center', color='blue')
    _ax.text(_x_sc_end + 0.1, _Ev_bulk, '$E_v$', fontsize=14, va='center', color='red')
    _ax.text(_x_sc_end + 0.1, _EF, '$E_F$', fontsize=14, va='center', color='green')
    _ax.text(_x_sc_end + 0.1, _Ei_bulk, '$E_{Fi}$', fontsize=14, va='center', alpha=0.6)

    _ax.plot([-1.2, -1.2], [_Ev_ox_m, _Ec_ox_m], 'k-', lw=1.5)
    _ax.plot([0, 0], [_Ev_ox_s, _Ec_ox_s], 'k-', lw=1.5)

    _ax.annotate(
        '', xy=(-1.05, (_Ec_ox_m + _Ec_ox_s) / 2),
        xytext=(-0.15, (_Ec_ox_m + _Ec_ox_s) / 2),
        arrowprops=dict(arrowstyle='->', color='brown', lw=2),
    )
    _ax.text(-0.6, (_Ec_ox_m + _Ec_ox_s) / 2 + 0.2,
             r'$\mathcal{E}_{bi} < 0$', fontsize=14, color='brown', ha='center')

    _cap = 0.06
    _xann = 0.2
    _ax.plot([_xann, _xann], [_Ec_sc[0], _Ec_bulk], color='darkgreen', lw=1.5)
    _ax.plot([_xann - _cap, _xann + _cap], [_Ec_sc[0], _Ec_sc[0]], color='darkgreen', lw=1.5)
    _ax.plot([_xann - _cap, _xann + _cap], [_Ec_bulk, _Ec_bulk], color='darkgreen', lw=1.5)
    _ax.text(_xann + 0.15, (_Ec_sc[0] + _Ec_bulk) / 2, r'$q\phi_s < 0$',
             fontsize=14, color='darkgreen', va='center')

    _ax.text(-1.5, _Ev_bulk - 1.8, 'P$^+$\npoly-Si', fontsize=14, ha='center', fontweight='bold')
    _ax.text(1.5, _Ev_bulk - 1.8, 'P-type Si', fontsize=14, ha='center', fontweight='bold')
    _ax.set_title(r'$V_G = 0$, $\Psi_M > \Psi_S$: bands bend UP (accumulation)',
                  fontsize=14, fontweight='bold')
    _ax.set_xlim(-2.0, _x_sc_end + 0.6)
    _ax.set_ylim(_Ev_bulk - 2.3, max(_Ec_ox_m, _Ec_ox_s) + 0.5)
    _ax.set_xticks([])
    _ax.set_ylabel('Energy (eV)', fontsize=16)
    _ax.tick_params(labelsize=14)
    for _sp in ['top', 'right', 'bottom']:
        _ax.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig)

    _answer = mo.vstack([
        mo.as_html(_fig),
        mo.md(rf"""
    When $\Psi_M > \Psi_S$ (e.g., P$^+$ poly-Si gate on P-body):

    - $V_{{FB}} = \Psi_M - \Psi_S > 0$ (here $V_{{FB}}$ = +{_Vfb_pp_val:.2f} V)
    - At $V_G = 0$, the effective voltage is $V_G - V_{{FB}} < 0$ $\Rightarrow$ **accumulation**
    - Bands bend **upward** at the surface ($\phi_s < 0$)
    - The oxide field reverses: $\mathcal{{E}}_{{bi}} < 0$

    **What happens to $V_T$?**

    $V_T$ **increases**:

    $$V_T = \underbrace{{V_{{FB}}}}_{{{_Vfb_pp_val:+.2f}}} + 2\phi_B + \frac{{Q_{{dep,max}}}}{{C_{{ox}}}} = {_Vt_pplus:.2f} \text{{ V}}$$

    compared to the ideal $V_T$ = {_Vt_ideal:.2f} V. The positive $V_{{FB}}$ adds to the gate voltage
    needed to reach inversion, so the transistor requires **more voltage** to turn on.

    This is why P$^+$ poly-Si on P-body is **not a common combination**. 
    """),
    ])

    _accordion = mo.accordion({"CHECKPOINT Answer: Energy diagram for $\\Psi_M > \\Psi_S$ (P-type body)": _answer})

    mo.vstack([_vt_md, _accordion])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Nonideality: Oxide Charges

    $Q_{ox}$: **effective charge per area** in the oxide [C/cm$^2$].

    ### Causes

    - **Fixed ions**: structural defects near the Si–SiO$_2$ interface
    - **Mobile ions** (Na$^+$, e.g., from water contamination)
    - **Interface traps**: dangling bonds at the interface
    - **Voltage/temperature stress**: induced charge and traps $\Rightarrow$ also affects reliability

    ### Effect on gate voltage

    The oxide charge adds a potential in the oxide:

    $$\Delta V_{ox} = \frac{Q_{ox}}{C_{ox}}$$

    This modifies the gate voltage equation:

    $$V_G = \phi_s + V_{ox} \quad\Longrightarrow\quad V_G = \phi_s + V_{ox} - \frac{Q_{ox}}{C_{ox}}$$

    ### Modified flatband voltage

    $$\boxed{V_{FB} = \Psi_M - \Psi_S - \frac{Q_{ox}}{C_{ox}}}$$

    - **Positive $Q_{ox}$** $\Rightarrow$ shifts the C-V curve to the **left** (more negative $V_G$)
    - **Negative $Q_{ox}$** $\Rightarrow$ shifts the C-V curve to the **right** (more positive $V_G$)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Enhancement vs. Depletion MOS

    Due to $V_{FB}$, the MOS can be in **depletion**, **inversion**, or **accumulation** at $V_G = 0$.

    ### Enhancement type MOSFET

    - Transistor normally **"off"** (MOS is in depletion when $V_G = 0$)
    - Apply $V_G > V_T$ for the transistor to conduct

    ### Depletion type MOSFET

    - Transistor normally **"on"** (MOS is in inversion when $V_G = 0$)
    - Apply $V_G < V_T$ to turn off channel

    ### P$^+$ poly-Si gate with P-body

    - MOS is in **accumulation** when $V_G = 0$ (since $V_{FB} > 0$)
    - Must apply voltage to bring the MOS first to depletion and then to inversion
    - **Not power efficient**

    ### Most common combinations

    | Configuration | Why |
    |:---|:---|
    | **P-body with N$^+$ poly-Si gate** (NMOS) | $V_{FB} < 0$ → enhancement mode, lower $V_T$ |
    | **N-body with P$^+$ poly-Si gate** (PMOS) | $V_{FB} > 0$ → enhancement mode, lower $\vert V_T\vert$ |
    """)
    return


@app.cell(hide_code=True)
def _(eps_0, eps_Si, eps_ox, mo, np, plt, q):
    _poly_md = mo.md(r"""
    ## 4. Poly-Si Gate Depletion

    If a poly-Si is used for the gate, a **depletion layer can form in the gate**.

    - This effectively **increases $T_{ox}$**
      - Reduces $C_{ox,eff}$ and thus $Q_{inv}$ for a given $V_G$
      - Degrades MOSFET drive current and electrostatic control
    - Modification is more significant for **thin oxides**
    - New MOSFETs are using **metal gates** again!

    ### Series capacitance model

    $$\frac{1}{C_{ox,eff}} = \frac{1}{C_{ox}} + \frac{1}{C_{poly}}$$

    where

    $$C_{ox} = \frac{\varepsilon_{ox}}{T_{ox}}, \qquad C_{poly} = \frac{\varepsilon_s}{W_{dep,poly}}$$
    """)

    # ── Schematic MOS diagram with poly-Si depletion and circuit ──
    _fig, (_ax_mos, _ax_ckt) = plt.subplots(1, 2, figsize=(13, 4.5),
                                              gridspec_kw={'width_ratios': [1, 1.2]})

    _ax = _ax_mos
    _y_bot, _y_gate_top = 0.0, 3.0
    _y_ox_bot, _y_ox_top = 1.0, 1.6
    _y_depl_gate = 1.9
    _y_depl_si = 0.6

    _ax.fill_between([0, 4], _y_bot, _y_ox_bot, color='#ffcccc', alpha=0.5)
    _ax.text(2, 0.1, 'S', fontsize=18, ha='center', va='bottom', fontweight='bold')

    _ax.fill_between([0, 4], _y_depl_si, _y_ox_bot, color='#ffe0e0', alpha=0.8,
                     hatch='--', edgecolor='#cc6666', linewidth=0.5)
    for _yy in [_y_depl_si]:
        _ax.plot([0, 4], [_yy, _yy], 'k--', lw=0.8, alpha=0.6)

    for _xx in np.linspace(0.5, 3.5, 6):
        _ax.text(_xx, _y_ox_bot - 0.08, '−', fontsize=14, ha='center', va='top',
                 color='blue', fontweight='bold')

    _ax.fill_between([0, 4], _y_ox_bot, _y_ox_top, color='#99ccff', alpha=0.5)
    _ax.text(2, (_y_ox_bot + _y_ox_top) / 2, 'O', fontsize=18, ha='center', va='center',
             fontweight='bold')

    _ax.fill_between([0, 4], _y_ox_top, _y_gate_top, color='#ffcc99', alpha=0.5)
    _ax.text(2, 2.6, 'M', fontsize=18, ha='center', va='center', fontweight='bold')

    _ax.fill_between([0, 4], _y_ox_top, _y_depl_gate, color='#ffe0cc', alpha=0.8,
                     hatch='--', edgecolor='#cc9966', linewidth=0.5)
    for _yy in [_y_depl_gate]:
        _ax.plot([0, 4], [_yy, _yy], 'k--', lw=0.8, alpha=0.6)

    for _xx in np.linspace(0.5, 3.5, 6):
        _ax.text(_xx, _y_ox_top + 0.08, '+', fontsize=14, ha='center', va='bottom',
                 color='red', fontweight='bold')

    _ax.annotate('', xy=(4.3, _y_ox_top), xytext=(4.3, _y_depl_gate),
                 arrowprops=dict(arrowstyle='<->', color='brown', lw=1.5))
    _ax.text(4.5, (_y_ox_top + _y_depl_gate) / 2, 'Depletion\nlayer in gate',
             fontsize=12, va='center', color='brown')

    _ax.annotate('', xy=(4.3, _y_depl_si), xytext=(4.3, _y_ox_bot),
                 arrowprops=dict(arrowstyle='<->', color='brown', lw=1.5))
    _ax.text(4.5, (_y_depl_si + _y_ox_bot) / 2, 'Depletion\nlayer, $W_{dep}$',
             fontsize=12, va='center', color='brown')

    _ax.text(2, _y_gate_top + 0.15, '$V_G$', fontsize=16, ha='center', fontweight='bold')
    _ax.plot([2, 2], [_y_gate_top, _y_gate_top + 0.1], 'k-', lw=2)

    _ax.plot([2, 2], [_y_bot - 0.1, _y_bot], 'k-', lw=2)
    _gw = 0.3
    for _dy in [0.0, -0.06, -0.12]:
        _ax.plot([2 - _gw, 2 + _gw], [_y_bot - 0.1 + _dy, _y_bot - 0.1 + _dy], 'k-', lw=1.5)
        _gw -= 0.08

    _ax.set_xlim(-0.5, 7)
    _ax.set_ylim(-0.4, _y_gate_top + 0.5)
    _ax.set_aspect('equal')
    _ax.axis('off')
    _ax.set_title('MOS with Poly-Si Gate Depletion', fontsize=14, fontweight='bold')

    _ax2 = _ax_ckt
    _y_top, _y_bot_c = 3.0, 0.5

    _ax2.plot([3, 3], [_y_top, 2.5], 'k-', lw=2)
    _ax2.text(3, _y_top + 0.1, '$V_G$', fontsize=16, ha='center', fontweight='bold')

    _pw = 0.6
    _ax2.plot([3 - _pw, 3 + _pw], [2.5, 2.5], 'k-', lw=3)
    _ax2.plot([3 - _pw, 3 + _pw], [2.2, 2.2], 'k-', lw=3)
    _ax2.text(3 + _pw + 0.15, 2.35, '$C_{poly}$', fontsize=16, va='center')
    _ax2.plot([3, 3], [2.2, 1.7], 'k-', lw=2)

    _ax2.plot([3 - _pw, 3 + _pw], [1.7, 1.7], 'k-', lw=3)
    _ax2.plot([3 - _pw, 3 + _pw], [1.4, 1.4], 'k-', lw=3)
    _ax2.text(3 + _pw + 0.15, 1.55, '$C_{ox}$', fontsize=16, va='center')
    _ax2.plot([3, 3], [1.4, _y_bot_c], 'k-', lw=2)

    _gw = 0.3
    for _dy in [0.0, -0.06, -0.12]:
        _ax2.plot([3 - _gw, 3 + _gw], [_y_bot_c + _dy, _y_bot_c + _dy], 'k-', lw=1.5)
        _gw -= 0.08

    _ax2.text(1.0, 2.0, r'$\dfrac{1}{C_{ox,eff}} = \dfrac{1}{C_{ox}} + \dfrac{1}{C_{poly}}$',
              fontsize=14, ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='orange'))

    _ax2.set_xlim(-0.5, 6)
    _ax2.set_ylim(0, _y_top + 0.5)
    _ax2.set_aspect('equal')
    _ax2.axis('off')
    _ax2.set_title('Equivalent Circuit', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.close(_fig)

    # ── Estimate of W_dep,poly ──
    _Vox = 1.0
    _tox = 2e-7
    _Npoly = 1e20
    _eps_s = eps_Si * eps_0
    _Cox = eps_ox * eps_0 / _tox

    _Wdep_poly = _Cox * _Vox / (q * _Npoly)
    _Cpoly = _eps_s / _Wdep_poly
    _Ceff = 1.0 / (1.0 / _Cox + 1.0 / _Cpoly)

    _estimate_md = mo.md(rf"""
    ### Estimate of $W_{{dep,poly}}$

    **Given:** $V_{{ox}} = {_Vox}$ V, $T_{{ox}} = {_tox*1e7:.0f}$ nm,
    N$^+$ poly gate with $N_{{poly}} = 10^{{20}}$ cm$^{{-3}}$,
    substrate doping $N_A = 10^{{17}}$ cm$^{{-3}}$.

    From the boundary condition at the gate–oxide interface ($\varepsilon_{{ox}}\mathcal{{E}}_{{ox}} = q N_{{poly}} W_{{dep,poly}}$):

    $$W_{{dep,poly}} = \frac{{C_{{ox}} V_{{ox}}}}{{q N_{{poly}}}}
    = \frac{{{_Cox:.3e} \times {_Vox}}}{{1.6 \times 10^{{-19}} \times 10^{{20}}}}
    = {_Wdep_poly*1e7:.2f} \text{{ nm}}$$

    | Parameter | Value |
    |:---|:---|
    | $W_{{dep,poly}}$ | {_Wdep_poly*1e7:.2f} nm |
    | $C_{{ox}}$ | {_Cox:.3e} F/cm$^2$ |
    | $C_{{poly}} = \varepsilon_s / W_{{dep,poly}}$ | {_Cpoly:.3e} F/cm$^2$ |
    | $C_{{ox,eff}}$ | {_Ceff:.3e} F/cm$^2$ |
    | $C_{{ox,eff}} / C_{{ox}}$ | {_Ceff/_Cox:.3f} |

    The effective capacitance is reduced to **{_Ceff/_Cox*100:.1f}%** of $C_{{ox}}$!

    """)

    mo.vstack([_poly_md, mo.as_html(_fig), _estimate_md])
    return


@app.cell(hide_code=True)
def _(eps_Si, eps_ox, mo, np, plt):
    _tinv_md = mo.md(rf"""
    ## 5. Inversion and Accumulation Charge Thickness

    So far, we assumed the inversion and accumulation charge layers in the semiconductor are
    **infinitely thin**. In reality, they have non-zero thickness $T_{{inv}}$.

    The effective oxide thickness is called the **electrical** or **equivalent oxide thickness**, $T_{{oxe}}$:

    $$\boxed{{T_{{oxe}} = T_{{ox}} + \frac{{\varepsilon_{{ox}}}}{{\varepsilon_s}}\left(W_{{dep,poly}} + T_{{inv}}\right)}}
    \qquad \frac{{\varepsilon_{{ox}}}}{{\varepsilon_s}} \approx \frac{{1}}{{3}}$$

    The inversion layer charge is then:

    $$Q_{{inv}} = -\frac{{\varepsilon_{{ox}}}}{{T_{{oxe}}}}(V_G - V_T)$$

    For thin oxides ($T_{{ox}} < 2$ nm), both poly-depletion and charge-layer thickness are
    significant corrections.
    """)

    # ── Schematic of n(x) distribution ──
    _fig, _ax = plt.subplots(figsize=(9, 5))

    _tox = 2.0
    _Wdep_poly = 1.0
    _x_gate_left = -5.0
    _x_si_right = 5.0

    _ax.axvspan(_x_gate_left, -_tox, color='#ffcc99', alpha=0.3, label='Gate')
    _ax.axvspan(-_tox, 0, color='#99ccff', alpha=0.3, label='Ox.')
    _ax.axvspan(0, _x_si_right, color='#ffcccc', alpha=0.15, label='Si')

    _ax.axvline(x=-_tox, color='k', lw=1.5)
    _ax.axvline(x=0, color='k', lw=1.5)

    _ax.text((_x_gate_left - _tox) / 2, 0.92, 'Gate', fontsize=16, ha='center',
             fontweight='bold', transform=_ax.get_xaxis_transform())
    _ax.text(-_tox / 2, 0.92, 'Ox.', fontsize=16, ha='center',
             fontweight='bold', transform=_ax.get_xaxis_transform())
    _ax.text(2.5, 0.92, 'Si', fontsize=16, ha='center',
             fontweight='bold', transform=_ax.get_xaxis_transform())

    _x_si = np.linspace(0, _x_si_right, 300)
    _x_peak = 1.0
    _sigma = 0.8
    _n_x = np.exp(-(_x_si - _x_peak) ** 2 / (2 * _sigma ** 2)) * _x_si / _x_peak
    _n_x[_x_si < 0.1] = 0
    _n_x = _n_x / np.max(_n_x)

    _ax.plot(_x_si, _n_x, 'b-', linewidth=2.5)
    _ax.fill_between(_x_si, 0, _n_x, alpha=0.15, color='blue')
    _ax.text(_x_peak + 0.3, 0.85, '$n(x)$', fontsize=16, color='blue')

    _y_ann = -0.12
    _ax.annotate('', xy=(0, _y_ann), xytext=(-_tox, _y_ann),
                 arrowprops=dict(arrowstyle='<->', color='darkorange', lw=1.5),
                 annotation_clip=False)
    _ax.text(-_tox / 2, _y_ann - 0.06, '$T_{ox}$', fontsize=16, ha='center',
             color='darkorange', transform=_ax.get_xaxis_transform())

    _ax.annotate('', xy=(-_tox, _y_ann), xytext=(-_tox - _Wdep_poly, _y_ann),
                 arrowprops=dict(arrowstyle='<->', color='brown', lw=1.5),
                 annotation_clip=False)
    _ax.text(-_tox - _Wdep_poly / 2, _y_ann - 0.06, 'Poly-Si\ndepl.', fontsize=12,
             ha='center', color='brown', transform=_ax.get_xaxis_transform())

    _inv_extent = 2.0
    _y_ann2 = -0.22
    _ax.annotate('', xy=(_inv_extent, _y_ann2), xytext=(0, _y_ann2),
                 arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5),
                 annotation_clip=False)
    _ax.text(_inv_extent / 2, _y_ann2 - 0.06, 'Effective\ninv. layer', fontsize=12,
             ha='center', color='purple', transform=_ax.get_xaxis_transform())

    _y_ann3 = -0.35
    _eff_left = -_tox - _Wdep_poly
    _eff_right = _inv_extent
    _ax.annotate('', xy=(_eff_right, _y_ann3), xytext=(_eff_left, _y_ann3),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2),
                 annotation_clip=False)
    _ax.text((_eff_left + _eff_right) / 2, _y_ann3 - 0.06, 'Effective $T_{ox}$',
             fontsize=16, ha='center', color='red', fontweight='bold',
             transform=_ax.get_xaxis_transform())

    _ax.set_xlabel(r'$x$ (nm)', fontsize=16)
    _ax.set_ylabel(r'$n(x)$ (arb. units)', fontsize=16)
    _ax.set_xlim(_x_gate_left, _x_si_right)
    _ax.set_ylim(-0.05, 1.15)
    _ax.tick_params(labelsize=16)
    _ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.close(_fig)

    _eox_es = eps_ox / eps_Si
    _caption = mo.md(rf"""
    The effective oxide thickness $T_{{oxe}}$ includes contributions from:
    physical oxide ($T_{{ox}}$), poly-Si gate depletion ($W_{{dep,poly}}$), and
    finite inversion layer thickness ($T_{{inv}}$), all scaled by
    $\varepsilon_{{ox}}/\varepsilon_s \approx {_eox_es:.2f} \approx 1/3$.
    """)

    mo.vstack([_tinv_md, mo.as_html(_fig), _caption])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Capacitance-Voltage Relation

    C-V measurements are used to characterize the MOS capacitor: capacitance, existence of non-idealities, surface charges, oxide thickness, dopant concentration, etc.

    The C-V curve has three distinct regions:

    - **Accumulation** ($V_G < V_{FB}$): $C = C_{ox}$ — oxide is the only capacitance
    - **Depletion** ($V_{FB} < V_G < V_T$): $C$ decreases as $W_{dep}$ grows — series combination of $C_{ox}$ and $C_{dep}$
    - **Inversion** ($V_G > V_T$):
      - *Low frequency:* $C \to C_{ox}$ — inversion charge responds to AC signal
      - *High frequency:* $C = C_{min}$ — inversion charge cannot follow; depletion width is frozen at $W_{dep,max}$

    ### How non-idealities shift the C-V

    The work function mismatch and oxide charges shift $V_{FB}$ away from zero. Since the entire C-V
    curve is referenced to $V_{FB}$, the curve translates **horizontally** by $V_{FB}$:

    $$V_{FB} = \Psi_M - \Psi_S - \frac{Q_{ox}}{C_{ox}}$$

    Consequently, $V_T$ also shifts:

    $$V_T = V_{FB} + 2\phi_B + \frac{Q_{dep,max}}{C_{ox}}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cv_Vfb_slider = mo.ui.slider(
        start=-2.0, stop=2.0, step=0.05, value=0.0,
        label=r"$V_{FB}$ (work fn.) [V]",
    )
    cv_tox_slider = mo.ui.slider(
        start=1, stop=20, step=0.5, value=5,
        label=r"$T_{ox}$ [nm]",
    )
    cv_Na_slider = mo.ui.slider(
        start=15, stop=18, step=0.1, value=17,
        label=r"log$_{10}$($N_A$) [cm$^{-3}$]",
    )
    cv_Qox_slider = mo.ui.slider(
        start=-5, stop=5, step=0.25, value=0,
        label=r"$Q_{ox}/q$ [$\times 10^{11}$ cm$^{-2}$]",
    )
    cv_Wdp_slider = mo.ui.slider(
        start=0, stop=5, step=0.1, value=0,
        label=r"$W_{dep,poly}$ [nm]",
    )
    cv_Tinv_slider = mo.ui.slider(
        start=0, stop=3, step=0.1, value=0,
        label=r"$T_{inv}$ [nm]",
    )
    return (
        cv_Na_slider,
        cv_Qox_slider,
        cv_Tinv_slider,
        cv_Vfb_slider,
        cv_Wdp_slider,
        cv_tox_slider,
    )


@app.cell(hide_code=True)
def _(
    cv_Na_slider,
    cv_Qox_slider,
    cv_Tinv_slider,
    cv_Vfb_slider,
    cv_Wdp_slider,
    cv_tox_slider,
    eps_0,
    eps_Si,
    eps_ox,
    kT_300,
    mo,
    ni_Si,
    np,
    plt,
    q,
):
    _Na = 10 ** cv_Na_slider.value
    _tox = cv_tox_slider.value * 1e-7          # cm
    _Vfb_wf = cv_Vfb_slider.value              # V
    _Qox = q * cv_Qox_slider.value * 1e11      # C/cm²
    _Wdp = cv_Wdp_slider.value * 1e-7          # cm
    _Tinv = cv_Tinv_slider.value * 1e-7         # cm

    _eps_s = eps_Si * eps_0
    _eox_es = eps_ox / eps_Si

    # ── Physical oxide capacitance ──
    _Cox = eps_ox * eps_0 / _tox

    # ── Flatband voltage ──
    _dVfb_Qox = -_Qox / _Cox
    _Vfb_total = _Vfb_wf + _dVfb_Qox

    # ── Effective oxide capacitances ──
    # Accumulation: no poly-depletion (gate has majority carriers)
    _Toxe_acc = _tox + _eox_es * _Tinv
    _Cox_eff_acc = eps_ox * eps_0 / _Toxe_acc

    # Depletion & inversion: poly-depletion + charge-layer thickness
    _Toxe_inv = _tox + _eox_es * (_Wdp + _Tinv)
    _Cox_eff_inv = eps_ox * eps_0 / _Toxe_inv

    # Poly capacitance (for display)
    _Cpoly = _eps_s / _Wdp if _Wdp > 1e-10 else float('inf')

    # ── Semiconductor parameters ──
    _phi_B = kT_300 * np.log(_Na / ni_Si)
    _phi_st = 2 * _phi_B
    _Qdep_max = np.sqrt(2 * q * _eps_s * _Na * _phi_st)
    _Cdep_min = np.sqrt(q * _eps_s * _Na / (2 * _phi_st))

    _gamma_ideal = np.sqrt(2 * q * _eps_s * _Na) / _Cox
    _gamma_eff = np.sqrt(2 * q * _eps_s * _Na) / _Cox_eff_inv

    # ── Threshold voltages ──
    _Vt_ideal = _phi_st + _Qdep_max / _Cox
    _Vt_nonideal = _Vfb_total + _phi_st + _Qdep_max / _Cox_eff_inv

    # ── C-V solver ──
    def _solve_hf(Vg_arr, Vfb, Cox_acc, Cox_inv, gamma, Cox_norm):
        C = np.zeros_like(Vg_arr)
        Vt = Vfb + _phi_st + _Qdep_max / Cox_inv
        for i, Vg in enumerate(Vg_arr):
            Vg_eff = Vg - Vfb
            if Vg_eff < 0:
                C[i] = Cox_acc / Cox_norm
            elif Vg < Vt:
                ps = 0.01
                for _ in range(80):
                    if ps <= 0:
                        ps = 0.01
                    f = ps + gamma * np.sqrt(ps) - Vg_eff
                    df = 1 + gamma / (2 * np.sqrt(ps))
                    ps -= f / df
                    if abs(f) < 1e-8:
                        break
                ps = max(ps, 1e-10)
                Cd = np.sqrt(q * _eps_s * _Na / (2 * ps))
                C[i] = (1.0 / (1.0 / Cox_inv + 1.0 / Cd)) / Cox_norm
            else:
                C[i] = (1.0 / (1.0 / Cox_inv + 1.0 / _Cdep_min)) / Cox_norm
        return C

    def _solve_lf(Vg_arr, Vfb, Cox_acc, Cox_inv, gamma, Cox_norm):
        hf = _solve_hf(Vg_arr, Vfb, Cox_acc, Cox_inv, gamma, Cox_norm)
        lf = hf.copy()
        Vt = Vfb + _phi_st + _Qdep_max / Cox_inv
        Cmin_r = (1.0 / (1.0 / Cox_inv + 1.0 / _Cdep_min)) / Cox_norm
        Cox_r = Cox_inv / Cox_norm
        inv_mask = Vg_arr > Vt
        dv = np.maximum(Vg_arr[inv_mask] - Vt, 0)
        lf[inv_mask] = Cmin_r + (Cox_r - Cmin_r) * (1 - np.exp(-dv / 0.08))
        return lf

    _Vg = np.linspace(-4.0, 5.0, 800)

    _hf_ideal = _solve_hf(_Vg, 0.0, _Cox, _Cox, _gamma_ideal, _Cox)
    _lf_ideal = _solve_lf(_Vg, 0.0, _Cox, _Cox, _gamma_ideal, _Cox)
    _hf_real = _solve_hf(_Vg, _Vfb_total, _Cox_eff_acc, _Cox_eff_inv, _gamma_eff, _Cox)
    _lf_real = _solve_lf(_Vg, _Vfb_total, _Cox_eff_acc, _Cox_eff_inv, _gamma_eff, _Cox)

    # ── Plot ──
    _fig, _ax = plt.subplots(figsize=(10, 6))

    _ax.plot(_Vg, _hf_ideal, 'b--', linewidth=1.5, alpha=0.4, label='Ideal HF')
    _ax.plot(_Vg, _lf_ideal, 'r--', linewidth=1.5, alpha=0.3, label='Ideal LF')
    _ax.plot(_Vg, _hf_real, 'b-', linewidth=2.5, label='Non-ideal HF')
    _ax.plot(_Vg, _lf_real, 'r-', linewidth=2.5, label='Non-ideal LF')

    _ax.axhline(y=1.0, color='gray', ls='--', lw=1, alpha=0.5)
    _ax.text(-3.8, 1.02, '$C_{ox}$', fontsize=16, color='gray')

    # C_max annotations when reduced by non-idealities
    _Cmax_acc = _Cox_eff_acc / _Cox
    _Cmax_inv_lf = _Cox_eff_inv / _Cox
    if _Cmax_acc < 0.995:
        _ax.axhline(y=_Cmax_acc, color='darkorange', ls=':', lw=1.5, alpha=0.6)
        _ax.text(4.2, _Cmax_acc + 0.02, f'$C_{{max,acc}}$', fontsize=13,
                 color='darkorange', va='bottom')
    if _Cmax_inv_lf < _Cmax_acc - 0.005:
        _ax.axhline(y=_Cmax_inv_lf, color='purple', ls=':', lw=1.5, alpha=0.6)
        _ax.text(4.2, _Cmax_inv_lf - 0.02, f'$C_{{max,inv}}$', fontsize=13,
                 color='purple', va='top')

    # V_fb markers
    _ax.axvline(x=0, color='blue', ls=':', lw=1, alpha=0.25)
    _ax.axvline(x=_Vfb_total, color='blue', ls=':', lw=1.2, alpha=0.6)
    _ax.text(0, -0.07, '$V_{fb}^{\\mathrm{ideal}}$', fontsize=13, ha='center',
             color='blue', alpha=0.45, transform=_ax.get_xaxis_transform())
    _ax.text(_Vfb_total, -0.07, '$V_{fb}$', fontsize=16, ha='center',
             color='blue', transform=_ax.get_xaxis_transform())

    # V_t markers
    _ax.axvline(x=_Vt_ideal, color='red', ls=':', lw=1, alpha=0.25)
    _ax.axvline(x=_Vt_nonideal, color='red', ls=':', lw=1.2, alpha=0.6)
    _ax.text(_Vt_ideal, -0.07, '$V_t^{\\mathrm{ideal}}$', fontsize=13, ha='center',
             color='red', alpha=0.45, transform=_ax.get_xaxis_transform())
    _ax.text(_Vt_nonideal, -0.07, '$V_t$', fontsize=16, ha='center',
             color='red', transform=_ax.get_xaxis_transform())

    # V_FB shift arrow
    if abs(_Vfb_total) > 0.05:
        _ax.annotate('', xy=(_Vfb_total, 0.93), xytext=(0, 0.93),
                     arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        _ax.text(_Vfb_total / 2, 0.96, f'$V_{{FB}}$ = {_Vfb_total:+.2f} V',
                 ha='center', fontsize=14, color='green', fontweight='bold')

    # Region labels
    _mid_depl = (_Vfb_total + _Vt_nonideal) / 2
    _ax.text(_Vfb_total - 1.0, 0.55, 'acc.', fontsize=16, ha='center', color='#555555')
    _ax.text(_mid_depl, 0.18, 'depletion', fontsize=16, ha='center', color='#555555')
    _ax.text(_Vt_nonideal + 1.0, 0.55, 'inv.', fontsize=16, ha='center', color='#555555')

    _ax.set_xlabel(r'$V_G$ (V)', fontsize=16)
    _ax.set_ylabel(r'$C / C_{ox}$', fontsize=16)
    _ax.set_title('C-V with Non-Idealities (P-type substrate)', fontsize=16, fontweight='bold')
    _ax.legend(fontsize=13, loc='center right')
    _ax.tick_params(labelsize=16)
    _ax.set_ylim(0, 1.15)
    _ax.set_xlim(-4.0, 5.0)
    _ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.close(_fig)

    # ── Computed parameters table ──
    _Cpoly_str = f"{_Cpoly:.3e}" if np.isfinite(_Cpoly) else r"$\infty$ (no poly-depl.)"
    _Cox_eff_ratio = _Cox_eff_inv / _Cox

    _info = mo.md(rf"""
    | Parameter | Value | | Parameter | Value |
    |:---|:---|---|:---|:---|
    | $N_A$ | {_Na:.2e} cm$^{{-3}}$ | | $\phi_B$ | {_phi_B:.4f} V |
    | $T_{{ox}}$ | {cv_tox_slider.value} nm | | $C_{{ox}} = \varepsilon_{{ox}}/T_{{ox}}$ | {_Cox:.3e} F/cm² |
    | $V_{{FB}}$ (work fn.) | {_Vfb_wf:+.3f} V | | $\Delta V_{{FB}}$ ($Q_{{ox}}$) | {_dVfb_Qox:+.3f} V |
    | **$V_{{FB,total}}$** | **{_Vfb_total:+.3f} V** | | **$V_T$ (ideal)** | **{_Vt_ideal:.3f} V** |
    | $C_{{poly}} = \varepsilon_s / W_{{dep,poly}}$ | {_Cpoly_str} | | **$V_T$ (non-ideal)** | **{_Vt_nonideal:.3f} V** |
    | $C_{{ox,eff}}$ (inv.) | {_Cox_eff_inv:.3e} F/cm² | | $C_{{ox,eff}}/C_{{ox}}$ | {_Cox_eff_ratio:.4f} |
    | $T_{{oxe}}$ (acc.) | {_Toxe_acc*1e7:.2f} nm | | $T_{{oxe}}$ (inv.) | {_Toxe_inv*1e7:.2f} nm |
    """)

    _guide = mo.md(r"""
    **How each slider affects the C-V:**

    - **$V_{FB}$** and **$Q_{ox}$** shift the curve **horizontally** → $V_{fb}$ and $V_T$ move together
    - **$W_{dep,poly}$** reduces $C_{max}$ in **inversion** (poly-gate adds series capacitance $C_{poly}$)
    - **$T_{inv}$** reduces $C_{max}$ in **both accumulation and inversion** (charge layer adds to $T_{oxe}$)
    - **$T_{ox}$** sets the baseline $C_{ox}$; thinner oxides make poly-depletion and $T_{inv}$ effects more significant
    """)

    _controls = mo.vstack([
        mo.hstack([cv_Vfb_slider, cv_tox_slider, cv_Na_slider], justify="center"),
        mo.hstack([cv_Qox_slider, cv_Wdp_slider, cv_Tinv_slider], justify="center"),
    ])

    _header = mo.md(r"### Interactive C-V Relation with Non-Idealities")

    mo.vstack([_header, _controls, mo.as_html(_fig), _info, _guide])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Extracting Non-Idealities from C-V

    In the previous notebook (`mos_cv`), we extracted three **ideal** parameters from the C-V curve:

    | Parameter | How to extract (ideal MOS) |
    |:---|:---|
    | $t_{ox}$ | $C_{ox} = \varepsilon_{ox}/t_{ox}$ from $C_{max}$ in accumulation |
    | $N_A$ | Slope of $1/C^2$ vs. $V_G$ in the depletion region |
    | $V_T$ | Onset of strong inversion (where HF C-V reaches $C_{min}$) |

    With non-idealities, the measured C-V differs from the ideal curve. Here is how to extract each non-ideality.

    ---

    ### Step 1: Extract $C_{ox}$ and $t_{ox}$

    In the **ideal** case, $C_{max} = C_{ox}$ in accumulation. With non-idealities:

    - If $T_{inv} > 0$: $C_{max,acc} < C_{ox}$ because the accumulation charge layer adds to the effective oxide thickness.
    - If the gate is metal (no poly-depletion), the accumulation $C_{max}$ gives $C_{ox,eff}^{acc} = \varepsilon_{ox}/T_{oxe,acc}$, not the true $C_{ox}$.

    **In practice**, $t_{ox}$ is determined from ellipsometry (optical measurement) or from $C_{ox}$ measured on a **large-area** test capacitor where fringe effects are negligible and the accumulation charge layer correction is applied:

    $$t_{ox} = \frac{\varepsilon_{ox}}{C_{ox}}, \qquad C_{ox} = C_{max,acc} \text{ (if metal gate, thin corrections applied)}$$

    ---

    ### Step 2: Extract $N_A$ from the $1/C^2$ slope

    The slope of $1/C^2$ vs. $V_G$ in the **depletion** region is unaffected by $V_{FB}$, $Q_{ox}$, or poly-depletion (the depletion capacitance depends only on $N_A$ and $\phi_s$):

    $$\frac{d(1/C^2)}{dV_G} = \frac{2}{q\,\varepsilon_s\,N_A} \qquad \Rightarrow \qquad N_A = \frac{2}{q\,\varepsilon_s \cdot \text{slope}}$$

    This is the same method as in the ideal case (and analogous to PN junction and Schottky $1/C^2$ analysis).

    ---

    ### Step 3: Extract $V_{FB}$ from the flatband capacitance

    Once $t_{ox}$ and $N_A$ are known, compute the **flatband capacitance**:

    $$C_{FB} = \frac{C_{ox}\,C_{s,FB}}{C_{ox}+C_{s,FB}}, \qquad C_{s,FB} = \frac{\varepsilon_s}{L_D}, \qquad L_D = \sqrt{\frac{\varepsilon_s\,kT}{q^2\,N_A}}$$

    $V_{FB}$ is the gate voltage at which $C = C_{FB}$ on the measured HF C-V curve. This reveals the **total** flatband shift:

    $$V_{FB} = \Psi_M - \Psi_S - \frac{Q_{ox}}{C_{ox}}$$

    If the gate work function $\Psi_M$ is known, $Q_{ox}$ can be separated:

    $$Q_{ox} = -C_{ox}\left(V_{FB} - (\Psi_M - \Psi_S)\right)$$

    ---

    ### Step 4: Identify poly-Si gate depletion

    Compare $C_{max}$ in **accumulation** vs. $C_{max}$ in **LF inversion**:

    - **Ideal or metal gate:** both equal $C_{ox}$ (or both reduced equally by $T_{inv}$)
    - **Poly-Si gate:** $C_{max,inv} < C_{max,acc}$ because the poly-Si gate is depleted in inversion but not in accumulation

    The difference isolates $C_{poly}$:

    $$\frac{1}{C_{poly}} = \frac{1}{C_{max,inv}} - \frac{1}{C_{max,acc}} \qquad \Rightarrow \qquad W_{dep,poly} = \frac{\varepsilon_s}{C_{poly}}$$

    ---

    ### Step 5: Identify charge-layer thickness ($T_{inv}$)

    If $C_{max,acc} < C_{ox}$ (even after accounting for poly-depletion), the remaining reduction is due to the finite accumulation charge thickness:

    $$T_{inv} \approx \frac{\varepsilon_s}{\varepsilon_{ox}}\left(\frac{\varepsilon_{ox}}{C_{max,acc}} - t_{ox}\right)$$

    ---

    ### Step 6: Detect interface traps ($Q_{it}$)

    Compare the HF and LF C-V curves in the **depletion-to-inversion transition**:

    - **Without traps:** the HF and LF curves separate only in the inversion region (minority carrier response)
    - **With traps ($Q_{it}$):** the transition region is **stretched out** — the C-V curve changes more gradually from accumulation to inversion because traps exchange charge with the bands

    Interface traps act like additional capacitance $C_{it}$ in parallel with $C_s$. At low frequency, traps can follow the signal and contribute to the measured capacitance; at high frequency, they cannot. The difference between the LF and HF capacitance in the depletion region reveals $C_{it}$, and thus $D_{it} = C_{it}/q$.

    ---

    ### Extraction summary

    | What to extract | Where on the C-V | Method |
    |:---|:---|:---|
    | $t_{ox}$ | Accumulation $C_{max}$ | $t_{ox} = \varepsilon_{ox}/C_{ox}$ (with corrections) |
    | $N_A$ | Depletion $1/C^2$ slope | $N_A = 2/(q\varepsilon_s \cdot \text{slope})$ |
    | $V_{FB}$ | $V_G$ at $C = C_{FB}$ | Compute $C_{FB}$ from $t_{ox}$ and $N_A$ |
    | $Q_{ox}$ | $V_{FB}$ shift from expected $\Psi_M - \Psi_S$ | $Q_{ox} = -C_{ox}(V_{FB} - \Psi_M + \Psi_S)$ |
    | Poly-depletion | $C_{max,inv} < C_{max,acc}$ | $1/C_{poly} = 1/C_{max,inv} - 1/C_{max,acc}$ |
    | $T_{inv}$ | $C_{max,acc} < C_{ox}$ | Back-calculate from $T_{oxe,acc}$ |
    | $D_{it}$ | Stretch-out in depletion | $C_{it}$ from LF$-$HF difference; $D_{it} = C_{it}/q$ |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Non-ideality | Effect on $V_{FB}$ | Effect on C-V |
    |:---|:---|:---|
    | **Work function mismatch** ($\Psi_M \neq \Psi_S$) | $V_{FB} = \Psi_M - \Psi_S$ | Horizontal shift of entire curve |
    | **Oxide charge** ($Q_{ox}$) | $\Delta V_{FB} = -Q_{ox}/C_{ox}$ | Additional horizontal shift |
    | **Poly-gate depletion** | — | Reduces $C_{max}$ below $C_{ox}$ in inversion |
    | **Charge layer thickness** ($T_{inv}$) | — | Reduces $C_{max}$ in both acc. and inversion |
    | **Interface traps** ($Q_{it}$) | Frequency-dependent | **Stretch-out** of C-V transition |

    ### Modified threshold voltage

    $$V_T = \underbrace{(\Psi_M - \Psi_S) - \frac{Q_{ox}}{C_{ox}}}_{V_{FB}} + 2\phi_B + \frac{2}{C_{ox}}\sqrt{q N_A \varepsilon_s \phi_B}$$
    """)
    return


if __name__ == "__main__":
    app.run()
