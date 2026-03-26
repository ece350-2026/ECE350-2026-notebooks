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
app = marimo.App(width="medium", layout_file=None)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path

    q = 1.6e-19
    k = 1.381e-23
    kT_300 = 0.02585  # eV
    eps_0 = 8.854e-14  # F/cm
    eps_Si = 11.7
    eps_ox = 3.9
    ni_Si = 1.1e10  # cm^-3

    try:
        _test = Path(__file__).parent / "images"
        if _test.exists():
            ASSET_DIR = Path(__file__).parent
        else:
            raise FileNotFoundError
    except Exception:
        ASSET_DIR = None
    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/mosfet-intro/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    mo.md(r"""
    # MOSFET: Qualitative Overview of Operation
    **ECE350 Lecture 33**

    Hu, Chapter 6

    This notebook provides a qualitative overview of the Metal-Oxide-Semiconductor Field-Effect Transistor (MOSFET). We explore how the gate voltage $V_{GS}$ and drain voltage $V_{DS}$ control the charge distribution in the channel and, ultimately, the drain current $I_{DS}$.

    **Contents:**

    1. **MOSFET Structure**: NFET (NMOS) and PFET (PMOS), voltages, and symbols
    2. **CMOS**: complementary logic circuit
    3. **Interactive Explorer**: unified visualization of charge profile, channel, and IV characteristics
    4. **$I_{DS}$–$V_{GS}$ and Threshold Voltage**
    5. **Summary**
    """)
    return IMAGE_BASE, eps_0, eps_Si, kT_300, mo, mpatches, ni_Si, np, plt, q


@app.cell(hide_code=True)
def _(mo, mpatches, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    def _draw_mosfet(ax, ptype_body=True, title=""):
        body_color = '#ffe0e0' if ptype_body else '#e0e0ff'
        sd_color = '#4040ff' if ptype_body else '#ff4040'
        sd_label = 'n⁺' if ptype_body else 'p⁺'
        body_label = 'P-body' if ptype_body else 'N-body'

        body = mpatches.FancyBboxPatch((0.5, 0.0), 4.0, 2.0, boxstyle="round,pad=0.05",
                                        facecolor=body_color, edgecolor='black', linewidth=1.5)
        ax.add_patch(body)

        for xpos in [0.8, 3.5]:
            sd = mpatches.FancyBboxPatch((xpos, 1.0), 0.8, 1.0, boxstyle="round,pad=0.02",
                                          facecolor=sd_color, edgecolor='black', linewidth=1)
            ax.add_patch(sd)

        oxide = mpatches.FancyBboxPatch((1.8, 2.0), 1.5, 0.3, boxstyle="round,pad=0.02",
                                         facecolor='#ffffcc', edgecolor='black', linewidth=1)
        ax.add_patch(oxide)
        gate = mpatches.FancyBboxPatch((1.8, 2.3), 1.5, 0.5, boxstyle="round,pad=0.02",
                                        facecolor='#a0a0a0', edgecolor='black', linewidth=1.5)
        ax.add_patch(gate)

        if ptype_body:
            ax.plot([1.6, 3.3], [2.0, 2.0], color='cyan', linewidth=3, alpha=0.6)

        ax.text(1.2, 1.4, sd_label, fontsize=14, ha='center', va='center', color='white', fontweight='bold')
        ax.text(3.9, 1.4, sd_label, fontsize=14, ha='center', va='center', color='white', fontweight='bold')
        ax.text(2.55, 0.6, body_label, fontsize=14, ha='center', va='center')
        ax.text(2.55, 2.12, 'oxide', fontsize=10, ha='center', va='center', style='italic')
        ax.text(2.55, 2.55, 'Gate', fontsize=14, ha='center', va='center', fontweight='bold')

        ax.annotate('S', xy=(1.2, 2.0), xytext=(1.2, 3.0),
                    fontsize=16, ha='center', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate('D', xy=(3.9, 2.0), xytext=(3.9, 3.0),
                    fontsize=16, ha='center', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate('G', xy=(2.55, 2.8), xytext=(2.55, 3.4),
                    fontsize=16, ha='center', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate('B', xy=(2.55, 0.0), xytext=(2.55, -0.7),
                    fontsize=16, ha='center', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=1.5))

        ax.set_xlim(0, 5.1)
        ax.set_ylim(-1.0, 3.8)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')

    _draw_mosfet(_ax1, ptype_body=True, title="N-Channel MOSFET (NMOS)")
    _draw_mosfet(_ax2, ptype_body=False, title="P-Channel MOSFET (PMOS)")
    plt.tight_layout()
    plt.close(_fig)

    _header = mo.md(r"""
    ## 1. MOSFET Structure

    The MOSFET is built from **two pn junctions** and **one MOS capacitor**. It has four terminals:

    | Terminal | Symbol | Description |
    |----------|--------|-------------|
    | **Gate** | G | Controls the channel via the electric field through the oxide |
    | **Source** | S | Reference terminal (taken at 0 V for NFET) |
    | **Drain** | D | Collects the channel current |
    | **Body** | B | Substrate contact (often tied to source or ground) |

    ### N-Channel MOSFET (NFET / NMOS)
    - **P-type body** with **n⁺ source** and **n⁺ drain**; electrons flow S → D when $V_{GS} > V_t$; conventional current $I_{DS}$ flows D → S

    ### P-Channel MOSFET (PFET / PMOS)
    - **N-type body** with **p⁺ source** and **p⁺ drain**; holes flow S → D when $V_{GS} < V_t$ (negative $V_t$)

    **Key voltages:** $V_{GS}$ (gate-source), $V_{DS}$ (drain-source), $V_{DD}$ (supply).
    The **cyan line** in the NMOS diagram represents the inversion layer (channel) that forms when $V_{GS} > V_t$.
    """)

    mo.vstack([_header, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _cmos_img = mo.image(f"{IMAGE_BASE}/CMOS.png", width=200)
    _circuit_img = mo.image(f"{IMAGE_BASE}/CMOS-circuit.png", width=500)

    mo.vstack([
        mo.md(r"""
        ## 2. CMOS Inverter

        **Complementary MOS (CMOS):** Place a PFET and NFET in series between $V_{DD}$ and ground as switch. Only one transistor is on at a time, so there is no direct current path from $V_{DD}$ to ground. This leads to **low static power dissipation**!

        | Input $V_{in}$ | NFET | PFET | Output $V_{out}$ |
        |:---:|:---:|:---:|:---:|
        | $0$ V | OFF | **ON** | $\approx V_{DD}$ |
        | $V_{DD}$ | **ON** | OFF | $\approx 0$ V |

        The CMOS inverter is the fundamental building block of all modern digital logic.
        """),
        mo.hstack([_cmos_img, _circuit_img], justify="center"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Interactive MOSFET Explorer

    Use the sliders below to set $V_{GS}$ and $V_{DS}$ and see how the MOSFET responds:

    - **Top-left:** Cross-section showing the charge state under the gate (accumulation / depletion / inversion) and how the channel tapers with $V_{DS}$
    - **Top-right:** $I_{DS}$–$V_{DS}$ family of curves; the **red dot** marks the current operating point
    - **Bottom:** $Q_{inv}(x)$ along the channel from source to drain

    ### Physics Summary

    The inversion charge at position $x$ along the channel is:
    $$Q_{inv}(x) = -C_{oxe}\big[V_{GS} - V_t - V_{CS}(x)\big]$$

    where $V_{CS}(x)$ is the channel-to-source voltage (0 at source, $V_{DS}$ at drain). This gives:

    - **Linear / Triode** ($V_{DS} < V_{Dsat}$): channel tapers gradually toward the drain
    - **Saturation** ($V_{DS} \geq V_{Dsat} = V_{GS} - V_t$): channel is pinched off at drain end ($Q_{inv}(L) \to 0$), and $I_{DS}$ becomes ideally constant
    """)
    return


@app.cell
def _(mo):
    Vgs_slider = mo.ui.slider(start=-0.5, stop=3.0, value=1.5, step=0.05,
                               label=r"$V_{GS}$ (V)")
    Vds_slider = mo.ui.slider(start=0.0, stop=3.0, value=0.5, step=0.05,
                               label=r"$V_{DS}$ (V)")
    return Vds_slider, Vgs_slider


@app.cell(hide_code=True)
def _(Vds_slider, Vgs_slider, eps_0, eps_Si, kT_300, mo, ni_Si, np, plt, q):
    _Vgs = Vgs_slider.value
    _Vds = Vds_slider.value

    _Na = 1e17
    _tox = 5e-7  # 5 nm
    _Cox = 3.9 * eps_0 / _tox
    _phi_B = kT_300 * np.log(_Na / ni_Si)
    _Vfb = 0.0
    _Vt = _Vfb + 2 * _phi_B + np.sqrt(2 * eps_Si * eps_0 * q * _Na * 2 * _phi_B) / _Cox
    _Wdep_max = np.sqrt(2 * eps_Si * eps_0 * 2 * _phi_B / (q * _Na))
    _mu_ns = 300
    _W_over_L = 10.0
    _Vov = _Vgs - _Vt
    _Vdsat = max(_Vov, 0.001)

    if _Vgs < _Vfb:
        _regime = "Accumulation"
        _regime_color = '#cc3333'
    elif _Vgs < _Vt:
        _regime = "Depletion"
        _regime_color = '#cc8800'
    elif _Vds < _Vdsat:
        _regime = "Triode"
        _regime_color = '#2266cc'
    else:
        _regime = "Saturation"
        _regime_color = '#22aa44'

    if _Vgs <= _Vt:
        _Ids_op = 0.0
    elif _Vds < _Vdsat:
        _Ids_op = _mu_ns * _Cox * _W_over_L * (_Vov * _Vds - 0.5 * _Vds**2)
    else:
        _Ids_op = 0.5 * _mu_ns * _Cox * _W_over_L * _Vov**2
    _Ids_op = max(_Ids_op, 0)

    _x_norm = np.linspace(0, 1, 300)
    if _Vgs <= _Vt:
        _Qinv_norm = np.zeros_like(_x_norm)
    elif _Vds < _Vdsat:
        _Vcs = _Vds * _x_norm
        _Qinv_norm = np.maximum(1 - _Vcs / _Vov, 0) if _Vov > 0 else np.zeros_like(_x_norm)
    else:
        _Vcs = np.minimum(_Vdsat * _x_norm, _Vdsat)
        _Qinv_norm = np.maximum(1 - _Vcs / _Vov, 0) if _Vov > 0 else np.zeros_like(_x_norm)

    # ===================== FIGURE =====================
    _fig = plt.figure(figsize=(14, 10))
    _gs = _fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.35, wspace=0.3)
    _ax_xs = _fig.add_subplot(_gs[0, 0])
    _ax_iv = _fig.add_subplot(_gs[0, 1])
    _ax_ch = _fig.add_subplot(_gs[1, :])

    # -------- Top-left: MOSFET cross-section --------
    _ax = _ax_xs
    _ax.set_xlim(-0.3, 5.3)
    _ax.set_ylim(-0.2, 3.2)
    _ax.set_aspect('equal')
    _ax.axis('off')

    _ax.add_patch(plt.Rectangle((0, 0), 5, 1.8, facecolor='#ffe0e0', edgecolor='black', lw=1.5))
    _ax.text(2.5, 0.25, r'P-body ($N_A=10^{17}$)', fontsize=14, ha='center', color='#884444')

    _ax.add_patch(plt.Rectangle((0.5, 1.8), 4.0, 0.2, facecolor='#ffffcc', edgecolor='black', lw=1))
    _ax.add_patch(plt.Rectangle((0.5, 2.0), 4.0, 0.5, facecolor='#a0a0a0', edgecolor='black', lw=1.5))
    _ax.text(2.5, 1.88, 'SiO₂', fontsize=11, ha='center', va='center', color='#666600')
    _ax.text(2.5, 2.25, 'Gate', fontsize=16, ha='center', va='center', fontweight='bold')

    _ax.add_patch(plt.Rectangle((0, 0.9), 0.5, 0.9, facecolor='#4040ff', edgecolor='black', lw=1, zorder=3))
    _ax.add_patch(plt.Rectangle((4.5, 0.9), 0.5, 0.9, facecolor='#4040ff', edgecolor='black', lw=1, zorder=3))
    _ax.text(0.25, 1.3, 'n⁺', fontsize=14, ha='center', color='white', fontweight='bold')
    _ax.text(4.75, 1.3, 'n⁺', fontsize=14, ha='center', color='white', fontweight='bold')
    _ax.text(0.25, 2.65, 'S', fontsize=16, ha='center', fontweight='bold')
    _ax.text(4.75, 2.65, 'D', fontsize=16, ha='center', fontweight='bold')

    # --- Channel and depletion region visualization ---
    _ch_x = np.linspace(0.5, 4.5, 300)
    _ch_top = np.ones_like(_ch_x) * 1.8
    _dep_max_draw = 0.7
    _inv_draw = 0.12

    if _Vgs > _Vt:
        _dep_h_source = _dep_max_draw * 0.5
        # Exaggerate the Wdep widening toward drain for visual clarity
        _Vcb_drain = min(_Vds, _Vdsat)
        _dep_ratio = 1.0 + 2.5 * _Vcb_drain / max(_Vov, 0.01)
        _dep_h_drain = min(_dep_h_source * _dep_ratio, 1.5)
        _t = ((_ch_x - 0.5) / 4.0) ** 1.3  # slightly superlinear for visual punch
        _dep_h_profile = _dep_h_source + (_dep_h_drain - _dep_h_source) * _t

        # Inversion layer height along channel
        _ch_height = _Qinv_norm * _inv_draw
        _ch_bot = 1.8 - _ch_height

        if _regime == "Saturation":
            # In saturation the channel is pinched off near drain.
            # Where Qinv → 0, the depletion region extends all the way to the oxide.
            _dep_top = 1.8 - _ch_height  # depletion top = oxide minus inversion layer
            _dep_bot = _dep_top - _dep_h_profile
        else:
            _dep_top = np.ones_like(_ch_x) * (1.8 - _inv_draw)
            _dep_bot = _dep_top - _dep_h_profile

        # Depletion region fill (under gate)
        _ax.fill_between(_ch_x, _dep_bot, _dep_top,
                         color='#ccccff', alpha=0.4, zorder=1)
        _ax.plot(_ch_x, _dep_bot, color='#7777bb', lw=1.2, ls='--', alpha=0.7, zorder=1)

        # Extend depletion region under the n+ drain junction
        _drain_ext_x = np.linspace(4.5, 5.0, 30)
        _drain_dep_top = np.ones_like(_drain_ext_x) * 0.9  # bottom of n+ drain
        _drain_dep_bot_at_edge = _dep_bot[-1]
        _drain_dep_bot = _drain_dep_bot_at_edge + (0.9 - _drain_dep_bot_at_edge) * ((_drain_ext_x - 4.5) / 0.5) ** 2
        _ax.fill_between(_drain_ext_x, _drain_dep_bot, _drain_dep_top,
                         color='#ccccff', alpha=0.4, zorder=1)
        _ax.plot(_drain_ext_x, _drain_dep_bot, color='#7777bb', lw=1.2, ls='--', alpha=0.7, zorder=1)

        # Inversion layer
        _ax.fill_between(_ch_x, _ch_bot, _ch_top, color='cyan', alpha=0.65, zorder=2)

        _mid_dep = 1.8 - _inv_draw - _dep_h_source * 0.5
        _ax.text(1.2, _mid_dep, '$W_{dep}$', fontsize=14, ha='center', va='center',
                 color='#555599', fontstyle='italic')
        if _regime == "Saturation":
            _ax.annotate('pinch-off', xy=(4.3, 1.75), xytext=(3.6, 0.4),
                         fontsize=14, ha='center', color='red', fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    elif _Vgs > _Vfb:
        _dep_frac = min(1.0, max(0.05, (_Vgs - _Vfb) / max(_Vt - _Vfb, 0.01)))
        _dep_h = _dep_frac * _dep_max_draw
        _ax.add_patch(plt.Rectangle((0.5, 1.8 - _dep_h), 4.0, _dep_h,
                                     facecolor='#ccccff', edgecolor='none', alpha=0.45))
        _ax.plot([0.5, 4.5], [1.8 - _dep_h, 1.8 - _dep_h],
                 color='#7777bb', lw=1.2, ls='--', alpha=0.7)
        _ax.text(2.5, 1.8 - _dep_h / 2, 'depletion', fontsize=14, ha='center', va='center',
                 color='#555599', alpha=0.8)
    else:
        _ax.add_patch(plt.Rectangle((0.5, 1.65), 4.0, 0.15,
                                     facecolor='#ff6666', edgecolor='none', alpha=0.5))
        _ax.text(2.5, 1.72, 'holes (accumulation)', fontsize=14, ha='center', va='center')

    _ax.text(2.5, 3.0, f'{_regime}', fontsize=16, ha='center', fontweight='bold',
             color=_regime_color,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor=_regime_color, lw=1.5))

    _ax.set_title('NMOS Cross-Section', fontsize=16, fontweight='bold')

    # -------- Top-right: IV family + operating point --------
    _ax = _ax_iv
    _Vds_arr = np.linspace(0, 3.5, 400)
    _vgs_family = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0]
    _colors_fam = plt.cm.Blues(np.linspace(0.3, 0.85, len(_vgs_family)))

    for _i, _vgs_f in enumerate(_vgs_family):
        _vov_f = _vgs_f - _Vt
        if _vov_f <= 0:
            continue
        _vdsat_f = _vov_f
        _ids_f = np.where(
            _Vds_arr < _vdsat_f,
            _mu_ns * _Cox * _W_over_L * (_vov_f * _Vds_arr - 0.5 * _Vds_arr**2),
            0.5 * _mu_ns * _Cox * _W_over_L * _vov_f**2
        )
        _ids_f = np.maximum(_ids_f, 0)
        _ax.plot(_Vds_arr, _ids_f * 1e3, color=_colors_fam[_i], lw=1.2, alpha=0.6,
                 label=f'{_vgs_f:.1f}' if _i == 0 or _i == len(_vgs_family)-1 else None)
        _ax.text(3.55, _ids_f[-1] * 1e3, f'{_vgs_f:.1f}', fontsize=10,
                 color=_colors_fam[_i], va='center', clip_on=True)

    if _Vov > 0:
        _ids_sel = np.where(
            _Vds_arr < _Vdsat,
            _mu_ns * _Cox * _W_over_L * (_Vov * _Vds_arr - 0.5 * _Vds_arr**2),
            0.5 * _mu_ns * _Cox * _W_over_L * _Vov**2
        )
        _ids_sel = np.maximum(_ids_sel, 0)
        _ax.plot(_Vds_arr, _ids_sel * 1e3, 'k-', lw=2.5, zorder=3)
        _ax.axvline(_Vdsat, color='gray', ls=':', lw=1, alpha=0.4)

    _ax.plot(_Vds, _Ids_op * 1e3, 'o', color='red', markersize=12, zorder=5,
             markeredgecolor='darkred', markeredgewidth=2)

    _ax.set_xlabel('$V_{DS}$ (V)', fontsize=16)
    _ax.set_ylabel('$I_{DS}$ (mA)', fontsize=16)
    _ax.set_title(r'$I_{DS}$–$V_{DS}$ ($V_{GS}$ labels at right)', fontsize=14, fontweight='bold')
    _ax.set_xlim(0, 3.8)
    _ids_max = 0.5 * _mu_ns * _Cox * _W_over_L * (3.0 - _Vt)**2 * 1e3
    _ax.set_ylim(0, _ids_max * 1.05)
    _ax.tick_params(labelsize=14)
    for _sp in ['top', 'right']:
        _ax.spines[_sp].set_visible(False)

    # -------- Bottom: Qinv along channel --------
    _ax = _ax_ch
    _ax.plot(_x_norm, _Qinv_norm, 'b-', lw=2.5)
    _ax.fill_between(_x_norm, 0, _Qinv_norm, color='cyan', alpha=0.3)
    _ax.axhline(0, color='k', lw=0.5)

    _ax.set_xlabel('Position along channel  ($x / L$)', fontsize=16)
    _ax.set_ylabel(r'$Q_{inv}(x)\; /\; C_{oxe}(V_{GS} - V_t)$', fontsize=16)
    _ax.set_xlim(0, 1)
    _ax.set_ylim(-0.08, 1.15)
    _ax.tick_params(labelsize=14)

    _ax.text(0.01, 1.05, 'Source', fontsize=14, va='bottom', fontweight='bold', color='#333')
    _ax.text(0.99, 0.05, 'Drain', fontsize=14, ha='right', va='bottom', fontweight='bold', color='#333')

    if _regime == "Saturation":
        _ax.annotate('Pinch-off: $Q_{inv}(L) \\to 0$', xy=(1.0, 0), xytext=(0.7, 0.4),
                     fontsize=13, ha='center', color='red', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    elif _regime == "Triode" and _Vds > 0.05:
        _ax.annotate(f'$Q_{{inv}}$ tapers\ntoward drain', xy=(0.85, _Qinv_norm[int(0.85*300)]),
                     xytext=(0.65, 0.3), fontsize=12, ha='center', color='#2266cc',
                     arrowprops=dict(arrowstyle='->', color='#2266cc', lw=1.2))

    for _sp in ['top', 'right']:
        _ax.spines[_sp].set_visible(False)

    plt.close(_fig)

    if _Vgs <= _Vt:
        _ids_str = "0 (below threshold)"
    else:
        _ids_str = f"{_Ids_op*1e3:.3f} mA"

    _info = mo.md(rf"""
    | Parameter | Value |
    |-----------|-------|
    | $V_{{GS}}$ | {_Vgs:.2f} V |
    | $V_{{DS}}$ | {_Vds:.2f} V |
    | $V_t$ | {_Vt:.2f} V |
    | $V_{{Dsat}} = V_{{GS}} - V_t$ | {_Vdsat:.2f} V |
    | **Regime** | **{_regime}** |
    | $I_{{DS}}$ | {_ids_str} |

    **Key relationships:**
    $Q_{{inv}}(x) = -C_{{oxe}}[V_{{GS}} - V_t - V_{{CS}}(x)]$, where $V_{{CS}}$ ranges from $0$ (source) to $V_{{DS}}$ (drain).
    The channel is pinched off when $V_{{DS}} \geq V_{{Dsat}} = V_{{GS}} - V_t$.
    """)

    _slider_row = mo.hstack([Vgs_slider, Vds_slider], justify="start")
    mo.vstack([_slider_row, mo.as_html(_fig), _info])
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _ids_vgs_img = mo.image(f"{IMAGE_BASE}/Ids_vs_Vgs.png", width=500)

    mo.md(rf"""
    ## 4. $I_{{DS}}$–$V_{{GS}}$ and Threshold Voltage

    To determine $V_t$ from measured data, plot $I_{{DS}}$ vs. $V_{{GS}}$ at a small $V_{{DS}}$. Two common methods:

    1. **Linear extrapolation**: In the linear region, $I_{{DS}} \propto (V_{{GS}} - V_t)$. Extrapolate the steepest part of the curve to the $V_{{GS}}$ axis.
    2. **Reference current**: Define $V_t$ as the $V_{{GS}}$ at which $I_{{DS}} = 0.1\,\mu\text{{A}} \times W/L$.

    Ideally, when $V_{{GS}} < V_t$, $I_{{DS}} = 0$. Any current below threshold contributes to **parasitic power dissipation** (the "off" state is not truly off). This subthreshold current will be discussed further in the next notebook.

    From textbook p. 206 (see also Lab 5)
    {mo.as_html(mo.hstack([_ids_vgs_img], justify="center"))}



    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Summary

    | Concept | Key Equation / Result |
    |---------|----------------------|
    | Inversion charge | $Q_{inv} = -C_{oxe}(V_{GS} - V_t)$ |
    | Threshold voltage | $V_t = V_{FB} + 2\phi_B + \dfrac{\sqrt{2\epsilon_s q N_A \cdot 2\phi_B}}{C_{oxe}}$ |
    | Saturation voltage | $V_{Dsat} = V_{GS} - V_t$ |
    | Triode current | $I_{DS} = \mu_{ns} C_{oxe} \dfrac{W}{L}\left[(V_{GS}-V_t)V_{DS} - \dfrac{V_{DS}^2}{2}\right]$ |
    | Saturation current | $I_{Dsat} = \dfrac{1}{2}\mu_{ns} C_{oxe}\dfrac{W}{L}(V_{GS}-V_t)^2$ |
    | Transconductance | $g_m = \dfrac{\partial I_{DS}}{\partial V_{GS}} = \mu_{ns} C_{oxe}\dfrac{W}{L}V_{Dsat}$ (max at $V_{Dsat}$) |

    - The gate voltage controls the channel via the inversion charge
    - $V_{DS}$ is dropped along the channel, tapering $Q_{inv}$ from source to drain
    - Pinch-off occurs at $V_{DS} = V_{Dsat}$, after which $I_{DS}$ saturates
    - Higher $C_{oxe}$ (thinner oxide) gives higher transconductance and better gate control
    """)
    return


if __name__ == "__main__":
    app.run()
