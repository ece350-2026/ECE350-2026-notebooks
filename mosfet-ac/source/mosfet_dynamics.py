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

    try:
        _test = Path(__file__).parent / "images"
        if _test.exists():
            ASSET_DIR = Path(__file__).parent
        else:
            raise FileNotFoundError
    except Exception:
        ASSET_DIR = None
    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/mosfet-ac/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    mo.md(r"""
    # MOSFET Dynamics, Small-Signal Model
    **ECE350: Lecture 35**

    Hu, Chapter 7

    This notebook covers the dynamic behavior of MOSFETs in both digital and analog applications.

    **Contents:**

    1. **Digital Switching Speed**: inverter delay and $I_{Dsat}$ ($I_{on}$)
    2. **Power Dissipation**: dynamic and static power
    3. **Small-Signal Model**: transconductance, output conductance, and cut-off frequency $f_T$
    """)
    return IMAGE_BASE, mo, np, plt


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _img_circuit = mo.image(f"{IMAGE_BASE}/switch-chain-circuit.png", width=500)
    _img_delay = mo.image(f"{IMAGE_BASE}/switch-prop-delay.png", width=500)

    _text_intro = mo.md(r"""
    ## 1. Digital Switching Speed and $I_{Dsat}$ ($I_{on}$)

    Consider a **chain of CMOS inverters** below. Each inverter consists of a PFET (pull-up) and NFET (pull-down). The gate capacitance $C_{gate}$ of the next stage must be charged or discharged through the on-transistor when the input switches. Here, $C_{gate}$ is the total capacitance of the gate, not the per unit area capacitance.
    """)

    _text_deriv = mo.md(r"""
    When $V_1$ turns on (goes high), the NFET at stage ① conducts and discharges node ②. The charge on the gate capacitance at ② decays exponentially:

    $$Q_{gate} = C_{gate} V_{DD}\, e^{-t/\tau}$$

    Differentiating:

    $$\frac{dQ_{gate}}{dt} = -\frac{C_{gate} V_{DD}}{\tau}\, e^{-t/\tau} = -\frac{Q_{gate}}{\tau}$$

    The gate current $I_{gate} \leq I_{Dsat}$ of the NFET at ①.

    ### Propagation Delay

    The time constant for $Q_{gate}$ to reach $C_{gate} V_{DD}/2$ is roughly:

    $$\boxed{\tau \sim \frac{C_{gate} V_{DD}}{2 I_{on}}} \qquad \text{where } I_{on} = I_{Dsat}$$

    More accurately, as the NFET at ① conducts, the PFET also has to turn off. Accounting for both transitions:

    $$\tau \sim \frac{1}{2}\left[\frac{C_{gate} V_{DD}}{2 I_{on,NFET}} + \frac{C_{gate} V_{DD}}{2 I_{on,PFET}}\right]$$

    **Higher $I_{on}$ leads to shorter delay. Higher $I_{Dsat}$ is preferred for high-speed digital switching!**
    """)

    mo.vstack([
        _text_intro,
        mo.hstack([_img_circuit], justify="center"),
        _text_deriv,
        mo.hstack([_img_delay], justify="center"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    Vdd_slider = mo.ui.slider(start=0.5, stop=1.8, value=1.0, step=0.1,
                               label=r"$V_{DD}$ (V)")
    Cgate_slider = mo.ui.slider(start=0.1, stop=5.0, value=1.0, step=0.1,
                                 label=r"$C_{gate}$ (fF)")
    Ion_slider = mo.ui.slider(start=50, stop=1000, value=300, step=50, label=r"$I_{on}$ (µA)")
    return Cgate_slider, Ion_slider, Vdd_slider


@app.cell(hide_code=True)
def _(Cgate_slider, Ion_slider, Vdd_slider, mo, np, plt):
    _Vdd = Vdd_slider.value
    _Cgate = Cgate_slider.value * 1e-15
    _Ion = Ion_slider.value * 1e-6
    _tau = _Cgate * _Vdd / (2 * _Ion)
    _tau_ps = _tau * 1e12

    _t = np.linspace(0, 5 * _tau, 500)

    _Vin_period = 10 * _tau
    _Vin = _Vdd * ((_t % _Vin_period) < _Vin_period / 2).astype(float)

    _Vout = np.zeros_like(_t)
    _vout_val = _Vdd
    _dt = _t[1] - _t[0]
    for _i in range(1, len(_t)):
        if _Vin[_i] > _Vdd / 2:
            _vout_val -= (_Ion / _Cgate) * _dt * max(0, _vout_val / _Vdd)
        else:
            _vout_val += (_Ion / _Cgate) * _dt * max(0, 1 - _vout_val / _Vdd)
        _vout_val = np.clip(_vout_val, 0, _Vdd)
        _Vout[_i] = _vout_val

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _ax1.plot(_t * 1e12, _Vin, 'b-', lw=2, label='$V_{in}$')
    _ax1.plot(_t * 1e12, _Vout, 'r-', lw=2.5, label='$V_{out}$')
    _ax1.axhline(_Vdd / 2, color='gray', ls='--', lw=1, alpha=0.5)
    _ax1.set_xlabel('Time (ps)', fontsize=16)
    _ax1.set_ylabel('Voltage (V)', fontsize=16)
    _ax1.set_title('Inverter Switching Waveforms', fontsize=16, fontweight='bold')
    _ax1.legend(fontsize=14)
    _ax1.tick_params(labelsize=14)
    _ax1.set_ylim(-0.1 * _Vdd, 1.3 * _Vdd)
    for _sp in ['top', 'right']:
        _ax1.spines[_sp].set_visible(False)

    _ion_arr = np.linspace(50e-6, 1000e-6, 200)
    _tau_arr = _Cgate * _Vdd / (2 * _ion_arr) * 1e12
    _ax2.plot(_ion_arr * 1e6, _tau_arr, 'b-', lw=2.5)
    _ax2.plot(_Ion * 1e6, _tau_ps, 'ro', markersize=10, zorder=5)
    _ax2.set_xlabel('$I_{on}$ (µA)', fontsize=16)
    _ax2.set_ylabel(r'Delay $\tau$ (ps)', fontsize=16)
    _ax2.set_title(r'Switching Delay vs. $I_{on}$', fontsize=16, fontweight='bold')
    _ax2.tick_params(labelsize=14)
    _ax2.annotate(f'τ = {_tau_ps:.1f} ps', xy=(_Ion * 1e6, _tau_ps),
                  xytext=(_Ion * 1e6 + 100, _tau_ps + 1),
                  fontsize=14, arrowprops=dict(arrowstyle='->', lw=1.5))
    for _sp in ['top', 'right']:
        _ax2.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig)

    _info = mo.md(rf"""
    **Delay:** $\tau = C_{{gate}} V_{{DD}} / (2\,I_{{on}})$ = {_tau_ps:.1f} ps

    $C_{{gate}}$ = {Cgate_slider.value} fF, $V_{{DD}}$ = {_Vdd} V, $I_{{on}}$ = {Ion_slider.value} µA

    Faster switching requires: higher $I_{{on}}$, lower $C_{{gate}}$, lower $V_{{DD}}$ (but $V_{{DD}}$ also affects noise margin).
    """)

    mo.vstack([
        mo.md("### Inverter Delay Parameters"),
        mo.hstack([Vdd_slider, Cgate_slider, Ion_slider], justify="start"),
        mo.as_html(_fig),
        _info,
    ])
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _img = mo.image(f"{IMAGE_BASE}/CMOS-switch.png", width=500)

    _text = mo.md(r"""
    ## 2. Power Dissipation

    Charge transferred to load at $V_{out}$: $Q = C_{gate} V_{DD}$

    Rate of charge transfer: $I = dQ/dt = C_{gate} V_{DD} f$

    Activity factor (transistor is not used all the time), $k$: $k C_{gate} V_{DD} f$, where $k < 1$

    - **Dynamic power:** $P_{dynamic} = V_{DD} \cdot I = k C_{gate} V_{DD}^2 f$
    - **Static (leakage) power:** $P_{static} = V_{DD} \cdot I_{off}$, where $I_{off}$ = sub-threshold current

    $$P_{total} = P_{dynamic} + P_{static} = k C_{gate} V_{DD}^2 f + V_{DD} I_{off}$$

    **Low power** $\Rightarrow$ low leakage (subthreshold) current, low $V_{DD}$, low load capacitance
    """)

    mo.vstack([
        _text,
        mo.hstack([_img], justify="center"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    Vdd_p_slider = mo.ui.slider(start=0.3, stop=1.2, value=0.7, step=0.1,
                                 label=r"$V_{DD}$ (V)")
    Cgate2_slider = mo.ui.slider(start=0.05, stop=1, value=.5, step=0.05,
                                 label=r"$C$ (fF)")
    freq_slider = mo.ui.slider(start=0.1, stop=5.0, value=2.0, step=0.1,
                                label=r"$f$ (GHz)")
    Ioff_slider = mo.ui.slider(start=-2.0, stop=2.0, value=0.0, step=0.5,
                                label=r"$\log_{10}(I_{off}/\text{nA})$")
    k_slider = mo.ui.slider(start=0.01, stop=1, value=0.1, step=0.01,
                             label=r"$k$ (activity factor)")
    numgates_slider = mo.ui.slider(start=1, stop=300, value=20, step=1, label=r"Number of transistors (B)")
    return (
        Cgate2_slider,
        Ioff_slider,
        Vdd_p_slider,
        freq_slider,
        k_slider,
        numgates_slider,
    )


@app.cell(hide_code=True)
def _(
    Cgate2_slider,
    Ioff_slider,
    Vdd_p_slider,
    freq_slider,
    k_slider,
    mo,
    np,
    numgates_slider,
    plt,
):
    _Vdd = Vdd_p_slider.value
    _f = freq_slider.value * 1e9
    _Ioff = 10 ** Ioff_slider.value * 1e-9
    _Cgate = Cgate2_slider.value * 1e-15
    _k = k_slider.value
    _N_gates = numgates_slider.value * 1e9

    _P_dyn = _k * _Cgate * _Vdd**2 * _f * _N_gates
    _P_stat = _Vdd * _Ioff * _N_gates
    _P_total = _P_dyn + _P_stat

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    _bars = _ax1.bar(['Dynamic', 'Static', 'Total'],
                     [_P_dyn, _P_stat, _P_total],
                     color=['#4CAF50', '#FF5722', '#2196F3'], edgecolor='black', lw=1.5)
    _ax1.set_ylabel('Power (W)', fontsize=16)
    _ax1.set_title(f'Power Breakdown ({_N_gates/1e9:.0f}B gates)', fontsize=16, fontweight='bold')
    _ax1.tick_params(labelsize=14)
    for _bar, _val in zip(_bars, [_P_dyn, _P_stat, _P_total]):
        _ax1.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + _P_total * 0.02,
                  f'{_val:.2f} W', ha='center', fontsize=13, fontweight='bold')
    for _sp in ['top', 'right']:
        _ax1.spines[_sp].set_visible(False)

    _vdd_arr = np.linspace(0.3, 1.8, 200)
    _pdyn_arr = _k * _Cgate * _vdd_arr**2 * _f * _N_gates
    _pstat_arr = _vdd_arr * _Ioff * _N_gates
    _ax2.plot(_vdd_arr, _pdyn_arr, 'g-', lw=2.5, label=r'$P_{dynamic} \propto V_{DD}^2$')
    _ax2.plot(_vdd_arr, _pstat_arr, 'r-', lw=2.5, label=r'$P_{static} \propto V_{DD}$')
    _ax2.plot(_vdd_arr, _pdyn_arr + _pstat_arr, 'b--', lw=2, label='$P_{total}$')
    _ax2.axvline(_Vdd, color='gray', ls=':', lw=1.5, alpha=0.7)
    _ax2.set_xlabel('$V_{DD}$ (V)', fontsize=16)
    _ax2.set_ylabel('Power (W)', fontsize=16)
    _ax2.set_title('Power vs. Supply Voltage', fontsize=16, fontweight='bold')
    _ax2.legend(fontsize=12)
    _ax2.tick_params(labelsize=14)
    for _sp in ['top', 'right']:
        _ax2.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig)

    _power_table = mo.md(rf"""
    | Component | Value |
    |:-----------|:-------:|
    | $P_{{dynamic}}$ | {_P_dyn:.0f} W |
    | $P_{{static}}$ | {_P_stat:.0f} W |
    | **$P_{{total}}$** | **{_P_total:.0f} W** |
    """)

    _param_table = mo.md(rf"""
    | Parameter | Value |
    |:----------|:-----:|
    | $V_{{DD}}$ | {_Vdd} V |
    | $f$ | {_f/1e9:.1f} GHz |
    | $C_{{gate}}$ | {_Cgate/1e-15:.2f} fF |
    | $k$ (activity factor) | {_k} |
    | $I_{{off}}$ | {_Ioff/1e-9:.1e} nA |
    | Number of transistors | {_N_gates/1e9:.0f} B |
    """)

    _comparison = mo.md(r"""

    ## Comparison Table


    | Group | Processor | Category | Transistors | Clock (Base/Boost) | Power |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **NVIDIA** | B200 | Data Center GPU | 208B | est. ~2 GHz | 1000W |
    | **NVIDIA** | RTX 6000 Pro | Professional GPU | 92.2B | ~2 GHz | 600W |
    | **NVIDIA** | RTX 5060 | Gaming GPU | 21.9B | 2.28 / 2.50 GHz | 145W |
    | **AMD** | EPYC 9654 | Server CPU | 90B | 2.40 / 3.70 GHz | 360W (320-400W TDP)|
    | **AMD** | Ryzen 9 9950X | Desktop CPU | est. 20.6B | 4.30 / 5.70 GHz | 170W (230W Max PPT) |
    | **AMD** | Ryzen 7 9800X3D | Gaming CPU | est. 16.5B | 4.70 / 5.20 GHz | 120W (162W Max PPT) |
    | **Intel** | Xeon 8592+ | Server CPU | est. 61B | 1.90 / 3.90 GHz | 350W |
    | **Intel** | Core Ultra 9 285K| Desktop CPU | 17.8B | 3.70 / 5.70 GHz | 125W (250W max turbo) |
    | **Apple** | M3 Ultra | Workstation SoC | 184B | est. 2.75 / 4.05 GHz? | 140W? |
    | **Apple** | M4 Max | Laptop SoC | 95B | est. 2.59 / 4.51 GHz | 95W? |
    | **Apple** | M4 | Mobile SoC | 28B | est. 2.89 / 4.51 GHz | 24W? |

    ### Technical Context & Definitions

    #### 1. TDP vs. TGP vs. PPT
    * **TDP (Thermal Design Power):** Used primarily by Intel and AMD for CPUs. It represents the cooling required for the chip at base clock speeds.
    * **Max Turbo Power / PPT (Package Power Tracking):** The actual power draw of a CPU under heavy load. For example, an Intel chip with a 125W TDP can pull **250W** during intensive tasks.
    * **TGP/TBP (Total Graphics/Board Power):** Used by NVIDIA. This includes the power for the GPU die, memory, and voltage regulation.

    #### 2. Architectural Differences
    * **GPU Density:** GPUs pack more transistors into a specific area compared to CPUs. They use many small, identical cores (CUDA/Tensor cores) optimized for parallel math.
    * **Unified Memory (Apple Silicon):** Apple's M-series chips include the CPU, GPU, and Neural Engine on a single die or interconnected package. This is referred to as system-on-chip (SoC). This is why their transistor counts are much higher than standalone desktop CPUs. Apple does not provide details on their chips. The numbers in the table are best estimates.
    * **Chiplet Design:** Modern high-end chips (AMD EPYC, Ryzen, and Intel Core Ultra) are made of multiple smaller pieces of silicon (dies) connected inside one package. The transistor count provided above is the total package sum.

    ###Observation

    To match the computed power consumption with the actual values, the activity factor is very low (1 to 3%). Most transistors are dedicated to memory, control logic, or idle blocks, and modern chips turn off parts of the chip that are not in use (power gating, clock gating). Only a small fraction of the transistors contribute to the effective switched capacitance at any moment!

    """)

    mo.vstack([
        mo.md("### Power Dissipation Estimate"),
        mo.hstack([Vdd_p_slider, Cgate2_slider, freq_slider], justify="start"),
        mo.hstack([k_slider, Ioff_slider, numgates_slider], justify="start"),
        mo.as_html(_fig),
        mo.hstack([_power_table, _param_table], justify="start", widths=[0.4, 0.6]),
        _comparison,
    ])
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _img1 = mo.image(f"{IMAGE_BASE}/small_signal_blackbox.png", width=400)

    _text_blackbox = mo.md(r"""
    ## 3. Analog Small-Signal High-Frequency Response

    Consider the MOSFET as a black-box three-terminal device with DC bias plus small AC signals:

    $$I_{ds} = I_{ds,DC}(V_{gs,DC}, V_{ds,DC}) + i_{ds}(t)$$

    $$V_{ds} = V_{ds,DC} + v_{ds}(t) \qquad V_{gs} = V_{gs,DC} + v_{gs}(t)$$
    """)

    _text_ss = mo.md(r"""
    ### Small-Signal Model

    The small-signal drain current is obtained by Taylor expansion about the DC operating point:

    $$i_{ds} = I_{ds} - I_{ds,DC} = v_{ds}\frac{\partial I_{ds}}{\partial V_{ds}}\bigg|_{DC} + v_{gs}\frac{\partial I_{ds}}{\partial V_{gs}}\bigg|_{DC}$$

    This defines the two key small-signal parameters:

    - $g_d = \dfrac{\partial I_{ds}}{\partial V_{ds}}\bigg|_{DC}$: **output conductance**
    - $g_m = \dfrac{\partial I_{ds}}{\partial V_{gs}}\bigg|_{DC}$: **transconductance**

    $$\boxed{i_{ds} = g_d \, v_{ds} + g_m \, v_{gs}}$$
    """)

    mo.vstack([
        _text_blackbox,
        mo.hstack([_img1], justify="center"),
        _text_ss,
    ])
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _img2 = mo.image(f"{IMAGE_BASE}/small_signal_circuit.png", width=600)
    _img3 = mo.image(f"{IMAGE_BASE}/fT-vs-CMOS-node.png", width=600)

    _text2_intro = mo.md(r"""
    ### Small-Signal Circuit Model and $f_T$

    The MOSFET cross-section shows the intrinsic capacitances $C_{gs}$ and $C_{gd}$ between the gate and the source/drain regions. The equivalent circuit includes:

    - $C_{gs}$, $C_{gd}$: gate-source and gate-drain capacitances
    - $g_m v_{gs}$: voltage-controlled current source
    - $1/g_d$: output resistance
    """)

    _text2_nodes = mo.md(r"""
    **Node analysis (at node ①, gate):**

    $$i_{in} = j\omega C_{gd}(v_{gs} - v_{ds}) + j\omega C_{gs} v_{gs}$$

    **Node analysis (at node ②, drain):**

    $$i_{ds} = g_m v_{gs} + g_d v_{ds} - j\omega C_{gd}(v_{gs} - v_{ds})$$

    ### Deriving $f_T$

    Assume **saturation**, so $g_d = 0$, and short-circuit the output ($v_{ds} = 0$). Then $i_{ds} \approx g_m v_{gs}$ and $i_{in} \approx j\omega C_{gs} v_{gs}$ (neglecting $C_{gd}$), giving:

    $$\left|\frac{i_{ds}}{i_{in}}\right| = \frac{g_m}{2\pi f C_{gs}}$$

    Define the **cut-off frequency** $f_T$ as when $|i_{ds}/i_{in}| = 1$:

    $$\boxed{f_T = \frac{g_m}{2\pi C_{gs}}}$$

    Since $C_{gs} \approx C_{oxe} W L$ and $g_m = \mu_{ns} C_{oxe} (W/L)(V_{gs,DC} - V_t)$:

    $$\boxed{f_T = \frac{\mu_{ns}}{2\pi L^2}(V_{gs,DC} - V_t)}$$

    **$f_T$ increases as $L$ decreases** — scaling makes devices faster!
    """)

    mo.vstack([
        _text2_intro,
        mo.hstack([_img2], justify="center"),
        _text2_nodes,
        mo.hstack([mo.vstack([_img3, mo.md(r"*C.H. Jan et al. (Intel), IEDM, 2010.*")], align="center")], justify="center"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    L_ft_slider = mo.ui.slider(start=0.02, stop=1.0, value=0.1, step=0.02,
                                label=r"$L$ (µm)")
    Vov_ft_slider = mo.ui.slider(start=0.1, stop=1.5, value=0.5, step=0.1,
                                  label=r"$V_{GS} - V_t$ (V)")
    mu_ft_slider = mo.ui.slider(start=100, stop=500, value=250, step=50,
                                 label=r"$\mu_{ns}$ (cm²/Vs)")
    return L_ft_slider, Vov_ft_slider, mu_ft_slider


@app.cell(hide_code=True)
def _(L_ft_slider, Vov_ft_slider, mo, mu_ft_slider, np, plt):
    _L_um = L_ft_slider.value
    _L = _L_um * 1e-4
    _Vov = Vov_ft_slider.value
    _mu = mu_ft_slider.value

    _fT = _mu * _Vov / (2 * np.pi * _L**2)
    _fT_GHz = _fT / 1e9

    _fig, _ax = plt.subplots(figsize=(7, 5))

    _L_arr = np.linspace(0.02, 1.0, 200)
    _fT_arr = _mu * _Vov / (2 * np.pi * (_L_arr * 1e-4)**2) / 1e9

    _ax.plot(_L_arr, _fT_arr, 'b-', lw=2.5)
    _ax.plot(_L_um, _fT_GHz, 'ro', markersize=10, zorder=5)
    _ax.set_xlabel('Channel Length $L$ (µm)', fontsize=16)
    _ax.set_ylabel('$f_T$ (GHz)', fontsize=16)
    _ax.set_title(r'Cut-off Frequency vs. Channel Length', fontsize=16, fontweight='bold')
    _ax.set_yscale('log')
    _ax.tick_params(labelsize=14)
    _ax.annotate(f'$f_T$ = {_fT_GHz:.0f} GHz', xy=(_L_um, _fT_GHz),
                 xytext=(_L_um + 0.15, _fT_GHz * 1.5),
                 fontsize=14, arrowprops=dict(arrowstyle='->', lw=1.5))
    for _sp in ['top', 'right']:
        _ax.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig)

    _info = mo.md(rf"""
    $$f_T = \frac{{\mu_{{ns}}(V_{{GS}} - V_t)}}{{2\pi L^2}} = \frac{{{_mu} \times {_Vov}}}{{2\pi \times ({_L_um}\times 10^{{-4}})^2}} = {_fT_GHz:.0f}\;\text{{GHz}}$$

    $f_T$ increases when:
    - $L$ **decreases** (most powerful lever — $f_T \propto 1/L^2$)
    - $\mu_{{ns}}$ increases (high-mobility channels)
    - $V_{{GS}} - V_t$ increases
    """)

    mo.vstack([
        mo.md(r"### Interactive: Cut-off Frequency $f_T$"),
        mo.hstack([L_ft_slider, Vov_ft_slider, mu_ft_slider], justify="start"),
        mo.as_html(_fig),
        _info,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Topic |  Result |
    |:-------|:-----------:|
    | **Switching delay** | $\tau \approx C_{gate} V_{DD} / I_{on}$; higher $I_{on}$ → faster |
    | **Dynamic power** | $P_{dyn} = k C V_{DD}^2 f$ |
    | **Static power** | $P_{stat} = V_{DD} I_{off}$ |
    | **Cut-off frequency** | $f_T = \dfrac{\mu_{ns}(V_{GS}-V_t)}{2\pi L^2}$ |


    Shrinking $L$ increases speed ($f_T \propto 1/L^2$) and density, but worsens short-channel effects (DIBL, $V_t$ roll-off, higher $I_{off}$). Advanced device architectures (FinFET, nanosheets) are designed to maintain gate control as $L$ scales below 20 nm.
    """)
    return


if __name__ == "__main__":
    app.run()
