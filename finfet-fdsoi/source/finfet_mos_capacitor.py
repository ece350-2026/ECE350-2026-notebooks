# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.4.4",
#     "matplotlib==3.10.8",
#     "scipy==1.17.1",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def title():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import constants
    from pathlib import Path

    # Physical constants
    q = constants.e           # Elementary charge (C)
    eps_0 = constants.epsilon_0  # Vacuum permittivity (F/m)
    k_B = constants.k         # Boltzmann constant (J/K)
    T = 300                   # Temperature (K)
    V_T = k_B * T / q         # Thermal voltage at 300K

    try:
        _test = Path(__file__).parent / "images"
        if _test.exists():
            ASSET_DIR = Path(__file__).parent
        else:
            raise FileNotFoundError
    except Exception:
        ASSET_DIR = None
    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/finfet-fdsoi/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    mo.md(
        r"""
        # FinFET MOS Capacitor: Electrostatics and Threshold Voltage

        ECE350: Extra topic, for interest

        This notebook derives and visualizes the electrostatics of symmetric double-gate FinFETs and compares them with FDSOI. We solve Poisson's equation under the fully-depleted condition for the FinFET. We find that the threshold voltage for FinFET and FDSOI FET is close to the flat-band voltage and exhibits a lower variability than bulk MOSFETs.

        1. **Device Structure and Parameters**: FinFET/Nanosheet symmetric structure
        2. **Derivation**: Charge density, electric field, and potential from Poisson's equation
        3. **FinFET Electrostatics**: Interactive charge, field, potential & energy band diagrams
        4. **Interactive FD-SOI Electrostatics**: Coupled front/back gate with back body bias
        5. **Threshold Voltage Scaling**: $V_t$ vs. body thickness for FinFET, FDSOI, and bulk
        """
    )
    return IMAGE_BASE, eps_0, mo, np, plt, q


@app.cell
def device_parameters(IMAGE_BASE, mo):
    _img = mo.image(f"{IMAGE_BASE}/sheet-fin-moscap.png", width=600)

    mo.md(rf"""
    ## Device Structure and Parameters

    ### FinFET/Nanosheet Symmetric Structure

    {mo.hstack([_img], justify="center")}

    The FinFET has gates on both sides of the silicon fin, creating a **symmetric**
    structure. For analysis, we assume the fin/sheet is infinitely long out of the plane, and we ignore the finite height of the fin or length of the sheet to use a 1D model along the thickness of the body. We use

    - Potential reference at $t_{{Si}}/2$: $V(t_{{Si}}/2) = 0$ at the center of the fin
    - Symmetry condition: $\mathcal{{E}}|_{{x=t_{{Si}}/2}} = - dV/dx|_{{x=t_{{Si}}/2}} = 0$ (zero electric field at center)

    The symmetry requirement arises since $V(x=-t_{{ox}}) = V(x=t_{{Si}}+t_{{ox}})$. If $\mathcal{{E}}|_{{x=t_{{Si}}/2}} \neq 0$, it cannot be both positive and negative at the same time.

    This symmetry allows us to solve only half the structure ($0 \leq x \leq t_{{Si}}/2$).
    """)
    return


@app.cell
def sliders(mo):
    # Device parameter sliders
    t_Si_slider = mo.ui.slider(
        start=3, stop=15, step=0.5, value=7,
        label=r"Silicon thickness $t_{Si}$ (nm)"
    )
    t_ox_slider = mo.ui.slider(
        start=0.5, stop=3, step=0.1, value=1.0,
        label=r"Oxide thickness $t_{ox}$ (nm)"
    )
    N_A_slider = mo.ui.slider(
        start=14, stop=17, step=0.5, value=16,
        label=r"log$_{10}$($N_A$) (cm⁻³)"
    )
    eps_ox_slider = mo.ui.slider(
        start=3.9, stop=25, step=0.5, value=4.0,
        label=r"Oxide relative permittivity $\epsilon_{ox}$"
    )
    V_FB_slider = mo.ui.slider(
        start=-0.5, stop=0.5, step=0.05, value=0.0,
        label=r"Flat-band voltage $V_{FB}$ (V)"
    )
    Vg_slider = mo.ui.slider(
        start=-1.5, stop=1.5, step=0.05, value=0.0,
        label=r"Gate voltage $V_G$ (V)", full_width=False
    )
    return (
        N_A_slider,
        V_FB_slider,
        Vg_slider,
        eps_ox_slider,
        t_Si_slider,
        t_ox_slider,
    )


@app.cell
def compute_finfet(
    N_A_slider,
    V_FB_slider,
    eps_0,
    eps_ox_slider,
    np,
    q,
    t_Si_slider,
    t_ox_slider,
):
    # Extract slider values
    t_Si = t_Si_slider.value * 1e-9      # Convert nm to m
    t_ox = t_ox_slider.value * 1e-9      # Convert nm to m
    N_A = 10**N_A_slider.value * 1e6     # Convert cm⁻³ to m⁻³
    eps_rOX = eps_ox_slider.value
    eps_rSi = 11.7                        # Silicon relative permittivity
    V_FB = V_FB_slider.value

    # Permittivities
    eps_Si = eps_rSi * eps_0
    eps_OX = eps_rOX * eps_0

    # Oxide capacitance per unit area
    C_oxe = eps_OX / t_ox

    # --- FinFET Electrostatics (symmetric DG) ---
    # For 0 ≤ x ≤ t_Si/2, solving Poisson in depletion:
    # d²φ/dx² = qN_A/ε_Si
    # With BCs: φ(t_Si/2) = 0, dφ/dx|_{t_Si/2} = 0

    # Electric field in silicon (linear)
    # ξ(x) = (qN_A/ε_Si)(t_Si/2 - x)

    # Surface electric field (at x=0)
    xi_S_finfet = q * N_A * t_Si / (2 * eps_Si)

    # Electric field in oxide (from continuity: ε_Si·ξ_S = ε_OX·ξ_OX)
    xi_OX_finfet = (eps_Si / eps_OX) * xi_S_finfet

    # Surface potential (at x=0)
    # φ_S = (qN_A/2ε_Si)(t_Si/2)² = qN_A·t_Si²/(8ε_Si)
    phi_S_finfet = q * N_A * t_Si**2 / (8 * eps_Si)

    # Voltage drop across oxide
    V_OX_finfet = xi_OX_finfet * t_ox

    # Alternative: V_OX = qN_A·t_Si / (2·C_oxe)
    V_OX_finfet_alt = q * N_A * t_Si / (2 * C_oxe)

    # Threshold voltage for FinFET
    V_t_finfet = V_FB + V_OX_finfet + phi_S_finfet

    # Create spatial arrays for plotting
    x_Si = np.linspace(0, t_Si/2, 200)
    x_full = np.linspace(0, t_Si, 400)
    x_ox_left = np.linspace(-t_ox, 0, 50)
    x_ox_right = np.linspace(t_Si, t_Si + t_ox, 50)

    # Electric field profile (half structure)
    xi_Si = (q * N_A / eps_Si) * (t_Si/2 - x_Si)
    xi_ox_left = np.ones_like(x_ox_left) * xi_OX_finfet

    # Full structure electric field (symmetric, antisymmetric about center)
    xi_full = np.zeros_like(x_full)
    for i, x in enumerate(x_full):
        if x <= t_Si/2:
            xi_full[i] = (q * N_A / eps_Si) * (t_Si/2 - x)
        else:
            xi_full[i] = -(q * N_A / eps_Si) * (x - t_Si/2)

    # Potential profile (half structure)
    phi_Si = (q * N_A / (2 * eps_Si)) * (t_Si/2 - x_Si)**2
    phi_ox_left = -xi_OX_finfet * (x_ox_left + t_ox) + phi_S_finfet + V_OX_finfet

    # Full structure potential (symmetric about center)
    phi_full = np.zeros_like(x_full)
    for i, x in enumerate(x_full):
        if x <= t_Si/2:
            phi_full[i] = (q * N_A / (2 * eps_Si)) * (t_Si/2 - x)**2
        else:
            phi_full[i] = (q * N_A / (2 * eps_Si)) * (x - t_Si/2)**2
    return C_oxe, N_A, V_FB, V_t_finfet, eps_OX, eps_Si, t_Si, t_ox


@app.cell
def theory_derivation(mo):
    mo.md(r"""
    ## Derivation (Half Structure, Using Symmetry)

    By symmetry we solve only half: $-t_{ox} \leq x \leq t_{Si}/2$.

    **Boundary conditions** at the center ($x = t_{Si}/2$):

    - $V = 0$ &ensp;(reference)
    - $\mathcal{E} = 0$ &ensp;(symmetry — the field must vanish at the mirror plane)



    ### Step 1 — Charge Density $\rho(x)$

    $$\rho(x) = \begin{cases} 0 & -t_{ox} \leq x < 0 \quad\text{(oxide)}\\ -qN_A & 0 \leq x \leq \dfrac{t_{Si}}{2} \quad\text{(depleted Si)} \end{cases}$$



    ### Step 2 — Electric Field from Poisson's Relation

    $$\frac{d\mathcal{E}}{dx} = \frac{\rho(x)}{\epsilon}$$

    Integrating with $\mathcal{E}(t_{Si}/2) = 0$:

    $$\mathcal{E}(x) = \begin{cases} \mathcal{E}_{OX} = \dfrac{\epsilon_s}{\epsilon_{OX}}\,\mathcal{E}_S & -t_{ox} \leq x < 0 \\[8pt] \dfrac{qN_A}{\epsilon_s}\!\left(\dfrac{t_{Si}}{2} - x\right) & 0 \leq x \leq \dfrac{t_{Si}}{2} \end{cases}$$

    The **surface field** is $\mathcal{E}_S = \dfrac{qN_A\, t_{Si}}{2\epsilon_s}$, and the oxide field follows from the displacement continuity condition $\epsilon_s\,\mathcal{E}_S = \epsilon_{OX}\,\mathcal{E}_{OX}$.



    ### Step 3 — Potential from $\dfrac{dV}{dx} = -\mathcal{E}(x)$

    Integrating with $V(t_{Si}/2) = 0$:

    $$V(x) = \begin{cases} -\mathcal{E}_{OX}\, x \;+\; \dfrac{qN_A\, t_{Si}^2}{8\epsilon_s} & -t_{ox} \leq x < 0 \\[8pt] \dfrac{qN_A}{2\epsilon_s}\!\left(\dfrac{t_{Si}}{2} - x\right)^{\!2} & 0 \leq x \leq \dfrac{t_{Si}}{2} \end{cases}$$

    **Surface potential** (at $x = 0$): &ensp; $\phi_s = \dfrac{qN_A\, t_{Si}^2}{8\epsilon_s}$



    ### Threshold Voltage

    $$V_t = V_{FB} + V_{ox} + \phi_s = V_{FB} + \frac{qN_A\, t_{Si}}{2C_{oxe}} + \frac{qN_A\, t_{Si}^2}{8\epsilon_s}$$

    where $V_{FB} = \Psi_{ms} = \Psi_m - \Psi_s$ (for an ideal structure with no oxide charges).

    For ultra-thin bodies ($t_{Si} < 10$ nm) and light doping ($N_A \sim 10^{16}$ cm$^{-3}$), both $V_{ox}$ and $\phi_s$ are negligible → **$V_t \approx V_{FB} = \Psi_{ms}$**
    """)
    return


@app.cell(hide_code=True)
def plot_electrostatics(
    C_oxe,
    N_A,
    N_A_slider,
    V_FB,
    V_FB_slider,
    V_t_finfet,
    Vg_slider,
    eps_Si,
    eps_ox_slider,
    mo,
    np,
    plt,
    q,
    t_Si,
    t_Si_slider,
    t_ox,
    t_ox_slider,
):
    _Vg = Vg_slider.value
    _t_Si_nm = t_Si * 1e9
    _t_ox_nm = t_ox * 1e9
    _metal_w_nm = max(0.8, _t_ox_nm * 0.5)

    _x_metal_L = np.array([-_t_ox_nm - _metal_w_nm, -_t_ox_nm])
    _x_metal_R = np.array([_t_Si_nm + _t_ox_nm, _t_Si_nm + _t_ox_nm + _metal_w_nm])
    _x_ox_L = np.linspace(-_t_ox_nm, 0, 50)
    _x_ox_R = np.linspace(_t_Si_nm, _t_Si_nm + _t_ox_nm, 50)
    _x_Si = np.linspace(0, t_Si, 400)
    _x_Si_nm = _x_Si * 1e9

    _xlim = (-_t_ox_nm - _metal_w_nm - 0.3, _t_Si_nm + _t_ox_nm + _metal_w_nm + 0.3)

    # ── Determine operating regime (symmetric DG) ──
    if _Vg < V_FB:
        _regime = "Accumulation"
    elif _Vg > V_t_finfet:
        _regime = "Inversion"
    else:
        _regime = "Depletion"

    _is_depleted = (_regime != "Accumulation")

    # ── Compute electrostatics from physics ──
    _phi_S_max = q * N_A * t_Si**2 / (8 * eps_Si)  # max surface potential at threshold

    if _is_depleted:
        _V_OX_dep = q * N_A * t_Si / (2 * C_oxe)
        _phi_S = min(_phi_S_max, max(0, _Vg - V_FB - _V_OX_dep))
        _xi_S = q * N_A * t_Si / (2 * eps_Si)
        _V_OX = (_Vg - V_FB - _phi_S)

        # Full structure: E(x) and phi(x) from Poisson
        _xi_full = np.zeros_like(_x_Si)
        _phi_full = np.zeros_like(_x_Si)
        for _i, _x in enumerate(_x_Si):
            if _x <= t_Si / 2:
                _xi_full[_i] = (q * N_A / eps_Si) * (t_Si / 2 - _x)
                _phi_full[_i] = (q * N_A / (2 * eps_Si)) * (t_Si / 2 - _x)**2
            else:
                _xi_full[_i] = -(q * N_A / eps_Si) * (_x - t_Si / 2)
                _phi_full[_i] = (q * N_A / (2 * eps_Si)) * (_x - t_Si / 2)**2
    else:
        _phi_S = 0.0
        _xi_S = 0.0
        _V_OX = 0.0
        _xi_full = np.zeros_like(_x_Si)
        _phi_full = np.zeros_like(_x_Si)

    _E_OX = _V_OX / t_ox if t_ox > 0 else 0.0
    _phi_gate = _Vg - V_FB

    # ── Interface charges (per gate, C/m²) ──
    _Q_dep_half = q * N_A * t_Si / 2  # depletion charge seen by each gate

    if _regime == "Accumulation":
        _Q_M = C_oxe * (_Vg - V_FB)
        _Q_inv = 0.0
        _Q_acc = -_Q_M
    elif _regime == "Inversion":
        _Q_inv = -C_oxe * (_Vg - V_t_finfet)
        _Q_M = _Q_dep_half - _Q_inv
        _Q_acc = 0.0
    else:
        _Q_M = _Q_dep_half
        _Q_inv = 0.0
        _Q_acc = 0.0

    # ── Helper: shade regions ──
    def _shade(_ax):
        _ax.axvspan(-_t_ox_nm - _metal_w_nm, -_t_ox_nm, alpha=0.25, color='#6baed6')
        _ax.axvspan(-_t_ox_nm, 0, alpha=0.15, color='orange')
        _ax.axvspan(_t_Si_nm, _t_Si_nm + _t_ox_nm, alpha=0.15, color='orange')
        _ax.axvspan(_t_Si_nm + _t_ox_nm, _t_Si_nm + _t_ox_nm + _metal_w_nm, alpha=0.25, color='#6baed6')
        for _xv in [-_t_ox_nm, 0, _t_Si_nm, _t_Si_nm + _t_ox_nm]:
            _ax.axvline(x=_xv, color='gray', linestyle=':', alpha=0.4)
        _ax.axvline(x=_t_Si_nm / 2, color='red', linestyle='--', alpha=0.4)
        _ax.set_xlim(_xlim)
        _ax.grid(True, alpha=0.3)
        _ax.tick_params(labelsize=13)

    # ═══════════════════════════════════════════
    # 1×3 figure: ρ, E, V
    # ═══════════════════════════════════════════
    _fig, (_ax_rho, _ax_E, _ax_V) = plt.subplots(1, 3, figsize=(18, 5))
    _fig.subplots_adjust(wspace=0.35)

    # ── (0) Charge density ρ(x) ──
    _shade(_ax_rho)
    _rho_body_Ccm3 = -q * N_A * 1e-6
    _rho_ref = abs(_rho_body_Ccm3) if abs(_rho_body_Ccm3) > 1e-20 else 1.0

    if _is_depleted:
        _ax_rho.fill_between([0, _t_Si_nm], _rho_body_Ccm3, 0,
                             color='blue', alpha=0.3, step='mid')
        _ax_rho.plot([0, 0, _t_Si_nm, _t_Si_nm],
                     [0, _rho_body_Ccm3, _rho_body_Ccm3, 0], 'b-', lw=2)
        _ax_rho.text(_t_Si_nm / 2, _rho_body_Ccm3 * 0.5,
                     f'$-qN_A$\n= {_rho_body_Ccm3:.2e}',
                     ha='center', va='center', fontsize=13, color='blue')

    _ax_rho.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    _ax_rho.plot(_x_metal_L, [0, 0], 'b-', lw=2)
    _ax_rho.plot([-_t_ox_nm, 0], [0, 0], 'b-', lw=2)
    _ax_rho.plot([_t_Si_nm, _t_Si_nm + _t_ox_nm], [0, 0], 'b-', lw=2)
    _ax_rho.plot(_x_metal_R, [0, 0], 'b-', lw=2)

    # Thin interface charge bars
    _bw = max(0.2, (_xlim[1] - _xlim[0]) * 0.010)
    _Q_ref = _Q_dep_half if _Q_dep_half > 1e-20 else 1.0

    def _draw_bar(_ax, x, Q, label, color):
        if abs(Q) < 1e-14:
            return
        _bh = (Q / _Q_ref) * _rho_ref
        _ax.bar(x, _bh, width=_bw, color=color, alpha=0.8,
                edgecolor='k', linewidth=1, zorder=5, align='center')
        _off = np.sign(_bh) * _rho_ref * 0.07
        _ax.text(x, _bh + _off, label, fontsize=10, ha='center',
                 fontweight='bold', color=color)

    _sgn_M = '+' if _Q_M > 0 else '\u2212'
    _draw_bar(_ax_rho, -_t_ox_nm, _Q_M, f'{_sgn_M}$Q_M$', 'darkorange')
    _draw_bar(_ax_rho, _t_Si_nm + _t_ox_nm, _Q_M, f'{_sgn_M}$Q_M$', 'darkorange')

    if abs(_Q_inv) > 1e-14:
        _draw_bar(_ax_rho, 0, _Q_inv, '$-Q_{inv}$', 'purple')
        _draw_bar(_ax_rho, _t_Si_nm, _Q_inv, '$-Q_{inv}$', 'purple')
    if abs(_Q_acc) > 1e-14:
        _draw_bar(_ax_rho, 0, _Q_acc, '$+Q_{acc}$', 'red')
        _draw_bar(_ax_rho, _t_Si_nm, _Q_acc, '$+Q_{acc}$', 'red')

    _ax_rho.text(-_t_ox_nm - _metal_w_nm / 2, _rho_body_Ccm3 * 0.3,
                 'Metal', ha='center', va='center', fontsize=11, color='#2171b5')
    _ax_rho.text(_t_Si_nm + _t_ox_nm + _metal_w_nm / 2, _rho_body_Ccm3 * 0.3,
                 'Metal', ha='center', va='center', fontsize=11, color='#2171b5')
    _ax_rho.set_xlabel('Position $x$ (nm)', fontsize=16)
    _ax_rho.set_ylabel(r'$\rho$ (C/cm³)', fontsize=16)
    _ax_rho.set_title(rf'Charge Density — {_regime}', fontsize=14)

    # ── (1) Electric field E(x) ──
    _shade(_ax_E)
    _E_OX_Vcm = _E_OX * 1e-2
    _ax_E.plot(_x_metal_L, [0, 0], 'b-', lw=2)
    _ax_E.plot(_x_ox_L, np.ones_like(_x_ox_L) * _E_OX_Vcm, 'b-', lw=2)
    if _is_depleted:
        _ax_E.plot(_x_Si_nm, _xi_full * 1e-2, 'b-', lw=2)
    else:
        _ax_E.plot([0, _t_Si_nm], [0, 0], 'b-', lw=2)
    _ax_E.plot(_x_ox_R, -np.ones_like(_x_ox_R) * _E_OX_Vcm, 'b-', lw=2)
    _ax_E.plot(_x_metal_R, [0, 0], 'b-', lw=2)
    _ax_E.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    _ax_E.plot([-_t_ox_nm]*2, [0, _E_OX_Vcm], 'b:', lw=1, alpha=0.5)
    _ax_E.plot([_t_Si_nm + _t_ox_nm]*2, [0, -_E_OX_Vcm], 'b:', lw=1, alpha=0.5)
    _ax_E.set_xlabel('Position $x$ (nm)', fontsize=16)
    _ax_E.set_ylabel(r'$\mathcal{E}$ (V/cm)', fontsize=16)
    _ax_E.set_title(r'Electric Field $\mathcal{E}(x)$', fontsize=14)

    # ── (2) Electrostatic potential V(x) ──
    _shade(_ax_V)
    _phi_ox_L = _phi_gate - _E_OX * (_x_ox_L * 1e-9 + t_ox)
    _phi_ox_R = _phi_S + _E_OX * (_x_ox_R * 1e-9 - t_Si)

    _ax_V.plot(_x_metal_L, [_phi_gate * 1e3] * 2, 'g-', lw=2)
    _ax_V.plot(_x_ox_L, _phi_ox_L * 1e3, 'g-', lw=2)
    if _is_depleted:
        _ax_V.plot(_x_Si_nm, _phi_full * 1e3, 'g-', lw=2)
    else:
        _ax_V.plot([0, _t_Si_nm], [0, 0], 'g-', lw=2)
    _ax_V.plot(_x_ox_R, _phi_ox_R * 1e3, 'g-', lw=2)
    _ax_V.plot(_x_metal_R, [_phi_gate * 1e3] * 2, 'g-', lw=2)
    _ax_V.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    if abs(_phi_gate) > 1e-4:
        _ax_V.text(-_t_ox_nm - _metal_w_nm / 2, _phi_gate * 1e3 + 2,
                   f'{_phi_gate * 1e3:.1f} mV', ha='center', va='bottom',
                   fontsize=11, color='#2171b5')
    _ax_V.set_xlabel('Position $x$ (nm)', fontsize=16)
    _ax_V.set_ylabel(r'$V$ (mV)', fontsize=16)
    _ax_V.set_title(r'Electrostatic Potential $V(x)$', fontsize=14)

    plt.close(_fig)

    # ═══════════════════════════════════════════
    # Energy band diagram (same cell, below electrostatics)
    # ═══════════════════════════════════════════
    _Eg = 1.12
    _kT = 0.02585
    _ni_m3 = 1.0e16
    _dEc_ox = 3.1
    _dEv_ox = 4.78

    _phi_B = _kT * np.log(N_A / _ni_m3)
    _gamma = np.sqrt(2 * q * eps_Si * N_A) / C_oxe
    _phi_s_max = q * N_A * t_Si**2 / (8 * eps_Si)
    _V_ox_thresh = q * N_A * t_Si / (2 * C_oxe)
    _V_t = V_FB + _phi_s_max + _V_ox_thresh

    _Vg_eff = _Vg - V_FB
    if _Vg_eff > 0.001:
        _phi_s_band = 0.1
        for _ in range(100):
            _phi_s_band = max(_phi_s_band, 1e-8)
            _f = _phi_s_band + _gamma * np.sqrt(_phi_s_band) - _Vg_eff
            _df = 1 + _gamma / (2 * np.sqrt(_phi_s_band))
            _phi_s_band -= _f / _df
            if abs(_f) < 1e-10:
                break
        _phi_s_band = min(max(_phi_s_band, 0), _phi_s_max)
    elif _Vg_eff < -0.001:
        _phi_s_band = _Vg_eff * 0.3
    else:
        _phi_s_band = 0.0

    _V_ox_band = _Vg - V_FB - _phi_s_band

    _EF = 0.0
    _Ei_bulk = _phi_B
    _Ec_bulk = _Ei_bulk + _Eg / 2
    _Ev_bulk = _Ei_bulk - _Eg / 2
    _EFm = _EF - _Vg

    _half_w = 1.5
    _xML = [-3.0, -2.5]
    _xOL = [-2.5, -_half_w]
    _xOR = [_half_w, 2.5]
    _xMR = [2.5, 3.0]
    _x_sc = np.linspace(-_half_w, _half_w, 400)

    if _phi_s_band > 1e-6:
        _W_dep_m = np.sqrt(2 * eps_Si * _phi_s_band / (q * N_A))
        _Wd = min(_half_w, _W_dep_m / (t_Si / 2) * _half_w)
        _d_L = _x_sc + _half_w
        _d_R = _half_w - _x_sc
        _bending = np.where(
            _d_L <= _Wd,
            _phi_s_band * (1 - _d_L / _Wd) ** 2,
            np.where(_d_R <= _Wd,
                     _phi_s_band * (1 - _d_R / _Wd) ** 2,
                     0.0))
    elif _phi_s_band < -1e-6:
        _bl = 0.3
        _d_L = _x_sc + _half_w
        _d_R = _half_w - _x_sc
        _bending = np.where(
            _d_L < _bl,
            _phi_s_band * (1 - _d_L / _bl) ** 2,
            np.where(_d_R < _bl,
                     _phi_s_band * (1 - _d_R / _bl) ** 2,
                     0.0))
    else:
        _bending = np.zeros_like(_x_sc)

    _Ec_sc = _Ec_bulk - _bending
    _Ev_sc = _Ev_bulk - _bending
    _Ei_sc = _Ei_bulk - _bending

    _Ec_ox_s_L = _Ec_sc[0] + _dEc_ox
    _Ev_ox_s_L = _Ev_sc[0] - _dEv_ox
    _Ec_ox_m_L = _Ec_ox_s_L - _V_ox_band
    _Ev_ox_m_L = _Ev_ox_s_L - _V_ox_band

    _Ec_ox_s_R = _Ec_sc[-1] + _dEc_ox
    _Ev_ox_s_R = _Ev_sc[-1] - _dEv_ox
    _Ec_ox_m_R = _Ec_ox_s_R - _V_ox_band
    _Ev_ox_m_R = _Ev_ox_s_R - _V_ox_band

    _fig_band, _ax = plt.subplots(figsize=(10, 6))

    _ax.fill_between(_xML, _EFm - 4.0, _EFm, color='#6baed6', alpha=0.3)
    _ax.plot(_xML, [_EFm, _EFm], 'g-', lw=2)
    _ax.text(_xML[0] + 0.05, _EFm + 0.12, '$E_{Fm}$', fontsize=14, color='green')
    _ax.fill_between(_xMR, _EFm - 4.0, _EFm, color='#6baed6', alpha=0.3)
    _ax.plot(_xMR, [_EFm, _EFm], 'g-', lw=2)

    _ax.fill_between(_xOL, [_Ev_ox_m_L, _Ev_ox_s_L], [_Ec_ox_m_L, _Ec_ox_s_L],
                     color='#ffffcc', alpha=0.5, edgecolor='none')
    _ax.plot(_xOL, [_Ec_ox_m_L, _Ec_ox_s_L], 'k-', lw=2)
    _ax.plot(_xOL, [_Ev_ox_m_L, _Ev_ox_s_L], 'k-', lw=2)

    _ax.fill_between(_xOR, [_Ev_ox_s_R, _Ev_ox_m_R], [_Ec_ox_s_R, _Ec_ox_m_R],
                     color='#ffffcc', alpha=0.5, edgecolor='none')
    _ax.plot(_xOR, [_Ec_ox_s_R, _Ec_ox_m_R], 'k-', lw=2)
    _ax.plot(_xOR, [_Ev_ox_s_R, _Ev_ox_m_R], 'k-', lw=2)

    _ax.axvspan(-_half_w, _half_w, alpha=0.05, color='blue')
    _ax.plot(_x_sc, _Ec_sc, 'b-', lw=2.5, label='$E_C$')
    _ax.plot(_x_sc, _Ev_sc, 'r-', lw=2.5, label='$E_V$')
    _ax.plot(_x_sc, _Ei_sc, 'k--', lw=1, alpha=0.5, label='$E_{Fi}$')
    _ax.plot([-_half_w, _half_w], [_EF, _EF], 'g-', lw=2, label='$E_F$')

    _ax.plot([-2.5] * 2, [_Ev_ox_m_L, _Ec_ox_m_L], 'k-', lw=1.5)
    _ax.plot([-_half_w] * 2, [_Ev_ox_s_L, _Ec_ox_s_L], 'k-', lw=1.5)
    _ax.plot([_half_w] * 2, [_Ev_ox_s_R, _Ec_ox_s_R], 'k-', lw=1.5)
    _ax.plot([2.5] * 2, [_Ev_ox_m_R, _Ec_ox_m_R], 'k-', lw=1.5)

    _ax.text(_half_w + 0.1, _Ec_bulk, '$E_C$', fontsize=13, va='center', color='blue')
    _ax.text(_half_w + 0.1, _Ev_bulk, '$E_V$', fontsize=13, va='center', color='red')
    _ax.text(_half_w + 0.1, _EF, '$E_F$', fontsize=13, va='center', color='green')

    if abs(_phi_s_band) > 0.05:
        _xann = -1.2
        _ax.annotate('', xy=(_xann, _Ec_bulk), xytext=(_xann, _Ec_sc[0]),
                     arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
        _ax.text(_xann + 0.12, (_Ec_sc[0] + _Ec_bulk) / 2, r'$q\phi_s$',
                 fontsize=14, color='darkgreen', va='center')

    if abs(_V_ox_band) > 0.1:
        _cap = 0.06
        _xvox = -2.3
        _ax.plot([_xvox] * 2, [_Ec_ox_m_L, _Ec_ox_s_L], color='darkorange', lw=1.5)
        _ax.plot([_xvox - _cap, _xvox + _cap], [_Ec_ox_m_L] * 2, color='darkorange', lw=1.5)
        _ax.plot([_xvox - _cap, _xvox + _cap], [_Ec_ox_s_L] * 2, color='darkorange', lw=1.5)
        _ax.text(_xvox - 0.12, (_Ec_ox_m_L + _Ec_ox_s_L) / 2, r'$qV_{ox}$',
                 fontsize=14, color='darkorange', va='center', ha='right')

    if abs(_Vg) > 0.1:
        _xvg = -2.75
        _cap = 0.06
        _ax.plot([_xvg] * 2, [_EFm, _EF], color='darkred', lw=1.5)
        _ax.plot([_xvg - _cap, _xvg + _cap], [_EFm] * 2, color='darkred', lw=1.5)
        _ax.plot([_xvg - _cap, _xvg + _cap], [_EF] * 2, color='darkred', lw=1.5)
        _ax.text(_xvg - 0.15, (_EFm + _EF) / 2, r'$qV_G$',
                 fontsize=14, color='darkred', va='center', ha='right')

    _cap = 0.06
    _xpB = 0.9
    _ax.plot([_xpB] * 2, [_EF, _Ei_bulk], color='purple', lw=1.5)
    _ax.plot([_xpB - _cap, _xpB + _cap], [_EF] * 2, color='purple', lw=1.5)
    _ax.plot([_xpB - _cap, _xpB + _cap], [_Ei_bulk] * 2, color='purple', lw=1.5)
    _ax.text(_xpB + 0.12, (_EF + _Ei_bulk) / 2, r'$q\phi_B$',
             fontsize=14, color='purple', va='center')

    if _Vg < V_FB - 0.02:
        _regime_band = "Accumulation"; _regime_color = '#d62728'
    elif abs(_Vg - V_FB) <= 0.02:
        _regime_band = "Flat Band"; _regime_color = '#7f7f7f'
    elif _Vg < _V_t:
        _regime_band = "Depletion"; _regime_color = '#2ca02c'
    else:
        _regime_band = "Inversion"; _regime_color = '#1f77b4'

    _ax.text(0, max(_Ec_ox_s_L, _Ec_ox_m_L) + 0.15, _regime_band,
             fontsize=18, ha='center', fontweight='bold', color=_regime_color,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=_regime_color, alpha=0.9))

    _ax.text(-2.0, max(_Ec_ox_m_L, _Ec_ox_s_L) + 0.15,
             '$E_{c,ox}$', fontsize=13, ha='center')
    _ax.text(-2.0, min(_Ev_ox_m_L, _Ev_ox_s_L) - 0.3,
             '$E_{v,ox}$', fontsize=13, ha='center')

    _y_label = min(_Ev_ox_s_L, _Ev_ox_m_L, _Ev_ox_s_R, _Ev_ox_m_R, _EFm) - 1.0
    _ax.text(-2.75, _y_label, 'Metal\n(Gate)', fontsize=14,
             ha='center', fontweight='bold')
    _ax.text(0, _y_label, 'Si Channel', fontsize=14,
             ha='center', fontweight='bold')
    _ax.text(2.75, _y_label, 'Metal\n(Gate)', fontsize=14,
             ha='center', fontweight='bold')

    _y_lo = min(_Ev_ox_s_L, _Ev_ox_m_L, _EFm) - 1.5
    _y_hi = max(_Ec_ox_s_L, _Ec_ox_m_L, _EFm + 1.0) + 0.8
    _ax.set_xlim(-3.3, 3.3)
    _ax.set_ylim(_y_lo, _y_hi)
    _ax.set_ylabel('Energy (eV)', fontsize=16)
    _ax.set_title('FinFET Energy Band Diagram (Symmetric DG)',
                  fontsize=16, fontweight='bold')
    _ax.set_xticks([])
    _ax.tick_params(labelsize=14)
    for _sp in ['top', 'right', 'bottom']:
        _ax.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig_band)

    # ── Display ──
    _controls_row1 = mo.hstack([Vg_slider, V_FB_slider, N_A_slider], justify="start")
    _controls_row2 = mo.hstack([t_Si_slider, t_ox_slider, eps_ox_slider], justify="start")

    _C_oxe_uF_cm2 = C_oxe * 1e2
    _xi_S_Vcm = _xi_S * 1e-2

    _info = mo.md(rf"""
    | **Regime** | $V_G$ | $V_{{FB}}$ | $V_t$ |
    |:--:|:--:|:--:|:--:|
    | {_regime} | {_Vg:.2f} V | {V_FB:.2f} V | {V_t_finfet:.4f} V |
    | **$C_{{oxe}}$** | **$\mathcal{{E}}_S$** | **$\phi_s$** | **$V_{{ox}}$** |
    | {_C_oxe_uF_cm2:.3f} µF/cm² | {_xi_S_Vcm:.2e} V/cm | {_phi_S * 1e3:.3f} mV | {_V_OX * 1e3:.3f} mV |
    | **$Q_M$** | **$Q_{{inv}}$** | **$Q_{{acc}}$** | |
    | {_Q_M:.2e} C/m² | {abs(_Q_inv):.2e} C/m² | {abs(_Q_acc):.2e} C/m² | |
    """)

    mo.vstack([
        mo.md("## FinFET Electrostatics: Charge, Field, Potential & Bands"),
        _controls_row1,
        _controls_row2,
        _info,
        mo.as_html(_fig),
        mo.as_html(_fig_band),
    ])
    return


@app.cell
def fdsoi_interactive_intro(mo):
    mo.md(r"""
    ## Interactive FD-SOI Electrostatics

    Explore the electrostatics of a **Fully-Depleted SOI** with independent front gate ($V_{GF}$) and back gate ($V_{GB}$) control.

    **Structure:** &ensp; Front gate &ensp;|&ensp; Front oxide ($t_{of}$) &ensp;|&ensp; Si body ($t_{Si}$) &ensp;|&ensp; BOX ($t_{BOX}$) &ensp;|&ensp; Back gate

    The coupled-gate equations for the front and back surface potentials ($\phi_{sf}$, $\phi_{sb}$):

    $$V_{GF} - V_{FB} = \phi_{sf}\!\left(1 + \frac{C_{Si}}{C_{of}}\right) - \phi_{sb}\,\frac{C_{Si}}{C_{of}} + \frac{qN_A t_{Si}}{2C_{of}}$$

    $$V_{GB} - V_{FB} = \phi_{sb}\!\left(1 + \frac{C_{Si}}{C_{ob}}\right) - \phi_{sf}\,\frac{C_{Si}}{C_{ob}} + \frac{qN_A t_{Si}}{2C_{ob}}$$

    where $C_{Si} = \varepsilon_{Si}/t_{Si}$, $C_{of} = \varepsilon_{ox}/t_{of}$, $C_{ob} = \varepsilon_{ox}/t_{BOX}$.

    Solving these simultaneously gives $\phi_{sf}$ and $\phi_{sb}$, from which the potential, field, and charge profiles follow. Surface potentials are pinned at $2\phi_B$ (inversion) or $0$ (accumulation) when the full-depletion solution exceeds those bounds.

    ### Back body bias for $V_t$ control
    - $V_{GB} = 0$ → back in depletion (typical operation)
    - $V_{GB} > 0$ → Forward body bias → lowers front $V_t$ (higher speed operation)
    - $V_{GB} < 0$ → Reverse body bias → raises front $V_t$ (leakage reduction)
    """)
    return


@app.cell
def fdsoi_interactive_sliders(mo):
    V_GF_slider = mo.ui.slider(
        start=-1.5, stop=2.0, step=0.05, value=0.5,
        label=r"Front gate $V_{GF}$ (V)"
    )
    V_GB_slider = mo.ui.slider(
        start=-2.0, stop=2.0, step=0.05, value=0.0,
        label=r"Back gate $V_{GB}$ (V)"
    )
    t_of_soi_slider = mo.ui.slider(
        start=0.5, stop=5.0, step=0.1, value=1.0,
        label=r"Front oxide $t_{of}$ (nm)"
    )
    t_ob_soi_slider = mo.ui.slider(
        start=5, stop=50, step=1, value=25,
        label=r"BOX $t_{BOX}$ (nm)"
    )
    t_Si_soi_slider = mo.ui.slider(
        start=3, stop=20, step=0.5, value=7,
        label=r"Si body $t_{Si}$ (nm)"
    )
    N_A_soi_slider = mo.ui.slider(
        start=14, stop=19, step=0.1, value=16,
        label=r"$\log_{10}(N_A)$ (cm$^{-3}$)"
    )
    V_FB_soi_slider = mo.ui.slider(
        start=-0.5, stop=0.5, step=0.05, value=0.0,
        label=r"$V_{FB}$ (V)"
    )
    return (
        N_A_soi_slider,
        V_FB_soi_slider,
        V_GB_slider,
        V_GF_slider,
        t_Si_soi_slider,
        t_ob_soi_slider,
        t_of_soi_slider,
    )


@app.cell(hide_code=True)
def fdsoi_interactive_plot(
    N_A_soi_slider,
    V_FB_soi_slider,
    V_GB_slider,
    V_GF_slider,
    eps_0,
    mo,
    np,
    plt,
    q,
    t_Si_soi_slider,
    t_ob_soi_slider,
    t_of_soi_slider,
):
    _V_GF = V_GF_slider.value
    _V_GB = V_GB_slider.value
    _t_of = t_of_soi_slider.value * 1e-9
    _t_ob = t_ob_soi_slider.value * 1e-9
    _t_Si = t_Si_soi_slider.value * 1e-9
    _N_A = 10**N_A_soi_slider.value * 1e6
    _V_FB = V_FB_soi_slider.value
    _eps_ox = 3.9 * eps_0
    _eps_Si = 11.7 * eps_0
    _kT = 0.02585
    _ni_m3 = 1.0e16

    _C_of = _eps_ox / _t_of
    _C_ob = _eps_ox / _t_ob
    _C_Si = _eps_Si / _t_Si

    _alpha_f = _C_Si / _C_of
    _alpha_b = _C_Si / _C_ob
    _D_f = q * _N_A * _t_Si / (2 * _C_of)
    _D_b = q * _N_A * _t_Si / (2 * _C_ob)
    _phi_B = _kT * np.log(_N_A / _ni_m3) if _N_A > _ni_m3 else 0
    _phi_inv = 2 * _phi_B

    # ── Solve coupled-gate equations (full depletion) ──
    _det = 1 + _alpha_f + _alpha_b
    _b1 = _V_GF - _V_FB - _D_f
    _b2 = _V_GB - _V_FB - _D_b
    _phi_sf_raw = ((1 + _alpha_b) * _b1 + _alpha_f * _b2) / _det
    _phi_sb_raw = (_alpha_b * _b1 + (1 + _alpha_f) * _b2) / _det

    # ── Pin surface potentials and determine regimes ──
    _phi_sf = _phi_sf_raw
    _phi_sb = _phi_sb_raw
    _front_regime = "Depletion"
    _back_regime = "Depletion"
    _sf_pinned = False
    _sb_pinned = False

    if _phi_sf > _phi_inv:
        _phi_sf = _phi_inv; _sf_pinned = True; _front_regime = "Inversion"
    elif _phi_sf < 0:
        _phi_sf = 0.0; _sf_pinned = True; _front_regime = "Accumulation"

    if _phi_sb > _phi_inv:
        _phi_sb = _phi_inv; _sb_pinned = True; _back_regime = "Inversion"
    elif _phi_sb < 0:
        _phi_sb = 0.0; _sb_pinned = True; _back_regime = "Accumulation"

    if _sf_pinned and not _sb_pinned:
        _phi_sb = (_b2 + _alpha_b * _phi_sf) / (1 + _alpha_b)
        if _phi_sb > _phi_inv:
            _phi_sb = _phi_inv; _back_regime = "Inversion"
        elif _phi_sb < 0:
            _phi_sb = 0.0; _back_regime = "Accumulation"
    elif _sb_pinned and not _sf_pinned:
        _phi_sf = (_b1 + _alpha_f * _phi_sb) / (1 + _alpha_f)
        if _phi_sf > _phi_inv:
            _phi_sf = _phi_inv; _front_regime = "Inversion"
        elif _phi_sf < 0:
            _phi_sf = 0.0; _front_regime = "Accumulation"

    _both_acc = (_front_regime == "Accumulation" and _back_regime == "Accumulation")

    # ── Electrostatics in Si body ──
    _x_Si = np.linspace(0, _t_Si, 400)
    if _both_acc:
        _phi_Si = np.zeros_like(_x_Si)
        _E_Si = np.zeros_like(_x_Si)
    else:
        _C1 = (_phi_sb - _phi_sf) / _t_Si - q * _N_A * _t_Si / (2 * _eps_Si)
        _phi_Si = (q * _N_A / (2 * _eps_Si)) * _x_Si**2 + _C1 * _x_Si + _phi_sf
        _E_Si = -(q * _N_A / _eps_Si) * _x_Si - _C1

    # ── Gate potentials & oxide fields ──
    _phi_gate_f = _V_GF - _V_FB
    _phi_gate_b = _V_GB - _V_FB

    _E_ox_f = (_phi_gate_f - _phi_sf) / _t_of
    _E_box = (_phi_sb - _phi_gate_b) / _t_ob

    # ── Interface & gate charges (C/m²) ──
    _Q_M_f = _C_of * (_V_GF - _V_FB - _phi_sf)
    _Q_M_b = _C_ob * (_V_GB - _V_FB - _phi_sb)
    _E_sf = _E_Si[0] if not _both_acc else 0.0
    _E_sb = _E_Si[-1] if not _both_acc else 0.0
    _Q_sf = _eps_Si * _E_sf - _Q_M_f
    _Q_sb = -_Q_M_b - _eps_Si * _E_sb

    # ── Convert to nm for plotting ──
    _t_of_nm = _t_of * 1e9
    _t_ob_nm = _t_ob * 1e9
    _t_Si_nm = _t_Si * 1e9
    _mw_f = max(1.5, _t_of_nm * 0.6)
    _mw_b = max(1.5, _t_ob_nm * 0.06)

    _x_Si_nm = _x_Si * 1e9
    _x_of_nm = np.linspace(-_t_of_nm, 0, 50)
    _x_box_nm = np.linspace(_t_Si_nm, _t_Si_nm + _t_ob_nm, 50)
    _x_mf_nm = np.array([-_t_of_nm - _mw_f, -_t_of_nm])
    _x_mb_nm = np.array([_t_Si_nm + _t_ob_nm, _t_Si_nm + _t_ob_nm + _mw_b])

    _xlim = (-_t_of_nm - _mw_f - 0.3, _t_Si_nm + _t_ob_nm + _mw_b + 0.3)

    _phi_of = _phi_gate_f + (_phi_sf - _phi_gate_f) * (_x_of_nm + _t_of_nm) / _t_of_nm
    _phi_box = _phi_sb + (_phi_gate_b - _phi_sb) * (_x_box_nm - _t_Si_nm) / _t_ob_nm

    def _shade(_ax):
        _ax.axvspan(-_t_of_nm - _mw_f, -_t_of_nm, alpha=0.25, color='#6baed6')
        _ax.axvspan(-_t_of_nm, 0, alpha=0.15, color='orange')
        _ax.axvspan(_t_Si_nm, _t_Si_nm + _t_ob_nm, alpha=0.15, color='orange')
        _ax.axvspan(_t_Si_nm + _t_ob_nm, _t_Si_nm + _t_ob_nm + _mw_b, alpha=0.25, color='#6baed6')
        for _xv in [-_t_of_nm, 0, _t_Si_nm, _t_Si_nm + _t_ob_nm]:
            _ax.axvline(x=_xv, color='gray', linestyle=':', alpha=0.4)
        _ax.set_xlim(_xlim)
        _ax.grid(True, alpha=0.3)
        _ax.tick_params(labelsize=13)

    # ═══════════════════════════════════════════
    # 1×3 figure: ρ, E, V  (matches FinFET layout)
    # ═══════════════════════════════════════════
    _fig_ev, (_ax_rho, _ax_E, _ax_V) = plt.subplots(1, 3, figsize=(18, 5))
    _fig_ev.subplots_adjust(wspace=0.35)

    # ── (0) Charge density ρ(x) ──
    _shade(_ax_rho)
    _rho_body_Ccm3 = -q * _N_A * 1e-6
    _rho_ref = abs(_rho_body_Ccm3) if abs(_rho_body_Ccm3) > 1e-20 else 1.0
    _Q_dep_total = q * _N_A * _t_Si

    if not _both_acc:
        _ax_rho.fill_between([0, _t_Si_nm], _rho_body_Ccm3, 0,
                             color='blue', alpha=0.3, step='mid')
        _ax_rho.plot([0, 0, _t_Si_nm, _t_Si_nm],
                     [0, _rho_body_Ccm3, _rho_body_Ccm3, 0], 'b-', lw=2)
        _ax_rho.text(_t_Si_nm / 2, _rho_body_Ccm3 * 0.5,
                     f'$-qN_A$\n= {_rho_body_Ccm3:.2e}',
                     ha='center', va='center', fontsize=13, color='blue')

    _ax_rho.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    _ax_rho.plot(_x_mf_nm, [0, 0], 'b-', lw=2)
    _ax_rho.plot([-_t_of_nm, 0], [0, 0], 'b-', lw=2)
    _ax_rho.plot([_t_Si_nm, _t_Si_nm + _t_ob_nm], [0, 0], 'b-', lw=2)
    _ax_rho.plot(_x_mb_nm, [0, 0], 'b-', lw=2)

    _bw = max(0.2, (_xlim[1] - _xlim[0]) * 0.010)
    _Q_ref = _Q_dep_total if _Q_dep_total > 1e-20 else 1.0

    def _draw_bar(_ax, x, Q, label, color):
        if abs(Q) < 1e-14:
            return
        _bh = (Q / _Q_ref) * _rho_ref
        _ax.bar(x, _bh, width=_bw, color=color, alpha=0.8,
                edgecolor='k', linewidth=1, zorder=5, align='center')
        _off = np.sign(_bh) * _rho_ref * 0.07
        _ax.text(x, _bh + _off, label, fontsize=10, ha='center',
                 fontweight='bold', color=color)

    _sgn_Mf = '+' if _Q_M_f > 0 else '\u2212'
    _sgn_Mb = '+' if _Q_M_b > 0 else '\u2212'
    _draw_bar(_ax_rho, -_t_of_nm, _Q_M_f, f'{_sgn_Mf}$Q_M$', 'darkorange')
    _draw_bar(_ax_rho, _t_Si_nm + _t_ob_nm, _Q_M_b, f'{_sgn_Mb}$Q_M$', 'darkorange')
    if abs(_Q_sf) > 1e-14:
        _lbl_sf = '$-Q_{inv}$' if _Q_sf < 0 else '$+Q_{acc}$'
        _col_sf = 'purple' if _Q_sf < 0 else 'red'
        _draw_bar(_ax_rho, 0, _Q_sf, _lbl_sf, _col_sf)
    if abs(_Q_sb) > 1e-14:
        _lbl_sb = '$-Q_{inv}$' if _Q_sb < 0 else '$+Q_{acc}$'
        _col_sb = 'purple' if _Q_sb < 0 else 'red'
        _draw_bar(_ax_rho, _t_Si_nm, _Q_sb, _lbl_sb, _col_sb)

    _ax_rho.text(-_t_of_nm - _mw_f / 2, _rho_body_Ccm3 * 0.3,
                 'Metal', ha='center', va='center', fontsize=11, color='#2171b5')
    _ax_rho.text(_t_Si_nm + _t_ob_nm + _mw_b / 2, _rho_body_Ccm3 * 0.3,
                 'Metal', ha='center', va='center', fontsize=11, color='#2171b5')
    _ax_rho.set_xlabel('Position $x$ (nm)', fontsize=16)
    _ax_rho.set_ylabel(r'$\rho$ (C/cm³)', fontsize=16)
    _ax_rho.set_title(rf'Charge Density — front: {_front_regime}, back: {_back_regime}',
                      fontsize=14)

    # ── (1) Electric field E(x) ──
    _shade(_ax_E)
    _ax_E.plot(_x_mf_nm, [0, 0], 'b-', lw=2)
    _ax_E.plot(_x_of_nm, np.ones_like(_x_of_nm) * _E_ox_f * 1e-2, 'b-', lw=2)
    if not _both_acc:
        _ax_E.plot(_x_Si_nm, _E_Si * 1e-2, 'b-', lw=2)
    else:
        _ax_E.plot([0, _t_Si_nm], [0, 0], 'b-', lw=2)
    _ax_E.plot(_x_box_nm, np.ones_like(_x_box_nm) * _E_box * 1e-2, 'b-', lw=2)
    _ax_E.plot(_x_mb_nm, [0, 0], 'b-', lw=2)
    _ax_E.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for _xi, _ev in [(-_t_of_nm, _E_ox_f * 1e-2), (_t_Si_nm + _t_ob_nm, _E_box * 1e-2)]:
        _ax_E.plot([_xi, _xi], [0, _ev], 'b:', lw=1, alpha=0.5)
    _ax_E.set_xlabel('Position $x$ (nm)', fontsize=16)
    _ax_E.set_ylabel(r'$\mathcal{E}$ (V/cm)', fontsize=16)
    _ax_E.set_title(r'Electric Field $\mathcal{E}(x)$', fontsize=14)

    # ── (2) Electrostatic potential V(x) ──
    _shade(_ax_V)
    _ax_V.plot(_x_mf_nm, [_phi_gate_f * 1e3] * 2, 'g-', lw=2)
    _ax_V.plot(_x_of_nm, _phi_of * 1e3, 'g-', lw=2)
    if not _both_acc:
        _ax_V.plot(_x_Si_nm, _phi_Si * 1e3, 'g-', lw=2)
    else:
        _ax_V.plot([0, _t_Si_nm], [0, 0], 'g-', lw=2)
    _ax_V.plot(_x_box_nm, _phi_box * 1e3, 'g-', lw=2)
    _ax_V.plot(_x_mb_nm, [_phi_gate_b * 1e3] * 2, 'g-', lw=2)
    _ax_V.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    _ax_V.text(-_t_of_nm - _mw_f / 2, _phi_gate_f * 1e3 + 2,
               f'{_phi_gate_f * 1e3:.1f} mV', ha='center', va='bottom',
               fontsize=11, color='#2171b5')
    _ax_V.text(_t_Si_nm + _t_ob_nm + _mw_b / 2, _phi_gate_b * 1e3 + 2,
               f'{_phi_gate_b * 1e3:.1f} mV', ha='center', va='bottom',
               fontsize=11, color='#2171b5')
    _ax_V.set_xlabel('Position $x$ (nm)', fontsize=16)
    _ax_V.set_ylabel(r'$V$ (mV)', fontsize=16)
    _ax_V.set_title(r'Electrostatic Potential $V(x)$', fontsize=14)

    plt.close(_fig_ev)

    # ═══════════════════════════════════════════
    # Separate band diagram (matches FinFET style)
    # ═══════════════════════════════════════════
    _Eg = 1.12
    _dEc_ox = 3.1
    _dEv_ox = 4.78
    _EF = 0.0
    _Ei_bulk = _phi_B
    _Ec_bulk = _Ei_bulk + _Eg / 2
    _Ev_bulk = _Ei_bulk - _Eg / 2

    _phi_Si_plot = _phi_Si if not _both_acc else np.zeros_like(_x_Si)
    _E_C = _Ec_bulk - _phi_Si_plot
    _E_V = _Ev_bulk - _phi_Si_plot
    _E_Fi = _Ei_bulk - _phi_Si_plot

    _V_ox_f = _phi_gate_f - _phi_sf
    _V_box = _phi_gate_b - _phi_sb
    _EFm_f = _EF - _V_FB - _phi_gate_f
    _EFm_b = _EF - _V_FB - _phi_gate_b

    _Ec_ox_Si_f = _E_C[0] + _dEc_ox
    _Ev_ox_Si_f = _E_V[0] - _dEv_ox
    _Ec_ox_M_f = _Ec_ox_Si_f - _V_ox_f
    _Ev_ox_M_f = _Ev_ox_Si_f - _V_ox_f

    _Ec_ox_Si_b = _E_C[-1] + _dEc_ox
    _Ev_ox_Si_b = _E_V[-1] - _dEv_ox
    _Ec_ox_M_b = _Ec_ox_Si_b - _V_box
    _Ev_ox_M_b = _Ev_ox_Si_b - _V_box

    _xO_f = [-_t_of_nm, 0]
    _xO_b = [_t_Si_nm, _t_Si_nm + _t_ob_nm]
    _xM_f = [-_t_of_nm - _mw_f, -_t_of_nm]
    _xM_b = [_t_Si_nm + _t_ob_nm, _t_Si_nm + _t_ob_nm + _mw_b]

    _fig_band, _ax = plt.subplots(figsize=(10, 6))

    _ax.fill_between(_xM_f, _EFm_f - 4.0, _EFm_f, color='#6baed6', alpha=0.3)
    _ax.plot(_xM_f, [_EFm_f] * 2, 'g-', lw=2)
    _ax.text(_xM_f[0] + 0.2, _EFm_f + 0.12, '$E_{Fm}$', fontsize=14, color='green')
    _ax.fill_between(_xM_b, _EFm_b - 4.0, _EFm_b, color='#6baed6', alpha=0.3)
    _ax.plot(_xM_b, [_EFm_b] * 2, 'g-', lw=2)

    _ax.fill_between(_xO_f, [_Ev_ox_M_f, _Ev_ox_Si_f],
                     [_Ec_ox_M_f, _Ec_ox_Si_f],
                     color='#ffffcc', alpha=0.5, edgecolor='none')
    _ax.plot(_xO_f, [_Ec_ox_M_f, _Ec_ox_Si_f], 'k-', lw=2)
    _ax.plot(_xO_f, [_Ev_ox_M_f, _Ev_ox_Si_f], 'k-', lw=2)

    _ax.fill_between(_xO_b, [_Ev_ox_Si_b, _Ev_ox_M_b],
                     [_Ec_ox_Si_b, _Ec_ox_M_b],
                     color='#ffffcc', alpha=0.5, edgecolor='none')
    _ax.plot(_xO_b, [_Ec_ox_Si_b, _Ec_ox_M_b], 'k-', lw=2)
    _ax.plot(_xO_b, [_Ev_ox_Si_b, _Ev_ox_M_b], 'k-', lw=2)

    _ax.axvspan(0, _t_Si_nm, alpha=0.05, color='blue')
    _ax.plot(_x_Si_nm, _E_C, 'b-', lw=2.5, label='$E_C$')
    _ax.plot(_x_Si_nm, _E_V, 'r-', lw=2.5, label='$E_V$')
    _ax.plot(_x_Si_nm, _E_Fi, 'k--', lw=1, alpha=0.5, label='$E_{Fi}$')
    _ax.axhline(y=_EF, color='green', linestyle=':', lw=1.5, label='$E_F$')

    _ax.plot([-_t_of_nm] * 2, [_Ev_ox_M_f, _Ec_ox_M_f], 'k-', lw=1.5)
    _ax.plot([0] * 2, [_Ev_ox_Si_f, _Ec_ox_Si_f], 'k-', lw=1.5)
    _ax.plot([_t_Si_nm] * 2, [_Ev_ox_Si_b, _Ec_ox_Si_b], 'k-', lw=1.5)
    _ax.plot([_t_Si_nm + _t_ob_nm] * 2, [_Ev_ox_M_b, _Ec_ox_M_b], 'k-', lw=1.5)

    _ax.text(_t_Si_nm + 0.2, _Ec_bulk, '$E_C$', fontsize=13, va='center', color='blue')
    _ax.text(_t_Si_nm + 0.2, _Ev_bulk, '$E_V$', fontsize=13, va='center', color='red')
    _ax.text(_t_Si_nm + 0.2, _EF, '$E_F$', fontsize=13, va='center', color='green')

    _bend_f = abs(_phi_sf)
    if _bend_f > 0.005:
        _xann_f = _t_Si_nm * 0.08
        _ax.annotate('', xy=(_xann_f, _Ec_bulk), xytext=(_xann_f, _E_C[0]),
                     arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
        _ax.text(_xann_f + 0.3, (_E_C[0] + _Ec_bulk) / 2, r'$q\phi_{sf}$',
                 fontsize=14, color='darkgreen', va='center')
    _bend_b = abs(_phi_sb)
    if _bend_b > 0.005:
        _xann_b = _t_Si_nm * 0.92
        _ax.annotate('', xy=(_xann_b, _Ec_bulk), xytext=(_xann_b, _E_C[-1]),
                     arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
        _ax.text(_xann_b - 0.3, (_E_C[-1] + _Ec_bulk) / 2, r'$q\phi_{sb}$',
                 fontsize=14, color='darkgreen', va='center', ha='right')

    _ax.text(-_t_of_nm / 2, max(_Ec_ox_M_f, _Ec_ox_Si_f) + 0.15,
             '$E_{c,ox}$', fontsize=13, ha='center')
    _ax.text(-_t_of_nm / 2, min(_Ev_ox_M_f, _Ev_ox_Si_f) - 0.3,
             '$E_{v,ox}$', fontsize=13, ha='center')
    _ax.text(_t_Si_nm + _t_ob_nm / 2, max(_Ec_ox_Si_b, _Ec_ox_M_b) + 0.15,
             '$E_{c,BOX}$', fontsize=13, ha='center')

    _y_label = min(_Ev_ox_Si_f, _Ev_ox_M_f, _Ev_ox_Si_b, _Ev_ox_M_b, _EFm_f, _EFm_b) - 1.0
    _ax.text(-_t_of_nm - _mw_f / 2, _y_label, 'Front\ngate', fontsize=14,
             ha='center', fontweight='bold', color='#2171b5')
    _ax.text(_t_Si_nm / 2, _y_label, 'Si Channel', fontsize=14,
             ha='center', fontweight='bold')
    _ax.text(_t_Si_nm + _t_ob_nm + _mw_b / 2, _y_label, 'Back\ngate', fontsize=14,
             ha='center', fontweight='bold', color='#2171b5')

    _y_lo = min(_Ev_ox_Si_f, _Ev_ox_M_f, _EFm_f, _Ev_ox_Si_b, _Ev_ox_M_b, _EFm_b) - 1.5
    _y_hi = max(_Ec_ox_Si_f, _Ec_ox_M_f, _Ec_ox_Si_b, _Ec_ox_M_b, _EFm_f + 1.0) + 0.8
    _ax.set_xlim(_xlim)
    _ax.set_ylim(_y_lo, _y_hi)
    _ax.set_ylabel('Energy (eV)', fontsize=16)
    _ax.set_title('FD-SOI Energy Band Diagram',
                  fontsize=16, fontweight='bold')
    _ax.set_xticks([])
    _ax.tick_params(labelsize=14)
    for _sp in ['top', 'right', 'bottom']:
        _ax.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig_band)

    # ── Display ──
    _controls_row1 = mo.hstack([V_GF_slider, V_GB_slider, V_FB_soi_slider], justify="start")
    _controls_row2 = mo.hstack([t_of_soi_slider, t_ob_soi_slider, t_Si_soi_slider, N_A_soi_slider],
                               justify="start")

    _phi_sf_mV = _phi_sf * 1e3
    _phi_sb_mV = _phi_sb * 1e3

    _V_T_front = (_V_FB + _D_f + _phi_inv
                  + _alpha_f / (1 + _alpha_b) * (_phi_inv - _V_GB + _V_FB + _D_b))

    _info = mo.md(rf"""
    | | Front surface | Back surface |
    |:--|:--:|:--:|
    | **Regime** | {_front_regime} | {_back_regime} |
    | $\phi_s$ | {_phi_sf_mV:.1f} mV | {_phi_sb_mV:.1f} mV |
    | $Q_{{sheet}}$ | {_Q_sf:.2e} C/m² | {_Q_sb:.2e} C/m² |
    | $Q_M$ (gate) | {_Q_M_f:.2e} C/m² | {_Q_M_b:.2e} C/m² |

    $\phi_B$ = {_phi_B*1e3:.1f} mV &emsp; $2\phi_B$ = {_phi_inv*1e3:.1f} mV &emsp;
    $V_T$ (front, at $V_{{GB}}$={_V_GB:.2f} V) = {_V_T_front:.3f} V
    """)

    mo.vstack([
        mo.md("## Interactive FD-SOI Electrostatics"),
        _controls_row1,
        _controls_row2,
        _info,
        mo.as_html(_fig_ev),
        mo.as_html(_fig_band),
    ])
    return


@app.cell
def plot_vt_scaling(N_A, V_FB, eps_OX, eps_Si, mo, np, plt, q, t_ox):

    _t_Si_range = np.linspace(3, 20, 100) * 1e-9  # 3 to 20 nm
    _C_oxe = eps_OX / t_ox

    _V_OX_fin = q * N_A * _t_Si_range / (2 * _C_oxe)
    _phi_S_fin = q * N_A * _t_Si_range**2 / (8 * eps_Si)
    _V_t_fin = V_FB + _V_OX_fin + _phi_S_fin

    _V_OX_fdsoi = q * N_A * _t_Si_range / _C_oxe
    _phi_S_fdsoi = q * N_A * _t_Si_range**2 / (2 * eps_Si)
    _V_t_fdsoi = V_FB + _V_OX_fdsoi + _phi_S_fdsoi

    _fig4, _ax = plt.subplots(figsize=(10, 6))

    _ax.plot(_t_Si_range*1e9, (_V_t_fin - V_FB)*1e3, 'b-', linewidth=2, 
             label='FinFET: $V_t - V_{FB}$')
    _ax.plot(_t_Si_range*1e9, (_V_t_fdsoi - V_FB)*1e3, 'r--', linewidth=2, 
             label='FDSOI: $V_t - V_{FB}$')

    _ax.axvspan(5, 10, alpha=0.2, color='green', label='Modern FinFET range')

    _ax.set_xlabel(r'Silicon Body Thickness $t_{Si}$ (nm)', fontsize=16)
    _ax.set_ylabel('$V_t - V_{FB}$ (mV)', fontsize=16)
    _ax.set_title(f'Threshold Voltage ($N_A$ = {N_A*1e-6:.0e} cm⁻³)', fontsize=16)
    _ax.legend(fontsize=16)
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(3, 20)

    plt.close(_fig4)

    _header = mo.md(r"""
            ## Threshold Voltage Scaling and Variation with Body Thickness

            - **FinFET:** $V_t - V_{FB} = \dfrac{qN_A t_{Si}}{2C_{oxe}} + \dfrac{qN_A t_{Si}^2}{8\epsilon_s}$
            - **FDSOI:** $V_t - V_{FB} = \dfrac{qN_A t_{Si}}{C_{oxe}} + \dfrac{qN_A t_{Si}^2}{2\epsilon_s}$ 
            - **Bulk:** $V_t - V_{FB} = 2\phi_B + \dfrac{\sqrt{2\epsilon_s q N_A (2\phi_B)}}{C_{oxe}}$, where $\phi_B = (kT/q)\ln(N_A/n_i)$ 

            For thin bodies with light doping, $V_t \approx V_{FB}$ (see plot below). Therefore, threshold voltage variation due to process-induced $\Delta t_{Si}$ is minimal

            In bulk MOSFETs with $N_A \sim 10^{{16}}$ cm$^{{-3}}$, $(V_t - V_{FB}) \sim 0.5- 1$ V. $V_t$ is sensitive to variations in doping $N_A$, which can be O(1). So $V_t$ variability in bulk MOSFET is of the order of ~100mV.

            **Intuition:**  $V_t$ is set by the total charge needed to deplete the body and reach inversion. The body in the FDSOI and FinFETs is fully depleted, and the number of charges that needs to be moved to deplete is restricted because of the volume set by $t_{Si}$ (which is thin). In contrast, bulk MOSFETs have a thick body, so more charges need to be moved to deplete and invert the surface. So $V_t - V_{FB}$ for the FD-SOI and FinFET is lower than bulk MOSFET. 

            FD-SOI and FinFETs offer better charge control by the gate voltage than bulk MOSFET.


            """)


    mo.vstack([
        _header,
        mo.as_html(_fig4),
    ])
    return


if __name__ == "__main__":
    app.run()
