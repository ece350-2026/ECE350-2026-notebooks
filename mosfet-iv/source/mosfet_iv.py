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
    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/mosfet-iv/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    mo.md(r"""
    # MOSFET $I_{DS}$–$V_{DS}$: Derivation and Modifications
    **ECE350 Lecture 34**

    Hu, Chapters 6, 7

    This notebook derives the MOSFET current-voltage relationship from first principles, introduces four important modifications to the ideal model, and covers key short-channel effects.

    **Contents:**

    1. Surface Mobility
    2. Derivation of the Square-Law IV Relation
    3. Transconductance
    4. Effective Oxide Capacitance
    5. Body Effect
    6. Velocity Saturation
    7. Parasitic Source-Drain Resistance
    8. Channel Length Modulation
    9. Interactive IV Comparison (all effects)
    10. Subthreshold Conduction
    11. $V_t$ Roll-Off and DIBL
    """)
    return IMAGE_BASE, eps_0, eps_Si, kT_300, mo, ni_Si, np, plt, q


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _img = mo.image(f"{IMAGE_BASE}/surface-mobility.png", width=400)

    mo.md(rf"""
    ## 1. Surface Mobility

    Carriers in the MOSFET channel are confined to a thin layer at the oxide-semiconductor interface. The **surface mobility** $\mu_{{ns}}$ (for electrons) is lower than the bulk mobility due to additional surface scattering, and depends on $V_{{GS}}$, $V_t$, and $t_{{oxe}}$.

    {mo.as_html(_img)}


    """)
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _coord_img = mo.image(f"{IMAGE_BASE}/MOSFET-coord.png", width=500)

    mo.md(rf"""
    ## 2. Derivation of MOSFET IV Relation

    The cross-section area for current flow is the channel width $W$ times the inversion layer thickness $h(x)$.

    The drift current density in the channel is:
    $$J_N = q n \mu_{{ns}} \mathcal{{E}} = q n \mu_{{ns}} \left(-\frac{{dV_{{CS}}}}{{dx}}\right)$$

    where $V_{{CS}}(x)$ is the channel-to-source voltage at position $x$ along the channel.

    ### Coordinate System

    {mo.as_html(_coord_img)}

    - $x$: along the channel from source ($x=0$) to drain ($x=L$)
    - $y$: perpendicular to M-O-S
    - $z$: along the channel width $W$
    - $h(x)$: inversion layer thickness at position $x$
    - $V_{{CS}}(x)$: channel-to-source voltage at position $x$ ($V_{{CS}}(0) = 0$, $V_{{CS}}(L) = V_{{DS}}$)

    ### Assumptions
    1. Current in the channel is due to **drift only**
    2. No current flows through the gate oxide
    3. Channel mobility $\mu_{{ns}}$ is constant along the channel
    4. $V_{{GS}} > V_t$ (above threshold)

    ### Step 1: Drift Current Density

    The drift current density in the channel and the lateral electric field are:
    $$J_N = q \mu_{{ns}} n \,\mathcal{{E}}_x, \qquad \mathcal{{E}}_x = -\frac{{dV_{{CS}}}}{{dx}}$$

    ### Step 2: Integrate Over the Cross-Section

    The drain current is $J_N$ integrated over the cross-section area in the $y$–$z$ plane. $W$ is the channel width (along $z$), and $h(x)$ is the channel height:

    $$I_{{DS}} = -W \int_0^{{h(x)}} J_N \, dy = W \mu_{{ns}} \frac{{dV_{{CS}}}}{{dx}} \int_0^{{h(x)}} q\, n(x,y)\, dy$$

    The negative sign arises because conventional current flows in the $-x$ direction (from drain to source inside the channel).

    But the **sheet charge density** $Q_{{inv}}$ is the volume charge density integrated over the thickness (height) of the channel:
    $$Q_{{inv}}(x) = -\int_0^{{h(x)}} q\, n(x,y)\, dy$$

    Therefore:
    $$\boxed{{I_{{DS}} = -W \mu_{{ns}} \frac{{dV_{{CS}}}}{{dx}}\, Q_{{inv}}(x)}}$$

    ### Step 3: Rearrange and Integrate Both Sides

    Rearrange and integrate: left side over $x$ from $0$ to $L$, right side over $V_{{CS}}$ from $0$ to $V_{{DS}}$:
    $$\int_0^L I_{{DS}}\, dx = -W \mu_{{ns}} \int_0^{{V_{{DS}}}} Q_{{inv}}(x)\, dV_{{CS}}$$

    Since $I_{{DS}}$ is constant along the channel (current continuity):
    $$I_{{DS}} = -\frac{{W \mu_{{ns}}}}{{L}} \int_0^{{V_{{DS}}}} Q_{{inv}}(x)\, dV_{{CS}}$$

    ### Step 4: Substitute $Q_{{inv}}$

    From the MOS capacitor analysis, the local inversion charge density is:
    $$Q_{{inv}}(x) = -C_{{oxe}}\big[V_{{GS}} - V_t - V_{{CS}}(x)\big]$$

    Substituting:
    $$I_{{DS}} = \frac{{W \mu_{{ns}} C_{{oxe}}}}{{L}} \int_0^{{V_{{DS}}}} \big(V_{{GS}} - V_t - V_{{CS}}\big)\, dV_{{CS}}$$

    $$\boxed{{I_{{DS}} = \mu_{{ns}} C_{{oxe}} \frac{{W}}{{L}}\left[(V_{{GS}} - V_t)V_{{DS}} - \frac{{1}}{{2}}V_{{DS}}^2\right]}}$$

    This is the **square-law** IV relation, valid for $0 < V_{{DS}} < V_{{Dsat}}$.
    """)
    return


@app.cell(hide_code=True)
def _(eps_0, eps_Si, kT_300, mo, ni_Si, np, plt, q):
    _Na = 1e17
    _tox = 5e-7  # cm
    _Cox = 3.9 * eps_0 / _tox
    _phi_B = kT_300 * np.log(_Na / ni_Si)
    _Vt = 2 * _phi_B + np.sqrt(2 * eps_Si * eps_0 * q * _Na * 2 * _phi_B) / _Cox
    _mu_ns = 300  # cm^2/V·s
    _W = 10e-4  # 10 µm
    _L = 1e-4   # 1 µm

    _Vds = np.linspace(0, 3.5, 500)
    _Vgs_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
    _colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    _Idsat_list = []
    _Vov_list = []

    for _i, _vgs in enumerate(_Vgs_vals):
        _Vov = _vgs - _Vt
        if _Vov <= 0:
            continue
        _Vdsat = _Vov
        _ids_triode = _mu_ns * _Cox * _W / _L * (_Vov * _Vds - 0.5 * _Vds**2)
        _Idsat = 0.5 * _mu_ns * _Cox * _W / _L * _Vov**2
        _ids = np.where(_Vds < _Vdsat, _ids_triode, _Idsat)
        _ids = np.maximum(_ids, 0)

        _ax1.plot(_Vds, _ids * 1e3, '-', color=_colors[_i], lw=2.5,
                  label=f'$V_{{GS}}$ = {_vgs:.1f} V')
        _ax1.plot(_Vdsat, _Idsat * 1e3, 'o', color=_colors[_i], markersize=7, zorder=5)

        _Idsat_list.append(_Idsat)
        _Vov_list.append(_Vov)

    _ax1.set_xlabel('$V_{DS}$ (V)', fontsize=16)
    _ax1.set_ylabel('$I_{DS}$ (mA)', fontsize=16)
    _ax1.set_title('Square-Law $I_{DS}$–$V_{DS}$', fontsize=16, fontweight='bold')
    _ax1.legend(fontsize=12)
    _ax1.tick_params(labelsize=14)
    _ax1.set_xlim(0, 3.5)
    _ax1.set_ylim(0, None)
    for _sp in ['top', 'right']:
        _ax1.spines[_sp].set_visible(False)

    _Vov_arr = np.array(_Vov_list)
    _Idsat_arr = np.array(_Idsat_list)
    _Vov_fit = np.linspace(0, max(_Vov_arr) * 1.1, 100)
    _Idsat_fit = 0.5 * _mu_ns * _Cox * _W / _L * _Vov_fit**2

    _ax2.plot(_Vov_fit, _Idsat_fit * 1e3, 'k--', lw=2,
              label=r'$\frac{1}{2}\mu_{ns}C_{oxe}\frac{W}{L}(V_{GS}-V_t)^2$')
    for _i in range(len(_Vov_arr)):
        _ax2.plot(_Vov_arr[_i], _Idsat_arr[_i] * 1e3, 'o', color=_colors[_i],
                  markersize=10, zorder=5)
    _ax2.set_xlabel('$V_{GS} - V_t$ (V)', fontsize=16)
    _ax2.set_ylabel('$I_{Dsat}$ (mA)', fontsize=16)
    _ax2.set_title('$I_{Dsat} \\propto (V_{GS} - V_t)^2$', fontsize=16, fontweight='bold')
    _ax2.legend(fontsize=13, loc='upper left')
    _ax2.tick_params(labelsize=14)
    _ax2.set_xlim(0, None)
    _ax2.set_ylim(0, None)
    for _sp in ['top', 'right']:
        _ax2.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig)

    _text = mo.md(rf"""
    ### Saturation

    The current reaches a maximum when $dI_{{DS}}/dV_{{DS}} = 0$:
    $$\frac{{dI_{{DS}}}}{{dV_{{DS}}}} = \mu_{{ns}} C_{{oxe}} \frac{{W}}{{L}}\big[(V_{{GS}} - V_t) - V_{{DS}}\big] = 0$$

    This gives the **saturation voltage**:
    $$V_{{Dsat}} = V_{{GS}} - V_t$$

    At this point, $Q_{{inv}}(L) = 0$ — the channel is "pinched off" at the drain end.

    The **saturation current** is:
    $$I_{{Dsat}} = \frac{{1}}{{2}}\mu_{{ns}} C_{{oxe}} \frac{{W}}{{L}}(V_{{GS}} - V_t)^2$$

    For $V_{{DS}} > V_{{Dsat}}$, the current remains at $I_{{Dsat}}$ (ideally).
    """)

    _caption = mo.md(rf"""
    **Device parameters:** $N_A$ = 10$^{{17}}$ cm$^{{-3}}$, $t_{{ox}}$ = 5 nm, $\mu_{{ns}}$ = 300 cm$^2$/V·s, $W/L$ = 10, $V_t$ = {_Vt:.3f} V. **Left:** $I_{{DS}}$-$V_{{DS}}$ for several $V_{{GS}}$. Dots mark the onset of saturation. **Right:** $I_{{Dsat}}$ vs. $V_{{GS}} - V_t$, the quadratic (square-law) dependence.
    """)

    mo.vstack([_text, mo.as_html(_fig), _caption])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Transconductance

    The **transconductance** $g_m$ measures how well the gate voltage controls the drain current:
    $$g_m = \frac{\partial I_{DS}}{\partial V_{GS}}$$

    In the triode region:
    $$g_m = \mu_{ns} C_{oxe} \frac{W}{L} V_{DS}$$

    At saturation ($V_{DS} = V_{Dsat}$), this is maximized:
    $$g_m = \mu_{ns} C_{oxe} \frac{W}{L}(V_{GS} - V_t)$$

    Higher $C_{oxe}$ (thinner oxide) leads to higher transconductance. $C_{oxe}$ provides **charge control**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Effective Oxide Capacitance

    In real devices, the oxide thickness is effectively wider than the physical $t_{ox}$ due to:
    - **Poly-gate depletion** — the gate electrode itself has a small depletion region
    - **Inversion layer thickness** — the channel charge centroid is slightly below the interface

    We define the **effective oxide thickness** $t_{oxe} > t_{ox}$ and use:
    $$C_{oxe} = \frac{\epsilon_{ox}}{t_{oxe}}$$

    in place of $C_{ox} = \epsilon_{ox}/t_{ox}$.

    **Effect:** The actual capacitance is slightly lower than the ideal value, reducing $I_{DS}$ and $g_m$.
    """)
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _body_mosfet_img = mo.image(f"{IMAGE_BASE}/body-effect-MOSFET.png", width=300)
    _body_model_img = mo.image(f"{IMAGE_BASE}/body-effect-circuit-model.png", width=200)


    mo.md(rf"""
    ## 5. Body Effect

    **Body bias** (or back bias): apply a voltage $V_b$ to the body to adjust the threshold voltage. $V_{{SB}}$ is the voltage at the source relative to the body.

    {mo.as_html(mo.hstack([_body_mosfet_img, _body_model_img], justify="center"))}

    Previously, for a MOS capacitor in inversion (without $V_{{CS}}$):
    $$Q_{{inv}} = -C_{{oxe}}(V_{{GS}} - V_{{t0}})$$
    where $V_{{t0}}$ is the threshold voltage without body bias. 

    Body bias adds charges to the depletion layer, modifying the inversion charge:
    $$Q_{{inv}} = -C_{{oxe}}(V_{{GS}} - V_{{t0}}) + C_{{dep}} V_{{SB}}$$
    where:
    $$C_{{dep}} = \frac{{\varepsilon_s}}{{W_{{dep,max}}}}, \qquad W_{{dep,max}} = \sqrt{{\frac{{2\varepsilon_s (2\phi_B)}}{{qN_a}}}}$$

    ### Redefining the Threshold Voltage

    Let's redefine the threshold voltage to absorb the body bias term:
    $$\boxed{{V_t = V_{{t0}} + \frac{{C_{{dep}}}}{{C_{{oxe}}}} V_{{SB}} \equiv V_{{t0}} + \alpha \, V_{{SB}}}}$$

    where $\alpha = C_{{dep}}/C_{{oxe}}$ is the **body effect parameter** and:
    $$V_{{t0}} = V_{{fb}} + 2\phi_B + \frac{{2}}{{C_{{oxe}}}}\sqrt{{qN_a \varepsilon_s \phi_B}}$$

    With this definition, $Q_{{inv}} = -C_{{oxe}}(V_{{GS}} - V_t)$, the same form as before.

    **Consequences:**
    - Keep $V_{{SB}} > 0$ so body-source and body-drain junctions remain in reverse bias
    - But $V_{{SB}} > 0$ means $V_t$ is increased relative to $V_{{t0}}$, and $I_{{DS}}$ is reduced at the same $V_{{GS}}$ and $V_{{DS}}$ → **reduces circuit speed**
    - In a circuit, multiple FETs share the same body but have different $V_S$, so $V_{{SB}}$ varies → body effect should be **minimized**
    - **Retrograde doping**: lightly doped surface with heavily doped body so $W_{{dep,max}}$ does not change much with $V_{{SB}}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Modification to $Q_{inv}(x)$ and IV Relation

    Including the effective oxide capacitance, body effect, and channel voltage $V_{CS}(x)$:

    Without $V_{CS}$:
    $$Q_{inv} = -C_{oxe}(V_{GS} - V_{t0}) + C_{dep} V_{SB}$$

    Including $V_{CS}(x)$:
    $$Q_{inv}(x) = -C_{oxe}(V_{GS} - V_{t0} - V_{CS}) + C_{dep}(V_{SB} + V_{CS})$$
    $$= -C_{oxe}(V_{GS} - m\, V_{CS} - V_t)$$

    where $m \equiv 1 + \alpha$ is the **bulk-charge factor** (ideal MOSFET: $m = 1$).

    The IV relation with body effect:
    $$\boxed{I_{DS} = \mu_{ns} C_{oxe} \frac{W}{L}\left[(V_{GS} - V_t)V_{DS} - \frac{m}{2}V_{DS}^2\right]}$$

    Saturation voltage, current, and transconductance:
    $$V_{Dsat} = \frac{V_{GS} - V_t}{m}, \qquad I_{Dsat} = \frac{W}{2mL}C_{oxe}\mu_{ns}(V_{GS} - V_t)^2, \qquad g_{m,sat} = \frac{W\mu_{ns}C_{oxe}}{mL}(V_{GS} - V_t)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Velocity Saturation

    So far, the surface mobility is assumed constant. In reality, the carrier velocity saturates at high electric fields:

    $$v = \frac{\mu_{ns}\, \mathcal{E}}{1 + \mathcal{E}/\mathcal{E}_{sat}}$$

    where $\mathcal{E}_{sat}$ is the saturation electric field (typically $\sim 10^5$ V/cm for electrons in Si).

    As $V_{DS}$ increases, the lateral electric field in the channel increases. When $\mathcal{E} \gg \mathcal{E}_{sat}$, the velocity saturates at $v_{sat} \approx 10^7$ cm/s.

    ### Modified IV Relation

    Including velocity saturation in the derivation modifies the drain current:

    $$I_{DS} = \frac{\mu_{ns} C_{oxe} \frac{W}{L}\left[(V_{GS} - V_t)V_{DS} - \frac{m}{2}V_{DS}^2\right]}{1 + \frac{V_{DS}}{\mathcal{E}_{sat} L}}$$

    The denominator is close to unity when $L$ is long or $V_{DS}$ is small, so the correction is negligible.

    **When does velocity saturation matter?**
    For $V_{DS} \sim 1$ V and $\mathcal{E}_{sat} \sim 10^5$ V/cm: significant when $L \lesssim 1\,\mu$m since $V_{DS}/(\mathcal{E}_{sat}L) \sim 0.1$.

    ### Effects:
    - $I_{DS}$ is **reduced** at the same $V_{GS}$, $V_{DS}$ → reduces switching speed
    - $I_{Dsat}$ varies roughly **linearly** with $(V_{GS} - V_t)$ rather than quadratically
    - $V_{Dsat}$ is lower than the long-channel value
    """)
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _rsd_img = mo.image(f"{IMAGE_BASE}/source-drain-resistance.png", width=500)

    mo.md(rf"""
    ## 7. Parasitic Source-Drain Resistance

    Real MOSFETs have ohmic resistance $R_S$ and $R_D$ in series with the source and drain. These resistances arise from the contact resistance and the sheet resistance of the n⁺ (or p⁺) regions.

    {mo.as_html(_rsd_img)}

    The effective drain-source voltage across the intrinsic channel is reduced:
    $$V_{{DS,\text{{intrinsic}}}} = V_{{DS}} - I_{{DS}}(R_S + R_D)$$

    To account for this, replace $V_{{DS}}$ with $V_{{DS}} - I_{{DS}}(R_S + R_D)$ in the IV relation.

    Similarly, $V_{{GS,\text{{intrinsic}}}} = V_{{GS}} - I_{{DS}} R_S$ (the source resistance also reduces the effective gate overdrive).

    **Effect:** Reduces $I_{{DS}}$ at the same external $V_{{DS}}$ and $V_{{GS}}$, especially at high currents.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Channel Length Modulation

    In the long-channel model, $I_{DS} = I_{Dsat}$ (constant) for $V_{DS} > V_{Dsat}$.

    In reality, increasing $V_{DS}$ beyond $V_{Dsat}$ pushes the pinch-off point slightly toward the source, effectively shortening the channel. A shorter channel means lower resistance, so $I_{DS}$ increases slightly:

    $$I_{DS} = I_{Dsat}(1 + \lambda V_{DS})$$

    where $\lambda$ is the **channel length modulation parameter**. The output conductance is:
    $$g_d = \lambda I_{Dsat}$$
    """)
    return


@app.cell
def _(mo):
    L_comp_slider = mo.ui.slider(start=0.05, stop=10.0, value=0.5, step=0.05,
                                  label=r"$L$ (µm)")
    m_comp_slider = mo.ui.slider(start=1.0, stop=1.5, value=1.0, step=0.05,
                                  label=r"$m$ (bulk-charge factor)")
    Esat_comp_slider = mo.ui.slider(start=1e4, stop=5e5, value=1e5, step=1e4,
                                     label=r"$\mathcal{E}_{sat}$ (V/cm)")
    Rsd_comp_slider = mo.ui.slider(start=0, stop=200, value=0, step=10,
                                    label=r"$R_S + R_D$ (Ω)")
    lambda_comp_slider = mo.ui.slider(start=0.0, stop=0.3, value=0.0, step=0.01,
                                       label=r"$\lambda$ (V⁻¹)")
    return (
        Esat_comp_slider,
        L_comp_slider,
        Rsd_comp_slider,
        lambda_comp_slider,
        m_comp_slider,
    )


@app.cell(hide_code=True)
def _(
    Esat_comp_slider,
    L_comp_slider,
    Rsd_comp_slider,
    eps_0,
    eps_Si,
    kT_300,
    lambda_comp_slider,
    m_comp_slider,
    mo,
    ni_Si,
    np,
    plt,
    q,
):
    _L_um = L_comp_slider.value
    _L = _L_um * 1e-4
    _m = m_comp_slider.value
    _Esat = Esat_comp_slider.value
    _Rsd = Rsd_comp_slider.value
    _lam = lambda_comp_slider.value

    _Na = 1e17
    _tox = 5e-7
    _Cox = 3.9 * eps_0 / _tox
    _phi_B = kT_300 * np.log(_Na / ni_Si)
    _Vt = 2 * _phi_B + np.sqrt(2 * eps_Si * eps_0 * q * _Na * 2 * _phi_B) / _Cox
    _mu_ns = 300
    _W = 10e-4

    _Vds = np.linspace(0, 3.5, 500)
    _Vgs_vals = [1.0, 1.5, 2.0, 2.5]
    _colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    _fig, _ax = plt.subplots(figsize=(10, 6))

    for _i, _Vgs in enumerate(_Vgs_vals):
        _Vov = _Vgs - _Vt
        if _Vov <= 0:
            continue

        _Vdsat_id = _Vov
        _ids_ideal = np.where(
            _Vds < _Vdsat_id,
            _mu_ns * _Cox * _W / _L * (_Vov * _Vds - 0.5 * _Vds**2),
            0.5 * _mu_ns * _Cox * _W / _L * _Vov**2
        )
        _ids_ideal = np.maximum(_ids_ideal, 0)

        _ids_mod = np.zeros_like(_Vds)
        for _j in range(len(_Vds)):
            _id_prev = 0
            for _iter in range(50):
                _vgs_int = _Vgs - _id_prev * _Rsd / 2
                _vds_int = max(_Vds[_j] - _id_prev * _Rsd, 0)
                _vov_int = max(_vgs_int - _Vt, 0)
                if _vov_int <= 0:
                    _id_new = 0
                    break
                _E = _Esat * _L
                _vdsat = _E * (-1 + np.sqrt(1 + 2 * _vov_int / (_m * _E)))
                if _vds_int < _vdsat:
                    _id_new = _mu_ns * _Cox * _W / _L * (_vov_int * _vds_int - _m / 2 * _vds_int**2) / (1 + _vds_int / (_Esat * _L))
                else:
                    _Idsat_int = _mu_ns * _Cox * _W / _L * (_vov_int * _vdsat - _m / 2 * _vdsat**2) / (1 + _vdsat / (_Esat * _L))
                    _id_new = _Idsat_int * (1 + _lam * (_vds_int - _vdsat))
                _id_new = max(_id_new, 0)
                if abs(_id_new - _id_prev) < 1e-10:
                    break
                _id_prev = 0.5 * _id_prev + 0.5 * _id_new
            _ids_mod[_j] = _id_new

        _ax.plot(_Vds, _ids_ideal * 1e3, '--', color=_colors[_i], lw=1.5, alpha=0.5)
        _ax.plot(_Vds, _ids_mod * 1e3, '-', color=_colors[_i], lw=2.5,
                 label=f'$V_{{GS}}$ = {_Vgs:.1f} V')

    _ax.set_xlabel('$V_{DS}$ (V)', fontsize=16)
    _ax.set_ylabel('$I_{DS}$ (mA)', fontsize=16)
    _ax.set_title(f'Ideal (dashed) vs. Modified (solid): $m$={_m:.2f}, $L$={_L_um} µm, '
                  f'$\\mathcal{{E}}_{{sat}}$={_Esat:.0e} V/cm, '
                  f'$R_{{SD}}$={_Rsd} Ω, $\\lambda$={_lam:.2f}',
                  fontsize=13, fontweight='bold')
    _ax.legend(fontsize=12)
    _ax.tick_params(labelsize=14)
    _ax.set_xlim(0, 3.5)
    _ax.set_ylim(0, None)
    for _sp in ['top', 'right']:
        _ax.spines[_sp].set_visible(False)

    plt.tight_layout()
    plt.close(_fig)

    _slider_row = mo.vstack([
        mo.hstack([L_comp_slider, m_comp_slider, Esat_comp_slider], justify="start"),
        mo.hstack([Rsd_comp_slider, lambda_comp_slider], justify="start"),
    ])


    _header = mo.md(rf"""
    ## 9. Interactive Comparison: $I_{{DS}}$ vs. $V_{{DS}}$ with Modifications

    """)

    _info = mo.md(rf"""
    Each modification affects $I_{{DS}}$:
    - **Body effect** ($m > 1$): increases effective $V_t$, reduces $V_{{Dsat}}$
    - **Velocity saturation** (short $L$): limits carrier velocity in channel
    - **Parasitic $R_{{SD}}$**: drops voltage outside the intrinsic channel
    - **Channel length modulation** ($\lambda > 0$): finite slope in saturation, increases $I_{{DS}}$ above $V_{{Dsat}}$
    """)

    mo.vstack([_header, _slider_row, mo.as_html(_fig), _info])
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _img = mo.image(f"{IMAGE_BASE}/subthreshold-swing.png", width=400)

    mo.md(rf"""
    ## 10. Subthreshold Conduction

    When $V_{{GS}} < V_t$, the MOSFET is not truly "off". A small **subthreshold current** flows due to diffusion of minority carriers over the source-channel barrier:

    $$I_{{DS,sub}} \propto \exp\left(\frac{{V_{{GS}}}}{{m \cdot kT/q}}\right)$$

    The **subthreshold swing** $S$ measures how many millivolts of $V_{{GS}}$ are needed to change $I_{{DS}}$ by one decade:

    $$S = m \cdot \frac{{kT}}{{q}} \cdot \ln(10) \approx 2.3 \cdot m \cdot \frac{{kT}}{{q}}$$

    At room temperature with $m = 1$: $S_{{min}} = 60$ mV/decade (the **Boltzmann limit**).

    Typical values: $S \sim 80$–$100$ mV/decade (since $m > 1$).

    {mo.as_html(mo.hstack([_img], justify="center"))}

    ### Off-State Current $I_{{off}}$

    $I_{{off}}$ is the drain current when $V_{{GS}} = 0$ V. Using the threshold definition $I_{{DS}}(V_t) = 100$ nA $\times W/L$:

    $$I_{{off}} = \frac{{100\,\text{{nA}} \cdot W}}{{L}} \cdot 10^{{-V_t / S}}$$

    $I_{{off}}$ is reduced when $V_t$ is higher or $S$ is lower. Shorter channels tend to have higher $I_{{off}}$.
    """)
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _rolloff_img = mo.image(f"{IMAGE_BASE}/Vt-roll-off.png", width=400)
    _dibl_img = mo.image(f"{IMAGE_BASE}/DIBL.png", width=600)

    mo.md(rf"""
    ## 11. $V_t$ Roll-Off and Drain-Induced Barrier Lowering (DIBL)

    As the channel length $L$ decreases, the threshold voltage drops. This is known as **$V_t$ roll-off**. 

    Reduction of $V_t$ means higher $I_{{off}}$.

    {mo.as_html(mo.hstack([_rolloff_img], justify="center"))}

    **Physical origin: Drain-Induced Barrier Lowering (DIBL)**

    In a long-channel device, the potential barrier between source and drain is fully controlled by the gate and reaches its maximum value. $V_t$ has its expected (designed) value.

    In a short-channel device, the drain electric field encroaches on the source-channel barrier. The barrier cannot reach its full height → $V_t$ is **reduced**.

    $$V_t(\text{{short}}) = V_t(\text{{long}}) - \eta \cdot V_{{DS}}$$

    where $\eta$ is the DIBL coefficient (typically 20–100 mV/V).

    **Consequences:**
    - Short-channel MOSFETs have **higher $I_{{off}}$** (lower $V_t$ means more subthreshold leakage)
    - There is a **minimum channel length** below which the transistor cannot be adequately turned off
    - DIBL makes $V_t$ depend on $V_{{DS}}$, which is undesirable for circuit design


    {mo.as_html(mo.hstack([_dibl_img], justify="center"))}


    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Modification | Effect on IV | Key Parameter |
    |:-------------|:-------------|:---------------:|
    | **Effective $C_{oxe}$** | Reduces $I_{DS}$ and $g_m$ | $t_{oxe} > t_{ox}$ |
    | **Body effect** | Increases $V_t$, reduces $I_{DS}$; $V_{Dsat} = (V_{GS}-V_t)/m$ | $m = 1 + C_{dep}/C_{oxe}$ |
    | **Velocity saturation** | Reduces $I_{DS}$; $I_{Dsat}$ becomes linear in $V_{GS}-V_t$ | $\mathcal{E}_{sat} L$ vs. $V_{DS}$ |
    | **Parasitic $R_{SD}$** | Reduces effective $V_{DS}$ and $V_{GS}$ | $R_S + R_D$ |
    | **Channel length modulation** | $I_{DS}$ has finite slope in saturation | $\lambda$ (gives $g_d = \lambda I_{Dsat}$) |
    | **Subthreshold swing** | Exponential turn-off below $V_t$; determines $I_{off}$ | $S \geq 60$ mV/dec |
    | **DIBL** | $V_t$ decreases with $V_{DS}$ in short channels | $\eta$ (mV/V) |

    ### Complete IV relation (triode, with all modifications):

    $$I_{DS} = \frac{\mu_{ns} C_{oxe} \dfrac{W}{L}\left[(V_{GS} - V_t) V_{DS} - \dfrac{m}{2}V_{DS}^2\right]}{1 + \dfrac{V_{DS}}{\mathcal{E}_{sat} L}}$$

    with $V_{DS}$ replaced by $V_{DS} - I_{DS}(R_S + R_D)$ to account for parasitic resistance.

    **The key trade-off in MOSFET scaling:** Shrinking $L$ increases speed and density, but worsens short-channel effects (DIBL, $V_t$ roll-off, higher $I_{off}$). Advanced device architectures (FinFET, nanosheets) are designed to maintain gate control as $L$ scales below 20 nm.
    """)
    return


if __name__ == "__main__":
    app.run()
