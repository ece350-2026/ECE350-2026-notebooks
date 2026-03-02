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
    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/pn-non-idealities/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    # Physical constants
    q = 1.6e-19       # C
    kT_eV = 0.02585   # eV at 300K
    kT_q = 0.02585    # V at 300K (thermal voltage)
    ni_Si = 1.1e10    # cm^-3 for Si at 300K
    Eg_Si = 1.12      # eV
    eps_r = 11.7
    eps_0 = 8.854e-14 # F/cm
    eps_Si = eps_r * eps_0

    # Typical mobility values for Si at 300K
    mu_n = 1350.0  # cm^2/V-s
    mu_p = 480.0   # cm^2/V-s
    D_n = kT_q * mu_n  # cm^2/s (Einstein relation)
    D_p = kT_q * mu_p  # cm^2/s

    mo.md(
        r"""
        # PN Junction: Approximations and Non-Idealities

        ECE350, Lecture 22-23

        This interactive notebook covers the quantitative current-voltage characteristics of PN junction diodes:

        1. P⁺N and N⁺P Diodes (One-Sided Junctions)
        2. Boltzmann Limitation
        3. E-Field in the Quasi-Neutral Regions
        4. Short Diode
        5. Non-Idealities and the Ideality Factor
        """
    )
    return IMAGE_BASE, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. P$^+$N and N$^+$P Diodes (One-Sided Junctions)

    The current density is

    $$J = q\left(\frac{D_n\, n_{P0}}{L_n} + \frac{D_p\, p_{N0}}{L_p}\right)\left(e^{qV/k_B T} - 1\right) \implies J = q\left(\frac{D_n\, n_i^2}{N_aL_n} + \frac{D_p\, n_i^2}{N_dL_p}\right)\left(e^{qV/k_B T} - 1\right)$$

    - If $N_a >> N_d$: current dominated by the second term (hole diffusion in the N-side)

    $$J = q \frac{D_p\, n_i^2}{N_dL_p}\left(e^{qV/k_B T} - 1\right)$$

    - If $N_d >> N_a$: current dominated by the first term (electron diffusion in the P-side)

    $$J = q\frac{D_n\, n_i^2}{N_aL_n}\left(e^{qV/k_B T} - 1\right)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Boltzmann Limitation

    - Thermodynamics limit us from getting the perfect switch with an infinitely sharp turn-on

    - To get 10x change in current, we need a voltage swing of:

    $$\frac{I_2}{I_1} = 10 \approx e^{q(V_2 - V_1)/k_BT}$$
    $$V_2 - V_1 = 26 \mathrm{mV} \ln(10) = 60 \mathrm{mV}$$

    - Swing is limited to 60mV/decade
        - We want to be as close to zero as possible
        - Non-idealities raise this value
        - Can we get below 60mV/dec at room temperature???
    """)
    return


@app.cell
def _(mo, np):
    # E-field in quasi-neutral regions (Si, Na=Nd=10^16, T=300K, V=0.4V)
    _q = 1.6e-19
    _kT_q = 0.02585
    _ni = 1.1e10
    _eps = 11.7 * 8.854e-14
    _mu_n = 1350.0
    _mu_p = 480.0
    _D_n = _kT_q * _mu_n
    _D_p = _kT_q * _mu_p
    _Na = 1e16
    _Nd = 1e16
    _V = 0.4
    _Ln = 10e-4
    _Lp = 10e-4
    _phi_bi = _kT_q * np.log(_Na * _Nd / _ni**2)
    _phi_eff = max(_phi_bi - _V, 0.01)
    _W_dep = 2 * np.sqrt(_eps * _phi_eff / (_q * _Nd))
    _E_dep_max = 2 * _phi_eff / _W_dep
    _J_0 = _q * (_D_n * _ni**2 / (_Ln * _Na) + _D_p * _ni**2 / (_Lp * _Nd))
    _J = _J_0 * (np.exp(_V / _kT_q) - 1)
    _E_N = _J / (_q * _mu_n * _Nd)
    _E_P = _J / (_q * _mu_p * _Na)
    mo.vstack([
        mo.md(r"""
        ## 3. E-Field in the Quasi-Neutral Regions

        We have assumed that the electric field in the quasi-neutral regions is zero. Let's check if this is a good approximation.

        **Plug in numbers:** Si step junction, $N_a = N_d = 10^{16}$ cm$^{-3}$, $T = 300$ K, forward bias $V = 0.4$ V.

        Compare the **electric field in the quasi-neutral N and P regions** to the **depletion layer field**.
        """),
        mo.md(
            f"""
            **Depletion region:** $\\phi_{{bi}}$ = {_phi_bi:.3f} V, $W_{{dep}}$ = {_W_dep*1e4:.2f} μm.  
            Peak field $|E_{{dep}}|$ ≈ $2\\phi_{{eff}}/W_{{dep}}$ ≈ **{_E_dep_max:.1f} V/cm** (order ~10$^4$ V/cm).

            **Quasi-neutral regions:** Current density $J$ ≈ {_J:.2e} A/cm².  
            In the N-region, $E \\approx J/(q\\mu_n N_d)$; in the P-region, $E \\approx J/(q\\mu_p N_a)$.  
            $|E_N|$ ≈ **{_E_N:.2e} V/cm**, $|E_P|$ ≈ **{_E_P:.2e} V/cm** (order ~10$^{{-4}}$ V/cm).

            **Order of magnitude:** The quasi-neutral fields are **many orders of magnitude smaller** than the depletion field (e.g. ~10$^8$×), so assuming $\\mathcal{{E}} \\approx 0$ in the quasi-neutral regions is an excellent approximation. Note the orders of magnitude for the electric field, depletion width, and current as in the problem statement.
            """
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Short Diode

    In many practical devices (e.g., the base of a BJT), the quasi-neutral region length $W$
    is **shorter** than the diffusion length: $W \ll L_p$ or $W \ll L_n$.

    ### Changed Boundary Condition

    Instead of $p'_N(x \to \infty) = 0$, for a short diode with an ohmic contact at
    $x = x_n + W$ (where $W$ is the quasi-neutral region length):

    $$p'_N(x = x_n + W) = 0$$

    ### Solution

    The excess carrier profile becomes **linear** (rather than exponential) when $W \ll L_p$:

    $$p'_N(x) \approx \frac{n_i^2}{N_d}\left[\exp\!\left(\frac{qV}{k_B T}\right) - 1\right]\left(1 - \frac{x - x_n}{W}\right)$$

    More generally, the solution involves hyperbolic functions:

    $$p'_N(x) = \frac{n_i^2}{N_d}\left[\exp\!\left(\frac{qV}{k_B T}\right) - 1\right]
    \frac{\sinh\!\left(\frac{x_n + W - x}{L_p}\right)}{\sinh\!\left(\frac{W}{L_p}\right)}$$

    The current density for a short diode replaces $L_p$ with $L_p \tanh(W/L_p)$:

    $$J_p^{short}(x_n) = \frac{qD_p}{L_p}\coth\!\left(\frac{W}{L_p}\right)
    \frac{n_i^2}{N_d}\left[\exp\!\left(\frac{qV}{k_B T}\right) - 1\right]$$

    For $W \ll L_p$: $\coth(W/L_p) \approx L_p / W$, and $J_p \propto D_p / W$ (no dependence on lifetime!).

    For $W \gg L_p$: $\coth(W/L_p) \approx 1$, recovering the long diode result.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Non-Idealities and the Ideality Factor

    Three main departures from the ideal diode:

    1. **SCR current:** generation if $V < 0$, recombination if $V > 0$ (space-charge region)
    2. **High injection** if $V > \phi_{bi}$
    3. **Series resistance** $R_S$ of quasi-neutral regions at $V > \phi_{bi}$, often combined with (2)

    Empirically we write:

    $$\boxed{I = I_0\left[\exp\!\left(\frac{qV}{n k_B T}\right) - 1\right]}, \qquad n = \text{ideality factor},\; 1 \leq n \leq 2$$

    ### Overview of the Ideality Factor

    | Regime | $n$ | Mechanism |
    |:---|:---:|:---|
    | Ideal | $n = 1$ | Diffusion current dominates (e.g. Ge: $n_i$ large) |
    | Recombination in depletion region | $n = 2$ | $I_{SCR}$ dominates (120 mV/decade) |
    | Mixed | $1 < n < 2$ | Both diffusion and recombination (Si, GaAs, InP, etc.) |
    | High injection | $n = 2$ | $n \to 2$ at $V > \phi_{bi}$ when heavy injection sets in |

    On a semi-log plot of $I$ vs $V$, the slope is $q/(n k_B T)$; larger $n$ gives a **shallower** slope.
    """)
    return


@app.cell
def _(IMAGE_BASE, mo):
    _text = mo.md(r"""
    ### 5.1 SCR Current (Generation and Recombination in the Depletion Region)

    - **Ideal** $n = 1$: slope 60 mV/decade (diffusion current).
    - **In the depletion region**, recombination is maximized when $n = p$.

        - Since $np = n_i^2 \exp(qV/k_B T)$, when $n \approx p$ we have $n \approx p \approx n_i \exp(qV/2k_B T)$.
    - Net recombination rate per unit volume:

    $$R = \frac{n - n_i}{\tau_{dep}} \approx \frac{n_i}{\tau_{dep}} \left[\exp\!\left(\frac{qV}{2k_B T}\right) - 1\right]$$

    - $\tau_{dep}$ is the effective recombination lifetime in the depletion region (from SRH with midgap traps and $\tau_n = \tau_p = \tau_{dep}$).

    $$\therefore I_{SCR} = q R\times \text{Volume} = qA\,\frac{ n_i W_{dep}}{\tau_{dep}}\left(e^{qV/2k_B T} - 1\right)$$

    - Total current (diffusion + SCR):

    $$I = I_0 \left(e^{qV/k_B T} - 1\right) + I_{SCR}$$

    - $I_{SCR}$ has a slope of $q/2k_B T$ (120 mV/decade) in forward bias

    - In reverse bias, the leakage current is higher than ideal due to generation:

    $$I_{leakage} = I_0 + A\,\frac{q n_i W_{dep}}{\tau_g}$$
    """)
    _img = mo.image(src=f"{IMAGE_BASE}/non-ideal-IV.png", width="75%")
    _centered_img = mo.hstack([_img], justify="center")
    _caption = mo.hstack([mo.md("*Fig. 4-22 from Hu. In forward bias, the slope exceeds 60 mV/decade. In reverse bias, the leakage current is higher than ideal due to generation.*")], justify="center")
    _img2 = mo.image(src=f"{IMAGE_BASE}/Ic-Vbe_plot.png", width="100%")
    _img3 = mo.image(src=f"{IMAGE_BASE}/probestation.png", width="100%")
    _img4 = mo.image(src=f"{IMAGE_BASE}/SiGe-HBT.png", width="100%")
    _img_row = mo.hstack([_img2, _img3, _img4], widths=[0.4, 0.4, 0.3], align="center")
    _caption2 = mo.hstack([mo.md("*Measurement from Prof. Sorin Voinigescu's research group. [S. Bonen et al., IEEE MWCL, 2022](https://www.eecg.utoronto.ca/~sorinv/papers/Bonen_2022_IEEEMWCL__Cryogenic-Characterization-of-the-High-Frequnecy-and-Noise-Performance-of-SiGe-HBTs-From-DC-to-70GHz-and-Down-to-2K.pdf). SiGe HBT cross-section from [The Aerospace Corporation](https://aerospace.org/story/sige-hbt-dream).*")], justify="center")
    mo.vstack([_text, _centered_img, _caption, _img_row, _caption2])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.2 High Injection ($V > \phi_{bi}$)

    - $n'_P = p'_N \gg N_a,\, N_d$: injected carriers comparable to doping
    - Consider $p_N n_N = n_i^2 \exp(qV/k_B T)$ and charge neutrality $n \approx p$: $n_N \approx p_N \approx n_i \exp(qV/2k_B T)$
    - Using the same argument as in the previous section, we get $J \sim \exp(qV/2k_B T)$ $\Rightarrow$ again **$n = 2$**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.3 Series Resistance at Large Current

    - Effective voltage across the junction is reduced by the voltage drop on the **series resistance** of the quasi-neutral n- and p-regions
        - More pronounced for high-injection due to higher voltage drop
    - $I$ increases at a **lower rate** than $\exp(qV/k_B T)$ or $\exp(qV/2k_B T)$

    $$I = I_0 \exp\!\left(\frac{V - I R_S}{n k_B T}\right) < I_0 \exp\!\left(\frac{V}{n k_B T}\right)$$
    """)
    return


@app.cell
def _(mo):
    # No leading underscore so next cell can read .value (marimo convention)
    n_slider = mo.ui.slider(
        start=1.0, stop=2.0, value=1.0, step=0.05,
        label=r"Ideality factor $n$"
    )
    IS2_slider = mo.ui.slider(
        start=-15, stop=-9, value=-12, step=0.5,
        label=r"log$_{10}$($I_0$) [A]"
    )
    ideality_controls = mo.vstack([
        mo.md("### 5.4 Ideality Factor Explorer"),
        mo.hstack([n_slider, IS2_slider], justify="start"),
    ])
    return IS2_slider, ideality_controls, n_slider


@app.cell
def _(IS2_slider, ideality_controls, mo, n_slider, np, plt):
    _n_val = n_slider.value
    _I0_val = 10 ** IS2_slider.value
    _kT_q_val = 0.02585  # V at 300K

    _V = np.linspace(0.0, 0.8, 500)

    # Current with ideality factor n
    _I_n = _I0_val * (np.exp(_V / (_n_val * _kT_q_val)) - 1)

    # Reference curves for n=1 and n=2
    _I_n1 = _I0_val * (np.exp(_V / (1.0 * _kT_q_val)) - 1)
    _I_n2 = _I0_val * (np.exp(_V / (2.0 * _kT_q_val)) - 1)

    _fig_n, _ax_n = plt.subplots(figsize=(10, 7))

    # Plot reference curves
    _ax_n.semilogy(_V, _I_n1, "b--", linewidth=1.5, alpha=0.4, label="$n$ = 1 (diffusion)")
    _ax_n.semilogy(_V, _I_n2, "r--", linewidth=1.5, alpha=0.4, label="$n$ = 2 (recombination)")

    # Plot selected n
    _ax_n.semilogy(_V, _I_n, "k-", linewidth=3, label=f"$n$ = {_n_val:.2f}")

    # Mark I_0 (format as ×10^n)
    def _x10_fmt_n(x):
        if x == 0:
            return "0"
        e = int(np.floor(np.log10(abs(x))))
        m = x / (10**e)
        if abs(m) >= 9.995:
            m /= 10
            e += 1
        return f"{m:.2f}×10$^{{{e}}}$"
    _ax_n.axhline(_I0_val, color="gray", linestyle=":", alpha=0.7, linewidth=1)
    _ax_n.text(0.01, _I0_val * 2, f"$I_0$ = {_x10_fmt_n(_I0_val)} A", fontsize=11, color="gray")

    # Annotate slope
    _slope_n = 1 / (_n_val * _kT_q_val * np.log(10))
    _slope_1 = 1 / (1.0 * _kT_q_val * np.log(10))
    _slope_2 = 1 / (2.0 * _kT_q_val * np.log(10))

    _ax_n.text(
        0.50, _I0_val * np.exp(0.45 / (_n_val * _kT_q_val)),
        f"Slope: {_slope_n:.1f} dec/V\n($n$ = {_n_val:.2f})",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    # Shade approximate regions where each n dominates
    _ax_n.axvspan(0, 0.3, alpha=0.05, color="red", label="Recombination regime")
    _ax_n.axvspan(0.4, 0.65, alpha=0.05, color="blue", label="Diffusion regime")

    _ax_n.text(0.15, _ax_n.get_ylim()[0] * 5, "$n \\approx 2$\nRecombination\ndominates",
               fontsize=10, ha="center", color="red", alpha=0.7)
    _ax_n.text(0.52, _ax_n.get_ylim()[0] * 5, "$n \\approx 1$\nDiffusion\ndominates",
               fontsize=10, ha="center", color="blue", alpha=0.7)

    _ax_n.set_xlabel(r"$V$ (V)", fontsize=14)
    _ax_n.set_ylabel(r"$I$ (A)", fontsize=14)
    _ax_n.set_title("Effect of Ideality Factor on Forward I-V Characteristic", fontsize=14, fontweight="bold")
    _ax_n.legend(fontsize=11, loc="upper left")
    _ax_n.grid(True, alpha=0.3, which="both")
    _ax_n.set_xlim(0, 0.8)
    _ax_n.set_ylim(_I0_val * 0.5, _I0_val * np.exp(0.8 / (1.0 * _kT_q_val)) * 2)

    plt.tight_layout()

    _info_ideality = mo.md(
        f"""
        **Slopes on semi-log plot:**
        $n = 1$: {_slope_1:.1f} dec/V |
        $n = 2$: {_slope_2:.1f} dec/V |
        $n = {_n_val:.2f}$: **{_slope_n:.1f} dec/V**

        At 300 K, the ideal slope ($n=1$) is approximately **16.7 decades/V** (or equivalently 60 mV/decade).
        Recombination current ($n=2$) gives about **8.4 decades/V** (120 mV/decade).
        """
    )

    mo.vstack([ideality_controls, _fig_n, _info_ideality])
    return


if __name__ == "__main__":
    app.run()
