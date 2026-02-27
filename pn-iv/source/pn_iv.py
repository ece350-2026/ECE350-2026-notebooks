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
app = marimo.App(width="medium", layout_file="layouts/pn_iv.slides.json")


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
    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/pn-iv/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

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
        # PN Junction: Current-Voltage Relation

        ECE350, Lecture 19-21

        This interactive notebook covers the quantitative current-voltage characteristics of PN junction diodes:

        1. Qualitative picture of current flow
        2. Ideal diode model (Shockley assumptions)
        3. Minority carrier boundary conditions (law of the junction)
        4. Solving the minority carrier diffusion equation
        5. Derivation of the Shockley diode equation
        6. Interactive I-V characteristics

        """
    )
    return (
        D_n,
        D_p,
        Eg_Si,
        IMAGE_BASE,
        eps_Si,
        kT_q,
        mo,
        mu_n,
        mu_p,
        ni_Si,
        np,
        plt,
        q,
    )


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(r"""## Qualitative Description""")
    _text = mo.md(r"""
    ### Zero Bias (Equilibrium)

    At thermal equilibrium, no net current flows.

    ### Forward Bias ($V > 0$)

    When a positive voltage is applied to the P-side:

    - The potential barrier is **reduced** from $q\phi_{bi}$ to $q(\phi_{bi} - V)$
    - Some holes diffuse from P to N and become minority carriers in the N side ("minority carrier injection")
    - Similarly, minority carrier injection of electrons from N to P
    - **Diffusion of minority carriers** in quasi-neutral regions $\rightarrow$ current
    - **Quasi-Fermi level split**: $E_{Fn} - E_{Fp} = qV$
    - **Depletion width narrows**: since the effective voltage across the junction is reduced, less charge is needed to support it

    ### Reverse Bias ($V < 0$)

    When a negative voltage is applied to the P-side:

    - The potential barrier is **increased** to $q(\phi_{bi} + |V|)$
    - Electrons in the P-side and holes in the N-side near the depletion edges are swept across by the electric field
    - This produces a small, voltage-independent **drift** current. We call this the **reverse saturation current** $I_0$
    - The current saturates because the supply of minority carriers is limited by thermal generation
    """)
    _img = mo.image(src=f"{IMAGE_BASE}/PN-biases.png", width="100%")
    mo.vstack([
        _header,
        mo.hstack([_text, _img], widths=[0.6, 0.4], align="start"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Ideal Diode Model (Shockley Model)

    To derive an analytical expression for the I-V characteristic, we make the following simplifying
    assumptions:

    1. **Steady state**: $\dfrac{\partial n}{\partial t} = \dfrac{\partial p}{\partial t} = 0$

    2. **Maxwell-Boltzmann statistics** apply

    3. **One-dimensional geometry**, infinitely long diode (contacts at $x = \pm\infty$)

    4. **Total current is constant**: electron and hole currents are continuous and constant in the depletion region (i.e., there is no recombination or generation in the depletion region): $\frac{dJ_n}{dx} = 0, \quad \frac{dJ_p}{dx} = 0$ in the depletion region

    5. **Low-level injection**: the applied voltage satisfies $V < \phi_{bi}$, and the voltage
       is dropped entirely in the depletion region. The P and N sides have $\mathcal{E} = 0$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Roadmap for Analysis

    Since the current is dominated by the diffusion of carriers, we find the carrier concentrations as a function of space.


    1. Find excess hole concentration in the N-side, $p'_N(x)$, and excess electron concentration in the P-side, $n'_P(x)$.

    2. Find the diffusion component of the current at the edges of the depletion layer: $J_{P,\text{diff}}(x = x_n)$ and $J_{N,\text{diff}}(x = -x_p)$.

    3. Since $J = J_N + J_P$ and the current is assumed to be constant in the depletion region,

    $$J = J_{P,\text{diff}}(x = x_n) + J_{N,\text{diff}}(x = -x_p)$$
    """)
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(r"""
    ## Step 1: Minority Carrier Densities
    """)
    _img = mo.hstack([mo.image(src=f"{IMAGE_BASE}/depletion-region.png", width="70%")], justify="center")
    _text = mo.md(r"""
    At the edges of the depletion region, the carrier concentrations are set by the quasi-Fermi levels.

    **Electron concentration at $x = -x_p$ (P-side edge):**

    $$n(-x_p) = N_c\, e^{-(E_c - E_{Fn})/k_B T} = N_c\, e^{-(E_c - E_{Fp})/k_B T}\, e^{(E_{Fn} - E_{Fp})/k_B T} = n_{P0}\, e^{qV/k_B T}$$

    where $n_{P0}$ is the equilibrium electron concentration on the P-side.

    **Hole concentration at $x = x_n$ (N-side edge):**

    $$p(x_n) = N_v\, e^{-(E_{Fp} - E_v)/k_B T} = N_v\, e^{-(E_{Fn} - E_v)/k_B T}\, e^{(E_{Fn} - E_{Fp})/k_B T} = p_{N0}\, e^{qV/k_B T}$$

    where $p_{N0}$ is the equilibrium hole concentration on the N-side.

    Since $n_{P0} \approx n_i^2 / N_a$ and $p_{N0} \approx n_i^2 / N_d$:

    $$\boxed{n(-x_p) \approx \frac{n_i^2}{N_a}\, e^{qV/k_B T} \qquad p(x_n) \approx \frac{n_i^2}{N_d}\, e^{qV/k_B T}}$$

    ### Excess Carriers

    $$\boxed{n_P'(-x_p) \equiv n(-x_p) - n_{P0} = n_{P0}\left(e^{qV/k_B T} - 1\right)}$$

    $$\boxed{p_N'(x_n) \equiv p(x_n) - p_{N0} = p_{N0}\left(e^{qV/k_B T} - 1\right)}$$

    - Under **forward bias** ($V > 0$): excess carriers are **injected** ($n_P' > 0$, $p_N' > 0$)
    - Under **reverse bias** ($V < 0$): minority carriers are **extracted** ($n_P' < 0$, $p_N' < 0$)
    """)
    mo.vstack([_header, _img, _text])
    return


@app.cell
def _(mo):
    # Sliders: N_a, N_d from 10^14 to 10^18 (step 0.5 in exponent), V from 0 to 0.6 V
    # No leading underscore so these are exported to the next cell (marimo convention)
    Na_log_slider = mo.ui.slider(
        start=14.0, stop=18.0, value=18.0, step=0.5,
        label=r"log₁₀(N_a) [cm⁻³]"
    )
    Nd_log_slider = mo.ui.slider(
        start=14.0, stop=18.0, value=16.0, step=0.5,
        label=r"log₁₀(N_d) [cm⁻³]"
    )
    V_slider = mo.ui.slider(
        start=0.0, stop=0.6, value=0.6, step=0.02,
        label=r"Applied voltage V (V)"
    )
    _calc_header = mo.md("### Step 1 — Example: Minority and Excess Carrier Concentrations")
    _controls = mo.hstack([
        Na_log_slider,
        Nd_log_slider,
        V_slider,
    ], justify="start")
    step1_header = mo.vstack([_calc_header, _controls])
    return Na_log_slider, Nd_log_slider, V_slider, step1_header


@app.cell
def _(
    Na_log_slider,
    Nd_log_slider,
    V_slider,
    kT_q,
    mo,
    ni_Si,
    np,
    step1_header,
):
    # Read slider values in a separate cell (marimo: no .value access in the cell that creates the UI element)
    _Na_cm3 = 10.0 ** Na_log_slider.value
    _Nd_cm3 = 10.0 ** Nd_log_slider.value
    _V = V_slider.value
    _n_P0 = ni_Si**2 / _Na_cm3
    _p_N0 = ni_Si**2 / _Nd_cm3
    _n_at_neg_xP = _n_P0 * np.exp(_V / kT_q)
    _p_at_xN = _p_N0 * np.exp(_V / kT_q)
    _excess_n_P = _n_at_neg_xP - _n_P0
    _excess_p_N = _p_at_xN - _p_N0

    def _x10(x):
        """Format number as mantissa × 10^exp for LaTeX (e.g. 1.23 × 10^4)."""
        if x == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(x))))
        mantissa = x / (10**exp)
        if abs(mantissa) >= 9.995:
            mantissa /= 10
            exp += 1
        return f"{mantissa:.2f} \\times 10^{{{exp}}}"

    _eq_nP0 = f"$$n_{{P0}} = \\frac{{n_i^2}}{{N_a}} = \\frac{{({_x10(ni_Si)})^2}}{{{_x10(_Na_cm3)}}} = {_x10(_n_P0)}\\ \\mathrm{{cm}}^{{-3}}$$"
    _eq_pN0 = f"$$p_{{N0}} = \\frac{{n_i^2}}{{N_d}} = \\frac{{({_x10(ni_Si)})^2}}{{{_x10(_Nd_cm3)}}} = {_x10(_p_N0)}\\ \\mathrm{{cm}}^{{-3}}$$"
    _part_equil = mo.vstack([
        mo.md(r"**Equilibrium minority carrier concentrations** (in the quasi-neutral P and N regions, far from the junction)"),
        mo.md(r"- **P-side:** electrons are minority carriers:"),
        mo.md(_eq_nP0),
        mo.md(r"- **N-side:** holes are minority carriers:"),
        mo.md(_eq_pN0),
    ])
    _n_eq = f"$$n(-x_p) = n_{{P0}}\\, e^{{qV/k_B T}} = n_{{P0}}\\, e^{{V/V_T}} = {_x10(_n_P0)} \\times e^{{{_V:.2f}/{kT_q:.3f}}} = {_x10(_n_at_neg_xP)}\\ \\mathrm{{cm}}^{{-3}}$$"
    _p_eq = f"$$p(x_n) = p_{{N0}}\\, e^{{qV/k_B T}} = p_{{N0}}\\, e^{{V/V_T}} = {_x10(_p_N0)} \\times e^{{{_V:.2f}/{kT_q:.3f}}} = {_x10(_p_at_xN)}\\ \\mathrm{{cm}}^{{-3}}$$"
    _part_i = mo.md(
        f"""
        **Part (i) — Minority carrier concentrations at depletion layer edges**

        - Electron concentration at $x=-x_p$ (p-side edge): $n(-x_p) = n_{{P0}}\\, e^{{qV/k_B T}}$, where $k_B T/q \\approx 0.026\\ \\mathrm{{V}}$ at 300K
        {_n_eq}

        - Hole concentration at $x=x_n$ (n-side edge): $p(x_n) = p_{{N0}}\\, e^{{qV/k_B T}}$, where $k_B T/q \\approx 0.026\\ \\mathrm{{V}}$ at 300K
        {_p_eq}
        """
    )
    _excess_n_eq = f"$$n_P'(-x_p) \\equiv n(-x_p) - n_{{P0}} = {_x10(_n_at_neg_xP)} - {_x10(_n_P0)} = {_x10(_excess_n_P)}\\ \\mathrm{{cm}}^{{-3}}$$"
    _excess_p_eq = f"$$p_N'(x_n) \\equiv p(x_n) - p_{{N0}} = {_x10(_p_at_xN)} - {_x10(_p_N0)} = {_x10(_excess_p_N)}\\ \\mathrm{{cm}}^{{-3}}$$"

    _takeaway = (
        f"**Takeaway:**\n"
        f"Compare the equilibrium values $n_{{P0}} = {_x10(_n_P0)}\\, \\mathrm{{cm}}^{{-3}}$ and $p_{{N0}} = {_x10(_p_N0)}\\, \\mathrm{{cm}}^{{-3}}$ to the concentrations at the depletion edges under forward bias: "
        f"$n(-x_p) = {_x10(_n_at_neg_xP)}\\, \\mathrm{{cm}}^{{-3}}$ and $p(x_n) = {_x10(_p_at_xN)}\\, \\mathrm{{cm}}^{{-3}}$.\n"
        "The minority carrier concentrations at the depletion edges are **completely dominated by the excess carriers** in forward bias "
        r"($n(-x_p) \approx n_P'(-x_p)$, $p(x_n) \approx p_N'(x_n)$ when $e^{qV/k_B T} \gg 1$)."
    )
    _part_ii = mo.vstack([
        mo.md(r"**Part (ii) — Excess minority carrier concentrations**"),
        mo.md(r"- Excess electron concentration at $x=-x_p$:"),
        mo.md(_excess_n_eq),
        mo.md(r"- Excess hole concentration at $x=x_n$:"),
        mo.md(_excess_p_eq),
        mo.md(_takeaway),
    ])
    mo.vstack([step1_header, _part_equil, _part_i, _part_ii])
    return


@app.cell
def _(IMAGE_BASE, mo):
    _coord_caption = mo.md(r"*Schematic: depletion edges at $-x_p$ and $x_n$.*")
    headerfig = mo.vstack([
        mo.md(r"## Minority Carrier Diffusion Equation"),
        mo.hstack([mo.image(src=f"{IMAGE_BASE}/depletion-region.png", width="60%")], justify="center"),
        _coord_caption,
    ])
    return (headerfig,)


@app.cell
def _(headerfig, mo):
    _text = mo.md(r"""
    ### Step 1 (continued): Solve for $p_N'(x)$ (excess holes in the N-region)

    **Domain:** $x_n < x < \infty$ (quasi-neutral N-side)

    **Differential equation:**
    $$\frac{d^2 p_N'}{dx^2} = \frac{p_N'}{L_p^2}$$
    where $L_p = \sqrt{D_p \tau_p}$ is the **hole diffusion length** and $\tau_p$ is the hole minority carrier lifetime.

    **Boundary conditions (infinitely long diode):**
    1. At the edge of the depletion region on the N-side: $p_N'(x_n) = p_{N0}\left(e^{qV/k_B T} - 1\right)$ (with $p_{N0} = n_i^2/N_d$).
    2. Far into the N-region: $p_N'(\infty) = 0$ (infinite diode approx.).

    **Solution:** General form is $p_N'(x) = A\, e^{(x-x_n)/L_p} + B\, e^{-(x-x_n)/L_p}$. The condition $p_N'(\infty) = 0$ gives $A = 0$. Applying the boundary condition at $x = x_n$:

    $$\boxed{p_N'(x) = p_{N0}\left(e^{qV/k_B T} - 1\right) e^{-(x-x_n)/L_p}, \qquad x \geq x_n}$$

    Excess holes decay **exponentially** away from the depletion edge with characteristic length $L_p$.
    """)
    mo.vstack([headerfig, _text])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 1 (continued): Solve for $n_P'(x)$ (excess electrons in the P-region)

    **Domain:** $-\infty < x < -x_p$ (quasi-neutral P-side)

    **Differential equation:**
    $$\frac{d^2 n_P'}{dx^2} = \frac{n_P'}{L_n^2}$$
    where $L_n = \sqrt{D_n \tau_n}$ is the **electron diffusion length**.

    **Boundary conditions (infinitely long diode):**
    1. At the edge of the depletion region on the P-side: $n_P'(-x_p) = n_{P0}\left(e^{qV/k_B T} - 1\right)$ (with $n_{P0} = n_i^2/N_a$).
    2. Far into the P-region: $n_P'(-\infty) = 0$ (infinite diode approx.).

    **Solution:** General form is $n_P'(x) = A\, e^{(x+x_p)/L_n} + B\, e^{-(x+x_p)/L_n}$. The condition $n_P'(-\infty) = 0$ gives $B = 0$. Applying the boundary condition at $x = -x_p$:

    $$\boxed{n_P'(x) = n_{P0}\left(e^{qV/k_B T} - 1\right) e^{(x+x_p)/L_n}, \qquad x \leq -x_p}$$

    Excess electrons decay **exponentially** away from the depletion edge with characteristic length $L_n$.
    """)
    return


@app.cell
def _(mo):
    # No leading underscore so these are exported to the next cell (marimo convention)
    Va_slider = mo.ui.slider(
        start=-2.0, stop=0.7, value=0.3, step=0.05,
        label=r"Applied voltage $V$ (V)"
    )
    Na_slider = mo.ui.slider(
        start=15, stop=18, value=16, step=0.5,
        label=r"log$_{10}$($N_a$) [cm$^{-3}$]"
    )
    Nd_slider = mo.ui.slider(
        start=15, stop=19, value=16, step=0.5,
        label=r"log$_{10}$($N_d$) [cm$^{-3}$]"
    )
    Lp_slider = mo.ui.slider(
        start=1, stop=20, value=5, step=1,
        label=r"$L_p$ (um)"
    )
    Ln_slider = mo.ui.slider(
        start=1, stop=20, value=5, step=1,
        label=r"$L_n$ (um)"
    )
    carrier_controls = mo.vstack([
        mo.md("### Settings"),
        mo.hstack([Va_slider, Na_slider, Nd_slider], justify="start"),
        mo.hstack([Lp_slider, Ln_slider], justify="start"),
    ])
    return (
        Ln_slider,
        Lp_slider,
        Na_slider,
        Nd_slider,
        Va_slider,
        carrier_controls,
    )


@app.cell
def _(
    D_n,
    D_p,
    Ln_slider,
    Lp_slider,
    Na_slider,
    Nd_slider,
    Va_slider,
    carrier_controls,
    eps_Si,
    kT_q,
    mo,
    ni_Si,
    np,
    plt,
    q,
):
    # Read slider values (sliders defined in previous cell; no leading underscore so they are exported)
    _Va = Va_slider.value
    _Na = 10 ** Na_slider.value
    _Nd = 10 ** Nd_slider.value
    _Lp_um = Lp_slider.value
    _Ln_um = Ln_slider.value

    # Convert diffusion lengths to cm
    _Lp = _Lp_um * 1e-4  # um to cm
    _Ln = _Ln_um * 1e-4  # um to cm

    # Built-in potential
    _phi_bi = kT_q * np.log(_Na * _Nd / ni_Si**2)

    # Depletion widths at applied voltage (must ensure phi_bi - Va > 0 for forward bias)
    _V_eff = _phi_bi - _Va
    if _V_eff < 0.01:
        _V_eff = 0.01  # prevent negative under barrier

    _xn = np.sqrt(2 * eps_Si * _V_eff / q * _Na / (_Nd * (_Na + _Nd)))
    _xp = np.sqrt(2 * eps_Si * _V_eff / q * _Nd / (_Na * (_Na + _Nd)))

    # Equilibrium minority carrier concentrations
    _pn0 = ni_Si**2 / _Nd  # holes in N-side
    _np0 = ni_Si**2 / _Na  # electrons in P-side

    # Excess carrier at depletion edges
    _exp_factor = np.exp(_Va / kT_q) - 1
    _delta_pn_edge = _pn0 * _exp_factor
    _delta_np_edge = _np0 * _exp_factor

    # Spatial grids (in cm, converted to um for plotting)
    _x_margin_n = 4.0 * _Lp  # show several diffusion lengths
    _x_margin_p = 4.0 * _Ln

    # N-side: x >= x_n
    _x_n_region = np.linspace(_xn, _xn + _x_margin_n, 300)
    _delta_pn = _delta_pn_edge * np.exp((_xn - _x_n_region) / _Lp)

    # P-side: x <= -x_p
    _x_p_region = np.linspace(-_xp - _x_margin_p, -_xp, 300)
    _delta_np = _delta_np_edge * np.exp((_xp + _x_p_region) / _Ln)

    # Total minority carrier concentrations
    _pn_total = _pn0 + _delta_pn
    _np_total = _np0 + _delta_np

    # Current densities at depletion edges
    _Jp_xn = q * D_p / _Lp * _pn0 * _exp_factor
    _Jn_xp = q * D_n / _Ln * _np0 * _exp_factor
    _Jtotal = _Jp_xn + _Jn_xp

    # --- Plotting --- (use _fig, _ax1, _ax2 so we don't redefine names from other cells)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Convert positions to um for display
    _x_n_um = _x_n_region * 1e4
    _x_p_um = _x_p_region * 1e4
    _xn_um = _xn * 1e4
    _xp_um = _xp * 1e4

    # Left plot: Excess minority carrier concentration
    _ax1.plot(_x_p_um, _delta_np, "b-", linewidth=2.5, label=r"$n'_P(x)$ (electrons in P)")
    _ax1.plot(_x_n_um, _delta_pn, "r-", linewidth=2.5, label=r"$p'_N(x)$ (holes in N)")

    # Shade depletion region between the two distributions
    _ax1.axvspan(-_xp_um, _xn_um, alpha=0.15, color="gray", label="Depletion region")
    _ax1.text(0.5 * (-_xp_um + _xn_um), 0.5, "Depletion", fontsize=12, ha="center", va="center", color="gray",
              transform=_ax1.get_xaxis_transform())
    _ax1.axhline(0, color="k", linewidth=0.5, linestyle="-")
    _ax1.axvline(-_xp_um, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    _ax1.axvline(_xn_um, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    _ax1.set_xlabel(r"Position $x$ ($\mu$m)", fontsize=13)
    _ax1.set_ylabel(r"Excess carrier concentration (cm$^{-3}$)", fontsize=13)
    _ax1.set_xlim(-40, 40)
    _bias_label = "Forward" if _Va > 0 else ("Reverse" if _Va < 0 else "Zero")
    _ax1.set_title(f"Excess Minority Carriers ({_bias_label} bias, $V$ = {_Va:.2f} V)", fontsize=13, fontweight="bold")
    _ax1.legend(fontsize=11, loc="best")
    _ax1.grid(True, alpha=0.3)

    # Annotate diffusion lengths (only in forward bias when excess > 0)
    if _Va > 0:
        # L_p on N-side (red curve)
        _ax1.annotate(
            "", xy=(_xn_um + _Lp_um, _delta_pn_edge * np.exp(-1)),
            xytext=(_xn_um, _delta_pn_edge * np.exp(-1)),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
        )
        _ax1.text(
            _xn_um + _Lp_um / 2, _delta_pn_edge * np.exp(-1) * 1.15,
            f"$L_p$ = {_Lp_um} μm", fontsize=10, color="red", ha="center",
        )
        # L_n on P-side (blue curve): arrow from -x_p leftward by L_n at 1/e height
        _y_ln = _delta_np_edge * np.exp(-1)
        _x_ln_start = -_xp_um
        _x_ln_end = -_xp_um - _Ln_um
        _ax1.annotate(
            "", xy=(_x_ln_end, _y_ln), xytext=(_x_ln_start, _y_ln),
            arrowprops=dict(arrowstyle="<->", color="blue", lw=1.5),
            clip_on=False,
        )
        _ylim = _ax1.get_ylim()
        _y_label_ln = max(_y_ln * 1.15, _ylim[0] + 0.02 * (_ylim[1] - _ylim[0]))
        _ax1.text(
            _x_ln_start + (_x_ln_end - _x_ln_start) / 2,
            _y_label_ln,
            f"$L_n$ = {_Ln_um} μm", fontsize=10, color="blue", ha="center", clip_on=False,
        )

    # Right plot: Total minority carrier concentration (log scale)
    _ax2.semilogy(_x_p_um, np.maximum(_np_total, 1e-1), "b-", linewidth=2.5,
                  label=r"$n_p(x)$ (electrons in P)")
    _ax2.semilogy(_x_n_um, np.maximum(_pn_total, 1e-1), "r-", linewidth=2.5,
                  label=r"$p_n(x)$ (holes in N)")

    # Equilibrium levels (format as ×10^n, not e+4)
    def _x10_label(x):
        if x == 0:
            return "0"
        e = int(np.floor(np.log10(abs(x))))
        m = x / (10**e)
        if abs(m) >= 9.995:
            m /= 10
            e += 1
        return f"{m:.2f} \\times 10^{{{e}}}"
    _ax2.axhline(_pn0, color="r", linestyle=":", alpha=0.6, label=f"$p_{{n0}}$ = {_x10_label(_pn0)}")
    _ax2.axhline(_np0, color="b", linestyle=":", alpha=0.6, label=f"$n_{{p0}}$ = {_x10_label(_np0)}")

    _ax2.axvspan(-_xp_um, _xn_um, alpha=0.15, color="gray")
    _ax2.axvline(-_xp_um, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    _ax2.axvline(_xn_um, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    _ax2.set_xlabel(r"Position $x$ ($\mu$m)", fontsize=13)
    _ax2.set_ylabel(r"Carrier concentration (cm$^{-3}$)", fontsize=13)
    _ax2.set_xlim(-40, 40)
    _ax2.set_title("Total Minority Carrier Profile (log scale)", fontsize=13, fontweight="bold")
    _ax2.legend(fontsize=10, loc="best")
    _ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    def _x10_fmt(x):
        """Format number as mantissa ×10^exp with 2 decimal places (for table)."""
        if x == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(x))))
        mantissa = x / (10**exp)
        if abs(mantissa) >= 9.995:
            mantissa /= 10
            exp += 1
        return f"{mantissa:.2f}×10$^{{{exp}}}$"

    _info = mo.md(
        "**Computed values:**\n\n"
        "| Quantity | Value |\n"
        "|----------|-------|\n"
        f"| $\\phi_{{bi}}$ | {_phi_bi:.2f} V |\n"
        f"| $x_n$ | {_xn*1e4:.2f} μm |\n"
        f"| $x_p$ | {_xp*1e4:.2f} μm |\n"
        f"| $p_{{n0}}$ | {_x10_fmt(_pn0)} cm$^{{-3}}$ |\n"
        f"| $n_{{p0}}$ | {_x10_fmt(_np0)} cm$^{{-3}}$ |\n"
        f"| $p'_N(x_n)$ | {_x10_fmt(_delta_pn_edge)} cm$^{{-3}}$ |\n"
        f"| $n'_P(-x_p)$ | {_x10_fmt(_delta_np_edge)} cm$^{{-3}}$ |\n"
    )

    _header = mo.md("### Step 1 — Interactive: Minority Carrier Distributions")
    # Last expression + return so marimo displays the output
    _out = mo.vstack([_header, carrier_controls, _fig, _info])
    _out
    return


@app.cell
def _(IMAGE_BASE, mo):
    mo.vstack([
        mo.md(r"""
        ### Observations
        - Minority carrier densities are increased relative to equilibrium values near the depletion region edges under forward bias (minority carrier injection).
        - Minority carrier densities are reduced relative to equilibrium values near the depletion region edges under reverse bias. 
        """),
        mo.hstack([
        mo.image(src=f"{IMAGE_BASE}/minority-carrier-dist.png", width="100%"),], justify="center"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: Find $J_{P,\text{diff}}(x = x_n)$ and $J_{N,\text{diff}}(x = -x_p)$

    Since there is no electric field in the quasi-neutral regions (low-level injection), current
    is carried entirely by **diffusion** of minority carriers.

    ### 2.1 Hole diffusion current density ($J_{P,\text{diff}}$)

    **$J_{P,\text{diff}}(x = x_n)$**

    $$J_{p,\text{diff}} = -q D_p \frac{dp'_N(x)}{dx}
    = q \frac{D_p}{L_p}\, p_{N0}\left(e^{qV/k_B T} - 1\right) e^{-(x-x_n)/L_p}$$

    At the N-side edge of the depletion region ($x = x_n$, so $e^{-(x-x_n)/L_p} = 1$):

    $$\boxed{J_{p,\text{diff}}(x = x_n) = q \frac{D_p}{L_p}\, p_{N0}\left(e^{qV/k_B T} - 1\right)}$$

    (with $p_{N0} = n_i^2/N_d$.)

    ### 2.2 Electron diffusion current density ($J_{N,\text{diff}}$)

    **$J_{N,\text{diff}}(x = -x_p)$**

    $$J_{n,\text{diff}} = q D_n \frac{dn'_P(x)}{dx}
    = q \frac{D_n}{L_n}\, n_{P0}\left(e^{qV/k_B T} - 1\right) e^{(x+x_p)/L_n}$$

    At the P-side edge of the depletion region ($x = -x_p$, so $e^{(x+x_p)/L_n} = 1$):

    $$\boxed{J_{n,\text{diff}}(x = -x_p) = q \frac{D_n}{L_n}\, n_{P0}\left(e^{qV/k_B T} - 1\right)}$$

    (with $n_{P0} = n_i^2/N_a$.)

    ## Step 3: Total current

    - $J = J_{P,\text{diff}}(x = x_n) + J_{N,\text{diff}}(x = -x_p)$

    $$\boxed{J = q\left(\frac{D_n\, n_{P0}}{L_n} + \frac{D_p\, p_{N0}}{L_p}\right)\left(e^{qV/k_B T} - 1\right)}$$

    **Diode I–V relation:**

    $$I = I_0\left(e^{qV/k_B T} - 1\right), \qquad I_0 = qA\left(\frac{D_n\, n_{P0}}{L_n} + \frac{D_p\, p_{N0}}{L_p}\right)$$

    ($A$ = diode cross-sectional area; $I_0$ is the reverse saturation current.)

    **Simplification** (for uniformly doped diodes):

    $$n_{P0} \approx \frac{n_i^2}{N_a}, \qquad p_{N0} \approx \frac{n_i^2}{N_d}$$
    """)
    return


@app.cell
def _(
    D_n,
    D_p,
    Ln_slider,
    Lp_slider,
    Na_slider,
    Nd_slider,
    Va_slider,
    carrier_controls,
    eps_Si,
    kT_q,
    mo,
    ni_Si,
    np,
    plt,
    q,
):
    # Hole and electron currents vs x (reuse Minority Carrier Distribution sliders)
    _Va = Va_slider.value
    _Na = 10 ** Na_slider.value
    _Nd = 10 ** Nd_slider.value
    _Lp_um = Lp_slider.value
    _Ln_um = Ln_slider.value
    _Lp = _Lp_um * 1e-4
    _Ln = _Ln_um * 1e-4

    _phi_bi = kT_q * np.log(_Na * _Nd / ni_Si**2)
    _V_eff = _phi_bi - _Va
    if _V_eff < 0.01:
        _V_eff = 0.01
    _xn = np.sqrt(2 * eps_Si * _V_eff / q * _Na / (_Nd * (_Na + _Nd)))
    _xp = np.sqrt(2 * eps_Si * _V_eff / q * _Nd / (_Na * (_Na + _Nd)))

    _pn0 = ni_Si**2 / _Nd
    _np0 = ni_Si**2 / _Na
    _exp_factor = np.exp(_Va / kT_q) - 1
    _Jp_xn = q * D_p / _Lp * _pn0 * _exp_factor
    _Jn_xp = q * D_n / _Ln * _np0 * _exp_factor
    _Jtotal = _Jp_xn + _Jn_xp

    # Spatial grid spanning P, depletion, N (in cm)
    _x_margin_p = 4.0 * _Ln
    _x_margin_n = 4.0 * _Lp
    _x_p = np.linspace(-_xp - _x_margin_p, -_xp, 200)
    _x_dep = np.linspace(-_xp, _xn, 80)
    _x_n = np.linspace(_xn, _xn + _x_margin_n, 200)
    _x_all = np.concatenate([_x_p, _x_dep[1:-1], _x_n])  # avoid duplicate edges

    # Current densities: diffusion from expressions; majority = J_tot - J_diffusion
    # P-side (x <= -x_p): J_n = diffusion (electrons), J_p = J_tot - J_n (drift majority)
    _Jn_diff_P = q * D_n / _Ln * _np0 * _exp_factor * np.exp((_x_p + _xp) / _Ln)
    _Jp_P = _Jtotal - _Jn_diff_P
    # Depletion (-x_p <= x <= x_n): J_n and J_p constant (no G-R)
    _Jn_dep = np.full_like(_x_dep, _Jn_xp)
    _Jp_dep = np.full_like(_x_dep, _Jp_xn)
    # N-side (x >= x_n): J_p = diffusion (holes), J_n = J_tot - J_p (drift majority)
    _Jp_diff_N = q * D_p / _Lp * _pn0 * _exp_factor * np.exp(-(_x_n - _xn) / _Lp)
    _Jn_N = _Jtotal - _Jp_diff_N

    _Jn_plot = np.concatenate([_Jn_diff_P, _Jn_dep[1:-1], _Jn_N])
    _Jp_plot = np.concatenate([_Jp_P, _Jp_dep[1:-1], _Jp_diff_N])
    _x_um = _x_all * 1e4
    _xn_um = _xn * 1e4
    _xp_um = _xp * 1e4

    _fig_j, _ax_j = plt.subplots(figsize=(12, 6), constrained_layout=True)
    _ax_j.plot(_x_um, _Jn_plot * 1e3, "b-", linewidth=2.5, label=r"$J_n$ (electron)")
    _ax_j.plot(_x_um, _Jp_plot * 1e3, "r-", linewidth=2.5, label=r"$J_p$ (hole)")
    _ax_j.axhline(_Jtotal * 1e3, color="k", linestyle="-", linewidth=3.5, label=r"$J_{total}$")
    _ax_j.axvspan(-_xp_um, _xn_um, alpha=0.15, color="gray")
    _ax_j.axvline(-_xp_um, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    _ax_j.axvline(_xn_um, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    _ax_j.axhline(0, color="k", linewidth=0.5)
    _ax_j.set_xlabel(r"Position $x$ ($\mu$m)", fontsize=18)
    _ax_j.set_ylabel(r"Current density (mA/cm²)", fontsize=18)
    _ax_j.set_xlim(-40, 40)
    _ax_j.set_title(r"Hole and electron currents  ($J = J_p + J_n$ constant)", fontsize=18, fontweight="bold")
    _ax_j.legend(fontsize=11, loc="upper right")
    _ax_j.grid(True, alpha=0.3)
    _ax_j.text(-_xp_um - _Ln_um * 1.5, _Jtotal * 1e3 * 0.75, "drift\n(majority $J_p$)", fontsize=16, color="red", ha="center")
    _ax_j.text(_xn_um + _Lp_um * 1.5, _Jtotal * 1e3 * 0.75, "drift\n(majority $J_n$)", fontsize=16, color="blue", ha="center")
    _ax_j.text(-_xp_um - _Ln_um , _Jn_xp * 1e3 * 0.4, "diffusion\n($J_n$)", fontsize=16, color="blue", ha="center")
    _ax_j.text(_xn_um + _Lp_um , _Jp_xn * 1e3 * 0.4, "diffusion\n($J_p$)", fontsize=16, color="red", ha="center")

    _caption = mo.md(r"""
    - Same sliders as Minority Carrier Distribution. Changing them updates both plots.
    - **Current is constant in the diode:** $J_{total} = J_p + J_n$.  
    - In quasi-neutral regions, minority current is **diffusion**; majority current is due to **drift** $J_{drift} = J_{total} - J_{diffusion}$.
        - Since $J_{drift} \neq 0$, $\mathcal{E} \neq 0$ in the quasi-neutral region! 
    - In the depletion region, $J_n$ and $J_p$ are constant (no G–R). 
    """)
    _header = mo.md("### Interactive: Hole and Electron Currents (Steps 2 & 3)")

    mo.vstack([_header, carrier_controls, _fig_j, _caption])
    return


@app.cell
def _(carrier_controls, mo):
    T_slider_J = mo.ui.slider(
        start=250, stop=400, value=300, step=10,
        label="Temperature $T$ (K)"
    )
    J_fixed_log_slider = mo.ui.slider(
        start=0, stop=1.5, value=0.5, step=0.1,
        label=r"log₁₀(Fixed $J$ in mA/cm²) for d$V$/d$T$"
    )
    iv_controls = mo.vstack([
        carrier_controls,
        mo.hstack([T_slider_J, J_fixed_log_slider], justify="start"),
    ])
    return J_fixed_log_slider, T_slider_J, iv_controls


@app.cell
def _(
    Eg_Si,
    J_fixed_log_slider,
    Ln_slider,
    Lp_slider,
    Na_slider,
    Nd_slider,
    T_slider_J,
    iv_controls,
    mo,
    mu_n,
    mu_p,
    np,
    plt,
    q,
):
    # Silicon: temperature-dependent n_i and k_B T
    _k_B_eV = 8.617e-5  # eV/K
    _T = T_slider_J.value
    _kT_q = _k_B_eV * _T  # k_B T / q in V
    _Nc_300 = 2.86e19   # cm^-3 (Si)
    _Nv_300 = 3.10e19   # cm^-3 (Si)
    _Nc = _Nc_300 * (_T / 300) ** 1.5
    _Nv = _Nv_300 * (_T / 300) ** 1.5
    _ni_T = np.sqrt(_Nc * _Nv * np.exp(-Eg_Si / (_k_B_eV * _T)))

    _D_n = _kT_q * mu_n  # cm^2/s
    _D_p = _kT_q * mu_p  # cm^2/s

    _Na = 10 ** Na_slider.value
    _Nd = 10 ** Nd_slider.value
    _Ln_um = Ln_slider.value
    _Lp_um = Lp_slider.value
    _Ln = _Ln_um * 1e-4  # cm
    _Lp = _Lp_um * 1e-4  # cm

    _n_P0 = _ni_T**2 / _Na
    _p_N0 = _ni_T**2 / _Nd
    _J_0 = q * (_D_n * _n_P0 / _Ln + _D_p * _p_N0 / _Lp)

    _V = np.linspace(-2, 0.8, 500)
    _J = _J_0 * (np.exp(_V / _kT_q) - 1)  # J = J_0 (exp(q V / k_B T) - 1)
    _J_mA_cm2 = _J * 1e3  # mA/cm²

    _eq_block = mo.md(r"""
    ## Interactive I–V Characteristics

    ### Temperature dependence and I–V relation

    - Neglect the temperature dependence of $m_n^*$, $m_p^*$, diffusion coefficients, diffusion length, and $E_g$ (true for small perturbations in temperature)

    - Intrinsic concentration: $n_i^2(T) = N_c(T)\, N_v(T)\, \exp(-E_g/k_B T)$, with $N_c,\, N_v \propto T^{3/2}$.

    - Minority densities: $n_{P0} = n_i^2/N_a$, $\quad p_{N0} = n_i^2/N_d$

    - Saturation current density: $J_0 = q\left(\frac{D_n\, n_{P0}}{L_n} + \frac{D_p\, p_{N0}}{L_p}\right)$

    - Current density vs voltage: $\boxed{J = J_0\left(e^{qV/k_B T} - 1\right)}$
        """
    )

    _fig_iv, (_ax1_iv, _ax2_iv) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # --- Left: J vs V (linear) ---
    _ax1_iv.plot(_V, _J_mA_cm2, "b-", linewidth=2.5)
    _ax1_iv.axhline(0, color="k", linewidth=0.5)
    _ax1_iv.axvline(0, color="k", linewidth=0.5)
    _J_max = _J_0 * (np.exp(0.8 / _kT_q) - 1) * 1e3
    _ax1_iv.set_ylim(-abs(_J_0) * 1e3 * 1.2, max(_J_max * 1.1, 0.01))
    _ax1_iv.axhline(-_J_0 * 1e3, color="r", linestyle="--", alpha=0.7, linewidth=1.5)
    _ax1_iv.set_xlabel(r"$V$ (V)", fontsize=16)
    _ax1_iv.set_ylabel(r"$J$ (mA/cm²)", fontsize=16)
    _ax1_iv.set_title(r"$J$–$V$ (linear)", fontsize=16, fontweight="bold")
    _ax1_iv.grid(True, alpha=0.3)

    # --- Right: |J| vs V (semi-log) ---
    _J_plot = np.where(_J > 0, _J, -_J)
    _J_plot = np.maximum(_J_plot, 1e-20)
    _ax2_iv.semilogy(_V, _J_plot * 1e3, "b-", linewidth=2.5, label=r"$|J|$")
    _ax2_iv.axhline(_J_0 * 1e3, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    def _x10_fmt_iv(x):
        if x == 0:
            return "0"
        e = int(np.floor(np.log10(abs(x))))
        m = x / (10**e)
        if abs(m) >= 9.995:
            m /= 10
            e += 1
        return f"{m:.2f}×10$^{{{e}}}$"
    _ax2_iv.text(-1.8, _J_0 * 1e3 * 2, f"$J_0$ = {_x10_fmt_iv(_J_0*1e3)} mA/cm²", fontsize=11, color="gray")
    _slope = 1 / (_kT_q * np.log(10))
    _ax2_iv.text(0.4, _J_0 * np.exp(0.35 / _kT_q) * 1e3 * 0.4,
        f"Slope $\\approx q/k_B T\\cdot\\ln(10)^{{-1}}$\n= {_slope:.1f} dec/V\n($T$ = {_T} K)",
        fontsize=14, color="blue", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    _ax2_iv.set_xlabel(r"$V$ (V)", fontsize=16)
    _ax2_iv.set_ylabel(r"$|J|$ (mA/cm²)", fontsize=16)
    _ax2_iv.set_title(r"$J$–$V$ (semi-log)", fontsize=16, fontweight="bold")
    _ax2_iv.grid(True, alpha=0.3, which="both")
    _ax2_iv.set_xlim(-2, 0.85)

    _values = mo.md(
        f"**At $T$ = {_T} K:** "
        f"$n_i$ = {_x10_fmt_iv(_ni_T)} cm$^{{-3}}$; "
        f"$n_{{P0}}$ = {_x10_fmt_iv(_n_P0)} cm$^{{-3}}$; "
        f"$p_{{N0}}$ = {_x10_fmt_iv(_p_N0)} cm$^{{-3}}$; "
        f"$J_0$ = {_x10_fmt_iv(_J_0*1e3)} mA/cm²."
    )

    # dV/dT at fixed J: V(T) such that J = J_0(T)*(exp(V/(kT/q)) - 1), then numerical derivative
    _J_fixed_mA = 10.0 ** J_fixed_log_slider.value  # mA/cm²
    _J_fixed = _J_fixed_mA * 1e-3  # A/cm²
    _T_arr = np.linspace(260, 390, 300)
    _ni_arr = np.sqrt(
        _Nc_300 * _Nv_300 * (_T_arr / 300) ** 3 * np.exp(-Eg_Si / (_k_B_eV * _T_arr))
    )
    _kT_q_arr = _k_B_eV * _T_arr
    _Dn_arr = _kT_q_arr * mu_n
    _Dp_arr = _kT_q_arr * mu_p
    _nP0_arr = _ni_arr**2 / _Na
    _pN0_arr = _ni_arr**2 / _Nd
    _J0_arr = q * (_Dn_arr * _nP0_arr / _Ln + _Dp_arr * _pN0_arr / _Lp)
    # V = (k_B*T/q) * ln(1 + J/J_0(T))
    _V_at_J = _kT_q_arr * np.log(1.0 + _J_fixed / np.maximum(_J0_arr, 1e-30))
    _dV_dT = np.gradient(_V_at_J, _T_arr)  # V/K

    _fig_dV, _ax_dV = plt.subplots(figsize=(10, 5), constrained_layout=True)
    _ax_dV.plot(_T_arr, _dV_dT * 1e3, "b-", linewidth=2.5)  # mV/K for readability
    _ax_dV.axhline(0, color="k", linewidth=0.5)
    _ax_dV.set_xlabel(r"Temperature $T$ (K)", fontsize=16)
    _ax_dV.set_ylabel(r"$dV/dT$ (mV/K)", fontsize=16)
    _ax_dV.set_title(r"$dV/dT$ at fixed $J$ = " + f"{_J_fixed_mA:.3g} mA/cm²", fontsize=14, fontweight="bold")
    _ax_dV.grid(True, alpha=0.3)

    mo.vstack([
        _eq_block, iv_controls, _values, _fig_iv,
        _fig_dV,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary of Key Equations

    | Quantity | Expression |
    |:---|:---|
    | np product | $np = n_i^2 \exp\!\left(\dfrac{qV}{k_B T}\right)$ |
    | Excess holes (N-side) | $p_N'(x) = \dfrac{n_i^2}{N_d}\left[e^{qV/k_B T} - 1\right] e^{-(x-x_n)/L_p}$ |
    | Excess electrons (P-side) | $n_P'(x) = \dfrac{n_i^2}{N_a}\left[e^{qV/k_B T} - 1\right] e^{(x+x_p)/L_n}$ |
    | Diffusion lengths | $L_p = \sqrt{D_p \tau_p}$, $\quad L_n = \sqrt{D_n \tau_n}$ |
    | Hole current at $x_n$ | $J_p = \dfrac{qD_p}{L_p}\dfrac{n_i^2}{N_d}\left[e^{qV/k_B T} - 1\right]$ |
    | Electron current at $-x_p$ | $J_n = \dfrac{qD_n}{L_n}\dfrac{n_i^2}{N_a}\left[e^{qV/k_B T} - 1\right]$ |
    | **Saturation current** | $J_0 = q\left(\dfrac{D_p}{L_p}\dfrac{n_i^2}{N_d} + \dfrac{D_n}{L_n}\dfrac{n_i^2}{N_a}\right)$ |
    | **Shockley diode equation** | $I = I_0\left[\exp\!\left(\dfrac{qV}{k_B T}\right) - 1\right]$ |
    """)
    return


if __name__ == "__main__":
    app.run()
