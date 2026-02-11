# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "matplotlib==3.10.8",
# ]
# ///

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # For WASM (Pyodide), __file__ is not meaningful; use GitHub Pages URL
    try:
        _test = Path(__file__).parent / "images"
        if _test.exists():
            ASSET_DIR = Path(__file__).parent
        else:
            raise FileNotFoundError
    except Exception:
        ASSET_DIR = None

    IMAGE_BASE = "https://joyce-poon.github.io/ECE350/pn-electrostatics/images" if ASSET_DIR is None else str(ASSET_DIR / "images")

    # Physical constants
    q = 1.6e-19  # C
    kT = 0.02585  # eV at 300K
    ni_Si = 1.1e10  # cm^-3 for Si at 300K
    eps_r = 11.7  # relative permittivity of Si
    eps_0 = 8.854e-14  # F/cm (vacuum permittivity in CGS)
    eps_s = eps_r * eps_0  # F/cm
    Eg_Si = 1.12  # eV

    mo.md(
        r"""
        # PN Junction Electrostatics

        ECE350, Lecture 17

        This notebook derives the electrostatic quantities of an abrupt junction at **thermal equilibrium**. Main concepts

        1. The depletion approximation
        2. Built-in potential derivation
        3. Charge $\rightarrow$ Electric field $\rightarrow$ Electrostatic potential 
        4. Depletion width
        5. Interactive plots: $\rho(x)$, $\mathcal{E}(x)$, $V(x)$, and energy band diagram


        ---
        """
    )
    return ASSET_DIR, Eg_Si, IMAGE_BASE, eps_s, kT, mo, ni_Si, np, plt, q


@app.cell
def _(IMAGE_BASE, mo):
    mo.vstack([
        mo.md(r"""
    ## 1. The Abrupt Junction & Depletion Approximation

    We solve for the electrostatics of the PN junction under the following simplifying assumptions:

    - **One-dimensional problem** -- all quantities depend only on $x$.
    - **Abrupt depletion edges** -- the depletion region extends from $x = -x_p$ (on the P-side) to $x = x_n$ (on the N-side), with abrupt transitions.
    - **Zero electric field outside the depletion region** -- $\mathcal{E} = 0$ for $x < -x_p$ and $x > x_n$.
    - **Complete ionization** -- $N_D^+ = N_D$ and $N_A^- = N_A$.
    - **Negligible free carriers in the depletion region** -- $n, p \ll N_A, N_D$ within the depletion region.
        """),
        mo.hstack([mo.image(src=f"{IMAGE_BASE}/lec13-09.png", width="50%")], justify="center"),
        mo.md(r"""
    Under these assumptions, the **charge density** is:

    $$\rho(x) = \begin{cases} -qN_A & -x_p \leq x \leq 0 \\ qN_D & 0 \leq x \leq x_n \\ 0 & \text{elsewhere} \end{cases}$$

    **Charge neutrality** requires the total positive charge to equal the total negative charge:

    $$qN_A \cdot x_p = qN_D \cdot x_n$$

    $$\boxed{N_A \, x_p = N_D \, x_n}$$

    The depletion region extends **further into the lightly doped side**.
        """),
    ])
    return


@app.cell
def _(IMAGE_BASE, mo):
    mo.vstack([
        mo.md(r"""
        ## 2. Built-in Potential Derivation
        """),
        mo.hstack([mo.image(
            src=f"{IMAGE_BASE}/lec14-04.png",
            width="50%"
        )], justify="center"),
        mo.md(r"""

        The **built-in potential** ($\phi_{bi}$) of the PN junction is the total potential difference across the depletion region at equilibrium.

        $$q\phi_{bi} = qB - qA \quad \text{where} \quad B = (E_c - E_F)_{\text{P-side}} \quad \text{and} \quad A = (E_c - E_F)_{\text{N-side}}$$

        $$qB =(E_c - E_F)_{\text{P-side}} = k_BT \ln\!\left(\frac{N_c}{n_{0,P}}\right) = k_BT \ln\!\left(\frac{N_c N_A}{n_i^2}\right)$$

        $$qA =(E_c - E_F)_{\text{N-side}} = k_BT \ln\!\left(\frac{N_c}{n_{0,N}}\right) = k_BT \ln\!\left(\frac{N_c}{N_D}\right)$$

        $$q\phi_{bi} = B - A = k_BT \ln\!\left(\frac{N_c N_A}{n_i^2}\right) - k_BT \ln\!\left(\frac{N_c}{N_D}\right)$$ 

        $$\therefore \boxed{q\phi_{bi} = k_BT \ln\!\left(\frac{N_A N_D}{n_i^2}\right)}$$        


    """)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Example: Magnitude of $\phi_{bi}$

    **Determine** the built-in potential of a Si PN junction at $T = 300$ K where $N_A = N_D = 10^{16}$ cm$^{-3}$.

    $$\phi_{bi} = \frac{kT}{q} \ln\!\left(\frac{N_A N_D}{n_i^2}\right) = 26\text{ mV} \times \ln\!\left(\frac{10^{16} \times 10^{16}}{1.21 \times 10^{20}}\right) = \boxed{0.71 \text{ V}}$$

    where we used $n_i^2 = (1.1 \times 10^{10})^2 = 1.21 \times 10^{20}$ cm$^{-6}$ for Si at 300 K.
    """)
    return


@app.cell
def _(IMAGE_BASE, mo):
    mo.hstack([
        mo.md(r"""
    ## 3. Charge $\rightarrow$ Electric Field $\rightarrow$ Electrostatic Potential

    Starting from Poisson's equation:

    $$\frac{d\mathcal{E}}{dx} = \frac{\rho(x)}{\varepsilon_s}$$

    we integrate in each region, applying the boundary conditions $\mathcal{E}(-x_p) = 0$ and $\mathcal{E}(x_n) = 0$:

    **P-side of depletion region** ($-x_p \leq x \leq 0$):

    $$\mathcal{E}(x) = -\frac{qN_A}{\varepsilon_s}(x + x_p)$$

    **N-side of depletion region** ($0 \leq x \leq x_n$):

    $$\mathcal{E}(x) = \frac{qN_D}{\varepsilon_s}(x - x_n)$$

    Key observations:
    - The electric field is **negative** everywhere in the depletion region.
    - The **maximum magnitude** occurs at the metallurgical junction ($x = 0$):

    $$\boxed{|\mathcal{E}_{\max}| = |\mathcal{E}(x=0)| = \frac{qN_A x_p}{\varepsilon_s} = \frac{qN_D x_n}{\varepsilon_s}}$$

    - We recover again our charge neutrality condition ($N_A x_p = N_D x_n$).
    - The depletion region extends further into the lightly doped side.
    - One side junctions: P$^+$N and N$^+$P have a heavily doped P or N side, respectively.
        """),
        mo.image(src=f"{IMAGE_BASE}/lec14-09.png", width="100%", caption="Derivation of electric field from charge density"),
    ], widths=[0.6, 0.4], align="center")
    return


@app.cell
def _(IMAGE_BASE, mo):
    mo.hstack([
        mo.md(r"""
    ### Electrostatic Potential

    The electrostatic potential is obtained from $\mathcal{E} = -dV/dx$, i.e., $V(x) = -\int \mathcal{E}\, dx$.

    Using the boundary condition $V(-x_p) = 0$ (reference potential on the P-side):

    **P-side of depletion region** ($-x_p \leq x \leq 0$):

    $$\frac{dV}{dx} = \frac{qN_A}{\varepsilon_s}(x + x_p)$$

    $$V(x) = \int \frac{qN_A}{\varepsilon_s}(x + x_p)\, dx = \frac{qN_A}{2\varepsilon_s}(x + x_p)^2 + C_1, \quad V(-x_p) = 0 \implies C_1 = 0$$

    $$\therefore \boxed{V(x) = \frac{qN_A}{2\varepsilon_s}(x + x_p)^2}$$

    **N-side of depletion region** ($0 \leq x \leq x_n$):

    $$\frac{dV}{dx} = -\frac{qN_D}{\varepsilon_s}(x - x_n)$$

    $$V(x) = - \frac{qN_D}{2\varepsilon_s}(x - x_n)^2 + C_2, \quad V(x_n) = \phi_{bi} \implies C_2 = \phi_{bi}$$

    $$\therefore \boxed{V(x) = - \frac{qN_D}{2\varepsilon_s}(x - x_n)^2 + \phi_{bi}}$$

    Therefore,

    $$
    V(x) = \begin{cases}
    0 & \text{for } x \leq -x_p \\[6pt]
    \displaystyle \frac{qN_A}{2\varepsilon_s}(x + x_p)^2 & \text{for } -x_p \leq x \leq 0 \\[6pt]
    \displaystyle \phi_{bi} - \frac{qN_D}{2\varepsilon_s}(x - x_n)^2 & \text{for } 0 \leq x \leq x_n \\[6pt]
    \phi_{bi} & \text{for } x > x_n
    \end{cases}
    $$
        """),
        mo.image(src=f"{IMAGE_BASE}/PN-potential.png", width="100%"),
    ], widths=[0.6, 0.4], align="center")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Depletion Width


    Matching $V(x)$ at $x = 0$,

    $$V(x=0) = \frac{qN_A}{2\varepsilon_s}x_p^2 = -\frac{qN_D}{2\varepsilon_s} x_n^2 + \phi_{bi}$$

    Substitute ($N_A x_p = N_D x_n$) into the equation above and solve for $x_n$ and $x_p$:

    $$\boxed{x_n = \sqrt{\frac{2\varepsilon_s}{q} \cdot \frac{N_A}{N_D(N_A + N_D)} \cdot \phi_{bi}}}$$

    $$\boxed{x_p = \sqrt{\frac{2\varepsilon_s}{q} \cdot \frac{N_D}{N_A(N_A + N_D)} \cdot \phi_{bi}}}$$

    The total depletion width is:

    $$\boxed{W = x_n + x_p = \sqrt{\frac{2\varepsilon_s}{q} \cdot \frac{N_A + N_D}{N_A N_D} \cdot \phi_{bi}}}$$

    **Typical values:** For $\phi_{bi} \approx 0.8$ V, $\varepsilon_s = 11.7\varepsilon_0$, and $N_A, N_D \approx 10^{17}$ cm$^{-3}$, the depletion width is $W \approx 0.1$ $\mu$m.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. One-Sided (Asymmetric) Junctions

       Sometimes, one side is doped much more heavily than the other. For example, in a $P^+N$ junction, $N_A \gg N_D$.

    ### Depletion width simplification

    When $N_A \gg N_D$:

    $$x_p = x_n \frac{N_D}{N_A} \ll x_n$$

    The depletion region is almost entirely on the **lightly doped** (N) side:

    $$W \approx x_n \approx \sqrt{\frac{2\varepsilon_s \phi_{bi}}{qN_D}}$$

    More generally, if $N_B$ is the lighter doping concentration:

    $$\boxed{W \approx \sqrt{\frac{2\varepsilon_s \phi_{bi}}{qN_B}}}$$

    ### Maximum electric field

    $$|\mathcal{E}_{\max}| = \frac{qN_B W}{\varepsilon_s} = \sqrt{\frac{2qN_B \phi_{bi}}{\varepsilon_s}}$$

    ### Why this matters

    - The one-sided approximation is common ($P^+N$ and $N^+P$ junctions).
    - The depletion width is controlled by the **lighter** doping.
    - This is the basis for the varactor diode (voltage-variable capacitor) and many other devices.
    - The breakdown voltage is also determined primarily by the lightly doped side.
    """)
    return


@app.cell
def _(mo):
    Na_slider = mo.ui.slider(
        14, 18, value=17, step=0.5,
        label=r"log(N_A) [cm⁻³]", show_value=True
    )
    Nd_slider = mo.ui.slider(
        14, 18, value=16, step=0.5,
        label=r"log(N_D) [cm⁻³]", show_value=True
    )
    return Na_slider, Nd_slider


@app.cell
def _(Eg_Si, Na_slider, Nd_slider, eps_s, kT, mo, ni_Si, np, plt, q):
    # Read slider values
    _Na = 10 ** Na_slider.value
    _Nd = 10 ** Nd_slider.value

    # Built-in potential
    _phi_bi = kT * np.log(_Na * _Nd / ni_Si**2)

    # Depletion widths (cm)
    _xn = np.sqrt(2 * eps_s * _phi_bi / q * _Na / (_Nd * (_Na + _Nd)))
    _xp = np.sqrt(2 * eps_s * _phi_bi / q * _Nd / (_Na * (_Na + _Nd)))
    _W = _xn + _xp

    # Maximum electric field magnitude (V/cm)
    _E_max = q * _Na * _xp / eps_s

    # Spatial grid (cm)
    _margin = 2.5
    _x_extent = max(_xn, _xp) * _margin
    _x = np.linspace(-_x_extent, _x_extent, 1000)

    # --- Charge density (C/cm^3) ---
    _rho = np.piecewise(
        _x,
        [_x < -_xp, (_x >= -_xp) & (_x <= 0), (_x > 0) & (_x <= _xn), _x > _xn],
        [0, -q * _Na, q * _Nd, 0],
    )

    # --- Electric field (V/cm) ---
    _E_field = np.zeros_like(_x)
    _mask_p = (_x >= -_xp) & (_x <= 0)
    _mask_n = (_x > 0) & (_x <= _xn)
    _E_field[_mask_p] = -q * _Na / eps_s * (_x[_mask_p] + _xp)
    _E_field[_mask_n] = q * _Nd / eps_s * (_x[_mask_n] - _xn)

    # --- Electrostatic potential (V) ---
    _V = np.zeros_like(_x)
    _mask_left = _x <= -_xp
    _mask_right = _x > _xn
    _V[_mask_left] = 0.0
    _V[_mask_p] = q * _Na / (2 * eps_s) * (_x[_mask_p] + _xp) ** 2
    _V[_mask_n] = _phi_bi - q * _Nd / (2 * eps_s) * (_x[_mask_n] - _xn) ** 2
    _V[_mask_right] = _phi_bi

    # Convert x-axis to micrometers for display
    _x_um = _x * 1e4
    _xp_um = _xp * 1e4
    _xn_um = _xn * 1e4

    # --- Four-panel plot ---
    _fig, _axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # (a) Charge density -- plot rho/q in units of cm^-3
    _axes[0].plot(_x_um, _rho / q * 1e-15, "b-", linewidth=2)
    _axes[0].set_ylabel(r"$\rho\,/\,q$ ($\times 10^{15}$ cm$^{-3}$)", fontsize=16)
    _fig.suptitle(
        r"Interactive Calculator of $\rho$, $\mathcal{E}$, $V$, and Energy Bands from $N_A$ and $N_D$",
        fontsize=15, fontweight="bold", y=1.01,
    )
    _axes[0].set_title(
        r"$N_A$ = " + f"{_Na:.1e}" + r" cm$^{-3}$, $N_D$ = " + f"{_Nd:.1e}" + r" cm$^{-3}$",
        fontsize=13,
    )
    _axes[0].axhline(0, color="gray", linewidth=0.5)
    _axes[0].axvline(-_xp_um, color="red", linestyle="--", alpha=0.6,
                     label=f"$-x_p$ = {-_xp_um:.4f} \u00b5m")
    _axes[0].axvline(_xn_um, color="blue", linestyle="--", alpha=0.6,
                     label=f"$x_n$ = {_xn_um:.4f} \u00b5m")
    _axes[0].fill_between(_x_um, _rho / q * 1e-15, 0, alpha=0.15, color="blue")
    _axes[0].legend(fontsize=16)
    _axes[0].axvline(0, color="black", linestyle=":", alpha=0.7, label="junction")
    _axes[0].grid(True, alpha=0.3)

    # (b) Electric field (V/cm)
    _axes[1].plot(_x_um, _E_field, "r-", linewidth=2)
    _axes[1].set_ylabel(r"$\mathcal{E}$ (V/cm)", fontsize=16)
    _axes[1].axhline(0, color="gray", linewidth=0.5)
    _axes[1].axvline(-_xp_um, color="red", linestyle="--", alpha=0.6)
    _axes[1].axvline(_xn_um, color="blue", linestyle="--", alpha=0.6)
    _axes[1].fill_between(_x_um, _E_field, 0, alpha=0.15, color="red")
    _axes[1].annotate(
        f"|$\\mathcal{{E}}_{{\\max}}$| = {_E_max:.2e} V/cm",
        xy=(0, np.min(_E_field)),
        xytext=(0.3 * _xn_um + 0.5 * _xp_um, np.min(_E_field) * 0.5),
        fontsize=16, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )
    _axes[1].axvline(0, color="black", linestyle=":", alpha=0.7)
    _axes[1].grid(True, alpha=0.3)

    # (c) Electrostatic potential (V)
    _axes[2].plot(_x_um, _V, "g-", linewidth=2)
    _axes[2].set_ylabel(r"$V$ (V)", fontsize=16)
    _axes[2].axhline(0, color="gray", linewidth=0.5)
    _axes[2].axhline(_phi_bi, color="green", linestyle="--", alpha=0.6,
                     label=rf"$\phi_{{bi}}$ = {_phi_bi:.3f} V")
    _axes[2].axvline(-_xp_um, color="red", linestyle="--", alpha=0.6)
    _axes[2].axvline(_xn_um, color="blue", linestyle="--", alpha=0.6)
    _axes[2].fill_between(_x_um, _V, 0, alpha=0.10, color="green")
    _axes[2].axvline(0, color="black", linestyle=":", alpha=0.7)
    _axes[2].legend(fontsize=16)
    _axes[2].grid(True, alpha=0.3)

    # --- Energy bands from E_c = -V(x) + offset, with E_F = 0 ---
    _EF_bd = 0.0
    _Ec_offset = Eg_Si / 2 + kT * np.log(_Na / ni_Si)  # (E_c - E_F) on P-side
    _Ec = _Ec_offset - _V
    _Ev = _Ec - Eg_Si
    _Ei = (_Ec + _Ev) / 2

    # (d) Energy band diagram
    _axes[3].plot(_x_um, _Ec, "b-", linewidth=2.5, label=r"$E_c$")
    _axes[3].plot(_x_um, _Ev, "b-", linewidth=2.5, label=r"$E_v$")
    _axes[3].plot(_x_um, _Ei, "g:", linewidth=1.5, label=r"$E_i$")
    _axes[3].axhline(_EF_bd, color="red", linestyle="--", linewidth=2, label=r"$E_F$")
    _axes[3].axvspan(-_xp_um, _xn_um, alpha=0.08, color="gray")
    _axes[3].axvline(0, color="black", linestyle=":", alpha=0.7)
    _axes[3].axvline(-_xp_um, color="red", linestyle="--", alpha=0.6)
    _axes[3].axvline(_xn_um, color="blue", linestyle="--", alpha=0.6)
    _Ec_P_far = _Ec[0]
    _Ec_N_far = _Ec[-1]
    _x_arrow = _x_um[-1] * 0.85
    _axes[3].annotate(
        "", xy=(_x_arrow, _Ec_N_far), xytext=(_x_arrow, _Ec_P_far),
        arrowprops=dict(arrowstyle="<->", color="purple", lw=2),
    )
    _axes[3].text(
        _x_arrow * 1.05, (_Ec_P_far + _Ec_N_far) / 2,
        rf"$q\phi_{{bi}}$ = {_phi_bi:.3f} eV",
        fontsize=16, color="purple", va="center",
    )
    _axes[3].set_ylabel("Energy (eV)", fontsize=16)
    _axes[3].set_xlabel("$x$ (\u00b5m)", fontsize=16)
    _axes[3].legend(fontsize=16, loc="lower left")
    _axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    # --- Computed values summary ---
    _info = mo.md(
        f"""
        **Computed Values at Thermal Equilibrium:**

        | Quantity | Symbol | Value |
        |:---------|:------:|------:|
        | Built-in potential | $\\phi_{{bi}}$ | {_phi_bi:.2f} V |
        | N-side depletion width | $x_n$ | {_xn * 1e4:.3f} \u00b5m |
        | P-side depletion width | $x_p$ | {_xp * 1e4:.3f} \u00b5m |
        | Total depletion width | $W_{{dep}}$ | {_W * 1e4:.3f} \u00b5m |
        | Maximum electric field | $\\lvert\\mathcal{{E}}_{{\\max}}\\rvert$ | {_E_max:.2e} V/cm |
        """
    )

    _controls = mo.hstack([Na_slider, Nd_slider], justify="center")

    mo.vstack([_controls, plt.gca(), _info])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary of Equations (Abrupt 1D Junction)

    | Quantity | Expression |
    |:---------|:-----------|
    | Built-in potential | $\displaystyle \phi_{bi} = \frac{kT}{q}\ln\!\left(\frac{N_A N_D}{n_i^2}\right)$ |
    | Charge neutrality | $N_A\, x_p = N_D\, x_n$ |
    | Charge density | $\rho = -qN_A$ (P-side), $\rho = +qN_D$ (N-side) |
    | Electric field (P-side) | $\displaystyle \mathcal{E}(x) = -\frac{qN_A}{\varepsilon_s}(x + x_p)$ |
    | Electric field (N-side) | $\displaystyle \mathcal{E}(x) = \frac{qN_D}{\varepsilon_s}(x - x_n)$ |
    | Max electric field | $\displaystyle \vert \mathcal{E}_{\max} \vert = \frac{qN_A x_p}{\varepsilon_s} = \frac{qN_D x_n}{\varepsilon_s}$ |
    | Potential (P-side) | $\displaystyle V(x) = \frac{qN_A}{2\varepsilon_s}(x + x_p)^2$ |
    | Potential (N-side) | $\displaystyle V(x) = \phi_{bi} - \frac{qN_D}{2\varepsilon_s}(x - x_n)^2$ |
    | N-side depletion width | $\displaystyle x_n = \sqrt{\frac{2\varepsilon_s}{q}\cdot\frac{N_A}{N_D(N_A + N_D)}\cdot\phi_{bi}}$ |
    | P-side depletion width | $\displaystyle x_p = \sqrt{\frac{2\varepsilon_s}{q}\cdot\frac{N_D}{N_A(N_A + N_D)}\cdot\phi_{bi}}$ |
    | Total depletion width | $\displaystyle W = \sqrt{\frac{2\varepsilon_s}{q}\cdot\frac{N_A + N_D}{N_A N_D}\cdot\phi_{bi}}$ |
    | One-sided ($N_A \gg N_D$) | $\displaystyle W \approx \sqrt{\frac{2\varepsilon_s\,\phi_{bi}}{qN_B}}$, where $N_B$ is the lighter doping |
    """)
    return


if __name__ == "__main__":
    app.run()
