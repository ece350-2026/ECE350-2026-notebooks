# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "matplotlib==3.10.8",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    q = 1.6e-19  # C
    # For local editing, use Path(__file__).parent
    # For WASM export, images are in the same directory as index.html
    import sys
    if "pyodide" in sys.modules:
        ASSET_DIR = Path(".")  # WASM: images in same directory
    else:
        ASSET_DIR = Path(__file__).parent  # Local: images in script directory

    mo.md(
        r"""
        # Carrier Transport: Drift

        Lecture 12

        Jan. 30, 2026

        This interactive notebook covers carrier drift in semiconductors, including:

        1. Drift velocity and mobility
        2. Scattering mechanisms
        3. Drift current and resistivity

        """
    )
    return ASSET_DIR, mo, np, plt, q


@app.cell
def _(ASSET_DIR, mo):
    _md1 = mo.md(r"""
    ## Carrier Drift and Mobility

    ### Physical Concept

    **Drift** is the net carrier motion caused by an applied electric field $\vec{\mathcal{E}}$.

    #### Microscopic Picture

    Even at equilibrium (no applied field), carriers are in constant random thermal motion with velocities ~$10^7$ cm/s at room temperature. Carriers scatter off:

    - Lattice vibrations (phonons)
    - Ionized impurities
    - Crystal defects

    But the net displacement averages to zero.

    The thermal velocity is given by:

    $$v_{th} = \sqrt{\frac{3k_BT}{m^*}},$$

    At room temperature, the thermal velocity is typically around $10^7$ cm/s for a free electron.
    """)
    _img = mo.vstack([mo.image(src=str(ASSET_DIR / "lecture14_15_carrier_transport_drift_img-1.png"), width="45%", caption="At zero field, carriers undergo random thermal motion with scattering events. The net displacement averages to zero.")], align="center")

    _md2 = mo.md(r"""
    When an electric field is applied ($\mathcal{E} \neq 0$):

    - The mean time between scattering events is $\tau_c \sim 0.1$ ps, and the mean free path is $\lambda = v_{th}\tau_c \sim 20$ nm.
    - Carriers experience a force $\vec{F} = \pm q\vec{\mathcal{E}}$ that accelerates them between collisions.
    - This creates a net **drift velocity** superimposed on the random thermal motion:

    $$v_{drift} \ll v_{th}$$

    - The drift velocity is lower than the thermal velocity (typically $v_{drift} \sim 10^4$–$10^6$ cm/s), but it produces a net current.""")
    _img2 = mo.vstack([mo.image(
        src=str(ASSET_DIR / "lecture14_15_carrier_transport_drift_img-2.png"),
        width="45%",
        caption="Schematic of electron motion for ℰ < 0"
    )], align="center")

    _md3 = mo.md(r"""
    At **low electric fields**, we approximate the drift velocity as being linearly proportional to the field:

    $$\boxed{\vec{v}_n = -\mu_n \vec{\mathcal{E}}} \quad \text{(electrons)}$$

    $$\boxed{\vec{v}_p = \mu_p \vec{\mathcal{E}}} \quad \text{(holes)}$$

    The constant of proportionality, $\mu$, is called the **mobility** (units: cm²/V·s):

    $$\mu_n = \frac{q\tau_{c,n}}{m_n^*} \quad \mu_p = \frac{q\tau_{c,p}}{m_p^*}$$

    - $\tau_c$ = mean free time between collisions (~0.1 ps)
    - $m^*$ = effective mass
    - $q$ = elementary charge

    The low-field approximation is valid when:

    1. The energy gained between collisions is small compared to thermal energy $k_BT$
    2. The scattering rate $1/\tau_c$ remains approximately constant

    **Side note:** If the device is shorter than the mean free path, then carriers are in the regime of ballistic transport, and the drift velocity is not limited by scattering. In modern devices with dimensions in the range of 10-100nm, ballistic transport is possible.
    """)
    mo.vstack([_md1, _img, _md2, _img2, _md3])
    return


@app.cell
def _(ASSET_DIR, mo):
    _md1 = mo.md(r"""
    ### Velocity Saturation at High Fields

    At high fields ($\mathcal{E} > 10^4$ V/cm in Si), due to increased scattering,
    the drift velocity **saturates**. The velocity has the form:

    $$ v = \frac{\mu \mathcal{E}}{1 + \frac{\mathcal{E}}{\mathcal{E}_{sat}}} $$

    where $\mathcal{E}_{sat}$ is the field when the carrier velocity is halved. The saturation velocity $v_{sat}$ is reached when $\mathcal{E} \gg \mathcal{E}_{sat}$. 
    """)

    _img = mo.vstack([mo.image(
        src=str(ASSET_DIR / "lecture14_15_carrier_transport_drift_img-16.png"),
        width="70%",
        caption="Drift velocity vs. electric field showing saturation at high fields"
    )], align="center")

    _md2 = mo.md(r"""

    """)

    mo.vstack([_md1, _img, _md2])
    return


@app.cell
def _(ASSET_DIR, mo):
    _md1 = mo.md(r"""
    ### Scattering Mechanisms

    Two main scattering mechanisms limit carrier mobility:

    1. **Phonon (Lattice) Scattering**:
       - Dominant at high temperatures
       - This is due to the scattering of carriers by lattice vibrations (phonons)
       - $\mu_{phonon} \propto T^{-3/2}$

    2. **Ionized Impurity  Scattering**:
       - Dominant at low temperatures and high doping
       - This is due to the scattering of carriers by ionized impurities
       - $\mu_{impurity} \propto \frac{T^{3/2}}{N_a + N_d}$

    The total mobility is given by:

    $$\frac{1}{\mu_{total}} = \frac{1}{\mu_{phonon}} + \frac{1}{\mu_{impurity}}$$
    """)
    _img = mo.vstack([mo.image(src=str(ASSET_DIR / "lecture14_15_carrier_transport_drift_img-6.png"), width="60%", caption="Electron mobility in silicon vs. temperature for various doping concentrations. At low T, impurity scattering dominates. At high T, lattice scattering dominates.")], align="center")
    mo.vstack([_md1, _img])
    return


@app.cell
def _(np, plt):
    def _():
        # Doping dependence at 300K according to Hu Eq. 2.2.8, 2.2.9
        fig, ax2 = plt.subplots(figsize=(10, 5))

        Impurity_range = np.logspace(14, 20, 100)
        mu_n300K = 1318 / (1 + (Impurity_range/1e17)**0.85) + 92
        mu_p300k = 420 / (1 + (Impurity_range/1.6e17)**0.7) + 50

        ax2.semilogx(Impurity_range, mu_n300K, 'b-', linewidth=2)
        ax2.semilogx(Impurity_range, mu_p300k, 'r-', linewidth=2)
        ax2.set_xlabel('Impurity Concentration (N$_A$ + N$_D$) (cm⁻³)', fontsize=16)
        ax2.set_ylabel('Mobility (cm²/V·s)', fontsize=16)
        ax2.set_title('Mobility vs. Impurity Concentration (T=300K) [Hu Eq. 2.2.8, 2.2.9]', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend([r'$\mu_n$', r'$\mu_p$'], fontsize=16)

        plt.tight_layout()
        return plt.gca()


    _()
    return


@app.cell
def _(ASSET_DIR, mo):
    _md1 = mo.md(r"""
    ### Drift Current

    The drift current density [A/cm²] is given by:

    $$J_{n,drift} = qnv_n = qn\mu_n\mathcal{E} \quad \text{(electrons)}$$

    $$J_{p,drift} = qpv_p = qp\mu_p\mathcal{E} \quad \text{(holes)}$$
    """)

    _img = mo.vstack([mo.image(
        src=str(ASSET_DIR / "current_density.png"),
        width="60%",
        caption="Current density: charge flowing through a cross-sectional area"
    )], align="center")

    _md2 = mo.md(r"""
    **Total drift current:**

    $$\boxed{J_{drift} = J_{n,drift} + J_{p,drift} = \sigma\mathcal{E} = (qn\mu_n + qp\mu_p)\mathcal{E}}$$

    where $\sigma$ is the **conductivity** (units: (Ω·cm)⁻¹):

    $$\boxed{\sigma = qn\mu_n + qp\mu_p}$$

    The **resistivity** is $\boxed{\rho = 1/\sigma}$ (units: Ω·cm). 

    The resistance is $\boxed{R = \rho \frac{L}{A}}$, where $L$ is the length and $A$ is the cross-sectional area.

    Because the mobility and carrier concentration are temperature-dependent, the resistance is also strongly temperature-dependent.
    """)

    mo.vstack([_md1, _img, _md2])
    return


@app.cell
def _(mo):
    temp_slider_rho = mo.ui.slider(150, 500, value=300, step=25, label="Temperature [K]", show_value=True)
    return (temp_slider_rho,)


@app.cell
def _(mo, np, plt, q, temp_slider_rho):
    # Plot resistivity vs doping with temperature-dependent mobility (Pierret Eq. 6.9)
    T_rho = temp_slider_rho.value
    N_range_res = np.logspace(14, 20, 200)

    # Temperature-dependent mobility parameters for electrons
    mu_nmin = 92 * (T_rho/300)**-0.57
    mu_n0 = 1268 * (T_rho/300)**-2.33
    N_nref = 1.3e17 * (T_rho/300)**2.4
    alpha_n = 0.91 * (T_rho/300)**-0.146

    # Temperature-dependent mobility parameters for holes
    mu_pmin = 54.3 * (T_rho/300)**-0.57
    mu_p0 = 406.9 * (T_rho/300)**-2.23
    N_pref = 235e17 * (T_rho/300)**2.4
    alpha_p = 0.88 * (T_rho/300)**-0.146

    # n-type: mobility depends on donor concentration
    mu_n_curve = mu_nmin + mu_n0 / (1 + (N_range_res/N_nref)**alpha_n)
    sigma_n = q * N_range_res * mu_n_curve / 1e4
    rho_n = 1 / sigma_n

    # p-type: mobility depends on acceptor concentration
    mu_p_curve = mu_pmin + mu_p0 / (1 + (N_range_res/N_pref)**alpha_p)
    sigma_p = q * N_range_res * mu_p_curve / 1e4
    rho_p = 1 / sigma_p

    fig_rho, ax_rho = plt.subplots(figsize=(10, 6))
    ax_rho.loglog(N_range_res, rho_n, 'b-', linewidth=2, label='n-type Si')
    ax_rho.loglog(N_range_res, rho_p, 'r-', linewidth=2, label='p-type Si')

    ax_rho.set_xlabel('Doping Concentration (cm⁻³)', fontsize=16)
    ax_rho.set_ylabel('Resistivity (Ω·cm)', fontsize=16)
    ax_rho.set_title(f'Resistivity vs. Doping Concentration (Si, T = {T_rho} K) (Pierret Eq. 6.9)', fontsize=16, fontweight='bold')
    ax_rho.legend(fontsize=16)
    ax_rho.grid(True, alpha=0.3, which='both')
    ax_rho.set_xlim([1e14, 1e20])
    ax_rho.set_ylim([1, 1e7])

    # Add annotations
    ax_rho.axhline(2300, color='gray', linestyle='--', alpha=0.5)
    ax_rho.text(1e15, 3000, 'Intrinsic Si (300K)', fontsize=16, color='gray')

    plt.tight_layout()

    mo.vstack([temp_slider_rho, plt.gca()])
    return


@app.cell
def _(ASSET_DIR, mo):
    _md1 = mo.md(r"""

    ## Hall Effect (Lab 3)

    This effect can be used for measuring carrier concentration and mobility in semiconductors.

    ### Setup

    Consider a semiconductor bar with:
    - Current $I_x$ flowing in the $x$-direction
    - Magnetic field $B_z$ applied in the $z$-direction
    - Hall voltage $V_H$ measured in the $y$-direction
    """)

    _img = mo.vstack([mo.image(
        src=str(ASSET_DIR / "Hall_effect.png"),
        width="60%",
        caption="Hall effect geometry for a p-type semiconductor"
    )], align="center")

    _md2 = mo.md(r"""
    ### How It Works

    1. **Lorentz Force**: Consider a positive charge moving with drift velocity $v_x$ in a magnetic field $B_z$. It experiences a Lorentz force:

    $$\vec{F} = q(\vec{v} \times \vec{B})$$

    $$F_y = qv_xB_z$$

    For a positive charge with a positive $v_x$ and positive $B_z$, the force is in the negative $y$-direction.

    2. **Charge Accumulation**: This force deflects charges toward $y=0$ end of the sample. This sets up an electric field in the $+\hat{y}$ direction. This field is called the **Hall field**, $\mathcal{E}_H$. The **Hall voltage** is:

    $$V_H = \mathcal{E}_H \cdot W$$

    3. **Steady State**: Equilibrium is reached when the electric force balances the magnetic force:

    $$F =0 = q(\mathcal{E}_H\hat{y} + \vec{v} \times \vec{B})  = q(\mathcal{E}_H - v_xB_z)\hat{y}$$

    $$\therefore \mathcal{E}_H = v_xB_z \implies \boxed{V_H = \mathcal{E}_H \cdot W = v_xB_zW}$$

    4. The drift velocity is given by:

    $$v_x = \frac{J_x}{qp} = \frac{I_x}{Wt}\frac{1}{qp}$$

    $$\therefore V_H = \frac{I_x B_z W}{qpWt} \implies \boxed{V_H = \frac{I_x B_z}{qpt}}$$

    Thus, the hole concentration can be deduced from the Hall voltage.

    For an N-type semiconductor, $V_H < 0$. So one can determine the carrier type from the sign of $V_H$.

    5. The Hall coefficient is:

    $$\boxed{R_H = \frac{V_H t}{I_x B_z} = \frac{1}{qp}}$$

    You can determine the carrier concentration from $R_H$ and the carrier type from the sign of $R_H$. For an N-type semiconductor, $R_H = -\frac{1}{qn} < 0$. 

    6. After the carrier concentration is determined, the Hall mobility and resistivity can be calculated. Since $\sigma = qp\mu_p$, the resistivity is $\boxed{\rho = \frac{1}{\sigma} = \frac{1}{qp\mu_p}}$.

    $$\therefore \boxed{\mu_H = |R_H| \sigma = \frac{1}{qp} \cdot qp\mu_p = \mu_p}$$
    """)

    mo.vstack([_md1, _img, _md2])
    return


if __name__ == "__main__":
    app.run()
