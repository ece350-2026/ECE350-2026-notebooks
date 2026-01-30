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
async def _():
    # Install plotly in WASM environment (Pyodide)
    import sys
    if "pyodide" in sys.modules:
        import micropip
        await micropip.install("plotly")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    q = 1.6e-19  # C
    ASSET_DIR = Path(__file__).parent

    mo.md(
        r"""
        # Band Bending and Electrostatics

        This interactive notebook covers the concept spatial energy band diagrams in semiconductors, including:

        1. Poisson's equation and electrostatics
        2. Relationship to energy bands
        3. Band bending 

        """
    )
    return ASSET_DIR, mo, np, plt


@app.cell
def _(ASSET_DIR, mo):
    _md1 = mo.md(r"""
    ## Energy Bands and the Electric Field

    - The parabolic band approximation gives: 

    $$E  = E_c + \frac{\hbar^2 k^2}{2m_n^*} \quad \text{for electrons}$$

    $$E = E_v - \frac{\hbar^2 k^2}{2m_p^*} \quad \text{for holes}$$

    - So we can view $E_c$ and $E_v$ as the potential energy of the electrons and holes, respectively, and $\frac{\hbar^2 k^2}{2m_n^*}$ and $\frac{\hbar^2 k^2}{2m_p^*}$ as the kinetic energy. 
        - At $E_c$ and $E_v$, the group velocity for the carriers is zero, so one can think of the carriers as having zero (net) kinetic energy.
        - **Electrons** tend to fall in the energy diagram (toward lower $E_c$)
        - **Holes** "bubble" up (toward higher $E_v$)

    - Notice that the energy is defined in a positive sense for **electrons** (negative charge) 
        - For electrons, $E_c$ is the minimum energy and more energetic electrons have higher energies. 
            - Consider the lowest energy to be $E_c - E_{ref}$, where $E_{ref}$ is the reference energy. 
        - For holes (positive charge), $E_v$ is the minimum energy and more energetic holes have lower energies. 
            - Consider the lowest energy to be $E_{ref} - E_v$, where $E_{ref}$ is the reference energy. 

    """)

    _img1 = mo.hstack([mo.image(src=str(ASSET_DIR / "electron-hole-energy.png"), width=300, caption="Electrons fall toward E<sub>c</sub>, holes bubble up toward  E<sub>v</sub>.")], justify="center")

    _md2 = mo.md(r"""
    - If there is a voltage (potential difference) between two points in the semiconductor, the potential energy for the carriers is no longer spatially uniform. This means $E_c(x)$ and $E_v(x)$ vary with position $x$.
    - This is called **band bending**.
    """)

    mo.vstack([_md1, _img1, _md2])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Electrostatics Review

    The chain of relationships: **Charge → Electric Field → Potential**

    ### 1. Charge → Electric Field (Poisson's Equation)

    $$\boxed{\nabla \cdot \vec{\mathcal{E}} = \frac{\rho}{\epsilon_s}}$$

    - $\rho$ is the charge density
    - $\epsilon_s$ is the permittivity of the material
        - $\epsilon_s = \epsilon_0 \epsilon_r$
            - $\epsilon_0 = 8.85 \times 10^{-12} F/m$ is the permittivity of free space
            - $\epsilon_r$ is the relative permittivity of the material ($\approx 12$ for silicon)

    In a semiconductor, the charge density is
    $$\rho = q(p - n + N_D^+ - N_A^-)$$

    ### 2. Electric Field → Potential

    For a **positive charge**, the electric field is related to the electrostatic potential, $\Phi$, and the voltage, $V$, by:

    $$\boxed{\vec{\mathcal{E}} = -\frac{1}{q} \nabla \Phi = -\nabla V} \quad \text{in 1D: } \mathcal{E} = -\frac{dV}{dx}$$

    $\boxed{\Phi = qV}$ is the potential energy of a positive charge.

    Combining:

    $$\boxed{\nabla^2 V = -\frac{\rho}{\epsilon_s}} \quad \text{in 1D: } \frac{d^2V}{dx^2} = -\frac{\rho}{\epsilon_s}$$
    """)
    return


@app.cell
def _(ASSET_DIR, mo):
    _md = mo.md(r"""
    ## Connection to Energy Bands 

    Since $E_c(x)$ and $E_v(x)$ represent the potential energy of the electrons and holes, respectively, when we relate to the electrostatic potential, we get:

    $$E_{ref} - E_v(x) = \Phi(x) = qV(x)  \quad \text{for holes}$$

    $$E_c(x) - E_{ref} = -\Phi(x) =-qV(x)  \quad \text{for electrons}$$

    $$\therefore \boxed{\mathcal{E} = -\frac{dV}{dx} = \frac{1}{q} \frac{dE_v}{dx} = \frac{1}{q} \frac{dE_c}{dx} }$$

    - The bandgap $E_g = E_c - E_v$ remains constant

    """)

    _img = mo.hstack([mo.image(src=str(ASSET_DIR / "bandbend_roll.png"), width=300, caption="Energy bands under an applied electric field. Electrons roll down the hill, holes roll up the hill.")], justify="center")

    mo.vstack([_md, _img])
    return


@app.cell
def _(mo):
    # Slider for applied voltage (created in separate cell)
    mo.md("## Band Diagram Visualization")

    voltage_slider = mo.ui.slider(
        start=-1.0,
        stop=1.0,
        step=0.05,
        value=0.7,
        label="**Applied Voltage** $V_a$ (V) at $x = 0$:"
    )
    return (voltage_slider,)


@app.cell
def _(ASSET_DIR, mo, np, plt, voltage_slider):
    # Parameters
    E_g = 1.0  # Bandgap in eV
    L_mm = 1.0  # Device length in mm
    L = L_mm * 1e-3  # Convert to meters for calculations
    V_a = voltage_slider.value  # Applied voltage from slider

    # Position array (in mm for display)
    x_mm = np.linspace(0, L_mm, 200)
    x = x_mm * 1e-3  # in meters

    # Linear voltage drop: V(0) = V_a, V(L) = 0
    V = V_a * (1 - x_mm / L_mm)

    # Electric field: E = -dV/dx (in V/m, then convert to V/mm for display)
    E_field_V_per_m = -np.gradient(V, x[1] - x[0])
    E_field_V_per_mm = E_field_V_per_m * 1e-3  # V/mm

    # Energy bands: E_c(x) = E_c,ref - V(x), E_v(x) = E_c(x) - E_g
    E_c_ref = E_g / 2  # Reference so bands centered at x=L
    E_c = E_c_ref - V
    E_v = E_c - E_g

    # Intrinsic Fermi level at mid-gap
    E_Fi = (E_c + E_v) / 2

    # Fixed axis limits based on slider range (-1V to 1V)
    V_min, V_max = -1.0, 1.0  # Slider range
    E_c_max = E_c_ref - V_min  # E_c when V is at minimum
    E_c_min = E_c_ref - V_max  # E_c when V is at maximum
    E_v_max = E_c_max - E_g
    E_v_min = E_c_min - E_g
    E_field_max = V_max / L_mm  # Max E-field magnitude

    # Create figure with 3 subplots
    fig, (ax_V, ax_band, ax_field) = plt.subplots(3, 1, figsize=(10, 8))

    # Plot 1: Electrostatic Potential V(x)
    ax_V.plot(x_mm, V, 'k-', linewidth=2)
    ax_V.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_V.set_ylabel('V (V)', fontsize=16)
    ax_V.set_title(f'Voltage:  $V(0) = V_a = {V_a:.2f}$ V,  $V(L) = 0$', fontsize=16, fontweight='bold')
    ax_V.grid(True, alpha=0.3)
    ax_V.set_xlim([0, L_mm])
    ax_V.set_ylim([V_min - 0.1, V_max + 0.1])  # Fixed y-axis
    ax_V.set_xticklabels([])
    ax_V.tick_params(axis='both', labelsize=14)
    # Add equation
    ax_V.text(0.98, 0.95, r'$V(x) = V_a \left(1 - \frac{x}{L}\right)$', transform=ax_V.transAxes, 
              fontsize=16, verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Energy Bands
    ax_band.plot(x_mm, E_c, 'b-', linewidth=2, label='$E_c$')
    ax_band.plot(x_mm, E_v, 'r-', linewidth=2, label='$E_v$')
    ax_band.plot(x_mm, E_Fi, 'k:', linewidth=1.5, label='$E_{Fi}$')
    ax_band.fill_between(x_mm, E_c, E_c_max + 0.2, alpha=0.1, color='blue')
    ax_band.fill_between(x_mm, E_v, E_v_min - 0.2, alpha=0.1, color='red')
    ax_band.set_ylabel('Energy (eV)', fontsize=16)
    ax_band.set_title(f'Energy Band Diagram  ($E_g = {E_g}$ eV)', fontsize=16, fontweight='bold')
    ax_band.legend(loc='best', fontsize=14)
    ax_band.grid(True, alpha=0.3)
    ax_band.set_xlim([0, L_mm])
    ax_band.set_ylim([E_v_min - 0.3, E_c_max + 0.3])  # Fixed y-axis
    ax_band.set_xticklabels([])
    ax_band.tick_params(axis='both', labelsize=14)
    # Add equation
    ax_band.text(0.98, 0.05, 
                 r'$E_c(x) = - qV(x) + \text{constant}$' + '\n' + r'$E_v(x) = E_c(x) - E_g$', 
                 transform=ax_band.transAxes, fontsize=16, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Electric Field
    ax_field.plot(x_mm, E_field_V_per_mm, 'purple', linewidth=2)
    ax_field.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_field.set_xlabel(r'Position $x$ (mm)', fontsize=16)
    ax_field.set_ylabel(r'$\mathcal{E}$ (V/mm)', fontsize=16)
    E_field_avg = V_a / L_mm if L_mm != 0 else 0
    ax_field.set_title(f'Electric Field:  $\\mathcal{{E}} = {E_field_avg:.2f}$ V/mm', fontsize=16, fontweight='bold')
    ax_field.grid(True, alpha=0.3)
    ax_field.set_xlim([0, L_mm])
    ax_field.set_ylim([-E_field_max - 0.1, E_field_max + 0.1])  # Fixed y-axis
    ax_field.tick_params(axis='both', labelsize=14)
    # Add equation
    ax_field.text(0.98, 0.95, r'$\mathcal{E} = -\frac{dV}{dx} = \frac{V_a}{L}$', transform=ax_field.transAxes, 
                  fontsize=16, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Display device length and plot
    mo.vstack([
        mo.md("## Band Diagram Visualization"),
        mo.hstack([mo.image(src=str(ASSET_DIR / "voltage_bandbend.png"), width=300)], justify="center"),
        voltage_slider,
        mo.md(f"**Device length:** $L = {L_mm}$ mm  |  **Bandgap:** $E_g = {E_g}$ eV"),
        mo.hstack([plt.gca()], justify="center"),
        mo.md(r'$$\because \frac{d\mathcal{E}}{dx} = \frac{\rho}{\epsilon_s}\text{, } \rho = 0$$')
    ])
    return


if __name__ == "__main__":
    app.run()
