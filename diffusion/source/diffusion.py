# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "matplotlib==3.10.8",
#     "scipy==1.17.0",
# ]
# ///

import marimo

__generated_with = "0.19.7"
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
    import scipy.constants as const
    import sys

    q = 1.6e-19  # C
    # WASM-compatible ASSET_DIR
    if "pyodide" in sys.modules:
        ASSET_DIR = Path(".")  # WASM: images in same directory
    else:
        ASSET_DIR = Path(__file__).parent  # Local: images in script directory

    mo.md(
        r"""
        # Carrier Transport: Diffusion

        Lecture 13

        Feb. 2, 2026

        This interactive notebook covers carrier diffusion in semiconductors:

        1. Diffusion current
        2. Non-uniformly doped semiconductor and built-in electric field
        3. Einstein relation
        """
    )
    return ASSET_DIR, const, mo, np, plt


@app.cell
def _(ASSET_DIR, mo):
    _md = mo.md(r"""
        ## Carrier Diffusion

        ### Physical Concept

        **Diffusion** is carrier motion due to concentration gradients. Particles move from high to low concentration regions (increasing the entropy of the system).
        """)
    _img = mo.hstack([mo.image(src=str(ASSET_DIR / "diffusion.png"), width=500, caption="Diffusion of particles from high to low concentration regions. Hu, Fig. 2.9")], justify="center")
    mo.vstack([_md, _img])
    return


@app.cell
def _(ASSET_DIR, mo):
    _md = mo.md(r"""
        ### Diffusion Equation

        The diffusion current density is proportional to the **concentration gradient**, with $qD$ as the constant of proportionality:

        $$\boxed{J_{n,diff} = qD_n\frac{dn}{dx}} \quad \text{(electrons)}$$

        $$\boxed{J_{p,diff} = -qD_p\frac{dp}{dx}} \quad \text{(holes)}$$

        where $D$ is the **diffusion coefficient** (units: cm²/s).

        Even when $n$ and $p$ are low, it is possible to have a significant diffusion current (if the concentration gradient is large).

        **Notice the Sign!!:**
        - Electrons diffusing in the -x direction (dn/dx > 0) create current in the +x direction
        - Holes diffusing in the -x direction (dp/dx > 0) create current in the -x direction
        """)
    _img = mo.hstack([mo.image(src=str(ASSET_DIR / "diffusion_current_direction.png"), width=500, caption="Diffusion current direction. Hu, Fig. 2.10")], justify="center")
    mo.vstack([_md, _img])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Built-in Electric Field in a Non-Uniformly Doped Semiconductor

    Consider a **non-uniformly doped semiconductor** where the impurity concentration varies with position.

    Carriers **diffuse** from regions of high concentration to regions of low concentration.

    **BUT** as carriers (e.g., electrons) diffuse, they leave behind ionized  donors or acceptors (the nuclei and low energy electrons of the donors and acceptors are immobile)

    This leads to a charge separation, which causes an **electric field**. This field is called the **built-in field**

    The built-in field is in the opposite direction of the diffusion.

    At **thermal equilibrium**, the built-in electric field balances the diffusion. **Diffusion does not completely wash out the carrier concentration gradient!**
    """)
    return


@app.cell
def _(mo):
    # Animation slider for particle visualization
    time_slider = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        value=0,
        label="**Time evolution** (arbitrary units):"
    )
    return (time_slider,)


@app.cell
def _(mo, np, plt, time_slider):
    def make_plot():
        # Particle animation showing diffusion and immobilized ion cores
        np.random.seed(42)

        t = time_slider.value / 100.0  # Normalize to 0-1

        # Donor positions (fixed, distributed with gradient - more on left)
        n_particles = 30
        donor_x = np.concatenate([
            np.random.uniform(0.08, 0.4, 20),  # More donors on left
            np.random.uniform(0.4, 0.92, 10)   # Fewer on right
        ])
        donor_y = np.random.uniform(0.15, 0.85, n_particles)

        # Initial electron positions: offset from donor at 45 degrees (upper-right)
        offset = 0.025  # Small offset distance
        electron_x_initial = donor_x + offset * np.cos(np.pi/4)  # 45 degrees
        electron_y_initial = donor_y + offset * np.sin(np.pi/4)

        # Electrons diffuse to the right over time with random motion
        # Each electron has its own random trajectory
        np.random.seed(123)  # Different seed for diffusion randomness

        # Diffusion stops when built-in field reaches equilibrium (at t=75%)
        t_equilibrium = 0.75
        t_eff = min(t, t_equilibrium)  # Freeze positions after equilibrium

        # Random diffusion amounts (some electrons move more than others)
        random_factors = np.random.uniform(0.5, 1.5, n_particles)
        base_diffusion = t_eff * 0.35 * random_factors

        # Add random jitter that increases with time (simulating random walk)
        jitter_x = t_eff * 0.08 * np.random.randn(n_particles)
        jitter_y = t_eff * 0.15 * np.random.randn(n_particles)

        electron_x = electron_x_initial + base_diffusion + jitter_x
        electron_y = electron_y_initial + jitter_y
        # Electrons can leave the plot area (no clipping)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 5))

        # Draw semiconductor region (covers full plot area)
        ax.axvspan(-0.05, 1.05, alpha=0.1, color='gray')


        # Plot fixed donor ions (+ charges) - these don't move!
        ax.scatter(donor_x, donor_y, s=200, c='red', marker='+', linewidths=3, 
                   label='Ionized donors $N_D^+$', zorder=2)

        # Plot electrons (- charges) - these diffuse
        ax.scatter(electron_x, electron_y, s=150, c='blue', marker='o', alpha=0.8,
                   label='Electrons $n$', zorder=3)

        # Add electric field arrow if time > 0
        if t > 0.1:
            # Net positive charge on left, net negative on right
            # E-field points from + to - (left to right initially, but opposes diffusion)
            field_strength = min(t * 0.8, 0.6)
            ax.annotate('', xy=(0.25, 0.5), xytext=(0.25 + field_strength * 0.4, 0.5),
                        arrowprops=dict(arrowstyle='<-', color='black', lw=5))
            ax.text(0.25 + field_strength * 0.2, 0.58, r'$\vec{\mathcal{E}}_{built-in}$', 
                    fontsize=24, color='black', ha='center', fontweight='bold')

        # Labels
        ax.text(0.25, 0.02, 'High doping', ha='center', fontsize=14, style='italic')
        ax.text(0.75, 0.02, 'Low doping', ha='center', fontsize=14, style='italic')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Position x', fontsize=14)
        ax.set_ylabel('', fontsize=14)
        ax.set_title(f'Time = {int(t*100)}', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=12)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.tight_layout()
        return mo.vstack([
            mo.md("### Animation: Non-Uniformly Doped Semiconductor and Built-in Electric Field"),
            time_slider,
            mo.hstack([plt.gca()], justify="center"),
            mo.md(r"""
            - **Red (+)**: Ionized donor atoms $N_D^+$ (immobile) 
            - **Blue (−)**: Conduction band electrons (mobile, diffuse)
            - As electrons diffuse to the right, they leave behind uncompensated positive charge (ionized donors)
            - The charge separation creates the **built-in electric field** $\vec{\mathcal{E}}$
            - The field opposes further diffusion at thermal equilibrium
            """)
        ])


    make_plot()
    return


@app.cell
def _(mo):
    # Slider for doping gradient in band diagram
    gradient_slider = mo.ui.slider(
        start=-0.5,
        stop=0.5,
        step=0.1,
        value=0.5,
        label="**Doping gradient coefficient, g:**"
    )
    return (gradient_slider,)


@app.cell
def _(gradient_slider, mo, np, plt):
    # Band diagram showing built-in field from doping gradient
    g = gradient_slider.value
    g2 = 5

    # Position
    x = np.linspace(0, 1, 200)

    # Doping profile: exponential decay from left to right
    N_D0 = 1e16  # Doping at x=0
    N_D = N_D0 * np.exp(-g * g2 * x)  

    # Built-in potential from equilibrium condition (Fermi level flat)
    # V(x) = (kT/q) * ln(N_D(x) / N_D_ref), using x=1 as reference (V=0)
    kT_q = 0.026  # kT/q at 300K in Volts
    V_builtin = kT_q * np.log(N_D / N_D[-1])  # V(x) relative to V(x=1)=0

    # Energy bands bend with potential: E_c(x) = E_c,ref - qV(x)
    E_g = 1.12  # Silicon bandgap
    E_c_ref = 0.5  # E_c at reference point (x=1)
    E_c = E_c_ref - V_builtin  # Band bends down to the right (higher V = lower E_c)
    E_v = E_c - E_g

    # Intrinsic Fermi level: E_Fi = (E_c + E_v)/2 + (kT/2)*ln(N_v/N_c)
    # For Si at 300K: N_c = 2.8e19, N_v = 1.04e19
    N_c = 2.8e19
    N_v = 1.04e19
    E_Fi = (E_c + E_v) / 2 + (kT_q / 2) * np.log(N_v / N_c)  # Slightly below midgap

    # Fermi level is FLAT at equilibrium!
    # E_c - E_F = kT * ln(N_c / N_D) for n-type
    # Use reference point (x=1) to set E_F
    E_F = E_c[-1] - kT_q * np.log(N_c / N_D[-1])

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Plot 1: Doping profile
    ax1.semilogy(x, N_D, 'b-', linewidth=2)
    ax1.set_ylabel(r'$N_D$ (cm$^{-3}$)', fontsize=14)
    ax1.set_title(f'Doping Profile $N_D = 10^{{16}} \exp(-{g2} \cdot {g} \cdot x)$', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([N_D0*1e-2, N_D0*1e2])

    # Plot 2: Band diagram
    ax2.plot(x, E_c, 'b-', linewidth=2, label=r'$E_c$')
    ax2.plot(x, E_v, 'r-', linewidth=2, label=r'$E_v$')
    ax2.plot(x, E_Fi, 'g:', linewidth=1.5, label=r'$E_{Fi}$')
    ax2.axhline(E_F, color='k', linestyle='--', linewidth=2, label=r'$E_F$')
    ax2.fill_between(x, E_c, E_c + 0.3, alpha=0.1, color='blue')
    ax2.fill_between(x, E_v - 0.3, E_v, alpha=0.1, color='red')
    ax2.set_ylabel('Energy (eV)', fontsize=14)
    ax2.set_title('Energy Band Diagram at Equilibrium', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, E_F - 0.2, r'$E_F$ constant', 
             ha='center', fontsize=14, color='black', fontweight='bold')
    # Add equations
    ax2.text(0.02, E_v[0] + 0.05, 
             r'$E_c - E_F = kT \ln\left(\frac{N_c}{N_D}\right)$', 
             fontsize=12, color='black', ha='left', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 3: Electric field E = -dV/dx = (1/q) * dE_c/dx
    # Since E_c = E_c,ref - qV, we have dE_c/dx = -q*dV/dx, so E = (1/q)*dE_c/dx
    E_field = np.gradient(E_c, x[1] - x[0])  # E = (1/q) * dE_c/dx (in eV, so q=1)
    ax3.plot(x, E_field, 'g-', linewidth=2)
    ax3.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Position x (normalized)', fontsize=14)
    ax3.set_ylabel(r'$\mathcal{E}$ (V/unit length)', fontsize=14)
    ax3.set_title(r'Built-in Electric Field: $\mathcal{E} = -\frac{dV}{dx} = \frac{1}{q}\frac{dE_c}{dx}$', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-0.08, 0.08])

    plt.tight_layout()

    mo.vstack([
        mo.md("### Built-in Field from Doping Gradient"),
        gradient_slider,
        mo.hstack([plt.gca()], justify="center"),
        mo.md(r"""
        **Observations:**
        1. Electron density gradient means the energy bands are not flat, but the Fermi level is flat at equilibrium.
        2. Built-in potential: $V(x) = \frac{k_BT}{q} \ln\left(\frac{N_D(x)}{N_{D,ref}}\right)$
        3. Built-in field: $\mathcal{E} = -\frac{dV}{dx} = \frac{1}{q}\frac{dE_c}{dx}$
        4. At equilibrium: $J_{drift} + J_{diff} = 0$ (no net current!)
        """)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Einstein Relation


    The Einstein relation connects the diffusion coefficient $D$ and mobility $\mu$:

    $$\boxed{\frac{D_n}{\mu_n} = \frac{k_BT}{q}} \quad \text{and} \quad \boxed{\frac{D_p}{\mu_p} = \frac{k_BT}{q}}$$

    At room temperature (T = 300 K), $\frac{k_BT}{q} = 26 \text{ mV} = 0.026 \text{ V}$.


    ### Derivation

    Consider a non-uniformly doped semiconductor at thermal equilibrium:
    - Built-in electric field exists due to doping gradient
    - No net current flows: $J_{drift} + J_{diffusion} = 0$

    **Step 1:** Since the electron concentration is $n(x) = N_c \exp\left(\frac{E_F - E_c(x)}{k_BT}\right) = N_c \exp\left(\frac{qV(x)}{k_BT}\right)$, the gradient is

    $$\frac{dn}{dx} = N_c \exp\left(\frac{qV(x)}{k_BT}\right) \cdot \frac{q}{k_BT} \cdot \frac{dV}{dx}  = \frac{qn(x)}{k_BT} \cdot \frac{dV}{dx}$$

    **Step 2:** Apply equilibrium condition $J_{drift} + J_{diff} = 0$:

    $$qn \mu_n \mathcal{E} + qD_n\frac{dn}{dx} = 0$$

    **Step 3:** Substitute $\mathcal{E} = -\frac{dV}{dx}$ and the expression for $\frac{dn}{dx}$:

    $$-qn\mu_n\frac{dV}{dx} + qD_n
    \left(\frac{qn}{k_BT}\frac{dV}{dx}\right) = 0$$

    **Step 4:** Simplifying the equation, we get the Einstein relation:

    $$\boxed{\frac{D}{\mu} = \frac{kT}{q}}$$
    """)
    return


@app.cell
def _(mo):
    # Einstein relation calculator - create sliders
    mobility_input = mo.ui.slider(100, 2000, value=1400, step=50, label="Mobility μ (cm²/V·s)")
    temp_input = mo.ui.slider(200, 400, value=300, step=10, label="Temperature (K)")
    return mobility_input, temp_input


@app.cell
def _(const, mo, mobility_input, temp_input):
    mu_einstein = mobility_input.value
    T_einstein = temp_input.value

    thermal_voltage = const.k * T_einstein / const.e
    D_einstein = thermal_voltage * mu_einstein

    mo.vstack([
        mobility_input,
        temp_input,
        mo.md(
            f"""
            **Results:**
            - Mobility: μ = {mu_einstein} cm²/V·s
            - Temperature: T = {T_einstein} K
            - kT/q = {thermal_voltage*1000:.1f} mV
            - Diffusion coefficient: D = {D_einstein:.2f} cm²/s

            **Verification:** D/μ = {D_einstein/mu_einstein:.4f} V = {thermal_voltage:.4f} V ✓
            """
        )
    ])
    return


if __name__ == "__main__":
    app.run()
