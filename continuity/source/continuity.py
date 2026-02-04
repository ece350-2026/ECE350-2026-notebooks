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
    import sys as _sys
    if "pyodide" in _sys.modules:
        import micropip
        _ = await micropip.install("plotly")
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
        # Carrier Transport: Continuity Equation 

        This interactive notebook covers:

        1. Continuity equation 
        2. Diffusion equation and diffusion length

        Reference: Hu, Ch. 4.7

        """
    )
    return ASSET_DIR, mo, np, plt


@app.cell
def _(ASSET_DIR, mo):
    _md1 = mo.md(r"""
    ## Continuity of Carriers
    """)

    _img = mo.hstack([mo.image(src=str(ASSET_DIR / "continuity.png"), width="80%")], justify="center")

    _md2 = mo.md(r"""
    Consider a volume of semiconductor of length $\Delta x$ and cross-sectional area $A$. Let the total number of holes in the volume be $P = pA\Delta x$. 

    The time rate of change of $P$ is

    $$
    \frac{\partial P}{\partial t} = \frac{1}{q} J_p(x)A - \frac{1}{q} J_p(x + \Delta x)A + (G_p - R_p)A \Delta x
    $$

    $$\therefore \boxed{\frac{\partial p}{\partial t} = -\frac{1}{q} \frac{\partial J_p}{\partial x} + (G_p - R_p)}$$

    Similarly, for electrons:

    $$\boxed{\frac{\partial n}{\partial t} = \frac{1}{q}\frac{\partial J_n}{\partial x} + (G_n - R_n)}$$

    Since $J_n = q n \mu_n \mathcal{E} + q D_n \frac{\partial n}{\partial x}$ and $J_p = q p \mu_p \mathcal{E} - q D_p \frac{\partial p}{\partial x}$, we can substitute these into the continuity equations to get:

    $$\boxed{\frac{\partial n}{\partial t} = n \mu_n \frac{\partial \mathcal{E}}{\partial x} + \mu_n \mathcal{E} \frac{\partial n}{\partial x} + D_n \frac{\partial^2 n}{\partial x^2} + G_n - R_n}$$

    $$\boxed{\frac{\partial p}{\partial t} = -p \mu_p \frac{\partial \mathcal{E}}{\partial x} - \mu_p \mathcal{E} \frac{\partial p}{\partial x} + D_p \frac{\partial^2 p}{\partial x^2} + G_p - R_p}$$

    In addition, we have Poisson's equation:

    $$\boxed{\frac{\partial \mathcal{E}}{\partial x} =  \frac{\rho}{\epsilon_s}}$$



    """)

    mo.vstack([_md1, _img, _md2])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Diffusion Length

    Consider the steady-state solution of the continuity equation with $\mathcal{E} = 0$ aand the recombination rate of $R_n = n/ \tau_n$:

    $$\frac{\partial n}{\partial t} = 0 = D_n \frac{\partial^2 n}{\partial x^2} - \frac{n}{\tau_n}$$

    $$\boxed{\frac{\partial^2 n}{\partial x^2} = \frac{n}{L_n^2} \quad \text{where} \quad L_n = \sqrt{D_n \tau_n}}$$

    Similarly, for holes:

    $$\frac{\partial p}{\partial t} = 0 = D_p \frac{\partial^2 p}{\partial x^2} - \frac{p}{\tau_p}$$

    $$\boxed{\frac{\partial^2 p}{\partial x^2} = \frac{p}{L_p^2} \quad \text{where} \quad L_p = \sqrt{D_p \tau_p}}$$

    $L_n$, $L_p$ are the **diffusion lengths** of the electrons and holes.

    Intuitively, $L_n$, $L_p$ are the average distances a minority carrier diffuses before recombining.
    """)
    return


@app.cell
def _(mo):
    # Sliders for diffusion length calculation
    D_param_slider = mo.ui.slider(5, 50, value=12, step=1, label="Diffusion coefficient D_p (cm²/s)")
    tau_p_slider = mo.ui.slider(0.1, 100, value=10, step=0.5, label="Lifetime τₚ (μs)")
    return D_param_slider, tau_p_slider


@app.cell
def _(D_param_slider, mo, np, tau_p_slider):
    D_param = D_param_slider.value
    tau_p_us = tau_p_slider.value

    L_p_cm = np.sqrt(D_param * tau_p_us * 1e-6)
    L_p_um = L_p_cm * 1e4

    _result = mo.md(
        f"""
        **Example Calculation:**

        With D_p = {D_param:.1f} cm²/s and τ_p = {tau_p_us:.1f} μs:

        $$L_p = \\sqrt{{D_p \\tau_p}} = \\sqrt{{{D_param:.1f} \\times {tau_p_us*1e-6:.1e}}} \\text{{cm}} = {L_p_um:.1f} \\text{{ μm}}$$

        """
    )

    mo.vstack([
        mo.hstack([D_param_slider, tau_p_slider], justify="start"),
        _result
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Steady-State Carrier Diffusion

    We can break down the total carrier density into the equilibrium carrier density and the excess carrier density:

    $$p = p_0 + p'$$

    $$n = n_0 + n'$$

    where:
    - $p_0$ = equilibrium hole concentration
    - $n_0$ = equilibrium electron concentration
    - $p'$ = excess hole concentration (perturbation)
    - $n'$ = excess electron concentration (perturbation)

    Consider the case of steady-state carrier diffusion in a homogeneous semiconductor with no generation ($G = 0$):

    $$\frac{d^2 (p_0 + p')}{dx^2} - \frac{(p_0 + p')}{L_p^2} = 0$$

    $$\implies \frac{d^2 p'}{dx^2} = \frac{p_0 + p'}{L_p^2}$$

    **General solution:** $p'(x) = A e^{-x/L_p} + B e^{x/L_p}$

    The excess carrier densities $p'(x)$ and $n'(x)$ are usually more important for the minority carriers, because their equilibrium concentrations are very low to begin with.
    """)
    return


@app.cell
def _(ASSET_DIR, mo):
    # Build markdown block and image separately
    _md1 = mo.md(r"""
    ## Example: Steady-State Carrier Diffusion in a Homogeneous Semiconductor

    Consider **boundary conditions:**
    - $p'(x = 0) = \Delta p_0$ (constant excess hole injection)
    - $p'(x = W) = 0$ (all excess holes extracted)
    """)

    _img = mo.image(src=str(ASSET_DIR / "shortbase.png"), width="80%")

    _md2 = mo.md(r"""
    Assume **solution**: $p'(x) = A \sinh(x/L_p) + B \cosh(x/L_p)$

    At $x = 0$: $p'(0) = B = \Delta p_0$

    At $x = W$: $p'(W) = A \sinh(W/L_p) + \Delta p_0 \cosh(W/L_p) = 0$

    $$\implies A = -\Delta p_0 \frac{\cosh(W/L_p)}{\sinh(W/L_p)} = -\Delta p_0 \coth(W/L_p)$$

    **Solution:**

    $$p'(x) = \Delta p_0 \left[ -\coth(W/L_p) \sinh(x/L_p) + \cosh(x/L_p) \right]$$

    Using the identity $\cosh(A)\sinh(B) - \sinh(A)\cosh(B) = -\sinh(A-B)$:

    $$\boxed{p'(x) = \Delta p_0 \frac{\sinh((W-x)/L_p)}{\sinh(W/L_p)}}$$

    **Current density:**

    $$J_p = -q D_p \frac{dp'}{dx} = -q D_p \cdot \Delta p_0 \cdot \frac{-1}{L_p} \cdot \frac{\cosh((W-x)/L_p)}{\sinh(W/L_p)}$$

    $$\boxed{J_p(x) = \frac{q D_p \Delta p_0}{L_p} \cdot \frac{\cosh((W-x)/L_p)}{\sinh(W/L_p)}}$$

    At $x = 0$: $J_p(0) = \frac{q D_p \Delta p_0}{L_p} \coth(W/L_p)$
    """)

    # Display as vertical stack
    mo.vstack([_md1, _img, _md2])
    return


@app.cell
def _(mo):
    # Sliders for W/Lp ratio visualization
    W_Lp_slider = mo.ui.slider(0.5, 5.0, value=2.0, step=0.1, label="W/Lₚ ratio")
    return (W_Lp_slider,)


@app.cell
def _(W_Lp_slider, mo, np, plt):
    W_Lp = W_Lp_slider.value

    # Normalized coordinates
    x_norm = np.linspace(0, 1, 200)  # x/W from 0 to 1

    # p'(x)/Δp₀ = sinh((W-x)/Lp) / sinh(W/Lp) = sinh(W/Lp * (1 - x/W)) / sinh(W/Lp)
    p_norm = np.sinh(W_Lp * (1 - x_norm)) / np.sinh(W_Lp)

    # J normalized: proportional to cosh((W-x)/Lp) / sinh(W/Lp)
    J_norm = np.cosh(W_Lp * (1 - x_norm)) / np.sinh(W_Lp)

    # Create figure
    _fig_sol, (_ax_p, _ax_j) = plt.subplots(1, 2, figsize=(14, 8))

    # Plot carrier concentration
    _ax_p.fill_between(x_norm, 0, p_norm, alpha=0.3, color='blue')
    _ax_p.plot(x_norm, p_norm, 'b-', linewidth=2.5)
    _ax_p.axhline(0, color='gray', linestyle='--', alpha=0.5)
    _ax_p.set_xlabel('Position x/W', fontsize=18)
    _ax_p.set_ylabel(r"$p'(x) / \Delta p_0$", fontsize=18)
    _ax_p.set_title(f"Excess Carrier Profile (W/Lₚ = {W_Lp:.1f})", fontsize=18, fontweight='bold')
    _ax_p.set_xlim([0, 1])
    _ax_p.set_ylim([0, 1.1])
    _ax_p.grid(True, alpha=0.3)
    _ax_p.tick_params(axis='both', labelsize=18)

    # Mark boundary conditions
    _ax_p.plot(0, 1, 'ro', markersize=10, label=r"$p'(0) = \Delta p_0$")
    _ax_p.plot(1, 0, 'go', markersize=10, label=r"$p'(W) = 0$")
    _ax_p.legend(fontsize=18, loc='upper right')

    # Plot current density
    _ax_j.plot(x_norm, J_norm, 'r-', linewidth=2.5)
    _ax_j.fill_between(x_norm, 0, J_norm, alpha=0.3, color='red')
    _ax_j.set_xlabel('Position x/W', fontsize=18)
    _ax_j.set_ylabel(r"$J_p(x) \cdot L_p / (q D_p \Delta p_0)$", fontsize=18)
    _ax_j.set_title(f"Current Density Profile (W/Lₚ = {W_Lp:.1f})", fontsize=18, fontweight='bold')
    _ax_j.set_xlim([0, 1])
    _ax_j.set_ylim(bottom=0)  # Set the minimum y limit to zero
    _ax_j.grid(True, alpha=0.3)
    _ax_j.tick_params(axis='both', labelsize=18)


    plt.tight_layout()

    mo.vstack([
        mo.md(f"### Interactive: Carrier Profile with Injection at $x = 0$"),
        W_Lp_slider,
        plt.gca(),
        mo.md(r"""
        **Observations:**

        1. As $W/L_p \to \infty$, the p'(x) approaches a decaying exponential and the current density approaches zero at $W$ (infinitely long device)
        2. As $W/L_p \to 0$, the p'(x) approaches a straight line and the current is nearly constant and is high (short device)
        3. Side note: We will see this again in the bipolar junction transistor (BJT). This carrier profile is in the base region of the BJT.
        """)
    ])
    return


@app.cell
def _(mo):
    # Sliders for animation parameters (no time slider - we animate from 0 to 10 ns)
    mu_anim_slider = mo.ui.slider(800, 2000, value=1400, step=100, label="Mobility μ (cm²/V·s)")
    T_anim_slider = mo.ui.slider(200, 500, value=300, step=10, label="Temperature T (K)")
    E_anim_slider = mo.ui.slider(-30, 30, value=10, step=2, label="Electric field ℰ (V/cm)")
    tau_anim_slider = mo.ui.slider(1, 100, value=20, step=5, label="Recombination lifetime τ (ns)")
    return E_anim_slider, T_anim_slider, mu_anim_slider, tau_anim_slider


@app.cell
def _(E_anim_slider, T_anim_slider, mo, mu_anim_slider, np, tau_anim_slider):
    from scipy.integrate import solve_ivp
    import plotly.graph_objects as go

    # Physical parameters from sliders
    mu_anim = mu_anim_slider.value  # cm²/V·s
    T_anim = T_anim_slider.value  # K
    E_field_anim = E_anim_slider.value  # V/cm
    tau_ns = tau_anim_slider.value  # ns
    tau_s = tau_ns * 1e-9  # s

    # Animation time range: 0 to 10 ns
    t_max_ns = 10.0
    n_frames = 101  # More frames for smoother animation
    time_points_ns = np.linspace(0, t_max_ns, n_frames)

    # Constants
    k_B_anim = 1.38e-23  # J/K
    q_anim = 1.6e-19  # C

    # Einstein relation: D = (kT/q) * μ
    kT_q_anim = k_B_anim * T_anim / q_anim  # V
    D_anim = kT_q_anim * mu_anim  # cm²/s

    # Drift velocity (electrons move opposite to E-field)
    v_drift_anim = -mu_anim * E_field_anim  # cm/s

    # Diffusion length
    L_diff = np.sqrt(D_anim * tau_s) * 1e4  # μm

    # Spatial grid (in μm for display, cm for calculation)
    N_points = 600  # More points for smoother pulse
    x_min_um, x_max_um = -20, 20
    x_um = np.linspace(x_min_um, x_max_um, N_points)
    dx_um = x_um[1] - x_um[0]
    dx_cm = dx_um * 1e-4  # cm

    # Initial condition: Gaussian pulse centered at x=0
    sigma_0_um = 0.5  # μm initial width
    sigma_0_cm = sigma_0_um * 1e-4  # cm
    n_0 = 1e10  # cm⁻³ peak concentration (excess minority carriers)
    x_cm = x_um * 1e-4

    n_initial = n_0 * np.exp(-x_cm**2 / (2 * sigma_0_cm**2))

    # Define the RHS of the continuity equation using finite differences
    # ∂n/∂t = D ∂²n/∂x² - v_d ∂n/∂x - n/τ
    def continuity_rhs(t, n):
        dndt = np.zeros_like(n)

        # Interior points (vectorized for speed)
        diffusion = D_anim * (n[2:] - 2*n[1:-1] + n[:-2]) / dx_cm**2

        if v_drift_anim >= 0:
            drift = v_drift_anim * (n[1:-1] - n[:-2]) / dx_cm
        else:
            drift = v_drift_anim * (n[2:] - n[1:-1]) / dx_cm

        recombination = n[1:-1] / tau_s

        dndt[1:-1] = diffusion - drift - recombination

        # Outflow boundary conditions (zero-gradient for smooth exit)
        # Use one-sided differences at boundaries
        # Left boundary (i=0): forward difference for second derivative
        diff_left = D_anim * (n[2] - 2*n[1] + n[0]) / dx_cm**2
        drift_left = v_drift_anim * (n[1] - n[0]) / dx_cm if v_drift_anim >= 0 else v_drift_anim * (n[1] - n[0]) / dx_cm
        dndt[0] = diff_left - drift_left - n[0] / tau_s

        # Right boundary (i=-1): backward difference for second derivative
        diff_right = D_anim * (n[-1] - 2*n[-2] + n[-3]) / dx_cm**2
        drift_right = v_drift_anim * (n[-1] - n[-2]) / dx_cm if v_drift_anim >= 0 else v_drift_anim * (n[-1] - n[-2]) / dx_cm
        dndt[-1] = diff_right - drift_right - n[-1] / tau_s

        return dndt

    # Solve the PDE for all time points
    t_final_s = t_max_ns * 1e-9
    t_eval_s = time_points_ns * 1e-9
    sol = solve_ivp(continuity_rhs, [0, t_final_s], n_initial,
                   method='RK45', t_eval=t_eval_s, max_step=t_final_s/400)

    # Store all solutions
    all_solutions = sol.y  # Shape: (N_points, n_frames)

    # Calculate statistics for each time point
    total_initial = np.trapz(n_initial, x_cm)

    def calc_stats(n_profile):
        total = np.trapz(n_profile, x_cm)
        if total > 1e-10 * total_initial:
            x_center = np.trapz(x_cm * n_profile, x_cm) / total
            variance = np.trapz((x_cm - x_center)**2 * n_profile, x_cm) / total
            sigma = np.sqrt(max(variance, 0))
            return x_center * 1e4, sigma * 1e4  # Convert to μm
        return 0, sigma_0_um

    # Create Plotly animation (WASM compatible)
    # Build frames for animation - each frame includes all traces
    frames = []
    for i in range(n_frames):
        n_current = all_solutions[:, i]
        t_ns = time_points_ns[i]
        x_center_um, sigma_um = calc_stats(n_current)
        n_peak = np.max(n_current)

        frame = go.Frame(
            data=[
                # Present pulse (blue, animated)
                go.Scatter(x=x_um, y=n_current, mode='lines', fill='tozeroy',
                          fillcolor='rgba(0, 0, 255, 0.3)', line=dict(color='blue', width=2),
                          showlegend=False),
                # Vertical line at centroid position
                go.Scatter(x=[x_center_um, x_center_um], y=[0, n_0 * 1.1], mode='lines',
                          line=dict(color='darkblue', width=2, dash='dot'),
                          showlegend=False),
            ],
            name=str(i),
            layout=go.Layout(
                annotations=[
                    dict(x=0.98, y=0.98, xref='paper', yref='paper', showarrow=False,
                         text=f"t = {t_ns:.1f} ns<br>Peak: {n_peak:.2e} cm⁻³<br>Center: {x_center_um:.2f} μm<br>Width σ: {sigma_um:.2f} μm",
                         font=dict(size=14), align='right',
                         bgcolor='rgba(224, 255, 255, 0.9)', bordercolor='gray', borderwidth=1),
                    dict(x=0.02, y=0.98, xref='paper', yref='paper', showarrow=False,
                         text=f"D<sub>n</sub> = {D_anim:.1f} cm²/s<br>v<sub>d</sub> = {v_drift_anim:.1e} cm/s",
                         font=dict(size=14), align='left',
                         bgcolor='rgba(255, 255, 255, 0.9)', bordercolor='gray', borderwidth=1),
                ]
            )
        )
        frames.append(frame)

    # Initial data
    x_center_0, sigma_0 = calc_stats(n_initial)

    fig = go.Figure(
        data=[
            # Present pulse (blue, animated)
            go.Scatter(x=x_um, y=n_initial, mode='lines', fill='tozeroy',
                      fillcolor='rgba(0, 0, 255, 0.3)', line=dict(color='blue', width=2),
                      showlegend=False),
            # Vertical line at centroid position
            go.Scatter(x=[x_center_0, x_center_0], y=[0, n_0 * 1.1], mode='lines',
                      line=dict(color='darkblue', width=2, dash='dot'),
                      showlegend=False),
        ],
        layout=go.Layout(
            title=dict(text='Continuity Equation: Drift + Diffusion + Recombination', font=dict(size=18)),
            xaxis=dict(title='Position x (μm)', range=[-5, 5], tickfont=dict(size=14)),
            yaxis=dict(title="Excess Electron Concentration n' (cm⁻³)", range=[0, n_0 * 1.1], tickfont=dict(size=14), exponentformat='e'),
            showlegend=False,
            annotations=[
                dict(x=0.98, y=0.98, xref='paper', yref='paper', showarrow=False,
                     text=f"t = 0.0 ns<br>Peak: {n_0:.2e} cm⁻³<br>Center: {x_center_0:.2f} μm<br>Width σ: {sigma_0:.2f} μm",
                     font=dict(size=14), align='right',
                     bgcolor='rgba(224, 255, 255, 0.9)', bordercolor='gray', borderwidth=1),
                dict(x=0.02, y=0.98, xref='paper', yref='paper', showarrow=False,
                     text=f"D<sub>n</sub> = {D_anim:.1f} cm²/s<br>v<sub>d</sub> = {v_drift_anim:.1e} cm/s",
                     font=dict(size=14), align='left',
                     bgcolor='rgba(255, 255, 255, 0.9)', bordercolor='gray', borderwidth=1),
            ],
            updatemenus=[
                dict(type='buttons', showactive=False, x=0.1, y=-0.18, xanchor='right',
                     buttons=[
                         dict(label='▶ Play', method='animate',
                              args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, mode='immediate')]),
                         dict(label='⏸ Pause', method='animate',
                              args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                     ])
            ],
            sliders=[
                dict(active=0, yanchor='top', xanchor='left', x=0.1, y=-0.08, len=0.85,
                     currentvalue=dict(prefix='Time: ', suffix=' ns', font=dict(size=14), visible=True, xanchor='center'),
                     steps=[dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                                 label=f'{time_points_ns[i]:.1f}', method='animate') for i in range(0, n_frames, 5)])
            ],
            height=600,
            margin=dict(b=140)
        ),
        frames=frames
    )

    mo.vstack([
        mo.md("### Animation: Spatio-temporal evolution of excess carriers"),
        mo.md(r"""
        This animation shows the numerical solution to the continuity equation for excess electrons in a constant electric field (low-injection regime)

        $$\frac{\partial n'}{\partial t} = D_n \frac{\partial^2 n'}{\partial x^2} + \mu_n \mathcal{E} \frac{\partial n'}{\partial x} - \frac{n'}{\tau}$$

        Adjust the parameters and press **Play** or drag the slider.
        """),
        mo.hstack([mu_anim_slider, T_anim_slider], justify="start"),
        mo.hstack([E_anim_slider, tau_anim_slider], justify="start"),
        fig
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Observations:**

    1. **Drift**: Pulse center moves at $v_d = -\mu \mathcal{E}$

    2. **Diffusion** ($D_n \frac{\partial^2 n}{\partial x^2}$): Pulse spreads over time
       - Width grows as $\sigma(t) \approx \sqrt{\sigma_0^2 + 2Dt}$

    3. **Recombination** ($-n/\tau$): Pulse amplitude decays exponentially
       - Total charge decays as $\sim e^{-t/\tau}$

    **Experiments to try:**
    - Vary the recombination rate to see decay
    - Vary the electric field and mobility to observe the velocity of the pulse
    """)
    return


if __name__ == "__main__":
    app.run()
