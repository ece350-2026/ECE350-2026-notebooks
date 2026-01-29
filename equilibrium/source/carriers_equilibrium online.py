import marimo

__generated_with = "0.19.4"
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
    from matplotlib.patches import Rectangle
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Carriers at Thermal Equilibrium

    Lectures 9 - 10

    Jan. 23 and 26, 2026

    References: Pierret Ch. 4, Hu Ch. 1.6-1.11
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Introduction

    In this notebook, we learn about the following concepts:

    1. **Density of States (DoS)** - The number of available states at an energy
    2. **Fermi-Dirac Statistics** - The probability that a state is occupied
    3. **Carrier Concentrations** - Combining DoS and occupation probability

    By thermal equilibrium, we mean that
    - The semiconductor is at a fixed temperature
    - No other external forces or excitations are applied
    - No net current flow

    **IMPORTANT**: Distinguish between availability of a (quantum) state and the occupation of a state!
    - Quantum states are the eigenfunction solutions (i.e., eigenstates) of the time-independent Schrödinger equation. The existence of an eigenstate does not imply that an electron in that eigenstate. Whether an eigenstate is occupied by an electron depends on the occupation probability.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Density of States (DoS)

    - The **density of states** $D(E)$ tells us how many quantum states are available per unit energy per unit volume:

    $$D(E) = \frac{\text{number of available states}}{\text{unit volume} \times \text{unit energy}} \quad [\text{eV}^{-1}\text{cm}^{-3}]$$



    - The density of states is normalized by the volume because there are more states when there are more atoms

    ### How to think about this?

    - Consider a single atom with states (1s, 2s, 2p, 3s, 3p, 3d, ...) which increase in energy. As energy increases, there are more orbitals available within a given energy range $\Delta E$. Similarly, in a crystal, the density of states increases with energy.
    """)
    return


@app.cell
def _(mo):
    # Interactive sliders for the DOS demonstration

    E_center_slider = mo.ui.slider(
        start=1,
        stop=3,
        step=0.1,
        value=1,
        label="Energy window center"
    )
    return (E_center_slider,)


@app.cell
def _(E_center_slider, mo, plt):
    """
    Interactive diagram showing atomic energy levels to explain
    the concept of Density of States (DOS).

    As the energy window ΔE is moved through different regions,
    the number of states within that window changes - illustrating
    that DOS varies with energy.
    """

    # Define atomic-like energy levels (arbitrary units, mimicking s, p, d orbitals)
    # Format: (energy, label, color, degeneracy/number of lines)
    energy_levels = [
        (1.0, '1s', 'black', 1),
        (1.5, '2s', 'blue', 1),
        (1.75, '2p', 'blue', 3),
        (2.25, '3s', 'green', 1),
        (2.5, '3p', 'green', 3),
        (2.75, '4s', 'orange', 1),
        (2.9, '3d', 'green', 5),
        (3.1, '4p', 'orange', 3),
    ]

    # Get slider values
    delta_E = 0.35
    E_center = E_center_slider.value
    E_min_window = E_center - delta_E / 2
    E_max_window = E_center + delta_E / 2

    # Create figure
    fig_dos_concept, ax = plt.subplots(figsize=(8, 4))

    # Draw energy axis
    ax.annotate('', xy=(0, 3.5), xytext=(0, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(-0.3, 3.0, '$E$', fontsize=16, fontweight='bold')

    # Draw vertical dots at top to indicate continuation
    for i in range(3):
        ax.plot(0.5, 6.5 + i * 0.15, 'ko', markersize=4)

    # Draw each energy level
    line_width = 0.3  # Width of each horizontal line
    x_start = 0.3

    for energy, label, color, degeneracy in energy_levels:
        # Draw multiple lines for degenerate states (slightly offset)
        for i in range(degeneracy):
            offset = (i - (degeneracy - 1) / 2) * 0.08
            x_left = x_start + offset
            x_right = x_left + line_width
            ax.hlines(energy, x_left, x_right, colors=color, linewidth=3)

        # Add label to the right
        ax.text(x_start + line_width + 0.15, energy, label, 
                fontsize=12, va='center', color=color, fontweight='bold')

    # Draw the ΔE window (shaded region)
    window_x_min = -0.1
    window_x_max = 1.2
    ax.fill_between([window_x_min, window_x_max], E_min_window, E_max_window,
                    alpha=0.25, color='yellow', edgecolor='orange', linewidth=2)

    # Draw ΔE bracket on the right side
    bracket_x = 1.0
    ax.annotate('', xy=(bracket_x, E_max_window), xytext=(bracket_x, E_min_window),
                arrowprops=dict(arrowstyle='<->', color='darkorange', lw=2))
    ax.text(bracket_x + 0.1, E_center, '$\\Delta E$', fontsize=14, 
            color='darkorange', fontweight='bold', va='center')

    # Count states within the window
    states_in_window = 0
    levels_in_window = []
    for energy, label, color, degeneracy in energy_levels:
        if E_min_window <= energy <= E_max_window:
            states_in_window += degeneracy * 2  # Factor of 2 for spin
            levels_in_window.append(f"{label} ({degeneracy}×2 = {degeneracy*2} states)")

    # Add text showing count
    info_text = f"States in window: {states_in_window}"
    if levels_in_window:
        info_text += "\n" + "\n".join(levels_in_window)
    else:
        info_text += "\n(no levels in window)"

    ax.text(1.5, 2.5, info_text, fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            va='center')


    # Formatting
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('DoS of an atom\nCounting States in Energy Window $\\Delta E$', 
                 fontsize=14, pad=0)

    plt.tight_layout()
    #mo.mpl.interactive(fig_dos_concept)

    mo.vstack([
        mo.md("### Adjust the energy window to count states of a single atom:"),
        E_center_slider,
        mo.mpl.interactive(fig_dos_concept)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.1 Derivation of DoS in 3D (Bulk Semiconductor)

    Let's derive the DoS step-by-step.

    #### Step 1: Volume of a state in $k$ space

    Consider an electron confined in a 3D cube of semiconductor crystal with side length $L$ (volume $V = L^3$).

    Recall **periodic boundary conditions:**

    $$\psi(x+L, y, z) = \psi(x, y, z), \psi(x, y+L, z) = \psi(x, y, z), \psi(x, y, z+L) = \psi(x, y, z)$$

    **Bloch wavefunction** implies

    $$\psi(x+L, y, z) = e^{ik_xL}\psi(x, y, z), \quad \text{similarly for y and z}$$

    $$\therefore e^{ik_x L} = 1 \quad \Rightarrow \quad k_x L = 2\pi n_x, \quad \text{where } n_x \text{ is an integer}$$

    $$\therefore k_x = \frac{2\pi n_x}{L}, \quad k_y = \frac{2\pi n_y}{L}, \quad k_z = \frac{2\pi n_z}{L}$$

    In $k$-space, the states are spaced at:

    $$\Delta k_x = \Delta k_y = \Delta k_z = \frac{2\pi}{L}$$

    Each state occupies a volume in k-space of:

    $$\boxed{\text{Volume per state in k-space} = \Delta k_x \Delta k_y \Delta k_z = \left(\frac{2\pi}{L}\right)^3 = \frac{(2\pi)^3}{V}}$$

    Taking the reciprocal, we have **the number of states per unit k-space volume** (i.e., the density of states):

    $$\rho_k = \frac{V}{(2\pi)^3} \quad \text{states per unit k-space volume}$$

    Each value of $(k_x, k_y, k_z)$ actually represents 2 states due to spin degeneracy (spin up and spin down), so:

    $$\text{Number of states per k-space volume} = \rho_k = \frac{2V}{(2\pi)^3} \quad \text{(including spin)}$$
    """)
    return


@app.cell
def _(go, make_subplots, np):
    def _():
        # Visualization of k-space quantization in 3D using Plotly for interactivity

        L_crystal = 5  # nm
        dk = 2 * np.pi / L_crystal

        # Create grid of allowed k-points in 3D (in units of 2π/L)
        n_max = 3
        n_points = np.arange(-n_max, n_max+1)
        NX, NY, NZ = np.meshgrid(n_points, n_points, n_points)

        # Create the 3D figure
        fig_kspace_3d = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
            column_widths=[0.6, 0.4],
            subplot_titles=("Eigenstate (k<sub>x</sub>, k<sub>y</sub>, k<sub>z</sub>) in 3D k-space", "")
        )

        # Plot all k-states as scatter points (in normalized units)
        fig_kspace_3d.add_trace(
            go.Scatter3d(
                x=NX.flatten(),
                y=NY.flatten(),
                z=NZ.flatten(),
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.7),
                name='k-states',
                showlegend= False,
                hovertemplate='n<sub>x</sub>: %{x}<br>n<sub>y</sub>: %{y}<br>n<sub>z</sub>: %{z}<extra></extra>'
            ),
            row=1, col=1
        )

        # Draw arrows showing Δk spacing (from origin to (1,0,0), (0,1,0), (0,0,1))
        # Arrow along x
        fig_kspace_3d.add_trace(
            go.Scatter3d(
                x=[0, 1], y=[0, 0], z=[0, 0],
                mode='lines',
                line=dict(color='red', width=6),
                name='Δk<sub>x</sub>',
                showlegend=False
            ),
            row=1, col=1
        )
        fig_kspace_3d.add_trace(
            go.Cone(x=[1], y=[0], z=[0], u=[0.3], v=[0], w=[0],
                    colorscale=[[0, 'red'], [1, 'red']], showscale=False,
                    sizemode='absolute', sizeref=0.15),
            row=1, col=1
        )

        # Arrow along y
        fig_kspace_3d.add_trace(
            go.Scatter3d(
                x=[0, 0], y=[0, 1], z=[0, 0],
                mode='lines',
                line=dict(color='red', width=6),
                name='Δk<sub>y</sub>',
                showlegend=False
            ),
            row=1, col=1
        )
        fig_kspace_3d.add_trace(
            go.Cone(x=[0], y=[1], z=[0], u=[0], v=[0.3], w=[0],
                    colorscale=[[0, 'red'], [1, 'red']], showscale=False,
                    sizemode='absolute', sizeref=0.15),
            row=1, col=1
        )

        # Arrow along z
        fig_kspace_3d.add_trace(
            go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[0, 1],
                mode='lines',
                line=dict(color='red', width=6),
                name='Δk<sub>z</sub>',
                showlegend=False
            ),
            row=1, col=1
        )
        fig_kspace_3d.add_trace(
            go.Cone(x=[0], y=[0], z=[1], u=[0], v=[0], w=[0.3],
                    colorscale=[[0, 'red'], [1, 'red']], showscale=False,
                    sizemode='absolute', sizeref=0.15),
            row=1, col=1
        )

        # Add labels for Δk near the arrows
        fig_kspace_3d.add_trace(
            go.Scatter3d(
                x=[0.5, -0.4, -0.4],
                y=[-0.4, 0.5, -0.4],
                z=[-0.4, -0.4, 0.5],
                mode='text',
                text=['Δkₓ', 'Δkᵧ', 'Δk_z'],
                textfont=dict(size=12, color='red'),
                showlegend=False
            ),
            row=1, col=1
        )

        # Draw a cube showing the volume per state (unit cube from 0 to 1)
        cube_edges = [
            # Bottom face
            ([0, 1], [0, 0], [0, 0]),
            ([1, 1], [0, 1], [0, 0]),
            ([1, 0], [1, 1], [0, 0]),
            ([0, 0], [1, 0], [0, 0]),
            # Top face
            ([0, 1], [0, 0], [1, 1]),
            ([1, 1], [0, 1], [1, 1]),
            ([1, 0], [1, 1], [1, 1]),
            ([0, 0], [1, 0], [1, 1]),
            # Vertical edges
            ([0, 0], [0, 0], [0, 1]),
            ([1, 1], [0, 0], [0, 1]),
            ([1, 1], [1, 1], [0, 1]),
            ([0, 0], [1, 1], [0, 1]),
        ]

        for edge in cube_edges:
            fig_kspace_3d.add_trace(
                go.Scatter3d(
                    x=edge[0], y=edge[1], z=edge[2],
                    mode='lines',
                    line=dict(color='green', width=4),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

        # Add label for volume per k-state
        fig_kspace_3d.add_trace(
            go.Scatter3d(
                x=[0.5],
                y=[0.5],
                z=[1.3],
                mode='text',
                text=['Volume per<br>k-state'],
                textfont=dict(size=14, color='green'),
                showlegend=False
            ),
            row=1, col=1
        )

        # Update 3D scene
        fig_kspace_3d.update_scenes(
            xaxis=dict(
                title=dict(text='kₓ (units of 2π/L)', font=dict(size=12, color='black')),
                range=[-n_max-0.5, n_max+0.5],
                tickvals=list(range(-n_max, n_max+1)),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            yaxis=dict(
                title=dict(text='kᵧ (units of 2π/L)', font=dict(size=12, color='black')),
                range=[-n_max-0.5, n_max+0.5],
                tickvals=list(range(-n_max, n_max+1)),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            zaxis=dict(
                title=dict(text='k_z (units of 2π/L)', font=dict(size=12, color='black')),
                range=[-n_max-0.5, n_max+0.5],
                tickvals=list(range(-n_max, n_max+1)),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        )

        # Right panel: Formulas and explanation (using annotations)
        annotations_text = [
            dict(x=0.78, y=0.62, xref='paper', yref='paper',
                 text='<b>Volume per k-state:</b>',
                 showarrow=False, font=dict(size=14)),
            dict(x=0.78, y=0.52, xref='paper', yref='paper',
                 text='Δk<sub>x</sub>·Δk<sub>y</sub>·Δk<sub>z</sub> = (2π/L)³ = (2π)³/V',
                 showarrow=False, font=dict(size=14)),
            dict(x=0.78, y=0.38, xref='paper', yref='paper',
                 text='<b>Density of k-states:</b>',
                 showarrow=False, font=dict(size=14)),
            dict(x=0.78, y=0.28, xref='paper', yref='paper',
                 text='ρ<sub>k</sub> = V/(2π)³ states per unit k-volume',
                 showarrow=False, font=dict(size=14)),
            dict(x=0.78, y=0.14, xref='paper', yref='paper',
                 text='Including spin: ρ<sub>k</sub> = 2V/(2π)³',
                 showarrow=False, font=dict(size=14)),
        ]

        # Hide the right subplot axes (we're using it just for text)
        fig_kspace_3d.update_xaxes(visible=False, row=1, col=2)
        fig_kspace_3d.update_yaxes(visible=False, row=1, col=2)

        fig_kspace_3d.update_layout(
            height=600,
            width=1000,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            annotations=annotations_text,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig_kspace_3d


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Step 2: Counting States in k-space

    For a parabolic band, the energy depends only on the magnitude of $\mathbf{k}$:

    $$E = E_c + \frac{\hbar^2 k^2}{2m_n^*}$$

    $$E = E_v - \frac{\hbar^2 k^2}{2m_p^*}$$

    where $k = |\mathbf{k}| = \sqrt{k_x^2 + k_y^2 + k_z^2}$.

    States with the same $k$ (same energy) lie on a sphere of radius $k$ in k-space.

    **Number of states between $k$ and $k + dk$:**

    The volume of a spherical shell of radius $k$ and thickness $dk$ is:

    $$\text{Volume of shell} = 4\pi k^2 dk$$

    Number of states in this shell:

    $$dN = \rho_k \times \text{Volume of shell} = \frac{2V}{(2\pi)^3} \times 4\pi k^2 dk$$

    $$\boxed{dN = \frac{V k^2 dk}{\pi^2}}$$
    """)
    return


@app.cell
def _(go, make_subplots, np):
    def _():
        # Visualization of spherical shells in 3D k-space using Plotly for interactivity

        # Shell parameters (in units of 2π/L)
        k_highlight = 1.35
        dk_highlight = 0.2

        # Create grid of allowed k-points in 3D (in units of 2π/L)
        n_max = 3
        n_points = np.arange(-n_max, n_max+1)
        NX, NY, NZ = np.meshgrid(n_points, n_points, n_points)

        # Create the 3D figure
        fig_shell_3d = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
            column_widths=[0.6, 0.4],
            subplot_titles=("Spherical Shell in 3D k-space", "")
        )

        # Plot all k-states as scatter points (blue dots for permissible k values)
        fig_shell_3d.add_trace(
            go.Scatter3d(
                x=NX.flatten(),
                y=NY.flatten(),
                z=NZ.flatten(),
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.6),
                name='Allowed k-states',
                hovertemplate='n<sub>x</sub>: %{x}<br>n<sub>y</sub>: %{y}<br>n<sub>z</sub>: %{z}<extra></extra>'
            ),
            row=1, col=1
        )

        # Create sphere meshes
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)

        # Inner sphere (k)
        x_inner = k_highlight * np.outer(np.cos(u), np.sin(v))
        y_inner = k_highlight * np.outer(np.sin(u), np.sin(v))
        z_inner = k_highlight * np.outer(np.ones(np.size(u)), np.cos(v))

        fig_shell_3d.add_trace(
            go.Surface(
                x=x_inner, y=y_inner, z=z_inner,
                opacity=0.2,
                colorscale=[[0, 'blue'], [1, 'blue']],
                showscale=False,
                name='k (inner sphere)',
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Outer sphere (k + dk)
        x_outer = (k_highlight + dk_highlight) * np.outer(np.cos(u), np.sin(v))
        y_outer = (k_highlight + dk_highlight) * np.outer(np.sin(u), np.sin(v))
        z_outer = (k_highlight + dk_highlight) * np.outer(np.ones(np.size(u)), np.cos(v))

        fig_shell_3d.add_trace(
            go.Surface(
                x=x_outer, y=y_outer, z=z_outer,
                opacity=0.15,
                colorscale=[[0, 'red'], [1, 'red']],
                showscale=False,
                name='k+dk (outer sphere)',
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Draw coordinate axes
        axis_length = 2.2

        # X axis
        fig_shell_3d.add_trace(
            go.Scatter3d(
                x=[0, axis_length], y=[0, 0], z=[0, 0],
                mode='lines',
                line=dict(color='black', width=4),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        fig_shell_3d.add_trace(
            go.Cone(x=[axis_length], y=[0], z=[0], u=[0.2], v=[0], w=[0],
                    colorscale=[[0, 'black'], [1, 'black']], showscale=False,
                    sizemode='absolute', sizeref=0.1, hoverinfo='skip'),
            row=1, col=1
        )

        # Y axis
        fig_shell_3d.add_trace(
            go.Scatter3d(
                x=[0, 0], y=[0, axis_length], z=[0, 0],
                mode='lines',
                line=dict(color='black', width=4),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        fig_shell_3d.add_trace(
            go.Cone(x=[0], y=[axis_length], z=[0], u=[0], v=[0.2], w=[0],
                    colorscale=[[0, 'black'], [1, 'black']], showscale=False,
                    sizemode='absolute', sizeref=0.1, hoverinfo='skip'),
            row=1, col=1
        )

        # Z axis
        fig_shell_3d.add_trace(
            go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[0, axis_length],
                mode='lines',
                line=dict(color='black', width=4),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        fig_shell_3d.add_trace(
            go.Cone(x=[0], y=[0], z=[axis_length], u=[0], v=[0], w=[0.2],
                    colorscale=[[0, 'black'], [1, 'black']], showscale=False,
                    sizemode='absolute', sizeref=0.1, hoverinfo='skip'),
            row=1, col=1
        )

        # Draw k vector (from origin to a point on the inner sphere)
        theta_k = np.pi / 4  # 45 degrees from z-axis
        phi_k = np.pi / 5    # azimuthal angle
        kx = k_highlight * np.sin(theta_k) * np.cos(phi_k)
        ky = k_highlight * np.sin(theta_k) * np.sin(phi_k)
        kz = k_highlight * np.cos(theta_k)

        # Draw the k vector as a line
        fig_shell_3d.add_trace(
            go.Scatter3d(
                x=[0, kx], y=[0, ky], z=[0, kz],
                mode='lines',
                line=dict(color='green', width=6),
                name='k vector',
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Add cone at tip of k vector
        dir_x = kx / k_highlight
        dir_y = ky / k_highlight
        dir_z = kz / k_highlight
        fig_shell_3d.add_trace(
            go.Cone(x=[kx], y=[ky], z=[kz], u=[dir_x*0.15], v=[dir_y*0.15], w=[dir_z*0.15],
                    colorscale=[[0, 'green'], [1, 'green']], showscale=False,
                    sizemode='absolute', sizeref=0.1, hoverinfo='skip'),
            row=1, col=1
        )

        # Add a point at the tip of k vector
        fig_shell_3d.add_trace(
            go.Scatter3d(
                x=[kx], y=[ky], z=[kz],
                mode='markers',
                marker=dict(size=6, color='green'),
                showlegend=False,
                hovertemplate='k vector tip<br>|k| = 1.5<extra></extra>'
            ),
            row=1, col=1
        )

        # Draw dk arrow (radial direction, showing shell thickness)
        kx_outer = (k_highlight + dk_highlight) * np.sin(theta_k) * np.cos(phi_k)
        ky_outer = (k_highlight + dk_highlight) * np.sin(theta_k) * np.sin(phi_k)
        kz_outer = (k_highlight + dk_highlight) * np.cos(theta_k)

        fig_shell_3d.add_trace(
            go.Scatter3d(
                x=[kx, kx_outer], y=[ky, ky_outer], z=[kz, kz_outer],
                mode='lines',
                line=dict(color='orange', width=6),
                name='dk (shell thickness)',
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        fig_shell_3d.add_trace(
            go.Cone(x=[kx_outer], y=[ky_outer], z=[kz_outer], 
                    u=[dir_x*0.08], v=[dir_y*0.08], w=[dir_z*0.08],
                    colorscale=[[0, 'orange'], [1, 'orange']], showscale=False,
                    sizemode='absolute', sizeref=0.08, hoverinfo='skip'),
            row=1, col=1
        )

        # Draw circles at equator to show shell structure
        theta_eq = np.linspace(0, 2 * np.pi, 100)

        # Inner equator circle
        fig_shell_3d.add_trace(
            go.Scatter3d(
                x=k_highlight * np.cos(theta_eq),
                y=k_highlight * np.sin(theta_eq),
                z=np.zeros_like(theta_eq),
                mode='lines',
                line=dict(color='blue', width=4),
                name=f'k = {k_highlight} (inner)',
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Outer equator circle
        fig_shell_3d.add_trace(
            go.Scatter3d(
                x=(k_highlight + dk_highlight) * np.cos(theta_eq),
                y=(k_highlight + dk_highlight) * np.sin(theta_eq),
                z=np.zeros_like(theta_eq),
                mode='lines',
                line=dict(color='red', width=4, dash='dash'),
                name=f'k + dk = {k_highlight + dk_highlight} (outer)',
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Draw a meridian to show shell structure
        phi_mer = np.linspace(0, 2 * np.pi, 100)

        # Inner meridian
        fig_shell_3d.add_trace(
            go.Scatter3d(
                x=k_highlight * np.sin(phi_mer),
                y=np.zeros_like(phi_mer),
                z=k_highlight * np.cos(phi_mer),
                mode='lines',
                line=dict(color='blue', width=4),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Outer meridian
        fig_shell_3d.add_trace(
            go.Scatter3d(
                x=(k_highlight + dk_highlight) * np.sin(phi_mer),
                y=np.zeros_like(phi_mer),
                z=(k_highlight + dk_highlight) * np.cos(phi_mer),
                mode='lines',
                line=dict(color='red', width=4, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Update 3D scene
        fig_shell_3d.update_scenes(
            xaxis=dict(
                title=dict(text='kₓ (units of 2π/L)', font=dict(size=12)),
                range=[-2.5, 2.5],
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            yaxis=dict(
                title=dict(text='kᵧ (units of 2π/L)', font=dict(size=12)),
                range=[-2.5, 2.5],
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            zaxis=dict(
                title=dict(text='kᵤ (units of 2π/L)', font=dict(size=12)),
                range=[-2.5, 2.5],
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        )

       # Right panel: Formula explanation (using annotations)
        fig_shell_3d.update_layout(
            annotations=[
                dict(x=0.78, y=0.95, xref='paper', yref='paper',
                     text='Spherical Shell in k-space<',
                     showarrow=False, font=dict(size=16)),
                dict(x=0.78, y=0.88, xref='paper', yref='paper',
                     text='The k vector:',
                     showarrow=False, font=dict(size=13)),
                dict(x=0.78, y=0.82, xref='paper', yref='paper',
                     text='k = kₓx̂ + kᵧŷ + kᵤẑ',
                     showarrow=False, font=dict(size=12)),
                dict(x=0.78, y=0.75, xref='paper', yref='paper',
                     text='|k| = √(kₓ² + kᵧ² + kᵤ²)',
                     showarrow=False, font=dict(size=12, color='green'),
                     bgcolor='lightgreen', borderpad=4),
                dict(x=0.78, y=0.62, xref='paper', yref='paper',
                     text='Volume of Spherical Shell:',
                     showarrow=False, font=dict(size=13)),
                dict(x=0.78, y=0.55, xref='paper', yref='paper',
                     text='Surface area: 4πk²',
                     showarrow=False, font=dict(size=12)),
                dict(x=0.78, y=0.48, xref='paper', yref='paper',
                     text='Shell thickness: dk',
                     showarrow=False, font=dict(size=12)),
                dict(x=0.78, y=0.40, xref='paper', yref='paper',
                     text='Shell Volume = 4πk² dk',
                     showarrow=False, font=dict(size=14, color='blue'),
                     bgcolor='lightblue', borderpad=4),
                dict(x=0.78, y=0.28, xref='paper', yref='paper',
                     text='Number of states in shell:',
                     showarrow=False, font=dict(size=13)),
                dict(x=0.78, y=0.20, xref='paper', yref='paper',
                     text='dN = (Shell Vol.) × (States/Vol.) × (Spin)',
                     showarrow=False, font=dict(size=11)),
                dict(x=0.78, y=0.13, xref='paper', yref='paper',
                     text='dN = 4πk²dk × V/(2π)³ × 2',
                     showarrow=False, font=dict(size=12)),
                dict(x=0.78, y=0.05, xref='paper', yref='paper',
                     text='dN = Vk²dk / π²',
                     showarrow=False, font=dict(size=14, color='purple'),
                     bgcolor='#E6E6FA', borderpad=4),
            ],
            title=dict(
                text='Counting States in a Spherical Shell',
                font=dict(size=18),
                x=0.5
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            width=1100,
            height=600,
            margin=dict(l=10, r=10, t=60, b=10)
        )

        #mo.ui.plotly(fig_shell_3d)

        return fig_shell_3d
    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Step 3: Converting from k to Energy

    We have the number of states as a function of $k$, but it is much more convenient to use it as a function of energy $E$.

    $k$ and $E$ are related through the dispersion relation.

    Near the bottom of the conduction band, $E = E_c + \frac{\hbar^2 k^2}{2m_n^*}$, so we get

    $$k = \sqrt{\frac{2m^*(E - E_c)}{\hbar^2}}$$

    Differentiate both sides with respect to $E$:

    $$\frac{dk}{dE} = \frac{d}{dE}\left[\sqrt{\frac{2m^*(E - E_c)}{\hbar^2}}\right]$$

    $$\frac{dk}{dE} = \frac{1}{2}\sqrt{\frac{2m^*}{\hbar^2(E - E_c)}}$$

    $$\therefore \boxed{dk = \frac{1}{2}\sqrt{\frac{2m^*}{\hbar^2(E - E_c)}} dE}$$

    (Similarly for the holes)
    """)
    return


@app.cell
def _(fig_final, mo):
    mo.vstack([

    mo.md(r"""

    #### Step 4: Final Expression for Density of States

    Recall that $dN = \frac{V k^2}{\pi^2} dk$

    Substitute $k = \sqrt{\frac{2m_n^*(E-E_c)}{\hbar^2}}$ and $dk = \frac{1}{2}\sqrt{\frac{2m_n^*}{\hbar^2(E-E_c)}} dE$:

    $$dN = \frac{V}{\pi^2} \left(\frac{2m_n^*(E-E_c)}{\hbar^2}\right) \times \frac{1}{2}\sqrt{\frac{2m_n^*}{\hbar^2(E-E_c)}} dE$$

    $$dN = \frac{V}{2\pi^2} \left(\frac{2m_n^*}{\hbar^2}\right)^{3/2} \sqrt{E-E_c} \, dE$$

    **The density of states per unit volume for electrons:**

    $$\boxed{D_c(E) \equiv \frac{1}{V}\frac{dN}{dE} = \frac{1}{2\pi^2} \left(\frac{2m_n^*}{\hbar^2}\right)^{3/2} \sqrt{E-E_c} = \frac{4\pi (2m_n^*)^{3/2}}{h^3} \sqrt{E - E_c}}$$

    Similarly, **the density of states per unit volume for holes:**

    $$\boxed{D_v(E) \equiv \frac{1}{V}\frac{dN}{dE} = \frac{1}{2\pi^2} \left(\frac{2m_p^*}{\hbar^2}\right)^{3/2} \sqrt{E_v-E} =  \frac{4\pi (2m_p^*)^{3/2}}{h^3} \sqrt{E_v - E}}$$

    **Notes:**
    - $D(E) \propto \sqrt{E - E_c}$ (square root dependence)
    - The DoS increases with the energy 
    - In the bandgap, $D(E) = 0$
    """),
        mo.mpl.interactive(fig_final)
    ])
    return


@app.cell
def _(np, plt):
    """
    Final visualization: The complete 3D DOS with breakdown of contributing factors
    """

    fig_final, ax_final = plt.subplots(figsize=(6, 4))

    _E_c_plot = 0  # Reference
    _E_above_Ec = np.linspace(0.001, 0.5, 500)

    # D(E) ∝ k² × (dk/dE) ∝ (E-Ec) × 1/√(E-Ec) = √(E-Ec)
    _dos_final = np.sqrt(_E_above_Ec)
    ax_final.fill_between(_E_above_Ec, 0, _dos_final, alpha=0.4, color='purple')
    ax_final.plot(_E_above_Ec, _dos_final, 'purple', linewidth=2.5)
    ax_final.axvline(_E_c_plot, color='red', linestyle='--', label=r'$E_c$')
    ax_final.set_xlabel(r'$E - E_c$', fontsize=12)
    ax_final.set_ylabel(r'$D_c(E)$', fontsize=12)
    ax_final.set_title(r'3D Density of States: $D(E)$', fontsize=14, fontweight='bold')
    ax_final.set_xlim(-0.1, 0.5)
    ax_final.set_ylim(0, 0.8)
    ax_final.grid(True, alpha=0.3)
    ax_final.legend()

    # Add equation box
    ax_final.text(0.25, 0.6, r'$D_c(E) = \frac{4\pi(2m^*)^{3/2}}{h^3}\sqrt{E-E_c}$', 
             fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    #mo.mpl.interactive(fig_final)
    return (fig_final,)


@app.cell
def _(fig, mo):
    mo.vstack([
    mo.md(r"""

    ### 1.2 DoS for Different Dimensionalities

    - You can repeat the 3D analysis procedure and instead assume 1, 2, and 3 dimensions are quantum confined (homework)
        - For example, for 1D quantum confinement, fix $k_z = 0$ (or another specific value), and apply periodic boundary condition in $x$ and $y$ only.
        - For 2D quantum confinement, fix $k_z$ and $k_y$ to specific values, and apply periodic boundary condition in $x$ only.
        - For 3D quantum confinement, fix $k_z$, $k_y$, and $k_z$ to specific values.
    - Result

    | Structure | Quantum Confinement | DoS  |
    |:-----------:|:-------------:|:----------------------:|
    | **Bulk (3D)** | None | $D(E) \propto \sqrt{E-E_c}$ |
    | **Quantum Well (2D)** | 1D  | $D(E) \propto \text{constant (step function)}$ |
    | **Quantum Wire (1D)** | 2D  | $D(E) \propto (E-E_c)^{-1/2}$ |
    | **Quantum Dot (0D)** | 3D  | $D(E) \propto \delta(E-E_n)$ (discrete) |
    """),
    mo.mpl.interactive(fig)
    ])
    return


@app.cell
def _(np, plt):
    # Interactive plot of DOS for different dimensionalities

    E_range = np.linspace(0, 2, 1000)
    E_c = 0.5

    # 3D (Bulk)
    DOS_3D = np.where(E_range > E_c, np.sqrt(E_range - E_c), 0)

    # 2D (Quantum Well) - step functions at quantized energies
    DOS_2D = np.zeros_like(E_range)
    for n in range(1, 5):
        E_n = E_c + 0.2 * n
        mask = E_range >= E_n
        DOS_2D[mask] += 1.0

    # 1D (Quantum Wire)
    DOS_1D = np.zeros_like(E_range)
    for n in range(1, 5):
        E_n = E_c + 0.3 * n
        mask = E_range > E_n
        DOS_1D[mask] += 1.0 / np.sqrt(E_range[mask] - E_n + 0.01)

    # 0D (Quantum Dot) - delta functions
    E_dots = [E_c + 0.3*n for n in range(1, 5)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    # 3D
    axes[0, 0].plot(E_range, DOS_3D, 'b-', linewidth=2)
    axes[0, 0].axvline(E_c, color='r', linestyle='--', label='$E_c$')
    axes[0, 0].set_xlabel('Energy (eV)', fontsize=12)
    axes[0, 0].set_ylabel('DOS (arb. units)', fontsize=12)
    axes[0, 0].set_title('3D (Bulk): $D(E) \propto \sqrt{E-E_c}$', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2D
    axes[0, 1].plot(E_range, DOS_2D, 'g-', linewidth=2)
    axes[0, 1].axvline(E_c, color='r', linestyle='--', label='$E_c$')
    axes[0, 1].set_xlabel('Energy (eV)', fontsize=12)
    axes[0, 1].set_ylabel('DOS (arb. units)', fontsize=12)
    axes[0, 1].set_title(r'2D (Quantum Well): $D(E) = \mathrm{constant}$', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 1D
    axes[1, 0].plot(E_range, DOS_1D, 'm-', linewidth=2)
    axes[1, 0].axvline(E_c, color='r', linestyle='--', label='$E_c$')
    axes[1, 0].set_xlabel('Energy (eV)', fontsize=12)
    axes[1, 0].set_ylabel('DOS (arb. units)', fontsize=12)
    axes[1, 0].set_title('1D (Quantum Wire): $D(E) \propto (E-E_c)^{-1/2}$', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 10])
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 0D
    for E_dot in E_dots:
        axes[1, 1].axvline(E_dot, color='orange', linewidth=3)
    axes[1, 1].axvline(E_c, color='r', linestyle='--', label='$E_c$')
    axes[1, 1].set_xlabel('Energy (eV)', fontsize=12)
    axes[1, 1].set_ylabel('DOS (arb. units)', fontsize=12)
    axes[1, 1].set_title('0D (Quantum Dot): $D(E) = \delta(E-E_n)$', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlim([0, 2])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    #plt.gca()
    return (fig,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Fermi-Dirac Statistics

    - The **Fermi-Dirac distribution** gives the probability that a state at energy $E$ is occupied by an electron
        - Statistics comes from Pauli's exclussion principle and indistinguishable particles

    $$\boxed{f(E) = \frac{1}{1 + e^{(E-E_F)/k_BT}}}$$

    where:
    - $E_F$ is the **Fermi energy** (or Fermi level)
    - $k_B = 8.617 \times 10^{-5}$ eV/K is Boltzmann's constant
    - $T$ is temperature in Kelvin

    ###  Properties:

    - At $E = E_F$: $f(E_F) = 1/2$
    - At $T = 0$ K: Step function ($f(E< E_F) = 1$, $f(E>E_F) = 0$)
    - At $E \ll E_F$: $f(E) \approx 1$ (states are filled)
    - At $E \gg E_F$: $f(E) \approx e^{-(E-E_F)/k_BT}$ (Boltzmann approximation)

    $$\boxed{\text{At thermal equilibrium, $E_F$ is constant throughout the system.}}$$
    """)
    return


@app.cell
def _(mo):
    # Interactive Fermi-Dirac distribution

    temperature_slider = mo.ui.slider(
        start=0,
        stop=600,
        step=50,
        value=300,
        label="Temperature (K):",
        show_value=True
    )


    return (temperature_slider,)


@app.cell
def _(go, mo, np, temperature_slider):
    # Plot Fermi-Dirac distribution

    T = temperature_slider.value
    E_F_pos = 0

    k_B = 8.617e-5  # eV/K

    E_plot = np.linspace(-0.3, 0.3, 1000)

    if T > 0:
        f_E = 1 / (1 + np.exp((E_plot - E_F_pos) / (k_B * T)))
    else:
        f_E = np.where(E_plot < E_F_pos, 1.0, 0.0)

    fig_fd = go.Figure()

    fig_fd.add_trace(go.Scatter(
        x=E_plot,
        y=f_E,
        mode='lines',
        name=f'T = {T} K',
        line=dict(width=3, color='blue')
    ))

    # Add Fermi level line
    fig_fd.add_vline(
        x=E_F_pos,
        line_dash="dash",
        line_color="red",
    )

    # Add horizontal line at f=0.5
    fig_fd.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="gray",
        annotation_text="f = 0.5"
    )

    fig_fd.update_layout(
        title=f"Fermi-Dirac Distribution at T = {T} K",
        xaxis_title='Energy (E - E<sub>F</sub>) (eV)',
        yaxis_title="Occupation Probability f(E)",
        height=500,
        showlegend=False,
        hovermode='x unified'
    )

    fig_fd.update_xaxes(range=[-0.3, 0.3])
    fig_fd.update_yaxes(range=[0, 1.05])

    #fig_fd

    mo.vstack([
        mo.md(f"""
        ### Interactive Fermi-Dirac Distribution

        Adjust the temperature to see how the occupation probability changes:"""),

        temperature_slider,
    
        mo.ui.plotly(fig_fd)
    ])
    return


@app.cell
def _(np, plt):
    # Comparison of Fermi-Dirac vs Maxwell-Boltzmann

    fig_compare, axes_compare = plt.subplots(figsize=(14, 6))

    E_comp = np.linspace(-0.15, 0.15, 1000)
    E_F_comp = 0.0
    k_B_comp = 8.617e-5
    T_comp = 300

    # Fermi-Dirac
    f_FD = 1 / (1 + np.exp((E_comp - E_F_comp) / (k_B_comp * T_comp)))

    # Maxwell-Boltzmann (with same chemical potential)
    f_MB = np.exp(-(E_comp - E_F_comp) / (k_B_comp * T_comp))

    # Left panel: Linear scale
    axes_compare.plot(E_comp, f_FD, 'b-', linewidth=3, label='Fermi-Dirac')
    axes_compare.plot(E_comp, f_MB, 'r--', linewidth=3, label='Maxwell-Boltzmann')
    axes_compare.axvline(E_F_comp, color='black', linestyle=':', linewidth=2, alpha=0.5)
    axes_compare.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)


    axes_compare.set_xlabel(r'Energy $(E - E_F)$ (eV)', fontsize=12, fontweight='bold')
    axes_compare.set_ylabel('Occupation Probability (f(E))', fontsize=12, fontweight='bold')
    axes_compare.set_title('Fermi-Dirac vs. Maxwell-Boltzmann (T = 300 K)', fontsize=14, fontweight='bold')
    axes_compare.legend(fontsize=11)
    axes_compare.grid(True, alpha=0.3)
    axes_compare.set_ylim([0, 1.2])



    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Derivation of the Fermi-Dirac Distribution (optional)

    See also Pierret Ch. 4.2 for an alternative derivation based on counting the number of configurations. This derivation is based on statistical mechanics.

    ### 1. Postulate of Statistical Mechanics

    At thermal equilibrium, a system explores all accessible microstates with equal probability (microcanonical ensemble). For a system in contact with a heat reservoir at temperature $T$ (canonical ensemble), the probability of finding the system in a state with energy $E_i$ is

    $$P(E_i) \propto e^{-E_i/k_BT},$$

    where $k_B$ is Boltzmann's constant and $T$ is the absolute temperature.

    ### 2: The Grand Canonical Ensemble

    For a system that can exchange both energy AND particles with a reservoir, we use the **grand canonical ensemble**.

    **Chemical Potential:** The reservoir is characterized by $T$ and the chemical potential $\mu$ (also called the Fermi energy $E_F$ for electrons).  $\mu$ represents the energy cost to add one particle to the system.

    **Grand Canonical Probability:** The probability of finding the system in a state with energy $E$ and $N$ particles is:

    $$P(E, N) \propto e^{-(E - \mu N)/k_BT} = e^{-E/k_BT} \cdot e^{\mu N/k_BT}$$

    ### 3: Single Quantum State Occupancy

    Now consider a **single quantum state** at energy $E$. This state can be in one of two configurations:

    1. **Empty** (0 electrons): $N = 0$, Energy contribution = 0
    2. **Occupied** (1 electron): $N = 1$, Energy contribution = $E$

    From Pauli's exclusion principle, a quantum state with a given spin can hold at most ONE electron.

    Probability of state being empty: $P_0 \propto e^{-(0 - \mu \cdot 0)/k_BT} = 1$

    Probability of state being occupied: $P_1 \propto e^{-(E - \mu \cdot 1)/k_BT} = e^{-(E-\mu)/k_BT}$

    The total probability must equal 1, so $P_0 + P_1 = 1$

    Therefore,

    $$P_0 = \frac{1}{1 + e^{-(E-\mu)/k_BT}}, \quad P_1 = \frac{e^{-(E-\mu)/k_BT}}{1 + e^{-(E-\mu)/k_BT}}$$


    For electrons, instead of the chemical potential, we use the term "Fermi Energy". Letting $\mu = E_F$ (the Fermi energy), the occupation probability $P_1$ simplifies to the Fermi-Dirac distribution

    $$\boxed{f(E) = \frac{1}{1 + e^{(E-E_F)/k_BT}}}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Carrier Concentrations

    - To find the carrier concentrations, we multiply the density of states by the occupation probability and integrate over the energy range.

    - For electron concentration, we integrate over the conduction band edge to infinity. The band has a finite energy range, but it is ok to extend to infinity because $f(E) \to 0$. The integration to infinity is an approximation, but it turns out to be good most of the time because the energy bands are wider than a few $k_B T$.

    - Therefore, the equilibrium electron concentration, $n_0$, is

    $$n_0 = \int_{E_c}^{\infty} D_c(E) f(E) dE$$

    - For the hole concentration, we integrate from $-\infty$ to $E_v$. Holes are empty states and have occupation probability $1-f(E)$.

    $$p_0 = \int_{-\infty}^{E_v} D_v(E) [1-f(E)] dE$$

    ### With Boltzmann Approximation:

    - Approximating $f(E) \approx e^{-(E-E_F)/k_B T}$, we arrive at

    $$\boxed{n_0 = N_c e^{-(E_c-E_F)/k_BT}, \quad
    \text{where } N_c = 2\left(\frac{2\pi m_n^* k_B T}{h^2}\right)^{3/2}}$$

    $$\boxed{p_0 = N_v e^{-(E_F-E_v)/k_BT}, \quad \text{where } N_v = 2\left(\frac{2\pi m_p^* k_B T}{h^2}\right)^{3/2}}$$

    - $N_c$ and  $N_v$ are called the **effective density of states**

    ### Example Values (at 300 K):

    | Material | $N_c$ (cm$^{-3}$) | $N_v$ (cm$^{-3}$) |  $E_g$ (eV) |
    |----------|-------------------|-------------------|-------------|
    | Si       | $2.8 \times 10^{19}$ | $1.04 \times 10^{19}$ |  1.12 |
    | GaAs     | $4.7 \times 10^{17}$ | $7.0 \times 10^{18}$ |  1.42 |
    | Ge       | $1.04 \times 10^{19}$ | $6.0 \times 10^{18}$ |  0.66 |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Interactive Carrier Density Explorer for Several Semiconductors

    This interactive tool shows how carrier densities change using effective masses from a few semiconductors.

    **Observations:**
    - When $E_F$ is near $E_c$: High electron concentration (n-type)
    - When $E_F$ is near $E_v$: High hole concentration (p-type)
    - $n_0 p_0 = n_i^2$ remains constant
    """)
    return


@app.cell
def _(mo):
    # Interactive carrier density explorer

    material_select = mo.ui.dropdown(
        options=["Silicon", "GaAs", "Germanium", "InP"],
        value="Silicon",
        label="Select Material:"
    )

    temp_slider_2 = mo.ui.slider(
        start=0,
        stop=500,
        step=25,
        value=300,
        label="Temperature (K):",
        show_value=True
    )

    ef_position_slider = mo.ui.slider(
        start=-0.56,
        stop=0.56,
        step=0.02,
        value=0.0,
        label="Fermi Level Position (eV from midgap):",
        show_value=True
    )
    return ef_position_slider, material_select, temp_slider_2


@app.cell
def _(
    ef_position_slider,
    go,
    make_subplots,
    material_select,
    mo,
    np,
    temp_slider_2,
):
    # Calculate and plot carrier concentrations with temperature-dependent Nc and Nv

    # Material parameters including effective masses (in units of m0)
    _material_params = {
        "Silicon": {"mn_eff": 1.08, "mp_eff": 0.81, "Eg": 1.12},
        "GaAs": {"mn_eff": 0.067, "mp_eff": 0.45, "Eg": 1.42},
        "Germanium": {"mn_eff": 0.55, "mp_eff": 0.37, "Eg": 0.66},
        "InP" : {"mn_eff": 0.08, "mp_eff": 0.6, "Eg":1.34}
    }

    # Physical constants
    _m0 = 9.109e-31  # kg (electron rest mass)
    _k_B_J = 1.381e-23  # J/K
    _k_B_eV = 8.617e-5  # eV/K
    _h = 6.626e-34  # J·s
    _hbar = _h / (2 * np.pi)  # reduced Planck constant
    _eV_to_J = 1.602e-19  # conversion factor from eV to Joules

    _mat = material_select.value
    _params = _material_params[_mat]
    _T_2 = temp_slider_2.value
    _E_F_pos_2 = ef_position_slider.value

    # Calculate temperature-dependent Nc and Nv
    # Nc = 2 * (2π * m_n* * k_B * T / h²)^(3/2)
    _mn_star = _params["mn_eff"] * _m0
    _mp_star = _params["mp_eff"] * _m0

    _Nc_calc = 2 * ((2 * np.pi * _mn_star * _k_B_J * _T_2) / (_h**2))**(3/2) / 1e6  # convert to cm^-3
    _Nv_calc = 2 * ((2 * np.pi * _mp_star * _k_B_J * _T_2) / (_h**2))**(3/2) / 1e6  # convert to cm^-3

    # Band edges
    _E_c_2 = _params["Eg"] / 2
    _E_v_2 = -_params["Eg"] / 2

    # Plot range for energy
    _Emax = 1.25
    _Emin = -_Emax

    # Calculate n and p
    _n_calc = _Nc_calc * np.exp(-(_E_c_2 - _E_F_pos_2) / (_k_B_eV * _T_2))
    _p_calc = _Nv_calc * np.exp(-(_E_F_pos_2 - _E_v_2) / (_k_B_eV * _T_2))

    # Calculate intrinsic carrier concentration
    _ni_calc = np.sqrt(_Nc_calc * _Nv_calc) * np.exp(-_params["Eg"] / (2 * _k_B_eV * _T_2))

    # Verify mass action law
    _np_product = _n_calc * _p_calc
    _ni_squared = _ni_calc**2

    # Create energy range for plotting
    _E_band_2 = np.linspace(_Emin, _Emax, 1000)

    # Calculate the actual Density of States using the formula:
    # D_c(E) = (1/(2π²)) * (2m_n*/ℏ²)^(3/2) * √(E-E_c)
    # Units: We calculate in SI (J^-1 m^-3) then convert to cm^-3 eV^-1

    # Prefactor for conduction band DOS: (1/(2π²)) * (2m_n*/ℏ²)^(3/2)
    _DOS_prefactor_c = (1 / (2 * np.pi**2)) * ((2 * _mn_star) / (_hbar**2))**(3/2)  # in J^(-3/2) m^(-3)
    _DOS_prefactor_v = (1 / (2 * np.pi**2)) * ((2 * _mp_star) / (_hbar**2))**(3/2)  # in J^(-3/2) m^(-3)

    # Convert prefactor to cm^-3 eV^-1:
    # D(E) [SI] has units of J^(-3/2) m^(-3) * sqrt(J) = J^(-1) m^(-3)
    # To convert to cm^-3 eV^-1:
    # - J^(-1) to eV^(-1): multiply by eV_to_J (since 1 J^-1 = eV_to_J eV^-1)
    # - m^(-3) to cm^(-3): multiply by 1e-6 (since 1 m^-3 = 1e-6 cm^-3)
    # - sqrt(E) in SI needs sqrt(eV_to_J) factor when E is in eV

    _m3_to_cm3 = 1e-6  # m^-3 to cm^-3
    _sqrt_conversion = np.sqrt(_eV_to_J)  # for sqrt(E) term when E is in eV
    _energy_conversion = _eV_to_J  # J^-1 to eV^-1

    _DOS_prefactor_c_converted = _DOS_prefactor_c * _sqrt_conversion * _energy_conversion * _m3_to_cm3  # cm^-3 eV^-1 per sqrt(eV)
    _DOS_prefactor_v_converted = _DOS_prefactor_v * _sqrt_conversion * _energy_conversion * _m3_to_cm3  # cm^-3 eV^-1 per sqrt(eV)

    # Calculate actual DOS (in cm^-3 eV^-1)
    _DOS_c_2 = np.where(_E_band_2 > _E_c_2, 
                        _DOS_prefactor_c_converted * np.sqrt(_E_band_2 - _E_c_2), 
                        0)
    _DOS_v_2 = np.where(_E_band_2 < _E_v_2, 
                        _DOS_prefactor_v_converted * np.sqrt(_E_v_2 - _E_band_2), 
                        0)

    # Fermi function
    _f_val_2 = 1 / (1 + np.exp((_E_band_2 - _E_F_pos_2) / (_k_B_eV * _T_2)))

    # Carrier distributions (in cm^-3 eV^-1)
    _n_dist_2 = _DOS_c_2 * _f_val_2
    _p_dist_2 = _DOS_v_2 * (1 - _f_val_2)

    # Mask for plotting only in valid regions
    _mask_cb_2 = _E_band_2 >= _E_c_2
    _mask_vb_2 = _E_band_2 <= _E_v_2

    # Create interactive plotly figure with 4 panels
    _fig_carrier_explorer = make_subplots(
        rows=1, cols=4,
        subplot_titles=("Energy Bands", "Density of States D(E)", "Occupation Probability f(E)", "Carrier Distribution D(E)f(E)"),
        horizontal_spacing=0.08
    )

    # Panel 1: Band diagram
    _fig_carrier_explorer.add_trace(
        go.Scatter(x=[0, 1], y=[_E_c_2, _E_c_2], mode='lines', 
                   line=dict(color='blue', width=3), name='E_c', showlegend=False),
        row=1, col=1
    )
    _fig_carrier_explorer.add_trace(
        go.Scatter(x=[0, 1], y=[_E_v_2, _E_v_2], mode='lines', 
                   line=dict(color='blue', width=3), name='E_v', showlegend=False),
        row=1, col=1
    )
    _fig_carrier_explorer.add_trace(
        go.Scatter(x=[0, 1], y=[_E_F_pos_2, _E_F_pos_2], mode='lines', 
                   line=dict(color='red', width=3, dash='dash'), name='E_F', showlegend=False),
        row=1, col=1
    )
    # Conduction band shading
    _fig_carrier_explorer.add_trace(
        go.Scatter(x=[0, 1, 1, 0, 0], y=[_E_c_2, _E_c_2, _Emax, _Emax, _E_c_2],
                   fill='toself', fillcolor='rgba(173, 216, 230, 0.4)',
                   line=dict(color='rgba(0,0,0,0)'), name='CB', showlegend=False),
        row=1, col=1
    )
    # Valence band shading
    _fig_carrier_explorer.add_trace(
        go.Scatter(x=[0, 1, 1, 0, 0], y=[_E_v_2, _E_v_2, _Emin, _Emin, _E_v_2],
                   fill='toself', fillcolor='rgba(173, 216, 230, 0.4)',
                   line=dict(color='rgba(0,0,0,0)'), name='VB', showlegend=False),
        row=1, col=1
    )
    # Add labels
    _fig_carrier_explorer.add_annotation(x=1.1, y=_E_c_2, text="E<sub>c</sub>", showarrow=False, 
                                         font=dict(size=12, color='blue'), xref='x1', yref='y1')
    _fig_carrier_explorer.add_annotation(x=1.1, y=_E_v_2, text="E<sub>v</sub>", showarrow=False, 
                                         font=dict(size=12, color='blue'), xref='x1', yref='y1')
    _fig_carrier_explorer.add_annotation(x=1.1, y=_E_F_pos_2, text="E<sub>F</sub>", showarrow=False, 
                                         font=dict(size=12, color='red'), xref='x1', yref='y1')

    # Panel 2: DOS with effective mass labels (actual values)
    _fig_carrier_explorer.add_trace(
        go.Scatter(x=_DOS_c_2[_mask_cb_2], y=_E_band_2[_mask_cb_2], mode='lines', 
                   line=dict(color='blue', width=2), name='D_c(E)', showlegend=False,
                   hovertemplate='D(E) = %{x:.2e} cm⁻³eV⁻¹<br>E = %{y:.3f} eV<extra></extra>'),
        row=1, col=2
    )
    _fig_carrier_explorer.add_trace(
        go.Scatter(x=_DOS_v_2[_mask_vb_2], y=_E_band_2[_mask_vb_2], mode='lines', 
                   line=dict(color='blue', width=2), name='D_v(E)', showlegend=False,
                   hovertemplate='D(E) = %{x:.2e} cm⁻³eV⁻¹<br>E = %{y:.3f} eV<extra></extra>'),
        row=1, col=2
    )
    # Reference lines
    _fig_carrier_explorer.add_hline(y=_E_c_2, line=dict(color='gray', width=1, dash='dot'), row=1, col=2)
    _fig_carrier_explorer.add_hline(y=_E_v_2, line=dict(color='gray', width=1, dash='dot'), row=1, col=2)
    _fig_carrier_explorer.add_hline(y=_E_F_pos_2, line=dict(color='red', width=1, dash='dash'), row=1, col=2)

    # Add effective mass annotations in DOS panel
    _fig_carrier_explorer.add_annotation(
        x=2 * max(_DOS_c_2), y=_E_c_2 + 0.15, 
        text=f"m<sub>n</sub>* = {_params['mn_eff']:.3f} m₀",
        showarrow=False, font=dict(size=10, color='darkblue'),
        bgcolor='rgba(255,255,255,0.8)', xref='x2', yref='y2'
    )
    _fig_carrier_explorer.add_annotation(
        x=0.5 * max(_DOS_v_2), y=_E_v_2 - 0.15, 
        text=f"m<sub>p</sub>* = {_params['mp_eff']:.3f} m₀",
        showarrow=False, font=dict(size=10, color='darkblue'),
        bgcolor='rgba(255,255,255,0.8)', xref='x2', yref='y2'
    )

    _fig_carrier_explorer.update_xaxes(exponentformat='power', showexponent='all', row=1, col=2)


    # Panel 3: Fermi function
    _fig_carrier_explorer.add_trace(
        go.Scatter(x=_f_val_2, y=_E_band_2, mode='lines', 
                   line=dict(color='green', width=2), name='f(E)', showlegend=False),
        row=1, col=3
    )
    _fig_carrier_explorer.add_hline(y=_E_F_pos_2, line=dict(color='red', width=1, dash='dash'), row=1, col=3)
    _fig_carrier_explorer.add_vline(x=0.5, line=dict(color='gray', width=1, dash='dot'), row=1, col=3)


    # Panel 4: Carrier distributions (only in valid regions)
    _fig_carrier_explorer.add_trace(
        go.Scatter(x=_n_dist_2[_mask_cb_2], y=_E_band_2[_mask_cb_2], mode='lines', 
                   line=dict(color='red', width=2), name='Electrons', showlegend=False,
                   hovertemplate='N(E) = %{x:.2e} cm⁻³eV⁻¹<br>E = %{y:.3f} eV<extra></extra>'),
        row=1, col=4
    )
    _fig_carrier_explorer.add_trace(
        go.Scatter(x=_p_dist_2[_mask_vb_2], y=_E_band_2[_mask_vb_2], mode='lines', 
                   line=dict(color='orange', width=2), name='Holes', showlegend=False,
                   hovertemplate='N(E) = %{x:.2e} cm⁻³eV⁻¹<br>E = %{y:.3f} eV<extra></extra>'),
        row=1, col=4
    )
    _fig_carrier_explorer.update_xaxes(exponentformat='power', showexponent='all', row=1, col=4)
    # Reference lines
    _fig_carrier_explorer.add_hline(y=_E_c_2, line=dict(color='gray', width=1, dash='dot'), row=1, col=4)
    _fig_carrier_explorer.add_hline(y=_E_v_2, line=dict(color='gray', width=1, dash='dot'), row=1, col=4)
    _fig_carrier_explorer.add_hline(y=_E_F_pos_2, line=dict(color='red', width=1, dash='dash'), row=1, col=4)

    # Update layout
    _fig_carrier_explorer.update_xaxes(showticklabels=False, row=1, col=1)
    _fig_carrier_explorer.update_xaxes(title_text="DoS cm<sup>-3</sup> eV<sup>-1</sup>", range=[0, None], row=1, col=2)
    _fig_carrier_explorer.update_xaxes(title_text="f(E)", range=[0, 1], row=1, col=3)
    _fig_carrier_explorer.update_xaxes(title_text="Carrier Density cm<sup>-3</sup> eV<sup>-1</sup>",  row=1, col=4)

    _fig_carrier_explorer.update_yaxes(title_text="Energy (eV)", range=[-1.25, 1.25], row=1, col=1)
    _fig_carrier_explorer.update_yaxes(range=[-1.25, 1.25], row=1, col=2)
    _fig_carrier_explorer.update_yaxes(range=[-1.25, 1.25], row=1, col=3)
    _fig_carrier_explorer.update_yaxes(range=[-1.25, 1.25], row=1, col=4)

    _fig_carrier_explorer.update_layout(
        height=500,
        width=1100,
        title_text=f"Carrier Distribution at T = {_T_2} K, (E<sub>F</sub> - E<sub>mid-gap</sub>) = {_E_F_pos_2:.2f} eV",
        showlegend=True,
        legend=dict(x=0.85, y=0.5),
        hovermode='closest'
    )

    #_fig_carrier_explorer

    mo.vstack([
        mo.md("### Interactive Carrier Density vs. Fermi Level Position"),
        material_select,
        temp_slider_2,
        ef_position_slider,
        mo.ui.plotly(_fig_carrier_explorer)
    ])

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Intrinsic Carrier Concentration ($n_i$) & $n_0 p_0$ Product:

    Multiplying the expressions of $n_0$ and $p_0$, we get

    $$n_0 p_0 = N_c N_v e^{-(E_c - E_v)/k_BT} = N_c N_v e^{-E_g/k_BT}$$

    Consider the special case of an **intrinsic** (i.e., undoped) semiconductor, $n_0 = p_0 = n_i$, where $n_i$ is the intrinsic carrier concentration.

    For the above equation to hold, it means

    $$\boxed{n_i^2 = N_c N_v e^{-E_g/k_BT}}$$

    $$\therefore \boxed{n_0 p_0 = n_i^2}$$

    **For Si, $n_i \approx 10^{10} \text{ cm}^{-3}$ at $T = 300 \text{ K}$.**

    ### Intrinsic Fermi Level ($E_{Fi}$)

    - $E_{Fi}$ is a good reference energy because it is independent of doping and is a property of a pure semiconductor

    - For the intrinsic semiconductor, $n_i = N_c e^{-(E_c-E_{Fi})/k_BT} = n_i = N_v e^{-(E_{Fi}-E_{v})/k_BT}$.

    $$N_c = n_i e^{(E_c - E_{Fi})/k_BT}, \quad N_v = n_i e^{(E_{Fi} - E_v)/k_BT}$$

    $$\therefore \boxed{n_0 = n_i e^{(E_F - E_{Fi})/k_BT}, \quad p_0 = n_i e^{(E_{Fi} - E_F)/k_BT}}$$

    $$\boxed{E_F - E_{Fi} = k_B T \ln\left(\frac{n_0}{n_i}\right) = -k_B T \ln\left(\frac{p_0}{n_i}\right)}$$

    - $E_{Fi}$ is midgap for $N_c = N_v$. Because $N_c$ and $N_v$ are usually different, $E_{Fi}$ is not in the middle of the bandgap.

    ## Other Terminology
    - Extrinsic semiconductor = doped semiconductor
    - Compensated semiconductor = semiconductor containing both acceptors and donors
    - Majority carriers = the higher concentration carrier type in a doped semiconductor (n-type: electrons; p-type: holes)
    - Minority carriers = the low concentration carrier type (n-type: holes; p-type: electrons)

    | Material | Majority carrier | Minority carrier |
    |:----------:|:-----------------:|:--------------:|
    | n-type   | electrons | holes |
    | p-type   | holes | electrons |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Dopant and Carrier Concentrations

    ### Dopants

    - **Donor concentration: N<sub>d</sub>** [cm<sup>-3</sup>]
      - Doping with donors (impurities with excess electrons compared to the nominal atoms in the crystal) make the semiconductor n-type

    - **Acceptor concetration: N<sub>a</sub>** [cm<sup>-3</sup>]
      - Doping with acceptors (impurities lacking electrons compared to the nominal atoms in the crystal) make the semiconductor p-type

    ### Charge Neutrality

    - A doped semiconductor is electrically neutral.

    $$\therefore n_0 + N_a^- = p_0 + N_d^+,$$

    $$N_a^- \text{: concentration of ionized acceptors}$$

    $$N_d^+ \text{: concentration of ionized donors}$$

    - "Ionized" means that the impurity atom (donor or acceptor) has given up or took in the extra electron so it can bond with the atoms in crystal.

    - Assuming complete ionization ($N_a^- = N_a$ and $N_d^+ = N_d$):

    $$n_0 + N_a = p_0 + N_d$$

    - Combined with $n_0 = n_i^2/p_0$ and $p_0 = n_i^2/n_0$, we solve for $n_0$ and $p_0$ to arrive at

    $$\boxed{n_0 = \frac{N_d - N_a}{2} + \sqrt{\left(\frac{N_d - N_a}{2}\right)^2 + n_i^2}}$$

    $$\boxed{p_0 = \frac{N_a - N_d}{2} + \sqrt{\left(\frac{N_a - N_d}{2}\right)^2 + n_i^2}}$$

    ### Simplified Cases:

    **N-type** ($N_d \gg N_a, n_i$):
    - $n_0 \approx N_d - N_a \approx N_d$
    - $p_0 \approx n_i^2 / N_d$ (minority carrier)

    **P-type** ($N_a \gg N_d, n_i$):
    - $p_0 \approx N_a - N_d \approx N_a$
    - $n_0 \approx n_i^2 / N_a$ (minority carrier)

    Typically, $N_a$, $N_d$ are in the range of $10^{14}$ to $10^{17}$ cm$^{-3}$, which is much higher than $n_i$ at room temperature (for Si, $n_i \approx 10^{10} \text{ cm}^{-3}$ at 300 K.
    """)
    return


@app.cell
def _(mo):
    # Doping calculator

    nd_log_slider = mo.ui.slider(
        start=10,
        stop=18,
        step=0.25,
        value=16,
        label="Donor concentration N<sub>d</sub> (log cm⁻³):",
        show_value=True
    )

    na_log_slider = mo.ui.slider(
        start=10,
        stop=18,
        step=0.25,
        value=10,
        label="Acceptor concentration N<sub>a</sub> (log cm⁻³):",
        show_value=True
    )

    temp_input_slider = mo.ui.slider(
        start=200,
        stop=500,
        value=300,
        label="Temperature (K):",
        step=10
    )

    return na_log_slider, nd_log_slider, temp_input_slider


@app.cell
def _(mo, na_log_slider, nd_log_slider, np, temp_input_slider):
    # Convert log values to actual concentrations
    nd_input = 10 ** nd_log_slider.value
    na_input = 10 ** na_log_slider.value
    temp_input = temp_input_slider.value

    # Calculate doped semiconductor properties

    N_d_val = nd_input
    N_a_val = na_input
    T_dope = temp_input

    # Silicon parameters
    m_n_eff_Si = 1.08  # in units of m_0
    m_p_eff_Si = 0.81  # in units of m_0
    E_g_Si = 1.12

    # Physical constants for calculation
    m_0_calc = 9.109e-31  # kg
    k_B_J_calc = 1.381e-23  # J/K
    h_calc = 6.626e-34  # J·s
    k_B_dope = 8.617e-5  # eV/K

    # Calculate temperature-dependent effective density of states
    m_n_star_Si = m_n_eff_Si * m_0_calc
    m_p_star_Si = m_p_eff_Si * m_0_calc

    N_c_Si = 2 * ((2 * np.pi * m_n_star_Si * k_B_J_calc * T_dope) / (h_calc**2))**(3/2) / 1e6  # cm^-3
    N_v_Si = 2 * ((2 * np.pi * m_p_star_Si * k_B_J_calc * T_dope) / (h_calc**2))**(3/2) / 1e6  # cm^-3

    # Calculate temperature-dependent intrinsic carrier concentration
    n_i_Si = np.sqrt(N_c_Si * N_v_Si) * np.exp(-E_g_Si / (2 * k_B_dope * T_dope))

    # Calculate n and p
    n_doped = (N_d_val - N_a_val) / 2 + np.sqrt(((N_d_val - N_a_val) / 2)**2 + n_i_Si**2)
    p_doped = (N_a_val - N_d_val) / 2 + np.sqrt(((N_a_val - N_d_val) / 2)**2 + n_i_Si**2)

    # Calculate E_F position
    E_c_Si = E_g_Si / 2
    E_v_Si = -E_g_Si / 2
    E_i_Si = 0  # midgap

    E_F_doped = E_c_Si - k_B_dope * T_dope * np.log(N_c_Si / n_doped)

    # Determine semiconductor type
    if n_doped > p_doped:
        sc_type = "N-type"
        majority = "Electrons"
        minority = "Holes"
    elif p_doped > n_doped:
        sc_type = "P-type"
        majority = "Holes"
        minority = "Electrons"
    else:
        sc_type = "Intrinsic"
        majority = "N/A"
        minority = "N/A"


    def sci_fmt(val):
        if val == 0:
            return "0"
        if np.isnan(val):
            return "NaN"
        if np.isinf(val):
            if val > 0:
                return "∞"
            else:
                return "-∞"
        if abs(val) < 1e-300:
            return "~0"
        exp = int(np.floor(np.log10(abs(val))))
        coeff = val / (10**exp)
        # Show only one decimal if not integer
        if abs(coeff - round(coeff)) < 1e-8:
            coeff_str = f"{int(round(coeff))}"
        else:
            coeff_str = f"{coeff:.1f}"
        return f"{coeff_str} × 10<sup>{exp}</sup>"


    mo.vstack([
        mo.md("""### Carrier Concentration vs. Doping Calculator for Silicon"""),

        nd_log_slider,
    #    mo.md(f"""**N<sub>d</sub> = {sci_fmt(10 ** nd_log_slider.value)} cm<sup>-3</sup>**"""),
        na_log_slider,
    #    mo.md(f"""**N<sub>a</sub> = {sci_fmt(10 ** na_log_slider.value)} cm<sup>-3</sup>**"""),
        temp_input_slider,
        mo.md(f"""**Temperature** = {temp_input_slider.value} K"""),

        mo.md(rf"""

        **Doping:** N<sub>d</sub> = {sci_fmt(N_d_val)} cm<sup>-3</sup>, N<sub>a</sub> = {sci_fmt(N_a_val)} cm<sup>-3</sup>
    
        **Semiconductor Type:** {sc_type}  
        **Majority carriers:** {majority}  
        **Minority carriers:** {minority}

        **Carrier Concentrations:**
        - Electron concentration: n<sub>0</sub> = {sci_fmt(n_doped)} cm<sup>-3</sup>
        - Hole concentration: p<sub>0</sub> = {sci_fmt(p_doped)} cm<sup>-3</sup>
        - Intrinsic concentration: n<sub>i</sub> = {sci_fmt(n_i_Si)} cm<sup>-3</sup>

        **Verification:**
        - n<sub>0</sub>p<sub>0</sub> product: {sci_fmt(n_doped * p_doped)} cm<sup>-6</sup>
        - n<sub>i</sub><sup>2</sup>: {sci_fmt(n_i_Si**2)} cm<sup>-6</sup>

    """)
    ])

     #   **Fermi Level:**
     #   - E<sub>F</sub> = {sci_fmt(E_F_doped)} eV (from midgap)
     #   - E<sub>F</sub> - E<sub>i</sub> = {sci_fmt(E_F_doped - E_i_Si)} eV
     #   - E<sub>c</sub> - E<sub>F</sub> = {sci_fmt(E_c_Si - E_F_doped)} eV
     #   - E<sub>F</sub> - E<sub>v</sub> = {sci_fmt(E_F_doped - E_v_Si)} eV

    return


@app.cell
def _(mo):
    # Use absolute URL for WASM compatibility (image hosted on GitHub Pages)
    _base_url = "https://joyce-poon.github.io/ECE350/equilibrium"
    
    mo.vstack([
        mo.md(r"## Carrier concentrations at extreme temperatures"),
        mo.Html(f'<div style="text-align: center;"><img src="{_base_url}/carrier_vs_T.png" width="600" alt="Carrier concentration vs Temperature"><p style="font-style: italic; color: #666;">Pierret Fig. 4.18</p></div>'),
        mo.md(r"""
    - **Freeze out**: At low temperatures ($T \to 0$ K), there is insufficient thermal energy to ionize the impurities. Therefore, there are very few free carriers. 
    - **Intrinsic T-region**: At high temperatures, intrinsic carrier generation dominates over dopant ionization, leading to intrinsic-like behavior regardless of doping.
    - **Extrinsic T-region**: The equations we have derived for $n_0$, $p_0$ hold
        """)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary of Equations: Carrier Concentrations at Equilibrium

    ### Density of States (3D Bulk):

    $$\boxed{D_c(E) = \frac{8\pi m_n^*}{h^3}\sqrt{2m_n^*(E-E_c)}}$$

    $$\boxed{D_v(E) = \frac{8\pi m_p^*}{h^3}\sqrt{2m_p^*(E_v-E)}}$$

    ### Fermi-Dirac Distribution:

    $$\boxed{f(E) = \frac{1}{1 + e^{(E-E_F)/k_BT}}}$$

    ### Equilibirum Carrier Concentrations (Boltzmann Approximation):

    $$\boxed{n_0 = N_c e^{-(E_c-E_F)/k_BT}}$$

    $$\boxed{p_0 = N_v e^{-(E_F-E_v)/k_BT}}$$

    where the effective densities of states are

    $$N_c = 2\left(\frac{2\pi m_n^* k_B T}{h^2}\right)^{3/2}, \quad N_v = 2\left(\frac{2\pi m_p^* k_B T}{h^2}\right)^{3/2}$$

    $$\boxed{n_0 p_0 = n_i^2 = N_c N_v e^{-E_g/k_BT}} \quad n_i \text{ is the instrinsic carrier concentration}$$


    ### Intrinsic Fermi Level:

    $$\boxed{n_0 = n_i e^{(E_F - E_{Fi})/k_BT}, \quad p_0 = n_i e^{(E_{Fi} - E_F)/k_BT}}$$


    ### Doped Semiconductors:

    Charge neutrality:

    $$\boxed{n_0 + N_a^- = p_0 + N_d^+}$$

    Carrier concentration from doping:

    $$\boxed{n_0 = \frac{N_d^+ - N_a^-}{2} + \sqrt{\left(\frac{N_d^+ - N_a^-}{2}\right)^2 + n_i^2}} \quad \text{for } (N_d^+ \gg N_a^-, n_i): n_0 \approx N_d^+, p_0 \approx n_i^2/N_d^+$$

    $$\boxed{p_0 = \frac{N_a^- - N_d^+}{2} + \sqrt{\left(\frac{N_a^- - N_d^+}{2}\right)^2 + n_i^2}} \quad \text{for } (N_a^- \gg N_d^+, n_i): p_0 \approx N_a^-, n_0 \approx n_i^2/N_a^-$$
    """)
    return


if __name__ == "__main__":
    app.run()
