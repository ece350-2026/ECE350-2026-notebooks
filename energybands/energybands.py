import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


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
    # Energy Bands

    Lectures 6-7

    January 16-19, 2026
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Introduction

    This notebook explores the **Bloch theorem** and the 1D energy diagram using **Kronig-Penney model**.

    Reference: Pierret Ch. 3.2.1 - 3.2.3
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Consider a 1D periodic potential. $U(x)$ is the potential energy for **electrons**.

    The potential energy approaches $-\infty$ at the locations of the nucleii.
    """)
    return


@app.cell
def _(np, plt):
    # Create figure
    _fig, _ax = plt.subplots(figsize=(12, 6))

    # Parameters
    _a = 1.0  # lattice spacing
    _num_periods = 5
    _U0 = 2.0  # potential depth scale
    _x_min_offset = 0.05  # offset to avoid singularity at x=0

    # Create x array
    _x_total = np.linspace(-0.5*_a, (_num_periods - 0.5)*_a, 2000)

    # Create periodic potential with 1/|x-x_n| form
    # Include extra periods beyond visible range to make edge potentials look the same
    _U = np.zeros_like(_x_total)
    _extra_periods = 10  # Add extra atoms beyond visible range for uniform appearance

    for _n in range(-_extra_periods, _num_periods + _extra_periods):
        _x_center = _n * _a
        # Distance from center of each well
        _distance = np.abs(_x_total - _x_center)
        # Avoid singularity by setting minimum distance
        _distance = np.maximum(_distance, _x_min_offset)
        # Add attractive potential (negative 1/r)
        _U += -_U0 / _distance

    # Clip the potential for visualization (avoid too large values)
    _U_clipped = np.clip(_U, -35, 5)

    # Plot the potential
    _ax.plot(_x_total, _U_clipped, 'b-', linewidth=2.5, label='U(x)')
    _ax.fill_between(_x_total, _U_clipped, -35, alpha=0.2, color='blue')

    # Add vertical dashed lines at lattice sites
    for _n in range(_num_periods):
        _x_site = _n * _a
        _ax.axvline(x=_x_site, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        # Add atom markers
        _ax.plot(_x_site, -25, 'ro', markersize=12, markeredgecolor='darkred', 
                markeredgewidth=2, zorder=5)

    # Add horizontal line at U=0
    _ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # Add lattice spacing annotations
    # Arrow showing lattice spacing 'a'
    _arrow_y = -15
    _ax.annotate('', xy=(_a, _arrow_y), xytext=(0, _arrow_y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    _ax.text(_a/2, _arrow_y + 0.8, r'$a$', fontsize=20, ha='center')
            #bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))

    # Add another lattice spacing annotation
    _ax.annotate('', xy=(2*_a, _arrow_y), xytext=(_a, _arrow_y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    _ax.text(1.5*_a, _arrow_y + 0.8, r'$a$', fontsize=20, ha='center')
            #bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))

    # Add periodicity annotation
    _period_y = -20
    #_ax.annotate('', xy=((_num_periods-1)*_a, _period_y), xytext=(0, _period_y),
    #            arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
    _ax.text((_num_periods-1)*_a/2, _period_y + 10, 
            r'Periodic: $U(x+a) = U(x)$', fontsize=16, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))

    # Add labels for atomic positions
    _ax.text(0, -28, r'$x_0$', fontsize=14, ha='center', color='darkred', weight='bold')
    _ax.text(_a, -28, r'$x_0+a$', fontsize=14, ha='center', color='darkred', weight='bold')
    _ax.text(2*_a, -28, r'$x_0+2a$', fontsize=14, ha='center', color='darkred', weight='bold')
    _ax.text(3*_a, -28, r'$x_0+3a$', fontsize=14, ha='center', color='darkred', weight='bold')
    _ax.text(4*_a, -28, r'$x_0+4a$', fontsize=14, ha='center', color='darkred', weight='bold')

    # Add annotation showing potential form
    #_ax.text(0.15*_a, -8, r'$U(x) \propto -\frac{1}{|x-x_n|}$', fontsize=16,
    #        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))

    # Labels and title
    _ax.set_xlabel(r'Position $x [a]$', fontsize=18, weight='bold')
    _ax.set_ylabel(r'Potential Energy $U(x)$', fontsize=18, weight='bold')
    _ax.set_title(r'1D Periodic Potential with Coulomb-like Form', fontsize=20, weight='bold', pad=20)

    # Set axis limits
    _ax.set_xlim(-0.5*_a, (_num_periods - 0.5)*_a)
    _ax.set_ylim(-30, -5)

    # Grid
    _ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)

    # Tick parameters
    _ax.tick_params(labelsize=14)

    # Remove y-axis tick labels
    _ax.set_yticklabels([])

    # Legend
    _ax.legend(fontsize=14, loc='upper right')

    # Tight layout
    plt.tight_layout()

    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bloch's Theorem

    ###In 1D
    If $U(x+a) = U(x)$, where $a$ is the lattice constant, the electron wavefunction has the form

    $$\boxed{\psi(x+a) = e^{ika}\psi(x)}$$

          This is equivalent to

    $$\boxed{\psi(x) = e^{ikx}u_k(x)},$$

    where $u_k(x) = u_k(x+a)$ is a perioidic function with the same perioidicity as the lattice.

    You can check the equivalence by substitution $x$ with $(x+a)$ in the second equation:

    $$\psi(x+a) = e^{ik(x+a)}u_k(x+a) = \left[e^{ikx}u_k(x)\right]e^{ika} = e^{ika}\psi(x)$$

    ###Verify Bloch's Theorem

    Let's verify Bloch's theorem by substituting $\psi(x+a) = e^{ika}\psi(x)$ into the time-independent Schrödinger equation

    $$-\frac{\hbar^2}{2m} \frac{d^2\psi}{dx^2} + U(x) \psi = E \psi$$

    At $x^\prime = x+a$,

    $$-\frac{\hbar^2}{2m} \frac{d^2}{dx^2}\psi(x+a) + U(x+a) \psi = E \psi(x+a)$$

    So

    $$-\frac{\hbar^2}{2m} \frac{d^2}{dx^2}(\psi(x) e^{ika}) + U(x) \psi(x) e^{ika} = E \psi(x) e^{ika}$$

    $$\implies -\frac{\hbar^2}{2m} \frac{d^2}{dx^2}(\psi(x) ) + U(x) \psi(x)  = E \psi(x)$$

    Therefore, Schrödinger equation is satistied as before, and $\psi(x+a) = e^{ika}\psi(x)$ is a valid eigenstate.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualization of the Bloch Wavefunction

    - A periodic function $u_k$ is modulated by a plane wave ($e^{ikx}$)
    """)
    return


@app.cell
def _(mo):
    # Create slider for k value
    k_slider_bloch = mo.ui.slider(
        start=-1.0, stop=1.0, step=0.05, value=0.3,
        label="Wavevector k (in units of π/a)",
        show_value=True
    )
    return (k_slider_bloch,)


@app.cell
def _(go, k_slider_bloch, make_subplots, mo, np):

    # Get current k value from slider
    _k_val_bloch = k_slider_bloch.value * np.pi  # k in units of π/a
    _a_val_bloch = 1.0

    # Create x array spanning multiple lattice periods
    _num_lattice_periods = 8
    _x_bloch = np.linspace(0, _num_lattice_periods * _a_val_bloch, 1000)

    # Plane wave component: e^(ikx) - we'll show the real part
    _plane_wave = np.cos(_k_val_bloch * _x_bloch / _a_val_bloch)

    # Periodic modulation u_k(x) with lattice periodicity
    # Using Gaussians localized at each atomic site
    _sigma_bloch = 0.15 * _a_val_bloch  # Width of Gaussians
    _u_k = np.zeros_like(_x_bloch)
    for _site in range(_num_lattice_periods + 1):
        _x_center = _site * _a_val_bloch
        _u_k += np.exp(-(_x_bloch - _x_center)**2 / (2 * _sigma_bloch**2))

    # Normalize u_k to have max amplitude around 0.6
    _u_k = 0.6 * _u_k / np.max(_u_k)

    # Bloch wavefunction: ψ(x) = e^(ikx) * u_k(x)
    # Real part: Re[ψ] = u_k(x) * cos(kx)
    _psi_real = _u_k * np.cos(_k_val_bloch * _x_bloch / _a_val_bloch)
    _psi_imag = _u_k * np.sin(_k_val_bloch * _x_bloch / _a_val_bloch)
    _psi_prob = _psi_real**2 + _psi_imag**2  # |ψ|²

    # Create the figure with subplots
    fig_bloch = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Components of a Bloch Wavefunction",
            "Bloch Wavefunction ψ(x) = e<sup>ikx</sup>u<sub>k</sub>(x)",
            "Probability Density |ψ(x)|²"
        ),
        vertical_spacing=0.12,
        row_heights=[0.35, 0.35, 0.3]
    )

    # Plot 1: Components
    fig_bloch.add_trace(
        go.Scatter(x=_x_bloch, y=_plane_wave, mode='lines',
                   name='cos(kx)', line=dict(color='blue', width=2, dash='dash')),
        row=1, col=1
    )
    fig_bloch.add_trace(
        go.Scatter(x=_x_bloch, y=_u_k, mode='lines',
                   name='Periodic function u<sub>k</sub>(x)', line=dict(color='green', width=2)),
        row=1, col=1
    )
    #fig_bloch.add_trace(
    #    go.Scatter(x=_x_bloch, y=-_u_k, mode='lines',
    #               name='Envelope ±u<sub>k</sub>(x)', line=dict(color='green', width=2),
    #               showlegend=False),
    #    row=1, col=1
    #)

    # Plot 2: Bloch wavefunction
    fig_bloch.add_trace(
        go.Scatter(x=_x_bloch, y=_psi_real, mode='lines',
                   name='Re[ψ(x)]', line=dict(color='red', width=2.5)),
        row=2, col=1
    )
    fig_bloch.add_trace(
        go.Scatter(x=_x_bloch, y=_psi_imag, mode='lines',
                   name='Im[ψ(x)]', line=dict(color='orange', width=2.5, dash='dot')),
        row=2, col=1
    )


    # Plot 3: Probability density
    fig_bloch.add_trace(
        go.Scatter(x=_x_bloch, y=_psi_prob, mode='lines',
                   name='|ψ(x)|²', line=dict(color='purple', width=2.5),
                   fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.2)'),
        row=3, col=1
    )

    # Add vertical lines at lattice sites
    for _n in range(_num_lattice_periods + 1):
        _x_site = _n * _a_val_bloch
        fig_bloch.add_vline(x=_x_site, line_dash="dot", line_color="gray", 
                             line_width=1, opacity=0.5, row=1, col=1)
        fig_bloch.add_vline(x=_x_site, line_dash="dot", line_color="gray", 
                             line_width=1, opacity=0.5, row=2, col=1)
        fig_bloch.add_vline(x=_x_site, line_dash="dot", line_color="gray", 
                             line_width=1, opacity=0.5, row=3, col=1)

    # Update layout
    fig_bloch.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        title=dict(
            text=f"Bloch Wavefunction: ψ(x) = e<sup>ikx</sup>u<sub>k</sub>(x)<br>" +
                 f"<sup>k = {k_slider_bloch.value:.2f}π/a, lattice constant a = 1.0</sup>",
            x=0.5
        )
    )

    fig_bloch.update_xaxes(title_text="Lattice Position x [a]", row=3, col=1)
    fig_bloch.update_yaxes(title_text="Amplitude", row=1, col=1, range=[-1,1])
    fig_bloch.update_yaxes(title_text="ψ(x)", row=2, col=1)
    fig_bloch.update_yaxes(title_text="|ψ|²", row=3, col=1)

    mo.vstack([
        k_slider_bloch,
        mo.ui.plotly(fig_bloch)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Bloch's theorm in 3D

    For a periodic potential $U(\vec{r}+\vec{R}) = U(\vec{r})$, where $\vec{r} = (x,y,z)$ and $\vec{R} = n_1 \vec{a}_1 + n_2 \vec{a}_2 + n_3 \vec{a}_3$ is a lattice vector, the electron wavefunction is

    $$\boxed{\psi(\vec{r} + \vec{R}) = e^{i\vec{k}\cdot\vec{r}}\psi(\vec{r})}$$

    $$\boxed{\psi(\vec{r}) = e^{i\vec{k}\cdot\vec{r}}u_{\vec{k}}(\vec{r})}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Periodic Boundary Conditions

    - For crystals with a finite extent, it is common to use the periodic boundary condition. (Admittedly, this is rather artificial, but interfaces are very complicated to deal with!)

    - For a 1D finite crystal with $N$ periods, the periodic boundary condition is

    $$\boxed{\psi(x)=\psi(x+Na)}$$

    - But $\psi(x+Na) =e^{ikNa}\psi(x)$, so $\psi(x) = e^{ikNa}\psi(x)$.

    $$\therefore \boxed{k = \frac{2\pi n}{Na}}, \quad \text{where } n = 0, \pm 1, \pm 2, \ldots$$

    - So the values of $k$ are discrete.

    - Restricting $kNa \in (-\pi, +\pi]$ for the set of unique solutions,

        - If $N$ is odd, then $n=0, \pm 1, \pm 2, \ldots {N-1}/2$. Thus, the number of discrete $k$ values is $N$.

        - If $N$ is even, then $n=0, \pm 1, \pm 2, \ldots N/2$ ($n=-N/2$) leads to the same wavefunction as $n=+N/2$). The number of discrete $k$ values is also $N$.

    - As $N \rightarrow \infty$, $k$ becomes continuous.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Kronig-Penney Model

    Pierret 3.2.1

    The model consists of:
    - **Region I** (width $a$): Potential $U(x) = 0$ (well region)
    - **Region II** (width $b$): Potential $U(x) = U_0$ (barrier region)
    - **Lattice period**: $a + b$
    """)
    return


@app.cell
def _(np, plt):
    # Create figure
    _fig, _ax = plt.subplots(figsize=(10, 4))

    # Parameters
    _a = 0.7   # well width
    _b = 0.3   # barrier width
    _U0 = 5.0  # barrier height
    _num_periods = 3
    _period = _a + _b

    # Create x array
    # Start slightly before first barrier to show periodicity context
    _x_start = -_b - 0.5*_a
    _x_end = _num_periods * _period
    _x = np.linspace(_x_start, _x_end, 1000)

    # Create potential U(x)
    _U = np.zeros_like(_x)

    # Fill in barriers
    # Barriers are in range [-b, 0], [a, a+b], [2a+b, 2a+2b], etc.
    # Pattern repeats every (a+b)
    # Base barrier is at [-b, 0]
    # In general logic: if (x % period) is in [a, a+b] -> U0
    # But simpler: check periodicity relative to a standard cell
    # Let's define the cell from [-b, a] where [-b, 0] is barrier, [0, a] is well

    # Using modulo to determine region
    # Shift x by b so that 0 is start of barrier
    _x_shifted = _x + _b
    _x_mod = np.mod(_x_shifted, _period)

    # If using [-b, 0] as barrier (width b), then in modulo [0, period]:
    # Barrier is in [0, b], Well is in [b, a+b]
    _mask_barrier = (_x_mod < _b)
    _U[_mask_barrier] = _U0

    # Plot the potential
    _ax.plot(_x, _U, 'b-', linewidth=2, label='U(x)')
    _ax.fill_between(_x, _U, 0, alpha=0.3, color='blue', where=_U>0)

    # Styling
    _ax.set_title('Kronig-Penney Rectangular Potential', fontsize=16, weight='bold')
    _ax.set_xlabel('Position x [$a + b$]', fontsize=14)
    _ax.set_ylabel('Potential Energy U(x)', fontsize=14)
    _ax.set_ylim(-1, _U0 * 1.5)
    _ax.axhline(0, color='black', linewidth=1)

    # Annotations
    # Annotate Region I (Well)
    _ax.text(_a/2, _U0/2, 'Region I \nU=0', ha='center', va='center', fontsize=16)

    # Annotate Region II (Barrier)
    _ax.text(-_b/2, _U0 + 0.5, 'Region II\nU=U₀', ha='center', va='bottom', fontsize=16)

    # Width annotations
    # Width a
    _y_width = -0.25
    _ax.annotate('', xy=(0, _y_width), xytext=(_a, _y_width),
                arrowprops=dict(arrowstyle='<->', color='black'))
    _ax.text(_a/2, _y_width-0.025, 'a', ha='center', va='top', fontsize=16)

    # Width b
    _ax.annotate('', xy=(-_b, _y_width), xytext=(0, _y_width),
                arrowprops=dict(arrowstyle='<->', color='black'))
    _ax.text(-_b/2, _y_width-0.025, 'b', ha='center', va='top', fontsize=16)

    #_ax.grid(alpha=0.3)

    # Remove y-axis tick labels
    _ax.set_yticklabels([])

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Schrödinger Equation Solutions

    **Region I** ($0 < x < a$):

    $$\frac{d^2\psi_I}{dx^2} + \alpha^2\psi_I = 0, \quad \textrm{where }\alpha = \sqrt{\frac{2mE}{\hbar^2}}$$

    $$\psi_I(x) = A\sin(\alpha x) + B\cos(\alpha x)$$

    **Region II** ($-b < x < 0$):
    - If $E \geq U_0$:

    $$\psi_{II}(x) = C\sin(\beta x) + D\cos(\beta x), \qquad \textrm{where } \beta = \sqrt{\frac{2m(E-U_0)}{\hbar^2}}$$

    - If $E < U_0$:

    $$ \psi_{II}(x) = C\sinh(\gamma x) + D\cosh(\gamma x), \qquad \textrm{where } \gamma = \sqrt{\frac{2m(U_0-E)}{\hbar^2}}$$



    **Apply Bloch's theorem and boundary conditions**

    - Let $\Lambda = a+b$


    | # | Condition | Equation |
    |:---:|:-----------:|:----------:|
    | 1 | $\psi$ continuous at $x=0$ | $B = D$ |
    | 2 | $d\psi/dx$ continuous at $x=0$ | $\alpha A = \beta C$ |
    | 3 | Bloch: $\psi_I(a) = e^{ik\Lambda}\psi_{II}(-b)$ | $A\sin(\alpha a) + B\cos(\alpha a) = e^{ik\Lambda}[-C\sin(\beta b) + D\cos(\beta b)]$ |
    | 4 | Bloch: $\psi'_I(a) = e^{ik\Lambda}\psi'_{II}(-b)$ | $\alpha A\cos(\alpha a) - \alpha B\sin(\alpha a) = e^{ik\Lambda}[\beta C\cos(\beta b) + \beta D\sin(\beta b)]$ |



    - The four boundary conditions form a homogeneous linear system $\mathbf{M}\vec{c} = \vec{0}$:

    $$\begin{pmatrix} 0 & 1 & 0 & -1 \\ \alpha & 0 & -\beta & 0 \\ \sin(\alpha a) & \cos(\alpha a) & e^{ik\Lambda}\sin(\beta b) & -e^{ik\Lambda}\cos(\beta b) \\ \alpha\cos(\alpha a) & -\alpha\sin(\alpha a) & -\beta e^{ik\Lambda}\cos(\beta b) & -\beta e^{ik\Lambda}\sin(\beta b) \end{pmatrix} \begin{pmatrix} A \\ B \\ C \\ D \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 0 \end{pmatrix}$$

    **Non-trivial solutions require:**

    $$\det(\mathbf{M}) = 0$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Energy Band Diagram

    - Solving the $\det(\mathbf{M}) = 0$ leads to the **dispersion relation**:


    **For $E \geq U_0$:**

    $$\boxed{-\frac{\alpha^2 + \beta^2}{2\alpha\beta}\sin(\alpha a)\sin(\beta b) + \cos(\alpha a)\cos(\beta b) = \cos(k\Lambda)}$$

    **For $E < U_0$:** (substitute $\beta \to i\gamma$, using $\sin(ix) = i\sinh(x)$, $\cos(ix) = \cosh(x)$)

    $$\boxed{\frac{\gamma^2 - \alpha^2}{2\alpha\gamma}\sin(\alpha a)\sinh(\gamma b) + \cos(\alpha a)\cosh(\gamma b) = \cos(k\Lambda)}$$

    where $\Lambda = a + b$ is the lattice period, $\alpha = \sqrt{2mE/\hbar^2}$, $\beta = \sqrt{2m(E-U_0)/\hbar^2}$, $\gamma = \sqrt{2m(U_0-E)/\hbar^2}$

    ### Physical Interpretation

    Since $|\cos(k\Lambda)| \leq 1$, only energies where $|\text{LHS}| \leq 1$ are allowed. This creates:

    - **Allowed bands**: energies with real $k$ solutions
    - **Forbidden gaps**: energies where $|\text{LHS}| > 1$
    - **Band edges**: occur at $k = 0$ and $k = \pm\pi/\Lambda$ where $dE/dk = 0$
    - $k$ is the electron wavenumber; $\hbar k$ is also known as the crystal momentum

    ### E vs. k Diagram

    - Plot the allowed energy vs. $k (a+b)$
    - To find $k(a+b)$, solve $\text{LHS} = \cos(k(a+b))$

    ### Reduced Zone and Extended Zone Representations
    - Due to periodicity of $\cos(k(a+b))$, $k (a+b)$ is equivalent to $k (a+b) + 2\pi$
    - Extended zone representation: plot the solution for all values of $k$
    - Reduced zone representation: restrict $k$ to $[-\pi/(a+b), +\pi/(a+b)]$
        - $-\pi/(a+b) \leq k \leq +\pi/(a+b)$ is known as the first **Brillouin zone**
        - Energy band diagrams almost always use reduced zone representation
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path as _Path
    _image_path_kp = _Path(__file__).resolve().parent / "KP_LHS.png"
    _image_path_kp2 = _Path(__file__).resolve().parent / "KP_bands.png"
    _image_path_kp3 = _Path(__file__).resolve().parent / "KP_extended_reduced_zone.png"

    mo.vstack(
        [
            mo.md(
                r"""
                ## Visualizations

                Left hand side of the Kronig-Penney Dispersion Relation
                """
            ),
            mo.hstack([mo.image(str(_image_path_kp), width=500, caption="Pierret Fig. 3.3")], justify="center"),
            mo.md(
                r"""
               Energy Bands
                """
            ),
            mo.hstack([mo.image(str(_image_path_kp2), width=500, caption="Pierret Fig. 3.5")], justify="center"),
            mo.md(
                r"""
               E vs. k diagrams in reduced zone and extended zone representations, and in free-space
                """
            ),
            mo.hstack([mo.image(str(_image_path_kp3), width=600, caption="Pierret Fig. 3.6")], justify="center"),
        ],
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualization with Interactive Parameters
    """)
    return


@app.cell
def _(mo):
    # Create sliders for the combined visualization
    combined_U0_slider = mo.ui.slider(
        start=0.05, stop=8.0, step=0.1, value=3.0,
        label="Barrier Height U₀ (eV)",
        show_value=True
    )

    combined_a_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.01, value=0.4,
        label="Well Width a (nm)",
        show_value=True
    )

    combined_b_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.01, value=0.1,
        label="Barrier Width b (nm)",
        show_value=True
    )
    return combined_U0_slider, combined_a_slider, combined_b_slider


@app.cell
def _(
    combined_U0_slider,
    combined_a_slider,
    combined_b_slider,
    go,
    make_subplots,
    mo,
    np,
):
    # Helper function for dispersion calculation
    def _compute_dispersion_lhs_local(E, U0, a, b):
        """Compute the left-hand side of the Kronig-Penney dispersion relation"""
        _hbar = 1.054571817e-34
        _m_e = 9.1093837015e-31
        _eV_to_J = 1.602176634e-19

        E_J = E * _eV_to_J
        U0_J = U0 * _eV_to_J
        a_m = a * 1e-9
        b_m = b * 1e-9

        if E <= 0:
            return np.nan

        alpha = np.sqrt(2 * _m_e * E_J) / _hbar

        if E >= U0:
            beta = np.sqrt(2 * _m_e * (E_J - U0_J)) / _hbar
            if beta == 0:
                LHS = np.cos(alpha * a_m)
            else:
                term1 = -(alpha**2 + beta**2) / (2 * alpha * beta) * np.sin(alpha * a_m) * np.sin(beta * b_m)
                term2 = np.cos(alpha * a_m) * np.cos(beta * b_m)
                LHS = term1 + term2
        else:
            gamma = np.sqrt(2 * _m_e * (U0_J - E_J)) / _hbar
            term1 = (gamma**2 - alpha**2) / (2 * alpha * gamma) * np.sin(alpha * a_m) * np.sinh(gamma * b_m)
            term2 = np.cos(alpha * a_m) * np.cosh(gamma * b_m)
            LHS = term1 + term2

        return LHS

    # Helper function to calculate the E-K diagram
    def _compute_E_k_diagram_local(U0, a, b, k_points=200, E_max_factor=3):
        """
        Compute the E-k diagram by finding energies for each k value
        """
        # k values in the first Brillouin zone
        _period = (a + b) * 1e-9  # Convert to meters
        _k_max = np.pi / _period
        _k_values = np.linspace(-_k_max, _k_max, k_points)

        # Energy range to search
        E_max = 12.5
        E_search = np.linspace(0.01, E_max, 2000)

        # For each k, find energies where f(E) = cos(k*(a+b))
        E_k_solutions = []

        for k in _k_values:
            target = np.cos(k * _period)

            # Find energies where f(E) ≈ target
            f_E = np.array([_compute_dispersion_lhs_local(E, U0, a, b) for E in E_search])

            # Find crossings
            diff = f_E - target

            # Find sign changes (crossings)
            sign_changes = np.where(np.diff(np.sign(diff)))[0]

            energies_for_k = []
            for idx in sign_changes:
                # Linear interpolation for better accuracy
                if idx < len(E_search) - 1:
                    E1, E2 = E_search[idx], E_search[idx + 1]
                    f1, f2 = diff[idx], diff[idx + 1]
                    E_cross = E1 - f1 * (E2 - E1) / (f2 - f1)
                    energies_for_k.append(E_cross)

            E_k_solutions.append((k, energies_for_k))

        return _k_values, E_k_solutions

    # Helper function to calculate the extended zone E-K diagram
    def _compute_E_k_diagram_extended(U0, a, b, k_points=400, E_max_factor=3):
        """
        Compute the E-k diagram for extended zone representation (-2pi to +2pi)
        """
        # k values extended to -2pi to +2pi (normalized)
        _period = (a + b) * 1e-9  # Convert to meters
        _k_max = 2 * np.pi / _period  # Extended to 2pi
        _k_values = np.linspace(-_k_max, _k_max, k_points)

        # Energy range to search
        E_max = 12.5
        E_search = np.linspace(0.01, E_max, 2000)

        # For each k, find energies where f(E) = cos(k*(a+b))
        E_k_solutions = []

        for k in _k_values:
            target = np.cos(k * _period)

            # Find energies where f(E) ≈ target
            f_E = np.array([_compute_dispersion_lhs_local(E, U0, a, b) for E in E_search])

            # Find crossings
            diff = f_E - target

            # Find sign changes (crossings)
            sign_changes = np.where(np.diff(np.sign(diff)))[0]

            energies_for_k = []
            for idx in sign_changes:
                # Linear interpolation for better accuracy
                if idx < len(E_search) - 1:
                    E1, E2 = E_search[idx], E_search[idx + 1]
                    f1, f2 = diff[idx], diff[idx + 1]
                    E_cross = E1 - f1 * (E2 - E1) / (f2 - f1)
                    energies_for_k.append(E_cross)

            E_k_solutions.append((k, energies_for_k))

        return _k_values, E_k_solutions

    # Get current slider values
    _U0 = combined_U0_slider.value
    _a = combined_a_slider.value
    _b = combined_b_slider.value
    _energy_res = 2000

    # Create the combined figure with 4 subplots
    _fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=(
            "Periodic Potential",
            "Dispersion Function LHS",
            "E-k Diagram (Extended Zone) <br> k(a+b) ∈ [-2π, +2π]",
            "E-k Diagram (Reduced Zone) <br> k(a+b) ∈ [-π, +π]"
        ),
        horizontal_spacing=0.08,
        column_widths=[0.25, 0.25, 0.25, 0.25]
    )



    # ===== Plot 1: Periodic Potential =====
    _num_periods = 5
    _period = _a + _b
    _x_total = _num_periods * _period
    _x = np.linspace(-_b, _x_total, 2000)
    _U = np.zeros_like(_x)

    for _i in range(_num_periods + 1):
        _x_start = _i * _period - _b
        _x_end = _i * _period
        _mask = (_x >= _x_start) & (_x < _x_end)
        _U[_mask] = _U0

    _fig.add_trace(
        go.Scatter(
            x=_x, y=_U,
            mode='lines',
            name='Potential U(x)',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.2)',
            showlegend=False
        ),
        row=1, col=1,
    )

    _fig.add_annotation(
        x=_a/2, y=_U0*0.5,
        text=f"Region I<br>U=0<br>a={_a:.2f} nm",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        row=1, col=1
    )

    _fig.add_annotation(
        x= -_b/2+2*(_a+_b), y=_U0*1.15,
        text=f"Region II<br>U₀={_U0:.1f} eV<br>b={_b:.2f} nm",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        row=1, col=1
    )

    # ===== Plot 2: Dispersion Function LHS =====
    _E_max = _U0 * 3
    _E_values = np.linspace(0.01, _E_max, _energy_res)
    _LHS_values = np.array([_compute_dispersion_lhs_local(E, _U0, _a, _b) for E in _E_values])

    _fig.add_trace(
        go.Scatter(
            x=_E_values, y=_LHS_values,
            mode='lines',
            name='LHS',
            line=dict(color='blue', width=2),
            showlegend=False
        ),
        row=1, col=2
    )

    _fig.add_hline(y=1, line_dash="dash", line_color="red", row=1, col=2)
    _fig.add_hline(y=-1, line_dash="dash", line_color="red", row=1, col=2)

    # Find allowed bands
    _allowed_mask = np.abs(_LHS_values) <= 1
    _in_band = False
    _band_start = None
    _bands_found = []

    for _i, (_E, _allowed) in enumerate(zip(_E_values, _allowed_mask)):
        if _allowed and not _in_band:
            _band_start = _E
            _in_band = True
        elif not _allowed and _in_band:
            _bands_found.append((_band_start, _E_values[_i-1]))
            _in_band = False

    if _in_band:
        _bands_found.append((_band_start, _E_values[-1]))

    # Calculate bandgaps
    _bandgaps = []
    for _i in range(len(_bands_found) - 1):
        _gap_start = _bands_found[_i][1]  # End of current band
        _gap_end = _bands_found[_i + 1][0]  # Start of next band
        _gap_value = _gap_end - _gap_start
        _bandgaps.append((_gap_start, _gap_end, _gap_value))

    # Shade allowed bands
    _colors = ['rgba(0,255,0,0.2)', 'rgba(0,200,100,0.2)', 'rgba(100,200,0,0.2)', 
              'rgba(0,150,150,0.2)', 'rgba(50,200,50,0.2)']

    for _i, (_E_start, _E_end) in enumerate(_bands_found[:5]):
        _fig.add_vrect(
            x0=_E_start, x1=_E_end,
            fillcolor=_colors[_i % len(_colors)], 
            opacity=0.5,
            layer="below", 
            line_width=0,
            row=1, col=2
        )

    _fig.add_vline(x=_U0, line_dash="dot", line_color="purple", row=1, col=2)

    # ===== Plot 3: E-k Diagram (Extended Zone) =====
    _k_vals_ext, _E_k_sols_ext = _compute_E_k_diagram_extended(_U0, _a, _b, k_points=400, E_max_factor=3)

    # Organize solutions by band
    _max_bands_ext = max(len(energies) for _, energies in _E_k_sols_ext) if _E_k_sols_ext else 0

    _band_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for _band_idx in range(min(_max_bands_ext, 7)):
        _k_band_ext = []
        _E_band_ext = []

        for _k, _energies in _E_k_sols_ext:
            if _band_idx < len(_energies):
                _k_band_ext.append(_k * (_a + _b) * 1e-9)  # Normalize k by period
                _E_band_ext.append(_energies[_band_idx])

        if _k_band_ext:
            _fig.add_trace(go.Scatter(
                x=_k_band_ext, y=_E_band_ext,
                mode='lines',
                name=f'Band {_band_idx + 1}',
                line=dict(width=2, color=_band_colors[_band_idx % len(_band_colors)]),
                showlegend=False
            ), row=1, col=3)

    # Add horizontal line at U0 in extended E-k diagram
    _fig.add_hline(y=_U0, line_dash="dash", line_color="red", row=1, col=3)

    # Add vertical lines at zone boundaries for extended zone
    _fig.add_vline(x=-2*np.pi, line_dash="dot", line_color="gray", row=1, col=3)
    _fig.add_vline(x=-np.pi, line_dash="dot", line_color="gray", row=1, col=3)
    _fig.add_vline(x=0, line_dash="dot", line_color="gray", row=1, col=3)
    _fig.add_vline(x=np.pi, line_dash="dot", line_color="gray", row=1, col=3)
    _fig.add_vline(x=2*np.pi, line_dash="dot", line_color="gray", row=1, col=3)

    # ===== Plot 4: E-k Diagram (Reduced Zone) =====
    _k_vals, _E_k_sols = _compute_E_k_diagram_local(_U0, _a, _b, k_points=200, E_max_factor=3)

    # Organize solutions by band
    _max_bands = max(len(energies) for _, energies in _E_k_sols) if _E_k_sols else 0

    for _band_idx in range(min(_max_bands, 7)):
        _k_band = []
        _E_band = []

        for _k, _energies in _E_k_sols:
            if _band_idx < len(_energies):
                _k_band.append(_k * (_a + _b) * 1e-9)  # Normalize k by period
                _E_band.append(_energies[_band_idx])

        if _k_band:
            _fig.add_trace(go.Scatter(
                x=_k_band, y=_E_band,
                mode='lines',
                name=f'Band {_band_idx + 1}',
                line=dict(width=2, color=_band_colors[_band_idx % len(_band_colors)]),
                showlegend=False
            ), row=1, col=4)

    # Add horizontal line at U0 in reduced E-k diagram
    _fig.add_hline(y=_U0, line_dash="dash", line_color="red", row=1, col=4)

    # Add vertical lines at zone boundaries for reduced zone
    _fig.add_vline(x=-np.pi, line_dash="dot", line_color="gray", row=1, col=4)
    _fig.add_vline(x=np.pi, line_dash="dot", line_color="gray", row=1, col=4)


    _fig.update_xaxes(title_text="Position x (nm)", row=1, col=1)
    _fig.update_yaxes(title_text="U(x) (eV)", row=1, col=1)

    _fig.update_xaxes(title_text="Energy E (eV)", row=1, col=2)
    _fig.update_yaxes(title_text="LHS", row=1, col=2)

    _fig.update_xaxes(title_text="k(a+b)", row=1, col=3)
    _fig.update_yaxes(title_text="Energy E (eV)", range=[0,12], row=1, col=3)

    _fig.update_xaxes(title_text="k(a+b)", row=1, col=4)
    _fig.update_yaxes(title_text="Energy E (eV)", range=[0,12], row=1, col=4)

    mo.vstack([
        mo.hstack([combined_U0_slider, combined_a_slider, combined_b_slider]),
        mo.ui.plotly(_fig)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ###Explorations
    - How do the magnitude of the bandgaps change with barrier height? Does a higher barrier lead to a larger or smaller bandgap?
    - Does reducing the barrier height and/or width lead to the energy diagram consistent with free-space?
    - Are the ranges of permissible energies increasing or decreasing with increasing energy? Is that reasonable?
    - How do the energy bands change with $a$ (well width) and $b$ (barrier width)?
    """)
    return


if __name__ == "__main__":
    app.run()
