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
    return go, make_subplots, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Energy Bands - Part 2

    Lecture 8-9

    January 21 and 23, 2026

    Reference: Pierret Ch. 3.2.4 - 3.3
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Effective Mass

    - A **wavepacket** = QM analog of a classical particle localized in space
    - The wavepacket travels with the group velocity $v_g$. This is the velocity of the peak of the wavepacket.
    - From wave physics, the group velocity is $v_g = \frac{d \omega}{d k}$.
    - In quantum physics, $\omega = E / \hbar$, so we have

        $$\boxed{v_g = \frac{1}{\hbar} \frac{\partial E}{\partial k}}$$


    - Consider the momentum of a particle

        - The classical momentum $p = m v$, where $m$ is the mass and $v$ is the velocity.
        - In quantum physics (de Broglie relation), the momentum is $p = \hbar k$.

        - Therefore, adopting a semi-classical picture for a wavepacket, we have

    $$p = m \frac{d \omega}{dk} = \frac{m}{\hbar} \frac{\partial E}{\partial k} = \hbar k $$

    - From the third equality above, differentiate both sides by $k$, we arrive at

    $$\frac{m}{\hbar} \frac{\partial^2 E}{\partial k^2} = \hbar \implies m = \frac{\hbar^2}{\frac{\partial^2 E}{\partial k^2}}$$

    - In a crystal, $E$ is related to $k$ by the band dispersion relation $E(k)$.

    - We define the **effective mass** as

        $$\boxed{m^* = \frac{\hbar^2}{\frac{d^2 E}{\partial k^2}}}$$

    - The **effective mass** encapsulates how electrons respond to forces in the crystal. It allows us to use classical equations of motion for electrons in crystals!

    ### Key Properties of the Effective Mass
    - $m^* > 0$ at the **bottom** of a band (electrons)
    - $m^* < 0$ at the **top** of a band (holes)
    - Inversely proportional to band curvature
    - **Very different from the free electron mass!!**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Effective Mass Visualizations

    The conduction and valence band edges are approximately parabolic. The band edge is the minimum of the conduction band and the maximum of the valence band. Most of the carriers of interest (i.e., responsible for electrical conduction) are near the band edges.

    - **Conduction band:** $E(k) \approx E_c + \frac{\hbar^2 k^2}{2 m^*}$ → **positive** curvature → $m^* > 0$ for electrons at the bottom of the band
    - **Valence band:** $E(k) \approx E_v - \frac{\hbar^2 k^2}{2 |m^*|}$ → **negative** curvature → $m^* < 0$ for electrons at the top of the band

    Higher curvature (steeper parabola) corresponds to **smaller** effective mass.
    """)
    return


@app.cell
def _(go, make_subplots, mo, np):
    # k-axis (dimensionless, for illustration)
    _k = np.linspace(-1.0, 1.0, 400)

    # Two curvatures to illustrate effective mass
    _alpha_low = 0.5   # low curvature -> larger effective mass
    _alpha_high = 2  # high curvature -> smaller effective mass

    # Conduction band parabolas (upward)
    _Ec = 1.0
    _E_c_low = _Ec + _alpha_low * _k**2
    _E_c_high = _Ec + _alpha_high * _k**2

    # Valence band parabolas (downward)
    _Ev = 0.0
    _E_v_low = _Ev - _alpha_low * _k**2
    _E_v_high = _Ev - _alpha_high * _k**2

    _fig_eff_mass = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Low Curvature (Large m*)",
            "High Curvature (Small m*)"
        ),
        horizontal_spacing=0.12
    )

    # Low curvature
    _fig_eff_mass.add_trace(
        go.Scatter(x=_k, y=_E_c_low, mode="lines", name="Conduction (m*>0)",
                   line=dict(color="#1f77b4", width=2)),
        row=1, col=1
    )
    _fig_eff_mass.add_trace(
        go.Scatter(x=_k, y=_E_v_low, mode="lines", name="Valence (m*<0)",
                   line=dict(color="#d62728", width=2)),
        row=1, col=1
    )

    # High curvature
    _fig_eff_mass.add_trace(
        go.Scatter(x=_k, y=_E_c_high, mode="lines", showlegend=False,
                   line=dict(color="#1f77b4", width=2)),
        row=1, col=2
    )
    _fig_eff_mass.add_trace(
        go.Scatter(x=_k, y=_E_v_high, mode="lines", showlegend=False,
                   line=dict(color="#d62728", width=2)),
        row=1, col=2
    )

    # Define common axis ranges
    _x_range = [-1.0, 1.0]
    _y_range = [-2.5, 3.5]

    # Visual guides
    for _col in [1, 2]:
        _fig_eff_mass.add_hline(y=_Ev, line_dash="dot", line_color="gray", row=1, col=_col)
        _fig_eff_mass.update_xaxes(title_text="k (arb. units)", range=_x_range, row=1, col=_col)
        _fig_eff_mass.update_yaxes(title_text="Energy E (arb. units)", range=_y_range, row=1, col=_col)

    # Add annotations for m* > 0 (conduction band) and m* < 0 (valence band)
    # Low curvature plot (col 1)
    _fig_eff_mass.add_annotation(
        x=0, y=_Ec + _alpha_low * 0.6**2 + 0.3,
        text="Heavy<br>m* > 0",
        showarrow=False,
        font=dict(size=18, color="#1f77b4"),
        align="center",
        xref="x", yref="y",
        row=1, col=1
    )
    _fig_eff_mass.add_annotation(
        x=0, y=_Ev - _alpha_low * 0.6**2 - 0.3,
        text="Heavy<br>m* < 0",
        showarrow=False,
        font=dict(size=18, color="#d62728"),
        align="center",
        xref="x", yref="y",
        row=1, col=1
    )

    # High curvature plot (col 2)
    _fig_eff_mass.add_annotation(
        x=0, y=_Ec + _alpha_high * 0.5**2 + 0.3,
        text="Light<br>m* > 0",
        showarrow=False,
        font=dict(size=18, color="#1f77b4"),
        align="center",
        xref="x2", yref="y2",
        row=1, col=2
    )
    _fig_eff_mass.add_annotation(
        x=0, y=_Ev - _alpha_high * 0.5**2 - 0.3,
        text="Light<br>m* < 0",
        showarrow=False,
        font=dict(size=18, color="#d62728"),
        align="center",
        xref="x2", yref="y2",
        row=1, col=2
    )

    _fig_eff_mass.update_layout(
        height=420,
        width=900,
        showlegend=False,
        #legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=60, b=40, l=50, r=20),
    )

    mo.ui.plotly(_fig_eff_mass)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Electrons, holes, and conduction

    - Each point on the E-k diagram represents 2 electronic states (spin up and spin down)

    - Each state may be occupied with an **electron** or left empty (**hole**).

    - Suppose we have a 1D crystal with a length of $L = Na$, where $N$ is a large number and $a$ is the lattice constant.

        - Recall the periodic boundary conditions lead to the set of allowed $\{k\}$ of $k = 2\pi m/(Na)$, where $m$ is an integer and $ka \in (-\pi, +\pi]$.

        - The current conducted through $L$ by 1 electron is $i = \frac{charge}{time} = -q\frac{v_g}{L}$, where $v_g$ is the group velocity.

        - Over the energy range of a band, the net flow of current due to $N$ electrons is

    $$I = -\frac{q}{L} \sum_{i=1}^N v_{g,i}$$


    - For many common materials (including Si), the energy bands are symmetric about $k$, so $E(k) = E(-k)$ and $v_g(-k) = -v_g(k)$ (recall that $v_g = \frac{1}{\hbar}\frac{dE}{dk}$)
        - Exceptions: materials without centro-symmetric symmetry (i.e., $U(\vec{r}) \neq U(-\vec{r})$), materials under a magnetic field

    - $\therefore$ A completely empty or symmetrically filled band means there is no current flow.

        - There are as many electrons moving to the left as there are moving the right.


    - If a band is not symmetrically filled, then

    $$I = -\frac{q}{L} \sum_{filled}^N v_{g,i} = -\frac{q}{L} \left( \sum_{all}^N v_{g,i} -\sum_{empty}^N v_i \right) = +\frac{q}{L} \sum_{empty}^N v_{g,i}$$

    - The current is **the same as placing positively charged particles ($+q$) in the empty states**.  So we can describe current flow in terms of holes instead of electrons.

    - But under an applied electric field, $v_{g,i}$ of a positively-charged particle is oppositely signed compared to a negatively-charged particle, so its momentum is oppositely signed compared to an electron ($p = mv$)

    - If we flip the sign of the charge from $-q$ to $+q$ for holes, we should flip the sign of the mass so the equation of motion is consistent. Therefore, the hole effective mass is

    $$\boxed{m_p^\ast = -\frac{\hbar^2}{ \frac{\partial^2 E}{\partial k^2}}}$$

    $$ m* < 0 \text{ near the top of a band, so } m_p^\ast > 0 \text{ at the top of the valence band}$$

    - The conduction band electrons remain negatively charged, and they can use the same effective mass defition as before

    $$\boxed{m_n^\ast = \frac{\hbar^2} { \frac{\partial^2 E}{\partial k^2}}}$$

    $$ m* > 0 \text{ near the bottom of a band, so } m_n^\ast > 0 \text{ at the bottom of the conduction band}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Visualization: Filled Bands Carry No Net Current

    The diagram below illustrates why completely filled or symmetrically filled bands do not contribute to current flow. Each dot represents a filled electron state at a discrete $k$ value.

    - **Band 1**: Completely filled → For every electron at $+k$ with velocity $v_g$, there's one at $-k$ with velocity $-v_g$ → Net current = 0
    - **Band 2**: Symmetrically filled around $k=0$ → Same cancellation → Net current = 0
    - **Bands 3 & 4**: Empty or partially filled bands can carry current if filling is asymmetric
    """)
    return


@app.cell
def _(go, make_subplots, mo, np):
    # Create E-k diagram showing filled bands carry no net current

    # Define k values (discrete points in first Brillouin zone)
    _N = 20  # Number of unit cells (determines k-point density)
    _k_discrete = np.linspace(-np.pi, np.pi, _N + 1)[:-1]  # Exclude endpoint to avoid double counting
    _k_continuous = np.linspace(-np.pi, np.pi, 200)

    # Define energy bands (simple parabolic/cosine shapes for illustration)
    def _band1(k):
        return 0.5 - 0.3 * np.cos(k)  # Lowest band, centered around E=0.5

    def _band2(k):
        return 2 + 0.5 * np.cos(k)  # Second band

    def _band3(k):
        return 4 - 0.8 * np.cos(k)  # Third band (inverted curvature at center)


    # Create subplots with two columns
    _fig_bands = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Symmetric Filling (No Current)", "Asymmetric Filling (Current Flow)"),
        horizontal_spacing=0.08
    )

    # ============ LEFT PLOT: Symmetric filling (no current) ============

    # Plot continuous band curves
    _fig_bands.add_trace(go.Scatter(
        x=_k_continuous, y=_band1(_k_continuous),
        mode='lines', name='Band 1',
        line=dict(color='black', width=2),
        showlegend=False
    ), row=1, col=1)
    _fig_bands.add_trace(go.Scatter(
        x=_k_continuous, y=_band2(_k_continuous),
        mode='lines', name='Band 2',
        line=dict(color='black', width=2),
        showlegend=False
    ), row=1, col=1)
    _fig_bands.add_trace(go.Scatter(
        x=_k_continuous, y=_band3(_k_continuous),
        mode='lines', name='Band 3',
        line=dict(color='black', width=2),
        showlegend=False
    ), row=1, col=1)

    # Add filled states (dots) for Band 1 - completely filled
    _fig_bands.add_trace(go.Scatter(
        x=_k_discrete, y=_band1(_k_discrete),
        mode='markers', name='Filled (Band 1)',
        marker=dict(color='#1f77b4', size=10, symbol='circle'),
        showlegend=False
    ), row=1, col=1)

    # Add filled states for Band 2 - symmetrically filled
    _k_band2_filled = _k_discrete[np.abs(_k_discrete) > 0.5]
    _fig_bands.add_trace(go.Scatter(
        x=_k_band2_filled, y=_band2(_k_band2_filled),
        mode='markers', name='Filled (Band 2)',
        marker=dict(color='#1f77b4', size=10, symbol='circle'),
        showlegend=False
    ), row=1, col=1)
    _k_band2_empty = _k_discrete[np.abs(_k_discrete) <= 0.5]
    _fig_bands.add_trace(go.Scatter(
        x=_k_band2_empty, y=_band2(_k_band2_empty),
        mode='markers', name='Empty (Band 2)',
        marker=dict(color='white', size=10, symbol='circle', line=dict(color='#1f77b4', width=2)),
        showlegend=False
    ), row=1, col=1)

    # Add states for Band 3 - symmetrically filled
    _k_band3_filled = _k_discrete[np.abs(_k_discrete) <= 0.5]
    _fig_bands.add_trace(go.Scatter(
        x=_k_band3_filled, y=_band3(_k_band3_filled),
        mode='markers', name='Filled (Band 3)',
        marker=dict(color='#1f77b4', size=10, symbol='circle'),
        showlegend=False
    ), row=1, col=1)
    _k_band3_empty = _k_discrete[np.abs(_k_discrete) > 0.5]
    _fig_bands.add_trace(go.Scatter(
        x=_k_band3_empty, y=_band3(_k_band3_empty),
        mode='markers', name='Empty (Band 3)',
        marker=dict(color='white', size=10, symbol='circle', line=dict(color='#1f77b4', width=2)),
        showlegend=False
    ), row=1, col=1)

    # ============ RIGHT PLOT: Asymmetric filling (current flow) ============

    # Plot continuous band curves
    _fig_bands.add_trace(go.Scatter(
        x=_k_continuous, y=_band1(_k_continuous),
        mode='lines', name='Band 1',
        line=dict(color='black', width=2),
        showlegend=False
    ), row=1, col=2)
    _fig_bands.add_trace(go.Scatter(
        x=_k_continuous, y=_band2(_k_continuous),
        mode='lines', name='Band 2',
        line=dict(color='black', width=2),
        showlegend=False
    ), row=1, col=2)
    _fig_bands.add_trace(go.Scatter(
        x=_k_continuous, y=_band3(_k_continuous),
        mode='lines', name='Band 3',
        line=dict(color='black', width=2),
        showlegend=False
    ), row=1, col=2)

    # Add filled states (dots) for Band 1 - completely filled
    _fig_bands.add_trace(go.Scatter(
        x=_k_discrete, y=_band1(_k_discrete),
        mode='markers', name='Filled (Band 1)',
        marker=dict(color='#1f77b4', size=10, symbol='circle'),
        showlegend=False
    ), row=1, col=2)

    # Add filled states for Band 2 - asymmetrically filled (shifted to positive k)
    # Leave one more state empty for k>0 (changed upper bound from 1.2 to 1.5)
    _k_band2_filled_asym = _k_discrete[(_k_discrete < -0.5) | (_k_discrete > 1.5)]  # More states at positive k
    _fig_bands.add_trace(go.Scatter(
        x=_k_band2_filled_asym, y=_band2(_k_band2_filled_asym),
        mode='markers', name='Filled (Band 2)',
        marker=dict(color='#1f77b4', size=10, symbol='circle'),
        showlegend=False
    ), row=1, col=2)
    _k_band2_empty_asym = _k_discrete[(_k_discrete >= -0.5) & (_k_discrete <= 1.5)]  # One more empty state
    _fig_bands.add_trace(go.Scatter(
        x=_k_band2_empty_asym, y=_band2(_k_band2_empty_asym),
        mode='markers', name='Empty (Band 2)',
        marker=dict(color='white', size=10, symbol='circle', line=dict(color='#1f77b4', width=2)),
        showlegend=False
    ), row=1, col=2)

    # Add states for Band 3 - asymmetrically filled (shifted to positive k)
    # Fill one more state for k>0 (changed upper bound from 1.2 to 1.5)
    _k_band3_filled_asym = _k_discrete[(_k_discrete >= -0.5) & (_k_discrete <= 1.5)]  # Asymmetric filling
    _fig_bands.add_trace(go.Scatter(
        x=_k_band3_filled_asym, y=_band3(_k_band3_filled_asym),
        mode='markers', name='Filled (Band 3)',
        marker=dict(color='#1f77b4', size=10, symbol='circle'),
        showlegend=False
    ), row=1, col=2)
    _k_band3_empty_asym = _k_discrete[((_k_discrete < -0.5) | (_k_discrete > 1.5))]
    _fig_bands.add_trace(go.Scatter(
        x=_k_band3_empty_asym, y=_band3(_k_band3_empty_asym),
        mode='markers', name='Empty (Band 3)',
        marker=dict(color='white', size=10, symbol='circle', line=dict(color='#1f77b4', width=2)),
        showlegend=False
    ), row=1, col=2)
    # Add annotations for left plot
    _fig_bands.add_annotation(
        x=3.5, y=0.7,
        text="Filled",
        showarrow=False,
        font=dict(size=12, color='black'),
        xanchor='left',
        yshift=-15,
        xref="x", yref="y"
    )
    _fig_bands.add_annotation(
        x=3.5, y=2,
        text="Sym. filled",
        showarrow=False,
        font=dict(size=12, color='black'),
        xanchor='left',
        yshift=-15,
        xref="x", yref="y"
    )
    _fig_bands.add_annotation(
        x=3.5, y=3.7,
        text="Sym. filled",
        showarrow=False,
        font=dict(size=12, color='black'),
        xanchor='left',
        yshift=-15,
        xref="x", yref="y"
    )

    # Add annotations for right plot
    _fig_bands.add_annotation(
        x=3.5, y=0.7,
        text="Filled",
        showarrow=False,
        font=dict(size=12, color='black'),
        xanchor='left',
        yshift=-15,
        xref="x2", yref="y2"
    )
    _fig_bands.add_annotation(
        x=3.5, y=2,
        text="Asym. filled",
        showarrow=False,
        font=dict(size=12, color='#d62728'),
        xanchor='left',
        yshift=-15,
        xref="x2", yref="y2"
    )
    _fig_bands.add_annotation(
        x=3.5, y=3.7,
        text="Asym. filled",
        showarrow=False,
        font=dict(size=12, color='#d62728'),
        xanchor='left',
        yshift=-15,
        xref="x2", yref="y2"
    )

    # Add current flow arrow on right plot
    _fig_bands.add_annotation(
        x=-1, y=5.5,
        ax=-80, ay=0,
        text="Net current <0",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='#d62728',
        font=dict(size=14, color='#d62728'),
        xanchor='left',
        xref="x2", yref="y2"
    )

    # Add "E" labels
    _fig_bands.add_annotation(
        x=0, y=6,
        text="E",
        showarrow=False,
        font=dict(size=16, color='black'),
        xanchor='right',
        yanchor='bottom',
        xshift=-10,
        xref="x", yref="y"
    )
    _fig_bands.add_annotation(
        x=0, y=6,
        text="E",
        showarrow=False,
        font=dict(size=16, color='black'),
        xanchor='right',
        yanchor='bottom',
        xshift=-10,
        xref="x2", yref="y2"
    )

    # Update layout for both subplots
    _fig_bands.update_layout(
        xaxis=dict(
            title="k",
            tickvals=[-np.pi, 0, np.pi],
            ticktext=["-π/a", "0", "π/a"],
            range=[-4, 4],
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor='black',
        ),
        xaxis2=dict(
            title="k",
            tickvals=[-np.pi, 0, np.pi],
            ticktext=["-π/a", "0", "π/a"],
            range=[-4, 4],
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor='black',
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            zeroline=False,
            range=[-0.25, 6.5],
        ),
        yaxis2=dict(
            title="",
            showticklabels=False,
            zeroline=False,
            range=[-0.25, 6.5],
        ),
        showlegend=False,
        height=500,
        width=1000,
        plot_bgcolor='white',
        margin=dict(l=50, r=100, t=50, b=50),
    )

    # Add vertical lines at Brillouin zone boundaries for both plots
    _fig_bands.add_vline(x=-np.pi, line_width=1, line_color="grey", line_dash="dash", row=1, col=1)
    _fig_bands.add_vline(x=np.pi, line_width=1, line_color="grey", line_dash="dash", row=1, col=1)
    _fig_bands.add_vline(x=-np.pi, line_width=1, line_color="grey", line_dash="dash", row=1, col=2)
    _fig_bands.add_vline(x=np.pi, line_width=1, line_color="grey", line_dash="dash", row=1, col=2)

    # Add horizontal axis lines
    _fig_bands.add_hline(y=0, line_width=1, line_color="black", row=1, col=1)
    _fig_bands.add_hline(y=0, line_width=1, line_color="black", row=1, col=2)

    mo.vstack([
        mo.ui.plotly(_fig_bands),
        mo.md(r"""
        **Left:** In a symmetrically filled or fully filled band, contributions to current from $+k$ and $-k$ states **exactly cancel**. Thus, there is no current.

        **Right:** An applied external (electric) field leads to asymmetrically filled bands. There are more conduction band electrons with $k>0$ (and positive group velocity) than $k<0$. There are also more holes with $k>0$ (and negative group velocity) than holes with $k<0$. So the current flow is negative.
        """)])
    return


@app.cell
def _(mo):
    # Use absolute URLs for WASM compatibility (images hosted on GitHub Pages)
    _base_url = "https://joyce-poon.github.io/ECE350/effectivemass"

    mo.vstack(
        [
            mo.md(
                r"""
                ## 3D Band diagrams
                - Energy bands depend on the **3D wavevector** $\vec{k} = (k_x, k_y, k_z)$.
                - Energy diagrams are plotted along major crystal directions.
                - Here are the $E$ vs. $k$ diagrams for **Germainum**, **Silicon**, and **Gallium Arsenide**:
                """
            ),
            mo.Html(f'<div style="text-align: center;"><img src="{_base_url}/energy_diagrams.png" width="700" alt="E vs k diagrams"><p style="font-style: italic; color: #666;">Pierret, Fig. 3.13</p></div>'),
            mo.md(
                r"""
                **Crystal Symmetry Points:**
                - $\Gamma$ (Gamma): Zone center $(0,0,0)$
                - $X$: Center of square faces ($k$ along $\langle 100 \rangle$)
                - $L$: Center of hexagonal faces ($k$ along $\langle 111 \rangle$)
                - $K$: Midpoint of hexagon edges ($k$ along $\langle 110 \rangle$)
                """
            ),
            mo.md(
                r"""
                **Constant energy surfaces** for (a) Ge, (b) Si, and (c) GaAs. (d) Truncated in the Brillouin zone.
                """
            ),

            mo.Html(f'<div style="text-align: center;"><img src="{_base_url}/constant_energy_surfaces.png" width="700" alt="Constant energy surfaces"><p style="font-style: italic; color: #666;">Pierret, Fig. 3.14</p></div>'),

            mo.md(r"""
    **Effective Masses and Bandgaps of Selected Semiconductors (at room temperature):**

    | Material | $m_n^*$  | $m_p^*$ | $E_g$ |
    |:----------:|:------------------------:|:---------------------:|:---------------------:|
    | **Germanium (Ge)** | $0.12 m_0$ | $0.28 m_0$  | 0.67 eV |
    | **Silicon (Si)** | $0.26  m_0$ | $0.39 m_0$   | 1.12 eV |
    | **Gallium Arsenide (GaAs)** | $0.068 m_0$ | $0.5 m_0$ | 1.4 eV |
    | **Indium Phosphide (InP)** | $0.025 m_0$ | $0.45 m_0$ | 1.35 eV |
    | **Gallium Nitride (GaN)** | $0.19 m_0$ | $0.32 m_0$ | 3.4 eV |

    Note: $m_0 = 9.109 \times 10^{-31}$ kg is the free electron mass.
    """)
        ],
    )
    return


@app.cell
def _(mo):
    # Use absolute URLs for WASM compatibility
    _base_url = "https://joyce-poon.github.io/ECE350/effectivemass"

    mo.vstack([
        mo.md(r"""
    ## Direct vs. Indirect Bandgap Semiconductors

    - **Direct bandgap**: The valence band maximum and conduction band minimum occur at the **same k-value** (typically at $\Gamma$ point, $k=0$)
    - **Indirect bandgap**: The valence band maximum and conduction band minimum occur at **different k-values**

    ### Optical Transitions

    - Consider an optically induced transition (photon absorption or emission):

    $$k_{\text{final,electron}} = k_{\text{initial,electron}} \pm k_{\text{photon}}$$

    - Since $\lambda_{\text{optical}} \gg \lambda_{\text{electron}}$, $k_{\text{photon}} \ll k_{\text{electron}}$.

    $$\therefore \boxed{\text{optical transitions: } k_{\text{final,electron}} = k_{\text{initial,electron}}}$$

    - **Direct bandgap**: A photon can directly excite an electron from valence to conduction band if $E_{\text{photon}} \geq E_g$

    - **Indirect bandgap**: Momentum conservation requires a **phonon** (lattice vibration) to provide the momentum difference. A transition between conduction and valence bands requires 2 interactions (photon absorption/emission + phonon), this transition has a lower probability of occurring compared to transition in a direct bandgap semiconductor. 
        - Indirect semiconductors are less efficient in emitting and absorbing light than direct bandgap semiconductors.

    $$k_{\text{final,electron}} \approx k_{\text{initial,electron}} + k_{\text{phonon}}$$

       """),
        mo.Html(f'<div style="text-align: center;"><img src="{_base_url}/direct_indirect.png" width="700" alt="Direct vs Indirect Bandgap"></div>'),
        mo.md(r"""
    ### Examples

    - The photon energy required for absorption is:

    $$E_{\text{photon}} = \frac{hc}{\lambda}$$

    where $h$ is Planck's constant, $c$ is the speed of light, $\lambda$ is the wavelength of light.

    Light is absorbed when $E_{\text{photon}} > E_g$

    | Material | Bandgap (eV) | Wavelength (nm) | Type |
    |:--------:|:------------:|:---------------:|:----:|
    | **Si** | 1.12 | ~1100 | Indirect |
    | **Ge** | 0.67 | ~1850 | Indirect |
    | **GaAs** | 1.4 | ~875 | Direct |
    | **GaP** | 2.25 | ~550 | Indirect |
    | **GaN** | 3.4 | ~365 | Direct |

    - **Impact**: Direct bandgap materials are used in optoelectronics (LEDs, lasers). The most efficient solar cells are made with direct semiconductors.
        - [National Renewable Energy Lab (NREL) solar cell efficiency records](https://www.nrel.gov/pv/interactive-cell-efficiency)

      """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    ### Bloch Wavefunctions
    - The Bloch wavefunction $\psi(x) = e^{ikx}u_k(x)$ consists of
        - **Plane wave** $e^{ikx}$: Describes propagation through the crystal
        - **Periodic function** $u_k(x)$: Arises from the lattice periodicity

    ### Energy Bands and Gaps
    - **Allowed bands**: Electrons can propagate freely (real $k$)
    - **Band gaps**: No propagating states exist (complex $k$)

    #### Effect of Parameters:
    - Larger $U_0$: Wider band gaps, stronger confinement
    - Larger $a$, larger $b$: More bands in a given energy range
    - Higher energy bands resemble free-electron dispersion relation

    ### Effective Mass
    - Encapsulates crystal forces
    - For quasi-classical description
    - Different for electrons and holes
    """)
    return


if __name__ == "__main__":
    app.run()
