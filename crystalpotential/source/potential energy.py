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
    return go, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Potential Energy Visualizations in 2D and 3D

    Assuming a potential of a Coulombic form $U(r) \propto - \frac{1}{r}$, the potential energy for the electrons is lowest near the nuclei.

    In practice, the potential energy is lowest at the atomic radius $r_0$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2D Periodic Potential

    For a 2D lattice, the potential profile, $U(x,y)$, looks like hills and valleys.

    Below is an example potential for a hexagonal lattice. The red spheres show the positions of the atoms.
    """)
    return


@app.cell
def _(go, np):
    # Parameters
    _a_hex = 1.0  # lattice spacing
    _U0_hex = 2.0  # potential depth scale
    _min_distance = 0.05 # offset to avoid singularity

    # Number of unit cells to calculate (extended for periodic boundary effects)
    _n_cells_total = 15  

    # Number of unit cells to display (smaller to avoid edge effects)
    _n_cells_display = 15  # Only show 4x4 region

    # Create hexagonal lattice points
    # Hexagonal lattice basis vectors
    _a1_hex = np.array([_a_hex, 0])
    _a2_hex = np.array([_a_hex/2, _a_hex*np.sqrt(3)/2])

    # Generate ALL lattice sites (15x15) for potential calculation
    _lattice_sites_all = []
    for _i in range(_n_cells_total):
        for _j in range(_n_cells_total):
            _site = _i * _a1_hex + _j * _a2_hex
            _lattice_sites_all.append(_site)

    _lattice_sites_all = np.array(_lattice_sites_all)

    # Generate VISIBLE lattice sites for display (filtered by display range)
    _x_min_display = 8
    _x_max_display = 12
    _y_min_display = 3
    _y_max_display = 7

    _lattice_sites_visible = []
    for _i in range(_n_cells_total):
        for _j in range(_n_cells_total):
            _site = _i * _a1_hex + _j * _a2_hex
            # Only include sites within the visible region
            if (_x_min_display <= _site[0] <= _x_max_display and 
                _y_min_display <= _site[1] <= _y_max_display):
                _lattice_sites_visible.append(_site)

    _lattice_sites_visible = np.array(_lattice_sites_visible)

    # Create mesh grid for potential calculation - only for visible region
    _x_range_hex = np.linspace(_x_min_display, _x_max_display, 200)
    _y_range_hex = np.linspace(_y_min_display, _y_max_display, 200)
    _X_hex, _Y_hex = np.meshgrid(_x_range_hex, _y_range_hex)

    # Calculate potential at each point using NEAREST NEIGHBOR only
    # First, compute distances to all lattice sites and find minimum
    _distances_all = np.zeros((_X_hex.shape[0], _X_hex.shape[1], len(_lattice_sites_all)))
    for _idx, _site in enumerate(_lattice_sites_all):
        _distances_all[:, :, _idx] = np.sqrt((_X_hex - _site[0])**2 + (_Y_hex - _site[1])**2)

    # Find the minimum distance to any lattice site (nearest neighbor)
    _min_dist_to_atom = np.min(_distances_all, axis=2)

    # Avoid singularity
    _min_dist_to_atom = np.maximum(_min_dist_to_atom, _min_distance)

    # Calculate potential using only nearest neighbor contribution
    _U_hex = -_U0_hex*0.25 / _min_dist_to_atom

    # Clip potential to the range [-100, -30]
    _clip_min = -10
    _clip_max = 0
    _U_clipped = np.clip(_U_hex, _clip_min, _clip_max)

    # Create interactive 3D plot with Plotly
    _fig_hex = go.Figure()

    # Plot surface
    _fig_hex.add_trace(go.Surface(
        x=_X_hex,
        y=_Y_hex,
        z=_U_clipped,
        colorscale='Viridis_r',
        cmin=_clip_min,
        cmax=_clip_max,
        opacity=0.9,
        showscale=True,
        colorbar=dict(
            title=dict(text='Potential Energy U(x,y)', font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        name='Potential'
    ))

    # Mark ONLY visible lattice sites - RED CIRCLES
    _atom_z = 0 # Place atoms at U = -10
    if len(_lattice_sites_visible) > 0:
        _fig_hex.add_trace(go.Scatter3d(
            x=_lattice_sites_visible[:, 0],
            y=_lattice_sites_visible[:, 1],
            z=[_atom_z] * len(_lattice_sites_visible),
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                line=dict(color='darkred', width=2)
            ),
            name='Atoms',
            showlegend=False
        ))

        # Draw vertical lines from atoms down to bottom
        for _site in _lattice_sites_visible:
            _fig_hex.add_trace(go.Scatter3d(
                x=[_site[0], _site[0]],
                y=[_site[1], _site[1]],
                z=[_atom_z, _clip_min],
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                showlegend=False
            ))


    # Update layout
    _fig_hex.update_layout(
        title=dict(
            text='2D Hexagonal Lattice with a Coulombic potential',
            font=dict(size=20),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(title='x [a]', range=[_x_min_display, _x_max_display]),
            yaxis=dict(title='y [a]', range=[_y_min_display, _y_max_display]),
            zaxis=dict(title='Potential U(x,y)', range=[_clip_min, _clip_max]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        width=900,
        height=700
    )

    _fig_hex
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3D Potential Visualization

    In 3D, each point in space would have an associated value of potential energy.

    In the plot below, we visualize the equipotential surfaces of a FCC lattice. Use the slider below to set the value of the potential energy to be displayed in the unit cell. The red spheres are atoms. The regions with the same potential are surfaces.
    """)
    return


@app.cell
def _(mo):
    # Create slider for equipotential value
    potential_slider = mo.ui.slider(
        start=-20.0,
        stop=-3,
        step=0.1,
        value=-10.0,
        label="Equipotential Value U (arb. units)",
        show_value=True
    )
    potential_slider
    return (potential_slider,)


@app.cell
def _(go, np, potential_slider):
    # FCC Lattice Parameters
    _a_fcc = 1.0  # lattice constant
    _U0_fcc = 1.0  # potential depth scale
    _min_distance = 0.05  # offset to avoid singularity

    # Number of unit cells in each direction for display
    _n_cells = 1

    # Generate FCC lattice sites for the displayed unit cell
    # FCC has 4 atoms per conventional unit cell:
    # (0,0,0), (a/2,a/2,0), (a/2,0,a/2), (0,a/2,a/2)
    _fcc_basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * _a_fcc

    # Generate lattice sites for display - include all visible atoms
    # This includes corner atoms and face-center atoms that complete the unit cell visually
    _lattice_sites_display = []

    # Add all atoms from neighboring cells that are on the boundary of our display region
    for _i in range(-1, _n_cells + 2):
        for _j in range(-1, _n_cells + 2):
            for _k in range(-1, _n_cells + 2):
                _origin = np.array([_i, _j, _k]) * _a_fcc
                for _basis_atom in _fcc_basis:
                    _site = _origin + _basis_atom
                    # Include atoms that are within or on the boundary of the unit cell
                    if (0 <= _site[0] <= _n_cells * _a_fcc and 
                        0 <= _site[1] <= _n_cells * _a_fcc and 
                        0 <= _site[2] <= _n_cells * _a_fcc):
                        # Check if this site is already added (avoid duplicates)
                        is_duplicate = False
                        for _existing in _lattice_sites_display:
                            if np.allclose(_site, _existing, atol=1e-6):
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            _lattice_sites_display.append(_site)

    _lattice_sites_display = np.array(_lattice_sites_display)

    # Generate extended lattice sites (including neighbors) for potential calculation
    # Include MORE neighboring unit cells to ensure all equivalent sites see the same environment
    _n_extend = 2  # increased from 2 to 5 to ensure proper symmetry
    _lattice_sites_extended = []
    for _i in range(-_n_extend, _n_cells + _n_extend):
        for _j in range(-_n_extend, _n_cells + _n_extend):
            for _k in range(-_n_extend, _n_cells + _n_extend):
                _origin = np.array([_i, _j, _k]) * _a_fcc
                for _basis_atom in _fcc_basis:
                    _site = _origin + _basis_atom
                    _lattice_sites_extended.append(_site)

    _lattice_sites_extended = np.array(_lattice_sites_extended)

    # Create 3D grid for potential calculation
    _n_points = 50  # grid resolution (increased for better surface quality)
    _x_range_fcc = np.linspace(0, _n_cells * _a_fcc, _n_points)
    _y_range_fcc = np.linspace(0, _n_cells * _a_fcc, _n_points)
    _z_range_fcc = np.linspace(0, _n_cells * _a_fcc, _n_points)

    _X_fcc, _Y_fcc, _Z_fcc = np.meshgrid(_x_range_fcc, _y_range_fcc, _z_range_fcc, indexing='ij')

    # Flatten grid points for vectorized distance calculation
    _grid_points = np.stack([_X_fcc.flatten(), _Y_fcc.flatten(), _Z_fcc.flatten()], axis=1)

    # Calculate potential using NEAREST NEIGHBOR only
    # Compute distances to all extended lattice sites and find minimum for each grid point
    _distances_all = np.zeros((len(_grid_points), len(_lattice_sites_extended)))
    for _idx, _site in enumerate(_lattice_sites_extended):
        _distances_all[:, _idx] = np.sqrt(np.sum((_grid_points - _site)**2, axis=1))

    # Find the minimum distance to any lattice site (nearest neighbor)
    _min_dist_to_atom = np.min(_distances_all, axis=1)

    # Avoid singularity
    _min_dist_to_atom = np.maximum(_min_dist_to_atom, _min_distance)

    # Calculate potential using only nearest neighbor contribution
    _U_fcc = (-_U0_fcc / _min_dist_to_atom).reshape(_X_fcc.shape)

    # Get equipotential value from slider
    _iso_value = potential_slider.value

    # Create figure
    _fig_fcc = go.Figure()

    # Add isosurface for equipotential using Volume trace with isosurface
    _fig_fcc.add_trace(go.Isosurface(
        x=_X_fcc.flatten(),
        y=_Y_fcc.flatten(),
        z=_Z_fcc.flatten(),
        value=_U_fcc.flatten(),
        isomin=_iso_value,
        isomax=_iso_value,
        surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale=[[0, 'rgba(100, 150, 255, 0.7)'], [1, 'rgba(100, 150, 255, 0.7)']],
        opacity=0.7,
        showscale=False,
        name=f'U = {_iso_value:.1f}'
    ))


    # Add lattice sites as red spheres (all visible atoms in the unit cell)
    _fig_fcc.add_trace(go.Scatter3d(
        x=_lattice_sites_display[:, 0],
        y=_lattice_sites_display[:, 1],
        z=_lattice_sites_display[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            line=dict(color='darkred', width=2)
        ),
        name='Atoms',
        showlegend= False
    ))

    # Draw unit cell edges for reference
    # Unit cell corners
    _corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ]) * _a_fcc

    # Edges of the unit cell
    _edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]

    for _edge in _edges:
        _fig_fcc.add_trace(go.Scatter3d(
            x=[_corners[_edge[0], 0], _corners[_edge[1], 0]],
            y=[_corners[_edge[0], 1], _corners[_edge[1], 1]],
            z=[_corners[_edge[0], 2], _corners[_edge[1], 2]],
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            showlegend=False
        ))

    # Update layout
    _fig_fcc.update_layout(
        title=dict(
            text=f'FCC Lattice with Equipotential Surface (U = {_iso_value:.2f})',
    #        text=f'FCC Lattice Equipotential Surface (U = {_iso_value:.2f}) - Nearest Neighbor Only<br><sup>Potential range in grid: [{_U_fcc.min():.1f}, {_U_fcc.max():.1f}] | {len(_lattice_sites_display)} atoms displayed</sup>',
            font=dict(size=20),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(title='x [a]', range=[0, _n_cells * _a_fcc]),
            yaxis=dict(title='y [a]', range=[0, _n_cells * _a_fcc]),
            zaxis=dict(title='z [a]', range=[0, _n_cells * _a_fcc]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectmode='cube'
        ),
        width=900,
        height=750,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    _fig_fcc
    return


if __name__ == "__main__":
    app.run()
