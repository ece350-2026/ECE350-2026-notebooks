import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/lecture02_03_crystal_basics_v2.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from pathlib import Path
    from ipywidgets import interact, IntSlider, FloatSlider, Dropdown

    # Get the directory where images are stored
    IMAGE_DIR = Path(__file__).parent / "lecture02_03_images"
    return IMAGE_DIR, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Introduction to Crystals
    **ECE350 Lectures 2-3**

    **January 7-9, 2026**

    **References:**
    - Pierret, *Advanced Semiconductor Fundamentals*, Ch.1
    - Hu, Ch.1
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Motivation

    Electronic devices are made in materials classified as **semiconductors**. Many semiconductors are **crystals**. The crystal structure and orientation of a device with respect to the crystal have a significant impact on performance.
    """)
    return


@app.cell
def _(IMAGE_DIR, mo):
    mo.vstack([
        mo.md("""
    ## 2. Semiconductors
    The variable resistivity of semiconductors are used to create functional devices.

    """),
        mo.hstack([
            mo.md("""
    | Material Type | Resistivity |
    |:------------------:|:--------------:|
    | **Insulators** | ρ > 10⁸ Ω·cm |
    | **Conductors** | ρ < 10⁻³ Ω·cm |
    | **Semiconductors** | **controllable** <br> (e.g., by doping, temperature, light) |
    """),
            mo.image(src=IMAGE_DIR / "resistivity_chart.jpg", width="70%")
        ], justify="start", align="center", gap=2)
    ], align="start")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. What is a Crystal?

    - A **crystal** is an infinite periodic arrangement of atoms or group of atoms in space
    - A crystal has **discrete translational invariance**
    - Many important electronic materials (including silicon) are crystalline
    - Material properties depend on crystal structure and vary with the direction and plane
    """)
    return


@app.cell
def _(IMAGE_DIR, mo):
    mo.vstack([
        mo.md("""
    | Type | Description | Example | Applications |
    |:----:|:-----------:|:-------:|:------------:|
    | **Crystalline** | Periodic crystal structure | Single-crystal Si | Most semiconductor devices |
    | **Polycrystalline** | Crystal domains, short-range order | Poly-Si | Transistor gates, resistors |
    | **Amorphous** | No crystal structure | a-Si | Low-cost solar cells |
    """),
        mo.hstack([
            mo.vstack([
                mo.md("**Crystalline**"),
                mo.image(src=str(IMAGE_DIR / "crystalline-Si.jpg"), width=200),
            ], align="center"),
            mo.vstack([
                mo.md("**Polycrystalline**"),
                mo.image(src=str(IMAGE_DIR / "poly-Si.jpg"), width=200),
            ], align="center"),
            mo.vstack([
                mo.md("**Amorphous**"),
                mo.image(src=str(IMAGE_DIR / "a-Si.jpg"), width=200),
            ], align="center"),
        ], justify="space-around"),
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    ## 4. Crystal Definitions
    """),
        mo.vstack([
            mo.md(r"""
    | Term | Definition |
    |:------:|:------------:|
    | **Lattice** | A set of points in space <br> $\vec{R} = u_1 \vec{a}_1 + u_2 \vec{a}_2 + u_3 \vec{a}_3$, <br> where $\{u_i\}$ are integers, $\{\vec{a}_i\}$ are vectors |
    | **Basis** | The group of atoms at each lattice point <br> $\vec{b}_j = v_{j,1} \vec{a}_1 + v_{j,2} \vec{a}_2 + v_{j,3} \vec{a}_3$, <br>where $\{\vec{b}_j\}$ is the position of each atom in the group, <br>$v_{j,i}$ is a real number|
    | **Unit cell** | The volume/area that is repeated to fill up all space |
    | **Primitive cell** | The smallest unit cell, defined by "primitive lattice vectors" |
    | **Lattice constant** | The periodicity of the lattice |
    """),
            mo.md(r"""
    $$\boxed{\textbf{Lattice} + \textbf{Basis}  = \textbf{Crystal Structure}}$$
    """),
        ], align="center"),
    ])
    return


@app.cell
def _(IMAGE_DIR, mo):
    mo.vstack([
       mo.hstack([
            mo.vstack([
                mo.md("**2D Square Lattice**"),
                mo.image(src=IMAGE_DIR / "lattice_2Dsquare.jpg", width="100%"),
            ], align="start"),
            mo.vstack([
                mo.md("**1-Atom Basis**"),
                mo.image(src=IMAGE_DIR / "basis_1atom.jpg", width="100%"),
            ], align="start"),
            mo.vstack([
                mo.md("**2-Atom Basis**"),
                mo.image(src=IMAGE_DIR / "basis_2atom.jpg", width="100%"),
            ], align="start"),
            mo.vstack([
                mo.md("**Unit Cell**"),
                mo.image(src=IMAGE_DIR / "unitcell.jpg", width="100%"),
            ], align="start"),
        ], align="start", widths="equal", justify="space-between"),

        mo.md("**3D Primitive Vectors**"),
        mo.image(src=IMAGE_DIR / "bcc_primitive_vectors.jpg", width=300)
    ], align="start")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.1 Crystal Structure Key Features

    1. There is only **one lattice point per primitive cell** but lattice points and edges are shared among adjacent cells
    2. The primitive basis is the minimal set of atoms needed to generate the full crystal
    3. The primitive unit cell is not unique!
    4. The **number of atoms** in a primitive cell is the same

    ### 4.2 Wigner-Seitz Method (to choose a primitive cell)

    1. Draw lines to connect a given lattice point to all nearby lattice points
    2. At the midpoint and normal to the lines, draw new lines (2D) or planes (3D)
    3. The smallest enclosed area/volume is the Wigner-Seitz cell
    """)
    return


@app.cell
def _():
    #mo.vstack([
    #    mo.md(f"""
    ### 4.3 Example: Graphene (2D Lattice)
    #- Lattice constant: **a = 246 pm**
    #- Basis: 2 carbon atoms per unit cell
    #- Structure: Honeycomb lattice (hexagonal)

    #**Questions:**
    #1. Identify the lattice points
    #1. What is the basis?
    #1. Find a primitive cell
    #1. Find primitive vectors

    #"""),
    #    mo.image(src=IMAGE_DIR / "graphene_3d.jpg", width=300)
    #])
    return


@app.cell
def _():
    # mo.md(r"""
    # *Graphene lattice showing:*
    # - **Red dots**: Lattice points
    # - **Blue circles**: Atoms
    # - **Green circle**: Primitive cell (Wigner-Seitz)
    # - **Green arrows**: Primitive lattice vectors ($\vec{a}_1$, $\vec{a}_2$) at 60° and 120° angles
    # - **Black oval**: Primitive basis (2 atoms)
    # """)
    return


@app.cell
def _():
    # Graphene annotated with primitive cell, lattice points, etc.
    #mo.image(src=IMAGE_DIR / "graphene_annotated.png", width=800)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. 3D Crystal Lattices

    There are **14 Bravais lattices**. Some common ones:

    ### 5.1 Simple Cubic (SC)
    - Atoms at corners of cube
    - 1 atom per cubic unit cell

    ### 5.2 Body-Centered Cubic (BCC)
    - Atoms at corners + 1 atom at center
    - 2 atoms per cubic unit cell

    ### 5.3 Face-Centered Cubic (FCC)
    - Atoms at corners + atoms at face centers
    - 4 atoms per cubic unit cell
    """)
    return


@app.cell
def _(np):
    # 3D visualization of cubic lattices using plotly for interactivity
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create subplots
    _fig3d = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Simple Cubic (SC)', 'Body-Centered Cubic (BCC)', 'Face-Centered Cubic (FCC)'),
        horizontal_spacing=0.05
    )

    # Corner atoms for all structures
    _corners = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], 
                        [1,1,0], [1,0,1], [0,1,1], [1,1,1]])

    # Cube edges
    _edges = [
        ([0,1], [0,0], [0,0]), ([0,0], [0,1], [0,0]), ([0,0], [0,0], [0,1]),
        ([1,1], [0,1], [0,0]), ([1,1], [0,0], [0,1]), ([0,1], [1,1], [0,0]),
        ([0,0], [1,1], [0,1]), ([0,1], [0,0], [1,1]), ([0,0], [0,1], [1,1]),
        ([1,1], [1,1], [0,1]), ([1,1], [0,1], [1,1]), ([0,1], [1,1], [1,1])
    ]

    # Arrow colors
    _arrow_colors = ['red', 'green', 'orange']

    # Function to add arrow (line + cone) for primitive vectors
    def _add_arrow(fig, start, end, color, name, row, col):
        # Add line for the arrow shaft
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                mode='lines', line=dict(color=color, width=6),
                name=name, showlegend=False
            ),
            row=row, col=col
        )
        # Add cone for the arrow head
        direction = np.array(end) - np.array(start)
        fig.add_trace(
            go.Cone(
                x=[end[0]], y=[end[1]], z=[end[2]],
                u=[direction[0]*0.3], v=[direction[1]*0.3], w=[direction[2]*0.3],
                colorscale=[[0, color], [1, color]],
                showscale=False,
                sizemode='absolute',
                sizeref=0.15,
                name=name, showlegend=False
            ),
            row=row, col=col
        )

    # Simple Cubic - primitive vectors are the conventional unit cell vectors
    _sc_vectors = [
        ([0,0,0], [1,0,0], 'a₁'),
        ([0,0,0], [0,1,0], 'a₂'),
        ([0,0,0], [0,0,1], 'a₃')
    ]

    _fig3d.add_trace(
        go.Scatter3d(x=_corners[:,0], y=_corners[:,1], z=_corners[:,2],
                     mode='markers', marker=dict(size=10, color='blue'),
                     name='Corner atoms', showlegend=False),
        row=1, col=1
    )
    for _edge in _edges:
        _fig3d.add_trace(
            go.Scatter3d(x=_edge[0], y=_edge[1], z=_edge[2],
                         mode='lines', line=dict(color='gray', width=2),
                         showlegend=False),
            row=1, col=1
        )
    # Add SC primitive vectors
    for _i, (_start, _end, _label) in enumerate(_sc_vectors):
        _add_arrow(_fig3d, _start, _end, _arrow_colors[_i], _label, 1, 1)

    # BCC - primitive vectors from origin
    # a1 = (1/2)(a, a, -a) -> to corner via center, but commonly shown as:
    # a1 = (1, 0, 0), a2 = (0, 1, 0), a3 = (1/2, 1/2, 1/2) for conventional
    # Or true primitive: a1 = (-0.5, 0.5, 0.5), a2 = (0.5, -0.5, 0.5), a3 = (0.5, 0.5, -0.5)
    _bcc_vectors = [
        ([0,0,0], [0.5, 0.5, 0.5], 'a₁'),   # to body center
        ([0,0,0], [1, 0, 0], 'a₂'),
        ([0,0,0], [0, 1, 0], 'a₃')
    ]
    # True primitive vectors for BCC (from origin)
    _bcc_primitive = [
        ([0,0,0], [-0.5, 0.5, 0.5], 'a₁'),
        ([0,0,0], [0.5, -0.5, 0.5], 'a₂'),
        ([0,0,0], [0.5, 0.5, -0.5], 'a₃')
    ]

    _fig3d.add_trace(
        go.Scatter3d(x=_corners[:,0], y=_corners[:,1], z=_corners[:,2],
                     mode='markers', marker=dict(size=10, color='blue'),
                     name='Corner atoms', showlegend=False),
        row=1, col=2
    )
    _fig3d.add_trace(
        go.Scatter3d(x=[0.5], y=[0.5], z=[0.5],
                     mode='markers', marker=dict(size=12, color='red'),
                     name='Center atom', showlegend=False),
        row=1, col=2
    )
    for _edge in _edges:
        _fig3d.add_trace(
            go.Scatter3d(x=_edge[0], y=_edge[1], z=_edge[2],
                         mode='lines', line=dict(color='gray', width=2),
                         showlegend=False),
            row=1, col=2
        )
    # Add BCC primitive vectors (true primitive from center atom)
    _bcc_true_primitive = [
        ([0.5, 0.5, 0.5], [1, 1, 0], 'a₁'),
        ([0.5, 0.5, 0.5], [1, 0, 1], 'a₂'),
        ([0.5, 0.5, 0.5], [0, 1, 1], 'a₃')
    ]
    for _i, (_start, _end, _label) in enumerate(_bcc_true_primitive):
        _add_arrow(_fig3d, _start, _end, _arrow_colors[_i], _label, 1, 2)

    # FCC - primitive vectors
    # a1 = (0.5, 0.5, 0), a2 = (0.5, 0, 0.5), a3 = (0, 0.5, 0.5)
    _fcc_vectors = [
        ([0,0,0], [0.5, 0.5, 0], 'a₁'),
        ([0,0,0], [0.5, 0, 0.5], 'a₂'),
        ([0,0,0], [0, 0.5, 0.5], 'a₃')
    ]

    _face_centers = np.array([[0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5],
                             [0.5,0.5,1], [0.5,1,0.5], [1,0.5,0.5]])
    _fig3d.add_trace(
        go.Scatter3d(x=_corners[:,0], y=_corners[:,1], z=_corners[:,2],
                     mode='markers', marker=dict(size=10, color='blue'),
                     name='Corner atoms', showlegend=False),
        row=1, col=3
    )
    _fig3d.add_trace(
        go.Scatter3d(x=_face_centers[:,0], y=_face_centers[:,1], z=_face_centers[:,2],
                     mode='markers', marker=dict(size=10, color='green'),
                     name='Face center atoms', showlegend=False),
        row=1, col=3
    )
    for _edge in _edges:
        _fig3d.add_trace(
            go.Scatter3d(x=_edge[0], y=_edge[1], z=_edge[2],
                         mode='lines', line=dict(color='gray', width=2),
                         showlegend=False),
            row=1, col=3
        )
    # Add FCC primitive vectors
    for _i, (_start, _end, _label) in enumerate(_fcc_vectors):
        _add_arrow(_fig3d, _start, _end, _arrow_colors[_i], _label, 1, 3)

    # Add legend for primitive vectors using lines (arrows) instead of markers
    _fig3d.add_trace(
        go.Scatter3d(x=[None, None], y=[None, None], z=[None, None], mode='lines',
                     line=dict(width=3, color='red'), name=r'$\vec{a}_1$', showlegend=True),
        row=1, col=1
    )
    _fig3d.add_trace(
        go.Scatter3d(x=[None, None], y=[None, None], z=[None, None], mode='lines',
                     line=dict(width=3, color='green'), name=r'$\vec{a}_2$', showlegend=True),
        row=1, col=1
    )
    _fig3d.add_trace(
        go.Scatter3d(x=[None, None], y=[None, None], z=[None, None], mode='lines',
                     line=dict(width=3, color='orange'), name=r'$\vec{a}_3$', showlegend=True),
        row=1, col=1
    )

    # Update layout
    _fig3d.update_layout(
        height=500,
        width=1200,
        title_text="Interactive 3D Cubic Lattices with Primitive Vectors (click and drag to rotate)",
        legend=dict(
            yanchor="top",
            y= -0.2,
            xanchor="left",
            x=0.0,
            orientation="h",
        ),
        margin=dict(b=80),
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        scene2=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        scene3=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
    )

    _fig3d
    return (go,)


@app.cell
def _(IMAGE_DIR, mo):
    mo.vstack([
        mo.md(r"""
    ### 5.4 Diamond
    """),
        mo.hstack([
            mo.md(r"""
    - Diamond lattice is:
      - FCC with a **2-atom basis** at positions:
        - $\vec{b}_1 = (0, 0, 0)$
        - $\vec{b}_2 = (\frac{1}{4}, \frac{1}{4}, \frac{1}{4})$

    - Diamond is **two interlaced FCC lattices**!
    - **Silicon and Germanium** (Group IV) have a diamond crystal structure.
    """),
            mo.image(src=IMAGE_DIR / "diamond_structure.jpg", width="50%")
        ], justify="start",  gap=2)
    ])
    return


@app.cell
def _(np):
    def _():
        # Interactive Diamond Lattice Visualization - 8 Unit Cells (2x2x2)
        import plotly.graph_objects as go

        # First FCC lattice (blue) - corners and face centers for one unit cell
        _fcc1_corners_base = np.array([
            [0,0,0], [1,0,0], [0,1,0], [0,0,1],
            [1,1,0], [1,0,1], [0,1,1], [1,1,1]
        ])

        _fcc1_faces_base = np.array([
            [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.5, 0.5, 1], [0.5, 1, 0.5], [1, 0.5, 0.5]
        ])

        # Second FCC lattice offset
        _offset = np.array([0.25, 0.25, 0.25])

        # Generate 8 unit cells (2x2x2 arrangement)
        _fcc1_all_list = []
        _fcc2_all_list = []

        for _ix in range(2):
            for _iy in range(2):
                for _iz in range(2):
                    _shift = np.array([_ix, _iy, _iz])
                    _fcc1_all_list.append(_fcc1_corners_base + _shift)
                    _fcc1_all_list.append(_fcc1_faces_base + _shift)
                    _fcc2_all_list.append(_fcc1_corners_base + _offset + _shift)
                    _fcc2_all_list.append(_fcc1_faces_base + _offset + _shift)

        _fcc1_all = np.vstack(_fcc1_all_list)
        _fcc2_all = np.vstack(_fcc2_all_list)

        # Remove duplicate atoms (atoms at shared boundaries)
        def _unique_atoms(atoms, tol=0.01):
            unique = [atoms[0]]
            for atom in atoms[1:]:
                is_duplicate = False
                for u in unique:
                    if np.linalg.norm(atom - u) < tol:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique.append(atom)
            return np.array(unique)

        _fcc1_unique = _unique_atoms(_fcc1_all)

        # Filter atoms that are within or close to the 2x2x2 unit cell region
        def _filter_atoms(atoms, margin=0.01, x_max=2, y_max=2, z_max=2):
            mask = np.all((atoms >= -margin) & (atoms <= np.array([x_max, y_max, z_max]) + margin), axis=1)
            return atoms[mask]

        _fcc2_filtered = _filter_atoms(_fcc2_all)
        _fcc2_unique = _unique_atoms(_fcc2_filtered)

        # Create the figure
        _fig_diamond = go.Figure()

        # Add first FCC lattice (blue) - trace index 0
        _fig_diamond.add_trace(go.Scatter3d(
            x=_fcc1_unique[:, 0], y=_fcc1_unique[:, 1], z=_fcc1_unique[:, 2],
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.9),
            name='FCC Lattice 1 (blue)',
            visible=True
        ))

        # Add second FCC lattice (red) - trace index 1
        _fig_diamond.add_trace(go.Scatter3d(
            x=_fcc2_unique[:, 0], y=_fcc2_unique[:, 1], z=_fcc2_unique[:, 2],
            mode='markers',
            marker=dict(size=10, color='red', opacity=0.9),
            name='FCC Lattice 2 (red, offset by a/4)',
            visible=True
        ))

        # Add tetrahedral bonds between nearest neighbors
        # Bond length in diamond is sqrt(3)/4 * a ≈ 0.433a
        _bond_length = np.sqrt(3) / 4 + 0.01  # with small tolerance

        # Store bond trace indices for toggling
        _bond_trace_start = 2  # Bonds start after the two atom traces

        # Find all bonds between FCC1 and FCC2 atoms
        for _p1 in _fcc2_unique:
            for _p2 in _fcc1_unique:
                _dist = np.linalg.norm(_p1 - _p2)
                if _dist < _bond_length:
                    _fig_diamond.add_trace(go.Scatter3d(
                        x=[_p1[0], _p2[0]], y=[_p1[1], _p2[1]], z=[_p1[2], _p2[2]],
                        mode='lines',
                        line=dict(color='gray', width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

        _bond_trace_end = len(_fig_diamond.data)  # Number of traces after adding bonds
        _num_bond_traces = _bond_trace_end - _bond_trace_start

        # Add cube edges for all 8 unit cells
        for _ix in range(2):
            for _iy in range(2):
                for _iz in range(2):
                    _cube_edges = [
                        ([0+_ix, 1+_ix], [0+_iy, 0+_iy], [0+_iz, 0+_iz]), 
                        ([0+_ix, 0+_ix], [0+_iy, 1+_iy], [0+_iz, 0+_iz]), 
                        ([0+_ix, 0+_ix], [0+_iy, 0+_iy], [0+_iz, 1+_iz]),
                        ([1+_ix, 1+_ix], [0+_iy, 1+_iy], [0+_iz, 0+_iz]), 
                        ([1+_ix, 1+_ix], [0+_iy, 0+_iy], [0+_iz, 1+_iz]), 
                        ([0+_ix, 1+_ix], [1+_iy, 1+_iy], [0+_iz, 0+_iz]),
                        ([0+_ix, 0+_ix], [1+_iy, 1+_iy], [0+_iz, 1+_iz]), 
                        ([0+_ix, 1+_ix], [0+_iy, 0+_iy], [1+_iz, 1+_iz]), 
                        ([0+_ix, 0+_ix], [0+_iy, 1+_iy], [1+_iz, 1+_iz]),
                        ([1+_ix, 1+_ix], [1+_iy, 1+_iy], [0+_iz, 1+_iz]), 
                        ([1+_ix, 1+_ix], [0+_iy, 1+_iy], [1+_iz, 1+_iz]), 
                        ([0+_ix, 1+_ix], [1+_iy, 1+_iy], [1+_iz, 1+_iz])
                    ]

                    for _edge in _cube_edges:
                        _fig_diamond.add_trace(go.Scatter3d(
                            x=_edge[0], y=_edge[1], z=_edge[2],
                            mode='lines',
                            line=dict(color='black', width=1.5, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

        _total_traces = len(_fig_diamond.data)
        _num_edge_traces = _total_traces - _bond_trace_end

        # Create visibility arrays for buttons
        # All visible
        _vis_all = [True] * _total_traces

        # Only FCC1 (blue) - hide red atoms and bonds
        _vis_fcc1_only = [True] + [False] + [False] * _num_bond_traces + [True] * _num_edge_traces

        # Only FCC2 (red) - hide blue atoms and bonds
        _vis_fcc2_only = [False] + [True] + [False] * _num_bond_traces + [True] * _num_edge_traces

        # Both lattices, no bonds
        _vis_no_bonds = [True, True] + [False] * _num_bond_traces + [True] * _num_edge_traces

        # Update layout with buttons below the visualization
        _fig_diamond.update_layout(
            title=dict(
                text='Diamond Lattice: 8 Unit Cells (2×2×2)<br><sub>Two Interpenetrating FCC Lattices | Click and drag to rotate | Scroll to zoom</sub>',
                x=0.5
            ),
            scene=dict(
                xaxis_title='x [a]',
                yaxis_title='y [a]',
                zaxis_title='z [a]',
                aspectmode='data',
                camera=dict(eye=dict(x=2.0, y=2.0, z=1.5)),
                xaxis=dict(range=[-0.2, 2.3]),
                yaxis=dict(range=[-0.2, 2.3]),
                zaxis=dict(range=[-0.2, 2.3]),
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                itemclick="toggle",
                itemdoubleclick="toggleothers"
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.5,
                    y=-0.05,
                    xanchor="center",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Show Both Lattices",
                            method="update",
                            args=[{"visible": _vis_all}]
                        ),
                        dict(
                            label="FCC 1 Only (Blue)",
                            method="update",
                            args=[{"visible": _vis_fcc1_only}]
                        ),
                        dict(
                            label="FCC 2 Only (Red)",
                            method="update",
                            args=[{"visible": _vis_fcc2_only}]
                        ),
                        dict(
                            label="Both (No Bonds)",
                            method="update",
                            args=[{"visible": _vis_no_bonds}]
                        ),
                    ]
                )
            ],
            annotations=[
                dict(
                    text="Toggle Lattices:",
                    x=0.15,
                    y=-0.05,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12)
                )
            ],
            height=750,
            width=900,
            margin=dict(l=0, r=0, t=80, b=80)
        )
        return _fig_diamond


    _()
    return


@app.cell
def _(IMAGE_DIR, mo):
    mo.vstack([
        mo.md(r"""
    ### 5.5 Zinc Blende 
    """),
        mo.hstack([
            mo.md(r"""
    - Similar to diamond
    - One atom of the basis is **Group III** (e.g., Ga, In) and the other is **Group V** (e.g., As, P)

    - **Examples:** GaAs, InP

    - III-V compounds are important for **optoelectronics** and **high-speed electronics**.
    """),
            mo.image(src=IMAGE_DIR / "zinc_blende_structure.jpg", width="40%")
        ], justify="start", gap=2)
    ])
    return


@app.cell
def _(IMAGE_DIR, mo):
    mo.vstack([
        mo.md(rf"""
    ## 6. Elemental and Compound Semiconductors

    - Elemental semiconductors are made of 1 element
        - e.g., Group IV (Si, Ge, C)
    - Compound semiconductors are made of more than 1 element 
        - e.g., III-V (GaAs, InP, GaN, Al$_x$Ga$_{{1-x}}$As)
    """),
        mo.image(src=IMAGE_DIR / "periodictable.jpg", width="70%")
    ])
    return


@app.cell
def _(IMAGE_DIR, mo):
    mo.vstack([
        mo.md(r"""
    ### 6.1 Crystal Structure & Atomic Bonding
    - **Covalent bond**: A bond between 2 atoms is formed by sharing a pair of electrons
    - **Co-ordination number**: Number of atoms directly bonded to an atom

    - Examples
        - Si (atomic number 14, Group IV) has 4 outermost electrons available for bonding
            - Forms covalent bonds with 4 other Si atoms
            - In the diamond lattice, each atom has 4 nearest neighbours. Coordination number = 4
        - Indium (Group III) has 3 valence electrons bond with Phosphorus (Group V) with 5 valence electrons
    """),
        mo.image(src=IMAGE_DIR / "diamond_coordinates.jpg", width="60%")
    ])
    return


@app.cell
def _(IMAGE_DIR, mo):
    # Common semiconductor materials
    mo.image(src=IMAGE_DIR / "semiconductormaterials.jpg", width=800)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Density

    $$\text{Density} = \frac{\text{Number of atoms/cm}^3 \times \text{Atomic weight (g/mol)}}{\text{Avogadro's number}}$$

    where Avogadro's number = $6.02 \times 10^{23}$ atoms/mol
    """)
    return


@app.cell
def _(mo):
    # Interactive density calculator
    def _calculate_density(lattice_constant_nm, atoms_per_cell, atomic_weight):
        """Calculate density of a crystalline material."""
        a_cm = lattice_constant_nm * 1e-7  # nm to cm
        volume_cm3 = a_cm ** 3
        atoms_per_cm3 = atoms_per_cell / volume_cm3
        avogadro = 6.022e23
        density = atoms_per_cm3 * atomic_weight / avogadro
        return atoms_per_cm3, density

    # Material database
    _materials = {
        "Silicon (Si) - Diamond": {"lattice_constant": 0.543, "atoms_per_cell": 8, "atomic_weight": 28.086},
        "Germanium (Ge) - Diamond": {"lattice_constant": 0.566, "atoms_per_cell": 8, "atomic_weight": 72.63},
        "Carbon (C) - Diamond": {"lattice_constant": 0.357, "atoms_per_cell": 8, "atomic_weight": 12.011},
        "Gold (Au) - FCC": {"lattice_constant": 0.408, "atoms_per_cell": 4, "atomic_weight": 196.967},
    }

    # Create a table showing all material properties including calculated density
    _material_table = """
    | Material - Lattice | Lattice Constant (nm) | Atoms per Unit Cell | Atomic Weight (g/mol) | Density (g/cm³) |
    |:---------|:---------------------:|:--------------:|:---------------------:|:---------------:|
    """

    for _mat_name, _mat_props in _materials.items():
        _atoms_per_cm3, _density = _calculate_density(
            _mat_props['lattice_constant'],
            _mat_props['atoms_per_cell'],
            _mat_props['atomic_weight']
        )
        _material_table += f"| {_mat_name} | {_mat_props['lattice_constant']} | {_mat_props['atoms_per_cell']} | {_mat_props['atomic_weight']} | {_density:.2f} |\n"

    mo.vstack([
        mo.md("### 7.1 Examples"),
        mo.md(_material_table),
    ])
    return


@app.cell
def _(IMAGE_DIR, mo):
    mo.vstack([
        mo.md(r"""
    ## 8. Miller Indices

    A system for labeling **planes** and **directions** in crystals.

    ### 8.1 Recipe for Planes
    """),
        mo.hstack([
            mo.md(r"""
    1. Set up axes along unit cell
    2. Identify intercepts in terms of lattice constant: e.g., $(a, 3a, 2a)$
    3. Take reciprocals: $(1, \frac{1}{3}, \frac{1}{2})$
    4. Multiply by lowest common multiple to form integers: $(6, 2, 3)$
    5. Enclose in round brackets: **(623)**
    """),
            mo.image(src=IMAGE_DIR / "miller_plane_623.jpg", width=300)
        ], justify="start", align="center", gap=2)
    ], align="start")
    return


@app.cell
def _(IMAGE_DIR, mo):
    mo.vstack([
        mo.md(r"""
    ### 8.2 Special Cases

    - **No intercept**: Take intercept at infinity. Therefore, reciprocal = 0
      - E.g., plane parallel to z-axis: $(hk0)$
    - **Negative values**: Use overbar notation: $(\bar{1}00)$
    - **Family of equivalent planes**: Use curly brackets
        - E.g., in a cubic lattice (100), (010), (001), $(\bar{1}00)$, $(0\bar{1}0)$, $(00\bar{1})$ planes are equivalent 
        - This is the  $\{100\}$ family of planes
    """),
        mo.hstack([mo.md("**Examples of planes in cubic crystals**")]),
        mo.image(src=IMAGE_DIR / "miller_planes_100_110_111.jpg", width="60%")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 8.3 Miller Indices for Directions

    - Direction: $\vec{r} = h\vec{a}_1 + k\vec{a}_2 + l\vec{a}_3$
    - Notation: **[hkl]** (square brackets)
    - Equivalent directions: **⟨hkl⟩** (angle brackets)

    - **USEFUL:** In cubic crystals, the direction **[hkl]** is **perpendicular** to the plane **(hkl)**.
    """)
    return


@app.cell
def _(np, plt):
    fig, ax = plt.subplots(figsize=(4, 4))

    # Origin
    origin = np.array([0.15, 0.15])

    # Axis endpoints
    x_end = np.array([0.85, 0.15])
    y_end = np.array([0.75, 0.45])
    z_end = np.array([0.15, 0.85])

    # Intercept points on axes
    x_int = origin + 0.5 * (x_end - origin)  # m/h on x
    y_int = origin + 0.7 * (y_end - origin)  # m/k on y
    z_int = origin + 0.6 * (z_end - origin)  # m/l on z

    # Draw axes (black arrows)
    arrow_props = dict(head_width=0.025, head_length=0.02, fc='black', ec='black', linewidth=1.5)
    ax.arrow(origin[0], origin[1], (x_end - origin)[0]*0.95, (x_end - origin)[1], **arrow_props)
    ax.arrow(origin[0], origin[1], (y_end - origin)[0]*0.95, (y_end - origin)[1]*0.95, **arrow_props)
    ax.arrow(origin[0], origin[1], (z_end - origin)[0], (z_end - origin)[1]*0.95, **arrow_props)

    # Axis labels
    ax.text(x_end[0] + 0.03, x_end[1], r'$x$', fontsize=14, ha='left', va='center')
    ax.text(y_end[0] + 0.03, y_end[1] + 0.02, r'$y$', fontsize=14, ha='left', va='center')
    ax.text(z_end[0], z_end[1] + 0.05, r'$z$', fontsize=14, ha='center', va='bottom')

    # Draw the triangular plane (filled)
    triangle = plt.Polygon([x_int, y_int, z_int], fill=True, facecolor='lightgray', 
                           edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(triangle)

    # Draw red vectors v1 and v2 on the plane
    # v1: from z_int to x_int
    ax.annotate('', xy=x_int, xytext=z_int,
                arrowprops=dict(arrowstyle='->', color='#C44536', lw=2.5))
    # v2: from y_int to x_int  
    ax.annotate('', xy=x_int, xytext=y_int,
                arrowprops=dict(arrowstyle='->', color='#C44536', lw=2.5))

    # Intercept labels
    ax.text(x_int[0], x_int[1] - 0.06, r'$\frac{m}{h}$', fontsize=13, ha='center', va='top')
    ax.text(y_int[0] + 0.05, y_int[1], r'$\frac{m}{k}$', fontsize=13, ha='left', va='center')
    ax.text(z_int[0] - 0.05, z_int[1], r'$\frac{m}{l}$', fontsize=13, ha='right', va='center')

    # Vector labels
    v1_mid = 0.5 * (z_int + x_int)
    v2_mid = 0.5 * (y_int + x_int)
    ax.text(v1_mid[0] - 0.02, v1_mid[1] + 0.06, r'$\vec{v}_1 = \left(\frac{m}{h}\vec{a}_1 - \frac{m}{l}\vec{a}_3\right)$', 
            fontsize=10, ha='center', va='bottom')
    ax.text(v2_mid[0] + 0.12, v2_mid[1] - 0.03, r'$\vec{v}_2 = \left(\frac{m}{h}\vec{a}_1 - \frac{m}{k}\vec{a}_2\right)$', 
            fontsize=10, ha='left', va='top')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    miller_proof_fig = fig
    return (miller_proof_fig,)


@app.cell
def _(miller_proof_fig, mo):
    mo.hstack([
        mo.md(r"""
    ### 8.4 Proof: $(hkl)$ is normal to $[hkl]$

    1. Find intercepts of the plane $(hkl)$
       - $m$ is an integer
       - Intercepts are at $\frac{m}{h}$, $\frac{m}{k}$, $\frac{m}{l}$ along the $x$, $y$, $z$ axes

    2. Identify 2 vectors on this plane — Any vector can be expressed as a linear superposition of two vectors: $\vec{v}_1 = \left(\frac{m}{h}\vec{a}_1 - \frac{m}{l}\vec{a}_3\right)$ and $\vec{v}_2 = \left(\frac{m}{h}\vec{a}_1 - \frac{m}{k}\vec{a}_2\right)$

    3. An arbitrary vector on $(hkl)$ is $\vec{s} = n_1\vec{v}_1 + n_2\vec{v}_2$

    4. Dot product with direction $[hkl]$: $\vec{s} \cdot (h\vec{a}_1 + k\vec{a}_2 + l\vec{a}_3) = 0$

    5. Since any vector on $(hkl)$ is perpendicular to $[hkl]$, the plane $(hkl)$ is normal to $[hkl]$.
    """),
        miller_proof_fig
    ], justify="space-between", align="center", widths=[0.6, 0.4])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 8.5 Useful Formulas

    -  Angle between two directions $[x_1 \, y_1 \, z_1]$ and $[x_2 \, y_2 \, z_2]$:

    $$\cos(\alpha) = \frac{x_1 x_2 + y_1 y_2 + z_1 z_2}{\sqrt{x_1^2 + y_1^2 + z_1^2} \cdot \sqrt{x_2^2 + y_2^2 + z_2^2}}$$

    - Example: Angle between [100] and [110] is 45°


    - Distance between adjacent  (hkl) planes in a cubic lattice with lattice constant $a$:

    $$d_{(hkl)} = \frac{a}{\sqrt{h^2 + k^2 + l^2}}$$

    - Examples:
        - Distance between (100) planes: $a$
        - Distance between (110) planes: $a/\sqrt{2}$
        - Distance between (111) planes: $a/\sqrt{3}$
    """)
    return


@app.cell
def _(go, mo, np):
    def plot_miller_plane(h, k, l):
        """Plot the Miller indices plane in a unit cell by showing intercepts."""
        fig_miller = go.Figure()

        # Draw unit cell edges
        cube_edges = [
            ([0, 1], [0, 0], [0, 0]), ([0, 0], [0, 1], [0, 0]), ([0, 0], [0, 0], [0, 1]),
            ([1, 1], [0, 1], [0, 0]), ([1, 1], [0, 0], [0, 1]), ([0, 1], [1, 1], [0, 0]),
            ([0, 0], [1, 1], [0, 1]), ([0, 1], [0, 0], [1, 1]), ([0, 0], [0, 1], [1, 1]),
            ([1, 1], [1, 1], [0, 1]), ([1, 1], [0, 1], [1, 1]), ([0, 1], [1, 1], [1, 1])
        ]

        for edge in cube_edges:
            fig_miller.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines', line=dict(color='black', width=2),
                showlegend=False, hoverinfo='skip'
            ))

        # Draw solid x, y, z axes
        fig_miller.add_trace(go.Scatter3d(
            x=[-2, 2], y=[0, 0], z=[0, 0],
            mode='lines', line=dict(color='red', width=4),
            showlegend=False, hoverinfo='skip'
        ))
        fig_miller.add_trace(go.Scatter3d(
            x=[0, 0], y=[-2, 2], z=[0, 0],
            mode='lines', line=dict(color='green', width=4),
            showlegend=False, hoverinfo='skip'
        ))
        fig_miller.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-2, 2],
            mode='lines', line=dict(color='blue', width=4),
            showlegend=False, hoverinfo='skip'
        ))

        # Add axes labels
        fig_miller.add_trace(go.Scatter3d(
            x=[2.2], y=[0], z=[0], mode='text', text=['x'], 
            textfont=dict(size=14, color='red'), showlegend=False
        ))
        fig_miller.add_trace(go.Scatter3d(
            x=[0], y=[2.2], z=[0], mode='text', text=['y'],
            textfont=dict(size=14, color='green'), showlegend=False
        ))
        fig_miller.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[2.2], mode='text', text=['z'],
            textfont=dict(size=14, color='blue'), showlegend=False
        ))

        if h == 0 and k == 0 and l == 0:
            fig_miller.add_annotation(
                text="Invalid: (0,0,0) is not a valid Miller index",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='red')
            )
        else:
            # Calculate intercepts: intercept = 1/index (if index != 0)
            x_int = 1.0 / h if h != 0 else None
            y_int = 1.0 / k if k != 0 else None
            z_int = 1.0 / l if l != 0 else None

            # Collect valid intercept points for the plane
            intercept_points = []

            if x_int is not None and -2 <= x_int <= 2:
                fig_miller.add_trace(go.Scatter3d(
                    x=[x_int], y=[0], z=[0], mode='markers',
                    marker=dict(size=10, color='red'), name=f'x-intercept: {x_int:.2f}a'
                ))
                intercept_points.append([x_int, 0, 0])
            if y_int is not None and -2 <= y_int <= 2:
                fig_miller.add_trace(go.Scatter3d(
                    x=[0], y=[y_int], z=[0], mode='markers',
                    marker=dict(size=10, color='green'), name=f'y-intercept: {y_int:.2f}a'
                ))
                intercept_points.append([0, y_int, 0])
            if z_int is not None and -2 <= z_int <= 2:
                fig_miller.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[z_int], mode='markers',
                    marker=dict(size=10, color='blue'), name=f'z-intercept: {z_int:.2f}a'
                ))
                intercept_points.append([0, 0, z_int])

            # Draw the plane by connecting intercepts
            if len(intercept_points) == 3:
                # Three intercepts: draw a triangular plane using Mesh3d
                pts = np.array(intercept_points)
                fig_miller.add_trace(go.Mesh3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    i=[0], j=[1], k=[2],
                    opacity=0.6,
                    color='cyan',
                    name=f'({h}{k}{l}) plane',
                    showlegend=True
                ))
                # Draw edges of the triangle
                for i in range(3):
                    p1 = intercept_points[i]
                    p2 = intercept_points[(i + 1) % 3]
                    fig_miller.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode='lines', line=dict(color='darkblue', width=4),
                        showlegend=False, hoverinfo='skip'
                    ))
            elif len(intercept_points) == 2:
                # Two intercepts (one index is 0): plane is parallel to one axis
                # Draw a rectangular plane extending in the direction of the zero index
                p1, p2 = intercept_points

                # Determine which axis has zero index (plane extends infinitely along it)
                if h == 0:
                    # Plane parallel to x-axis, extend in x direction
                    vertices = [
                        [p1[0] - 2, p1[1], p1[2]],
                        [p1[0] + 2, p1[1], p1[2]],
                        [p2[0] + 2, p2[1], p2[2]],
                        [p2[0] - 2, p2[1], p2[2]]
                    ]
                elif k == 0:
                    # Plane parallel to y-axis, extend in y direction
                    vertices = [
                        [p1[0], p1[1] - 2, p1[2]],
                        [p1[0], p1[1] + 2, p1[2]],
                        [p2[0], p2[1] + 2, p2[2]],
                        [p2[0], p2[1] - 2, p2[2]]
                    ]
                else:  # l == 0
                    # Plane parallel to z-axis, extend in z direction
                    vertices = [
                        [p1[0], p1[1], p1[2] - 2],
                        [p1[0], p1[1], p1[2] + 2],
                        [p2[0], p2[1], p2[2] + 2],
                        [p2[0], p2[1], p2[2] - 2]
                    ]

                verts = np.array(vertices)
                fig_miller.add_trace(go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=[0, 0], j=[1, 2], k=[2, 3],
                    opacity=0.6,
                    color='cyan',
                    name=f'({h}{k}{l}) plane',
                    showlegend=True
                ))
                # Draw the line connecting the two intercepts
                fig_miller.add_trace(go.Scatter3d(
                    x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                    mode='lines', line=dict(color='darkblue', width=4),
                    showlegend=False, hoverinfo='skip'
                ))
            elif len(intercept_points) == 1:
                # One intercept (two indices are 0): plane perpendicular to one axis
                p = intercept_points[0]

                if h != 0:  # Plane perpendicular to x-axis (yz plane at x = 1/h)
                    vertices = [
                        [p[0], -2, -2], [p[0], 2, -2],
                        [p[0], 2, 2], [p[0], -2, 2]
                    ]
                elif k != 0:  # Plane perpendicular to y-axis (xz plane at y = 1/k)
                    vertices = [
                        [-2, p[1], -2], [2, p[1], -2],
                        [2, p[1], 2], [-2, p[1], 2]
                    ]
                else:  # l != 0, plane perpendicular to z-axis (xy plane at z = 1/l)
                    vertices = [
                        [-2, -2, p[2]], [2, -2, p[2]],
                        [2, 2, p[2]], [-2, 2, p[2]]
                    ]

                verts = np.array(vertices)
                fig_miller.add_trace(go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=[0, 0], j=[1, 2], k=[2, 3],
                    opacity=0.6,
                    color='cyan',
                    name=f'({h}{k}{l}) plane',
                    showlegend=True
                ))

            # Draw normal vector from center of unit cell
            normal = np.array([h, k, l], dtype=float)
            normal_length = np.linalg.norm(normal)
            if normal_length > 0:
                normal_unit = normal / normal_length * 0.5
                center = np.array([0.5, 0.5, 0.5])

                fig_miller.add_trace(go.Scatter3d(
                    x=[center[0], center[0] + normal_unit[0]],
                    y=[center[1], center[1] + normal_unit[1]],
                    z=[center[2], center[2] + normal_unit[2]],
                    mode='lines', line=dict(color='magenta', width=5),
                    name=f'Normal [{h}{k}{l}]'
                ))

                fig_miller.add_trace(go.Cone(
                    x=[center[0] + normal_unit[0]],
                    y=[center[1] + normal_unit[1]],
                    z=[center[2] + normal_unit[2]],
                    u=[normal_unit[0] * 0.3],
                    v=[normal_unit[1] * 0.3],
                    w=[normal_unit[2] * 0.3],
                    colorscale=[[0, 'magenta'], [1, 'magenta']],
                    showscale=False, sizemode='absolute', sizeref=0.1,
                    showlegend=False
                ))

        if h != 0 or k != 0 or l != 0:
            d_spacing = 1.0 / np.sqrt(h**2 + k**2 + l**2)
            title_text = f'Plane ({h}{k}{l}) | Interplane spacing = {d_spacing:.3f}a'
        else:
            title_text = 'Miller Plane Calculator'

        fig_miller.update_layout(
            title=dict(text=title_text, x=0.5),
            scene=dict(
                xaxis_title='x [a]', yaxis_title='y [a]', zaxis_title='z [a]',
                aspectmode='cube',
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
                xaxis=dict(range=[-0.5, 1.5]),
                yaxis=dict(range=[-0.5, 1.5]),
                zaxis=dict(range=[-0.5, 1.5]),
            ),
            height=600, width=700,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        return fig_miller

    h_input = mo.ui.slider(value=1, start=-3, stop=3, step=1, label="h")
    k_input = mo.ui.slider(value=1, start=-3, stop=3, step=1, label="k")
    l_input = mo.ui.slider(value=1, start=-3, stop=3, step=1, label="l")
    return h_input, k_input, l_input, plot_miller_plane


@app.cell
def _(h_input, k_input, l_input, mo, plot_miller_plane):
    mo.vstack([
        mo.md("### 8.6 Interactive Miller Indices"),
        mo.md("Enter the Miller indices (h, k, l) to visualize the crystal plane:"),
        mo.hstack([h_input, k_input, l_input], justify="start", gap=2),
        plot_miller_plane(h_input.value, k_input.value, l_input.value)
    ], align="center")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Volumetric Packing Fraction (APF)

    The **atomic packing factor (APF)** measures what fraction of the unit cell volume is occupied by atoms (modeled as hard spheres). We assume atoms touch along the closest-packed direction:

    | Lattice | Atoms touch along | Atomic radius $r$ | APF |
    |:--------:|:--------------:|:-----------------:|:----:|
    | SC      | Edge              | $a/2$             | $\pi/6 \approx 0.524$ |
    | BCC     | Body diagonal     | $\sqrt{3}\,a/4$   | $\pi\sqrt{3}/8 \approx 0.680$ |
    | FCC     | Face diagonal     | $\sqrt{2}\,a/4$   | $\pi\sqrt{2}/6 \approx 0.740$ |
    | Diamond | ¼ body diagonal   | $\sqrt{3}\,a/8$   | $\pi\sqrt{3}/16 \approx 0.340$ |
    """)
    return


@app.cell
def _(go, mo, np):
    def get_lattice_apf_info(lattice_type, a):
        """Return atomic radius, atoms per unit cell, and APF for the lattice."""
        if lattice_type == 'SC':
            r = a / 2
            atoms_per_cell = 1
            apf = np.pi / 6
        elif lattice_type == 'BCC':
            r = np.sqrt(3) * a / 4
            atoms_per_cell = 2
            apf = np.pi * np.sqrt(3) / 8
        elif lattice_type == 'FCC':
            r = np.sqrt(2) * a / 4
            atoms_per_cell = 4
            apf = np.pi * np.sqrt(2) / 6
        elif lattice_type == 'Diamond':
            r = np.sqrt(3) * a / 8
            atoms_per_cell = 8
            apf = np.pi * np.sqrt(3) / 16
        else:
            r, atoms_per_cell, apf = a/2, 1, np.pi/6
        return r, atoms_per_cell, apf


    def create_apf_figure(lattice_type='SC', a=1.0, show_spheres=True):
        """Create a plotly figure for APF visualization."""
        r, atoms_per_cell, apf_theory = get_lattice_apf_info(lattice_type, a)

        # Get atom positions in one unit cell
        if lattice_type == 'SC':
            positions = np.array([[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a],
                                  [a, a, 0], [a, 0, a], [0, a, a], [a, a, a]])
        elif lattice_type == 'BCC':
            corners = np.array([[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a],
                                [a, a, 0], [a, 0, a], [0, a, a], [a, a, a]])
            center = np.array([[a/2, a/2, a/2]])
            positions = np.vstack([corners, center])
        elif lattice_type == 'FCC':
            corners = np.array([[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a],
                                [a, a, 0], [a, 0, a], [0, a, a], [a, a, a]])
            faces = np.array([[a/2, a/2, 0], [a/2, a/2, a],
                              [a/2, 0, a/2], [a/2, a, a/2],
                              [0, a/2, a/2], [a, a/2, a/2]])
            positions = np.vstack([corners, faces])
        elif lattice_type == 'Diamond':
            corners = np.array([[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a],
                                [a, a, 0], [a, 0, a], [0, a, a], [a, a, a]])
            faces = np.array([[a/2, a/2, 0], [a/2, a/2, a],
                              [a/2, 0, a/2], [a/2, a, a/2],
                              [0, a/2, a/2], [a, a/2, a/2]])
            fcc_pts = np.vstack([corners, faces])
            basis_pts = fcc_pts + np.array([a/4, a/4, a/4])
            mask = np.all((basis_pts >= -1e-6) & (basis_pts <= a + 1e-6), axis=1)
            basis_in = basis_pts[mask]
            positions = np.vstack([fcc_pts, basis_in])
        else:
            positions = np.array([[a/2, a/2, a/2]])

        fig = go.Figure()

        # Draw unit cell edges
        edges = [
            ([0, a], [0, 0], [0, 0]), ([0, 0], [0, a], [0, 0]), ([0, 0], [0, 0], [0, a]),
            ([a, a], [0, a], [0, 0]), ([a, a], [0, 0], [0, a]), ([0, a], [a, a], [0, 0]),
            ([0, 0], [a, a], [0, a]), ([0, a], [0, 0], [a, a]), ([0, 0], [0, a], [a, a]),
            ([a, a], [a, a], [0, a]), ([a, a], [0, a], [a, a]), ([0, a], [a, a], [a, a])
        ]

        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines', line=dict(color='black', width=2),
                showlegend=False, hoverinfo='skip'
            ))

        if show_spheres:
            # Create sphere mesh for each atom
            for pos in positions:
                u_vals = np.linspace(0, 2 * np.pi, 20)
                v_vals = np.linspace(0, np.pi, 15)
                x_sphere = pos[0] + r * np.outer(np.cos(u_vals), np.sin(v_vals)).flatten()
                y_sphere = pos[1] + r * np.outer(np.sin(u_vals), np.sin(v_vals)).flatten()
                z_sphere = pos[2] + r * np.outer(np.ones(len(u_vals)), np.cos(v_vals)).flatten()

                fig.add_trace(go.Mesh3d(
                    x=x_sphere, y=y_sphere, z=z_sphere,
                    alphahull=0,
                    color='steelblue',
                    opacity=0.7,
                    showlegend=False
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                mode='markers',
                marker=dict(size=8, color='steelblue', line=dict(color='navy', width=1)),
                showlegend=False
            ))

        fig.update_layout(
            title=dict(
                text=f'{lattice_type}: APF = {apf_theory:.3f} ({apf_theory*100:.1f}%)<br>'
                     f'<sub>r = {r:.3f}, atoms/cell = {atoms_per_cell} | Drag to rotate</sub>',
                x=0.5
            ),
            scene=dict(
                xaxis_title='x', yaxis_title='y', zaxis_title='z',
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                xaxis=dict(range=[-0.3*a, 1.3*a]),
                yaxis=dict(range=[-0.3*a, 1.3*a]),
                zaxis=dict(range=[-0.3*a, 1.3*a]),
            ),
            height=500,
            width=600,
            margin=dict(l=0, r=0, t=80, b=0)
        )

        return fig


    # Create marimo UI controls
    lattice_dropdown = mo.ui.dropdown(
        options=['SC', 'BCC', 'FCC', 'Diamond'],
        value='SC',
        label='Lattice Type'
    )

    a_slider = mo.ui.slider(
        start=0.5,
        stop=2.0,
        step=0.1,
        value=1.0,
        label='Lattice constant a'
    )

    display_dropdown = mo.ui.dropdown(
        options={'Spheres', 'Points'},
        value='Spheres',
        label='Display Mode'
    )
    return (
        a_slider,
        create_apf_figure,
        display_dropdown,
        get_lattice_apf_info,
        lattice_dropdown,
    )


@app.cell
def _(
    a_slider,
    create_apf_figure,
    display_dropdown,
    get_lattice_apf_info,
    lattice_dropdown,
    mo,
):
    mo.vstack([
        mo.md("### 9.1 Atomic Packing Factor (APF) Visualization"),
        mo.hstack([lattice_dropdown, a_slider, display_dropdown], justify="start", gap=2),
        create_apf_figure(lattice_dropdown.value, a_slider.value, display_dropdown.value == 'Spheres'),
        mo.md(f"""
    **Lattice:** {lattice_dropdown.value}  
    **Lattice constant a:** {a_slider.value:.3f}  
    **Atomic radius r:** {get_lattice_apf_info(lattice_dropdown.value, a_slider.value)[0]:.4f}  
    **Atoms per unit cell:** {get_lattice_apf_info(lattice_dropdown.value, a_slider.value)[1]}  
    **APF:** {get_lattice_apf_info(lattice_dropdown.value, a_slider.value)[2]:.4f} = {get_lattice_apf_info(lattice_dropdown.value, a_slider.value)[2]*100:.2f}%
    """)
    ], align="center")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Planar Packing Fraction in Miller Planes

    The **planar (areal) packing fraction** measures how densely atoms are packed in a particular crystal plane:

    $$\text{Planar PF} = \frac{n_{\text{atoms}} \times \pi r^2}{A_{\text{2D cell}}}$$

    where $n_{\text{atoms}}$ is the number of atoms per 2D unit cell and $A_{\text{2D cell}}$ is the area of that cell.

    **Example values for common planes:**

    | Lattice | Plane | $n_{\text{atoms}}$| r               | $A_{\text{2D}}$ | Areal PF |
    |:--------:|:-----:|:-----------------:|:---------------:|:---------------:|:--------:|
    | FCC     | (111) | 1                 |  $\sqrt{2}/4 a$ |$\sqrt{3}a^2/4$ |  90.7% (close-packed) |
    | FCC     | (100) | 2                 | $\sqrt{2}/4 a$  | $a^2$           | 78.5% |
    | FCC     | (110) | 2                 | $\sqrt{2}/4 a$  | $\sqrt{2}a^2$   | 55.5% |
    | BCC     | (110) | 2                 | $\sqrt{3}/4 a$  | $\sqrt{2}a^2$   | 83.3% (close-packed) |
    | BCC     | (100) | 1                 | $\sqrt{3}/4 a$ | $a^2$           | 58.9% |
    | BCC     | (111) | 1                 | $\sqrt{3}/4 a$ |$\sqrt{3}a^2$   | 34.0% |
    """)
    return


@app.cell
def _(get_lattice_apf_info, mo, np):
    def get_plane_cell_intersection(h, k, l, a):
        """Find intersection polygon of plane hx + ky + lz = a with the unit cell [0,a]^3."""
        if h == 0 and k == 0 and l == 0:
            return np.array([])  # Invalid plane

        # Edges of the unit cell: 12 edges, each defined by a start point and direction
        edges = []
        for fixed_dim in range(3):  # Which dimension is fixed for this set of edges
            for v1 in [0, a]:
                for v2 in [0, a]:
                    start = [0, 0, 0]
                    start[fixed_dim] = 0
                    start[(fixed_dim + 1) % 3] = v1
                    start[(fixed_dim + 2) % 3] = v2
                    direction = [0, 0, 0]
                    direction[fixed_dim] = a
                    edges.append((np.array(start), np.array(direction)))

        normal = np.array([h, k, l], dtype=float)
        d = a  # Plane equation: h*x + k*y + l*z = a

        intersection_pts = []
        for start, direction in edges:
            denom = np.dot(normal, direction)
            if abs(denom) < 1e-10:
                continue  # Edge parallel to plane
            t = (d - np.dot(normal, start)) / denom
            if -1e-10 <= t <= 1 + 1e-10:  # Intersection within edge
                pt = start + t * direction
                # Check if point is within unit cell bounds
                if all(-1e-10 <= pt[i] <= a + 1e-10 for i in range(3)):
                    intersection_pts.append(pt)

        if len(intersection_pts) < 3:
            return np.array([])

        pts = np.array(intersection_pts)
        # Remove duplicate points
        unique_pts = [pts[0]]
        for p in pts[1:]:
            if all(np.linalg.norm(p - u) > 1e-6 for u in unique_pts):
                unique_pts.append(p)
        pts = np.array(unique_pts)

        if len(pts) < 3:
            return np.array([])

        # Sort points to form a convex polygon (by angle around centroid)
        centroid = pts.mean(axis=0)
        # Project onto plane and sort by angle
        # Use two basis vectors in the plane
        v0 = pts[0] - centroid
        v0 = v0 / (np.linalg.norm(v0) + 1e-10)
        v1 = np.cross(normal, v0)
        v1 = v1 / (np.linalg.norm(v1) + 1e-10)

        angles = []
        for p in pts:
            diff = p - centroid
            angle = np.arctan2(np.dot(diff, v1), np.dot(diff, v0))
            angles.append(angle)

        sorted_idx = np.argsort(angles)
        return pts[sorted_idx]

    def generate_sc(n, a=1.0):
        """Simple cubic lattice points in an n × n × n array of unit cells."""
        xs, ys, zs = np.mgrid[0:n, 0:n, 0:n]
        pts = np.vstack((xs.ravel(), ys.ravel(), zs.ravel())).T * a
        return pts


    def generate_bcc(n, a=1.0):
        """Body-centred cubic lattice: SC + body centres."""
        sc = generate_sc(n, a)
        centers = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    centers.append(np.array([i + 0.5, j + 0.5, k + 0.5]) * a)
        centers = np.array(centers)
        return np.vstack((sc, centers))


    def generate_fcc(n, a=1.0):
        """Face-centred cubic lattice: SC + centres of each face."""
        sc = generate_sc(n, a)
        shifts = np.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ])
        faces = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    origin = np.array([i, j, k], dtype=float)
                    for s in shifts:
                        faces.append((origin + s) * a)
        faces = np.array(faces)
        return np.vstack((sc, faces))


    def generate_diamond(n, a=1.0):
        """Diamond structure = FCC Bravais lattice + 2-atom basis.

        Basis at (0,0,0) and (1/4, 1/4, 1/4).
        """
        fcc_pts = generate_fcc(n, a)
        basis_shift = np.array([0.25, 0.25, 0.25]) * a
        diamond_pts = np.vstack((fcc_pts, fcc_pts + basis_shift))
        return diamond_pts



    def get_atoms_on_plane(lattice_type, h, k, l, a, tol=1e-4):
        """Find atoms that lie on the Miller plane hx + ky + lz = a."""
        if lattice_type == 'SC':
            pts = generate_sc(3, a)
        elif lattice_type == 'BCC':
            pts = generate_bcc(3, a)
        elif lattice_type == 'FCC':
            pts = generate_fcc(3, a)
        elif lattice_type == 'Diamond':
            pts = generate_diamond(3, a)
        else:
            pts = generate_sc(3, a)

        # Plane equation: h*x + k*y + l*z = a
        normal = np.array([h, k, l], dtype=float)
        if np.linalg.norm(normal) < 1e-10:
            return np.array([]), 0

        # Find points on the plane
        distances = np.abs(h * pts[:, 0] + k * pts[:, 1] + l * pts[:, 2] - a)
        on_plane = pts[distances < tol]

        return on_plane, len(on_plane)


    def get_areal_pf_calculation(lattice_type, h, k, l, a):
        """Return areal packing fraction with full calculation details.

        Returns: (apf_value, n_atoms, area_2d, r, area_formula, description)
        """
        # Normalize Miller indices to identify equivalent planes
        indices = tuple(sorted([abs(h), abs(k), abs(l)], reverse=True))

        # Get atomic radius
        r, _, _ = get_lattice_apf_info(lattice_type, a)
        r_sq = r**2

        if lattice_type == 'SC':
            # SC: r = a/2, r² = a²/4
            if indices == (1, 0, 0):
                # (100): square a × a, 4 corners × 1/4 = 1 atom
                n_atoms, area_2d = 1, a**2
                area_formula = "a²"
                desc = "Square cell with 4 corners × ¼"
            elif indices == (1, 1, 0):
                # (110): rectangle a × √2·a, atoms at corners and edge centers
                n_atoms, area_2d = 1, np.sqrt(2) * a**2
                area_formula = "√2·a²"
                desc = "Rectangular cell"
            elif indices == (1, 1, 1):
                # (111): hexagonal with NN = √2·a
                n_atoms, area_2d = 1, np.sqrt(3) * a**2
                area_formula = "√3·a²"
                desc = "Hexagonal cell, NN distance = √2·a"
            else:
                return None, None, None, r, None, None

        elif lattice_type == 'FCC':
            # FCC: r = √2·a/4, r² = a²/8
            if indices == (1, 1, 1):
                # (111): close-packed hexagonal, NN = a/√2
                # Primitive cell area = (√3/2) × (a/√2)² = √3·a²/4
                n_atoms, area_2d = 1, np.sqrt(3) * a**2 / 4
                area_formula = "√3·a²/4"
                desc = "Close-packed hexagonal, NN = a/√2"
            elif indices == (1, 0, 0):
                # (100): square a × a, 4 corners × 1/4 + 1 face center = 2 atoms
                n_atoms, area_2d = 2, a**2
                area_formula = "a²"
                desc = "Square cell: 4 corners × ¼ + 1 center"
            elif indices == (1, 1, 0):
                # (110): rectangle a × √2·a, 4 corners × 1/4 + 2 edges × 1/2 = 2 atoms
                n_atoms, area_2d = 2, np.sqrt(2) * a**2
                area_formula = "√2·a²"
                desc = "Rectangular cell: 4 corners × ¼ + 2 edges × ½"
            else:
                return None, None, None, r, None, None

        elif lattice_type == 'BCC':
            # BCC: r = √3·a/4, r² = 3a²/16
            if indices == (1, 1, 0):
                # (110): close-packed for BCC, rectangle a × √2·a
                # 4 corners × 1/4 + 1 center = 2 atoms
                n_atoms, area_2d = 2, np.sqrt(2) * a**2
                area_formula = "√2·a²"
                desc = "Close-packed for BCC: 4 corners × ¼ + 1 center"
            elif indices == (1, 0, 0):
                # (100): square a × a, only corner atoms (body center on different plane)
                # 4 corners × 1/4 = 1 atom
                n_atoms, area_2d = 1, a**2
                area_formula = "a²"
                desc = "Square cell: 4 corners × ¼ (body centers on different plane)"
            elif indices == (1, 1, 1):
                # (111): only SC sublattice atoms (body centers on different planes)
                # Hexagonal with NN = √2·a, area = √3·a²
                n_atoms, area_2d = 1, np.sqrt(3) * a**2
                area_formula = "√3·a²"
                desc = "Hexagonal (SC sublattice only), NN = √2·a"
            else:
                return None, None, None, r, None, None

        elif lattice_type == 'Diamond':
            # Diamond: r = √3·a/8, r² = 3a²/64
            if indices == (1, 1, 1):
                # (111): Two interlocking triangular sublattices
                # 2 atoms per FCC primitive cell area = √3·a²/4
                n_atoms, area_2d = 2, np.sqrt(3) * a**2 / 4
                area_formula = "√3·a²/4"
                desc = "2 atoms per hexagonal cell (FCC + basis)"
            elif indices == (1, 0, 0):
                # (100): 4 atoms per a² (2 FCC + 2 basis atoms visible)
                n_atoms, area_2d = 4, a**2
                area_formula = "a²"
                desc = "Square cell: 4 atoms (FCC + basis layers)"
            elif indices == (1, 1, 0):
                # (110): rectangular cell with zigzag chains
                n_atoms, area_2d = 4, np.sqrt(2) * a**2
                area_formula = "√2·a²"
                desc = "Rectangular cell with zigzag chains"
            else:
                return None, None, None, r, None, None
        else:
            return None, None, None, r, None, None

        # Calculate APF
        apf = n_atoms * np.pi * r_sq / area_2d

        return apf, n_atoms, area_2d, r, area_formula, desc


    def _get_plane_basis_vectors(h, k, l):
        """Get two orthonormal basis vectors that lie in the plane (hkl)."""
        normal = np.array([h, k, l], dtype=float)
        normal = normal / np.linalg.norm(normal)

        # Find a vector not parallel to normal
        if abs(normal[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])

        # First basis vector (in plane)
        u = np.cross(normal, temp)
        u = u / np.linalg.norm(u)

        # Second basis vector (in plane, perpendicular to u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        return u, v


    def _draw_circle_on_plane(ax, center, radius, h, k, l, n_points=50, **kwargs):
        """Draw a circle on the plane (hkl) centered at the given point."""
        u, v = _get_plane_basis_vectors(h, k, l)

        # Generate circle points
        theta = np.linspace(0, 2 * np.pi, n_points)
        circle_pts = np.array([
            center + radius * (np.cos(t) * u + np.sin(t) * v)
            for t in theta
        ])

        ax.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2], **kwargs)


    def plot_areal_packing(lattice_type='SC', h=1, k=0, l=0, a=1.0):
        """Visualize atoms on a Miller plane and show areal packing fraction calculation."""
        import plotly.graph_objects as go

        if h == 0 and k == 0 and l == 0:
            return mo.md("**Invalid Miller indices (0,0,0)**")

        # Get atomic radius for this lattice
        r, _, _ = get_lattice_apf_info(lattice_type, a)

        # Get plane intersection polygon
        plane_pts = get_plane_cell_intersection(h, k, l, a)

        if len(plane_pts) < 3:
            return mo.md(f"**Plane ({h}{k}{l}) does not intersect the unit cell properly.**")

        # Get atoms on the plane
        atoms_on_plane, n_atoms_found = get_atoms_on_plane(lattice_type, h, k, l, a)

        # Get areal packing fraction calculation
        apf, n_atoms, area_2d, r_calc, area_formula, desc = get_areal_pf_calculation(lattice_type, h, k, l, a)

        if apf is None:
            return mo.md(f"**Areal packing fraction not available for {lattice_type} ({h}{k}{l})**")

        # Create Plotly figure
        fig = go.Figure()

        # Draw unit cell edges
        edges = [
            ([0, a], [0, 0], [0, 0]), ([0, 0], [0, a], [0, 0]), ([0, 0], [0, 0], [0, a]),
            ([a, a], [0, a], [0, 0]), ([a, a], [0, 0], [0, a]), ([0, a], [a, a], [0, 0]),
            ([0, 0], [a, a], [0, a]), ([0, a], [0, 0], [a, a]), ([0, 0], [0, a], [a, a]),
            ([a, a], [a, a], [0, a]), ([a, a], [0, a], [a, a]), ([0, a], [a, a], [a, a])
        ]
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Draw the plane polygon using Mesh3d
        if len(plane_pts) >= 3:
            # Create triangulation for the polygon
            n_pts = len(plane_pts)
            # Fan triangulation from first vertex
            i_indices = [0] * (n_pts - 2)
            j_indices = list(range(1, n_pts - 1))
            k_indices = list(range(2, n_pts))

            fig.add_trace(go.Mesh3d(
                x=plane_pts[:, 0],
                y=plane_pts[:, 1],
                z=plane_pts[:, 2],
                i=i_indices,
                j=j_indices,
                k=k_indices,
                color='cyan',
                opacity=0.4,
                name=f'({h}{k}{l}) plane',
                showlegend=True
            ))

            # Draw plane edges
            closed_pts = np.vstack([plane_pts, plane_pts[0]])
            fig.add_trace(go.Scatter3d(
                x=closed_pts[:, 0],
                y=closed_pts[:, 1],
                z=closed_pts[:, 2],
                mode='lines',
                line=dict(color='darkblue', width=4),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Draw atoms on the plane
        if len(atoms_on_plane) > 0:
            fig.add_trace(go.Scatter3d(
                x=atoms_on_plane[:, 0],
                y=atoms_on_plane[:, 1],
                z=atoms_on_plane[:, 2],
                mode='markers',
                marker=dict(size=12, color='steelblue', line=dict(color='navy', width=2)),
                name='Atoms on plane',
                showlegend=True
            ))

            # Draw circles representing atomic radius on the plane
            u, v = _get_plane_basis_vectors(h, k, l)
            for atom in atoms_on_plane:
                theta = np.linspace(0, 2 * np.pi, 50)
                circle_pts = np.array([atom + r * (np.cos(t) * u + np.sin(t) * v) for t in theta])
                fig.add_trace(go.Scatter3d(
                    x=circle_pts[:, 0],
                    y=circle_pts[:, 1],
                    z=circle_pts[:, 2],
                    mode='lines',
                    line=dict(color='blue', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{lattice_type} - Plane ({h}{k}{l})<br><sub>Areal Packing Fraction = {apf*100:.1f}% | Drag to rotate</sub>',
                x=0.5
            ),
            scene=dict(
                xaxis_title='x [a]',
                yaxis_title='y [a]',
                zaxis_title='z [a]',
                aspectmode='cube',
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
                xaxis=dict(range=[-0.2, a + 0.5]),
                yaxis=dict(range=[-0.2, a + 0.5]),
                zaxis=dict(range=[-0.2, a + 0.5]),
            ),
            height=600,
            width=800,
            margin=dict(l=0, r=0, t=80, b=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Create calculation summary
        calc_text = f"""
    **Calculation for {lattice_type} ({h}{k}{l}):**

    - **Description:** {desc}
    - **Atomic radius:** r = {r_calc:.4f}a
    - **Atoms per 2D cell:** n = {n_atoms}
    - **2D cell area:** A = {area_formula} = {area_2d:.4f}a²
    - **Areal PF:** n × πr² / A = {n_atoms} × π × ({r_calc:.4f})² / {area_2d:.4f} = **{apf*100:.1f}%**
    """

        return mo.vstack([fig, mo.md(calc_text)], align="center")

    lattice_type_dropdown2 = mo.ui.dropdown(
        options=['SC', 'BCC', 'FCC', 'Diamond'],
        value='FCC',
        label='Lattice Type'
    )

    h_slider2 = mo.ui.slider(start=0, stop=1, step=1, value=1, label="h")
    k_slider2 = mo.ui.slider(start=0, stop=1, step=1, value=1, label="k")
    l_slider2 = mo.ui.slider(start=0, stop=1, step=1, value=1, label="l")
    return (
        h_slider2,
        k_slider2,
        l_slider2,
        lattice_type_dropdown2,
        plot_areal_packing,
    )


@app.cell
def _(
    h_slider2,
    k_slider2,
    l_slider2,
    lattice_type_dropdown2,
    mo,
    plot_areal_packing,
):
    mo.vstack([
        mo.md("### 10.1 Planar Packing Fraction Visualization"),
        mo.hstack([lattice_type_dropdown2, h_slider2, k_slider2, l_slider2], justify="start", gap=2),
        plot_areal_packing(
            lattice_type=lattice_type_dropdown2.value,
            h=h_slider2.value,
            k=k_slider2.value,
            l=l_slider2.value,
            a=1.0
        )
    ], align="center")
    return


@app.cell
def _(IMAGE_DIR, mo):
    mo.vstack([
        mo.md(r"""
    ## 11. Wafers and Wafer Flats
    """),
        mo.hstack([
            mo.md(r"""
    - Silicon wafers have specific orientations marked by **flats** or **notches**:
        - **Primary flat (or notch)**: Longer flat edge along [011] direction
        - **Secondary flat**: Shorter flat edge
    - The wafer orientation affects carrier mobility, etch rates, defect densities, etc.
    """),
            mo.image(src=IMAGE_DIR / "wafer_flats.jpg", width="60%")
        ], justify="start",  gap=2)
    ])
    return


if __name__ == "__main__":
    app.run()
