# ECE350 Interactive Notebooks Documentation

This folder contains interactive HTML-based educational notebooks for ECE350 - Semiconductor Electronic Devices.

## Contents

- **Crystal Structures** - Introduction to crystal structures, lattices, and Miller indices (Lectures 2-3)
- **Energy Bands** - Bloch's theorem, Kronig-Penney model, and E-k diagrams (Lectures 6-7)
- **Crystal Potential Visualizations** - Interactive 2D/3D visualizations of periodic potentials
- **Effective Mass** - Effective mass concept and band diagrams (Lectures 8-9)
- **Carriers at Thermal Equilibrium** - Density of states and Fermi-Dirac distribution (Lectures 9-11)
- **Carrier Drift** - Drift velocity, mobility, and scattering mechanisms (Lecture 12)
- **Band Bending** - Electrostatics and energy diagrams (Lecture 13)
- **Diffusion** - Diffusion, built-in field, Einstein's relation (Lecture 13)
- **Generation and Recombination** - G-R, excess carriers, quasi-Fermi level (Lectures 13-14)
- **Continuity Equqation** - Spatio-temporal evolution of carrier densities (Lectures 15)
- **PN Junctions** - Overview, electrostatics, reverse bias, IV, small-signal, optoelectronics (Lecture 16 - 23)
- **Bipolar Junction Transistors** - Overview, electrostatics, energy band diagrams, heterostructures (Lecture 24-26)
- **Metal-Semiconductor Interfaces** - Electrostatics, energy diagrams, Schottky diode, Ohmic contact (Lecture 27-28)
- **MOS Capacitors** - Electrostatics, energy diagrams (Lecture 29- )
- To be updated as the course progresses

## Viewing the Documentation

Open `index.html` in a web browser to access all notebooks.

## After re-exporting marimo notebooks

Marimo WASM exports overwrite `index.html` and remove noindex meta tags. To prevent search indexing again, from the repo root run:

```bash
python3 docs/add-noindex.py
```

## Separate Repository

This docs folder can be maintained as a separate repository to facilitate sharing and collaboration while staying synced with the main ECE350 repository.

### Repository Structure

- **Main Repository**: `joyce-poon/ECE350` - Contains all course materials including docs (Private)
- **Docs Repository**: A separate repository containing only this docs folder (for sharing)

### Syncing Changes

See [DOCS_REPO_SETUP.md](../DOCS_REPO_SETUP.md) in the root of the main repository for detailed instructions on:
- Creating the separate docs repository
- Pushing changes from main repo to docs repo
- Pulling changes from docs repo back to main repo
- Automated syncing with GitHub Actions

## License

This repository contains both software code and educational content.

- **Code** (e.g., `.py`, `.ipynb` code cells, scripts, utilities) is licensed under the **MIT License** — see `LICENSE`.
- **Educational content** (e.g., explanatory text, figures, diagrams, and written material) is licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)** — see `LICENSE-CC-BY`.

### Attribution request
If you reuse these materials, please acknowledge:

**Joyce Poon**, *ECE350 Interactive Notebooks*, University of Toronto, 2026.

If you made changes, please indicate that your version is modified.
