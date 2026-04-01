# ECE350 Interactive Notebooks Documentation

This folder contains interactive HTML-based educational notebooks for ECE350 - Semiconductor Electronic Devices.

## Contents

| Notebook | Lectures | Folder |
|:---------|:---------|:-------|
| Crystal Structures | 2–3 | `crystals/` |
| Energy Bands | 6–7 | `energybands/` |
| Crystal Potential Visualizations | — | `crystalpotential/` |
| Effective Mass | 8–9 | `effectivemass/` |
| Carriers at Thermal Equilibrium | 9–11 | `equilibrium/` |
| Carrier Drift | 12 | `drift/` |
| Band Bending and Electrostatics | 13 | `bandbend/` |
| Diffusion | 13 | `diffusion/` |
| Generation, Recombination & Quasi-Fermi Levels | 14 | `gen_recomb/` |
| Continuity Equation | 15 | `continuity/` |
| PN Junction Electrostatics | 17 | `pn-electrostatics/` |
| PN Junction under Reverse Bias | 18 | `pn-revbias/` |
| PN Junction I–V Characteristics | 19–21 | `pn-iv/` |
| PN Junction: Approximations and Non-Idealities | 21–22 | `pn-non-idealities/` |
| PN Junction: Small-Signal Models | 22 | `pn-small-sig/` |
| PN Junction: Light Absorption | 23 | `pn-light-absorption/` |
| BJT Current-Voltage Relations | 25–26 | `bjt-iv/` |
| Metal-Semiconductor Interfaces | 27–28 | `m-s-interface/` |
| MOS Capacitor: Energy Bands & Electrostatics | 29–30 | `moscap/` |
| MOS Capacitor: C-V Characteristics | 31 | `mos-cv/` |
| MOS Capacitor: Non-Idealities | 32 | `mos-nonidealities/` |
| MOSFET: Introduction & Operation | 33 | `mosfet-intro/` |
| MOSFET I_DS–V_DS: Derivation and Modifications | 34 | `mosfet-iv/` |
| MOSFET Dynamics & Small-Signal Model | 35 | `mosfet-ac/` |
| FinFET & FD-SOI MOS Capacitor | 36 (Special topic) | `finfet-fdsoi/` |

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
