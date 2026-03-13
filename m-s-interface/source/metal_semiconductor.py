# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "matplotlib==3.10.8",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    q = 1.6e-19        # C
    kT = 0.02585       # eV at 300K
    eps_0 = 8.854e-14  # F/cm
    eps_Si = 11.7
    ni_Si = 1.1e10     # cm^-3
    Eg_Si = 1.12       # eV
    chi_Si = 4.05      # eV (electron affinity of Si)
    Nc_Si = 2.86e19    # cm^-3 (effective DOS, conduction band, 300K)

    metal_data = {
        "Mg": 3.7, "Al": 4.1, "Ti": 4.3, "Cr": 4.5,
        "W": 4.6, "Mo": 4.6, "Pd": 5.1, "Au": 5.1, "Pt": 5.7,
    }

    A_star_n = 110  # A/cm^2/K^2 (N-Si)
    A_star_p = 32   # A/cm^2/K^2 (P-Si)
    T = 300         # K
    return (
        A_star_n,
        Eg_Si,
        Nc_Si,
        T,
        chi_Si,
        eps_0,
        eps_Si,
        kT,
        metal_data,
        mo,
        ni_Si,
        np,
        plt,
        q,
    )


@app.cell
def _():
    from pathlib import Path as _Path
    try:
        _test = _Path(__file__).parent / "images"
        if _test.exists():
            ASSET_DIR = _Path(__file__).parent
        else:
            raise FileNotFoundError
    except Exception:
        ASSET_DIR = None
    IMAGE_BASE = (
        "https://joyce-poon.github.io/ECE350/metal-semiconductor/images"
        if ASSET_DIR is None
        else str(ASSET_DIR / "images")
    )
    return (IMAGE_BASE,)


@app.cell
def _(Eg_Si, chi_Si, np):
    def schottky_bands(phi_m, phi_s, x_dep_vis, x_total=3.0):
        """Band profile after contact. EF = 0 reference.
        Works for N-type, P-type, rectifying, and ohmic.
        Ec_junction = phi_m - chi, Ec_bulk = phi_s - chi.
        Band bending delta = phi_m - phi_s.
        """
        Ec_bulk = phi_s - chi_Si
        Ec_junc = phi_m - chi_Si

        x_dep = np.linspace(0, x_dep_vis, 200)
        x_flat = np.linspace(x_dep_vis, x_total, 50)

        delta = Ec_junc - Ec_bulk
        Ec_dep = Ec_bulk + delta * (1 - x_dep / x_dep_vis) ** 2
        Ec_fb = np.full_like(x_flat, Ec_bulk)

        x_all = np.concatenate([x_dep, x_flat])
        Ec_all = np.concatenate([Ec_dep, Ec_fb])
        Ev_all = Ec_all - Eg_Si

        return x_all, Ec_all, Ev_all

    def schottky_bands_biased(phi_m, phi_s, Va, x_dep_vis, x_total=3.0):
        """Band profile under bias. EF_metal = 0, EF_SC = Va.
        Barrier phi_Bn = phi_m - chi is independent of bias.
        Effective band bending = phi_bi - Va.
        """
        Ec_bulk = (phi_s - chi_Si) + Va
        Ec_junc = phi_m - chi_Si

        x_dep = np.linspace(0, x_dep_vis, 200)
        x_flat = np.linspace(x_dep_vis, x_total, 50)

        delta = Ec_junc - Ec_bulk
        Ec_dep = Ec_bulk + delta * (1 - x_dep / x_dep_vis) ** 2
        Ec_fb = np.full_like(x_flat, Ec_bulk)

        x_all = np.concatenate([x_dep, x_flat])
        Ec_all = np.concatenate([Ec_dep, Ec_fb])
        Ev_all = Ec_all - Eg_Si

        return x_all, Ec_all, Ev_all

    def energy_marker(ax, x, E_lo, E_hi, label, color, cap=0.1,
                      text_offset=0.15, fontsize=14, guides=None, label_side='right'):
        """Flat-cap energy marker: vertical bar with horizontal caps.
        guides: list of (x_from, E_level) — dotted lines showing where levels come from.
        """
        ax.plot([x, x], [E_lo, E_hi], color=color, lw=1.5, solid_capstyle='butt')
        ax.plot([x - cap, x + cap], [E_lo, E_lo], color=color, lw=1.5)
        ax.plot([x - cap, x + cap], [E_hi, E_hi], color=color, lw=1.5)
        if label_side == 'right':
            ax.text(x + text_offset, (E_lo + E_hi) / 2, label,
                    fontsize=fontsize, color=color, va='center')
        else:
            ax.text(x - text_offset, (E_lo + E_hi) / 2, label,
                    fontsize=fontsize, color=color, va='center', ha='right')
        if guides:
            for gx, gy in guides:
                x_lo_g = min(gx, x - cap)
                x_hi_g = max(gx, x + cap)
                ax.plot([x_lo_g, x_hi_g], [gy, gy], color=color,
                        lw=0.8, ls=':', alpha=0.5)

    def width_marker(ax, x_lo, x_hi, y, label, color, cap=0.1,
                     text_offset=-0.25, fontsize=14):
        """Flat-cap width marker: horizontal bar with vertical caps."""
        ax.plot([x_lo, x_hi], [y, y], color=color, lw=1.5)
        ax.plot([x_lo, x_lo], [y - cap, y + cap], color=color, lw=1.5)
        ax.plot([x_hi, x_hi], [y - cap, y + cap], color=color, lw=1.5)
        ax.text((x_lo + x_hi) / 2, y + text_offset, label,
                fontsize=fontsize, color=color, ha='center', va='center')

    return energy_marker, schottky_bands, schottky_bands_biased, width_marker


@app.cell
def _(mo):
    mo.md(r"""
    # Metal-Semiconductor Interfaces

    ECE350, Lectures 27-28

    Reference: Hu, Chapter 4  Part III (4.16-4.21)

    This notebook covers the physics of metal-semiconductor junctions:

    1. Rectifying Contact: Schottky Diode (N-type and P-type)
    2. Schottky Barrier and Built-in Potential under Bias
    3. Schottky Diode Electrostatics (interactive)
    4. Interactive Explorer: Rectifying vs. Ohmic Contacts
    5. IV Characteristics: Thermionic Emission
    6. Ohmic Contacts
    7. Ohmic Contacts in Practice: Tunneling Junctions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Metal-Semiconductor Interfaces

    Metal-semiconductor junctions exhibit two fundamentally different behaviours depending on the relative work functions:

    - **Rectifying** contact → Schottky diode
    - **Resistive** contact → Ohmic contact

    The behaviour is determined by the relationship between the metal work function $\Psi_M$ and the semiconductor work function $\Psi_S$.
    """)
    return


@app.cell(hide_code=True)
def _(Eg_Si, chi_Si, energy_marker, mo, np, plt):
    _fig_def, (_ax_m, _ax_s) = plt.subplots(1, 2, figsize=(10, 4.5))

    _phi_m_ex = 4.6
    _Evac = 0.0
    _EF_m = _Evac - _phi_m_ex

    _ax_m.fill_between([-1, 1], _EF_m - 2.0, _EF_m, color='#6baed6', alpha=0.35, label='Filled states')
    _ax_m.fill_between([-1, 1], _EF_m, _EF_m + 0.5, color='#6baed6', alpha=0.10)
    _ax_m.plot([-1, 1], [_Evac, _Evac], 'k-', linewidth=2)
    _ax_m.text(1.05, _Evac, '$E_{vac}$', fontsize=16, va='center')
    _ax_m.plot([-1, 1], [_EF_m, _EF_m], 'g--', linewidth=2.5)
    _ax_m.text(1.05, _EF_m, '$E_{F}$', fontsize=16, va='center', color='green')
    energy_marker(_ax_m, 0, _EF_m, _Evac, r'$q\Psi_M$', 'purple',
                  cap=0.12, text_offset=0.18, fontsize=18)
    _ax_m.set_xlim(-1.5, 2.0)
    _ax_m.set_ylim(_EF_m - 2.5, _Evac + 1.0)
    _ax_m.set_ylabel('Energy (eV)', fontsize=16)
    _ax_m.set_title('Metal', fontsize=18, fontweight='bold')
    _ax_m.set_xticks([])
    _ax_m.tick_params(labelsize=14)
    for sp in ['top', 'right', 'bottom']:
        _ax_m.spines[sp].set_visible(False)

    _phi_s_ex = chi_Si + Eg_Si / 2 - 0.02585 * np.log(1e16 / 1.1e10)
    _Ec = _Evac - chi_Si
    _Ev = _Ec - Eg_Si
    _EF_s = _Evac - _phi_s_ex
    _Ei = (_Ec + _Ev) / 2

    _ax_s.plot([-1, 1], [_Ec, _Ec], 'b-', linewidth=2.5)
    _ax_s.text(1.05, _Ec, '$E_c$', fontsize=16, va='center', color='blue')
    _ax_s.plot([-1, 1], [_Ev, _Ev], 'r-', linewidth=2.5)
    _ax_s.text(1.05, _Ev, '$E_v$', fontsize=16, va='center', color='red')
    _ax_s.plot([-1, 1], [_Ei, _Ei], 'k:', linewidth=1.5, alpha=0.5)
    _ax_s.text(1.05, _Ei, '$E_i$', fontsize=14, va='center', alpha=0.6)
    _ax_s.plot([-1, 1], [_EF_s, _EF_s], 'g--', linewidth=2.5)
    _ax_s.text(1.05, _EF_s, '$E_F$', fontsize=16, va='center', color='green')
    _ax_s.plot([-1, 1], [_Evac, _Evac], 'k-', linewidth=2)
    _ax_s.text(1.05, _Evac, '$E_{vac}$', fontsize=16, va='center')

    energy_marker(_ax_s, -0.5, _Ec, _Evac, r'$q\chi$', 'darkorange',
                  cap=0.12, text_offset=0.18, fontsize=18)
    energy_marker(_ax_s, 0.3, _EF_s, _Evac, r'$q\Psi_S$', 'purple',
                  cap=0.12, text_offset=0.18, fontsize=18)
    energy_marker(_ax_s, -0.8, _Ev, _Ec, '$E_g$', 'gray',
                  cap=0.12, text_offset=0.18, fontsize=16)

    _ax_s.set_xlim(-1.5, 2.0)
    _ax_s.set_ylim(_Ev - 1.0, _Evac + 1.0)
    _ax_s.set_title('N-type Semiconductor', fontsize=18, fontweight='bold')
    _ax_s.set_xticks([])
    _ax_s.tick_params(labelsize=14)
    for sp in ['top', 'right', 'bottom']:
        _ax_s.spines[sp].set_visible(False)

    plt.tight_layout()

    _defs = mo.md(r"""
    ## Definitions

    | Quantity | Symbol | Definition |
    |:---------|:------:|:-----------|
    | **Work function** | $\Psi_M$, $\Psi_S$ | Energy required to free an electron from the Fermi level to vacuum |
    | **Electron affinity** | $\chi$ | Energy difference between vacuum level and conduction band edge ($E_{vac} - E_c$); a material property of the semiconductor |

    For silicon: $\chi_{Si}$ = 4.05 eV, $E_g$ = 1.12 eV.

    $\Psi_S$ and $\chi$ are material properties. $\Psi_S$ depends on doping; $\chi$ does not.
    """)



    mo.vstack([_defs, _fig_def])
    return


@app.cell(hide_code=True)
def _(
    Eg_Si,
    chi_Si,
    energy_marker,
    kT,
    mo,
    ni_Si,
    np,
    plt,
    schottky_bands,
    width_marker,
):
    _phi_m = 4.6   # W (Tungsten)
    _Nd = 1e16
    _phi_n = kT * np.log(_Nd / ni_Si)
    _phi_s = chi_Si + (Eg_Si / 2 - _phi_n)
    _phi_Bn = _phi_m - chi_Si
    _phi_bi = _phi_m - _phi_s

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # === LEFT: Before contact ===
    _Evac = max(_phi_m, _phi_s) + 0.5
    _EF_m = _Evac - _phi_m
    _EF_s = _Evac - _phi_s
    _Ec_s = _Evac - chi_Si
    _Ev_s = _Ec_s - Eg_Si

    _ml, _mr = -2.0, -0.3
    _sl, _sr = 0.3, 3.0

    _ax1.fill_between([_ml, _mr], _EF_m - 1, _EF_m, color='#6baed6', alpha=0.4)
    _ax1.plot([_ml, _mr], [_EF_m, _EF_m], 'g--', linewidth=2)
    _ax1.text(_ml + 0.1, _EF_m + 0.08, '$E_{F,M}$', fontsize=14, color='green')
    _ax1.plot([_ml, _mr], [_Evac, _Evac], 'k-', linewidth=1.5)
    energy_marker(_ax1, _ml + 0.3, _EF_m, _Evac, r'$q\Psi_M$', 'purple')

    _ax1.plot([_sl, _sr], [_Ec_s, _Ec_s], 'b-', linewidth=2.5)
    _ax1.plot([_sl, _sr], [_Ev_s, _Ev_s], 'r-', linewidth=2.5)
    _ax1.plot([_sl, _sr], [_EF_s, _EF_s], 'g--', linewidth=2)
    _ax1.text(_sr - 0.5, _EF_s + 0.08, '$E_{F,S}$', fontsize=14, color='green')
    _ax1.plot([_sl, _sr], [_Evac, _Evac], 'k-', linewidth=1.5)
    energy_marker(_ax1, _sr - 0.3, _Ec_s, _Evac, r'$q\chi$', 'orange')
    energy_marker(_ax1, _sl + 0.3, _EF_s, _Evac, r'$q\Psi_S$', 'purple')

    _ax1.annotate('', xy=(-0.15, (_EF_m + _EF_s) / 2), xytext=(0.15, (_EF_m + _EF_s) / 2),
                  arrowprops=dict(arrowstyle='->', color='#e6550d', lw=2.5))
    _ax1.text(0.0, (_EF_m + _EF_s) / 2 + 0.15, '$e^-$', fontsize=14, ha='center', color='#e6550d')

    _ax1.text((_ml + _mr) / 2, _EF_m - 1.3, 'Metal', fontsize=16, ha='center', fontweight='bold')
    _ax1.text((_sl + _sr) / 2, _Ev_s - 0.4, 'N-type', fontsize=16, ha='center', fontweight='bold')
    _ax1.text(0, _Evac + 0.15, '$E_{vac}$', fontsize=14, ha='center')
    _ax1.set_xlim(_ml - 0.3, _sr + 0.5)
    _ax1.set_ylim(min(_Ev_s, _EF_m) - 1.5, _Evac + 0.5)
    _ax1.set_ylabel('Energy (eV)', fontsize=16)
    _ax1.set_title(r'Before Contact ($\Psi_M > \Psi_S$)', fontsize=16, fontweight='bold')
    _ax1.set_xticks([])
    _ax1.tick_params(labelsize=14)

    # === RIGHT: After contact (EF = 0 reference) ===
    _x_dep_vis = 1.5
    _x_all, _Ec_all, _Ev_all = schottky_bands(_phi_m, _phi_s, _x_dep_vis)
    _EF = 0.0

    _ax2.fill_between([-2.0, 0], _EF - 1.5, _EF, color='#6baed6', alpha=0.4)
    _ax2.axvline(0, color='k', lw=1.5, zorder=3)
    _ax2.plot([-2.0, 0], [_EF, _EF], 'g--', linewidth=2)
    _ax2.plot([0, 3.0], [_EF, _EF], 'g--', linewidth=2, label='$E_F$')
    _ax2.plot(_x_all, _Ec_all, 'b-', linewidth=2.5, label='$E_c$')
    _ax2.plot(_x_all, _Ev_all, 'r-', linewidth=2.5, label='$E_v$')

    _Evac_sc = _Ec_all + chi_Si
    _ax2.plot(_x_all, _Evac_sc, 'k-', linewidth=1, alpha=0.4)
    _ax2.plot([-2, 0], [_Evac_sc[0], _Evac_sc[0]], 'k-', linewidth=1, alpha=0.4)

    energy_marker(_ax2, 0.05, _EF, _Ec_all[0], r'$q\Phi_{Bn}$', 'purple',
                  fontsize=16,
                  guides=[(0, _Ec_all[0])])
    energy_marker(_ax2, 2.0, _Ec_all[-1], _Ec_all[0], r'$q\phi_{bi}$', 'brown',
                  fontsize=16,
                  guides=[(0, _Ec_all[0])])
    width_marker(_ax2, 0, _x_dep_vis, _Ev_all.min() - 0.3, '$W_{dep}$', 'gray')
    _ax2.axvline(_x_dep_vis, color='gray', ls=':', lw=1, alpha=0.5)

    _ax2.text(-1.0, _EF - 1.7, 'Metal', fontsize=16, ha='center', fontweight='bold')
    _ax2.text(2.0, _Ev_all[-1] - 0.4, 'N-type', fontsize=16, ha='center', fontweight='bold')
    _ax2.set_xlim(-2.5, 3.5)
    _ax2.set_ylim(min(_Ev_all.min(), _EF) - 1.5, _Ec_all[0] + 1.0)
    _ax2.set_ylabel('Energy (eV)', fontsize=16)
    _ax2.set_title('After Contact — Rectifying (Schottky)', fontsize=16, fontweight='bold')
    _ax2.set_xticks([])
    _ax2.tick_params(labelsize=14)
    _ax2.legend(fontsize=13, loc='upper right')

    plt.tight_layout()

    _caption = mo.md(rf"""
    **Example:** W ($\Psi_M$ = {_phi_m} eV) on N-Si ($N_d$ = {_Nd:.0e} cm$^{{-3}}$).
    **Before:** More electrons in the semiconductor than in the metal at the same energy → electrons diffuse from SC to metal.
    **After:** Fermi levels align. A **depletion layer** forms with bands bending upward.
    $\Phi_{{Bn}}$ = $\Psi_M - \chi$ = {_phi_Bn:.2f} eV (Schottky barrier) &nbsp;|&nbsp;
    $\phi_{{bi}}$ = $\Psi_M - \Psi_S$ = {_phi_bi:.2f} eV (built-in potential).
    The N-side looks like that for a one-sided PN diode → **rectifying behaviour**.
    """)

    _intro = mo.md(r"""


    ## 1. Rectifying Contact: Schottky Diode

    ### N-type Semiconductor ($\Psi_M > \Psi_S$)

    When a metal with $\Psi_M > \Psi_S$ is brought into contact with an N-type semiconductor:

    - In the energy range where there are vacant states, at the same energy, more electrons are in the semiconductor than in the metal.
    - Electrons **diffuse from semiconductor to metal** until the Fermi levels align.
    - This creates a **depletion region** with upward band bending in the semiconductor.
    - The result is a **rectifying** contact known as a **Schottky diode**.
    """)

    _key_rels = mo.md(r"""

    - **Schottky barrier height:** $\boxed{\Phi_{Bn} = \Psi_M - \chi}$; (depends only on metal and semiconductor; independent of doping and bias)
    - **Built-in potential:** $\boxed{\phi_{bi} = \Psi_M - \Psi_S = \Phi_{Bn} - (E_c - E_F)/q}$
    """)

    mo.vstack([_intro, _fig, _key_rels])
    return


@app.cell(hide_code=True)
def _(
    Eg_Si,
    chi_Si,
    energy_marker,
    kT,
    mo,
    ni_Si,
    np,
    plt,
    schottky_bands,
    width_marker,
):
    _phi_m_p = 4.1   # Al
    _Na = 1e16
    _phi_p = kT * np.log(_Na / ni_Si)
    _phi_s_p = chi_Si + (Eg_Si / 2 + _phi_p)
    _phi_Bp = Eg_Si + chi_Si - _phi_m_p
    _phi_bi_p = _phi_s_p - _phi_m_p

    _fig_p, (_ax1p, _ax2p) = plt.subplots(1, 2, figsize=(14, 4.5))

    # === LEFT: Before contact ===
    _Evac = max(_phi_m_p, _phi_s_p) + 0.5
    _EF_m = _Evac - _phi_m_p
    _EF_s = _Evac - _phi_s_p
    _Ec_s = _Evac - chi_Si
    _Ev_s = _Ec_s - Eg_Si

    _ml, _mr = -2.0, -0.3
    _sl, _sr = 0.3, 3.0

    _ax1p.fill_between([_ml, _mr], _EF_m - 1, _EF_m, color='#6baed6', alpha=0.4)
    _ax1p.plot([_ml, _mr], [_EF_m, _EF_m], 'g--', linewidth=2)
    _ax1p.text(_ml + 0.1, _EF_m + 0.08, '$E_{F,M}$', fontsize=14, color='green')
    _ax1p.plot([_ml, _mr], [_Evac, _Evac], 'k-', linewidth=1.5)
    energy_marker(_ax1p, _ml + 0.3, _EF_m, _Evac, r'$q\Psi_M$', 'purple')

    _ax1p.plot([_sl, _sr], [_Ec_s, _Ec_s], 'b-', linewidth=2.5)
    _ax1p.plot([_sl, _sr], [_Ev_s, _Ev_s], 'r-', linewidth=2.5)
    _ax1p.plot([_sl, _sr], [_EF_s, _EF_s], 'g--', linewidth=2)
    _ax1p.text(_sr - 0.5, _EF_s - 0.15, '$E_{F,S}$', fontsize=14, color='green')
    _ax1p.plot([_sl, _sr], [_Evac, _Evac], 'k-', linewidth=1.5)
    energy_marker(_ax1p, _sr - 0.3, _Ec_s, _Evac, r'$q\chi$', 'orange')
    energy_marker(_ax1p, _sl + 0.3, _EF_s, _Evac, r'$q\Psi_S$', 'purple')

    _ax1p.annotate('', xy=(0.15, (_EF_m + _EF_s) / 2), xytext=(-0.15, (_EF_m + _EF_s) / 2),
                   arrowprops=dict(arrowstyle='->', color='#e6550d', lw=2.5))
    _ax1p.text(0.0, (_EF_m + _EF_s) / 2 + 0.15, '$e^-$', fontsize=14, ha='center', color='#e6550d')

    _ax1p.text((_ml + _mr) / 2, _EF_m - 1.3, 'Metal', fontsize=16, ha='center', fontweight='bold')
    _ax1p.text((_sl + _sr) / 2, _Ev_s - 0.5, 'P-type', fontsize=16, ha='center', fontweight='bold')
    _ax1p.text(0, _Evac + 0.15, '$E_{vac}$', fontsize=14, ha='center')
    _ax1p.set_xlim(_ml - 0.3, _sr + 0.5)
    _ax1p.set_ylim(min(_Ev_s, _EF_s) - 1.0, _Evac + 0.5)
    _ax1p.set_ylabel('Energy (eV)', fontsize=16)
    _ax1p.set_title(r'Before Contact ($\Psi_M < \Psi_S$)', fontsize=16, fontweight='bold')
    _ax1p.set_xticks([])
    _ax1p.tick_params(labelsize=14)

    # === RIGHT: After contact ===
    _x_dep_vis = 1.5
    _x_all, _Ec_all, _Ev_all = schottky_bands(_phi_m_p, _phi_s_p, _x_dep_vis)
    _EF = 0.0

    _ax2p.fill_between([-2.0, 0], _EF - 1.5, _EF, color='#6baed6', alpha=0.4)
    _ax2p.axvline(0, color='k', lw=1.5, zorder=3)
    _ax2p.plot([-2.0, 0], [_EF, _EF], 'g--', linewidth=2)
    _ax2p.plot([0, 3.0], [_EF, _EF], 'g--', linewidth=2, label='$E_F$')
    _ax2p.plot(_x_all, _Ec_all, 'b-', linewidth=2.5, label='$E_c$')
    _ax2p.plot(_x_all, _Ev_all, 'r-', linewidth=2.5, label='$E_v$')

    energy_marker(_ax2p, 0.05, _Ev_all[0], _EF, r'$q\Phi_{Bp}$', 'purple',
                  fontsize=16,
                  guides=[(0, _Ev_all[0])])
    energy_marker(_ax2p, 2.0, _Ev_all[0], _Ev_all[-1], r'$q\phi_{bi}$', 'brown',
                  fontsize=16,
                  guides=[(0, _Ev_all[0])])
    width_marker(_ax2p, 0, _x_dep_vis, _Ev_all.min() - 0.3, '$W_{dep}$', 'gray')
    _ax2p.axvline(_x_dep_vis, color='gray', ls=':', lw=1, alpha=0.5)

    _ax2p.text(-1.0, _EF - 1.7, 'Metal', fontsize=16, ha='center', fontweight='bold')
    _ax2p.text(2.0, _EF - 1.7, 'P-type', fontsize=16, ha='center', fontweight='bold')
    _ax2p.set_xlim(-2.5, 3.5)
    _ax2p.set_ylim(min(_Ev_all.min(), _EF) - 1.5, _Ec_all[0] + 1.0)
    _ax2p.set_ylabel('Energy (eV)', fontsize=16)
    _ax2p.set_title('After Contact — Rectifying (Schottky)', fontsize=16, fontweight='bold')
    _ax2p.set_xticks([])
    _ax2p.tick_params(labelsize=14)
    _ax2p.legend(fontsize=13, loc='upper right')

    plt.tight_layout()

    _intro_p = mo.md(r"""
    ### P-type Semiconductor ($\Psi_M < \Psi_S$)

    When a metal with $\Psi_M < \Psi_S$ is brought into contact with a P-type semiconductor:

    - Electrons **diffuse from metal to semiconductor**.
    - A **depletion region** forms with bands bending downward in the semiconductor.
    - The P-side looks like that for a PN diode (if the metal is taken to be like the N-side) → **rectifying behaviour**.

    $$\boxed{\Phi_{Bp} = E_g/q - (\Psi_M - \chi) = E_g/q + \chi - \Psi_M}$$

    $$\boxed{q\phi_{bi} = q(\Psi_S - \Psi_M)}$$
    """)

    _caption_p = mo.md(rf"""
    **Example:** Al ($\Psi_M$ = {_phi_m_p} eV) on P-Si ($N_a$ = {_Na:.0e} cm$^{{-3}}$).
    **Before:** Electrons diffuse from metal to semiconductor.
    **After:** Bands bend *downward*. The hole barrier $\Phi_{{Bp}}$ = $E_g/q + \chi - \Psi_M$ = {_phi_Bp:.2f} eV.
    Built-in potential $\phi_{{bi}}$ = $\Psi_S - \Psi_M$ = {_phi_bi_p:.2f} V.
    """)

    mo.vstack([_intro_p, _fig_p])
    return


@app.cell(hide_code=True)
def _(mo):
    _heading = mo.md(r"""


    ## 2. Schottky Barrier and Built-in Potential under Bias
    """)

    _table_header = mo.md(r"""### Schottky Barrier Heights and Work Functions 

    The Schottky barrier $\Phi_{Bn}$ is a property of the metal-semiconductor interface. It does **not** change with applied voltage.

    Hu, Tables 4-4 and 4-5

    """)

    _table = mo.md(r"""
    | | **Mg** | **Ti** | **Cr** | **W** | **Mo** | **Pd** | **Au** | **Pt** |
    |---|---|---|---|---|---|---|---|---|
    | $\Phi_{Bn}$ (V) | 0.4 | 0.5 | 0.61 | 0.67 | 0.68 | 0.77 | 0.8 | 0.9 |
    | $\Phi_{Bp}$ (V) | | 0.61 | 0.50 | | 0.42 | | 0.3 | |
    | $\Psi_M$ (V) | 3.7 | 4.3 | 4.5 | 4.6 | 4.6 | 5.1 | 5.1 | 5.7 |
    """)

    _centered_table = mo.Html(
        f'<div style="width:100%; display:flex; justify-content:center;">'
        f'{_table.text}'
        f'</div>'
    )

    _table_footer = mo.md(r"""
    **Note:** In IC fabrication, **metal silicides** (e.g., TiSi$_2$, CoSi$_2$, NiSi) are much more frequently used instead of pure metals.
    """)


    _silicide_table = mo.md(r"""
    | | **ErSi$_{1.7}$** | **HfSi** | **MoSi$_2$** | **ZrSi$_2$** | **TiSi$_2$** | **CoSi$_2$** | **WSi$_2$** | **NiSi$_2$** | **Pd$_2$Si** | **PtSi** |
    |---|---|---|---|---|---|---|---|---|---|---|
    | $\Phi_{Bn}$ (V) | 0.28 | 0.45 | 0.55 | 0.55 | 0.61 | 0.65 | 0.67 | 0.67 | 0.75 | 0.87 |
    | $\Phi_{Bp}$ (V) | | | 0.55 | 0.55 | 0.49 | 0.45 | 0.43 | 0.43 | 0.35 | 0.23 |
    """)

    _centered_silicide = mo.Html(
        f'<div style="width:100%; display:flex; justify-content:center;">'
        f'{_silicide_table.text}'
        f'</div>'
    )

    mo.vstack([
        _heading,
        _table_header, _centered_table, _table_footer, _centered_silicide,
    ])
    return


@app.cell(hide_code=True)
def _(
    Eg_Si,
    IMAGE_BASE,
    chi_Si,
    energy_marker,
    kT,
    mo,
    ni_Si,
    np,
    plt,
    schottky_bands_biased,
):
    _bias_text = mo.md(r"""
    ### Built-in Potential under Bias

    - **Forward bias** ($V_a > 0$): Applied voltage reduces the built-in potential to $\phi_{bi} - V_a$, lowering the barrier seen by electrons in the semiconductor → current increases exponentially.
    - **Reverse bias** ($V_a < 0$): The potential barrier increases to $\phi_{bi} + |V_a|$ → only a small saturation current $I_0$ flows.
    """)

    _bias_img = mo.hstack(
        [mo.image(f"{IMAGE_BASE}/schottky-orientation.png", width=600)],
        justify="center",
    )

    _phi_m_b = 4.6   # W on n-Si
    _Nd_b = 1e16
    _phi_n_b = kT * np.log(_Nd_b / ni_Si)
    _phi_s_b = chi_Si + (Eg_Si / 2 - _phi_n_b)
    _phi_Bn_b = _phi_m_b - chi_Si
    _phi_bi_b = _phi_m_b - _phi_s_b

    _fig_bias, (_ax_fwd, _ax_rev) = plt.subplots(1, 2, figsize=(14, 4.5))

    for _ax, _Va, _title in [(_ax_fwd, 0.2, 'Forward Bias ($V_a > 0$)'),
                              (_ax_rev, -1.0, 'Reverse Bias ($V_a < 0$)')]:
        _Veff = _phi_bi_b - _Va
        _x_dep_vis = min(2.5, max(0.5, 0.8 + abs(_Veff) * 0.5))
        _x_all, _Ec_all, _Ev_all = schottky_bands_biased(
            _phi_m_b, _phi_s_b, _Va, _x_dep_vis)
        _EF_M = 0.0
        _EF_S = _Va

        _ax.fill_between([-2, 0], _EF_M - 1.5, _EF_M, color='#6baed6', alpha=0.3)
        _ax.axvline(0, color='k', lw=1.5, zorder=3)
        _ax.plot([-2, 0], [_EF_M, _EF_M], 'g--', linewidth=2)
        _ax.plot(_x_all, _Ec_all, 'b-', linewidth=2.5)
        _ax.plot(_x_all, _Ev_all, 'r-', linewidth=2.5)

        _ax.plot([max(0.5, _x_dep_vis * 0.5), 3.0], [_EF_S, _EF_S], 'g--', linewidth=2, alpha=0.7)
        _ax.text(-1.8, _EF_M + 0.08, '$E_{F,M}$', fontsize=14, color='green')
        _ax.text(2.2, _EF_S + 0.08, '$E_{F,S}$', fontsize=14, color='green')

        energy_marker(_ax, 0.05, _EF_M, _Ec_all[0], r'$q\Phi_{Bn}$', 'purple',
                      guides=[(0, _Ec_all[0])])
        energy_marker(_ax, 1.8, _Ec_all[-1], _Ec_all[0], r'$q(\phi_{bi} - V_a)$', 'brown',
                      fontsize=13,
                      guides=[(0, _Ec_all[0])])

        _ax.set_title(_title, fontsize=16, fontweight='bold')
        _ax.set_ylabel('Energy (eV)', fontsize=16)
        _ax.set_xlabel('Position', fontsize=16)
        _ax.set_xlim(-2.5, 3.5)
        _y_lo = min(_Ev_all.min(), _EF_M, _EF_S) - 1.5
        _y_hi = _Ec_all[0] + 1.0
        _ax.set_ylim(_y_lo, _y_hi)
        _ax.set_xticks([])
        _ax.tick_params(labelsize=14)
        _ax.text(-1.0, _y_lo + 0.3, 'Metal', fontsize=16, ha='center', fontweight='bold')
        _ax.text(2.0, _Ev_all[-1] - 0.4, 'N-type', fontsize=16, ha='center', fontweight='bold')

    plt.tight_layout()

    mo.vstack([_bias_text, _bias_img, _fig_bias])
    return


@app.cell(hide_code=True)
def _(metal_data, mo):
    metal_es_select = mo.ui.dropdown(
        options={name: name for name in metal_data},
        value="W",
        label="Metal"
    )
    sc_type_es_select = mo.ui.dropdown(
        options={"N-type": "n", "P-type": "p"},
        value="N-type",
        label="Semiconductor type"
    )
    log_Ndop_es_slider = mo.ui.slider(
        start=14, stop=19, value=16, step=0.5,
        label=r"log$_{10}$(Doping) [cm$^{-3}$]"
    )
    Va_es_slider = mo.ui.slider(
        start=-3.0, stop=1, value=0.0, step=0.05,
        label=r"$V_a$ (V)"
    )

    es_controls = mo.vstack([
        mo.md(r"""## 3. Interactive Explorer: Energy Bands & Electrostatics

    The electrostatics of a Schottky diode are identical to the one-sided PN junction (with the metal acting as the heavily-doped side). Using the depletion approximation:

    | | **N-type** ($\rho = +qN_d$) | **P-type** ($\rho = -qN_a$) |
    |---|---|---|
    | **Charge density** | $\rho(x) = qN_d$ | $\rho(x) = -qN_a$ |
    | **Electric field** | $\mathcal{E}(x) = -\dfrac{qN_d}{\epsilon_s}(W_{dep} - x)$ | $\mathcal{E}(x) = +\dfrac{qN_a}{\epsilon_s}(W_{dep} - x)$ |
    | **Depletion width** | $W_{dep} = \sqrt{\dfrac{2\epsilon_s(\phi_{bi} - V_a)}{qN_d}}$ | $W_{dep} = \sqrt{\dfrac{2\epsilon_s(\phi_{bi} - V_a)}{qN_a}}$ |

    for $0 \le x \le W_{dep}$, where $V_a$ is the applied voltage (positive for forward bias).

    Select a metal, semiconductor type, doping concentration, and applied voltage to explore the energy bands and electrostatics.
    """),
        mo.hstack([metal_es_select, sc_type_es_select], justify="start"),
        mo.hstack([log_Ndop_es_slider, Va_es_slider], justify="start"),
    ])
    return (
        Va_es_slider,
        es_controls,
        log_Ndop_es_slider,
        metal_es_select,
        sc_type_es_select,
    )


@app.cell(hide_code=True)
def _(
    Eg_Si,
    Nc_Si,
    Va_es_slider,
    chi_Si,
    energy_marker,
    eps_0,
    eps_Si,
    es_controls,
    kT,
    log_Ndop_es_slider,
    metal_data,
    metal_es_select,
    mo,
    ni_Si,
    np,
    plt,
    q,
    sc_type_es_select,
    schottky_bands_biased,
):
    _phi_m = metal_data[metal_es_select.value]
    _sc_type = sc_type_es_select.value
    _N_dop = 10 ** log_Ndop_es_slider.value
    _Va = Va_es_slider.value
    _eps_s = eps_Si * eps_0

    if _sc_type == "n":
        _n = _N_dop
        _Ec_minus_EF = -kT * np.log(_n / Nc_Si)
        _phi_s = chi_Si + _Ec_minus_EF
        _phi_Bn = _phi_m - chi_Si
        _phi_bi = _phi_m - _phi_s
        _dop_label = f"$N_d$ = {_N_dop:.1e} cm$^{{-3}}$"
        _barrier_label = rf"$\Phi_{{Bn}}$ = {_phi_Bn:.3f} eV"
        _n_label = rf"$n \approx N_d$ = {_n:.2e} cm$^{{-3}}$"
    else:
        _n = ni_Si**2 / _N_dop
        _Ec_minus_EF = -kT * np.log(_n / Nc_Si)
        _phi_s = chi_Si + _Ec_minus_EF
        _phi_Bp = Eg_Si + chi_Si - _phi_m
        _phi_bi = _phi_s - _phi_m
        _dop_label = f"$N_a$ = {_N_dop:.1e} cm$^{{-3}}$"
        _barrier_label = rf"$\Phi_{{Bp}}$ = {_phi_Bp:.3f} eV"
        _n_label = rf"$n = n_i^2/N_a$ = {_n:.2e} cm$^{{-3}}$"

    _type_label = "N-type" if _sc_type == "n" else "P-type"
    _is_rectifying = _phi_bi > 0
    _contact_str = "Rectifying (Schottky)" if _is_rectifying else "Ohmic"
    _content = [es_controls]

    if not _is_rectifying:
        _content.append(
            mo.callout(
                mo.md(f"**Ohmic contact** — $\\phi_{{bi}}$ = {_phi_bi:.2f} V < 0. "
                      f"No depletion region forms; accumulation at interface."),
                kind="warn"
            )
        )

    # ── Energy band diagrams ──
    _fig_band, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    _Evac = max(_phi_m, _phi_s) + 0.5
    _EF_m_before = _Evac - _phi_m
    _EF_s_before = _Evac - _phi_s
    _Ec_s = _Evac - chi_Si
    _Ev_s = _Ec_s - Eg_Si

    _ml, _mr = -2.0, -0.3
    _sl, _sr = 0.3, 3.0

    _ax1.fill_between([_ml, _mr], _EF_m_before - 1, _EF_m_before, color='#6baed6', alpha=0.4)
    _ax1.plot([_ml, _mr], [_EF_m_before, _EF_m_before], 'g--', linewidth=2)
    _ax1.text(_ml + 0.1, _EF_m_before + 0.08, '$E_{F,M}$', fontsize=14, color='green')
    _ax1.plot([_ml, _mr], [_Evac, _Evac], 'k-', linewidth=1.5)
    energy_marker(_ax1, _ml + 0.3, _EF_m_before, _Evac, r'$q\Psi_M$', 'purple')

    _ax1.plot([_sl, _sr], [_Ec_s, _Ec_s], 'b-', linewidth=2.5, label='$E_c$')
    _ax1.plot([_sl, _sr], [_Ev_s, _Ev_s], 'r-', linewidth=2.5, label='$E_v$')
    _ax1.plot([_sl, _sr], [_EF_s_before, _EF_s_before], 'g--', linewidth=2)
    _ax1.text(_sr - 0.5, _EF_s_before + 0.08, '$E_{F,S}$', fontsize=14, color='green')
    _ax1.plot([_sl, _sr], [_Evac, _Evac], 'k-', linewidth=1.5)
    energy_marker(_ax1, _sr - 0.3, _Ec_s, _Evac, r'$q\chi$', 'orange')
    energy_marker(_ax1, _sl + 0.3, _EF_s_before, _Evac, r'$q\Psi_S$', 'purple')

    _ax1.text((_ml + _mr) / 2, _EF_m_before - 1.3, 'Metal', fontsize=16, ha='center', fontweight='bold')
    _ax1.text((_sl + _sr) / 2, _Ev_s - 0.4, f'{_type_label} SC', fontsize=16, ha='center', fontweight='bold')
    _ax1.text(0, _Evac + 0.15, '$E_{vac}$', fontsize=14, ha='center')
    _ax1.set_xlim(_ml - 0.3, _sr + 0.5)
    _ax1.set_ylim(min(_Ev_s, _EF_m_before, _EF_s_before) - 1.5, _Evac + 0.5)
    _ax1.set_ylabel('Energy (eV)', fontsize=16)
    _ax1.set_title('Before Contact', fontsize=16, fontweight='bold')
    _ax1.set_xticks([])
    _ax1.tick_params(labelsize=14)

    _Veff_band = _phi_bi - _Va
    _x_dep_vis = min(2.5, max(0.3, 0.8 + abs(_Veff_band) * 0.5))
    _x_all, _Ec_all, _Ev_all = schottky_bands_biased(
        _phi_m, _phi_s, _Va, _x_dep_vis)
    _EF_M = 0.0
    _EF_S = _Va

    _ax2.fill_between([-2.0, 0], _EF_M - 1.5, _EF_M, color='#6baed6', alpha=0.4)
    _ax2.axvline(0, color='k', lw=1.5, zorder=3)
    _ax2.plot([-2.0, 0], [_EF_M, _EF_M], 'g--', linewidth=2)
    _ax2.plot(_x_all, _Ec_all, 'b-', linewidth=2.5, label='$E_c$')
    _ax2.plot(_x_all, _Ev_all, 'r-', linewidth=2.5, label='$E_v$')
    _ax2.plot([max(0.5, _x_dep_vis * 0.5), 3.0], [_EF_S, _EF_S], 'g--',
              linewidth=2, alpha=0.7)
    _ax2.text(-1.8, _EF_M + 0.08, '$E_{F,M}$', fontsize=14, color='green')
    _ax2.text(2.2, _EF_S + 0.08, '$E_{F,S}$', fontsize=14, color='green')

    if _is_rectifying:
        if _sc_type == "n":
            energy_marker(_ax2, 0.05, _EF_M, _Ec_all[0], r'$q\Phi_{Bn}$', 'purple',
                          guides=[(0, _Ec_all[0])])
            energy_marker(_ax2, 1.8, _Ec_all[-1], _Ec_all[0],
                          r'$q(\phi_{bi} - V_a)$', 'brown', fontsize=13,
                          guides=[(0, _Ec_all[0])])
        else:
            energy_marker(_ax2, 0.05, _Ev_all[0], _EF_M, r'$q\Phi_{Bp}$', 'purple',
                          guides=[(0, _Ev_all[0])])
            energy_marker(_ax2, 1.8, _Ev_all[0], _Ev_all[-1],
                          r'$q(\phi_{bi} - V_a)$', 'brown', fontsize=13,
                          guides=[(0, _Ev_all[0])])

    if _Va == 0:
        _title_bias = "Equilibrium"
    elif _Va > 0:
        _title_bias = f"Forward Bias ($V_a$ = {_Va:.2f} V)"
    else:
        _title_bias = f"Reverse Bias ($V_a$ = {_Va:.2f} V)"
    _ax2.set_title(f'After Contact — {_title_bias}', fontsize=16, fontweight='bold')
    _ax2.set_ylabel('Energy (eV)', fontsize=16)
    _ax2.set_xlabel('Position', fontsize=16)
    _ax2.tick_params(labelsize=14)
    _ax2.set_xlim(-2.5, 3.5)
    _y_lo = min(_Ev_all.min(), _EF_M, _EF_S) - 1.5
    _y_hi = _Ec_all[0] + 1.0
    _ax2.set_ylim(_y_lo, _y_hi)
    _ax2.text(-1.0, _y_lo + 0.3, 'Metal', fontsize=16, ha='center', fontweight='bold')
    _ax2.text(2.0, _Ev_all[-1] - 0.4, f'{_type_label}', fontsize=16, ha='center', fontweight='bold')

    plt.tight_layout()
    _content.append(_fig_band)

    # ── Electrostatics ──
    _Veff = max(_phi_bi - _Va, 0.001) if _is_rectifying else max(abs(_phi_bi) + abs(_Va), 0.001)
    _Wdep = np.sqrt(2 * _eps_s * _Veff / (q * _N_dop))
    _Wdep_um = _Wdep * 1e4

    _x = np.linspace(0, _Wdep, 500)
    _x_um = _x * 1e4
    _x_ext = np.linspace(_Wdep, _Wdep * 1.5, 50)
    _x_ext_um = _x_ext * 1e4

    if _sc_type == "n":
        _rho_scale = np.full_like(_x, _N_dop)
        _E_field = -q * _N_dop / _eps_s * (_Wdep - _x)
        _E_max = -q * _N_dop * _Wdep / _eps_s
        _V = -q * _N_dop / (2 * _eps_s) * (_Wdep - _x) ** 2 + _Veff
        _V_flat = _Veff
    else:
        _rho_scale = np.full_like(_x, -_N_dop)
        _E_field = q * _N_dop / _eps_s * (_Wdep - _x)
        _E_max = q * _N_dop * _Wdep / _eps_s
        _V = q * _N_dop / (2 * _eps_s) * (_Wdep - _x) ** 2 - _Veff
        _V_flat = -_Veff

    _fig_es, (_ax_rho, _ax_E, _ax_V) = plt.subplots(1, 3, figsize=(18, 4.5))

    _ax_rho.plot(_x_um, _rho_scale * 1e-15, 'b-', linewidth=2.5)
    _ax_rho.plot(_x_ext_um, np.zeros_like(_x_ext), 'b-', linewidth=2.5)
    _ax_rho.axhline(0, color='k', linewidth=0.5)
    _ax_rho.set_ylabel(r'$\rho / q$ ($\times 10^{15}$ cm$^{-3}$)', fontsize=16)
    _ax_rho.set_xlabel(r'Position ($\mu$m)', fontsize=16)
    _ax_rho.set_title(r'Charge density $\rho(x)$', fontsize=16, fontweight='bold')
    _ax_rho.fill_between(_x_um, 0, _rho_scale * 1e-15, alpha=0.2, color='blue')
    _ax_rho.tick_params(labelsize=14)
    _ax_rho.grid(True, alpha=0.3)

    _ax_E.plot(_x_um, _E_field, 'r-', linewidth=2.5)
    _ax_E.plot(_x_ext_um, np.zeros_like(_x_ext), 'r-', linewidth=2.5)
    _ax_E.axhline(0, color='k', linewidth=0.5)
    _ax_E.set_ylabel(r'$\mathcal{E}$ (V/cm)', fontsize=16)
    _ax_E.set_xlabel(r'Position ($\mu$m)', fontsize=16)
    _ax_E.set_title(r'Electric field $\mathcal{E}(x)$', fontsize=16, fontweight='bold')
    _ax_E.fill_between(_x_um, 0, _E_field, alpha=0.2, color='red')
    _ax_E.tick_params(labelsize=14)
    _ax_E.grid(True, alpha=0.3)
    _ax_E.text(0.95, 0.95, f'$\\mathcal{{E}}_{{max}}$ = {_E_max:.2e} V/cm',
               fontsize=13, transform=_ax_E.transAxes, va='top', ha='right',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    _ax_V.plot(_x_um, _V, 'g-', linewidth=2.5)
    _ax_V.plot(_x_ext_um, np.full_like(_x_ext, _V_flat), 'g-', linewidth=2.5)
    _ax_V.axhline(0, color='k', linewidth=0.5)
    _ax_V.set_ylabel(r'$V(x)$ (V)', fontsize=16)
    _ax_V.set_xlabel(r'Position ($\mu$m)', fontsize=16)
    _ax_V.set_title(r'Potential $V(x)$', fontsize=16, fontweight='bold')
    _ax_V.fill_between(_x_um, 0, _V, alpha=0.2, color='green')
    _ax_V.tick_params(labelsize=14)
    _ax_V.grid(True, alpha=0.3)

    _fig_es.suptitle(
        f'{metal_es_select.value} on {_type_label} Si — {_contact_str} '
        f'({_dop_label}, $V_a$ = {_Va:.2f} V)',
        fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    _content.append(_fig_es)

    # ── Info summary ──
    _contact_type = "**Rectifying** (Schottky)" if _is_rectifying else "**Ohmic**"
    _info_es = mo.md(
        f"""
        **Metal:** {metal_es_select.value} ($\\Psi_M$ = {_phi_m:.2f} eV) &nbsp;|&nbsp;
        $\\Psi_S$ = {_phi_s:.2f} eV &nbsp;|&nbsp;
        **Type:** {_type_label} ({_contact_str}) &nbsp;|&nbsp;
        {_barrier_label} &nbsp;|&nbsp;
        $\\phi_{{bi}}$ = {_phi_bi:.3f} V &nbsp;|&nbsp;
        $\\phi_{{bi}} - V_a$ = {_phi_bi - _Va:.3f} V &nbsp;|&nbsp;
        $W_{{dep}}$ = {_Wdep_um:.3f} $\\mu$m &nbsp;|&nbsp;
        $\\mathcal{{E}}_{{max}}$ = {abs(_E_max):.2e} V/cm
        """
    )

    _psi_s_calc = mo.md(
        rf"""
        {_n_label} &nbsp;|&nbsp;
        $(E_c - E_F)/q = -kT \ln(n/N_c) = -{kT:.4f} \times \ln({_n:.2e}/{Nc_Si:.2e}) = {_Ec_minus_EF:.3f}$ eV

        $\Psi_S = \chi + (E_c - E_F)/q = {chi_Si:.2f} + {_Ec_minus_EF:.3f} = {_phi_s:.3f}$ eV
        """
    )

    _pinning_note = mo.callout(
        mo.md(r"""The Schottky–Mott rule ($\Phi_{Bn} = \Psi_M - \chi$) used here is the **ideal** model.
    Measured barrier heights (Table 4-4) are different because of **Fermi level pinning** by interface states at the Si surface. The empirical values should be used for device design."""),
        kind="info"
    )

    _content.append(_info_es)
    _content.append(_psi_s_calc)
    _content.append(_pinning_note)

    mo.vstack(_content)
    return


@app.cell(hide_code=True)
def _(mo):
    phi_Bn_iv_slider = mo.ui.slider(
        start=0.3, stop=1.0, value=0.55, step=0.01,
        label=r"$\Phi_{Bn}$ (eV)"
    )
    area_slider = mo.ui.slider(
        start=-6, stop=-2, value=-4, step=0.5,
        label=r"log$_{10}$(Area) [cm$^2$]"
    )

    iv_controls = mo.vstack([
        mo.md("### Interactive I-V relation"),
        mo.hstack([phi_Bn_iv_slider, area_slider], justify="start"),
    ])
    return area_slider, iv_controls, phi_Bn_iv_slider


@app.cell(hide_code=True)
def _(
    A_star_n,
    T,
    area_slider,
    iv_controls,
    kT,
    mo,
    np,
    phi_Bn_iv_slider,
    plt,
):
    _intro_iv = mo.md(r"""


    ## 5. IV Characteristics

    The Schottky diode current is dominated by **thermionic emission**. Majority carriers are injected over the barrier:

    $$I = I_0 \left( e^{V_a / V_T} - 1 \right), \quad I_0 = A \cdot A^* T^2 \exp\left(-\frac{q\Phi_{Bn}}{kT}\right)$$

    - $A^*$ is the **Richardson constant**: 110 A/cm$^2$/K$^2$ (N-Si), 32 A/cm$^2$/K$^2$ (P-Si)
    - The Schottky diode is a **majority carrier device**, so can have faster switching than PN diodes (no minority carrier recombination)
    - $I_0$ is typically larger than for PN diodes → useful for low voltage, high current applications
    """)

    _phi_Bn = phi_Bn_iv_slider.value
    _area = 10 ** area_slider.value

    _I0 = _area * A_star_n * T**2 * np.exp(-_phi_Bn / kT)

    _Va_fwd = np.linspace(0, 0.6, 300)
    _Va_rev = np.linspace(-3, 0, 100)
    _Va_all = np.concatenate([_Va_rev, _Va_fwd])

    _I_all = _I0 * (np.exp(_Va_all / kT) - 1)
    _I_clamp = np.clip(_I_all, -_I0 * 10, _I0 * np.exp(0.6 / kT))

    _fig_iv, (_ax_lin, _ax_log) = plt.subplots(1, 2, figsize=(14, 4.5))

    _ax_lin.plot(_Va_all, _I_clamp * 1e3, 'b-', linewidth=2.5)
    _ax_lin.axhline(0, color='k', linewidth=0.5)
    _ax_lin.axvline(0, color='k', linewidth=0.5)
    _ax_lin.set_xlabel(r'$V_a$ (V)', fontsize=16)
    _ax_lin.set_ylabel(r'$I$ (mA)', fontsize=16)
    _ax_lin.set_title('IV Characteristic (Linear)', fontsize=16, fontweight='bold')
    _ax_lin.tick_params(labelsize=14)
    _ax_lin.grid(True, alpha=0.3)

    _Va_fwd_plot = np.linspace(0.01, 0.6, 300)
    _I_fwd = _I0 * (np.exp(_Va_fwd_plot / kT) - 1)
    _ax_log.semilogy(_Va_fwd_plot, _I_fwd, 'b-', linewidth=2.5, label='Schottky')
    _ax_log.axhline(_I0, color='r', linestyle='--', alpha=0.7, linewidth=1.5, label=f'$I_0$ = {_I0:.2e} A')
    _ax_log.set_xlabel(r'$V_a$ (V)', fontsize=16)
    _ax_log.set_ylabel(r'$I$ (A)', fontsize=16)
    _ax_log.set_title('IV Characteristic (Semi-log)', fontsize=16, fontweight='bold')
    _ax_log.legend(fontsize=13)
    _ax_log.tick_params(labelsize=14)
    _ax_log.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    _info_iv = mo.md(
        f"""
        $\\Phi_{{Bn}}$ = {_phi_Bn:.2f} eV &nbsp;|&nbsp;
        Area = {_area:.1e} cm$^2$ &nbsp;|&nbsp;
        $I_0$ = {_I0:.2e} A &nbsp;|&nbsp;
        $A^*$ = {A_star_n} A/cm$^2$/K$^2$
        """
    )

    mo.vstack([_intro_iv, iv_controls, _fig_iv, _info_iv])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Summary: Rectifying vs. Ohmic

    |  | N-type | P-type |
    |--|:--------:|:--------:|
    | $\Psi_M > \Psi_S$ | **Rectifying** (Schottky) | Ohmic |
    | $\Psi_M < \Psi_S$ | Ohmic | **Rectifying** (Schottky) |

    **Key relationships:**

    - **N-type Schottky barrier:** $\Phi_{Bn} = \Psi_M - \chi$
    - **P-type Schottky barrier:** $\Phi_{Bp} = E_g/q - (\Psi_M - \chi)$
    - **Built-in potential:** $\phi_{bi} = \lvert \Psi_M - \Psi_S \rvert$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _question = mo.md(r""" ## Checkpoint

    **Question:** If a metal is in contact with an N-type semiconductor and $\Psi_M < \Psi_S$, will the energy bands in the semiconductor bend **upward** or **downward** toward the metal?

    *Think about where the electrons are and which way they flow!*
    """)


    _answer = mo.ui.text(placeholder="Type your answer here...")
    _reveal = mo.accordion({
        "Reveal Answer": mo.md(r"""
        The bands bend **downward** toward the metal.

        Since $\Psi_M < \Psi_S$: more electrons are in the metal than the semiconductor at the same energy. Electrons flow from the metal to the semiconductor, creating an **accumulation** region near the interface. There is no barrier for electrons → **Ohmic** contact.

        The built-in potential $\phi_{bi} = \Psi_M - \Psi_S < 0$.
        """)
    })

    mo.vstack([_question, _answer, _reveal])
    return


@app.cell(hide_code=True)
def _(Eg_Si, chi_Si, kT, mo, ni_Si, np, plt, schottky_bands):
    _fig_ohm, (_ax_on, _ax_op) = plt.subplots(1, 2, figsize=(14, 4.5))

    # === LEFT: N-type Ohmic (Mg on n-Si) ===
    _phi_m_on = 3.7
    _Nd_on = 1e16
    _phi_n_on = kT * np.log(_Nd_on / ni_Si)
    _phi_s_on = chi_Si + (Eg_Si / 2 - _phi_n_on)
    _phi_Bn_on = _phi_m_on - chi_Si
    _phi_bi_on = _phi_m_on - _phi_s_on

    _x_dep_vis = 1.2
    _x_all_n, _Ec_all_n, _Ev_all_n = schottky_bands(_phi_m_on, _phi_s_on, _x_dep_vis)
    _EF = 0.0

    _ax_on.fill_between([-2, 0], _EF - 1.5, _EF, color='#6baed6', alpha=0.3)
    _ax_on.axvline(0, color='k', lw=1.5, zorder=3)
    _ax_on.plot([-2, 0], [_EF, _EF], 'g--', linewidth=2)
    _ax_on.plot([0, 3.0], [_EF, _EF], 'g--', linewidth=2)
    _ax_on.plot(_x_all_n, _Ec_all_n, 'b-', linewidth=2.5, label='$E_c$')
    _ax_on.plot(_x_all_n, _Ev_all_n, 'r-', linewidth=2.5, label='$E_v$')


    _ax_on.text(0.3, _Ec_all_n[-1] + 0.25, 'No barrier\nfor electrons', fontsize=13,
                ha='center', color='#e6550d', fontweight='bold')

    _ax_on.set_title(r'N-type Ohmic: $\Psi_M < \Psi_S$ (Mg on n-Si)', fontsize=15, fontweight='bold')
    _ax_on.text(-1.0, _EF - 1.7, 'Metal', fontsize=15, ha='center', fontweight='bold')
    _ax_on.text(2.0, _Ev_all_n[-1] - 0.4, 'N-type', fontsize=15, ha='center', fontweight='bold')
    _ax_on.legend(fontsize=13, loc='lower right')
    _ax_on.set_xlim(-2.5, 3.5)
    _ax_on.set_ylim(min(_Ev_all_n.min(), _EF) - 1.5, max(_Ec_all_n.max(), _EF) + 1.0)
    _ax_on.set_xticks([])
    _ax_on.set_ylabel('Energy (eV)', fontsize=16)
    _ax_on.tick_params(labelsize=14)

    # === RIGHT: P-type Ohmic (Pt on p-Si) ===
    _phi_m_op = 5.7
    _Na_op = 1e16
    _phi_p_op = kT * np.log(_Na_op / ni_Si)
    _phi_s_op = chi_Si + (Eg_Si / 2 + _phi_p_op)
    _phi_Bp_op = Eg_Si + chi_Si - _phi_m_op
    _phi_bi_op = _phi_s_op - _phi_m_op

    _x_all_p, _Ec_all_p, _Ev_all_p = schottky_bands(_phi_m_op, _phi_s_op, _x_dep_vis)
    _Ec_all_p = _Ec_all_p + 0.5
    _Ev_all_p = _Ev_all_p + 0.5

    _ax_op.fill_between([-2, 0], _EF - 1.5, _EF, color='#6baed6', alpha=0.3)
    _ax_op.axvline(0, color='k', lw=1.5, zorder=3)
    _ax_op.plot([-2, 0], [_EF, _EF], 'g--', linewidth=2)
    _ax_op.plot([0, 3.0], [_EF, _EF], 'g--', linewidth=2)
    _ax_op.plot(_x_all_p, _Ec_all_p, 'b-', linewidth=2.5, label='$E_c$')
    _ax_op.plot(_x_all_p, _Ev_all_p, 'r-', linewidth=2.5, label='$E_v$')

    _ax_op.text(0.3, _Ev_all_p[-1] - 0.5, 'No barrier\nfor holes', fontsize=13,
                ha='center', color='#e6550d', fontweight='bold')

    _ax_op.set_title(r'P-type Ohmic: $\Psi_M > \Psi_S$ (Pt on p-Si)', fontsize=15, fontweight='bold')
    _ax_op.text(-1.0, _EF - 0.5, 'Metal', fontsize=15, ha='center', fontweight='bold')
    _ax_op.text(2.0, _Ev_all_p[-1] - 0.9, 'P-type', fontsize=15, ha='center', fontweight='bold')
    _ax_op.legend(fontsize=13, loc='upper right')
    _ax_op.set_xlim(-2.5, 3.5)
    _ax_op.set_ylim(min(_Ev_all_p.min(), _EF) - 1.5, max(_Ec_all_p.max(), _EF) + 1.0)
    _ax_op.set_xticks([])
    _ax_op.set_ylabel('Energy (eV)', fontsize=16)
    _ax_op.tick_params(labelsize=14)

    plt.tight_layout()

    _intro_ohm = mo.md(r"""

    ## 6. Ohmic Contacts

    ### N-type Semiconductor ($\Psi_M < \Psi_S$)

    - More electrons are in the metal than the semiconductor at the same energy.
    - Electrons **diffuse from metal to semiconductor** → **No built-in barrier** for electrons
        - $V > 0$: Makes the built-in electric field even more positive → electrons from N-side move more easily into metal. **Positive current flow.**
        - $V < 0$: Reduces built-in electric field → electrons from metal move easily into semiconductor. **Negative current flow.**
    - In both directions, current flows readily → **Ohmic (resistive) contact** with linear IV relation.

    ### P-type Semiconductor ($\Psi_M > \Psi_S$)

    - Electrons diffuse from semiconductor to metal → No built-in barrier for holes

    """)

    _caption_ohm = mo.md(r"""
    **Left:** When $\Psi_M < \Psi_S$ for N-type, bands bend *downward* near the metal. There is no barrier for electron flow in either direction → **Ohmic** (linear IV).
    **Right:** When $\Psi_M > \Psi_S$ for P-type, bands bend *upward* near the metal. There is no barrier for hole flow → **Ohmic**.
    """)

    mo.vstack([_intro_ohm, _fig_ohm])
    return


@app.cell(hide_code=True)
def _(
    Eg_Si,
    IMAGE_BASE,
    chi_Si,
    energy_marker,
    kT,
    mo,
    ni_Si,
    np,
    plt,
    schottky_bands,
    width_marker,
):
    _fig_tun, (_ax_mod, _ax_heavy) = plt.subplots(1, 2, figsize=(14, 4.5))

    _phi_m_t = 4.6   # W

    for _ax, _Nd_t, _x_dep_vis, _title in [
        (_ax_mod, 1e16, 1.8, 'Moderate Doping — Rectifying'),
        (_ax_heavy, 1e20, 0.25, 'Heavy Doping — Tunneling (Ohmic)'),
    ]:
        _phi_n_t = kT * np.log(_Nd_t / ni_Si)
        _phi_s_t = chi_Si + (Eg_Si / 2 - _phi_n_t)
        _phi_Bn_t = _phi_m_t - chi_Si
        _phi_bi_t = _phi_m_t - _phi_s_t

        _x_all, _Ec_all, _Ev_all = schottky_bands(_phi_m_t, _phi_s_t, _x_dep_vis)
        _EF = 0.0

        _ax.fill_between([-2, 0], _EF - 1.5, _EF, color='#6baed6', alpha=0.3)
        _ax.axvline(0, color='k', lw=1.5, zorder=3)
        _ax.plot([-2, 0], [_EF, _EF], 'g--', linewidth=2)
        _ax.plot([0, 3.0], [_EF, _EF], 'g--', linewidth=2)
        _ax.plot(_x_all, _Ec_all, 'b-', linewidth=2.5)
        _ax.plot(_x_all, _Ev_all, 'r-', linewidth=2.5)

        energy_marker(_ax, 0.02, _EF, _Ec_all[0], r'$q\Phi_{Bn}$', 'purple',
                      guides=[(0, _Ec_all[0])])
        width_marker(_ax, 0, _x_dep_vis, _Ev_all.min() - 0.3, '$W_{dep}$', 'brown')

        _ax.set_title(_title, fontsize=15, fontweight='bold')
        _ax.set_ylabel('Energy (eV)', fontsize=16)
        _ax.set_xlabel('Position', fontsize=16)
        _ax.set_xticks([])
        _ax.tick_params(labelsize=14)
        _ax.set_xlim(-2.5, 3.5)
        _ax.set_ylim(_Ev_all.min() - 1.2, _Ec_all[0] + 0.8)
        _ax.text(-1.0, _EF - 1.5, 'Metal', fontsize=15, ha='center', fontweight='bold')
        _Nd_label = f'$N_d = 10^{{{int(np.log10(_Nd_t))}}}$'
        _ax.text(2.0, _Ev_all[-1] - 0.4, _Nd_label, fontsize=14, ha='center')

        if _x_dep_vis < 0.5:
            _ax.annotate('', xy=(_x_dep_vis * 0.9, _EF), xytext=(0.02, _EF),
                         arrowprops=dict(arrowstyle='->', color='red', lw=2.5,
                                         connectionstyle='arc3,rad=0.2'))
            _ax.text(_x_dep_vis / 2, _EF + 0.12, 'tunnel', fontsize=14, color='red',
                     ha='center', fontweight='bold')

    plt.tight_layout()

    _intro_tun = mo.md(r"""


    ## 7. Ohmic Contacts in Practice: Tunneling Junctions

    For Si ICs, it is impractical and expensive to use several metals for fabrication. In practice, **ohmic contacts** are formed using:

    1. **Heavy doping** of the semiconductor surface ($N_d$ or $N_a > 10^{19}$ cm$^{-3}$)
    2. **Metal silicides** (e.g., TiSi$_2$, CoSi$_2$, NiSi) — more stable than pure metals

    Even if $\Psi_M > \Psi_S$ (which would normally give a rectifying contact on N-type), heavy doping makes the depletion width so narrow that electrons can **tunnel** through the barrier.

    The tunnelling probability increases exponentially as $W_{dep}$ decreases → low-resistance ohmic contact.

    **Note:** For III-V circuits and devices, it is more common to use different metals for P and N contacts for work function compatibility.
    """)

    _caption_tun = mo.md(r"""
    **Left:** With moderate doping, $W_{dep}$ is wide and the barrier blocks current → **rectifying**.
    **Right:** With heavy doping ($N_d > 10^{19}$ cm$^{-3}$), $W_{dep}$ becomes so narrow (~nm) that electrons can **quantum-mechanically tunnel** through the barrier → **ohmic** behaviour regardless of $\Psi_M$ vs. $\Psi_S$.
    """)

    _tun_img = mo.hstack(
        [mo.image(f"{IMAGE_BASE}/tunnelling.png", width=500)],
        justify="center",
    )

    mo.vstack([_intro_tun, _fig_tun])
    return


@app.cell(hide_code=True)
def _(IMAGE_BASE, mo):
    _contact_r_text = mo.md(r"""
    ### Contact Resistance

    - $R_c$: Specific contact resistance [$\Omega \cdot \mathrm{cm}^2$] 
        - Larger cross-section area, lower resistance

    """)

    _contact_r_img = mo.hstack(
        [mo.image(f"{IMAGE_BASE}/contact-resistance.png", width=800)],
        justify="center",
    )

    mo.vstack([_contact_r_text, _contact_r_img])
    return


if __name__ == "__main__":
    app.run()
