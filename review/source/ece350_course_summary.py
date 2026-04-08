# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "matplotlib==3.10.8",
# ]
# ///

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    BASE = "https://joyce-poon.github.io/ECE350"

    q_C = 1.6e-19
    kT_eV = 0.02585
    ni_si = 1.1e10
    Eg_Si = 1.12
    eps_r_Si = 11.7
    eps_0_cgs = 8.854e-14
    eps_Si = eps_r_Si * eps_0_cgs
    eps_ox = 3.9 * eps_0_cgs
    Nc_Si = 2.86e19
    Nv_Si = 1.08e19
    chi_Si = 4.05
    mu_n_Si = 1350.0
    mu_p_Si = 480.0
    D_n_Si = kT_eV * mu_n_Si
    D_p_Si = kT_eV * mu_p_Si
    return (
        BASE,
        Eg_Si,
        Nc_Si,
        Nv_Si,
        chi_Si,
        eps_0_cgs,
        eps_Si,
        kT_eV,
        mo,
        mu_n_Si,
        mu_p_Si,
        ni_si,
        np,
        plt,
        q_C,
    )


@app.cell
def _(BASE, mo):
    mo.md(rf"""
    # ECE350: Semiconductor Electronic Devices — Course Review

    This notebook is a **complete review** of the course: each unit summarised with its core equations, computed physics figures, and links to the full interactive notebooks.

    Disclosure: This summary was first created by AI (Cursor with Opus 4.6) from the course notes then edited by the instructor.

    **Textbooks:** C. Hu, *Modern Semiconductor Devices for Integrated Circuits*; R. F. Pierret, *Advanced Semiconductor Fundamentals*.

    | Unit | Topic | Interactive notebooks |
    |:------:|:-------:|:-----------------------:|
    | 1 | Crystals & Energy Bands | [crystals]({BASE}/crystals/) · [crystal potential]({BASE}/crystalpotential/) · [energy bands]({BASE}/energybands/) · [effective mass]({BASE}/effectivemass/) |
    | 2 | Equilibrium Statistics | [equilibrium carriers]({BASE}/equilibrium/) |
    | 3 | Carrier Transport | [drift]({BASE}/drift/) · [diffusion]({BASE}/diffusion/) · [G/R]({BASE}/gen_recomb/) · [continuity]({BASE}/continuity/) · [band bending]({BASE}/bandbend/) |
    | 4 | PN Junction | [electrostatics]({BASE}/pn-electrostatics/) · [reverse bias]({BASE}/pn-revbias/) · [I-V]({BASE}/pn-iv/) · [non-idealities]({BASE}/pn-non-idealities/) · [small-signal]({BASE}/pn-small-sig/) · [light/absorption]({BASE}/pn-light-absorption/) |
    | 5 | BJT | [BJT I-V]({BASE}/bjt-iv/) |
    | 6 | Metal-Semiconductor | [M-S interface]({BASE}/m-s-interface/) |
    | 7 | MOS & MOSFET | [MOS cap]({BASE}/moscap/) · [non-idealities]({BASE}/mos-nonidealities/) · [MOSFET intro]({BASE}/mosfet-intro/) · [MOSFET I-V]({BASE}/mosfet-iv/) · [dynamics]({BASE}/mosfet-ac/) · [FinFET/FDSOI]({BASE}/finfet-fdsoi/) |

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How the course fits together

    ```
    Crystals & Bands ─────► What states exist? E(k), band gap, m*
        │
        ▼
    Equilibrium Statistics ─► How many carriers? n₀, p₀, Eꜰ
        │
        ▼
    Transport ─────────────► How do carriers move? J (drift+diffusion), continuity, G/R
        │
        ▼
    Devices
        │
        ▼
        ├──► PN Junction ─────► Diode I-V, depletion, minority injection and diffusion
        │        │
        │        └──► Optoelectronics ► Solar cells, photodetectors, LEDs, lasers
        │
        ├──► BJT ────► Minority diffusion control, β, Gummel
        │     │
        │     └──► Heterojunction bipolar transistors
        │
        ├──► Metal-Semiconductor ► Schottky barriers, ohmic contacts
        │
        └──► MOS / MOSFET ────► Gate-controlled channel, I-V, scaling
    ```

    Every device reuses the **same concepts**: band structure → carrier transport → electrostatics.

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Unit 1: Electron Waves in Crystals and Energy Bands

    *Hu Ch. 1; Pierret Ch. 2–3*

    **Goal:** Explain why semiconductors have **energy bands** and **band gaps**, define **crystal momentum** $\hbar k$ and **effective mass** $m^*$.

    ---

    ### 1.1 Crystal Structure

    - Semiconductors have **periodic** atomic arrangements (diamond cubic for Si, zincblende for GaAs).
    - The **lattice constant** $a$ sets the length scale (Si: $a = 5.43$ Å).
    - **Miller indices** $(hkl)$ describe crystal planes; important for fabrication (wafer orientation), defect behaviour, and anisotropic properties.

    ### 1.2 Bloch's Theorem

    In a periodic potential $U(\mathbf{r}) = U(\mathbf{r}+\mathbf{R})$, the electron wavefunction has the form:

    $$\boxed{\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} \, u_{n\mathbf{k}}(\mathbf{r})}$$

    where $u_{n\mathbf{k}}$ has the periodicity of the lattice.  $n$ is the **band index**, $\mathbf{k}$ is the **crystal momentum** (wave vector).

    ### 1.3 Kronig–Penney Model

    A 1D periodic square-well potential gives a **transcendental equation** whose solutions $E(k)$ show:

    - **Allowed energy bands** separated by **forbidden gaps**.
    - Band structure is periodic in $k$ with period $2\pi/a$ (reciprocal lattice vector).
    - We use the **reduced zone scheme** ($-\pi/a < k \le \pi/a$, first Brillouin zone).

    ### 1.4 Energy Bands in Real Semiconductors

    - **Direct gap** (GaAs): conduction band minimum and valence band maximum at the same $k$ → efficient optical transitions.
    - **Indirect gap** (Si, Ge): minima at different $k$ → phonon-assisted transitions, weaker absorption.
    - Si: $E_g = 1.12$ eV (indirect), GaAs: $E_g = 1.42$ eV (direct).

    ### 1.5 Effective Mass

    Near a band extremum, the dispersion is approximately parabolic:

    $$E(\mathbf{k}) \approx E_{\text{edge}} + \frac{\hbar^2 k^2}{2 m^*}$$

    $$\boxed{m^* = \hbar^2 \left(\frac{d^2E}{dk^2}\right)^{-1}}$$

    - **Electrons** near the conduction band minimum: $m_n^* > 0$.
    - **Holes** near the valence band maximum: $m_p^* > 0$ (defined with sign flip).
    - In general, $m^*$ is a **tensor** (Si has longitudinal and transverse masses).

    $m^*$ determines the **density of states** and the **mobility**. It is the compact bridge between band theory and device equations.

    ---
    """)
    return


@app.cell
def _(np, plt):
    _fig_ek, _ax_ek = plt.subplots(1, 1, figsize=(6, 4.5))

    _k = np.linspace(-np.pi, np.pi, 500)
    _E_free = _k**2 / (2 * np.pi**2) * 10

    _P = 3.0 * np.pi
    _lhs = _P * np.sinc(_k * 0) + np.cos(_k)

    _E1 = 1.0 - 0.4 * np.cos(_k)
    _E2 = 3.2 + 0.6 * np.cos(_k)
    _E3 = 6.0 - 0.9 * np.cos(_k)
    _E4 = 10.0 + 1.1 * np.cos(_k)

    _ax_ek.fill_between(_k / np.pi, _E1.max(), _E2.min(), alpha=0.15, color="gray")
    _ax_ek.fill_between(_k / np.pi, _E2.max(), _E3.min(), alpha=0.15, color="gray")
    _ax_ek.fill_between(_k / np.pi, _E3.max(), _E4.min(), alpha=0.15, color="gray")

    for _Eband, _c, _lbl in [
        (_E1, "#1f77b4", "Band 1 "),
        (_E2, "#d62728", "Band 2 (VB)"),
        (_E3, "#2ca02c", "Band 3 (CB)"),
        (_E4, "#9467bd", "Band 4"),
    ]:
        _ax_ek.plot(_k / np.pi, _Eband, color=_c, lw=2, label=_lbl)

    _ax_ek.set_xlabel(r"$ka/\pi$", fontsize=16)
    _ax_ek.set_ylabel(r"$E$ (arb. units)", fontsize=16)
    _ax_ek.set_title("Schematic E(k) in the reduced zone scheme", fontsize=16)
    _ax_ek.legend(fontsize=12)
    _ax_ek.set_xlim(-1, 1)
    _ax_ek.tick_params(labelsize=14)

    _ax_ek.annotate(
        "Band gap",
        xy=(0, (_E2.max() + _E3.min()) / 2),
        fontsize=14,
        ha="center",
        color="gray",
    )

    _fig_ek.tight_layout()
    _fig_ek
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Figure: Schematic E(k) dispersion in the reduced zone scheme. Grey shaded regions are **band gaps** (forbidden energies). The curvatures of each band at its extrema determine the effective mass for the bandgap.*

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Unit 2: Equilibrium Carrier Statistics

    *Hu Ch. 1; Pierret Ch. 3–4*

    **Goal:** Given doping and temperature, find the carrier concentrations $n_0$, $p_0$ and the Fermi level $E_F$.

    ---

    ### 2.1 Density of States

    For parabolic bands in 3D:

    $$g_c(E) = \frac{4\pi (2m_n^*)^{3/2}}{h^3}\sqrt{E - E_c}, \quad g_v(E) = \frac{4\pi (2m_p^*)^{3/2}}{h^3}\sqrt{E_v - E}$$

    ### 2.2 Fermi–Dirac Distribution

    $$\boxed{f(E) = \frac{1}{1 + \exp\!\left(\frac{E - E_F}{k_BT}\right)}}$$

    At $E = E_F$: $f = 1/2$.  For $E - E_F \gg k_BT$: $f \approx e^{-(E-E_F)/k_BT}$ (Boltzmann tail).

    ### 2.3 Carrier Concentrations (Non-Degenerate)

    Using the Boltzmann approximation:

    $$\boxed{n_0 = N_c \exp\!\left(-\frac{E_c - E_F}{k_BT}\right)}, \quad \boxed{p_0 = N_v \exp\!\left(-\frac{E_F - E_v}{k_BT}\right)}$$

    where the **effective densities of states** are:

    $$N_c = 2\left(\frac{2\pi m_n^* k_BT}{h^2}\right)^{3/2}, \quad N_v = 2\left(\frac{2\pi m_p^* k_BT}{h^2}\right)^{3/2}$$

    For Si at 300 K: $N_c = 2.86 \times 10^{19}$ cm$^{-3}$, $N_v = 1.08 \times 10^{19}$ cm$^{-3}$.

    ### 2.4 Intrinsic Semiconductor and $n_0p_0$ Product

    $$n_i = \sqrt{N_c N_v}\,\exp\!\left(-\frac{E_g}{2k_BT}\right) \approx 1.1 \times 10^{10}\;\text{cm}^{-3}\text{ (Si, 300 K)}$$

    $$\boxed{n_0 \, p_0 = n_i^2} \quad \text{(valid in equilibrium)}$$

    ### 2.5 Extrinsic Semiconductors (Doping)

    - **Donors** ($N_D$): donate electrons → $n_0 \approx N_D$ (n-type, $N_D \gg n_i, N_A$).
    - **Acceptors** ($N_A$): accept electrons → $p_0 \approx N_A$ (p-type, $N_A \gg n_i, N_D$).
    - **Charge neutrality:** $n_0 + N_A^- = p_0 + N_D^+$.

    ### 2.6 Fermi Level Position

    $$n_0 = n_i \exp\!\left(\frac{E_F - E_{F_i}}{k_BT}\right), \quad p_0 = n_i \exp\!\left(\frac{E_{F_i} - E_F}{k_BT}\right)$$

    For n-type: $E_F - E_{F_i} = k_BT \ln(N_D / n_i)$, so $E_F$ moves **toward** $E_c$.

    ---
    """)
    return


@app.cell
def _(kT_eV, np, plt):
    _fig_fd, (_ax_fd, _ax_np) = plt.subplots(1, 2, figsize=(12, 4.5))

    _E_range = np.linspace(-0.3, 0.3, 500)
    for _T_K, _ls, _lbl in [(77, "--", "77 K"), (300, "-", "300 K"), (600, ":", "600 K")]:
        _kT = kT_eV * _T_K / 300.0
        _f = 1.0 / (1.0 + np.exp(_E_range / _kT))
        _ax_fd.plot(_f, _E_range * 1000, ls=_ls, lw=2, label=_lbl)
    _ax_fd.axhline(0, color="k", lw=0.5, ls="--")
    _ax_fd.set_xlabel("$f(E)$", fontsize=16)
    _ax_fd.set_ylabel("$E - E_F$ (meV)", fontsize=16)
    _ax_fd.set_title("Fermi–Dirac distribution", fontsize=16)
    _ax_fd.legend(fontsize=13)
    _ax_fd.set_xlim(-0.05, 1.05)
    _ax_fd.tick_params(labelsize=13)

    _Nd_arr = np.logspace(13, 19, 300)
    _ni = 1.1e10
    _n0 = _Nd_arr
    _p0 = _ni**2 / _Nd_arr
    _ax_np.loglog(_Nd_arr, _n0, lw=2, label="$n_0$ (electrons)", color="#1f77b4")
    _ax_np.loglog(_Nd_arr, _p0, lw=2, label="$p_0$ (holes)", color="#d62728")
    _ax_np.axhline(_ni, color="gray", ls="--", lw=1)
    _ax_np.text(1e14, 2e10, "$n_i$", fontsize=14, color="gray")
    _ax_np.set_xlabel("$N_D$ (cm$^{-3}$)", fontsize=16)
    _ax_np.set_ylabel("Carrier concentration (cm$^{-3}$)", fontsize=16)
    _ax_np.set_title("n-type Si at 300 K", fontsize=16)
    _ax_np.legend(fontsize=13)
    _ax_np.tick_params(labelsize=13)

    _fig_fd.tight_layout()
    _fig_fd
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Left: Fermi–Dirac function at three temperatures. Right: Electron and hole concentrations vs donor doping in Si at 300 K.  As $N_D$ increases, $n_0 \approx N_D$ and $p_0 = n_i^2/N_D$ drops.*

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Unit 3: Carrier Transport

    *Hu Ch. 2, 4; Pierret Ch. 5–6*

    **Goal:** Relate electric fields and concentration gradients to **current densities**, and describe how carrier concentrations evolve in time and space.

    ---

    ### 3.1 Drift

    An electric field $\mathcal{E}$ accelerates carriers.  Scattering gives a **drift velocity** $v_d = \mu \mathcal{E}$ (low field).

    $$\boxed{J_{n,\text{drift}} = q n \mu_n \mathcal{E}, \quad J_{p,\text{drift}} = q p \mu_p \mathcal{E}}$$

    At high fields, $v_d$ saturates: $v_d \to v_{\text{sat}} \approx 10^7$ cm/s in Si.

    ### 3.2 Diffusion

    A **concentration gradient** drives a diffusion current:

    $$\boxed{J_{n,\text{diff}} = q D_n \frac{dn}{dx}, \quad J_{p,\text{diff}} = -q D_p \frac{dp}{dx}}$$

    ### 3.3 Einstein Relation

    Links diffusion coefficient and mobility (non-degenerate):

    $$\boxed{\frac{D}{\mu} = \frac{k_BT}{q}} \approx 26 \;\text{mV at 300 K}$$

    ### 3.4 Total Current Densities

    $$J_n = qn\mu_n\mathcal{E} + qD_n\frac{dn}{dx}, \quad J_p = qp\mu_p\mathcal{E} - qD_p\frac{dp}{dx}$$

    $$J_{\text{total}} = J_n + J_p$$

    ### 3.5 Generation and Recombination

    - **Thermal G/R:** $R - G = \Delta n / \tau$ for low-level injection ($\Delta n \ll$ majority).
    - **Radiative** (direct gap): $R \propto np$ — basis of LEDs and lasers.
    - **Auger:** $R \propto n^2p$ or $np^2$ — dominates at high carrier densities.
    - **SRH (trap-assisted):** dominant in indirect-gap Si; rate depends on trap energy and capture cross-sections.

    ### 3.6 Continuity Equation

    Conservation of carriers in a volume element:

    $$\boxed{\frac{\partial n}{\partial t} = \frac{1}{q}\frac{\partial J_n}{\partial x} + G_n - R_n}$$

    $$\boxed{\frac{\partial p}{\partial t} = -\frac{1}{q}\frac{\partial J_p}{\partial x} + G_p - R_p}$$

    ### 3.7 Minority Carrier Diffusion Equation

    In a uniformly doped, field-free region with low-level injection and linear recombination:

    $$\frac{\partial^2 \Delta p_n}{\partial x^2} = \frac{\Delta p_n}{L_p^2} \quad \text{(steady state)}$$

    $$\boxed{L_p = \sqrt{D_p \tau_p}, \quad L_n = \sqrt{D_n \tau_n}}$$

    $L$ is the **diffusion length** — the average distance a minority carrier diffuses before recombining.

    ### 3.8 Band Bending and Quasi-Fermi Levels

    - In equilibrium, $E_F$ is **flat** (constant throughout the device).
    - An electric field causes **band bending**: $\mathcal{E} = \frac{1}{q}\frac{dE_c}{dx} = \frac{1}{q}\frac{dE_v}{dx} = \frac{1}{q}\frac{dE_{F_i}}{dx}$.
    - Out of equilibrium, define **quasi-Fermi levels** $E_{F_n}$, $E_{F_p}$:

    $$n = n_i \exp\!\left(\frac{E_{F_n} - E_{F_i}}{k_BT}\right), \quad p = n_i \exp\!\left(\frac{E_{F_i} - E_{F_p}}{k_BT}\right)$$

    $$J_n = \mu_n n \frac{dE_{F_n}}{dx}, \quad J_p = \mu_p p \frac{dE_{F_p}}{dx}$$

    Current flows **where the quasi-Fermi level has a gradient**.

    ### 3.9 Poisson's Equation

    Ties the electric field to the charge distribution:

    $$\boxed{\frac{d\mathcal{E}}{dx} = \frac{\rho}{\epsilon_s} = \frac{q}{\epsilon_s}(p - n + N_D^+ - N_A^-)}$$

    Together, drift-diffusion + continuity + Poisson form the **semiconductor equations** — the basis for all device models.

    ---
    """)
    return


@app.cell
def _(mu_n_Si, mu_p_Si, np, plt):
    _fig_tr, (_ax_vd, _ax_diff) = plt.subplots(1, 2, figsize=(12, 4.5))

    _E_field = np.linspace(0, 1e5, 500)
    _vsat = 1.0e7
    for _mu, _label, _c in [
        (mu_n_Si, "Electrons", "#1f77b4"),
        (mu_p_Si, "Holes", "#d62728"),
    ]:
        _vd = _mu * _E_field / (1.0 + _mu * _E_field / _vsat)
        _ax_vd.plot(_E_field / 1e3, _vd / 1e6, lw=2, label=_label, color=_c)

    _ax_vd.set_xlabel(r"$\mathcal{E}$ (kV/cm)", fontsize=16)
    _ax_vd.set_ylabel(r"$v_d$ ($\times 10^6$ cm/s)", fontsize=16)
    _ax_vd.set_title("Drift velocity vs field (Si, 300 K)", fontsize=16)
    _ax_vd.legend(fontsize=13)
    _ax_vd.tick_params(labelsize=13)
    _ax_vd.set_ylim(0, 12)

    _x_um = np.linspace(0, 50, 300)
    _Lp_vals = [5.0, 10.0, 20.0]
    for _Lp in _Lp_vals:
        _dp = np.exp(-_x_um / _Lp)
        _ax_diff.plot(
            _x_um, _dp, lw=2, label=rf"$L_p = {_Lp:.0f}\;\mu$m"
        )
    _ax_diff.set_xlabel(r"$x$ ($\mu$m)", fontsize=16)
    _ax_diff.set_ylabel(r"$\Delta p_n(x) / \Delta p_n(0)$", fontsize=16)
    _ax_diff.set_title("Minority carrier decay", fontsize=16)
    _ax_diff.legend(fontsize=13)
    _ax_diff.tick_params(labelsize=13)

    _fig_tr.tight_layout()
    _fig_tr
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Left: Drift velocity vs electric field in Si showing saturation at $\sim 10^7$ cm/s.  Right: Exponential decay of excess minority carriers — shorter $L_p$ means faster spatial decay (stronger recombination relative to diffusion).*

    ---
    """)
    return


@app.cell
def _(BASE, mo):
    mo.md(rf"""
    # Unit 4: PN Junction

    *Hu Ch. 4; Pierret Ch. 6–8*

    **Goal:** Electrostatics in the depletion region, forward/reverse bias, **Shockley diode I-V**, small-signal and transient models.

    ---

    ### 4.1 PN Junction at Equilibrium

    When P-type ($N_A$) and N-type ($N_D$) regions are joined:

    - Carriers diffuse across → leaves behind **ionized dopants** → **depletion region** forms.
    - Electric field opposes further diffusion → **equilibrium** with $E_F$ flat.

    **Built-in potential:**

    $$\boxed{{\phi_{{bi}} = \frac{{k_BT}}{{q}} \ln\!\left(\frac{{N_A N_D}}{{n_i^2}}\right)}}$$

    Interactive: [{BASE}/pn-electrostatics/]({BASE}/pn-electrostatics/)

    ### 4.2 Depletion Approximation

    Charge density in the depletion region ($-x_p < x < x_n$):

    $$\rho(x) = \begin{{cases}} -qN_A & -x_p \le x \le 0 \\ qN_D & 0 \le x \le x_n \\ 0 & \text{{elsewhere}} \end{{cases}}$$

    **Charge neutrality:** $N_A x_p = N_D x_n$ (depletion extends further into the lightly-doped side).

    **Depletion width** (under bias $V_A$):

    $$\boxed{{W = x_p + x_n = \sqrt{{\frac{{2\epsilon_s}}{{q}} \left(\frac{{1}}{{N_A}} + \frac{{1}}{{N_D}}\right) (\phi_{{bi}} - V_A)}}}}$$

    ### 4.3 Electric Field and Potential

    From Poisson's equation in the depletion region:

    $$\mathcal{{E}}(x) = \begin{{cases}} -\frac{{qN_A}}{{\epsilon_s}}(x + x_p) & -x_p \le x \le 0 \\ \frac{{qN_D}}{{\epsilon_s}}(x - x_n) & 0 \le x \le x_n \end{{cases}}$$

    Peak field at $x = 0$: $\mathcal{{E}}_{{max}} = -qN_A x_p / \epsilon_s = -qN_D x_n / \epsilon_s$.

    ### 4.4 PN Junction Under Bias

    - **Forward bias** ($V_A > 0$): barrier reduced → exponential injection of minority carriers.
    - **Reverse bias** ($V_A < 0$): barrier increased → tiny reverse saturation current.

    **Law of the junction** (minority carrier boundary conditions):

    $$n_p(x_p) = \frac{{n_i^2}}{{N_A}} e^{{qV_A/k_BT}}, \quad p_n(x_n) = \frac{{n_i^2}}{{N_D}} e^{{qV_A/k_BT}}$$

    ### 4.5 Shockley Ideal Diode Equation

    Solving the minority-carrier diffusion equation with these boundary conditions:

    $$\boxed{{I = I_0\!\left(e^{{qV_A / k_BT}} - 1\right)}}$$

    $$I_0 = Aqn_i^2 \left(\frac{{D_p}}{{L_p N_D}} + \frac{{D_n}}{{L_n N_A}}\right)$$

    ### 4.6 Small-Signal Model

    - **Junction capacitance** (reverse bias): $C_J = \epsilon_s / W$ — varies as $(\phi_{{bi}} - V_A)^{{-1/2}}$.
    - **Diffusion capacitance** (forward bias): $C_d = \tau_s G_d$ where $G_d = qI / k_BT$.
    - **Conductance:** $G_d = dI/dV \approx qI/(k_BT)$ for forward bias.

    ### 4.7 Non-Idealities

    - **Recombination current** in depletion region: $I_r \propto e^{{qV_A / 2k_BT}}$ → ideality factor $n \approx 2$.
    - **High injection:** both carrier types flood the lightly-doped side → $n \to 2$.
    - **Series resistance:** $V_A \to V_A - IR_s$ at high current.
    - **Breakdown:** Zener (tunneling, heavily doped) and avalanche (impact ionization).

    ---
    """)
    return


@app.cell
def _(eps_Si, kT_eV, ni_si, np, plt, q_C):
    _fig_pn, ((_ax_rho, _ax_E), (_ax_V, _ax_IV)) = plt.subplots(
        2, 2, figsize=(12, 9)
    )

    _NA = 1e17
    _ND = 5e16
    _Vbi = kT_eV * np.log(_NA * _ND / ni_si**2)
    _Va = 0.0
    _W_tot = np.sqrt(2 * eps_Si / q_C * (1 / _NA + 1 / _ND) * (_Vbi - _Va))
    _xn = _W_tot * _NA / (_NA + _ND)
    _xp = _W_tot * _ND / (_NA + _ND)

    _x = np.linspace(-2 * _xp, 2 * _xn, 1000) * 1e4

    _rho = np.piecewise(
        _x,
        [_x < -_xp * 1e4, (_x >= -_xp * 1e4) & (_x < 0), (_x >= 0) & (_x < _xn * 1e4), _x >= _xn * 1e4],
        [0, -q_C * _NA, q_C * _ND, 0],
    )
    _ax_rho.plot(_x, _rho / q_C / 1e16, lw=2, color="#1f77b4")
    _ax_rho.axhline(0, color="k", lw=0.5)
    _ax_rho.set_ylabel(r"$\rho / q$ ($\times 10^{16}$ cm$^{-3}$)", fontsize=16)
    _ax_rho.set_xlabel(r"$x$ ($\mu$m)", fontsize=16)
    _ax_rho.set_title(r"Charge density $\rho(x)$", fontsize=16)
    _ax_rho.tick_params(labelsize=13)

    _x_cm = _x * 1e-4
    _Efield = np.piecewise(
        _x_cm,
        [
            _x_cm < -_xp,
            (_x_cm >= -_xp) & (_x_cm < 0),
            (_x_cm >= 0) & (_x_cm < _xn),
            _x_cm >= _xn,
        ],
        [
            0,
            lambda xx: -q_C * _NA / eps_Si * (xx + _xp),
            lambda xx: q_C * _ND / eps_Si * (xx - _xn),
            0,
        ],
    )
    _ax_E.plot(_x, _Efield, lw=2, color="#d62728")
    _ax_E.axhline(0, color="k", lw=0.5)
    _ax_E.set_ylabel(r"$\mathcal{E}$ (V/cm)", fontsize=16)
    _ax_E.set_xlabel(r"$x$ ($\mu$m)", fontsize=16)
    _ax_E.set_title(r"Electric field $\mathcal{E}(x)$", fontsize=16)
    _ax_E.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    _ax_E.tick_params(labelsize=13)

    _V = np.zeros_like(_x_cm)
    for _i in range(1, len(_x_cm)):
        _V[_i] = _V[_i - 1] - _Efield[_i] * (_x_cm[_i] - _x_cm[_i - 1])
    _V = _V - _V[0]

    _ax_V.plot(_x, _V, lw=2, color="#2ca02c")
    _ax_V.set_ylabel(r"$V(x)$ (V)", fontsize=16)
    _ax_V.set_xlabel(r"$x$ ($\mu$m)", fontsize=16)
    _ax_V.set_title(r"Electrostatic potential $V(x)$", fontsize=16)
    _ax_V.tick_params(labelsize=13)

    _Va_arr = np.linspace(-2, 0.7, 500)
    _I0 = 1e-14
    _I = _I0 * (np.exp(_Va_arr / kT_eV) - 1)
    _ax_IV.plot(_Va_arr, _I * 1e3, lw=2, color="#9467bd")
    _ax_IV.set_xlabel(r"$V_A$ (V)", fontsize=16)
    _ax_IV.set_ylabel(r"$I$ (mA)", fontsize=16)
    _ax_IV.set_title("Ideal diode I-V", fontsize=16)
    _ax_IV.set_ylim(-0.5, 5)
    _ax_IV.axhline(0, color="k", lw=0.5)
    _ax_IV.axvline(0, color="k", lw=0.5)
    _ax_IV.tick_params(labelsize=13)

    _fig_pn.suptitle(
        f"PN Junction: $N_A = {_NA:.0e}$, $N_D = {_ND:.0e}$ cm$^{{-3}}$, $\\phi_{{bi}} = {_Vbi:.3f}$ V",
        fontsize=16,
        y=1.01,
    )
    _fig_pn.tight_layout()
    _fig_pn
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Top-left: charge density $\rho(x)$ in depletion. Top-right: electric field peaks at the metallurgical junction. Bottom-left: electrostatic potential is parabolic (from Poisson). Bottom-right: ideal Shockley diode I-V.*

    ---
    """)
    return


@app.cell
def _(mo):
    log_na_pn = mo.ui.slider(14.0, 18.0, value=17.0, step=0.1, label="log₁₀ Nₐ (cm⁻³)")
    log_nd_pn = mo.ui.slider(14.0, 18.0, value=16.0, step=0.1, label="log₁₀ N_D (cm⁻³)")
    Va_slider = mo.ui.slider(-5.0, 0.6, value=0.0, step=0.05, label="Applied voltage V_A (V)")
    mo.vstack([
        mo.md("### Interactive: PN junction depletion width"),
        mo.hstack([log_na_pn, log_nd_pn, Va_slider]),
    ])
    return Va_slider, log_na_pn, log_nd_pn


@app.cell
def _(Va_slider, eps_Si, kT_eV, log_na_pn, log_nd_pn, mo, ni_si, np, q_C):
    _NA = 10.0 ** log_na_pn.value
    _ND = 10.0 ** log_nd_pn.value
    _Vbi = kT_eV * np.log(_NA * _ND / ni_si**2)
    _Va = Va_slider.value
    _Veff = _Vbi - _Va
    if _Veff < 0:
        _Veff = 0.001
    _W = np.sqrt(2 * eps_Si / q_C * (1 / _NA + 1 / _ND) * _Veff)
    _xn = _W * _NA / (_NA + _ND)
    _xp = _W * _ND / (_NA + _ND)
    _Emax = q_C * _NA * _xp / eps_Si
    _Cj = eps_Si / _W
    mo.md(
        f"""
    | Quantity | Value |
    |----------|-------|
    | $\\phi_{{bi}}$ | {_Vbi:.4f} V |
    | $\\phi_{{bi}} - V_A$ | {_Veff:.4f} V |
    | $W$ | {_W*1e4:.3f} $\\mu$m |
    | $x_p$ | {_xp*1e4:.3f} $\\mu$m |
    | $x_n$ | {_xn*1e4:.3f} $\\mu$m |
    | $\\mathcal{{E}}_{{max}}$ | {_Emax:.2e} V/cm |
    | $C_j / A$ | {_Cj*1e9:.2f} nF/cm² |
        """
    )
    return


@app.cell
def _(BASE, mo):
    mo.md(rf"""
    # Unit 5: Bipolar Junction Transistor (BJT)

    *Hu Ch. 8*

    **Goal:** Derive how a thin base region between two PN junctions allows a small base current to **control** a large collector current.

    Interactive: [{BASE}/bjt-iv/]({BASE}/bjt-iv/)

    ---

    ### 5.1 Structure and Modes of Operation

    An NPN BJT has: **n⁺ emitter** – **p base** – **n collector**.

    | Mode | EB junction | BC junction | Use |
    |------|------------|------------|-----|
    | **Forward active** | Forward | Reverse | Amplification |
    | Saturation | Forward | Forward | Digital "on" |
    | Cutoff | Reverse | Reverse | Digital "off" |
    | Reverse active | Reverse | Forward | (rare) |

    ### 5.2 Forward Active Mode — Minority Carrier Profiles

    - EB forward → electrons injected from emitter into base.
    - BC reverse → electrons swept into collector at the BC edge (boundary condition: $\Delta n_B(W_B) = 0$).
    - Base is thin ($W_B \ll L_B$) → profile is nearly **linear** → almost all injected electrons reach the collector.

    The minority electron profile in the quasi-neutral base ($0 \le x \le W_B$):

    $$\Delta n_B(x) = \Delta n_B(0) \left(1 - \frac{{x}}{{W_B}}\right) \quad \text{{(short base approx.)}}$$

    ### 5.3 Terminal Currents

    $$\boxed{{I_C \approx I_S \left(e^{{qV_{{BE}}/k_BT}} - 1\right)}}$$

    where the saturation current:

    $$I_S = \frac{{A_E q D_{{nB}} n_i^2}}{{G_B}}$$

    and $G_B = \int_0^{{W_B}} N_A(x)\,dx$ is the **base Gummel number** (total integrated base doping per unit area).

    ### 5.4 Current Gain $\beta_0$

    $$\beta_0 = \frac{{I_C}}{{I_B}} \approx \frac{{G_E}}{{G_B}}$$

    where $G_E$ is the emitter Gummel number.  High $\beta_0$ requires:

    - **Heavy emitter doping** (large $G_E$)
    - **Thin, lightly-doped base** (small $G_B$)

    ### 5.5 Base Transit Time and $f_T$

    $$\tau_B = \frac{{W_B^2}}{{2 D_{{nB}}}}, \quad \frac{{1}}{{2\pi f_T}} = \frac{{C_{{be}} + C_{{bc}}}}{{g_m}} + \tau_B + \tau_C$$

    Shorter base → faster transit → higher $f_T$.

    ### 5.6 Early Effect (Base Width Modulation)

    Increasing $V_{{CE}}$ widens the BC depletion region → shrinks $W_B$ → increases $I_C$ slightly.

    $$I_C \approx I_S e^{{qV_{{BE}}/k_BT}} \left(1 + \frac{{V_{{CE}}}}{{V_A}}\right)$$

    where $V_A$ is the **Early voltage** (typically 50–200 V).

    ### 5.7 Heterojunction BJTs (HBTs)

    - Replace emitter with wider-bandgap material (e.g. SiGe base, or III-V emitter).
    - **Wider gap in emitter** suppresses hole back-injection → much higher $\beta_0$ without needing extreme doping.
    - SiGe HBTs are used in high-speed communications, RF circuits.

    ---
    """)
    return


@app.cell
def _(kT_eV, np, plt):
    _fig_bjt, (_ax_min, _ax_gummel) = plt.subplots(1, 2, figsize=(12, 4.5))

    _Wb = 0.5
    _x_base = np.linspace(0, _Wb, 200)
    for _Vbe, _c in [(0.5, "#1f77b4"), (0.6, "#ff7f0e"), (0.7, "#d62728")]:
        _dn0 = np.exp(_Vbe / kT_eV)
        _dn = _dn0 * (1 - _x_base / _Wb)
        _ax_min.plot(_x_base, _dn / _dn0, lw=2, label=f"$V_{{BE}}$ = {_Vbe} V", color=_c)
    _ax_min.set_xlabel(r"$x / W_B$", fontsize=16)
    _ax_min.set_ylabel(r"$\Delta n_B(x) / \Delta n_B(0)$", fontsize=16)
    _ax_min.set_title("Base minority carrier profile", fontsize=16)
    _ax_min.legend(fontsize=13)
    _ax_min.tick_params(labelsize=13)

    _Vbe_arr = np.linspace(0, 0.75, 300)
    _Is_C = 1e-15
    _beta = 100
    _Is_B = _Is_C / _beta
    _Ic = _Is_C * np.exp(_Vbe_arr / kT_eV)
    _Ib = _Is_B * np.exp(_Vbe_arr / kT_eV)
    _ax_gummel.semilogy(_Vbe_arr, _Ic, lw=2, label="$I_C$", color="#1f77b4")
    _ax_gummel.semilogy(_Vbe_arr, _Ib, lw=2, label="$I_B$", color="#d62728")
    _ax_gummel.set_xlabel(r"$V_{BE}$ (V)", fontsize=16)
    _ax_gummel.set_ylabel(r"Current (A)", fontsize=16)
    _ax_gummel.set_title("Gummel plot (ideal)", fontsize=16)
    _ax_gummel.set_ylim(1e-15, 1e-1)
    _ax_gummel.legend(fontsize=13)
    _ax_gummel.tick_params(labelsize=13)

    _fig_bjt.tight_layout()
    _fig_bjt
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Left: Normalised minority-carrier (electron) profile in the base — linear for a short base, all profiles share the same slope shape but $\Delta n_B(0)$ increases exponentially with $V_{BE}$.  Right: Ideal Gummel plot — both $I_C$ and $I_B$ are exponential in $V_{BE}$, separated by $\beta_0$.*

    ---
    """)
    return


@app.cell
def _(BASE, mo):
    mo.md(rf"""
    # Unit 6: Metal–Semiconductor Junctions

    *Hu Ch. 4.20–4.22*

    Interactive: [{BASE}/m-s-interface/]({BASE}/m-s-interface/)

    ---

    ### 6.1 Schottky Barrier

    When a metal contacts a semiconductor, band bending creates a **Schottky barrier** $\phi_{{Bn}}$ (for n-type) at the interface:

    $$\phi_{{Bn}} = \phi_M - \chi_S$$

    where $\phi_M$ is the metal work function and $\chi_S$ is the semiconductor electron affinity.

    The built-in potential of the metal–semiconductor junction:

    $$V_{{bi}} = \phi_{{Bn}} - (E_c - E_F)_{{bulk}} = \phi_{{Bn}} - \frac{{k_BT}}{{q}}\ln\!\left(\frac{{N_c}}{{N_D}}\right)$$

    ### 6.2 Schottky Diode I-V (Thermionic Emission)

    $$\boxed{{I = I_S\!\left(e^{{qV_A/k_BT}} - 1\right), \quad I_S = AA^* T^2 \exp\!\left(-\frac{{q\phi_{{Bn}}}}{{k_BT}}\right)}}$$

    where $A^*$ is the **Richardson constant**.

    Compared to PN diodes: **faster switching** (majority carrier device, no minority charge storage), but higher leakage ($I_S$ is larger).

    ### 6.3 Ohmic Contacts

    For **high-doping** interfaces ($N_D > 10^{{19}}$ cm$^{{-3}}$): barrier is thin → tunnelling dominates → linear $I$-$V$ (ohmic behaviour).

    **Specific contact resistance:** $\rho_c \propto \exp(C\phi_B / \sqrt{{N_D}})$ — minimised by heavy doping and low barrier.

    ### 6.4 Fermi-Level Pinning

    In real semiconductors, **interface states** can pin $E_F$ near mid-gap, making $\phi_B$ nearly independent of the metal work function (especially in GaAs and other III-Vs).

    ---
    """)
    return


@app.cell
def _(Nc_Si, kT_eV, np, plt):
    _fig_ms, _ax_ms = plt.subplots(1, 1, figsize=(7, 4.5))

    _phi_Bn = 0.67
    _ND = 1e16
    _Vbi_ms = _phi_Bn - kT_eV * np.log(Nc_Si / _ND)
    _Va_arr = np.linspace(-2, 0.5, 400)
    _Astar = 120
    _Is_ms = _Astar * 300**2 * np.exp(-_phi_Bn / kT_eV) * 1e-4
    _I_ms = _Is_ms * (np.exp(_Va_arr / kT_eV) - 1)

    _ax_ms.plot(_Va_arr, _I_ms * 1e6, lw=2, color="#1f77b4", label="Schottky diode")

    _I0_pn = 1e-14
    _I_pn = _I0_pn * (np.exp(_Va_arr / kT_eV) - 1)
    _ax_ms.plot(_Va_arr, _I_pn * 1e6, lw=2, ls="--", color="#d62728", label="PN diode")

    _ax_ms.set_xlabel(r"$V_A$ (V)", fontsize=16)
    _ax_ms.set_ylabel(r"$I$ ($\mu$A)", fontsize=16)
    _ax_ms.set_title(r"Schottky vs PN diode I-V ($\phi_{Bn} = 0.67$ V)", fontsize=16)
    _ax_ms.set_ylim(-50, 500)
    _ax_ms.axhline(0, color="k", lw=0.5)
    _ax_ms.axvline(0, color="k", lw=0.5)
    _ax_ms.legend(fontsize=13)
    _ax_ms.tick_params(labelsize=13)
    _fig_ms.tight_layout()
    _fig_ms
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Schottky diodes have much higher reverse saturation current ($I_S$) than PN diodes (thermionic emission over a lower barrier vs minority-carrier injection), so they turn on at a lower forward voltage.*

    ---
    """)
    return


@app.cell
def _(BASE, mo):
    mo.md(rf"""
    # Unit 7: MOS Capacitor and MOSFET

    *Hu Ch. 5–7*

    Interactive: [MOS cap]({BASE}/moscap/) · [MOSFET intro]({BASE}/mosfet-intro/) · [MOSFET I-V]({BASE}/mosfet-iv/) · [dynamics]({BASE}/mosfet-ac/) · [FinFET/FDSOI]({BASE}/finfet-fdsoi/)

    ---

    ## 7A. MOS Capacitor

    ### Regimes of Operation (p-type body)

    | Regime | Gate bias condition | Surface condition |
    |--------|--------------------|--------------------|
    | **Accumulation** | $V_G < V_{{FB}}$ | Holes pile up at surface |
    | **Flat-band** | $V_G = V_{{FB}}$ | No band bending |
    | **Depletion** | $V_{{FB}} < V_G < V_T$ | Surface depleted |
    | **Inversion** | $V_G > V_T$ | Electron layer at surface ($n_s > N_A$) |

    ### Key Quantities

    **Surface potential at threshold** (onset of strong inversion):

    $$\phi_{{s,inv}} = 2\phi_B, \quad \phi_B = \frac{{k_BT}}{{q}}\ln\!\left(\frac{{N_A}}{{n_i}}\right)$$

    **Maximum depletion width:**

    $$\boxed{{W_{{dep,max}} = \sqrt{{\frac{{2\epsilon_s \cdot 2\phi_B}}{{qN_A}}}}}}$$

    **Threshold voltage:**

    $$\boxed{{V_T = V_{{FB}} + 2\phi_B + \frac{{\sqrt{{2q\epsilon_s N_A \cdot 2\phi_B}}}}{{C_{{ox}}^{{\prime}}}}}}$$

    where $C_{{ox}}^{{\prime}} = \epsilon_{{ox}} / t_{{ox}}$ and $V_{{FB}} = \phi_{{ms}} - Q_f / C_{{ox}}^{{\prime}}$ (flat-band voltage).

    ---

    ## 7B. MOSFET I-V

    ### Long-Channel Model (n-MOSFET, gradual channel)

    **Linear region** ($V_{{DS}} < V_{{DS,sat}}$):

    $$\boxed{{I_D = \mu_n C_{{ox}}^{{\prime}} \frac{{W}}{{L}} \left[(V_{{GS}} - V_T)V_{{DS}} - \frac{{V_{{DS}}^2}}{{2}}\right]}}$$

    **Saturation** ($V_{{DS}} \ge V_{{DS,sat}} = V_{{GS}} - V_T$):

    $$\boxed{{I_{{D,sat}} = \frac{{\mu_n C_{{ox}}^{{\prime}}}}{{2}} \frac{{W}}{{L}} (V_{{GS}} - V_T)^2}}$$

    **Transconductance:**

    $$g_m = \frac{{\partial I_D}}{{\partial V_{{GS}}}} = \mu_n C_{{ox}}^{{\prime}} \frac{{W}}{{L}} V_{{DS}} \quad \text{{(linear)}}; \quad g_m = \mu_n C_{{ox}}^{{\prime}} \frac{{W}}{{L}} (V_{{GS}} - V_T) \quad \text{{(sat.)}}$$

    ### Important Modifications

    | Effect | What it changes |
    |--------|----------------|
    | **Body effect** | $V_T$ shifts with $V_{{SB}}$: $\Delta V_T = \gamma(\sqrt{{2\phi_B + V_{{SB}}}} - \sqrt{{2\phi_B}})$ |
    | **Velocity saturation** | $v_d \to v_{{sat}}$ at high $\mathcal{{E}}$; $I_{{D,sat}}$ becomes **linear** in $V_{{GS}}-V_T$ |
    | **Channel length modulation** | $I_D \propto (1 + \lambda V_{{DS}})$ in saturation |
    | **Subthreshold conduction** | $I_D \propto e^{{qV_{{GS}}/(nk_BT)}}$; **subthreshold swing** $SS = n \times 60$ mV/dec (at 300 K) |
    | **DIBL** | $V_T$ decreases as $L$ shrinks (drain field penetrates into channel) |

    ### Small-Signal Model and $f_T$

    $$f_T = \frac{{g_m}}{{2\pi(C_{{gs}} + C_{{gd}})}} \approx \frac{{3\mu_{{eff}} (V_{{GS}} - V_T)}}{{4\pi L^2}} \quad \text{{(long channel)}}$$

    - In the **velocity-saturated** regime: $f_T \propto v_{{sat}} / L$ (linear in $1/L$, not $1/L^2$).

    ### Digital Switching

    - **Propagation delay:** $\tau_p \propto C_L V_{{DD}} / I_{{on}}$.
    - **Power:** $P_{{dyn}} = \alpha C_L V_{{DD}}^2 f$.
    - Scaling $V_{{DD}}$ and $L$ together → **Dennard scaling** → faster, lower power.

    ---
    """)
    return


@app.cell
def _(
    Eg_Si,
    Nc_Si,
    Nv_Si,
    chi_Si,
    eps_0_cgs,
    eps_Si,
    kT_eV,
    mo,
    ni_si,
    np,
    plt,
    q_C,
):
    _Na = 1e16
    _phi_B = kT_eV * np.log(_Na / ni_si)
    _Ec_minus_Ei = Eg_Si / 2 + kT_eV / 2 * np.log(Nc_Si / Nv_Si)
    _Ei_minus_Ev = Eg_Si / 2 + kT_eV / 2 * np.log(Nv_Si / Nc_Si)
    _dEc_ox = chi_Si - 0.95
    _dEv_ox = 9.0 - Eg_Si - _dEc_ox

    _EF = 0.0
    _Ei_bulk = _phi_B
    _Ec_bulk = _Ei_bulk + _Ec_minus_Ei
    _Ev_bulk = _Ei_bulk - _Ei_minus_Ev

    _eps_s = eps_Si * eps_0_cgs
    _phi_st = 2 * _phi_B
    _Wdep_max_cm = np.sqrt(2 * _eps_s * _phi_st / (q_C * _Na))
    _Cox = eps_0_cgs * 3.9 / 5e-7
    _Vt = _phi_st + q_C * _Na * _Wdep_max_cm / _Cox
    _gamma = np.sqrt(2 * q_C * _eps_s * _Na) / _Cox

    _regimes = [
        ("Accumulation\n($V_G < 0$)", -1.5, "#d62728"),
        ("Depletion\n($0 < V_G < V_T$)", _Vt * 0.5, "#2ca02c"),
        ("Threshold\n($V_G = V_T$)", _Vt, "#ff7f0e"),
        ("Inversion\n($V_G > V_T$)", _Vt + 1.5, "#1f77b4"),
    ]

    _fig_mos, _axes = plt.subplots(1, 4, figsize=(18, 6.5), sharey=True)

    for _idx, (_label, _Vg, _color) in enumerate(_regimes):
        _ax = _axes[_idx]

        if _Vg >= 0:
            _phi_s = 0.1
            for _ in range(100):
                if _phi_s < 1e-6:
                    _phi_s = 1e-6
                _f = _phi_s + _gamma * np.sqrt(_phi_s) - _Vg
                _df = 1 + _gamma / (2 * np.sqrt(_phi_s))
                _phi_s -= _f / _df
                if abs(_f) < 1e-8:
                    break
            _phi_s = min(_phi_s, _phi_st)
            _Wdep = np.sqrt(2 * _eps_s * _phi_s / (q_C * _Na))
        else:
            _phi_s = _Vg * 0.25
            _Wdep = 0

        _EFm = _EF - _Vg

        _xM = [-1.8, -1.2]
        _xO = [-1.2, 0.0]
        _x_sc_end = 2.5

        if _phi_s > 0.001:
            _Wdep_plot = min(2.5, max(0.3, _Wdep * 1e4 * 2.5))
            _x_dep = np.linspace(0, _Wdep_plot, 200)
            _x_flat = np.linspace(_Wdep_plot, _x_sc_end, 50)
            _bending = _phi_s * (1 - _x_dep / _Wdep_plot) ** 2
            _Ec_dep = _Ec_bulk - _bending
            _Ev_dep = _Ev_bulk - _bending
            _Ei_dep = _Ei_bulk - _bending
            _x_sc = np.concatenate([_x_dep, _x_flat])
            _Ec_sc = np.concatenate([_Ec_dep, np.full_like(_x_flat, _Ec_bulk)])
            _Ev_sc = np.concatenate([_Ev_dep, np.full_like(_x_flat, _Ev_bulk)])
            _Ei_sc = np.concatenate([_Ei_dep, np.full_like(_x_flat, _Ei_bulk)])
        elif _phi_s < -0.001:
            _bend_len = 0.5
            _x_acc = np.linspace(0, _bend_len, 100)
            _x_flat = np.linspace(_bend_len, _x_sc_end, 50)
            _bending = _phi_s * (1 - _x_acc / _bend_len) ** 2
            _Ec_acc = _Ec_bulk - _bending
            _Ev_acc = _Ev_bulk - _bending
            _Ei_acc = _Ei_bulk - _bending
            _x_sc = np.concatenate([_x_acc, _x_flat])
            _Ec_sc = np.concatenate([_Ec_acc, np.full_like(_x_flat, _Ec_bulk)])
            _Ev_sc = np.concatenate([_Ev_acc, np.full_like(_x_flat, _Ev_bulk)])
            _Ei_sc = np.concatenate([_Ei_acc, np.full_like(_x_flat, _Ei_bulk)])
        else:
            _x_sc = np.array([0, _x_sc_end])
            _Ec_sc = np.array([_Ec_bulk, _Ec_bulk])
            _Ev_sc = np.array([_Ev_bulk, _Ev_bulk])
            _Ei_sc = np.array([_Ei_bulk, _Ei_bulk])

        _ax.fill_between(_xM, _EFm - 2.5, _EFm, color="#6baed6", alpha=0.3)
        _ax.plot(_xM, [_EFm, _EFm], "g-", linewidth=2)
        _ax.text(_xM[0] + 0.05, _EFm + 0.12, "$E_{F,M}$", fontsize=11, color="green")

        _Ec_ox_s = _Ec_sc[0] + _dEc_ox
        _Ev_ox_s = _Ev_sc[0] - _dEv_ox
        _V_ox_tilt = _Vg - _phi_s
        _Ec_ox_m = _Ec_ox_s - _V_ox_tilt
        _Ev_ox_m = _Ev_ox_s - _V_ox_tilt

        _ax.fill_between(
            _xO, [_Ev_ox_m, _Ev_ox_s], [_Ec_ox_m, _Ec_ox_s],
            color="#ffffcc", alpha=0.4, edgecolor="none",
        )
        _ax.plot(_xO, [_Ec_ox_m, _Ec_ox_s], "k-", linewidth=2)
        _ax.plot(_xO, [_Ev_ox_m, _Ev_ox_s], "k-", linewidth=2)

        _ax.plot(_x_sc, _Ec_sc, "b-", linewidth=2.5)
        _ax.plot(_x_sc, _Ev_sc, "r-", linewidth=2.5)
        _ax.plot(_x_sc, _Ei_sc, "k--", linewidth=1, alpha=0.5)
        _ax.plot([0, _x_sc_end], [_EF, _EF], "g-", linewidth=2)

        _ax.plot([-1.2, -1.2], [_Ev_ox_m, _Ec_ox_m], "k-", lw=1.5)
        _ax.plot([0, 0], [_Ev_ox_s, _Ec_ox_s], "k-", lw=1.5)

        if _idx == 3:
            _ax.text(2.6, _Ec_bulk, "$E_c$", fontsize=12, va="center", color="blue")
            _ax.text(2.6, _Ev_bulk, "$E_v$", fontsize=12, va="center", color="red")
            _ax.text(2.6, _Ei_bulk, "$E_{Fi}$", fontsize=12, va="center", alpha=0.6)
            _ax.text(2.6, _EF, "$E_{Fs}$", fontsize=12, va="center", color="green")

        if abs(_phi_s) > 0.05:
            _cap = 0.06
            _xann = 0.15
            _ax.plot([_xann, _xann], [_Ec_sc[0], _Ec_bulk], color="darkgreen", lw=1.5)
            _ax.plot([_xann - _cap, _xann + _cap], [_Ec_sc[0], _Ec_sc[0]], color="darkgreen", lw=1.5)
            _ax.plot([_xann - _cap, _xann + _cap], [_Ec_bulk, _Ec_bulk], color="darkgreen", lw=1.5)
            _ax.text(
                _xann + 0.15, (_Ec_sc[0] + _Ec_bulk) / 2, r"$q\phi_s$",
                fontsize=12, color="darkgreen", va="center",
            )

        _cap = 0.06
        _xpB = 2.0
        _ax.plot([_xpB, _xpB], [_EF, _Ei_bulk], color="purple", lw=1.2)
        _ax.plot([_xpB - _cap, _xpB + _cap], [_EF, _EF], color="purple", lw=1.2)
        _ax.plot([_xpB - _cap, _xpB + _cap], [_Ei_bulk, _Ei_bulk], color="purple", lw=1.2)
        _ax.text(
            _xpB + 0.12, (_EF + _Ei_bulk) / 2, r"$q\phi_B$",
            fontsize=11, color="purple", va="center",
        )

        if abs(_V_ox_tilt) > 0.01:
            _xvox = -1.05
            _bt = _Ec_ox_s
            _bb = _Ec_ox_m
            _min_vis = 0.4
            if abs(_bt - _bb) < _min_vis:
                _mid = (_bt + _bb) / 2
                _sgn = 1 if _bt >= _bb else -1
                _bt = _mid + _sgn * _min_vis / 2
                _bb = _mid - _sgn * _min_vis / 2
            _ax.plot([_xvox, _xvox], [_bb, _bt], color="darkorange", lw=1.5)
            _ax.plot([_xvox - _cap, _xvox + _cap], [_bb, _bb], color="darkorange", lw=1.5)
            _ax.plot([_xvox - _cap, _xvox + _cap], [_bt, _bt], color="darkorange", lw=1.5)
            _ax.text(
                _xvox + 0.12, (_bb + _bt) / 2, r"$qV_{ox}$",
                fontsize=11, color="darkorange", va="center", ha="left",
            )

        _ax.set_title(_label, fontsize=13, fontweight="bold", color=_color)
        _ax.set_xticks([])
        _ax.tick_params(labelsize=12)
        _ax.set_xlim(-2.0, 3.0)
        if _idx == 0:
            _ax.set_ylabel("Energy (eV)", fontsize=14)
        for _sp in ["top", "right", "bottom"]:
            _ax.spines[_sp].set_visible(False)

    _axes[0].set_ylim(_Ev_bulk - _dEv_ox - 1.0, _Ec_bulk + _dEc_ox + 1.5)
    _fig_mos.suptitle(
        r"MOS Energy Bands in Each Regime (P-type body, $N_A = 10^{16}$ cm$^{-3}$)",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.close(_fig_mos)

    _caption = mo.md(rf"""
    **Four regimes of the MOS capacitor** (p-type body, ideal $\Psi_M = \Psi_S$).

    - **Accumulation:** $V_G < 0$, bands bend **up**, holes pile up at the surface.
    - **Depletion:** $V_G > 0$, bands bend **down**, depletion region forms ($\phi_s$ solved from $V_G = \phi_s + \gamma\sqrt{{\phi_s}}$).
    - **Threshold:** $\phi_s = 2\phi_B$, surface electron concentration equals bulk hole concentration ($V_T$ = {_Vt:.2f} V).
    - **Inversion:** $V_G > V_T$, strong electron inversion layer at surface; $\phi_s$ pinned at $\approx 2\phi_B$.

    Oxide $E_{{c,ox}}$ and $E_{{v,ox}}$ tilt under the oxide electric field.  Annotations show $q\phi_s$ (surface bending), $q\phi_B$ ($E_{{Fi}}-E_F$), and $qV_{{ox}}$ (oxide voltage drop).
    """)

    mo.vstack([mo.as_html(_fig_mos), _caption])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(np, plt):
    _fig_iv, (_ax_iv, _ax_sub) = plt.subplots(1, 2, figsize=(12, 4.5))

    _Cox = 3.9 * 8.854e-14 / 5e-7
    _mu = 400
    _WL = 10.0
    _VT_mos = 0.5

    _Vds = np.linspace(0, 3.0, 400)
    for _Vgs in [1.0, 1.5, 2.0, 2.5]:
        _Vov = _Vgs - _VT_mos
        if _Vov <= 0:
            continue
        _Vdsat = _Vov
        _Id_lin = _mu * _Cox * _WL * (_Vov * _Vds - _Vds**2 / 2)
        _Id_sat = _mu * _Cox * _WL / 2 * _Vov**2
        _Id = np.where(_Vds < _Vdsat, _Id_lin, _Id_sat)
        _ax_iv.plot(_Vds, _Id * 1e3, lw=2, label=f"$V_{{GS}}$ = {_Vgs} V")
    _ax_iv.set_xlabel("$V_{DS}$ (V)", fontsize=16)
    _ax_iv.set_ylabel("$I_D$ (mA)", fontsize=16)
    _ax_iv.set_title("Long-channel MOSFET I-V", fontsize=16)
    _ax_iv.legend(fontsize=12)
    _ax_iv.tick_params(labelsize=13)
    _ax_iv.set_ylim(bottom=0)

    _Vgs_sub = np.linspace(0, 1.0, 300)
    _n_sub = 1.3
    _Vds_sub = 0.5
    _Id_sub = 1e-7 * np.exp((_Vgs_sub - _VT_mos) / (_n_sub * 0.02585))
    _Id_sub = np.minimum(_Id_sub, 5e-3)
    _ax_sub.semilogy(_Vgs_sub, _Id_sub, lw=2, color="#d62728")
    _ax_sub.axvline(_VT_mos, color="gray", ls=":", lw=1)
    _ax_sub.text(_VT_mos + 0.02, 1e-10, "$V_T$", fontsize=14, color="gray")
    _SS_val = _n_sub * 60
    _ax_sub.annotate(
        f"SS = {_SS_val:.0f} mV/dec",
        xy=(0.25, 1e-10),
        fontsize=14,
        color="#d62728",
    )
    _ax_sub.set_xlabel("$V_{GS}$ (V)", fontsize=16)
    _ax_sub.set_ylabel("$I_D$ (A)", fontsize=16)
    _ax_sub.set_title("Subthreshold characteristics", fontsize=16)
    _ax_sub.set_ylim(1e-13, 1e-2)
    _ax_sub.tick_params(labelsize=13)

    _fig_iv.tight_layout()
    _fig_iv
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Left: Long-channel MOSFET output characteristics ($I_D$ vs $V_{DS}$) for several $V_{GS}$, showing linear and saturation regions. Right: Subthreshold (log-scale) transfer curve showing exponential turn-on below $V_T$ with subthreshold swing $SS \approx$ 78 mV/dec.*

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7C. Advanced FET Structures

    ### FinFET

    - 3D gate wraps around a thin **fin** → improved electrostatic control.
    - Reduces **short-channel effects** (DIBL, $V_T$ roll-off).
    - Used in production from 22 nm node onward (Intel, TSMC, Samsung).

    ### FDSOI (Fully Depleted Silicon-On-Insulator)

    - Thin body on buried oxide → body is fully depleted.
    - **Back-gate bias** provides an extra knob to tune $V_T$.
    - Good for low-power applications.

    ### Gate-All-Around (GAA) / Nanosheet

    - Gate surrounds the channel on **all four sides** → ultimate electrostatic control.
    - Transition from FinFET at ~3 nm node.
    - Each nanosheet is a thin horizontal Si slab; stack multiple for higher drive current.

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Bonus Topics: Optoelectronics and Quantum Devices

    *Interspersed throughout the course*

    ---

    ### Optical Absorption and Emission

    - Photon absorption: $E_{photon} = h\nu \ge E_g$ → electron-hole pair creation.
    - **Direct gap** materials (GaAs, InP): strong absorption, efficient emission.
    - **Indirect gap** (Si): weak absorption (phonon-assisted), poor emission → not used for lasers.

    ### Photodiodes and Solar Cells

    - Reverse-biased PN junction: photogenerated carriers swept by depletion field → photocurrent.
    - Solar cell = PN photodiode at zero or forward bias; power output = $I \times V$ in the fourth quadrant.

    ### LEDs and Lasers

    - Forward-biased PN junction in direct-gap material → radiative recombination → photon emission.
    - **LED:** spontaneous emission; **Laser:** stimulated emission with optical feedback (cavity).

    ### Quantum Confinement

    - When dimensions approach the de Broglie wavelength ($\sim$ nm), energy levels become **discrete**.
    - **Quantum well** (1D confinement): step-function density of states.
    - **Quantum wire** (2D), **quantum dot** (3D): increasingly sharp DoS peaks.
    - Applications: quantum dot lasers, single-photon sources, quantum computing qubits.

    ---
    """)
    return


@app.cell
def _(np, plt):
    _fig_opt, (_ax_abs, _ax_dos) = plt.subplots(1, 2, figsize=(12, 4.5))

    _E_ph = np.linspace(0.5, 3.5, 500)
    _Eg_Si = 1.12
    _Eg_GaAs = 1.42
    _alpha_Si = 1e3 * np.where(_E_ph > _Eg_Si, ((_E_ph - _Eg_Si) / 0.5) ** 2, 0)
    _alpha_GaAs = 1e4 * np.where(_E_ph > _Eg_GaAs, np.sqrt(np.maximum(_E_ph - _Eg_GaAs, 0) / 0.3), 0)

    _ax_abs.semilogy(_E_ph, _alpha_Si + 1, lw=2, label="Si (indirect)", color="#1f77b4")
    _ax_abs.semilogy(_E_ph, _alpha_GaAs + 1, lw=2, label="GaAs (direct)", color="#d62728")
    _ax_abs.set_xlabel("Photon energy (eV)", fontsize=16)
    _ax_abs.set_ylabel(r"$\alpha$ (cm$^{-1}$)", fontsize=16)
    _ax_abs.set_title("Optical absorption coefficient", fontsize=16)
    _ax_abs.set_xlim(0.5, 3.5)
    _ax_abs.set_ylim(1, 1e6)
    _ax_abs.legend(fontsize=13)
    _ax_abs.tick_params(labelsize=13)

    _E_qw = np.linspace(0, 1.5, 500)
    _dos_3d = np.where(_E_qw > 0, np.sqrt(_E_qw), 0)
    _E_levels = [0.1, 0.4, 0.9]
    _dos_qw = np.zeros_like(_E_qw)
    for _El in _E_levels:
        _dos_qw += np.where(_E_qw >= _El, 1.0, 0.0)

    _ax_dos.plot(_E_qw, _dos_3d / _dos_3d.max() * 3, lw=2, label="3D (bulk)", color="#1f77b4")
    _ax_dos.plot(_E_qw, _dos_qw, lw=2, label="2D (quantum well)", color="#d62728")
    _ax_dos.set_xlabel("$E - E_c$ (eV)", fontsize=16)
    _ax_dos.set_ylabel("Density of states (arb.)", fontsize=16)
    _ax_dos.set_title("DoS: bulk vs quantum well", fontsize=16)
    _ax_dos.legend(fontsize=13)
    _ax_dos.tick_params(labelsize=13)

    _fig_opt.tight_layout()
    _fig_opt
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Left: Schematic absorption coefficient — direct-gap GaAs has a sharp onset and much higher absorption than indirect-gap Si. Right: Density of states for 3D bulk ($\propto \sqrt{E}$) vs 2D quantum well (step function at each subband).*

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Key Equations Reference Card

    | Topic | Equation |
    |-------|----------|
    | Bloch theorem | $\psi_{nk} = e^{ikx} u_{nk}(x)$ |
    | Effective mass | $m^* = \hbar^2 (d^2E/dk^2)^{-1}$ |
    | Fermi–Dirac | $f(E) = [1 + e^{(E-E_F)/k_BT}]^{-1}$ |
    | Carrier conc. | $n_0 = N_c e^{-(E_c-E_F)/k_BT}$; $n_0 p_0 = n_i^2$ |
    | Drift current | $J_n = qn\mu_n\mathcal{E}$; $J_p = qp\mu_p\mathcal{E}$ |
    | Diffusion current | $J_n = qD_n\,dn/dx$; $J_p = -qD_p\,dp/dx$ |
    | Einstein | $D/\mu = k_BT/q$ |
    | Continuity | $\partial n/\partial t = (1/q)\partial J_n/\partial x + G - R$ |
    | Diffusion length | $L = \sqrt{D\tau}$ |
    | Poisson | $d\mathcal{E}/dx = \rho/\epsilon_s$ |
    | PN built-in | $\phi_{bi} = (k_BT/q)\ln(N_AN_D/n_i^2)$ |
    | Depletion width | $W = [2\epsilon_s(\phi_{bi}-V_A)/q \cdot (1/N_A+1/N_D)]^{1/2}$ |
    | Ideal diode | $I = I_0(e^{qV_A/k_BT}-1)$ |
    | BJT collector | $I_C = I_S\,e^{qV_{BE}/k_BT}$; $\beta_0 = G_E/G_B$ |
    | Schottky | $I_S = AA^*T^2 e^{-q\phi_{Bn}/k_BT}$ |
    | MOS threshold | $V_T = V_{FB} + 2\phi_B + Q_{dep}/C_{ox}'$ |
    | MOSFET (sat.) | $I_{D,sat} = \frac{\mu C_{ox}'}{2}\frac{W}{L}(V_{GS}-V_T)^2$ |
    | Subthreshold swing | $SS = n \times 60$ mV/dec (300 K) |
    | Transit frequency | $f_T = g_m / [2\pi(C_{gs}+C_{gd})]$ |

    ---
    """)
    return


@app.cell
def _(mo):
    log_na_vbi = mo.ui.slider(14.0, 18.0, value=17.0, step=0.1, label="log₁₀ Nₐ (cm⁻³)")
    log_nd_vbi = mo.ui.slider(14.0, 18.0, value=16.0, step=0.1, label="log₁₀ N_D (cm⁻³)")
    tau_p_calc = mo.ui.slider(0.01, 10.0, value=1.0, step=0.05, label="τₚ (μs)")
    D_p_calc = mo.ui.slider(5.0, 50.0, value=12.0, step=1.0, label="Dₚ (cm²/s)")
    mo.vstack([
        mo.md("### Interactive: Key parameter calculator"),
        mo.hstack([log_na_vbi, log_nd_vbi], justify="start"),
        mo.hstack([tau_p_calc, D_p_calc], justify="start"),
    ])
    return D_p_calc, log_na_vbi, log_nd_vbi, tau_p_calc


@app.cell
def _(D_p_calc, kT_eV, log_na_vbi, log_nd_vbi, mo, ni_si, np, tau_p_calc):
    _NA = 10.0 ** log_na_vbi.value
    _ND = 10.0 ** log_nd_vbi.value
    _Vbi = kT_eV * np.log(_NA * _ND / ni_si**2)
    _EF_n = kT_eV * np.log(_ND / ni_si)
    _EF_p = -kT_eV * np.log(_NA / ni_si)
    _p0_n = ni_si**2 / _ND
    _n0_p = ni_si**2 / _NA

    _Dp = D_p_calc.value
    _tau = tau_p_calc.value * 1e-6
    _Lp = np.sqrt(_Dp * _tau)

    mo.md(
        f"""
    | Quantity | Value |
    |----------|-------|
    | $\phi_{{bi}}$ | **{_Vbi:.4f} V** |
    | $E_F - E_{{F_i}}$ (n-side) | {_EF_n:.4f} eV |
    | $E_{{F_i}} - E_F$ (p-side) | {-_EF_p:.4f} eV |
    | $p_0$ on n-side ($n_i^2/N_D$) | {_p0_n:.2e} cm$^{{-3}}$ |
    | $n_0$ on p-side ($n_i^2/N_A$) | {_n0_p:.2e} cm$^{{-3}}$ |
    | Diffusion length $L_p$ | **{_Lp*1e4:.2f} $\\mu$m** |
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Exam Preparation Tips

    From the AI 😀, Professor agrees

    1. **Draw band diagrams** under bias for PN, BJT (forward active), MOS (all regimes), and MOSFET (with channel). Label $E_c$, $E_v$, $E_F$ (or $E_{F_n}$, $E_{F_p}$), depletion edges, and applied voltages.

    2. **Check and state your assumptions** before applying any formula: non-degenerate? abrupt junction? long channel? low injection? steady state?

    3. **Track signs** carefully: diffusion current signs flip between electrons and holes; recombination drives concentrations **toward** equilibrium.

    4. **Estimate before you calculate:** check if $\phi_{bi} \sim 0.7\text{–}0.9$ V makes sense for Si; check if $W \sim 0.1\text{–}1\;\mu$m is reasonable; check if $I_0 \sim 10^{-14}$ A is in the right ballpark.

    5. **Connect the units:** every device problem ultimately reduces to Poisson + continuity + boundary conditions. If you understand those three, you can derive rather than memorize.

    ---

    **Good luck on the exam, and thank you for a great term!**
    """)
    return


if __name__ == "__main__":
    app.run()
