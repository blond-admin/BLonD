# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from copy import deepcopy

    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.typing import NDArray as NumpyArray
    from scipy.integrate import (
        cumulative_simpson,  # type: ignore[import-untyped]
    )

    from blond import (
        Beam,
        DriftSimple,
        MagneticCyclePerTurn,
        Ring,
        Simulation,
        SingleHarmonicRFStation,
        backend,
        momentum_compaction_factor,
        proton,
    )
    from blond.beam_preparation.helpers import populate_beam
    from blond.core.beam.beams import ProbeBeam
    from blond.experimental.beam_preparation.bucket_filler_functions import (
        hamilton_to_density_by_max,
    )
    from blond.experimental.beam_preparation.semi_empiric_matcher import (
        SemiEmpiricMatcher,
    )
    from blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.callables import (
        occupation_per_equipotential_to_density,
        occupation_per_equipotential_to_histogram,
    )
    from blond.generals.formatting_ import si_format

    return (
        Beam,
        DriftSimple,
        MagneticCyclePerTurn,
        NumpyArray,
        ProbeBeam,
        Ring,
        SemiEmpiricMatcher,
        Simulation,
        SingleHarmonicRFStation,
        backend,
        cumulative_simpson,
        deepcopy,
        hamilton_to_density_by_max,
        momentum_compaction_factor,
        np,
        occupation_per_equipotential_to_density,
        occupation_per_equipotential_to_histogram,
        plt,
        populate_beam,
        proton,
        si_format,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using the Semi-Empiric Matcher

    The `SemiEmpiricMatcher` prepares a **matched** longitudinal beam distribution for BLonD simulations. A matched distribution is stationary in phase space: particles follow closed orbits on constant-Hamiltonian contours and the bunch shape does not evolve turn-by-turn.

    The approach is *semi-empiric*: the RF potential well is derived **numerically** from a single simulated turn, while the kinetic (drift) term of the Hamiltonian is computed **analytically** from the machine optics. This hybrid strategy handles arbitrary RF waveforms and impedance contributions without requiring a closed-form expression for the voltage.

    ## Define the Simulation Context
    """)
    return


@app.cell
def _(
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    np,
    proton,
):
    energy_cycle = MagneticCyclePerTurn.init_from_linspace(
        values=np.linspace(450e9, 470e9, int(1e4)),
        reference_particle=proton,
    )

    _ring = Ring(26658.883)

    _rf_station = SingleHarmonicRFStation(
        harmonic=35640, voltage=6e6, phi_rf=0
    )

    drift1 = DriftSimple(orbit_length=26658.883)
    drift1.momentum_compaction_factor = momentum_compaction_factor(
        transition_gamma=55.759505
    )

    beam1 = Beam(intensity=1e9, particle_type=proton)

    # marimo shares one namespace between all cells, so
    # `Simulation.from_locals(locals())` would also find the ring of the
    # second simulation further below
    _ring.add_elements((_rf_station, drift1), reorder=True)
    sim = Simulation(ring=_ring, magnetic_cycle=energy_cycle)
    return beam1, drift1, energy_cycle, sim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate the Potential Well

    To construct the Hamiltonian we need the RF potential well $V(\Delta T)$. Rather than integrating the RF voltage analytically, we derive it empirically: a zero-energy *probe bunch* is tracked for one turn and the $\Delta E$ kicks it receives are integrated.

    A `ProbeBeam` is a zero-intensity beam placed along the time axis at $\Delta E = 0$. It samples the RF phase at equidistant points across the bucket window.
    """)
    return


@app.cell
def _(ProbeBeam, beam1, deepcopy, energy_cycle, np):
    dt = np.linspace(0, 10e-9, 100)

    probe_bunch = ProbeBeam(
        dt=dt,
        particle_type=beam1.particle_type,
        intensity=0,
        reference_total_energy=energy_cycle.get_total_energy_init(),
    )
    probe_bunch_init = deepcopy(probe_bunch)
    return probe_bunch, probe_bunch_init


@app.cell
def _(plt, probe_bunch_init):
    plt.figure(figsize=(8, 1))
    probe_bunch_init.plot_scatter()
    plt.ylabel("Energy Offset (eV)")
    plt.xlabel("Time (s)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tracking the Probe Beam for One Turn

    We run exactly one simulation turn. Each probe particle receives an energy kick $\Delta E(\Delta t)$ from the RF cavity. The revolution period $T_\text{rev}$ is read back from the reference particle and used to normalise the potential in the next step.
    """)
    return


@app.cell
def _(deepcopy, probe_bunch, sim):
    probe_bunch_out = deepcopy(probe_bunch)

    deepcopy(sim).run_simulation(
        beams=(probe_bunch_out,),
        n_turns=1,
        show_progressbar=False,
        verbose=False,
    )
    return (probe_bunch_out,)


@app.cell
def _(probe_bunch_init, probe_bunch_out, si_format):
    t_rev = probe_bunch_out.reference.time - probe_bunch_init.reference.time
    si_format(t_rev) + "s"
    return (t_rev,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Energy Kicks Received by the Probe Particles

    Plotting $\Delta E$ vs. $\Delta t$ reveals the RF waveform sampled at the probe positions. The sinusoidal shape is the expected response of a single-harmonic RF cavity.
    """)
    return


@app.cell
def _(plt, probe_bunch_out):
    plt.plot(
        probe_bunch_out.dt.array_local,
        probe_bunch_out.dE.array_local,
        ".",
        c="C0",
    )
    plt.ylabel("Energy Offset (eV)")
    plt.xlabel("Time (s)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Integrating the Kicks into a Potential Well

    The potential well is obtained by integrating the energy kicks along the time axis:

    $$V(\Delta t) = -\int_0^\Delta t \Delta E(\Delta t')\,d\Delta t'$$

    A normalisation factor $T_\text{bucket} / T_\text{rev}$ converts from the probe-particle spacing to physical time. The minimum is then shifted to zero so that $V(\Delta t) \geq 0$. The built-in helper `sim.plot_potential_well_empiric` performs the same calculation internally.
    """)
    return


@app.cell
def _(plt, probe_bunch_init, probe_bunch_out):
    def plot_potential_well(potential_well):
        ax = plt.subplot(2, 1, 1)
        plt.plot(
            probe_bunch_init.dt.array_local,
            probe_bunch_out.dE.array_local,
            c="C0",
        )
        plt.ylabel("Energy Offset (eV)")

        plt.subplot(2, 1, 2, sharex=ax)
        plt.plot(probe_bunch_init.dt.array_local, potential_well, c="C1")
        plt.ylabel("Potential [arb. unit]")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        return plt.gca()

    return (plot_potential_well,)


@app.cell
def _(
    cumulative_simpson,
    np,
    plot_potential_well,
    probe_bunch_init,
    probe_bunch_out,
    t_rev,
):
    potential_well = -cumulative_simpson(
        probe_bunch_out.dE.copy_as_numpy(),
        initial=0,
    ) / len(probe_bunch_out.dE.array_local)
    norm = (
        probe_bunch_init.dt.array_local[-1]
        - probe_bunch_init.dt.array_local[0]
    ) / t_rev
    potential_well *= norm
    potential_well -= np.min(potential_well)  # make the minimum value 0
    plot_potential_well(potential_well)
    return (potential_well,)


@app.cell
def _(plt, probe_bunch_init, sim):
    sim.plot_potential_well_empiric(
        dt=probe_bunch_init.dt.array_local,
        particle_type=probe_bunch_init.reference.particle_type,
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construct the Hamiltonian

    This is why we call it *semi* empiric matching. The `drift_term` is used in its analytical form.
    """)
    return


@app.cell
def _(deltaE_grid, plt, time_grid):
    def plot_hamilton(hamilton_2D):
        plt.figure(figsize=(6, 5))
        plt.pcolormesh(
            time_grid, deltaE_grid, hamilton_2D, shading="auto", cmap="turbo"
        )
        plt.colorbar(label="Hamiltonian value")
        plt.ylabel("Energy Offset (eV)")
        plt.xlabel("Time (s)")
        return plt.gca()

    return (plot_hamilton,)


@app.cell
def _(drift1, probe_bunch_init):
    eta = drift1.eta_0(gamma=probe_bunch_init.reference.gamma)
    beta = probe_bunch_init.reference.beta
    E0 = probe_bunch_init.reference.total_energy
    return E0, beta, eta


@app.cell
def _(E0, beta, eta, np):
    drift_term = np.abs(eta) / (np.square(beta) * E0)  # [1/eV]
    return (drift_term,)


@app.cell
def _(drift_term, np, potential_well):
    dE_max = np.sqrt(  # automatically derive the plot limit
        (potential_well.max()) / (abs(drift_term))
    )
    return (dE_max,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Building the 2D Phase Space Grid

    We evaluate $\mathcal{H}$ on a regular $(\Delta t, \Delta E)$ mesh. The energy range is bounded by $\Delta E_\text{max}$, the energy at which the separatrix is reached (where the kinetic term equals the peak of the potential), keeping the grid tightly focused on the stable region.
    """)
    return


@app.cell
def _(dE_max, np, probe_bunch_init):
    _dE_base = np.linspace(-dE_max, dE_max, 100)
    time_grid, deltaE_grid = np.meshgrid(
        probe_bunch_init.dt.array_local, _dE_base, indexing="ij"
    )
    return deltaE_grid, time_grid


@app.cell
def _(deltaE_grid, drift_term, np, plot_hamilton, potential_well):
    V = potential_well[:, None]
    hamilton_2D = 0.5 * drift_term * np.square(deltaE_grid) + V  # [eV]
    plot_hamilton(hamilton_2D)
    return (hamilton_2D,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generating a Particle Distribution

    Once the Hamiltonian is known, a matched distribution requires only that the particle density $\rho$ is a function of $\mathcal{H}$ alone — each iso-Hamiltonian contour must be uniformly populated. The choice of $\rho(\mathcal{H})$ determines the projected bunch shape.

    BLonD provides the built-in `hamilton_to_density_by_max` algorithm. The four steps below show how it works before the full `SemiEmpiricMatcher` API is introduced.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Algorithm 1: Max Clipping

    #### Step 1 — Clip to the Separatrix

    Cells with $\mathcal{H} > \mathcal{H}_\text{sep} \cdot f_\text{cut}$ (outside or near the bucket edge) are replaced by the separatrix value. This hard-clips the density to the stable region. The fraction `hamilton_max` (e.g. 90\% of the separatrix value) controls how close to the edge the distribution is allowed to reach.
    """)
    return


@app.cell
def _(deepcopy, hamilton_2D, plot_hamilton, potential_well):
    density_clipped = deepcopy(hamilton_2D)
    select = 0.9 * potential_well[0] < hamilton_2D
    density_clipped[select] = potential_well[0]
    plot_hamilton(density_clipped)
    return (density_clipped,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Step 2 — Invert the Distribution

    After clipping, low-Hamiltonian (core) cells have small values and edge cells have large values. We invert by computing `density_max - density` so that the bunch core becomes the most densely populated region.
    """)
    return


@app.cell
def _(density_clipped, plot_hamilton):
    density_inverted = density_clipped.max() - density_clipped
    plot_hamilton(density_inverted)
    return (density_inverted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Step 3 — Shape with an Exponent

    Raising the inverted density to a power `n` (the `density_modifier`) sharpens the profile. Higher exponents concentrate more particles near the core and produce a more Gaussian-like line-density projection. The default is `n = 2`.
    """)
    return


@app.cell
def _(density_inverted, plot_hamilton):
    exponent = 3
    density_shaped = density_inverted**exponent
    plot_hamilton(density_shaped)
    return (density_shaped,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Step 4 — Sample Macro-Particles

    The final density grid is used as a 2D probability distribution from which macro-particles are drawn via rejection sampling. The resulting beam has `n_macroparticles` particles distributed according to the computed density.
    """)
    return


@app.cell
def _(
    deepcopy,
    deltaE_grid,
    density_shaped,
    populate_beam,
    probe_bunch,
    time_grid,
):
    beam = deepcopy(probe_bunch)
    populate_beam(
        beam=beam,
        time_grid=time_grid.T,
        deltaE_grid=deltaE_grid.T,
        density_grid=density_shaped.T,
        n_macroparticles=int(1e6),
        seed=0,
    )
    return (beam,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Verifying the Match

    A well-matched beam should look nearly identical after many turns. We run a short simulation and compare the phase-space portrait at turn 0 and after one passage. Significant filamentation or shape change indicates a mismatch.
    """)
    return


@app.cell
def _(beam, deltaE_grid, plt, sim_tmp, time_grid):
    def plot_before():
        ax = plt.subplot(1, 2, 1)
        plt.title("Turn 0")
        beam.plot_hist2d(
            range=(
                (time_grid.min(), 2e-9),
                (deltaE_grid.min(), deltaE_grid.max()),
            )
        )
        return ax

    def plot_after(ax):
        plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
        plt.title(f"Turn {sim_tmp.turn_counter.value}")
        beam.plot_hist2d(
            range=(
                (time_grid.min(), 2e-9),
                (deltaE_grid.min(), deltaE_grid.max()),
            )
        )
        return plt.gca()

    return plot_after, plot_before


@app.cell
def _(backend, deepcopy, sim):
    backend.set_specials("cpp")
    sim_tmp = deepcopy(sim)
    kwargs = dict(show_progressbar=False, verbose=False)
    return kwargs, sim_tmp


@app.cell
def _(beam, kwargs, plot_after, plot_before, sim_tmp):
    _ax = plot_before()
    sim_tmp.run_simulation(beams=(beam,), **kwargs)
    plot_after(_ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Providing a Custom Algorithm

    Not all use cases are covered by `hamilton_to_density_by_max`. You can supply any callable with the signature below to `SemiEmpiricMatcher` via `hamilton_to_density_function`. Additional keyword arguments are forwarded through `hamilton_to_density_kwargs`.
    """)
    return


@app.cell
def _(NumpyArray):
    def hamilton_to_density_by_user(
        time_grid: NumpyArray,
        deltaE_grid: NumpyArray,
        hamilton_2D: NumpyArray,
        your_argument_01,  # add user defined arguments
        your_argument_02,  # add user defined arguments
    ) -> NumpyArray: ...

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using the SemiEmpiricMatcher API

    The four steps above are wrapped inside `SemiEmpiricMatcher`. Pass it to `sim.prepare_beam` and it handles probe tracking, potential integration, Hamiltonian construction, and particle sampling in a single call.

    Setting `maxiter_intensity_effects=0` disables the self-consistent loop for space-charge and impedance-driven potential distortion. Increase it to account for intensity effects.
    """)
    return


@app.cell
def _(
    SemiEmpiricMatcher,
    deepcopy,
    hamilton_to_density_by_max,
    potential_well,
    probe_bunch,
    sim,
):
    beam_matched = deepcopy(probe_bunch)
    sim.prepare_beam(
        beam=beam_matched,
        preparation_routine=SemiEmpiricMatcher(
            time_limit=(0, 2.0e-9),
            n_macroparticles=1e6,
            seed=0,
            maxiter_intensity_effects=0,
            hamilton_to_density_function=hamilton_to_density_by_max,
            hamilton_to_density_kwargs=dict(
                density_modifier=2.0,  # Controls density profile sharpness
                hamilton_max=0.9
                * potential_well[0],  # Hamiltonian cutoff [eV]
            ),
            animate=True,
        ),
    )
    return (beam_matched,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validating the SemiEmpiricMatcher Output

    We repeat the before/after comparison — phase-space portrait at turn 0 vs. after the passage — but now starting from a beam generated by `SemiEmpiricMatcher`. A stable, non-filamenting portrait confirms that the matching succeeded.
    """)
    return


@app.cell
def _(backend, beam_matched, deepcopy, deltaE_grid, plt, sim, time_grid):
    _ax = plt.subplot(1, 2, 1)
    plt.title("Turn 0")
    beam_matched.plot_hist2d(
        range=((time_grid.min(), 2e-9), (deltaE_grid.min(), deltaE_grid.max()))
    )
    backend.set_specials("cpp")
    _sim_tmp = deepcopy(sim)
    _sim_tmp.run_simulation(
        beams=(beam_matched,),
        show_progressbar=False,
        verbose=False,
    )

    plt.subplot(1, 2, 2, sharex=_ax, sharey=_ax)
    plt.title(f"Turn {_sim_tmp.turn_counter.value}")
    beam_matched.plot_hist2d(
        range=((time_grid.min(), 2e-9), (deltaE_grid.min(), deltaE_grid.max()))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting Line Densities

    In many operational scenarios the target bunch shape must match a **measured** profile. The `ProfileMatcherAddon` extension solves for the *occupation-per-equipotential* vector $\mathbf{p}$ such that the projected line density reproduces the measured histogram.

    The following cells build up the concept from first principles before demonstrating the full API.
    """)
    return


@app.cell
def _(histogram, occupation_per_equipotential, plt):
    def plot_occupation(hamilton_2D, density):
        plt.figure(figsize=(6, 5))
        plt.subplot(4, 1, 1)
        plt.plot(
            occupation_per_equipotential, label="Population State", c="C0"
        )
        plt.legend()
        plt.subplot(4, 1, 2)
        plt.pcolormesh(hamilton_2D.T, shading="auto", cmap="turbo")
        plt.subplot(4, 1, 3)
        plt.pcolormesh(density.T, shading="auto", cmap="turbo")
        plt.subplot(4, 1, 4)
        plt.plot(histogram, label="Histogram", c="C1")
        plt.legend()
        plt.tight_layout()
        return plt.gca()

    return (plot_occupation,)


@app.cell
def _(hamilton_2D, np, probe_bunch_init):
    idx = np.argmax(probe_bunch_init.dt.array_local > 2e-9)
    hamilton_2D_cropped = hamilton_2D[:idx, :]
    return (hamilton_2D_cropped,)


@app.cell
def _(
    hamilton_2D_cropped,
    np,
    occupation_per_equipotential_to_density,
    occupation_per_equipotential_to_histogram,
):
    density_shell = np.empty_like(hamilton_2D_cropped)
    occupation_per_equipotential = np.zeros(hamilton_2D_cropped.shape[0])
    occupation_per_equipotential[len(occupation_per_equipotential) // 4] = 1

    occupation_per_equipotential_to_density(
        occupation_per_equipotential=occupation_per_equipotential,
        potential_2D=hamilton_2D_cropped,
        density_write=density_shell,
    )

    histogram = occupation_per_equipotential_to_histogram(
        occupation_per_equipotential=occupation_per_equipotential,
        potential_2D=hamilton_2D_cropped,
    )
    return density_shell, histogram, occupation_per_equipotential


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualising the Density from an Occupation Vector

    The 3-panel plot shows how a single non-zero entry in $\mathbf{p}$ (population concentrated on one equipotential shell) maps to a 2D density and then to a 1D line-density histogram. A full distribution is a superposition of contributions from all shells.
    """)
    return


@app.cell
def _(density_shell, hamilton_2D_cropped, plot_occupation):
    plot_occupation(hamilton_2D_cropped, density_shell)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fitting a Measured Profile with `ProfileMatcherAddon`

    `ProfileMatcherAddon` wraps an optimisation loop that finds the occupation vector $\mathbf{p}$ whose projected histogram best matches the target. It plugs directly into `SemiEmpiricMatcher` as a drop-in `hamilton_to_density_function`, so the rest of the workflow is unchanged.

    Key hyperparameters:

    | Parameter | Effect |
    |-----------|--------|
    | `smoothness` | Regularisation strength — higher values enforce a smoother occupation vector |
    | `atol` | Convergence tolerance on the histogram residual |
    | `recenter` | Shift the fitted distribution to the bunch centre before sampling |
    """)
    return


@app.cell
def _():
    from blond.experimental import ProfileMatcherAddon

    return (ProfileMatcherAddon,)


@app.cell
def _(np):
    def get_test_profile(noisy=False):
        """
        Create a histogram that looks like a measurement.

        Parameters
        ----------
        noisy
            If it should be generated with or without noise.

        Returns
        -------
        hist_x
            Histogram time axis.
        hist_y
            Histogram amplitude.
        """
        # Parameters
        mean = 2.5e-9 / 2  # Mean of the distribution
        std_dev = 2.5e-9 / 30  # Standard deviation
        size = 10000  # Number of data points

        if noisy:  # pragma: no cover
            # Generate random data from a Gaussian distribution
            data = np.random.normal(loc=mean, scale=std_dev, size=size)

            # Get the histogram (density=False for raw counts)
            hist_y, bin_edges = np.histogram(data, bins=512, density=False)
            hist_x = bin_edges[0:-1] + np.diff(bin_edges[:2])[0] / 2
        else:
            hist_x = np.linspace(0, 2.5e-9, 100)
            hist_y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
                -((hist_x - mean) ** 2) / (2 * std_dev**2)
            )
        return hist_x, hist_y

    return (get_test_profile,)


@app.cell
def _(
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    np,
    proton,
):
    _ring = Ring(26658.883)

    _rf_station = SingleHarmonicRFStation()
    _rf_station.harmonic = 35640
    _rf_station.voltage = 6e6
    _rf_station.phi_rf_design = 0

    _values = np.linspace(450e9, 450e9, int(1e3) + 1)
    _energy_cycle = MagneticCyclePerTurn(
        value_init=_values[0],
        values_after_turn=_values[1:],
        reference_particle=proton,
    )

    _drift = DriftSimple(
        orbit_length=26658.883,
    )
    _drift.momentum_compaction_factor = momentum_compaction_factor(
        transition_gamma=55.759505
    )
    beam_fit = Beam(
        intensity=1e9,
        particle_type=proton,
    )

    _ring.add_elements((_rf_station, _drift), reorder=True)
    sim_fit = Simulation(ring=_ring, magnetic_cycle=_energy_cycle)
    return beam_fit, sim_fit


@app.cell
def _(ProfileMatcherAddon, get_test_profile):
    hist_x, hist_y = get_test_profile()
    matcher_addon = ProfileMatcherAddon(hist_x=hist_x, hist_y=hist_y)
    matcher_addon.smoothness = 0.01
    matcher_addon.atol = 1e-3
    matcher_addon.recenter = True
    matcher_addon.animate_fitting = False
    matcher_addon.plot_result = True
    matcher_addon.plot_result_blocking = True
    return (matcher_addon,)


@app.cell
def _(SemiEmpiricMatcher, beam_fit, matcher_addon, sim_fit):
    sim_fit.prepare_beam(
        beam=beam_fit,
        preparation_routine=SemiEmpiricMatcher(
            time_limit=(0, 2.5e-9),
            n_macroparticles=int(1e6),
            seed=0,
            maxiter_intensity_effects=0,
            internal_grid_shape=(1024, 1024),
            hamilton_to_density_function=matcher_addon.hamilton_to_density_function,
            hamilton_to_density_kwargs=dict(),
            animate=False,
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Further Items Not Discussed Here

    The following topics are implemented or partially implemented but not covered in this tutorial:

    - **Intensity effects** — self-consistent iteration over impedance-driven potential distortion is available via `maxiter_intensity_effects > 0`
    - **Multiple RF buckets** — partial support as of March 2024; multi-bucket scenarios may need manual adjustments
    - **Ill-constrained line densities** — the optimiser can diverge if the target histogram is wider than the bucket or has features outside the separatrix
    - **Unstable operating points** — matching cannot converge above the intensity threshold
    - Anti-aliasing in `occupation_per_equipotential_to_density`
    - Histogram recentering edge cases
    - State-vector smoothing strategies
    - Iterative solver for line-density fitting
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Roadmap

    Planned improvements and research directions:

    - Account for synchrotron radiation damping and quantum excitation
    - Handle parametric resonances near low-order resonance conditions
    - Robust multi-drift / multi-RF support
    - Fully empiric implementation without an analytic drift term
    - Hybrid Lie-algebra + per-element numeric potential well
    - Per-bucket sparse matching with coupled-bunch intensity effects
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    - `SemiEmpiricMatcher` reliably produces matched distributions and is ready for user testing
    - Custom `hamilton_to_density` functions allow adaptation to specialised use cases
    - `ProfileMatcherAddon` extends the matcher to reproduce measured bunch profiles

    ### See Also

    - [Documentation: Semi-Empiric Matcher](https://blond-code.docs.cern.ch/models_new/semi_empiric_matcher.html)
    - `blond/examples/scripts/EX_06_Acceleration_match_density.py`
    - `blond/examples/scripts/EX_08_Semi_empiric_matcher.py`
    - `blond/experimental/beam_preparation/semi_empiric_matcher.py`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contributing and Feedback

    Bug reports and feature requests are welcome. If you encounter unexpected behaviour or have suggestions for improving the matcher, please open an issue on the BLonD repository.

    ---

    *Thank you for reading this tutorial.*
    """)
    return


if __name__ == "__main__":
    app.run()
