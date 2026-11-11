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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Getting Started

    This guide walks you through setting up and running your first BLonD simulation using the EX_01_Acceleration example.

    ## Overview

    A BLonD simulation consists of several key components:

    1. Physical components (Ring, RF stations, Drift sections)
    2. Beam definition
    3. Energy/magnetic cycle
    4. Beam preparation
    5. Observations
    6. Simulation execution

    ## Setting Up the Simulation

    ### Creating the Ring

    The ring defines the circumference of your accelerator and
    holds all elements that are executed for simulating the beam behaviour:
    """)
    return


@app.cell
def _():
    import os
    import tempfile
    from copy import deepcopy

    import numpy as np
    from matplotlib import pyplot as plt

    import blond
    from blond import Beam, Simulation, proton
    from blond.handle_results.helpers import callers_relative_path
    from blond.testing import pytest_active

    pytest_is_active = pytest_active()
    return (
        Beam,
        Simulation,
        blond,
        callers_relative_path,
        deepcopy,
        np,
        os,
        plt,
        proton,
        pytest_is_active,
        tempfile,
    )


@app.cell
def _(blond, pytest_is_active):
    if not pytest_is_active:  # pragma: no cover
        blond.backend.set_specials(
            "numba"
        )  # select a faster backend, if available.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setting Up the Energy Cycle

    The magnetic cycle defines how the beam energy evolves during the simulation. BLonD provides several types of cycles.

    #### Constant Energy

    For simulations with constant beam energy (no acceleration):
    """)
    return


@app.cell
def _():
    from blond import Ring

    ring = Ring(26658.883)  # circumference in meters
    return (ring,)


@app.cell
def _(proton):
    from blond import ConstantMagneticCycle

    constant_cycle = ConstantMagneticCycle(
        reference_particle=proton,
        value=450e9,  # energy value
        in_unit="total energy",  # can be 'momentum', 'total energy', 'kinetic energy', or 'bending field'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the simplest and most efficient option when the beam energy doesn't change.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Energy Ramp (Per Turn)

    For simulations with acceleration, where energy changes turn-by-turn:
    """)
    return


@app.cell
def _(np, proton):
    from blond import MagneticCyclePerTurn

    N_TURNS = int(1e3)

    ramp_cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        value_init=450e9,  # initial energy
        values_after_turn=np.linspace(
            450e9, 460e9, N_TURNS
        ),  # energy after each turn
        in_unit="total energy",
    )
    return (N_TURNS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cycle assumes each RF station provides an equal fraction of the energy kick per turn.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Advanced Cycle Options

    **Per-RF-Station Control (MagneticCyclePerTurnAllRfStations)**

    For full control over each RF station's contribution at each turn:
    """)
    return


@app.cell
def _(N_TURNS, np, proton):
    from blond import MagneticCyclePerTurnAllRFStations

    # REQUIRED: 2D array with shape (n_rf_stations, n_turns)
    # Each row represents one RF station, each column represents one turn
    # Example for 4 RF stations and 1000 turns:
    n_rf_stations = 4

    energy_per_rf_station = np.zeros((n_rf_stations, N_TURNS))
    # Write the desired values in `energy_per_rf_station`
    per_station_cycle = MagneticCyclePerTurnAllRFStations(
        reference_particle=proton,
        value_init=450e9,
        values_after_rf_station_per_turn=energy_per_rf_station,  # 2D array: (rf_stations, turns)
        in_unit="total energy",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Time-Based Cycle (MagneticCycleByTime)**

    For cycles defined by continuous time interpolation:
    """)
    return


@app.cell
def _(np, proton):
    from scipy.interpolate import interp1d

    from blond import MagneticCycleByTime

    time_points = np.array([0, 1.0, 2.0, 3.0])  # time in seconds
    energy_values = np.array([450e9, 500e9, 600e9, 700e9])

    energy_cycle = MagneticCycleByTime(
        reference_particle=proton,
        reference_time=time_points,
        reference_values=energy_values,
        in_unit="total energy",
        interpolator=interp1d,  # interpolation function
    )
    return (energy_cycle,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All cycle types support the ``in_unit`` parameter with options: ``'momentum'``, ``'total energy'``, ``'kinetic energy'``, or ``'bending field'``.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Defining RF Stations

    RF stations provide acceleration and longitudinal focusing. A single harmonic
    RF station is defined by its harmonic number, voltage, and phase:
    """)
    return


@app.cell
def _():
    from blond import SingleHarmonicRFStation

    rf_station1 = SingleHarmonicRFStation()
    rf_station1.harmonic = 35640  # harmonic number
    rf_station1.voltage = 6e6  # voltage in V
    rf_station1.phi_rf_design = 0  # RF phase in radians
    return (rf_station1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is also possible to plan the value of those variables ahead in time. This will also work for drifts.
    """)
    return


@app.cell
def _(N_TURNS, np, rf_station1):
    rf_station1.schedule("voltage", 6e6 * np.ones(N_TURNS))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Configuring Drift Sections

    Drift sections represent regions without RF stations:
    """)
    return


@app.cell
def _():
    from blond import DriftSimple, momentum_compaction_factor

    drift1 = DriftSimple(orbit_length=26658.883)
    drift1.momentum_compaction_factor = momentum_compaction_factor(
        transition_gamma=55.759505
    )
    return (drift1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setting up the Beam Profile
    """)
    return


@app.cell
def _(energy_cycle, rf_station1, ring):
    from blond import StaticProfile

    profile1 = StaticProfile(
        cut_left=0,
        cut_right=energy_cycle.get_t_rev_init(ring.circumference)
        / rf_station1.get_main_harmonic(),
        n_bins=256,
    )
    return (profile1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wakefields

    Wakefields/impedances will disturb the beam by inducing additional voltages that affect the beam dynamics.
    """)
    return


@app.cell
def _(profile1):
    from blond import Resonators, TimeDomainFftSolver, WakeField

    wakefield1 = WakeField(
        sources=(Resonators(200e3, 200e6, 10e10),),
        solver=TimeDomainFftSolver(),
        profile=profile1,
    )
    return (wakefield1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Assembling the Simulation

    The ``Simulation.from_locals()`` method automatically discovers and connects all components defined in the local scope:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Populating the Beam

    The beam object holds particle information:
    """)
    return


@app.cell
def _(Beam, proton):
    beam1 = Beam(
        intensity=1e9,  # number of real particles
        particle_type=proton,  # particle type
    )
    return (beam1,)


@app.cell
def _(Simulation, drift1, energy_cycle, rf_station1, ring, wakefield1):
    ring.add_elements((wakefield1, rf_station1, drift1))
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

    # or alternatively to automatically discover the variables
    # sim = Simulation.from_locals(locals())

    sim.print_one_turn_execution_order()  # optional: print execution order
    return (sim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ``locals()`` function returns a dictionary of all variables in the current scope. ``Simulation.from_locals()`` inspects this dictionary and automatically detects all BLonD components (ring, RF stations, drifts, beam, energy cycle) without requiring you to pass each one explicitly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before running the simulation, the beam phase space must be populated with macroparticles. BLonD offers several beam preparation methods:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### BiGaussian Distribution

    The most commonly used method creates a simple Gaussian distribution in both time and energy:
    """)
    return


@app.cell
def _(beam1, sim):
    from blond import BiGaussian

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=BiGaussian(
            sigma_dt=0.4e-9 / 4,  # time spread (standard deviation) in seconds
            sigma_dE=1e9 / 4,  # energy spread (standard deviation) in eV
            seed=1,  # random seed for reproducibility
            n_macroparticles=1e6,  # number of macroparticles
        ),
    )
    return


@app.cell
def _(beam1):
    beam1.plot_hist2d()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This creates $10^6$ macroparticles with a Gaussian distribution. It's simple, fast, and suitable for most applications.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Semi-Empiric Matcher (Experimental)

    A physics-based matching approach that uses the Hamiltonian to generate matched distributions:

    For this, you need to know how your potential well looks like and how much of it to populate.
    """)
    return


@app.cell
def _(np, plt, proton, sim):
    sim.plot_potential_well_empiric(
        dt=np.linspace(0, 2.5e-9),
        particle_type=proton,
    )
    hamilton_max = 40
    plt.axhline(hamilton_max)
    return (hamilton_max,)


@app.cell
def _(beam1, hamilton_max, sim):
    from blond.experimental.beam_preparation.semi_empiric_matcher import (
        SemiEmpiricMatcher,
    )

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=SemiEmpiricMatcher(
            time_limit=(0, 2.5e-9),
            n_macroparticles=1e6,
            seed=0,
            hamilton_to_density_kwargs=dict(
                density_modifier=2.0,  # Controls density profile sharpness
                hamilton_max=hamilton_max,  # Hamiltonian cutoff [eV]
            ),
            animate=True,
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This method calculates the Hamiltonian and generates particles following matched trajectories, making it ideal for stationary distributions.
    """)
    return


@app.cell
def _(beam1):
    beam1.plot_hist2d()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Other Methods

    Additional beam preparation methods are under development in the ``blond.experimental.beam_preparation`` module. Check the source code for the latest available options.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting Up Observations

    Observations allow you to record data during the simulation for later analysis. BLonD provides several observation classes for different purposes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Available Observations

    #### BeamObservationOncePerTurn - Complete Beam Distribution

    Records the full longitudinal phase space coordinates (time and energy) of all macroparticles at the end of each specified turn, allowing you to analyze beam evolution and create phase space plots.
    """)
    return


@app.cell
def _():
    from blond import BeamObservationOncePerTurn

    bunch_observation = BeamObservationOncePerTurn(
        each_turn_i=1,  # record every turn
    )
    return (bunch_observation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Access recorded data via:

    - ``bunch_observation.dts``: Time coordinates [s] for all particles (shape: n_turns × n_macroparticles)
    - ``bunch_observation.dEs``: Energy deviations [eV] for all particles (shape: n_turns × n_macroparticles)
    - ``bunch_observation.flags``: Particle status flags (e.g., lost particles)
    - ``bunch_observation.reference_time``: Reference time per turn [s]
    - ``bunch_observation.reference_total_energy``: Reference energy per turn [eV]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### BunchObservationMetaParams - Statistical Parameters

    Records mean and standard deviation of time and energy coordinates, plus the statistical emittance:
    """)
    return


@app.cell
def _(callers_relative_path, os, pytest_is_active, tempfile):
    from blond import BeamStatisticsOncePerTurn

    if pytest_is_active:  # keep test runs out of the package directory
        results_folder = tempfile.mkdtemp() + os.sep
    else:  # pragma: no cover
        results_folder = callers_relative_path("results/", stacklevel=1)
    os.makedirs(results_folder, exist_ok=True)
    stats_observation = BeamStatisticsOncePerTurn(
        each_turn_i=1,  # record every turn
        folder=results_folder,
    )
    return (stats_observation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Access recorded data via:

    - ``stats_observation.bunch_position``: Mean time coordinate [s]
    - ``stats_observation.bunch_length``: Time coordinate standard deviation [s]
    - ``stats_observation.energy_spread``: Energy deviation standard deviation [eV]
    - ``stats_observation.reference_time``:  Reference total energy [eV]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### RfStationPhaseObservation - RF Station Parameters

    Tracks the evolution of RF station parameters (phase, frequency, voltage):
    """)
    return


@app.cell
def _(rf_station1):
    from blond import RFStationPhaseObservation

    phase_observation = RFStationPhaseObservation(
        each_turn_i=1,  # record every turn
        rf_station=rf_station1,  # which RF station to observe
    )
    return (phase_observation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Access recorded data via:

    - ``phase_observation.phases``: RF phase [rad] (shape: n_turns × n_harmonics)
    - ``phase_observation.omegas``: Angular frequency [rad/s] (shape: n_turns × n_harmonics)
    - ``phase_observation.voltages``: RF voltage [V] (shape: n_turns × n_harmonics)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### DriftObservation - Drift Parameters
    """)
    return


@app.cell
def _(drift1):
    from blond import DriftObservation

    drift_observation = DriftObservation(
        each_turn_i=1,
        drift=drift1,
    )
    return (drift_observation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### StaticProfileObservation - Beam Profile (Fixed Bins)

    Observes a beam profile with fixed bin positions and widths:
    """)
    return


@app.cell
def _(profile1):
    from blond import StaticProfileObservation

    profile_observation = StaticProfileObservation(
        each_turn_i=1,
        profile=profile1,
    )
    return (profile_observation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Access recorded data via:

    - ``profile_obs.hist_y``: Histogram amplitudes (shape: n_observations × n_bins)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### StaticMultiProfileObservation - Many Profiles

    If you want to have acess to sucessive profiles via a common observation.
    """)
    return


@app.cell
def _(profile1):
    from blond import StaticMultiProfileObservation

    multi_profile_observation = StaticMultiProfileObservation(
        each_turn_i=1,
        profiles=(profile1,),
    )
    return (multi_profile_observation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Access recorded data via:

    - ``profile_obs.hist_y``: Histogram amplitudes (shape: n_turns × n_bins)
    - ``profile_obs.hist_x``: Bin center positions [s] (shape: n_turns × n_bins)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### WakeFieldObservation - Wake Field Effects

    Observes the induced voltage from wake fields. Note that this observation is independent of the wakefield solver or impedance model used - it simply records the calculated induced voltage regardless of whether you're using resonators, impedance tables, or other wake field implementations:
    """)
    return


@app.cell
def _(wakefield1):
    from blond import WakeFieldObservation

    wake_observation = WakeFieldObservation(
        each_turn_i=1,
        wakefield=wakefield1,  # observes the resulting induced voltage
    )
    return (wake_observation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Access recorded data via:

    - ``wake_obs.induced_voltage``: Induced voltage [V] from wake fields (independent of solver type)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Common Parameters

    All observations support these common parameters:

    - ``each_turn_i``: Controls recording frequency. Set to 1 to record every turn, 10 to record every 10th turn, etc.
    - ``folder``: Optional path prefix for saving/loading data to disk.
     Some observation support also:
    - ``obs_per_turn``: For observations that support it, allows multiple recordings per turn at different RF sections. Maximum value is the number of RF stations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Contributing New Observations

    If you need to observe quantities not provided by the existing observation classes, you can implement your own by:

    1. Inheriting from ``ObservablesEndOfTurnBase`` (in ``blond.handle_results.observables``)
    2. Implementing the required methods: ``on_run_simulation()`` and ``update()``
    3. Using ``DenseArrayRecorder`` to efficiently store time-series data

    We encourage you to **contribute your new observations to the main BLonD project** via a pull request on GitHub. This helps the entire community benefit from your work and ensures your observations are maintained and tested.

    For examples, see the existing implementations in ``blond/handle_results/observables.py``.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running the Simulation

    ### Basic Execution

    Run the simulation with the defined observations:
    """)
    return


@app.cell
def _(
    beam1,
    bunch_observation,
    deepcopy,
    drift_observation,
    multi_profile_observation,
    phase_observation,
    profile_observation,
    sim,
    stats_observation,
    wake_observation,
):
    sim.run_simulation(
        beams=(deepcopy(beam1),),
        n_turns=15,
        observe=(
            bunch_observation,
            stats_observation,
            phase_observation,
            drift_observation,
            profile_observation,
            multi_profile_observation,
            wake_observation,
        ),
    )
    return


@app.cell
def _(sim, stats_observation):
    sim.save_results(observe=(stats_observation,))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Custom Callbacks

    You can define custom actions to execute during the simulation:
    """)
    return


@app.cell
def _(Beam, Simulation, beam1, deepcopy, energy_cycle, plt, ring):
    from blond import copy_to_cpu

    def custom_action(simulation: Simulation, beam: Beam):
        if simulation.turn_counter.value % 10 != 0:
            return

        plt.scatter(
            copy_to_cpu(beam.read_partial_dt()),
            copy_to_cpu(beam.read_partial_dE()),
        )
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    # Resuming a `Simulation` is not supported: a second `run_simulation` on
    # `sim` would restart the turn counter and reallocate the observations of
    # the first run. Demonstrate the callback on its own `Simulation` instead.
    sim_callbacks = Simulation(ring=ring, magnetic_cycle=energy_cycle)

    plt.ion()
    sim_callbacks.run_simulation(
        beams=(deepcopy(beam1),),
        n_turns=10,
        callbacks=custom_action,  # custom action called each turn
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loading Cached Results

    BLonD can cache simulation results. Use a try-except block to load cached results or run a new simulation:
    """)
    return


@app.cell
def _(N_TURNS, beam1, sim, stats_observation):
    sim.load_results(
        beams=(beam1,),
        n_turns=N_TURNS,
        observe=(stats_observation,),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyzing Results

    After the simulation completes, you can access the recorded observations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting Phase Evolution
    """)
    return


@app.cell
def _(plt, profile_observation):
    plt.plot(profile_observation.hist_x.T, profile_observation.hist_y.T)
    plt.xlabel("Time [s]")
    plt.ylabel("Intensity [arb. unit]")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - Explore other examples in the ``blond/examples/`` directory
    - Learn about intensity effects and collective effects
    - Understand different RF station types and RF programs
    - Experiment with different beam preparation methods
    """)
    return


if __name__ == "__main__":
    app.run()
