.. _getting_started:

Getting Started
===============

This guide walks you through setting up and running your first BLonD simulation using the EX_01_Acceleration example.

Overview
--------

A BLonD simulation consists of several key components:

1. Physical components (Ring, RF stations, Drift sections)
2. Beam definition
3. Energy/magnetic cycle
4. Beam preparation
5. Observations
6. Simulation execution

Setting Up the Simulation
--------------------------

Creating the Ring
~~~~~~~~~~~~~~~~~

The ring defines the circumference of your accelerator and
holds all elements that are executed for simulating the beam behaviour:

.. code-block:: python

    from blond import Ring

    ring = Ring(26658.883)  # circumference in meters

Defining RF Stations
~~~~~~~~~~~~~~~~~~~~

RF stations provide acceleration and longitudinal focusing. A single harmonic
RF station is defined by its harmonic number, voltage, and phase:

.. code-block:: python

    from blond import SingleHarmonicRFStation

    rf_station1 = SingleHarmonicRFStation()
    rf_station1.harmonic = 35640         # harmonic number
    rf_station1.voltage = 6e6            # voltage in V
    rf_station1.phi_rf = 0               # RF phase in radians

Configuring Drift Sections
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drift sections represent regions without RF stations:

.. code-block:: python

    from blond import DriftSimple

    drift1 = DriftSimple(orbit_length=26658.883)
    drift1.transition_gamma = 55.759505

Setting Up the Energy Cycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The magnetic cycle defines how the beam energy evolves during the simulation. BLonD provides several types of cycles.

Constant Energy
^^^^^^^^^^^^^^^

For simulations with constant beam energy (no acceleration):

.. code-block:: python

    from blond import ConstantMagneticCycle, proton

    energy_cycle = ConstantMagneticCycle(
        reference_particle=proton,
        value=450e9,             # energy value
        in_unit="total energy",  # can be 'momentum', 'total energy', 'kinetic energy', or 'bending field'
    )

This is the simplest and most efficient option when the beam energy doesn't change.

Energy Ramp (Per Turn)
^^^^^^^^^^^^^^^^^^^^^^^

For simulations with acceleration, where energy changes turn-by-turn:

.. code-block:: python

    from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
    from blond import proton
    import numpy as np

    N_TURNS = int(1e3)

    energy_cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        value_init=450e9,                                      # initial energy
        values_after_turn=np.linspace(450e9, 7000e9, N_TURNS), # energy after each turn
        in_unit="total energy",
    )

The cycle assumes each RF station provides an equal fraction of the energy kick per turn.

Advanced Cycle Options
^^^^^^^^^^^^^^^^^^^^^^^

**Per-RF-Station Control (MagneticCyclePerTurnAllRFStations)**

For full control over each RF station's contribution at each turn:

.. code-block:: python

    from blond.cycles.magnetic_cycle import MagneticCyclePerTurnAllRFStations

    # REQUIRED: 2D array with shape (n_rf_stations, n_turns)
    # Each row represents one RF station, each column represents one turn
    # Example for 4 RF stations and 1000 turns:
    n_rf_stations = 4
    N_TURNS = 1000

    energy_per_rf_station = np.zeros((n_rf_stations, N_TURNS))
    # Write the desired values in `energy_per_rf_station`
    energy_cycle = MagneticCyclePerTurnAllRFStations(
        reference_particle=proton,
        value_init=450e9,
        values_after_rf_station_per_turn=energy_per_rf_station,  # 2D array: (rf_stations, turns)
        in_unit="total energy",
    )

**Time-Based Cycle (MagneticCycleByTime)**

For cycles defined by continuous time interpolation:

.. code-block:: python

    from blond.cycles.magnetic_cycle import MagneticCycleByTime

    time_points = np.array([0, 1.0, 2.0, 3.0])  # time in seconds
    energy_values = np.array([450e9, 500e9, 600e9, 700e9])

    energy_cycle = MagneticCycleByTime(
        reference_particle=proton,
        base_time=time_points,
        base_values=energy_values,
        in_unit="total energy",
        interpolator=np.interp,  # interpolation function
    )

All cycle types support the ``in_unit`` parameter with options: ``'momentum'``, ``'total energy'``, ``'kinetic energy'``, or ``'bending field'``.

Creating the Beam
~~~~~~~~~~~~~~~~~

The beam object holds particle information:

.. code-block:: python

    from blond import Beam, proton

    beam1 = Beam(
        intensity=1e9,          # number of real particles
        particle_type=proton,   # particle type
    )

Assembling the Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Simulation.from_locals()`` method automatically discovers and connects all components defined in the local scope:

.. code-block:: python

    from blond import Simulation

    sim = Simulation.from_locals(locals())
    sim.print_one_turn_execution_order()   # optional: print execution order

The ``locals()`` function returns a dictionary of all variables in the current scope. ``Simulation.from_locals()`` inspects this dictionary and automatically detects all BLonD components (ring, RF stations, drifts, beam, energy cycle) without requiring you to pass each one explicitly.

Beam Preparation
----------------

Before running the simulation, the beam phase space must be populated with macroparticles. BLonD offers several beam preparation methods:

BiGaussian Distribution
~~~~~~~~~~~~~~~~~~~~~~~

The most commonly used method creates a simple Gaussian distribution in both time and energy:

.. code-block:: python

    from blond import BiGaussian

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=BiGaussian(
            sigma_dt=0.4e-9 / 4,       # time spread (standard deviation) in seconds
            sigma_dE=1e9 / 4,          # energy spread (standard deviation) in eV
            seed=1,                    # random seed for reproducibility
            n_macroparticles=1e3,      # number of macroparticles
        ),
    )

This creates 1000 macroparticles with a Gaussian distribution. It's simple, fast, and suitable for most applications.

Empiric Matcher (Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more advanced beam preparation that matches specific target distributions using a grid-based approach:

.. code-block:: python

    from blond.experimental.beam_preparation.empiric_matcher import EmpiricMatcher

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=EmpiricMatcher(
            grid_base_dt=np.linspace(0, 2.5e-9, 100),
            grid_base_dE=np.linspace(-(777538700.0 * 2), 777538700.0 * 2, 100),
            n_macroparticles=1e6,
            seed=0,
            maxiter_intensity_effects=0,
        ),
    )

The EmpiricMatcher populates the beam by mapping a target density distribution onto the phase space grid.

Semi-Empiric Matcher (Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A physics-based matching approach that uses the Hamiltonian to generate matched distributions:

.. code-block:: python

    from blond.experimental.beam_preparation.semi_empiric_matcher import SemiEmpiricMatcher

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=SemiEmpiricMatcher(
            hamilton_max=1e-6,               # maximum Hamiltonian value
            n_macroparticles=1e6,
            seed=0,
            density_modifier=1.0,            # shape modifier for density distribution
        ),
    )

This method calculates the Hamiltonian and generates particles following matched trajectories, making it ideal for stationary distributions.

XSuite RF Bucket Matcher
~~~~~~~~~~~~~~~~~~~~~~~~~

For users working with XSuite, BLonD provides an interface to XSuite's RF bucket matching:

.. code-block:: python

    from blond.interfaces.xsuite.beam_preparation.rfbucket_matching import XsuiteRFBucketMatcher
    from xpart.longitudinal.rfbucket_matching import QGaussianDistribution

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=XsuiteRFBucketMatcher(
            distribution_type=QGaussianDistribution,  # or ThermalDistribution, ParabolicDistribution
            sigma_z=2.5e-9 / 4,
            n_macroparticles=1e3,
            seed=42,
        ),
    )

This requires the ``xpart`` package and allows using XSuite's advanced stationary distributions.

Other Methods
~~~~~~~~~~~~~

Additional beam preparation methods are under development in the ``blond.experimental.beam_preparation`` module. Check the source code for the latest available options.

Setting Up Observations
------------------------

Observations allow you to record data during the simulation for later analysis. BLonD provides several observation classes for different purposes.

Available Observations
~~~~~~~~~~~~~~~~~~~~~~

**BeamObservationEndOfTurn** - Complete Beam Distribution

Records the full longitudinal phase space coordinates (time and energy) of all macroparticles at the end of each specified turn, allowing you to analyze beam evolution and create phase space plots.

.. code-block:: python

    from blond import BeamObservationEndOfTurn

    bunch_observation = BeamObservationEndOfTurn(
        each_turn_i=1,    # record every turn
        beam=beam1,       # which beam to observe
    )

Access recorded data via:

- ``bunch_observation.dts``: Time coordinates [s] for all particles (shape: n_turns × n_macroparticles)
- ``bunch_observation.dEs``: Energy deviations [eV] for all particles (shape: n_turns × n_macroparticles)
- ``bunch_observation.flags``: Particle status flags (e.g., lost particles)
- ``bunch_observation.reference_time``: Reference time per turn [s]
- ``bunch_observation.reference_total_energy``: Reference energy per turn [eV]

**BunchObservationMetaParams** - Statistical Parameters

Records mean and standard deviation of time and energy coordinates, plus the statistical emittance:

.. code-block:: python

    from blond.handle_results.observables import BunchObservationMetaParams

    stats_observation = BunchObservationMetaParams(
        each_turn_i=1,       # record every turn
        beam=beam1,
        obs_per_turn=1,      # observations per turn (max = n_cavities)
    )

Access recorded data via:

- ``stats_observation.mean_dt``: Mean time coordinate [s]
- ``stats_observation.mean_dE``: Mean energy deviation [eV]
- ``stats_observation.sigma_dt``: Time coordinate standard deviation [s]
- ``stats_observation.sigma_dE``: Energy deviation standard deviation [eV]
- ``stats_observation.emittance_stat``: Statistical emittance

**RFStationPhaseObservation** - RF Station Parameters

Tracks the evolution of RF station parameters (phase, frequency, voltage):

.. code-block:: python

    from blond import RFStationPhaseObservation

    phase_observation = RFStationPhaseObservation(
        each_turn_i=1,        # record every turn
        rf_station=rf_station1,   # which RF station to observe
    )

Access recorded data via:

- ``phase_observation.phases``: RF phase [rad] (shape: n_turns × n_harmonics)
- ``phase_observation.omegas``: Angular frequency [rad/s] (shape: n_turns × n_harmonics)
- ``phase_observation.voltages``: RF voltage [V] (shape: n_turns × n_harmonics)

**StaticProfileObservation** - Beam Profile (Fixed Bins)

Observes a beam profile with fixed bin positions and widths:

.. code-block:: python

    from blond.handle_results.observables import StaticProfileObservation
    from blond import StaticProfile

    profile = StaticProfile(...)
    profile_obs = StaticProfileObservation(
        each_turn_i=1,
        profile=profile,
        obs_per_turn=1,      # can observe at multiple locations per turn
    )

Access recorded data via:

- ``profile_obs.hist_y``: Histogram amplitudes (shape: n_observations × n_bins)

**DynamicProfileConstNBinsObservation** - Beam Profile (Adaptive Bins)

Observes a beam profile where bin width adapt to the beam distribution but the number of bins stays constant:

.. code-block:: python

    from blond.handle_results.observables import DynamicProfileConstNBinsObservation
    from blond import DynamicProfileConstNBins

    profile = DynamicProfileConstNBins(beam=beam1, n_bins=256)
    profile_obs = DynamicProfileConstNBinsObservation(
        each_turn_i=1,
        profile=profile,
    )

Access recorded data via:

- ``profile_obs.hist_y``: Histogram amplitudes (shape: n_turns × n_bins)
- ``profile_obs.hist_x``: Bin center positions [s] (shape: n_turns × n_bins)

**WakeFieldObservation** - Wake Field Effects

Observes the induced voltage from wake fields. Note that this observation is independent of the wakefield solver or impedance model used - it simply records the calculated induced voltage regardless of whether you're using resonators, impedance tables, or other wake field implementations:

.. code-block:: python

    from blond.handle_results.observables import WakeFieldObservation
    from blond import WakeField

    # The WakeField can be configured with any solver/impedance model
    wakefield = WakeField(...)  # e.g., resonators, impedance tables, etc.

    wake_obs = WakeFieldObservation(
        each_turn_i=1,
        wakefield=wakefield,  # observes the resulting induced voltage
        obs_per_turn=1,
    )

Access recorded data via:

- ``wake_obs.induced_voltage``: Induced voltage [V] from wake fields (independent of solver type)

Common Parameters
~~~~~~~~~~~~~~~~~

All observations support these common parameters:

- ``each_turn_i``: Controls recording frequency. Set to 1 to record every turn, 10 to record every 10th turn, etc.
- ``folder``: Optional path prefix for saving/loading data to disk.
 Some observation support also:
- ``obs_per_turn``: For observations that support it, allows multiple recordings per turn at different RF sections. Maximum value is the number of RF stations.

Contributing New Observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to observe quantities not provided by the existing observation classes, you can implement your own by:

1. Inheriting from ``ObservablesEndOfTurnBase`` (in ``blond.handle_results.observables``)
2. Implementing the required methods: ``on_run_simulation()`` and ``update()``
3. Using ``DenseArrayRecorder`` to efficiently store time-series data

We encourage you to **contribute your new observations to the main BLonD project** via a pull request on GitHub. This helps the entire community benefit from your work and ensures your observations are maintained and tested.

For examples, see the existing implementations in ``blond/handle_results/observables.py``.

Running the Simulation
----------------------

Basic Execution
~~~~~~~~~~~~~~~

Run the simulation with the defined observations:

.. code-block:: python

    sim.run_simulation(
        beams=(beam1,),

        n_turns=N_TURNS,
        observe=(phase_observation, bunch_observation),
    )

Loading Cached Results
~~~~~~~~~~~~~~~~~~~~~~~

BLonD can cache simulation results. Use a try-except block to load cached results or run a new simulation:

.. code-block:: python

    try:
        sim.load_results(
            beams=(beam1,),

            n_turns=N_TURNS,
            observe=(phase_observation, bunch_observation),
        )
        print(f"Loaded {phase_observation.common_name}")
    except (FileNotFoundError, AssertionError):
        sim.run_simulation(
            beams=(beam1,),

            n_turns=N_TURNS,
            observe=(phase_observation, bunch_observation),
        )

Custom Callbacks
~~~~~~~~~~~~~~~~

You can define custom actions to execute during the simulation:

.. code-block:: python

    from matplotlib import pyplot as plt

    def custom_action(simulation: Simulation, beam: Beam):
        if simulation.turn_i.value % 10 != 0:
            return

        plt.scatter(
            beam.read_partial_dt(),
            beam.read_partial_dE(),
        )
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    sim.run_simulation(
        beams=(beam1,),

        n_turns=N_TURNS,
        observe=(phase_observation, bunch_observation),
        callback=custom_action,  # custom action called each turn
    )

Analyzing Results
-----------------

After the simulation completes, you can access the recorded observations.

Plotting Phase Evolution
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from matplotlib import pyplot as plt

    plt.plot(phase_observation.phases)
    plt.xlabel('Turn number')
    plt.ylabel('RF Phase [rad]')
    plt.show()

Animating Beam Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    for i in range(N_TURNS):
        plt.clf()
        plt.hist2d(
            bunch_observation.dts[i, :],   # time coordinates
            bunch_observation.dEs[i, :],   # energy coordinates
            bins=256,
            range=[[0, 2.5e-9], [-4e8, 4e8]],
        )
        plt.xlabel('Time [s]')
        plt.ylabel('Energy deviation [eV]')
        plt.draw()
        plt.pause(0.1)

    plt.show()

Complete Example
----------------

Here is the complete example putting it all together:

.. code-block:: python

    import numpy as np
    from blond import (
        Beam,
        BeamObservationEndOfTurn,
        BiGaussian,
        RFStationPhaseObservation,
        ConstantMagneticCycle,
        DriftSimple,
        Ring,
        Simulation,
        SingleHarmonicRFStation,
        proton,
    )

    # Setup components
    ring = Ring(26658.883)

    # Define RF station (SingleHarmonicCavity will be renamed to
    SingleHarmonicRFStation)
    rf_station1 = SingleHarmonicRFStation()
    rf_station1.harmonic = 35640
    rf_station1.voltage = 6e6
    rf_station1.phi_rf = 0

    N_TURNS = int(1e3)

    energy_cycle = ConstantMagneticCycle(
        reference_particle=proton,
        value=450e9,
        in_unit="total energy",
    )

    drift1 = DriftSimple(orbit_length=26658.883)
    drift1.transition_gamma = 55.759505

    beam1 = Beam(intensity=1e9, particle_type=proton)

    # Create simulation
    sim = Simulation.from_locals(locals())

    # Prepare beam
    sim.prepare_beam(
        beam=beam1,
        preparation_routine=BiGaussian(
            sigma_dt=0.4e-9 / 4,
            sigma_dE=1e9 / 4,
            seed=1,
            n_macroparticles=1e3,
        ),
    )

    # Setup observations
    phase_observation = RFStationPhaseObservation(each_turn_i=1,
    rf_station=rf_station1)
    bunch_observation = BeamObservationEndOfTurn(each_turn_i=1, beam=beam1)

    # Run simulation
    try:
        sim.load_results(
            beams=(beam1,),

            n_turns=N_TURNS,
            observe=(phase_observation, bunch_observation),
        )
    except (FileNotFoundError, AssertionError):
        sim.run_simulation(
            beams=(beam1,),

            n_turns=N_TURNS,
            observe=(phase_observation, bunch_observation),
        )

Next Steps
----------

- Explore other examples in the ``blond/examples/`` directory
- Learn about intensity effects and collective effects
- Understand different RF station types and RF programs
- Experiment with different beam preparation methods
