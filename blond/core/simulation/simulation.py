# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Holds the `Simulation` class.

Notes
-----
Authors:
S. Lauber
L. Thiele
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from pstats import SortKey
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_simpson  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore

from blond.core.backends.backend import backend
from blond.core.base import (
    DynamicParameter,
    Preparable,
    SimulationElementBase,
)
from blond.core.helpers import (
    find_instances_with_method,
    int_from_float_with_warning,
)
from blond.core.ring.helpers import get_elements, get_required_order
from blond.cycles.magnetic_cycle import MagneticCycleBase
from blond.generals.warnings_ import NotTestedWarning, PerformanceWarning

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond import BiGaussian
    from blond.beam_preparation.base import BeamPreparationRoutine
    from blond.core.beam.base import BeamBaseClass
    from blond.core.beam.particle_types import ParticleType
    from blond.core.ring.ring import Ring
    from blond.experimental.beam_preparation.empiric_matcher import (
        EmpiricMatcher,
    )
    from blond.experimental.beam_preparation.semi_empiric_matcher import (
        SemiEmpiricMatcher,
    )
    from blond.handle_results.observables import ObservablesOncePerTurnBase
    from blond.interfaces.xsuite.beam_preparation.rfbucket_matching import (
        XsuiteRFBucketMatcher,
    )


logger = logging.getLogger(__name__)


def _single_beam_to_tuple(
    maybe_beams: BeamBaseClass | tuple[BeamBaseClass, ...],
) -> tuple[BeamBaseClass, ...]:
    """
    Guarantee that the result is a tuple of beams.

    Parameters
    ----------
    maybe_beams
        Single beam instance or multiple beams.

    Returns
    -------
    beams
        Tuple of at leat one beam.
    """
    return maybe_beams if isinstance(maybe_beams, Sequence) else (maybe_beams,)


class Simulation(Preparable):
    """
    Main simulation manager for beam dynamics in synchrotrons.

    The ``Simulation`` class is the central object that coordinates all aspects of a
    beam dynamics simulation. It manages the ring structure, energy evolution, beam
    tracking through RF stations and drift sections, and data collection via observables.

    Typical workflow:
        1. Create a ``Simulation`` using ``Simulation.from_locals()`` to auto-detect components
        2. Prepare the beam with ``sim.prepare_beam()`` to populate the phase space
        3. Run the simulation with ``sim.run_simulation()`` to track the beam evolution
        4. Analyze results from observables or save/load them for later use

    Parameters
    ----------
    ring
        The ``Ring`` object representing the synchrotron, which contains all physical
        elements (RF stations, drifts, etc.) that the beam encounters each turn.
    magnetic_cycle
        Defines how the beam energy evolves during the simulation. This can be a
        constant energy, a turn-by-turn energy ramp, or a time-based interpolation.

    Attributes
    ----------
    turn_i
        Counter tracking the current turn number during simulation. Can be subscribed
        to for notifications when the turn changes. Value is ``None`` when not running.
    section_i
        Counter tracking the current section (element) within a turn. Value is ``None``
        when not running.
    _ring
        The synchrotron ring (read-only property).
    _magnetic_cycle
        The energy evolution program (read-only property).

    See Also
    --------
    from_locals : Automatically create a Simulation from defined components.
    prepare_beam : Populate the beam phase space with macroparticles.
    run_simulation : Execute the beam dynamics tracking.

    Examples
    --------
    Create a simple simulation automatically from local variables:

    >>> from blond import Ring, ConstantMagneticCycle, SingleHarmonicRfStation, DriftSimple, proton
    >>> ring = Ring(26658.883)
    >>> energy_cycle = ConstantMagneticCycle(proton, value=450e9, in_unit="total energy")
    >>> rf_station1 = SingleHarmonicRfStation()
    >>> drift1 = DriftSimple(orbit_length=26658.883)
    >>> sim = Simulation.from_locals(locals())
    """

    def __init__(
        self,
        ring: Ring,
        magnetic_cycle: MagneticCycleBase,
    ) -> None:
        assert ring.elements.n_elements > 0, f"{ring.elements.n_elements=}"

        from blond.core.simulation.intensity_effect_manager import (
            IntensityEffectManager,
        )

        super().__init__()
        self._ring: Ring = ring

        self._magnetic_cycle: MagneticCycleBase = magnetic_cycle

        self.turn_i = DynamicParameter(None)
        self.section_i = DynamicParameter(None)
        self.intensity_effect_manager = IntensityEffectManager(simulation=self)

        self._exec_on_init_simulation()

        self._particle_performance_waning_threshold = int(1e3)

    def profiling(
        self,
        beams: tuple[BeamBaseClass],
        profile_start_turn_i: int,
        profile_n_turns: int,
        sortby: SortKey = SortKey.CUMULATIVE,
    ) -> None:
        """
        Profile the simulation to identify performance bottlenecks.

        Runs the Python profiler during simulation execution to measure where time
        is being spent. This is useful for optimizing slow simulations by identifying
        which functions or elements are consuming the most CPU time.

        The profiler can start after a few turns to exclude initialization overhead
        and focus on the main simulation loop performance.

        Parameters
        ----------
        beams
            Beams to simulate during profiling (typically just one).
        profile_start_turn_i
            Turn number at which to begin profiling.
        profile_n_turns
            Number of turns to profile after starting.
        sortby
            How to sort the profiling results. Options include:
                - SortKey.CUMULATIVE: Sort by cumulative time (default, most useful)
                - SortKey.TIME: Sort by internal time
                - SortKey.CALLS: Sort by call count

        See Also
        --------
        run_simulation : Execute the beam dynamics tracking.

        Notes
        -----
        - Profiling adds overhead and makes the simulation slower.
        - Use this for development and optimization, not for production runs.
        - Results show function call counts and time spent in each function.

        Examples
        --------
        Profile 100 turns after a 10-turn warmup:
        >>> from blond import Simulation, Beam
        >>> sim = Simulation(...)
        >>> beam1 = Beam(...)
        >>> from pstats import SortKey
        >>> sim.profiling(
        ...     beams=(beam1,),
        ...
        ...     profile_start_turn_i=10,  # Skip first 10 turns
        ...     profile_n_turns=100,       # Profile next 100 turns
        ...     sortby=SortKey.CUMULATIVE,
        ... )
        # Prints detailed timing statistics
        """
        assert profile_start_turn_i >= 0

        import cProfile
        import io
        import pstats

        pr = cProfile.Profile()

        # trigger profiling later than turn 0
        def start_profiling(simulation: Simulation, beam: BeamBaseClass):
            """
            Enable the profiling in the mainloop.

            Parameters
            ----------
            simulation
                `Simulation` context manager.
            beam
                The `Beam` object.
            """
            if simulation.turn_i.value == profile_start_turn_i:
                pr.enable()

        end_turn = profile_start_turn_i + profile_n_turns
        self.run_simulation(
            beams=beams,
            n_turns=end_turn,
            show_progressbar=False,
            callback=start_profiling,
        )

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def plot_potential_well_empiric(
        self,
        dt: NumpyArray,
        particle_type: ParticleType,
        subtract_min: bool = True,
        **kwargs_plot,
    ) -> None:
        """
        Plot the RF potential well by tracking particles through one turn.

        This is a convenience method that calculates and immediately plots the RF
        potential well. It internally calls ``get_potential_well_empiric()`` and
        creates a matplotlib plot of the result.

        The potential well shows the effective RF "landscape" that determines stable
        and unstable regions for particles in longitudinal phase space.

        Parameters
        ----------
        dt
            Array of time coordinates at which to sample the potential, in [s].
            Usually spans at least one RF bucket.
        particle_type
            Type of particle to track (e.g., proton, electron). The particle charge
            affects the phase advance and thus the potential shape.
        subtract_min
            If True (default), normalizes the potential so its minimum is at zero.
            If False, normalizes so ``potential_well[0] = 0``.
        **kwargs_plot
            Additional keyword arguments passed to ``matplotlib.pyplot.plot()``
            for customizing the plot appearance (e.g., color='red', linewidth=2).

        See Also
        --------
        get_potential_well_empiric : Calculate RF potential well.

        Notes
        -----
        - This method creates the plot but does not call ``plt.show()``. You must
          call that separately to display the plot.
        - With multiple RF stations and drifts, the potential may show distortions
          due to phase advances between stations.

        Examples
        --------
        Plot the potential well with default settings:

        >>> import numpy as np
        >>> from blond import Simulation, Beam, proton
        >>> sim = Simulation(...)
        >>> beam1 = Beam(...)
        >>> dt_array = np.linspace(0, 2.5e-9, 500)
        >>> plt.title('RF Bucket Structure')
        >>> sim.plot_potential_well_empiric(dt=dt_array, particle_type=proton)
        >>> plt.show()

        Customize the plot appearance:

        >>> sim.plot_potential_well_empiric(
        ...     dt=dt_array,
        ...     particle_type=proton,
        ...     color='red',
        ...     linewidth=2,
        ...     label='Potential'
        ... )
        >>> plt.legend()
        >>> plt.show()
        """
        potential_well, _, _ = self.get_potential_well_empiric(
            dt=dt,
            particle_type=particle_type,
            subtract_min=subtract_min,
        )
        plt.plot(dt, potential_well, **kwargs_plot)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (arb. unit)")

    def get_drift_term_empiric(
        self,
        dE: NumpyArray,
        particle_type: ParticleType,
        intensity: int = 0,
    ) -> NumpyArray:
        """
        Calculate how particles drift in time as a function of energy offset.

        This method empirically determines the time drift (chromaticity effect) by
        tracking probe particles with different energy offsets through one complete turn.
        Particles with different energies travel at different velocities and orbits,
        causing them to arrive at different times.

        This is the complementary measurement to ``get_potential_well_empiric()``:
            - ``get_potential_well_empiric``: starts with time offsets, measures energy changes
            - ``get_drift_term_empiric``: starts with energy offsets, measures time changes

        Parameters
        ----------
        dE
            Array of energy offsets to probe, in [eV]. Typically spans from negative
            to positive values around the synchronous energy.
        particle_type
            Type of particle (e.g., proton, electron). Different particles have
            different velocity vs. energy relationships.
        intensity
            Beam intensity (number of real particles) to include collective effects.

        Returns
        -------
        drift_term
            The time drift accumulated in one turn for each energy offset, in [s].
            Shows how arrival time varies with energy.

        See Also
        --------
        get_potential_well_empiric : Calculate RF potential well.

        Notes
        -----
        - Higher energy particles typically travel a longer path
          (above transition) or shorter path (below transition), causing
          time shifts.
        - This is related to the slip factor (eta).
        - The method creates a temporary probe beam and tracks it for one turn.

        Examples
        --------
        Calculate and plot the chromatic drift:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from blond import Simulation, proton
        >>> sim = Simulation(...)
        >>>
        >>> dE_array = np.linspace(-1e8, 1e8, 500)  # Energy offsets in eV
        >>> drift = sim.get_drift_term_empiric(dE=dE_array, particle_type=proton)
        >>>
        >>> plt.plot(dE_array, drift)
        >>> plt.xlabel('Energy offset [eV]')
        >>> plt.ylabel('Time drift [s]')
        >>> plt.title('Chromatic Effect')
        >>> plt.show()
        """
        from blond.core.beam.beams import ProbeBeam

        probe_bunch = ProbeBeam(
            dE=dE,
            particle_type=particle_type,
            intensity=intensity,
        )
        t0 = probe_bunch.reference_time
        self.run_simulation(
            beams=(probe_bunch,),
            n_turns=1,
            show_progressbar=False,
        )
        t1 = probe_bunch.reference_time
        T = t1 - t0
        potential_well = (
            cumulative_simpson(probe_bunch.read_partial_dt(), x=dE, initial=0)
            / T
        )
        potential_well -= potential_well.min()
        return potential_well

    def get_potential_well_empiric(
        self,
        dt: NumpyArray,
        particle_type: ParticleType,
        subtract_min: bool = True,
        intensity: int = 0,
    ) -> tuple[NumpyArray, float, float]:
        """
        Calculate the RF potential well by tracking particles through one turn.

        The potential well represents the effective RF voltage "landscape" that particles
        experience as they move through the synchrotron. This method computes it empirically
        by tracking probe particles at different time offsets through one complete turn and
        measuring their energy changes.

        This is useful for:
            - Visualizing stable and unstable regions for particles
            - Understanding bucket shapes and separatrices
            - Verifying RF configurations
            - Analyzing beam matching

        The method accounts for realistic effects like phase advances between multiple
        RF stations and drift sections.

        Parameters
        ----------
        dt
            Array of time coordinates at which to sample the potential well, in [s].
            Typically spans one or more RF buckets.
        particle_type
            Type of particle (e.g., proton, electron). The particle charge affects
            phase advance and the potential shape.
        subtract_min
            If True (default), shifts the potential so its minimum is at zero.
            If False, shifts it so potential_well[0] = 0.
        intensity
            Beam intensity (number of real particles) to include collective effects.
            Default is 0 (no intensity effects).

        Returns
        -------
        potential_well
            The effective potential energy at each time coordinate, in arbitrary units
            (proportional to energy).
        factor
            Ratio of the sampled time span to the revolution period:
            ``(dt[-1] - dt[0]) / t_rev``. Indicates how many buckets were sampled.
        tilt_dt_per_dE
            Phase space tilt coefficient [s/eV], showing how time coordinates change
            with energy even at dE=0 due to chromatic effects.

        See Also
        --------
        plot_potential_well_empiric : Plot RF potential well.
        get_drift_term_empiric : Calculate time drift vs energy.

        Notes
        -----
        - This creates a temporary probe beam internally and runs it for one turn.
        - With multiple RF stations and drifts, the effective potential can appear
          distorted because particles drift in time between stations.
        - The calculation is accurate but slower than analytical approximations.

        Examples
        --------
        Calculate and plot the potential well:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>>
        >>> from blond import Simulation, Beam, proton
        >>>
        >>> sim = Simulation(...)
        >>> beam1 = Beam(...)
        >>> dt_array = np.linspace(0, 2.5e-9, 500)
        >>> potential, factor, tilt = sim.get_potential_well_empiric(
        ...     dt=dt_array,
        ...     particle_type=proton,
        ... )
        >>>
        >>> plt.plot(dt_array, potential)
        >>> plt.xlabel('Time [s]')
        >>> plt.ylabel('Potential [arb. units]')
        >>> plt.title('RF Potential Well')
        >>> plt.show()
        """
        from blond.core.beam.beams import ProbeBeam  # prevent circular import

        probe_bunch = ProbeBeam(
            dt=dt,
            particle_type=particle_type,
            intensity=intensity,
        )
        bunch_before = deepcopy(probe_bunch)
        t_0 = probe_bunch.reference_time
        deepcopy(self).run_simulation(
            beams=(probe_bunch,),
            n_turns=1,
            show_progressbar=False,
        )
        # Calculate passed time
        t_1 = probe_bunch.reference_time
        t_rev = t_1 - t_0
        # Calculate scaling factor
        factor = float((dt[-1] - dt[0]) / t_rev)

        # Calculate tilt of phase space
        change_t = probe_bunch._dt - bunch_before._dt
        change_E = probe_bunch._dE - bunch_before._dE
        idx = np.argmax(change_t)
        tilt_dt_per_dE = change_t[idx] / change_E[idx]

        # Derive potential well by integrating over energy change
        potential_well = -cumulative_simpson(
            probe_bunch.read_partial_dE()
            if backend.specials_mode != "cuda"
            else probe_bunch.read_partial_dE().get(),
            initial=0,
        ) / len(dt)

        if self.ring.is_below_transition(beam=probe_bunch):
            # peaks and troughs are flipped when below transition
            potential_well *= -1

        if subtract_min:
            # Align potential so that the visible minimum is 0
            potential_well -= potential_well.min()
        return (
            backend.array(potential_well, dtype=backend.float),
            factor,
            tilt_dt_per_dE,
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Hook called when the Simulation object is initialized.

        This is used by simulation elements to perform
        initialization tasks. It is called automatically
        when ``Simulation.__init__()`` is executed.

        All objects in the simulation hierarchy that have an ``on_init_simulation``
        method will have it called in a specific dependency order.

        Parameters
        ----------
        simulation
            The simulation instance being initialized (usually ``self``).

        See Also
        --------
        on_run_simulation : Hook called when run_simulation is invoked.

        Notes
        -----
        - This is called once when the Simulation is created, before any beams are prepared.
        - Subclasses and simulation elements can override this to set up initial state.
        - The base implementation does nothing.
        """
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Hook called when ``run_simulation`` is invoked.

        This is a lifecycle hook that can be overridden by subclasses or used by
        simulation elements to perform setup tasks before the main simulation loop
        begins. It is called automatically by ``finalize()``.

        All objects in the simulation hierarchy that have an ``on_run_simulation``
        method will have it called in a specific dependency order.

        Parameters
        ----------
        simulation
            The simulation instance (usually ``self``).
        beam
            The first beam that will be tracked (primary beam in multi-beam scenarios).
        n_turns
            Number of turns that will be simulated.
        **kwargs
            Additional keyword arguments for extendability.

        See Also
        --------
        on_init_simulation : Hook called when Simulation is initialized.
        finalize : Finalize setup before running simulation.

        Notes
        -----
        - This is called before each ``run_simulation()`` or ``load_results()`` call.
        - Useful for pre-allocating arrays, resetting state, or computing derived parameters.
        - The base implementation does nothing.
        """
        pass

    def _exec_all_in_tree(self, method: str, **kwargs) -> None:
        """
        Execute all methods that are somewhere in the attribute hierarchy of `Simulation`.

        Parameters
        ----------
        method
            Method name to execute everywhere.
        **kwargs
            Extra keyword arguments.
        """
        logger.debug(f"Calling all {method}({kwargs}) in {self}")
        instances = find_instances_with_method(self, f"{method}")
        logger.debug(f"Found {instances} to be initialized")
        ordered_classes = get_required_order(instances, f"{method}.requires")

        classes_check = set()
        for ins in instances:
            classes_check.add(type(ins))

        logger.info(f"Execution order for `{method}` is {ordered_classes}")

        for cls in ordered_classes:
            for element in instances:
                if type(element).__name__ != cls:
                    continue
                logger.info(f"Running `{method}` of {element}")
                getattr(element, method)(**kwargs)

    def _exec_on_init_simulation(self) -> None:
        """Execute all `on_init_simulation` in the attribute hierarchy of `Simulation`."""
        self._exec_all_in_tree("on_init_simulation", simulation=self)

    def _exec_on_run_simulation(
        self,
        beam: BeamBaseClass,
        n_turns: int,
    ) -> None:
        """
        Execute all `on_run_simulation` in the attribute hierarchy of `Simulation`.

        Parameters
        ----------
        beam
            The beam object to simulate.
        n_turns
            Number of turns to simulate.
        """
        self._exec_all_in_tree(
            "on_run_simulation",
            simulation=self,
            beam=beam,
            n_turns=n_turns,
        )

    @staticmethod
    def from_locals(
        locals: dict[str, Any], verbose: bool = False
    ) -> Simulation:
        """
        Automatically create a Simulation by discovering components in the current scope.

        It automatically detects and assembles all simulation components (
        Ring, RF stations, drifts, magnetic cycle, beams) from the local
        variables where it's called. This eliminates the need to
        manually pass each component to the ``Simulation`` constructor.

        The method searches for:
            - Exactly one ``Ring`` object (required)
            - Exactly one ``MagneticCycleBase`` object (required)
            - All``SimulationElementBase`` objects like RF stations, drifts, etc.
            - Any number of ``Beam`` objects

        All found elements are automatically added to the ring in the correct execution order.

        Parameters
        ----------
        locals
            The local variable dictionary, typically obtained by calling ``locals()``.
            This should contain all the simulation components you've defined.
        verbose
            If True, prints detailed information about discovered components and their
            execution order. Useful for debugging. Default is False.

        Returns
        -------
        simulation
            A fully initialized ``Simulation`` object with all components properly
            connected and ordered.

        Raises
        ------
        AssertionError
            If exactly one Ring is not found, or if exactly one magnetic cycle is not found.

        See Also
        --------
        Simulation.__init__ : Initialize Simulation manually.
        prepare_beam : Populate beam phase space.

        Notes
        -----
        - The execution order of elements within the ring is automatically determined based
          on their dependencies and types.
        - All RF stations and drifts must be defined before calling this method.
        - The beam does not need to be prepared yet - use ``sim.prepare_beam()`` after
          creating the simulation.

        Examples
        --------
        Basic usage with automatic component discovery:

        >>> from blond import *
        >>> ring = Ring(26658.883)
        >>> energy_cycle = ConstantMagneticCycle(proton, value=450e9, in_unit="total energy")
        >>> rf_station1 = SingleHarmonicRfStation()
        >>> rf_station1.harmonic = 35640
        >>> rf_station1.voltage = 6e6
        >>> drift1 = DriftSimple(orbit_length=26658.883)
        >>> beam1 = Beam(intensity=1e9, particle_type=proton)
        >>>
        >>> # Automatically discover and assemble everything:
        >>> sim = Simulation.from_locals(locals())

        With verbose output to see what was discovered:

        >>> sim = Simulation.from_locals(locals(), verbose=True)
        # Prints: Found locals: dict_keys([...])
        # Prints: Execution order information
        """
        from blond.core.beam.base import BeamBaseClass  # prevent cyclic import
        from blond.core.ring.ring import Ring  # prevent cyclic import

        locals_list = locals.values()
        msg1 = f"Found locals: {locals.keys()}"
        logger.debug(msg=msg1)
        if verbose:
            print(msg1)
        _rings = get_elements(locals_list, Ring)
        assert len(_rings) == 1, f"Found {len(_rings)} rings"
        ring = _rings[0]

        beams = get_elements(locals_list, BeamBaseClass)  # type: ignore

        _magnetic_cycle = get_elements(locals_list, MagneticCycleBase)  # type: ignore
        assert len(_magnetic_cycle) == 1, (
            f"Found {len(_magnetic_cycle)} energy cycles"
        )
        magnetic_cycle = _magnetic_cycle[0]

        elements = get_elements(locals_list, SimulationElementBase)
        ring.add_elements(elements=elements, reorder=True)

        logger.debug(f"{ring=}")
        logger.debug(f"{beams=}")
        logger.debug(f"{elements=}")

        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
        order_info = sim.ring.elements.get_order_info()
        logger.info(order_info)
        if verbose:
            print(order_info)
        return sim

    @property  # as readonly attributes
    def ring(self) -> Ring:
        """
        The synchrotron ring containing all simulation elements.

        Returns the ``Ring`` object that holds all physical elements (RF stations,
        drifts, impedances, etc.) through which the beam is tracked each turn.

        Returns
        -------
        Ring
            The ring object, which contains the element list and execution order.
        """
        return self._ring

    @property  # as readonly attributes
    def magnetic_cycle(self) -> MagneticCycleBase:
        """
        The energy evolution program for the synchrotron.

        Returns the ``MagneticCycleBase`` object that defines how the beam's reference
        energy changes during the simulation (constant, ramping, or time-dependent).

        Returns
        -------
        cycle
            The magnetic cycle object defining energy evolution.

        See Also
        --------
        ConstantMagneticCycle : Constant energy cycle.
        MagneticCyclePerTurn : Energy cycle defined per turn.
        MagneticCyclePerTurnAllRfStations : Energy cycle with all RF stations.
        MagneticCycleByTime : Time-based energy cycle.
        """
        return self._magnetic_cycle

    def print_one_turn_execution_order(self) -> None:
        """
        Display the order in which elements are executed during each turn.

        This prints a human-readable list showing the sequence of elements (RF stations,
        drifts, etc.) that the beam encounters during one complete turn through the ring.
        This is useful for verifying that elements are in the expected order and for
        understanding the simulation flow.

        The output includes:
            - Element type (e.g., SingleHarmonicRfStation, DriftSimple)
            - Element name or identifier
            - Section index for each element

        See Also
        --------
        from_locals : Automatically create Simulation from components.

        Notes
        -----
        This should be called after creating the simulation with ``from_locals()``
        but before running it, to verify the setup is correct.
        """
        self._ring.elements.print_order()

    def prepare_beam(
        self,
        beam: BeamBaseClass,
        preparation_routine: (
            BiGaussian
            | EmpiricMatcher
            | SemiEmpiricMatcher
            | XsuiteRFBucketMatcher
            | BeamPreparationRoutine
        ),
        turn_i: int = 0,
    ) -> None:
        """
        Populate the beam's with macroparticles.

        Before running a simulation, the beam must be "prepared" by populating its
        longitudinal particle distribution (time and energy coordinates) with
        macroparticles.
        This method applies a beam preparation routine to generate the initial
        distribution of particles.

        Common preparation routines include:
            - ``BiGaussian``: Simple Gaussian distribution in time and energy
            - ``EmpiricMatcher``: Grid-based distribution matching
            - ``SemiEmpiricMatcher``: Hamiltonian-based matched distribution
            - ``XsuiteRFBucketMatcher``: Interface to XSuite's RF bucket matching

        Parameters
        ----------
        beam
            The ``Beam`` object to populate with macroparticles. This beam should
            have already been defined with its intensity and particle type.
        preparation_routine
            The algorithm that generates the initial particle distribution. This
            determines the shape and characteristics of the beam in longitudinal
            phase space (dt, dE coordinates).
        turn_i
            The turn number at which to prepare the beam. Usually 0 (the default)
            for initial beam setup.

        See Also
        --------
        BiGaussian : Simple Gaussian beam distribution.
        EmpiricMatcher : Grid-based distribution matching.
        SemiEmpiricMatcher : Hamiltonian-based matched distribution.
        XsuiteRFBucketMatcher : XSuite RF bucket matching.

        Notes
        -----
        - This method should be called after creating the ``Simulation`` but
          before ``run_simulation()``.
        - The beam preparation uses the current simulation state (RF programs, energy,
          etc.) to calculate the appropriate phase space distribution.
        - Different preparation routines produce different beam characteristics and may
          be more or less matched to the RF bucket.

        Examples
        --------
        Prepare a beam with a simple Gaussian distribution:

        >>> from blond import Simulation, Beam, BiGaussian
        >>> sim = Simulation(...)
        >>> beam1 = Beam(...)
        >>> sim.prepare_beam(
        ...     beam=beam1,
        ...     preparation_routine=BiGaussian(
        ...         sigma_dt=0.4e-9 / 4,    # time spread in seconds
        ...         sigma_dE=1e9 / 4,       # energy spread in eV
        ...         n_macroparticles=1e3,
        ...         seed=1,
        ...     ),
        ... )

        Use a more advanced Hamiltonian-based matcher:
        >>> from blond.experimental.beam_preparation.semi_empiric_matcher import SemiEmpiricMatcher
        >>> # Define matching routine with typical parameters
        >>> matcher = SemiEmpiricMatcher(
        ...     time_limit=(-2e-9, 2e-9),              # Time window around bunch [s]
        ...     n_macroparticles=100_000,              # Number of macro-particles
        ...     hamilton_to_density_kwargs=dict(
        ...         density_modifier=2.0,              # Controls density profile sharpness
        ...         hamilton_max=1.0                   # Hamiltonian cutoff [eV]
        ...     ),
        ...     internal_grid_shape=(1023, 1023),      # Resolution of phase space grid
        ...     tolerance=1e-6,                        # Convergence threshold
        ...     maxiter_intensity_effects=100,         # Max iterations with wakefields
        ...     increment_intensity_effects_until_iteration_i=10,  # Intensity ramp-up steps
        ...     seed=42,                               # For reproducibility
        ...     verbose=True,                          # Print convergence info
        ...     animate=False,                         # Set True for live plotting
        ... )
        >>> # Perform beam matching
        >>> matcher.prepare_beam(sim, beam)
        """
        logger.info("Running `prepare_beam`")
        self.turn_i.value = turn_i
        preparation_routine.prepare_beam(simulation=self, beam=beam)

    def run_simulation(
        self,
        beams: BeamBaseClass | tuple[BeamBaseClass, ...],
        n_turns: int | None = None,
        observe: tuple[ObservablesOncePerTurnBase, ...] = (),
        show_progressbar: bool = True,
        callback: Callable[[Simulation, BeamBaseClass], None] | None = None,
        callback_each_turn: int = 1,
    ) -> None:
        """
        Execute the main beam dynamics simulation loop.

        This is the core method that tracks the beam(s) through the synchrotron for
        a specified number of turns. Each turn, the beam passes through all elements
        in the ring (RF stations, drifts, etc.) in order. Observables can record data
        during the simulation, and an optional callback can be used for custom actions.

        The simulation loop:
            1. For each turn up to ``n_turns``
            2. Track the beam through each element in the ring
            3. Update observables after each drift section
            4. Call the optional callback function
            5. Update the progress bar if enabled

        Parameters
        ----------
        beams
            Tuple of beam(s) to simulate. For single beam simulations, use ``(beam1,)``.
            For counter-rotating beam simulations, use ``(beam_corotating, beam_counter)``.
            The first beam must be co-rotating, the second counter-rotating.
        n_turns
            Number of turns to simulate. If None, uses the maximum number defined by
            the magnetic cycle (only valid for cycles with a defined endpoint).
            Default is None.
        observe
            Tuple of observable objects that record data during the simulation
            (e.g., ``RfStationPhaseObservation``, ``BeamObservationEndOfTurn``).
            Each observable is updated according to its own schedule. Default is empty tuple.
        show_progressbar
            If True, displays a progress bar showing simulation progress and turn rate.
            Useful for long simulations. Default is True.
        callback
            Optional user-defined function called at the end of each turn. Signature
            must be ``def callback(simulation: Simulation, beam: BeamBaseClass): ...``.
            Useful for custom data collection or live plotting. Default is None.
        callback_each_turn
            Specifies the the repitition rate at which `callback` is called.
            Deafault is 1, each turn.

        Raises
        ------
        ValueError
            If ``n_turns`` is None and the magnetic cycle has unlimited turns.
        NotImplementedError
            If more than two beams are provided (currently unsupported).

        See Also
        --------
        prepare_beam : Populate beam with macroparticles.
        setup_beam : Manually set beam coordinates.
        save_results : Save simulation results to disk.
        load_results : Load previously saved results.
        print_one_turn_execution_order : Display element execution order.

        Notes
        -----
        - The beam must be prepared with ``prepare_beam()`` or
        ``beam.setup_beam()`` before calling this method.
        - Observables are updated after each drift section, not after every element.
        - The progress bar shows turns per second, which helps estimate total runtime.
        - For counter-rotating beams, elements are traversed in opposite order for
          the counter-rotating beam.

        Examples
        --------
        Basic simulation for 1000 turns:

        >>> from blond import Simulation, Beam, BiGaussian
        >>> sim = Simulation(...)
        >>> beam1 = Beam(...)
        >>> sim.run_simulation(
        ...     beams=(beam1,),
        ...     n_turns=1000,
        ... )

        Simulation with observables to record data:

        >>> from blond import RfStationPhaseObservation, BeamObservationOncePerTurn
        >>> from blond import Simulation, Beam, BiGaussian, SingleHarmonicRfStation
        >>> sim = Simulation(...)
        >>> beam1 = Beam(...)
        >>> rf_station1 = SingleHarmonicRfStation(...)
        >>> phase_obs = RfStationPhaseObservation(each_turn_i=1, rf_station=rf_station1)
        >>> beam_obs = BeamObservationOncePerTurn(each_turn_i=1)
        >>>
        >>> sim.run_simulation(
        ...     beams=(beam1,),
        ...     n_turns=1000,
        ...     observe=(phase_obs, beam_obs),
        ... )
        >>> # After simulation, access data via phase_obs.phases, beam_obs.dts, etc.

        Simulation with custom callback for live plotting:

        >>> import matplotlib.pyplot as plt
        >>> def plot_beam(simulation, beam):
        ...     if simulation.turn_i.value % 100 == 0:  # Every 100 turns
        ...         plt.clf()
        ...         plt.scatter(beam.read_partial_dt(), beam.read_partial_dE())
        ...         plt.pause(0.01)
        >>>
        >>> sim.run_simulation(
        ...     beams=(beam1,),
        ...     n_turns=1000,
        ...     callback=plot_beam,
        ... )

        Continue a simulation from turn 500:

        >>> sim.run_simulation(
        ...     beams=(beam1,),
        ...     n_turns=500,      # Run 500 more turns
        ... )
        """
        beams = _single_beam_to_tuple(beams)
        logger.info(f"Running `run_simulation` with {locals()}")
        n_turns = (
            int_from_float_with_warning(n_turns, warning_stacklevel=2)
            if n_turns is not None
            else None
        )
        _n_turns = self.finalize(
            beams=beams,
            n_turns=n_turns,
            observe=observe,
        )

        if len(beams) == 1:  # NOQA: PLR2004
            self.mainloop_single_beam(
                beam=beams[0],
                n_turns=_n_turns,
                observe=observe,
                show_progressbar=show_progressbar,
                callback=callback,
                callback_each_turn=callback_each_turn,
            )
        elif len(beams) == 2:  # NOQA: PLR2004
            assert (
                beams[0].is_counter_rotating,
                beams[1].is_counter_rotating,
            ) == (
                False,
                True,
            ), (
                "First beam must be normal, second beam must be counter-rotating"
            )
            self.mainloop_counterrotating_beam(
                n_turns=_n_turns,
                observe=observe,
                show_progressbar=show_progressbar,
                callback=callback,
                beams=beams,  # type: ignore
            )
        else:
            raise NotImplementedError(
                f"Up to two beam supported, but got {len(beams)}"
            )

    def finalize(
        self,
        beams: BeamBaseClass | tuple[BeamBaseClass, ...],
        n_turns: int | None = None,
        observe: tuple[ObservablesOncePerTurnBase, ...] = (),
    ) -> int:
        """
        Initialize all simulation components before running or loading results.

        This method is called internally by both ``run_simulation()`` and ``load_results()``
        to set up the simulation state. It:
            1. Validates the number of turns against the magnetic cycle
            2. Checks for performance warnings
            3. Calls ``on_run_simulation()`` hooks on all components
            4. Prepares observables for data collection

        Users typically don't need to call this method directly.

        Parameters
        ----------
        beams
            Beams to be simulated. For two-beam simulations, first must be
            co-rotating, second counter-rotating.
        n_turns
            Number of turns to simulate. If None, uses the maximum from the
            magnetic cycle (if defined).
        observe
            Observable objects that will record data during simulation.

        Returns
        -------
        n_turns
            The validated number of turns that will be simulated.

        Raises
        ------
        ValueError
            If ``n_turns`` is None and the magnetic cycle has unlimited turns.
        AssertionError
            If ``n_turns`` exceeds the magnetic cycle's maximum.

        See Also
        --------
        run_simulation : Execute beam dynamics simulation.
        load_results : Load previously saved results.

        Notes
        -----
        - This method temporarily attaches observables and beams to the simulation
          object to allow discovery by the initialization system.
        - Performance warnings are issued if using Python backend with many particles.
        """
        beams = _single_beam_to_tuple(beams)
        self.ring.assert_circumference()
        max_turns = self.magnetic_cycle.n_turns
        if n_turns is not None:
            _n_turns = int_from_float_with_warning(
                n_turns, warning_stacklevel=2
            )
            if max_turns is not None:
                assert _n_turns <= max_turns, (
                    f"Max turn number is {self.magnetic_cycle.n_turns=}, "
                    f"but trying to simulate {(0 + _n_turns)} turns"
                )
        elif max_turns is None:
            raise ValueError(
                f"`n_turns` must be provided, because"
                f" {type(self.magnetic_cycle)=} has"
                f" unlimited turns."
            )
        else:
            _n_turns = max_turns
        if backend.specials_mode == "python":
            particles_above_threshold = any(
                b.common_array_size
                > self._particle_performance_waning_threshold
                for b in beams
            )
            if particles_above_threshold:
                warnings.warn(
                    f"There are more than"
                    f" {self._particle_performance_waning_threshold}"
                    f" particles in your beam."
                    f" Consider using another backend via\n"
                    f" >>> from blond.core.backends.backend import backend\n"
                    f" >>> backend.set_specials(mode=...)",
                    PerformanceWarning,
                    stacklevel=2,
                )
        # temporarily pin attributes
        self._observe = (
            observe  # to find `on_run_simulation` within `simulation`
        )
        self._beams = beams  # to find `on_run_simulation` within `simulation`
        self._exec_on_run_simulation(
            beam=beams[0],
            n_turns=_n_turns,
        )
        # unpin temporary attributes
        del self._observe
        del self._beams
        if len(beams) == 1:
            self.turn_i.value = 0
            self.section_i.value = None
            for observable in observe:
                observable.update(
                    simulation=self,
                )
        return _n_turns

    def mainloop_single_beam(
        self,
        beam: BeamBaseClass,
        n_turns: int,
        observe: tuple[ObservablesOncePerTurnBase, ...] = (),
        show_progressbar: bool = True,
        callback: Callable[[Simulation, BeamBaseClass], None] | None = None,
        callback_each_turn: int = 1,
    ) -> None:
        """
        Execute the beam dynamics simulation for only one beam.

        Parameters
        ----------
        beam
            The beam to simulate.
        n_turns
            Number of turns to simulate.
        observe
            List of observables to protocol of whats happening inside
            the simulation.
        show_progressbar
            If True, will show a progress bar indicating how many turns have
            been completed and other metrics.
        callback
            User defined function `def myfunction(simulation: Simulation): ...`
            that is called each turn.
        callback_each_turn
            Specifies the the repitition rate at which `callback` is called.
            Deafault is 1, each turn.

        Notes
        -----
        This method assumes that ``Simulation.finalize(...)`` was executed
        before.
        """
        logger.info("Starting simulation mainloop...")
        iterator = range(self.turn_i.value, self.turn_i.value + n_turns)
        if show_progressbar:
            iterator = tqdm(iterator, desc="BLonD3 mainloop")  # Add TQDM
            # display to iteration
        for turn_i in iterator:
            self.turn_i.value = turn_i
            for element in self._ring.elements.elements:
                self.section_i.value = element.section_index
                if element.is_active_this_turn(turn_i=self.turn_i.value):
                    element.track(beam)
            for observable in observe:
                if observable.is_active_this_turn(turn_i=self.turn_i.value):
                    observable.update(
                        simulation=self,
                    )
            if callback is not None and (turn_i % callback_each_turn) == 0:
                callback(self, beam)

    def mainloop_counterrotating_beam(
        self,
        beams: tuple[BeamBaseClass, BeamBaseClass],
        n_turns: int,
        observe: tuple[ObservablesOncePerTurnBase, ...] = (),
        show_progressbar: bool = True,
        callback: Callable[[Simulation, BeamBaseClass], None] | None = None,
    ) -> None:
        """
        Execute the beam dynamics simulation for counter-rotating beams.

        Parameters
        ----------
        beams
            Tuple of two beams (co-rotating, counter-rotating).
        n_turns
            Number of turns to simulate.
        observe
            List of observables to protocol of whats happening inside
            the simulation.
        show_progressbar
            If True, will show a progress bar indicating how many turns have
            been completed and other metrics.
        callback
            User defined function `def myfunction(simulation: Simulation): ...`
            that is called each turn.
        """
        warnings.warn("Untested code", NotTestedWarning, stacklevel=2)

        if callback is not None:
            warnings.warn(
                "Callbacks are currently not supported for simulations"
                " with counter-rotating beams.",
                UserWarning,
                stacklevel=2,
            )

        logger.info("Starting simulation mainloop...")
        iterator = range(n_turns)
        if show_progressbar:
            iterator = tqdm(iterator)  # Add TQDM display to iteration
        self.turn_i.value = 0

        num_elements = len(self._ring.elements.elements)

        for observable in observe:
            observable.update(
                simulation=self,
            )

        for turn_i in iterator:
            for element_ind, element in enumerate(
                self._ring.elements.elements
            ):
                self.turn_i.value = turn_i
                self.section_i.value = element.section_index

                if element.is_active_this_turn(turn_i=self.turn_i.value):
                    element.track(beams[0])  # [0] is expected to be corotating
                element_counterrot = self.ring.elements.elements[
                    num_elements - element_ind - 1
                ]
                if element_counterrot.is_active_this_turn(
                    turn_i=self.turn_i.value
                ):
                    element_counterrot.track(beams[1])
            for observable in observe:
                if observable.is_active_this_turn(turn_i=self.turn_i.value):
                    observable.update(
                        simulation=self,
                    )
        # reset counters to uninitialized again
        self.turn_i.value = None
        self.section_i.value = None

    def save_results(
        self,
        observe: tuple[ObservablesOncePerTurnBase, ...] = (),
        common_name: str | None = None,
    ) -> None:
        """
        Save observable data to disk for later analysis.

        After running a simulation, this method saves the data collected by observables
        (like RF phase evolution, beam profiles, etc.) to disk. This is useful for:
            - Analyzing results later without re-running the simulation
            - Sharing results with others

        Each observable is saved according to its own format (typically NumPy arrays).

        Parameters
        ----------
        observe
            Tuple of observable objects whose data should be saved. These should be
            the same observables that were passed to ``run_simulation()``.
            Default is empty tuple.
        common_name
            Optional prefix for all saved files. If provided, all observables will
            use this as part of their filename, making it easier to identify related
            files. If None, each observable uses its default naming. Default is None.

        See Also
        --------
        load_results : Load previously saved results.
        run_simulation : Execute beam dynamics simulation.

        Notes
        -----
        - Saved files are typically stored in the current working directory or a
          subdirectory defined by the observable.
        - Use ``load_results()`` to load saved data without re-running the simulation.

        Examples
        --------
        Save results after a simulation:

        >>> from blond import Simulation, Beam
        >>> sim = Simulation(...)
        >>> beam1 = Beam(...)
        >>> # Run simulation with observables
        >>> sim.run_simulation(
        ...     beams=(beam1,),
        ...     n_turns=1000,
        ...     observe=(phase_obs, beam_obs),
        ... )
        >>>
        >>> # Save the data
        >>> sim.save_results(observe=(phase_obs, beam_obs))

        Save with a custom name prefix:

        >>> from blond import Simulation
        >>> sim = Simulation(...)
        >>> sim.run_simulation(observe=(phase_obs, beam_obs), ...)
        >>> sim.save_results(
        ...     observe=(phase_obs, beam_obs),
        ...     common_name="simulation_450GeV_1000turns"
        ... )
        """
        for observable in observe:
            if common_name is not None:
                observable.rename(new_common_filepath=common_name)
            observable.to_disk()

    def load_results(
        self,
        beams: BeamBaseClass | tuple[BeamBaseClass, ...],
        n_turns: int | None = None,
        observe: tuple[ObservablesOncePerTurnBase, ...] = (),
        common_name: str | None = None,
    ) -> None:
        """
        Load previously saved observable data from disk.

        This method loads data from observables that were saved with ``save_results()``
        in a previous session. This allows you to analyze results without re-running
        the expensive simulation. The observables are initialized as if the simulation
        had just completed.

        Typical use case: Run a long simulation once, save the results, then load them
        multiple times for different analyses or plotting.

        Parameters
        ----------
        beams
            Tuple of beam objects. Must match the beams used when the data was saved.
        n_turns
            Number of turns that were simulated. Must match the saved data.
            If None, uses the maximum from the magnetic cycle. Default is None.
        observe
            Tuple of observable objects to load data into. These observables will
            be populated with the saved data. Default is empty tuple.
        common_name
            Common prefix used when saving the files. Must match the name used in
            ``save_results()`` if one was provided. Default is None.

        Raises
        ------
        AssertionError
            If ``n_turns`` exceeds the maximum turns defined by the
            magnetic cycle, or if beam ordering is incorrect for counter-rotating simulations.

        FileNotFoundError
            If the saved data files cannot be found.
        AssertionError
            If the loaded data dimensions don't match the specified parameters.

        See Also
        --------
        save_results : Save simulation results to disk.
        run_simulation : Execute beam dynamics simulation.

        Notes
        -----
        - The observables must be created with the same parameters as when the data
          was saved.
        - The simulation setup (ring, RF stations, etc.) should match the original
          simulation, though only the observables are populated with data.
        - This method calls ``finalize()`` internally to set up the simulation state.

        Examples
        --------
        Load previously saved results:

        >>> # Create observables (same as before)
        >>> phase_obs = RfStationPhaseObservation(each_turn_i=1, rf_station=rf_station1)
        >>> beam_obs = BeamObservationEndOfTurn(each_turn_i=1, beam=beam1)
        >>>
        >>> # Load the saved data
        >>> sim.load_results(
        ...     beams=(beam1,),
        ...     n_turns=1000,
        ...     observe=(phase_obs, beam_obs),
        ... )
        >>>
        >>> # Now analyze the loaded data
        >>> plt.plot(phase_obs.phases)
        >>> plt.show()

        Load with custom name prefix:

        >>> sim.load_results(
        ...     beams=(beam1,),
        ...     n_turns=1000,
        ...     observe=(phase_obs, beam_obs),
        ...     common_name="simulation_450GeV_1000turns"
        ... )

        Combine with run_simulation for caching:

        >>> try:
        ...     sim.load_results(beams=(beam1,), n_turns=1000, observe=(phase_obs, beam_obs))
        ...     print("Loaded cached results")
        ... except FileNotFoundError:
        ...     print("No cached results, running simulation")
        ...     sim.run_simulation(beams=(beam1,), n_turns=1000, observe=(phase_obs, beam_obs))
        ...     sim.save_results(observe=(phase_obs, beam_obs))
        """
        self.finalize(
            beams=beams,
            n_turns=n_turns,
            observe=observe,
        )
        for observable in observe:
            if common_name is not None:
                observable.rename(new_common_filepath=common_name)
            observable.from_disk()
