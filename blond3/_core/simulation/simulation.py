from __future__ import annotations

import logging
from functools import cached_property
from pstats import SortKey
from typing import TYPE_CHECKING, Callable

import numpy as np
from tqdm import tqdm

from ..base import BeamPhysicsRelevant, Preparable, HasPropertyCache
from ..base import DynamicParameter
from ..helpers import find_instances_with_method, int_from_float_with_warning
from ..ring.helpers import get_elements, get_init_order
from ...cycles.energy_cycle import EnergyCycleBase, EnergyCyclePerTurn

if TYPE_CHECKING:  # pragma: no cover
    from typing import (
        Optional,
        Tuple,
    )
    from numpy.typing import NDArray as NumpyArray
    from ..beam.base import BeamBaseClass
    from ..ring.ring import Ring
    from ...beam_preparation.base import BeamPreparationRoutine
    from ...handle_results.observables import Observables

logger = logging.getLogger(__name__)


class Simulation(Preparable, HasPropertyCache):
    """Context manager to perform beam physics simulations of synchrotrons

    Parameters
    ----------
    ring
        Ring a.k.a. synchrotron
    beams
        Base class to host particle coordinates and timing information
    energy_cycle
        Container object to handle the scheduled energy gain
        per turn or by time

    Attributes
    ----------
    turn_i
        Counter of the current turn during simulation,
        which can also handle subscriptions/callbacks
    section_i
        Counter of the current section during simulation,
        which can also handle subscriptions/callbacks
    """

    def __init__(
        self,
        ring: Ring,
        beams: Tuple[BeamBaseClass, ...],
        energy_cycle: NumpyArray | EnergyCycleBase,
    ):
        super().__init__()
        self._ring: Ring = ring
        assert beams != (), f"{beams=}"
        assert len(beams) <= 2, "Maximum two beams allowed"

        self._beams: Tuple[BeamBaseClass, ...] = beams

        if isinstance(energy_cycle, np.ndarray):
            energy_cycle = EnergyCyclePerTurn(energy_cycle)
        self._energy_cycle: EnergyCycleBase = energy_cycle

        self.turn_i = DynamicParameter(None)
        self.section_i = DynamicParameter(None)

        self._exec_on_init_simulation()

    def profiling(
        self, turn_i_init: int, profile_start_turn_i: int, profile_n_turns:
            int,
            sortby: SortKey =SortKey.CUMULATIVE

    ):
        """Executes the python profiler

        Parameters
        ----------
        turn_i_init
            Initial turn to start simulation
        profile_start_turn_i
            First turn to start profiling.
            Can be later than turn_i_init.
        profile_n_turns
            Total number of turns that are consideref for profiling
        sortby
            Order to sort the runtime table
        """
        assert profile_start_turn_i >= turn_i_init

        import cProfile, pstats, io

        pr = cProfile.Profile()

        def my_callback(simulation: Simulation):
            if simulation.turn_i.value == profile_start_turn_i:
                pr.enable()

        end_turn = profile_start_turn_i + profile_n_turns
        self.run_simulation(
            n_turns=end_turn - turn_i_init,
            turn_i_init=turn_i_init,
            show_progressbar=False,
            callback=my_callback,
        )

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def invalidate_cache(self):
        """Delete the stored values of functions with @cached_property"""

        pass  # TODO

    def get_potential_well_analytic(self):
        raise NotImplementedError  # TODO

    def get_potential_well_empiric(self):
        raise NotImplementedError  # TODO

        from ... import WakeField
        from ...physics.cavities import CavityBaseClass
        from ...physics.drifts import DriftBaseClass
        from ...physics.profiles import ProfileBaseClass

        from blond3._core.beam.beams import ProbeBunch

        profile = self.ring.elements.get_element(ProfileBaseClass)
        x = profile.hist_x

        bunch = ProbeBunch(dt=x.copy(), particle_type=self.beams[0].particle_type)
        for element in self.ring.elements.elements:
            if (
                isinstance(element, DriftBaseClass)
                or isinstance(element, CavityBaseClass)
                or isinstance(element, WakeField)
            ):
                element.track(bunch)
        potential_well = np.trapezoid(bunch.read_partial_dE(), x)
        potential_well -= potential_well.min()
        return potential_well

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        pass

    def _exec_all_in_tree(self, method: str, **kwargs):
        """Execute all methods that are somewhere in the attribute hierarchy of `Simulation`

        Parameters
        ----------
        method
            Method name to execute everywhere
        kwargs
            Extra keyword arguments
        """
        logger.debug(f"Calling all {method}({kwargs}) in {self}")
        instances = find_instances_with_method(self, f"{method}")
        logger.debug(f"Found {instances} to be initialized")
        ordered_classes = get_init_order(instances, f"{method}.requires")

        classes_check = set()
        for ins in instances:
            classes_check.add(type(ins))
        # assert len(classes_check) == len(ordered_classes), "BUG"
        if "ABCMeta" in ordered_classes:
            ordered_classes.pop(ordered_classes.index("ABCMeta"))
        logger.info(f"Execution order for `{method}` is {ordered_classes}")

        for cls in ordered_classes:
            for element in instances:
                if not type(element).__name__ == cls:
                    continue
                logger.info(f"Running `{method}` of {element}")
                getattr(element, method)(**kwargs)

    def _exec_on_init_simulation(self):
        """Execute all `on_init_simulation` in the attribute hierarchy of `Simulation`"""
        self._exec_all_in_tree("on_init_simulation", simulation=self)

    def _exec_on_run_simulation(self, n_turns: int, turn_i_init: int):
        """Execute all `on_run_simulation` in the attribute hierarchy of `Simulation`

        Parameters
        ----------
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        self._exec_all_in_tree(
            "on_run_simulation",
            simulation=self,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )

    @staticmethod
    def from_locals(locals: dict) -> Simulation:
        """Automatically instance simulation from all locals of where its called

        Parameters
        ----------
        locals
            Just hand `locals()` over
        """
        from ..beam.base import BeamBaseClass  # prevent cyclic import
        from ..ring.ring import Ring  # prevent cyclic import

        locals_list = locals.values()
        logger.debug(f"Found {locals.keys()}")
        _rings = get_elements(locals_list, Ring)
        assert len(_rings) == 1, f"Found {len(_rings)} rings"
        ring = _rings[0]

        beams = get_elements(locals_list, BeamBaseClass)

        _energy_cycles = get_elements(locals_list, EnergyCycleBase)
        assert len(_energy_cycles) == 1, f"Found {len(_energy_cycles)} energy cycles"
        energy_cycle = _energy_cycles[0]

        elements = get_elements(locals_list, BeamPhysicsRelevant)
        ring.add_elements(elements=elements, reorder=True)

        logger.debug(f"{ring=}")
        logger.debug(f"{beams=}")
        logger.debug(f"{elements=}")

        sim = Simulation(ring=ring, beams=beams, energy_cycle=energy_cycle)
        logger.info(sim.ring.elements.get_order_info())
        return sim

    @property  # as readonly attributes
    def ring(self) -> Ring:
        """Ring a.k.a. synchrotron"""
        return self._ring

    @property  # as readonly attributes
    def beams(self) -> Tuple[BeamBaseClass, ...]:
        """Base class to host particle coordinates and timing information"""
        return self._beams

    @property  # as readonly attributes
    def energy_cycle(self) -> EnergyCycleBase:
        """Programmed energy program of the synchrotron"""
        return self._energy_cycle

    @cached_property
    def get_separatrix(self):
        raise NotImplementedError
        return None

    @cached_property
    def get_potential_well(self):
        raise NotImplementedError
        return None

    @cached_property
    def get_hash(self):
        raise NotImplementedError
        return None

    def print_one_turn_execution_order(self) -> None:
        """Prints the execution order of the main simulation loop"""
        self._ring.elements.print_order()

    #  properties that have the @cached_property decorator
    cached_properties = (
        "get_separatrix",
        "get_potential_well",
        "get_hash",
    )

    def _invalidate_cache(
        self,
        # turn i needed to be
        # compatible with subscription
        turn_i: int,
    ) -> None:
        """Reset cache of `cached_property` attributes"""
        super()._invalidate_cache(Simulation.cached_properties)

    def on_prepare_beam(
        self, preparation_routine: BeamPreparationRoutine, turn_i: int = 0
    ) -> None:
        """Run the routine to prepare the beam

        Parameters
        ----------
        preparation_routine
            Algorithm to prepare the beam dt and dE coorinates
        turn_i
            Turn to prepare the beam for

        """
        logger.info("Running `on_prepare_beam`")
        self.turn_i.value = turn_i
        preparation_routine.on_prepare_beam(simulation=self)

    def run_simulation(
        self,
        n_turns: Optional[int] = None,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
        callback: Optional[Callable[[Simulation], None]] = None,
    ) -> None:
        """
        Execute the beam dynamics simulation


        Parameters
        ----------
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to start with simulation
        observe
            List of observables to protocol of whats happening inside
            the simulation
        show_progressbar
            If True, will show a progress bar indicating how many turns have
            been completed and other metrics
        callback
            User defined function `def myfunction(simulation: Simulation): ...`
            that is called each turn.

        """
        logger.info(f"Running `run_simulation` with {locals()}")
        n_turns = int_from_float_with_warning(n_turns, warning_stacklevel=2)

        max_turns = self.energy_cycle.n_turns
        if max_turns is not None:
            assert (turn_i_init + n_turns) <= max_turns, (
                f"Max turn number is {max_turns}, but trying to "
                f"simulate {(turn_i_init + n_turns)} turns"
            )
        self.observe = observe
        self._exec_on_run_simulation(
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )
        del self.observe
        if len(self._beams) == 1:
            self._run_simulation_single_beam(
                n_turns=n_turns,
                turn_i_init=turn_i_init,
                observe=observe,
                show_progressbar=show_progressbar,
                callback=callback,
            )
        elif len(self._beams) == 2:
            assert (
                self._beams[0].is_counter_rotating,
                self._beams[1].is_counter_rotating,
            ) == (
                False,
                True,
            ), "First beam must be normal, second beam must be counter-rotating"
            self._run_simulation_counterrotating_beam(
                n_turns=n_turns,
                turn_i_init=turn_i_init,
                observe=observe,
                show_progressbar=show_progressbar,
                callback=callback,
            )

    def _run_simulation_single_beam(
        self,
        n_turns: int,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
        callback: Optional[Callable[[Simulation], None]] = None,
    ) -> None:
        """
        Execute the beam dynamics simulation for only one beam


        Parameters
        ----------
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to start with simulation
        observe
            List of observables to protocol of whats happening inside
            the simulation
        show_progressbar
            If True, will show a progress bar indicating how many turns have
            been completed and other metrics
        callback
            User defined function `def myfunction(simulation: Simulation): ...`
            that is called each turn.

        """
        logger.info("Starting simulation mainloop...")
        iterator = range(turn_i_init, turn_i_init + n_turns)
        if show_progressbar:
            iterator = tqdm(iterator)  # Add TQDM display to iteration
        self.turn_i.on_change(self._invalidate_cache)
        self.turn_i.value = 0
        for observable in observe:
            observable.update(simulation=self)
        for turn_i in iterator:
            self.turn_i.value = turn_i
            for element in self._ring.elements.elements:
                self.section_i.current_group = element.section_index
                if element.is_active_this_turn(turn_i=self.turn_i.value):
                    element.track(self._beams[0])
            for observable in observe:
                if observable.is_active_this_turn(turn_i=self.turn_i.value):
                    observable.update(simulation=self)
            if callback is not None:
                callback(self)

        # reset counters to uninitialized again
        self.turn_i.value = None
        self.section_i.value = None

    def get_legacy_map(self):
        raise NotImplementedError
        from ...physics.cavities import (  # prevent cyclic import
            CavityBaseClass,
            SingleHarmonicCavity,
            MultiHarmonicCavity,
        )

        ring_length = self.ring.circumference
        bending_radius = self.ring.bending_radius
        drift = self.ring.elements.get_element(DriftBaseClass)
        alpha_0 = drift.alpha_0
        synchronous_data = self.energy_cycle._synchronous_data
        synchronous_data_type = self.energy_cycle._synchronous_data_type
        particle = self.beams[0].particle_type
        #  BLonD legacy Imports
        from blond.beam.beam import Beam
        from blond.beam.profile import Profile, CutOptions
        from blond.input_parameters.rf_parameters import RFStation
        from blond.input_parameters.ring import Ring
        from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
        from blond.impedances.impedance import TotalInducedVoltage

        ring_blond2 = Ring(
            ring_length=ring_length,
            alpha_0=alpha_0,
            synchronous_data=synchronous_data,
            synchronous_data_type=synchronous_data_type,
            particle=particle,
            bending_radius=bending_radius,
            n_sections=self.ring.elements.n_cavities,
            alpha_1=getattr(drift, "alpha_1", None),
            alpha_2=getattr(drift, "alpha_2", None),
            ring_options=None,
        )
        beam_blond2 = Beam(
            ring=ring_blond2,
            n_macroparticles=self.beams[0]._n_macroparticles__init,
            intensity=self.beams[0]._n_particles__init,
        )
        # todo handle multiple RF stations
        cavity_blond3: SingleHarmonicCavity | MultiHarmonicCavity = (
            self.ring.elements.get_element(CavityBaseClass)
        )
        rf_station = RFStation(
            ring=ring_blond2,
            harmonic=cavity_blond3.rf_program.harmonics,
            voltage=cavity_blond3.rf_program.effective_voltages,
            phi_rf_d=cavity_blond3.rf_program.phases,
            n_rf=len(cavity_blond3.rf_program.phases),
            section_index=0,
            omega_rf=cavity_blond3.rf_program.omegas_rf,
            phi_noise=cavity_blond3.rf_program.phase_noise,
            phi_modulation=cavity_blond3.rf_program.phase_modulation,
            rf_station_options=None,
        )
        profile_blond3 = self.ring.elements.get_element(ProfileBaseClass)
        profile_blond2 = Profile(
            beam=beam_blond2,
            cut_options=CutOptions(
                cut_left=profile_blond3.cut_left,
                cut_right=profile_blond3.cut_right,
                n_slices=profile_blond3.n_bins,
            ),
            fit_options=None,
            filter_options=None,
        )
        total_induced_voltage = TotalInducedVoltage(
            beam=beam_blond2, profile=profile_blond2, induced_voltage_list=None
        )
        ring_rf_tracker = RingAndRFTracker(
            rf_station=rf_station,
            beam=beam_blond2,
            solver=None,  # todo
            beam_feedback=None,  # todo as feature for blond3
            noise_feedback=None,  # todo as feature for blond3
            cavity_feedback=None,  # todo as feature for blond3
            periodicity=False,  # todo as feature for blond3
            # interpolation=None,
            profile=profile_blond2,
            total_induced_voltage=total_induced_voltage,
        )
        full_ring = FullRingAndRF(ring_and_rf_section=ring_rf_tracker)
        return [
            ring_blond2,
            beam_blond2,
            rf_station,
            profile_blond2,
            total_induced_voltage,
            ring_rf_tracker,
            full_ring,
        ]

    def _run_simulation_counterrotating_beam(
        self,
        n_turns: int,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
        callback: Optional[Callable[[Simulation], None]] = None,

    ) -> None:
        """
        Execute the beam dynamics simulation for only one beam


        Parameters
        ----------
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to start with simulation
        observe
            List of observables to protocol of whats happening inside
            the simulation
        show_progressbar
            If True, will show a progress bar indicating how many turns have
            been completed and other metrics
        callback
            User defined function `def myfunction(simulation: Simulation): ...`
            that is called each turn.

        """
        raise  NotImplementedError()
        pass  # todo

    def load_results(
        self,
        n_turns: int,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        callback: Callable[[Simulation], None] = None,
    ) -> SimulationResults:
        raise FileNotFoundError()
        return
