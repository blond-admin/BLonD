from __future__ import annotations

import logging
from copy import deepcopy
from functools import cached_property
from pstats import SortKey
from typing import TYPE_CHECKING, Callable
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_simpson, cumulative_trapezoid
from tqdm import tqdm  # type: ignore

from ..._generals._warnings import PerformanceWarning
from ...cycles.magnetic_cycle import MagneticCycleBase
from ...physics.profiles import ProfileBaseClass
from ..backends.backend import backend
from ..base import (
    BeamPhysicsRelevant,
    DynamicParameter,
    HasPropertyCache,
    Preparable,
)
from ..helpers import find_instances_with_method, int_from_float_with_warning
from ..ring.helpers import get_elements, get_init_order

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Dict, Optional, Tuple

    from numpy.typing import NDArray as NumpyArray

    from blond import (
        Beam,
        DriftSimple,
        MagneticCyclePerTurn,
        SingleHarmonicCavity,
    )
    from blond.legacy.blond2.beam.beam import Beam as Blond2Beam
    from blond.legacy.blond2.beam.profile import Profile as Blond2Profile
    from blond.legacy.blond2.impedances.impedance import (
        TotalInducedVoltage as Blond2TotalInducedVoltage,
    )
    from blond.legacy.blond2.input_parameters.rf_parameters import (
        RFStation as Blond2RFStation,
    )
    from blond.legacy.blond2.input_parameters.ring import Ring as Blond2Ring
    from blond.legacy.blond2.trackers.tracker import (
        FullRingAndRF as Blond2FullRingAndRF,
    )
    from blond.legacy.blond2.trackers.tracker import (
        RingAndRFTracker as Blond2RingAndRFTracker,
    )

    from ...beam_preparation.base import BeamPreparationRoutine
    from ...handle_results.observables import Observables
    from ..beam.base import BeamBaseClass
    from ..beam.particle_types import ParticleType
    from ..ring.ring import Ring

logger = logging.getLogger(__name__)


class Simulation(Preparable, HasPropertyCache):
    """Context manager to perform beam physics simulations of synchrotrons

    Parameters
    ----------
    ring
        Ring a.k.a. synchrotron
    magnetic_cycle
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
        magnetic_cycle: MagneticCycleBase,
    ) -> None:
        assert ring.elements.n_elements > 0, f"{ring.elements.n_elements=}"

        from .intensity_effect_manager import IntensityEffectManager

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
        beams: Tuple[BeamBaseClass],
        turn_i_init: int,
        profile_start_turn_i: int,
        profile_n_turns: int,
        sortby: SortKey = SortKey.CUMULATIVE,
    ) -> None:
        """Executes the python profiler

        Parameters
        ----------
        beams
            Beams that are used to perform the simulation
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

        import cProfile
        import io
        import pstats

        pr = cProfile.Profile()

        # trigger profiling later than turn 0
        def start_profiling(simulation: Simulation, beam: BeamBaseClass):
            if simulation.turn_i.value == profile_start_turn_i:
                pr.enable()

        end_turn = profile_start_turn_i + profile_n_turns
        self.run_simulation(
            beams=beams,
            n_turns=end_turn - turn_i_init,
            turn_i_init=turn_i_init,
            show_progressbar=False,
            callback=start_profiling,
        )

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def invalidate_cache(self):
        """Delete the stored values of functions with @cached_property"""

        pass  # TODO

    def plot_potential_well_empiric(
        self,
        ts: NumpyArray,
        particle_type: ParticleType,
        subtract_min: bool = True,
    ) -> None:
        potential_well = self.get_potential_well_empiric(
            ts=ts,
            particle_type=particle_type,
            subtract_min=subtract_min,
        )
        plt.plot(ts, potential_well)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (arb. unit)")

    def get_drift_term_empiric(
        self,
        dE: NumpyArray,
        particle_type: ParticleType,
        intensity: int = 0,
    ) -> NumpyArray:
        """
        Obtain the potential well by tracking a beam one turn

        Notes
        -----
        This function internally obtains `dE_out` of `dt_in`.
        During one turn with many drifts, the time coordinate will change
        and sample different positions of dt, which is the expected
        physical behaviour. The RF of successive station can thus appear
        phase shifted/distorted due to the inherent drift in between RF
        stations=.

        Parameters
        ----------
        dE
            Energy coordinates to probe the potential, in [eV]
        particle_type
            Type of particle to probe.
            The particle charge influences the phase advance per station
            and might exhibit different distortion of the potential well
            due to the side effects described in `Notes`

        Returns
        -------
        potential_well
            The effective voltage that lead to a change of `dE` in one turn.
        """
        from ..._core.beam.beams import ProbeBeam

        probe_bunch = ProbeBeam(
            dE=dE,
            particle_type=particle_type,
            intensity=intensity,
        )
        t0 = probe_bunch.reference_time
        self.run_simulation(
            beams=(probe_bunch,),
            n_turns=1,
            turn_i_init=0,
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
        ts: NumpyArray,
        particle_type: ParticleType,
        subtract_min: bool = True,
        intensity: int = 0,
    ) -> Tuple[NumpyArray, float]:
        """
        Obtain the potential well by tracking a beam one turn

        Notes
        -----
        This function internally obtains `dE_out` of `dt_in`.
        During one turn with many drifts, the time coordinate will change
        and sample different positions of dt, which is the expected
        physical behaviour. The RF of successive station can thus appear
        phase shifted/distorted due to the inherent drift in between RF
        stations=.

        Parameters
        ----------
        ts
            Time coordinates to probe the potential, in [s]
        particle_type
            Type of particle to probe.
            The particle charge influences the phase advance per station
            and might exhibit different distortion of the potential well
            due to the side effects described in `Notes`
        subtract_min
            If True, will always return min(potential_well) = 0.
            If False, potential_well[0] = 0.

        Returns
        -------
        potential_well
            The effective voltage that lead to a change of `dE` in one turn.
        factor
            The fraction of the time span of `ts` relative to the
            revolution time `t_rev`.
            ``(ts[-1] - ts[0]) / t_rev``
        """
        from ..._core.beam.beams import ProbeBeam

        probe_bunch = ProbeBeam(
            dt=ts,
            particle_type=particle_type,
            intensity=intensity,
        )
        bunch_before = deepcopy(probe_bunch)
        t_0 = probe_bunch.reference_time
        deepcopy(self).run_simulation(
            beams=(probe_bunch,),
            n_turns=1,
            turn_i_init=0,
            show_progressbar=False,
        )
        change_t = probe_bunch._dt - bunch_before._dt
        change_E = probe_bunch._dE - bunch_before._dE
        idx = np.argmax(change_t)
        dt_per_dE = change_t[idx] / change_E[idx]
        print(f"{(dt_per_dE)=}")
        t_1 = probe_bunch.reference_time
        t_rev = t_1 - t_0
        factor = (ts[-1] - ts[0]) / t_rev
        potential_well = -cumulative_simpson(
            probe_bunch.read_partial_dE(), initial=0
        ) / len(ts)
        if subtract_min:
            potential_well -= potential_well.min()
        return potential_well / particle_type.charge, factor

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        pass

    def _exec_all_in_tree(self, method: str, **kwargs) -> None:
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

    def _exec_on_init_simulation(self) -> None:
        """Execute all `on_init_simulation` in the attribute hierarchy of `Simulation`"""
        self._exec_all_in_tree("on_init_simulation", simulation=self)

    def _exec_on_run_simulation(
        self,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
    ) -> None:
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
            beam=beam,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )

    @staticmethod
    def from_locals(
        locals: dict[str, Any], verbose: bool = False
    ) -> Simulation:
        """Automatically instance simulation from all locals of where its called

        Parameters
        ----------
        locals
            Dictionary of elements that
            should be contained in the simulation.

        Examples
        --------
        >>> beam1 = Beam( ... )
        >>> ring = Ring( ... )
        >>> energy_cycle = MagneticCyclePerTurn( ... )
        >>> cavity1 = SingleHarmonicCavity( ... )
        >>> drift1 = DriftSimple( ... )
        >>> Simulation.from_locals(locals=locals(), verbose=True)

        """
        from ..beam.base import BeamBaseClass  # prevent cyclic import
        from ..ring.ring import Ring  # prevent cyclic import

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

        elements = get_elements(locals_list, BeamPhysicsRelevant)  # type: ignore
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
        """Ring a.k.a. synchrotron"""
        return self._ring

    @property  # as readonly attributes
    def magnetic_cycle(self) -> MagneticCycleBase:
        """Programmed energy program of the synchrotron"""
        return self._magnetic_cycle

    @cached_property
    def get_separatrix(self) -> None:
        raise NotImplementedError
        return None

    @cached_property
    def get_potential_well(self) -> None:
        raise NotImplementedError
        return None

    def print_one_turn_execution_order(self) -> None:
        """Prints the execution order of the main simulation loop"""
        self._ring.elements.print_order()

    #  properties that have the @cached_property decorator
    cached_properties = (
        "get_separatrix",
        "get_potential_well",
    )

    def _invalidate_cache_on_turn(
        self,
        turn_i: int,  # required by `turn_i.on_change`
    ) -> None:
        """
        Reset cache of `cached_property` attributes

        Parameters
        ----------
        Current turn

        Notes
        -----
        This method is subscribed to turn_i.on_change

        """
        self._invalidate_cache(Simulation.cached_properties)

    def prepare_beam(
        self,
        beam: BeamBaseClass,
        preparation_routine: BeamPreparationRoutine,
        turn_i: int = 0,
    ) -> None:
        """Run the routine to prepare the beam

        Parameters
        ----------
        beam
            Simulation beam object
        preparation_routine
            Algorithm to prepare the beam `dt` and `dE` coorinates
        turn_i
            Turn to prepare the beam for

        """
        logger.info("Running `prepare_beam`")
        self.turn_i.value = turn_i
        preparation_routine.prepare_beam(simulation=self, beam=beam)

    def run_simulation(
        self,
        beams: Tuple[BeamBaseClass],
        n_turns: Optional[int] = None,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
        callback: Optional[Callable[[Simulation, Beam], None]] = None,
    ) -> None:
        """
        Execute the beam dynamics simulation


        Parameters
        ----------
        beams
            Beams that are used to perform the simulation
        n_turns
            Number of turns to simulate.
            If None, will use the maximum number of turns given by the cycle.
        turn_i_init
            Initial turn to start with simulation
        observe
            List of observables to protocol of whats happening inside
            the simulation
        show_progressbar
            If True, will show a progress bar indicating how many turns have
            been completed and other metrics
        callback
            User defined function
            `def myfunction(simulation: Simulation, beam: Beam): ...`
            that is called each turn.

        """
        logger.info(f"Running `run_simulation` with {locals()}")
        _n_turns = self.finalze(
            beams=beams,
            n_turns=n_turns,
            observe=observe,
            turn_i_init=turn_i_init,
        )

        if len(beams) == 1:
            self._run_simulation_single_beam(
                beam=beams[0],
                n_turns=_n_turns,
                turn_i_init=turn_i_init,
                observe=observe,
                show_progressbar=show_progressbar,
                callback=callback,
            )
        elif len(beams) == 2:
            assert (
                beams[0].is_counter_rotating,
                beams[1].is_counter_rotating,
            ) == (
                False,
                True,
            ), (
                "First beam must be normal, second beam must be counter-rotating"
            )
            self._run_simulation_counterrotating_beam(
                n_turns=_n_turns,
                turn_i_init=turn_i_init,
                observe=observe,
                show_progressbar=show_progressbar,
                callback=callback,
            )

    def finalze(self, beams, n_turns, observe, turn_i_init):
        max_turns = self.magnetic_cycle.n_turns
        if n_turns is not None:
            _n_turns = int_from_float_with_warning(
                n_turns, warning_stacklevel=2
            )
            if max_turns is not None:
                assert (turn_i_init + _n_turns) <= max_turns, (
                    f"Max turn number is {self.magnetic_cycle.n_turns=}, "
                    f"but trying to simulate {(turn_i_init + _n_turns)} turns"
                )
        else:
            if max_turns is None:
                raise ValueError(
                    f"`n_turns` must be provided, because"
                    f" {type(self.magnetic_cycle)=} has"
                    f" unlimited turns."
                )
            else:
                _n_turns = max_turns
        if backend.specials_mode == "python":
            particles_above_threshold = any(
                [
                    b.common_array_size
                    > self._particle_performance_waning_threshold
                    for b in beams
                ]
            )
            if particles_above_threshold:
                warn(
                    f"There are more than"
                    f" {self._particle_performance_waning_threshold}"
                    f" particles in your beam."
                    f" Consider using another backend via\n"
                    f" >>> from blond._core.backends.backend import backend\n"
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
            turn_i_init=turn_i_init,
        )
        # unpin temporary attributes
        del self._observe
        del self._beams
        return _n_turns

    def _run_simulation_single_beam(
        self,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
        callback: Optional[Callable[[Simulation, Beam], None]] = None,
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
        self.turn_i.on_change(self._invalidate_cache_on_turn)
        self.turn_i.value = 0
        for observable in observe:
            observable.update(
                simulation=self,
                beam=beam,
            )
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
                        beam=beam,
                    )
            if callback is not None:
                callback(simulation=self, beam=beam)

        # reset counters to uninitialized again
        self.turn_i.value = None
        self.section_i.value = None

    def get_legacy_map(
        self,
    ) -> list[
        Blond2Ring
        | Blond2Beam
        | Blond2RFStation
        | Blond2Profile
        | Blond2TotalInducedVoltage
        | Blond2RingAndRFTracker
        | Blond2FullRingAndRF
    ]:
        raise NotImplementedError
        from ...physics.cavities import (  # prevent cyclic import
            CavityBaseClass,
            MultiHarmonicCavity,
            SingleHarmonicCavity,
        )

        ring_length = self.ring.closed_orbit_length
        bending_radius = self.ring.bending_radius
        drift = self.ring.elements.get_element(DriftBaseClass)  # type: ignore
        alpha_0 = drift.alpha_0
        synchronous_data = self.magnetic_cycle._synchronous_data
        synchronous_data_type = self.magnetic_cycle._synchronous_data_type
        particle = self.beams[0].particle_type
        #  BLonD legacy Imports
        from blond.legacy.blond2.beam.beam import Beam
        from blond.legacy.blond2.beam.profile import CutOptions, Profile
        from blond.legacy.blond2.impedances.impedance import (
            TotalInducedVoltage,
        )
        from blond.legacy.blond2.input_parameters.rf_parameters import (
            RFStation,
        )
        from blond.legacy.blond2.input_parameters.ring import Ring
        from blond.legacy.blond2.trackers.tracker import (
            FullRingAndRF,
            RingAndRFTracker,
        )

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
        # FIXME
        rf_station = RFStation(
            ring=ring_blond2,
            harmonic=cavity_blond3.rf_program.harmonics,  # type: ignore # FIXME
            voltage=cavity_blond3.rf_program.effective_voltages,  # type: ignore # FIXME
            phi_rf_d=cavity_blond3.rf_program.phases,  # type: ignore # FIXME
            n_rf=len(cavity_blond3.rf_program.phases),  # type: ignore # FIXME
            section_index=0,
            omega_rf=cavity_blond3.rf_program.omegas_rf,  # type: ignore # FIXME
            phi_noise=cavity_blond3.rf_program.phase_noise,  # type: ignore # FIXME
            phi_modulation=cavity_blond3.rf_program.phase_modulation,  # type: ignore # FIXME
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
        beams: Tuple[BeamBaseClass],
        n_turns: int,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
        callback: Optional[Callable[[Simulation, Beam], None]] = None,
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
        raise NotImplementedError()
        pass  # todo

    def save_results(
        self,
        observe: Tuple[Observables, ...] = tuple(),
        common_name: Optional[str] = None,
    ) -> None:
        """
        Save the given observables to the disk

        Parameters
        ----------
        observe
            List of observables to protocol of whats happening inside
            the simulation
        common_name
            A common filename for the files/arrays to save.

        """
        for observable in observe:
            if common_name is not None:
                observable.rename(common_name=common_name)
            observable.to_disk()

    def load_results(
        self,
        beams: Tuple[BeamBaseClass],
        n_turns: Optional[int] = None,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        common_name: Optional[str] = None,
    ) -> None:
        """
        Load the given observables from the disk

        Parameters
        ----------
        beams
            Beams that are used to perform the simulation
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to start with simulation
        observe
            List of observables to protocol of whats happening inside
            the simulation
        common_name
            A common filename for the files/arrays to save.

        """
        self.finalze(
            beams=beams,
            n_turns=n_turns,
            observe=observe,
            turn_i_init=turn_i_init,
        )
        for observable in observe:
            if common_name is not None:
                observable.rename(common_name=common_name)
            observable.from_disk()
