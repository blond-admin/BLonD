from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from ..base import BeamPhysicsRelevant, Preparable
from ..base import DynamicParameter
from ..helpers import find_instances_with_method
from ..ring.helpers import get_elements, get_init_order
from ...cycles.energy_cycle import EnergyCycle
from ...physics.drifts import DriftBaseClass
from ...physics.profiles import ProfileBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from typing import (
        Optional,
        Tuple,
    )
    from numpy.typing import NDArray as NumpyArray
    from ..beam.base import BeamBaseClass
    from ...beam_preparation.base import MatchingRoutine

    from ..ring.ring import Ring
    from ...handle_results.observables import Observables


class Simulation(Preparable):
    def __init__(
        self,
        ring: Ring,
        beams: Tuple[BeamBaseClass, ...],
        energy_cycle: NumpyArray | EnergyCycle,
    ):
        super().__init__()
        self._ring: Ring = ring
        assert beams != (), f"{beams=}"
        assert len(beams) <= 2, "Maximum two beams allowed"

        self._beams: Tuple[BeamBaseClass, ...] = beams

        if isinstance(energy_cycle, np.ndarray):
            energy_cycle = EnergyCycle(energy_cycle)
        self._energy_cycle: EnergyCycle = energy_cycle

        self.turn_i = DynamicParameter(None)
        self.section_i = DynamicParameter(None)

        self._exec_on_init_simulation()

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass

    def _exec_all_in_tree(self, method: str, **kwargs):
        instances = find_instances_with_method(self, f"{method}")
        for instance in instances:
            print(type(instance), instance)
        print()
        ordered_classes = get_init_order(instances, f"{method}.requires")
        for class_ in ordered_classes:
            print(class_)

        classes_check = set()
        for ins in instances:
            classes_check.add(type(ins))
        assert len(classes_check) == len(ordered_classes), "BUG"
        ordered_classes.pop(ordered_classes.index("ABCMeta"))

        for cls in ordered_classes:
            for element in instances:
                if not type(element).__name__ == cls:
                    continue
                print(element, kwargs)
                getattr(element, method)(**kwargs)

    def _exec_on_init_simulation(self):
        print("_exec_on_init_simulation")

        self._exec_all_in_tree("on_init_simulation", simulation=self)

    def _exec_on_run_simulation(self, n_turns: int, turn_i_init: int):
        print("_exec_on_run_simulation")

        self._exec_all_in_tree(
            "on_run_simulation",
            simulation=self,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )

    @staticmethod
    def from_locals(locals: dict):
        from ..beam.base import BeamBaseClass  # prevent cyclic import
        from ..ring.ring import Ring  # prevent cyclic import

        locals_list = locals.values()
        _rings = get_elements(locals_list, Ring)
        assert len(_rings) == 1, f"Found {len(_rings)} rings"
        ring = _rings[0]

        beams = get_elements(locals_list, BeamBaseClass)

        _energy_cycles = get_elements(locals_list, EnergyCycle)
        assert len(_energy_cycles) == 1, f"Found {len(_energy_cycles)} energy cycles"
        energy_cycle = _energy_cycles[0]

        elements = get_elements(locals_list, BeamPhysicsRelevant)
        ring.add_elements(elements=elements, reorder=True)

        sim = Simulation(ring=ring, beams=beams, energy_cycle=energy_cycle)
        return sim

    @property  # as readonly attributes
    def ring(self):
        return self._ring

    @property  # as readonly attributes
    def beams(self):
        return self._beams

    @property  # as readonly attributes
    def energy_cycle(self) -> EnergyCycle:
        return self._energy_cycle

    @cached_property
    def get_separatrix(self):
        return None

    @cached_property
    def get_potential_well(self):
        return None

    @cached_property
    def get_hash(self):
        return None

    def print_one_turn_execution_order(self):
        self._ring.elements.print_order()

    def invalidate_cache(
        self,
        # turn i needed to be
        # compatible with subscription
        turn_i: int,
    ):
        self.__dict__.pop("get_separatrix", None)
        self.__dict__.pop("get_potential_well", None)
        self.__dict__.pop("get_hash", None)

    def on_prepare_beam(self, preparation_routine: MatchingRoutine, turn_i: int = 0):
        print("on_prepare_beam")
        self.turn_i.value = turn_i
        preparation_routine.on_prepare_beam(simulation=self)

    def run_simulation(
        self,
        n_turns: Optional[int] = None,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
    ) -> None:
        self._exec_on_run_simulation(
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )
        if len(self._beams) == 1:
            self._run_simulation_single_beam(
                n_turns=n_turns,
                turn_i_init=turn_i_init,
                observe=observe,
                show_progressbar=show_progressbar,
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
            )

    def _run_simulation_single_beam(
        self,
        n_turns: int,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
    ) -> None:
        iterator = range(turn_i_init, turn_i_init + n_turns)
        if show_progressbar:
            iterator = tqdm(iterator)  # Add TQDM display to iteration
        self.turn_i.on_change(self.invalidate_cache)
        for turn_i in iterator:
            self.turn_i.value = turn_i
            for element in self._ring.elements.elements:
                self.section_i.current_group = element.section_index
                if element.is_active_this_turn(turn_i=self.turn_i.value):
                    element.track(self._beams[0])
            for observable in observe:
                if observable.is_active_this_turn(turn_i=self.turn_i.value):
                    observable.update(simulation=self)

        # reset counters to uninitialized again
        self.turn_i.value = None
        self.section_i.value = None

    def get_legacy_map(self):
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
            n_sections=self.ring.elements.count(CavityBaseClass),
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
    ) -> None:
        pass  # todo

    def load_results(
        self,
        n_turns: int,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
    ) -> SimulationResults:
        raise FileNotFoundError()
        return
