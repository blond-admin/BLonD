from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0

from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant, DynamicParameter, Schedulable

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional as LateInit
    from typing import (
        Optional,
    )

    from numpy.typing import NDArray as NumpyArray

    from .impedances.base import WakeField
    from .feedbacks.base import LocalFeedback
    from .. import Ring
    from .._core.beam.base import BeamBaseClass
    from .._core.simulation.simulation import Simulation
    from ..cycles.energy_cycle import EnergyCycleBase

TWOPIC0 = 2.0 * np.pi * c0


class CavityBaseClass(BeamPhysicsRelevant, Schedulable, ABC):
    def __init__(
        self,
        n_rf: int,
        section_index: int,
        local_wakefield: Optional[WakeField],
        cavity_feedback: Optional[LocalFeedback],
    ):
        super().__init__(section_index=section_index)
        if cavity_feedback is not None:
            cavity_feedback.set_owner(cavity=self)

        self._n_rf = n_rf
        self._local_wakefield = local_wakefield
        self._cavity_feedback = cavity_feedback

        self._turn_i: LateInit[DynamicParameter] = None
        self._energy_cycle: LateInit[EnergyCycleBase] = None
        self._ring: LateInit[Ring] = None

    @staticmethod
    def headless(
        n_rf: int,
        section_index: int,
        local_wakefield: Optional[WakeField],
        cavity_feedback: Optional[LocalFeedback],
    ):
        cav = CavityBaseClass(
            n_rf=n_rf,
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
        )
        from .._core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0
        cav.on_init_simulation(simulation=simulation)
        cav.on_run_simulation(
            simulation=simulation, n_turns=1, turn_i_init=simulation.turn_i.value
        )

        return cav

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._turn_i = simulation.turn_i
        self._energy_cycle = simulation.energy_cycle
        self._ring = simulation.ring

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass

    @property  # as readonly attributes
    def n_rf(self):
        return self._n_rf

    def track(self, beam: BeamBaseClass):
        self.apply_schedules(
            turn_i=self._turn_i.value,
            reference_time=beam.reference_time,
        )
        if self._cavity_feedback is not None:
            self._cavity_feedback.track(beam=beam)
        if self._local_wakefield is not None:
            self._local_wakefield.track(beam=beam)

    @abstractmethod
    def calc_omega(self, beam_beta: float, ring_circumference: float):
        pass


class SingleHarmonicCavity(CavityBaseClass):
    def __init__(
        self,
        section_index: int = 0,
        local_wakefield: Optional[WakeField] = None,
        cavity_feedback: Optional[LocalFeedback] = None,
    ):
        super().__init__(
            n_rf=1,
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
        )
        self.voltage: float | None = None
        self.phi_rf: float | None = None
        self.harmonic: float | None = None
        self.total_energy_target: float | None = None
        self._omegas = None

    def calc_omega(self, beam_beta: float, ring_circumference: float):
        return TWOPIC0 * self.harmonic * beam_beta / ring_circumference

    def on_init_simulation(self, simulation: Simulation) -> None:
        super().on_init_simulation(simulation=simulation)
        if (self.voltage is None) and "voltage" not in self.schedules.keys():
            raise ValueError(
                "You need to define `voltage` via `.voltage=...` "
                "or `.schedule(attribute='voltage', value=...)`"
            )
        if (self.phi_rf is None) and "phi_rf" not in self.schedules.keys():
            raise ValueError(
                "You need to define `phi_rf` via `.phi_rf=...` "
                "or `.schedule(attribute='phi_rf', value=...)`"
            )
        if (self.harmonic is None) and "harmonic" not in self.schedules.keys():
            raise ValueError(
                "You need to define `harmonic` via `.harmonic=...` "
                "or `.schedule(attribute='harmonic', value=...)`"
            )

    def track(self, beam: BeamBaseClass):
        super().track(beam=beam)
        reference_energy_change = (
            self._energy_cycle.total_energy[self.section_index, self._turn_i.value]
            - beam.reference_total_energy
        )
        self._omegas = self.calc_omega(
            beam_beta=beam.reference_beta,
            ring_circumference=self._ring.circumference,
        )
        backend.specials.kick_single_harmonic(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=self.voltage,
            phi_rf=self.phi_rf,
            omega_rf=self._omegas,
            charge=backend.float(beam.particle_type.charge),  #  FIXME
            acceleration_kick=-reference_energy_change,  # Mind the minus!
        )
        beam.reference_total_energy += reference_energy_change

    @staticmethod
    def headless(
        section_index: int,
        voltage: float,
        phi_rf: float,
        harmonic: float,
        circumference: float,
        total_energy: float,
        local_wakefield: Optional[WakeField] = None,
        cavity_feedback: Optional[LocalFeedback] = None,
    ):
        from .._core.simulation.simulation import Simulation
        from .._core.ring.ring import Ring
        from ..cycles.energy_cycle import EnergyCycleBase

        mhc = SingleHarmonicCavity(
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
        )

        mhc.voltage = backend.float(voltage)
        mhc.phi_rf = backend.float(phi_rf)
        mhc.harmonic = backend.float(harmonic)

        ring = Mock(Ring)
        ring.circumference = backend.float(circumference)

        energy_cycle = Mock(EnergyCycleBase)
        energy_cycle.total_energy = total_energy * np.ones(
            (section_index + 1, 1), dtype=backend.float
        )

        simulation = Mock(Simulation)
        simulation.ring = ring
        simulation.energy_cycle = energy_cycle
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0

        mhc.on_init_simulation(simulation=simulation)
        mhc.on_run_simulation(
            simulation=simulation, n_turns=1, turn_i_init=simulation.turn_i.value
        )
        return mhc


class MultiHarmonicCavity(CavityBaseClass):
    def __init__(
        self,
        n_harmonics: int,
        section_index: int = 0,
        local_wakefield: Optional[WakeField] = None,
        cavity_feedback: Optional[LocalFeedback] = None,
    ):
        super().__init__(
            n_rf=n_harmonics,
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
        )
        self.voltage: NumpyArray | None = None
        self.phi_rf: NumpyArray | None = None
        self.harmonic: NumpyArray | None = None
        self._omegas: NumpyArray | None = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        super().on_init_simulation(simulation=simulation)
        if (self.voltage is None) and "voltage" not in self.schedules.keys():
            raise ValueError(
                f"You need to define `voltage` for '{self.name}' via "
                f"`.voltage=...` or `.schedule(attribute='voltage', value=...)`"
            )
        if (self.phi_rf is None) and "phi_rf" not in self.schedules.keys():
            raise ValueError(
                f"You need to define `phi_rf` for '{self.name}' via "
                f"`.phi_rf=...` or `.schedule(attribute='phi_rf', value=...)`"
            )
        if (self.harmonic is None) and "harmonic" not in self.schedules.keys():
            raise ValueError(
                f"You need to define `harmonic` for '{self.name}' via "
                f"`.harmonic=...` or `.schedule(attribute='harmonic', value=...)`"
            )

    @staticmethod
    def headless(
        section_index: int,
        voltage: NumpyArray,
        phi_rf: NumpyArray,
        harmonic: NumpyArray,
        circumference: float,
        total_energy: float,
        local_wakefield: Optional[WakeField] = None,
        cavity_feedback: Optional[LocalFeedback] = None,
    ):
        from .._core.simulation.simulation import Simulation
        from .._core.ring.ring import Ring
        from ..cycles.energy_cycle import EnergyCycleBase

        mhc = MultiHarmonicCavity(
            n_harmonics=len(voltage),
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
        )

        mhc.voltage = voltage
        mhc.phi_rf = phi_rf
        mhc.harmonic = harmonic

        ring = Mock(Ring)
        ring.circumference = circumference

        energy_cycle = Mock(EnergyCycleBase)
        energy_cycle.total_energy = total_energy * np.ones((section_index + 1, 1))

        simulation = Mock(Simulation)
        simulation.ring = ring
        simulation.energy_cycle = energy_cycle
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0

        mhc.on_init_simulation(simulation=simulation)
        mhc.on_run_simulation(
            simulation=simulation, n_turns=1, turn_i_init=simulation.turn_i.value
        )
        return mhc

    def calc_omega(self, beam_beta: float, ring_circumference: float):
        return self.harmonic * (TWOPIC0 * beam_beta / ring_circumference)

    def track(self, beam: BeamBaseClass):
        super().track(beam=beam)
        reference_energy_change = (
            self._energy_cycle.total_energy[self.section_index, self._turn_i.value]
            - beam.reference_total_energy
        )

        omega_rf = self.calc_omega(
            beam_beta=beam.reference_beta,
            ring_circumference=self._ring.circumference,
        )
        self._omegas = omega_rf

        backend.specials.kick_multi_harmonic(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=self.voltage,
            phi_rf=self.phi_rf,
            omega_rf=omega_rf,
            charge=backend.float(beam.particle_type.charge),  # FIXME
            n_rf=self.n_rf,
            acceleration_kick=-reference_energy_change,  # Mind the minus!
        )
        beam.reference_total_energy += reference_energy_change
