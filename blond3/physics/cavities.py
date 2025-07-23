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
    from typing import Optional, Type

    from numpy.typing import NDArray as NumpyArray

    from .impedances.base import WakeField
    from .feedbacks.base import LocalFeedback
    from .. import Ring
    from .._core.beam.base import BeamBaseClass
    from .._core.simulation.simulation import Simulation
    from ..cycles.magnetic_cycle import MagneticCycleBase

TWOPI_C0 = 2.0 * np.pi * c0


class CavityBaseClass(BeamPhysicsRelevant, Schedulable, ABC):
    def __init__(
        self,
        n_rf: int,
        section_index: int,
        local_wakefield: Optional[WakeField],
        cavity_feedback: Optional[LocalFeedback],
        name: Optional[str] = None,
    ):
        """
        Base class to implement beam-rf interactions in synchrotrons

        Parameters
        ----------
        n_rf
            Number of different rf waves for interaction
        section_index
            Section index to group elements into sections
        local_wakefield
            Optional wakefield to interact with beam
        cavity_feedback
            Optional cavity feedback to change cavity parameters
        """
        from .feedbacks.base import LocalFeedback  # prevent cyclic import

        super().__init__(section_index=section_index, name=name)
        if cavity_feedback is None:
            pass
        elif isinstance(cavity_feedback, LocalFeedback):
            cavity_feedback.set_parent_cavity(cavity=self)
        else:
            raise ValueError(cavity_feedback)

        self._n_rf = n_rf
        self._local_wakefield = local_wakefield
        self._cavity_feedback = cavity_feedback

        self._turn_i: LateInit[DynamicParameter] = None
        self._energy_cycle: LateInit[MagneticCycleBase] = None
        self._ring: LateInit[Ring] = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        self._turn_i = simulation.turn_i
        self._energy_cycle = simulation.magnetic_cycle
        self._ring = simulation.ring

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
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

    @property  # as readonly attributes
    def n_rf(self):
        """Number of different rf waves for interaction"""
        return self._n_rf

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """

        self.apply_schedules(
            turn_i=self._turn_i.value,
            reference_time=beam.reference_time,
        )
        if self._cavity_feedback is not None:
            self._cavity_feedback.track(beam=beam)
        if self._local_wakefield is not None:
            self._local_wakefield.track(beam=beam)

    @abstractmethod
    def voltage_waveform_tmp(self, ts: NumpyArray):
        """
        Calculate voltage of cavity for current turn

        Parameters
        ----------
        ts
            Time array, in [s]
            to calculate voltage
        """
        pass

    @abstractmethod
    def calc_omega(
        self,
        beam_beta: float,
        closed_orbit_length: float,
    ):
        """
        Calculate angular frequency of cavity, in [rad/s]

        Parameters
        ----------
        beam_beta
            Beam reference fraction of speed of light (v/c0)
        closed_orbit_length
            Length of the closed orbit, in [m]
        Returns
        -------
        omega
            Angular frequency (2 PI f) of cavity, in [rad/s]
        """
        pass


class SingleHarmonicCavity(CavityBaseClass):
    """
    Cavity with only one RF wave for beam interaction

    Parameters
    ----------
    section_index
        Section index to group elements into sections
    local_wakefield
        Optional wakefield to interact with beam
    cavity_feedback
        Optional cavity feedback to change cavity parameters

    Attributes
    ----------
    voltage
        Cavity's effective voltage, in [V]
    phi_rf
        Cavity's design phase, in [deg]
    harmonic
        Cavity's design harmonic []
    """

    def __init__(
        self,
        section_index: int = 0,
        local_wakefield: Optional[WakeField] = None,
        cavity_feedback: Optional[LocalFeedback] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            n_rf=1,
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
            name=name,
        )
        self.voltage: float | None = None
        self.phi_rf: float | None = None
        self.harmonic: float | None = None
        self._omega = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
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

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """

        # order matters. set _omega correctly
        # for feedbacks etc. in super()
        self._omega = self.calc_omega(
            beam_beta=beam.reference_beta,
            closed_orbit_length=self._ring.closed_orbit_length,
        )

        super().track(beam=beam)

        target_total_energy = self._energy_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index,
            reference_time=beam.reference_time,
            particle_type=beam.particle_type,
        )
        reference_energy_change = target_total_energy - beam.reference_total_energy
        backend.specials.kick_single_harmonic(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=self.voltage,
            phi_rf=self.phi_rf,
            omega_rf=self._omega,
            charge=backend.float(beam.particle_type.charge),  #  FIXME
            acceleration_kick=-reference_energy_change,  # Mind the minus!
        )
        beam.reference_total_energy += reference_energy_change

    def calc_omega(
        self,
        beam_beta: float,
        closed_orbit_length: float,
    ) -> float:
        """
        Calculate angular frequency of cavity, in [rad/s]

        Parameters
        ----------
        beam_beta
            Beam reference fraction of speed of light (v/c0)
        closed_orbit_length
            Length of the closed orbit, in [m]
        Returns
        -------
        omega
            Angular frequency (2 PI f) of cavity, in [rad/s]
        """
        return self.harmonic * TWOPI_C0 * beam_beta / closed_orbit_length

    def voltage_waveform_tmp(self, ts: NumpyArray):
        """
        Calculate voltage of cavity for current turn

        Note
        ----
        This function is intended for small ts arrays
        and not executed in parallel.

        Parameters
        ----------
        ts
            Time array, in [s]
            to calculate voltage
        """

        voltage = self.voltage
        phi_rf = self.phi_rf
        omega_rf = self._omega
        return voltage * np.sin(omega_rf * ts + phi_rf)

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
    ) -> SingleHarmonicCavity:
        """
        Initialize object without simulation context

        Parameters
        ----------
        section_index
            Section index to group elements into sections
        voltage
            Cavity's effective voltage in [V]
        phi_rf
            Cavity's design phase in [deg]
        harmonic
            Cavity's design harmonic []
        circumference
            Synchrotron circumference in [m]
        total_energy
            Target total energy in [eV]
        local_wakefield
            Optional wakefield to interact with beam
        cavity_feedback
            Optional cavity feedback to change cavity parameters

        Returns
        -------
        single_harmonic_cavity
        """
        from .._core.simulation.simulation import Simulation
        from .._core.ring.ring import Ring
        from ..cycles.magnetic_cycle import ConstantMagneticCycle
        from .._core.beam.base import BeamBaseClass

        mhc = SingleHarmonicCavity(
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
        )

        mhc.voltage = backend.float(voltage)
        mhc.phi_rf = backend.float(phi_rf)
        mhc.harmonic = backend.float(harmonic)

        ring = Mock(Ring)
        ring.effective_circumference = backend.float(circumference)

        energy_cycle = Mock(ConstantMagneticCycle)
        energy_cycle.get_target_total_energy.return_value = total_energy

        simulation = Mock(Simulation)
        simulation.ring = ring
        simulation.magnetic_cycle = energy_cycle
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0

        mhc.on_init_simulation(simulation=simulation)
        mhc.on_run_simulation(
            simulation=simulation,
            n_turns=1,
            turn_i_init=simulation.turn_i.value,
            beam=Mock(BeamBaseClass),
        )
        return mhc


class MultiHarmonicCavity(CavityBaseClass):
    """
    Cavity with several RF wave for beam interaction

    Parameters
    ----------
    n_harmonics
        Number of different RF waves for interaction
    main_harmonic_idx
        Index of the cavity's main harmonic
        Used to calculate attributes that rely on only one harmonic
    section_index
        Section index to group elements into sections
    local_wakefield
        Optional wakefield to interact with beam
    cavity_feedback
        Optional cavity feedback to change cavity parameters

    Attributes
    ----------
    voltage
        Cavity's effective voltages (per harmonic) in [V]
    phi_rf
        Cavity's design phases (per harmonic) in [deg]
    harmonic
        Cavity's design harmonics (per harmonic) []
    """

    def __init__(
        self,
        n_harmonics: int,
        main_harmonic_idx: int,
        section_index: int = 0,
        local_wakefield: Optional[WakeField] = None,
        cavity_feedback: Optional[LocalFeedback] = None,
        name: Optional[str] = None,
    ):
        assert (
            main_harmonic_idx < n_harmonics
        ), f"{n_harmonics=}, but {main_harmonic_idx=}."

        super().__init__(
            n_rf=n_harmonics,
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
            name=name,
        )

        self.main_harmonic_idx = main_harmonic_idx

        self.voltage: Optional[NumpyArray] = None
        self.phi_rf: Optional[NumpyArray] = None
        self.harmonic: Optional[NumpyArray] = None
        self._omega: Optional[NumpyArray] = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
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

    def calc_omega(
        self,
        beam_beta: float,
        closed_orbit_length: float,
    ) -> NumpyArray:
        """
        Calculate angular frequency of cavity in [rad/s]

        Parameters
        ----------
        beam_beta
            Beam reference fraction of speed of light (v/c0)

        closed_orbit_length
            Length of the closed orbit, in [m]

        Returns
        -------
        omega
            Angular frequency (2 PI f) of cavity in [rad/s]
        """
        return self.harmonic * TWOPI_C0 * beam_beta / closed_orbit_length

    def voltage_waveform_tmp(self, ts: NumpyArray):
        """
        Calculate voltage of cavity for current turn

        Note
        ----
        This function is intended for small ts arrays
        and not executed in parallel.

        Parameters
        ----------
        ts
            Time array, in [s]
            to calculate voltage
        """
        raise NotImplementedError
        voltage = self.voltage[0] * np.sin(self._omega[0] * ts + self.phi_rf[0])
        for i in range(1, len(self.voltage)):
            voltage += self.voltage[i] * np.sin(self._omega[i] * ts + self.phi_rf[i])

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
    ) -> MultiHarmonicCavity:
        """
        Initialize object without simulation context

        Parameters
        ----------
        section_index
            Section index to group elements into sections
        voltage
            Cavity's effective voltages (per harmonic) in [V]
        phi_rf
            Cavity's design phases (per harmonic) in [deg]
        harmonic
            Cavity's design harmonics (per harmonic) []
        circumference
            Synchrotron circumference in [m]
        total_energy
            Target total energy in [eV]
        local_wakefield
            Optional wakefield to interact with beam
        cavity_feedback
            Optional cavity feedback to change cavity parameters

        Returns
        -------
        multi_harmonic_cavity
        """
        from .._core.simulation.simulation import Simulation
        from .._core.ring.ring import Ring
        from ..cycles.magnetic_cycle import ConstantMagneticCycle
        from .._core.beam.base import BeamBaseClass

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
        ring.effective_circumference = circumference

        energy_cycle = Mock(ConstantMagneticCycle)
        energy_cycle.get_target_total_energy.return_value = total_energy

        simulation = Mock(Simulation)
        simulation.ring = ring
        simulation.magnetic_cycle = energy_cycle
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0

        mhc.on_init_simulation(simulation=simulation)
        mhc.on_run_simulation(
            simulation=simulation,
            n_turns=1,
            turn_i_init=simulation.turn_i.value,
            beam=Mock(BeamBaseClass),
        )
        return mhc

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        super().track(beam=beam)
        target_total_energy = self._energy_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index,
            reference_time=beam.reference_time,
            particle_type=beam.particle_type,
        )
        reference_energy_change = target_total_energy - beam.reference_total_energy

        omega_rf = self.calc_omega(
            beam_beta=beam.reference_beta,
            closed_orbit_length=self._ring.closed_orbit_length,
        )
        self._omega = omega_rf

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
