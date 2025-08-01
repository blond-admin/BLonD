from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0

from .feedbacks.beam_feedback import Blond2BeamFeedback
from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant, DynamicParameter, Schedulable

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional as LateInit
    from typing import Optional

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
        beam_feedback: Optional[Blond2BeamFeedback],
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

        if beam_feedback is None:
            pass
        elif isinstance(beam_feedback, LocalFeedback):
            beam_feedback.set_parent_cavity(cavity=self)
        else:
            raise ValueError(beam_feedback)
        self._n_rf = n_rf
        self._local_wakefield = local_wakefield
        self._cavity_feedback = cavity_feedback
        self._beam_feedback = beam_feedback

        self._turn_i: LateInit[DynamicParameter] = None
        self._magnetic_cycle: LateInit[MagneticCycleBase] = None
        self._ring: LateInit[Ring] = None

        # TODO MOVE
        self._omega_rf: NumpyArray | None = None
        self.delta_omega_rf = backend.float(0.0)
        self._t_rf: float | None = None
        self._t_rev: float | None = None
        self.voltage: NumpyArray | None = None
        self.phi_rf: NumpyArray | None = None
        self.harmonic: NumpyArray | None = None
        self.phi_s: NumpyArray | None = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        self._turn_i = simulation.turn_i
        self._magnetic_cycle = simulation.magnetic_cycle
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

    def calc_phi_s_single_harmonic(self, beam: BeamBaseClass) -> float:
        """
        Calculates the main harmonic synchronous phase

        Parameters
        ----------
        beam
            Beam class to interact with this element

        Returns
        -------
        phi_s
            Synchronous phase for the current RF parameters, in [rad]
        """
        # TODO rewrite for efficiency
        target_total_energy = self._energy_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index,
            reference_time=beam.reference_time,
            particle_type=beam.particle_type,
        )
        reference_energy_change = target_total_energy - beam.reference_total_energy

        from blond3.acc_math.analytic.hammilton import calc_phi_s_single_harmonic

        phi_s = calc_phi_s_single_harmonic(
            charge=beam.particle_type.charge,
            voltage=self.voltage,
            phase=self.phi_rf,
            energy_gain=reference_energy_change,
            above_transition=beam.reference_gamma > self._ring.average_transition_gamma,
        )

        return phi_s

    @property  # as readonly attributes
    def n_rf(self):
        """Number of different rf waves for interaction"""
        return self._n_rf

    @abstractmethod  # pragma: no cover
    def _update_beam_based_attributes(self, beam: BeamBaseClass) -> None:
        pass

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

        # set design omega etc for this turn
        self._update_beam_based_attributes(beam=beam)

        current_turn = (
            self._turn_i.value
        )  # TODO incorrect for simulations that start later
        # Determine phase loop correction on RF phase and frequency
        if self._beam_feedback is not None and (
            current_turn >= self._beam_feedback.delay
        ):  # TODO incorrect for simulations that start later
            # domega_rf is updated later
            # this means domega_rf is effectively from last turn
            omega_increment = (
                self._beam_feedback.domega_rf  # dynamically updated by `update_domega_rf`
                * self.harmonic[:]
                / self.harmonic[0]
            )
            self.delta_omega_rf = omega_increment
        # Update the RF phase of all systems for the next turn
        # Accumulated phase offset due to beam phase loop or frequency offset
        if self.delta_omega_rf != 0:
            phi_increment = (
                2.0
                * np.pi
                * self.harmonic[:]
                * (self.delta_omega_rf)
                / self._omega_rf[:]
            )

            self.delta_phi_rf += phi_increment

        """
        # Add phase noise directly to the cavity RF phase
        if self.phi_noise is not None:
            if self.noiseFB is not None:
                self.phi_rf[:, current_turn] += \
                    self.noiseFB.x * self.phi_noise[:, current_turn]
            else:
                self.phi_rf[:, current_turn] += \
                    self.phi_noise[:, current_turn]

        # Add phase modulation directly to the cavity RF phase
        if self.phi_modulation is not None:
            self.phi_rf[:, current_turn] += \
                self.phi_modulation[0][:, current_turn]
            self.omega_rf[:, current_turn] += \
                self.phi_modulation[1][:, current_turn]
        """

        # Determine phase loop correction on RF phase and frequency
        if self._beam_feedback is not None:
            self._beam_feedback.update_domega_rf(beam=beam)  # will be applied next turn

        # Correction from cavity loop
        if self._cavity_feedback is not None:
            for feedback in self._cavity_feedback:
                if feedback is not None:
                    feedback.track()

        if self._local_wakefield is not None:
            self._local_wakefield.track(beam=beam)

    @abstractmethod  # pragma: no cover
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

    @abstractmethod  # pragma: no cover
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
        beam_feedback: Optional[Blond2BeamFeedback] = None,
        name: Optional[str] = None,
        voltage: Optional[float] = None,
        phi_rf: Optional[float] = None,
        harmonic: Optional[float] = None,
    ):
        super().__init__(
            n_rf=1,
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
            beam_feedback=beam_feedback,
            name=name,
        )
        self.voltage: float | None = voltage
        self.phi_rf: float | None = phi_rf
        self.harmonic: float | None = harmonic
        self.delta_phi_rf: NumpyArray | None = backend.float(0)

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

    def _update_beam_based_attributes(self, beam: BeamBaseClass) -> None:
        self._omega_rf = self.calc_omega(
            beam_beta=beam.reference_beta,
            ring_circumference=self._ring.circumference,
        )
        self._t_rf = (2 * np.pi) / self._omega_rf
        self._t_rev = self._t_rf * self.harmonic
        try:
            self.phi_s = self.calc_phi_s_single_harmonic(beam=beam)
        except Exception as exc:
            warnings.warn(str(exc))
            self.phi_s = np.nan

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """

        super().track(beam=beam)

        target_total_energy = self._magnetic_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index,
            reference_time=beam.reference_time,
            particle_type=beam.particle_type,
        )
        reference_energy_change = backend.float(
            target_total_energy - beam.reference_total_energy
        )
        backend.specials.kick_single_harmonic(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=backend.float(self.voltage),
            phi_rf=backend.float(self.phi_rf + self.delta_phi_rf),
            omega_rf=backend.float(self._omega_rf + self.delta_omega_rf),
            charge=backend.float(beam.particle_type.charge),  #  FIXME
            acceleration_kick=-reference_energy_change,  # Mind the minus!
        )
        beam.reference_total_energy += reference_energy_change

    def calc_omega(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> float:
        """
        Calculate angular frequency of cavity, in [rad/s]

        Parameters
        ----------
        beam_beta
            Beam reference fraction of speed of light (v/c0)
        ring_circumference
            Reference synchrotron circumference, in [m].
        Returns
        -------
        omega
            Angular frequency (2 PI f) of cavity, in [rad/s]
        """
        return self.harmonic * TWOPI_C0 * beam_beta / ring_circumference

    def voltage_waveform_tmp(self, ts: NumpyArray):
        """
        Calculate voltage of cavity for current turn

        Note
        ----
        This function is intended for small `ts` arrays
        and not executed in parallel.

        Parameters
        ----------
        ts
            Time array, in [s]
            to calculate voltage

        Returns
        -------
        voltages
            Cavity voltage in [V] at time `ts`
        """

        voltage = self.voltage
        phi_rf = self.phi_rf + self.delta_phi_rf
        omega_rf = self._omega_rf = self.delta_omega_rf
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
        ring.circumference = backend.float(circumference)

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
        beam_feedback: Optional[Blond2BeamFeedback] = None,
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
            beam_feedback=beam_feedback,
            name=name,
        )

        self.main_harmonic_idx = main_harmonic_idx

        self.voltage: Optional[NumpyArray] = None
        self.phi_rf: Optional[NumpyArray] = None
        self.harmonic: Optional[NumpyArray] = None
        self.delta_phi_rf: NumpyArray | None = np.zeros(1, dtype=np.float64)

        self._t_rf: NumpyArray | None = None
        self._t_rev: float | None = None

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

    def _update_beam_based_attributes(self, beam: BeamBaseClass) -> None:
        self._omega_rf = self.calc_omega(
            beam_beta=beam.reference_beta,
            ring_circumference=self._ring.circumference,
        )

        self._t_rf = (2 * np.pi) / self._omega_rf
        self._t_rev = self._t_rf[0] * self.harmonic[0]
        try:
            self.phi_s = self.calc_phi_s_single_harmonic(beam=beam)
        except Exception as exc:
            warnings.warn(str(exc))
            self.phi_s = np.nan

    def calc_omega(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> NumpyArray:
        """
        Calculate angular frequency of cavity in [rad/s]

        Parameters
        ----------
        beam_beta
            Beam reference fraction of speed of light (v/c0)

        ring_circumference
            Reference synchrotron circumference, in [m].

        Returns
        -------
        omega
            Angular frequency (2 PI f) of cavity in [rad/s]
        """
        return self.harmonic * TWOPI_C0 * beam_beta / ring_circumference

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
        voltage = self.voltage[0] * np.sin(
            self._omega_rf_effective[0] * ts + self.phi_rf[0] + self.delta_phi_rf[0]
        )
        for i in range(1, len(self.voltage)):
            voltage += self.voltage[i] * np.sin(
                self._omega_rf_effective[i] * ts + self.phi_rf[i] + self.delta_phi_rf[i]
            )

    @staticmethod
    def headless(
        section_index: int,
        voltage: NumpyArray,
        phi_rf: NumpyArray,
        harmonic: NumpyArray,
        circumference: float,
        total_energy: float,
        main_harmonic_idx: float,
        local_wakefield: Optional[WakeField] = None,
        cavity_feedback: Optional[LocalFeedback] = None,
        beam_feedback: Optional[Blond2BeamFeedback] = None,
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
            beam_feedback=beam_feedback,
            main_harmonic_idx=main_harmonic_idx,
        )

        mhc.voltage = voltage
        mhc.phi_rf = phi_rf
        mhc.harmonic = harmonic

        ring = Mock(Ring)
        ring.circumference = circumference

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
            main_harmonic_idx=main_harmonic_idx,
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
        target_total_energy = self._magnetic_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index,
            reference_time=beam.reference_time,
            particle_type=beam.particle_type,
        )
        reference_energy_change = target_total_energy - beam.reference_total_energy

        backend.specials.kick_multi_harmonic(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=self.voltage,
            phi_rf=self.phi_rf + self.delta_phi_rf,
            omega_rf=self._omega_rf + self.delta_omega_rf,
            charge=backend.float(beam.particle_type.charge),  # FIXME
            n_rf=self.n_rf,
            acceleration_kick=-reference_energy_change,  # Mind the minus!
        )
        beam.reference_total_energy += reference_energy_change
