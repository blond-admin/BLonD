"""Collection of implementations to handle lumped RF cavities in synchrotrons.

Authors
-------
Simon Lauber
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0

from blond.experimental.physics.feedbacks.beam_feedback import (
    BeamFeedbackBase,
    Blond2BeamFeedback,
)

from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant, DynamicParameter, Schedulable

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from .. import Ring
    from .._core.beam.base import BeamBaseClass
    from .._core.simulation.simulation import Simulation
    from ..cycles.magnetic_cycle import MagneticCycleBase
    from ..experimental.physics.feedbacks.base import LocalFeedback
    from .impedances.base import WakeField

TWOPI_C0 = 2.0 * np.pi * c0


class RfManipulationBaseClass(BeamPhysicsRelevant, Schedulable, ABC):
    """Base class to implement beam-rf any interactions in synchrotrons.

    This class is intended to come with barely any feature to host all
    beam-rf interactions, whereas `RfStationBaseClass` has already several
    class methods to group `SingleHarmonicRfStation`, and `MultiHarmonicRfStation`.

    Parameters
    ----------
    section_index
        Section index to group elements into sections
    name
        User given name of the element
    """

    def __init__(
        self,
        section_index: int,
        name: str | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ):
        super().__init__(
            section_index=section_index,
            name=name,
            **kwargs,  # for MRO of fused elements
        )
        self._turn_i: DynamicParameter | None = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)

        self._turn_i = simulation.turn_i

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        super().track(beam=beam)
        assert self._turn_i is not None
        if self.schedule_active:
            self.apply_schedules(
                turn_i=self._turn_i.value,
                reference_time=float(beam.reference_time),
            )


class RfStationBaseClass(RfManipulationBaseClass, Schedulable, ABC):
    """Base class to implement beam-rf interactions in synchrotrons.

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

    def __init__(
        self,
        n_rf: int,
        section_index: int,
        local_wakefield: WakeField | None,
        cavity_feedback: LocalFeedback | None,
        beam_feedback: Blond2BeamFeedback | None,
        name: str | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ):
        from blond.experimental.physics.feedbacks.base import LocalFeedback

        # prevent cyclic import

        super().__init__(
            section_index=section_index,
            name=name,
            **kwargs,  # for MRO of fused elements
        )
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

        self._magnetic_cycle: MagneticCycleBase | None = None
        self._ring: Ring | None = None

        # TODO MOVE
        self._omega_rf_design: NumpyArray | float | None = None
        self.delta_omega_rf = 0.0
        self._phi_rf_design: NumpyArray | float | None = None
        self.delta_phi_rf = 0.0
        self._t_rf: float | None = None
        self._t_rev: float | None = None
        self.voltage: NumpyArray | None = None
        self.harmonic: NumpyArray | None = None
        self.phi_s: NumpyArray | None = None
        self.omega_s0: NumpyArray | None = None

    @property
    def omega_rf(self):
        """RF angular frequency."""
        return self._omega_rf_design + self.delta_omega_rf

    @omega_rf.setter
    def omega_rf(self, value: float | NumpyArray):
        """Setting RF angular frequency."""
        self.delta_omega_rf = value - self._omega_rf_design

    @property
    def phi_rf(self):
        """RF phase."""
        return self._phi_rf_design + self.delta_phi_rf

    @phi_rf.setter
    def phi_rf(self, value: float | NumpyArray):
        """Setting RF angular frequency."""
        self.delta_phi_rf = value - self._phi_rf_design

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)
        self._magnetic_cycle = simulation.magnetic_cycle
        self._ring = simulation.ring

        if (self._cavity_feedback is not None) and (
            not hasattr(self._cavity_feedback, "__iter__")
        ):
            self._cavity_feedback = [self._cavity_feedback]

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called.

        simulation
            Simulation context manager
        beam
            Simulation `Beam` object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        # set design omega etc. for this turn
        self._update_beam_based_attributes(beam=beam)

    @abstractmethod  # pragma: no cover
    def get_main_harmonic(self) -> float:
        """Returns the harmonic number of the main harmonic."""
        pass

    @abstractmethod  # pragma: no cover
    def get_main_harmonic_voltage(self) -> float:
        """Returns the voltage of the main harmonic, in [V]."""
        pass

    @abstractmethod  # pragma: no cover
    def get_main_harmonic_phi_rf(self) -> float:
        """Returns the phi_rf of the main harmonic, in [rad]."""
        pass

    @abstractmethod  # pragma: no cover
    def get_main_harmonic_omega_rf_design(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> float:
        """Returns the omega_rf of the main harmonic, in [rad/s]."""
        pass

    def attach_beam_feedback(self, beam_feedback: BeamFeedbackBase):
        """Attach beam feedback to the RF station after initialization."""
        self._beam_feedback = beam_feedback

    def attach_cavity_feedback(self, cavity_feedback: LocalFeedback):
        """Attach cavity feedback to the RF station after initialization."""
        cavity_feedback.set_parent_cavity(cavity=self)
        self._cavity_feedback = cavity_feedback

    def calc_synchrotron_tune_single_harmonic(
        self,
        beam: BeamBaseClass,
        phi_s: float | None = None,
        eta_0: float | None = None,
    ):
        """Function calculating the turn-by-turn synchrotron tune.

        The calculation assumes a single-harmonic RF system and no intensity
        effects.

        Parameters
        ----------
        beam
            Beam class to interact with this element

        Returns
        -------
        Q_s
            Synchrotron tune.

        """
        if eta_0 is None:
            assert self._ring is not None
            eta_0 = self._ring.calc_average_eta_0(beam.reference_gamma)

        if phi_s is None:
            phi_s = self.calc_phi_s_single_harmonic(beam)

        from blond.acc_math.analytic.hamilton import (
            calc_synchrotron_tune_single_harmonic,
        )

        assert self.voltage is not None

        Q_s0 = calc_synchrotron_tune_single_harmonic(
            charge=beam.particle_type.charge,
            voltage=float(self.voltage),
            beta=beam.reference_beta,
            energy=beam.reference_total_energy,
            phi_s=phi_s,
            harmonic=self.harmonic,
            eta_0=eta_0,
        )

        return Q_s0

    def calc_phi_s_single_harmonic(
        self, beam: BeamBaseClass, enable_rf_phase: bool = True
    ) -> float:
        """Calculates the main harmonic synchronous phase.

        Parameters
        ----------
        beam
            Beam class to interact with this element

        Returns
        -------
        phi_s
            Synchronous phase for the current RF parameters, in [rad]
        """
        assert self._magnetic_cycle is not None
        assert self._turn_i is not None
        assert self._ring is not None
        # TODO rewrite for efficiency
        target_total_energy = self._magnetic_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index
            if not beam.is_counter_rotating
            else len(self._ring.section_lengths) - self.section_index - 1,
            reference_time=float(beam.reference_time),
            particle_type=beam.particle_type,
        )
        reference_energy_change = (
            target_total_energy - beam.reference_total_energy
        )

        from blond.acc_math.analytic.hamilton import (
            calc_phi_s_single_harmonic,
        )

        assert self.voltage is not None
        assert self.phi_rf is not None
        phi_s = calc_phi_s_single_harmonic(
            charge=beam.particle_type.charge,
            voltage=float(self.voltage[0]),  # TODO change indexing
            phase=float(self.phi_rf[0]) * int(enable_rf_phase),
            energy_gain=reference_energy_change,
            above_transition=beam.reference_gamma
            > self._ring.average_transition_gamma,
        )

        return phi_s

    @property  # as readonly attributes
    def n_rf(self) -> int:
        """Number of different rf waves for interaction."""
        return self._n_rf

    @abstractmethod  # pragma: no cover
    def _update_beam_based_attributes(self, beam: BeamBaseClass) -> None:
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        super().track(beam=beam)

        # set design omega etc. for this turn
        self._update_beam_based_attributes(beam=beam)

        # TODO incorrect for simulations that start later
        # Determine phase loop correction on RF phase and frequency
        """
        if self._beam_feedback is not None and (
            self._turn_i.value >= self._beam_feedback.delay
        ):  # TODO incorrect for simulations that start later
            # domega_rf is updated later
            # this means domega_rf is effectively from last turn
            assert self.harmonic is not None
            omega_increment = (
                self._beam_feedback.domega_rf  # dynamically updated by `update_domega_rf`
                * self.harmonic[:]
                / self.harmonic[0]
            )
            self.delta_omega_rf = omega_increment
        # Update the RF phase of all systems for the next turn
        # Accumulated phase offset due to beam phase loop or frequency offset
        if self.delta_omega_rf != 0:
            assert self.harmonic is not None
            assert self._omega_rf is not None
            phi_increment = (
                2.0
                * np.pi
                * self.harmonic[:]
                * (self.delta_omega_rf)
                / self._omega_rf[:]
            )

            self.delta_phi_rf += phi_increment
        """

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
        # if self._beam_feedback is not None:
        #    self._beam_feedback.track(beam=beam)

        # Correction from cavity loop
        if self._cavity_feedback is not None:
            for feedback in self._cavity_feedback:
                if feedback is not None:
                    feedback.track(beam=beam)

        if self._local_wakefield is not None:
            self._local_wakefield.track(beam=beam)

    @abstractmethod  # pragma: no cover
    def voltage_waveform_tmp(self, ts: NumpyArray):
        """Calculate voltage of cavity for current turn.

        Parameters
        ----------
        ts
            Time array, in [s]
            to calculate voltage
        """
        pass

    @abstractmethod  # pragma: no cover
    def calc_omega_rf_design(
        self,
        beam_beta: float,
        closed_orbit_length: float,
    ):
        """Calculate angular frequency of cavity, in [rad/s].

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

    def info_string(self, prefix="") -> str:
        """Inform that the feedback/wakefield is also executed within the track method."""
        content = ""
        if self._cavity_feedback is not None:
            content += (
                f"{self._cavity_feedback.info_string(prefix=prefix + ' ↓ ')}\n"
            )

        if self._local_wakefield is not None:
            content += (
                f"{self._local_wakefield.info_string(prefix=prefix + ' ↓ ')}\n"
            )
        content += f"{super().info_string(prefix=prefix)}"
        return content


class SingleHarmonicRfStation(RfStationBaseClass):
    """Cavity with only one RF wave for beam interaction.

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
        voltage: float | None = None,
        phi_rf: float | None = None,
        harmonic: float | None = None,
        section_index: int = 0,
        local_wakefield: WakeField | None = None,
        cavity_feedback: LocalFeedback | None = None,
        beam_feedback: Blond2BeamFeedback | None = None,
        name: str | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ):
        super().__init__(
            n_rf=1,
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
            beam_feedback=beam_feedback,
            name=name,
            **kwargs,  # for MRO of fused elements
        )
        self.voltage: float | None = voltage
        self._phi_rf_design: float | None = phi_rf
        self.harmonic: float | None = harmonic

    def get_main_harmonic(self) -> float:
        """Returns the harmonic number of the main harmonic."""
        return self.harmonic

    def get_main_harmonic_voltage(self) -> float:
        """Returns the voltage of the main harmonic, in [V]."""
        return self.voltage

    def get_main_harmonic_phi_rf(self) -> float:
        """Returns the phi_rf of the main harmonic, in [rad]."""
        return self._phi_rf_design

    def get_main_harmonic_omega_rf_design(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> float:
        """Returns the omega_rf of the main harmonic, in [rad/s]."""
        return self.calc_omega_rf_design(
            beam_beta=beam_beta,
            ring_circumference=ring_circumference,
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)
        if (self.voltage is None) and "voltage" not in self.schedules:
            raise ValueError(
                "You need to define `voltage` via `.voltage=...` "
                f"or `.schedule(attribute='voltage', value=...)` for {self.name}"
            )
        if (self.phi_rf is None) and "phi_rf" not in self.schedules:
            raise ValueError(
                "You need to define `phi_rf` via `.phi_rf=...` "
                f"or `.schedule(attribute='phi_rf', value=...)` for {self.name}"
            )
        if (self.harmonic is None) and "harmonic" not in self.schedules:
            raise ValueError(
                "You need to define `harmonic` via `.harmonic=...` "
                f"or `.schedule(attribute='harmonic', value=...)` for {self.name}"
            )

    def _update_beam_based_attributes(self, beam: BeamBaseClass) -> None:
        self._omega_rf_design = self.calc_omega_rf_design(
            beam_beta=beam.reference_beta,
            ring_circumference=self._ring.circumference,
        )
        """self._t_rf = (2 * np.pi) / self._omega_rf
        self._t_rev = self._t_rf * self.harmonic
        try:
            self.phi_s = self.calc_phi_s_single_harmonic(beam=beam)
        except Exception as exc:
            warnings.warn(str(exc))
            self.phi_s = np.nan"""

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        super().track(beam=beam)

        target_total_energy = self._magnetic_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index
            if not beam.is_counter_rotating
            else len(self._ring.section_lengths) - self.section_index - 1,
            reference_time=beam.reference_time,
            particle_type=beam.particle_type,
        )
        reference_energy_change = (
            target_total_energy - beam.reference_total_energy
        )
        backend.specials.kick_single_harmonic(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=self.voltage,
            phi_rf=self.phi_rf,
            omega_rf=self.omega_rf,
            charge=beam.particle_type.charge,  #  FIXME
            acceleration_kick=-reference_energy_change,  # Mind the minus!
        )
        beam.reference_total_energy += reference_energy_change

        if self._beam_feedback is not None and (
            self._turn_i.value >= self._beam_feedback.delay
        ):  # TODO incorrect for simulations that start later
            # domega_rf is updated later
            # this means domega_rf is effectively from last turn
            assert self.harmonic is not None
            omega_increment = (
                self._beam_feedback.domega_rf  # dynamically updated by `update_domega_rf`
            )
            self.delta_omega_rf = omega_increment

        # Update the RF phase of all systems for the next turn
        # Accumulated phase offset due to beam phase loop or frequency offset
        if self.delta_omega_rf != 0:
            assert self.harmonic is not None
            assert self.omega_rf is not None
            phi_increment = (
                2.0
                * np.pi
                * self.harmonic
                * (self.delta_omega_rf)
                / self._omega_rf_design
            )

            self.delta_phi_rf += phi_increment

    def calc_omega_rf_design(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> float:
        """Calculate angular frequency of cavity, in [rad/s].

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
        return self.harmonic * backend.float(
            TWOPI_C0 * beam_beta / ring_circumference
        )

    def voltage_waveform_tmp(self, ts: NumpyArray):
        """Calculate voltage of cavity for current turn.

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
        local_wakefield: WakeField | None = None,
        cavity_feedback: LocalFeedback | None = None,
    ) -> SingleHarmonicRfStation:
        """Initialize object without simulation context.

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
        from .._core.beam.base import BeamBaseClass
        from .._core.ring.ring import Ring
        from .._core.simulation.simulation import Simulation
        from ..cycles.magnetic_cycle import ConstantMagneticCycle

        mhc = SingleHarmonicRfStation(
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
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
        )
        return mhc


class MultiHarmonicRfStation(RfStationBaseClass):
    """Cavity with several RF wave for beam interaction.

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
        voltage: NumpyArray,
        phi_rf: NumpyArray,
        harmonics: NumpyArray,
        n_harmonics: int,
        main_harmonic_idx: int,
        section_index: int = 0,
        local_wakefield: WakeField | None = None,
        cavity_feedback: LocalFeedback | None = None,
        beam_feedback: Blond2BeamFeedback | None = None,
        name: str | None = None,
    ):
        assert main_harmonic_idx < n_harmonics, (
            f"{n_harmonics=}, but {main_harmonic_idx=}."
        )

        super().__init__(
            n_rf=n_harmonics,
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
            beam_feedback=beam_feedback,
            name=name,
        )

        self.main_harmonic_idx = main_harmonic_idx

        self.voltage = voltage
        self._phi_rf_design = phi_rf
        self.harmonic = harmonics
        self.delta_phi_rf: NumpyArray | None = backend.zeros(
            n_harmonics
        )  # TODO
        self.delta_omega_rf: NumpyArray | None = backend.zeros(
            n_harmonics
        )  # TODO

        self._t_rf: NumpyArray | None = None
        self._t_rev: float | None = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)
        if (self.voltage is None) and "voltage" not in self.schedules:
            raise ValueError(
                f"You need to define `voltage` for '{self.name}' via "
                f"`.voltage=...` or `.schedule(attribute='voltage', value=...)`"
            )
        if (self.phi_rf is None) and "phi_rf" not in self.schedules:
            raise ValueError(
                f"You need to define `phi_rf` for '{self.name}' via "
                f"`.phi_rf=...` or `.schedule(attribute='phi_rf', value=...)`"
            )
        if (self.harmonic is None) and "harmonic" not in self.schedules:
            raise ValueError(
                f"You need to define `harmonic` for '{self.name}' via "
                f"`.harmonic=...` or `.schedule(attribute='harmonic', value=...)`"
            )

    def _update_beam_based_attributes(self, beam: BeamBaseClass) -> None:
        self._omega_rf_design = self.calc_omega_rf_design(
            beam_beta=beam.reference_beta,
            ring_circumference=self._ring.circumference,
        )

        self._t_rf = (2 * np.pi) / self._omega_rf_design
        self._t_rev = (
            self._t_rf[0] * self.harmonic[0]
        )  # todo this should be main harmonic idx??
        try:
            self.phi_s = self.calc_phi_s_single_harmonic(beam=beam)
        except Exception as exc:
            warnings.warn(str(exc), stacklevel=1)
            self.phi_s = np.nan

    def calc_omega_rf_design(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> NumpyArray:
        """Calculate angular frequency of cavity in [rad/s].

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
        return self.harmonic * (TWOPI_C0 * beam_beta / ring_circumference)

    def get_main_harmonic(self) -> float:
        """Returns the harmonic number of the main harmonic."""
        return self.harmonic[self.main_harmonic_idx]

    def get_main_harmonic_voltage(self) -> float:
        """Returns the voltage of the main harmonic, in [V]."""
        return self.voltage[self.main_harmonic_idx]

    def get_main_harmonic_phi_rf(self) -> float:
        """Returns the phi_rf of the main harmonic, in [rad]."""
        return self.phi_rf[self.main_harmonic_idx]

    def get_main_harmonic_omega_rf_design(
        self, beam_beta: float, ring_circumference: float
    ) -> float:
        """Returns the omega_rf of the main harmonic, in [rad/s]."""
        return self.calc_omega_rf_design(
            beam_beta=beam_beta,
            ring_circumference=ring_circumference,
        )[self.main_harmonic_idx]

    def voltage_waveform_tmp(self, ts: NumpyArray):
        """Calculate voltage of cavity for current turn.

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
            self._omega_rf_effective[0] * ts
            + self.phi_rf[0]
            + self.delta_phi_rf[0]
        )
        for i in range(1, len(self.voltage)):
            voltage += self.voltage[i] * np.sin(
                self._omega_rf_effective[i] * ts
                + self.phi_rf[i]
                + self.delta_phi_rf[i]
            )

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        super().track(beam=beam)
        target_total_energy = self._magnetic_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index
            if not beam.is_counter_rotating
            else len(self._ring.section_lengths) - self.section_index - 1,
            reference_time=beam.reference_time,
            particle_type=beam.particle_type,
        )
        reference_energy_change = (
            target_total_energy - beam.reference_total_energy
        )

        backend.specials.kick_multi_harmonic(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=(self.voltage).astype(backend.float),
            phi_rf=(self.phi_rf).astype(backend.float),
            omega_rf=(self.omega_rf).astype(backend.float),
            charge=beam.particle_type.charge,
            n_rf=self.n_rf,
            acceleration_kick=-reference_energy_change,  # Mind the minus!
        )
        beam.reference_total_energy += reference_energy_change

        if self._beam_feedback is not None and (
            self._turn_i.value >= self._beam_feedback.delay
        ):  # TODO incorrect for simulations that start later
            # domega_rf is updated later
            # this means domega_rf is effectively from last turn
            assert self.harmonic is not None
            omega_increment = (
                self._beam_feedback.domega_rf
                * self.harmonic[:]
                / self.harmonic[
                    self.main_harmonic_idx
                ]  # dynamically updated by `update_domega_rf`
            )
            self.delta_omega_rf = omega_increment

        # Update the RF phase of all systems for the next turn
        # Accumulated phase offset due to beam phase loop or frequency offset
        if self.delta_omega_rf[self.main_harmonic_idx] != 0:
            assert self.harmonic is not None
            assert self.omega_rf is not None
            phi_increment = (
                2.0
                * np.pi
                * self.harmonic[:]
                * (self.delta_omega_rf[:])
                / self.omega_rf[:]
            )

            self.delta_phi_rf += phi_increment

    @staticmethod
    def headless(
        section_index: int,
        voltage: NumpyArray,
        phi_rf: NumpyArray,
        harmonic: NumpyArray,
        circumference: float,
        total_energy: float,
        main_harmonic_idx: float,
        local_wakefield: WakeField | None = None,
        cavity_feedback: LocalFeedback | None = None,
        beam_feedback: Blond2BeamFeedback | None = None,
    ) -> MultiHarmonicRfStation:
        """Initialize object without simulation context.

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
        from .._core.beam.base import BeamBaseClass
        from .._core.ring.ring import Ring
        from .._core.simulation.simulation import Simulation
        from ..cycles.magnetic_cycle import ConstantMagneticCycle

        mhc = MultiHarmonicRfStation(
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
