# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of implementations to handle lumped RF stations in synchrotrons."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0

from blond.acc_math.analytic.hamilton import (
    calc_phi_s_single_harmonic,
)
from blond.core.backends.backend import backend
from blond.core.base import (
    AltersReference,
    BeamPhysicsRelevant,
    DynamicParameter,
    Schedulable,
)
from blond.core.beam.beams import ProbeBeam
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.core.ring.helpers import requires
from blond.experimental.physics.feedbacks.beam_feedback import (
    BeamFeedbackBase,
)
from blond.physics.feedbacks.base import LocalFeedback

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond import Ring
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.cycles.magnetic_cycle import MagneticCycleBase
    from blond.physics.impedances.base import WakeField

TWOPI_C0 = 2.0 * np.pi * c0


class RFManipulationBaseClass(BeamPhysicsRelevant, Schedulable, ABC):
    """
    Base class to implement beam-rf any interactions in synchrotrons.

    This class is intended to come with barely any feature to host all
    beam-rf interactions, whereas `RFStationBaseClass` has already several
    class methods to group `SingleHarmonicRFStation`, and `MultiHarmonicRFStation`.

    Parameters
    ----------
    section_index
        Section index to group elements into sections.
    name
        User given name of the element.
    **kwargs
        Additional keyword arguments for MRO of fused elements.
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
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        super().on_init_simulation(simulation=simulation)

        self._turn_i = simulation.turn_i

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        super()._track(beam=beam)
        assert self._turn_i is not None
        if self.schedule_active:
            self.apply_schedules(
                turn_i=self._turn_i.value,
                reference_time=float(beam.reference.time),
            )


class RFStationBaseClass(
    RFManipulationBaseClass, AltersReference, Schedulable, ABC
):
    """
    Base class to implement beam-rf interactions in synchrotrons.

    Parameters
    ----------
    n_rf
        Number of different rf waves for interaction.
    section_index
        Section index to group elements into sections.
    local_wakefield
        Optional wakefield to interact with beam.
    cavity_feedback
        Optional cavity feedback to change cavity parameters.
    beam_feedback
        Optional beam feedback.
    name
        User given name of the element.
    **kwargs
        Additional keyword arguments for MRO of fused elements.
    """

    skip_find_instances_attributes = ["omega_rf_design"]

    def __init__(
        self,
        n_rf: int,
        section_index: int,
        local_wakefield: WakeField | None,
        cavity_feedback: LocalFeedback
        | tuple[LocalFeedback, ...]
        | None = None,
        beam_feedback: BeamFeedbackBase | None = None,
        name: str | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ):
        # prevent cyclic import

        super().__init__(
            section_index=section_index,
            name=name,
            **kwargs,  # for MRO of fused elements
        )
        self._cavity_feedback: (
            LocalFeedback | tuple[LocalFeedback, ...] | None
        ) = None
        if cavity_feedback is not None:
            self.attach_cavity_feedback(cavity_feedback=cavity_feedback)

        if beam_feedback is not None:
            if isinstance(beam_feedback, BeamFeedbackBase):
                self.attach_beam_feedback(beam_feedback)
            else:
                raise ValueError(beam_feedback)
        self._n_rf = n_rf
        self._local_wakefield = local_wakefield
        self._beam_feedback = beam_feedback

        self._magnetic_cycle: MagneticCycleBase | None = None
        self._ring: Ring | None = None

        # TODO MOVE
        self.omega_rf_design: NumpyArray | float | None = None
        self.delta_omega_rf: NumpyArray | float = 0.0
        self.phi_rf_design: NumpyArray | float | None = None
        self.delta_phi_rf: NumpyArray | float = 0.0
        self._dphi_rf_next: NumpyArray | float = 0.0
        self._t_rf: float | None = None
        self._t_rev: float | None = None
        self.voltage: NumpyArray | None = None
        self.harmonic: NumpyArray | None = None
        self.phi_s: NumpyArray | float | None = None
        self.omega_s0: NumpyArray | None = None

    @property
    def omega_rf(self) -> NumpyArray | float:
        """
        RF angular frequency.

        This might be altered by detuning/feedbacks.

        Returns
        -------
        omega_rf
            Actual angular rf frequency.
        """
        return self.omega_rf_design + self.delta_omega_rf

    @property
    def phi_rf(self) -> NumpyArray | float:
        """
        RF angular phase.

        This might be altered by detuning/feedbacks.

        Returns
        -------
        phi_rf
            Actual angular rf phase.
        """
        return self.phi_rf_design + self.delta_phi_rf

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        super().on_init_simulation(simulation=simulation)
        self._magnetic_cycle = simulation.magnetic_cycle
        self._ring = simulation.ring

        if (self.voltage is None) and "voltage" not in self.schedules:
            raise ValueError(
                f"You need to define `voltage` for '{self.name}' via "
                f"`.voltage=...` or `.schedule(attribute='voltage', value=...)`"
            )
        if (
            self.phi_rf_design is None
        ) and "phi_rf_design" not in self.schedules:
            raise ValueError(
                f"You need to define `phi_rf_design` for '{self.name}' via "
                f"`.phi_rf_design=...` or `.schedule(attribute='phi_rf_design', value=...)`"
            )
        if (self.harmonic is None) and "harmonic" not in self.schedules:
            raise ValueError(
                f"You need to define `harmonic` for '{self.name}' via "
                f"`.harmonic=...` or `.schedule(attribute='harmonic', value=...)`"
            )

    @requires(["BeamBaseClass"])
    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        # set design omega etc. for this turn
        self._update_beam_based_attributes(beam=beam)

    @abstractmethod  # pragma: no cover
    def get_main_harmonic(self) -> float:
        """
        Return the harmonic number of the main harmonic.

        Returns
        -------
        main_harmonic
            Harmonic number of the main harmonic.
        """
        pass

    @abstractmethod  # pragma: no cover
    def get_main_harmonic_voltage(self) -> float:
        """
        Return the voltage of the main harmonic, in [V].

        Returns
        -------
        main_harmonic_voltage
            Voltage of the main harmonic, in [V].
        """
        pass

    @abstractmethod  # pragma: no cover
    def get_main_harmonic_phi_rf(self) -> float:
        """
        Return the phi_rf of the main harmonic, in [rad].

        Returns
        -------
        main_harmonic_phi_rf
            The phi_rf of the main harmonic, in [rad].
        """
        pass

    @abstractmethod  # pragma: no cover
    def calc_main_harmonic_omega_rf_design(
        self,
        beam_beta: float,
        closed_orbit_length: float,
    ) -> float:
        """
        Calculate the omega_rf of the main harmonic, in [rad/s].

        Parameters
        ----------
        beam_beta
            Relativistic beta of the beam.
        closed_orbit_length
            Ring circumference, in [m].

        Returns
        -------
        main_harmonic_omega_rf
            The omega_rf of the main harmonic, in [rad/s].
        """
        pass

    @abstractmethod  # pragma: no cover
    def get_main_harmonic_omega_rf(self) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Returns
        -------
        main_harmonic_omega_rf
            The omega_rf of the main harmonic, in [rad/s].
        """
        pass

    def get_main_harmonic_t_rf(
        self,
    ) -> float:
        """
        Return the t_rf of the main harmonic, in [s].

        Returns
        -------
        main_harmonic_t_rf
            The t_rf of the main harmonic, in [s].
        """
        return (2 * np.pi) / self.get_main_harmonic_omega_rf()

    def calc_main_harmonic_t_rf(
        self, beam_beta: float, closed_orbit_length: float
    ) -> float:
        """
        Calculate the t_rf of the main harmonic, in [s].

        Parameters
        ----------
        beam_beta
            Relativistic beta of the beam.
        closed_orbit_length
            Ring circumference, in [m].

        Returns
        -------
        main_harmonic_t_rf
            The t_rf of the main harmonic, in [s].
        """
        return (2 * np.pi) / self.calc_main_harmonic_omega_rf_design(
            beam_beta, closed_orbit_length
        )

    def attach_beam_feedback(self, beam_feedback: BeamFeedbackBase):
        """
        Attach beam feedback to the RF station after initialization.

        Parameters
        ----------
        beam_feedback
            Beam feedback to be attached to the RF station.
        """
        self._beam_feedback = beam_feedback

    def attach_cavity_feedback(
        self, cavity_feedback: LocalFeedback | tuple[LocalFeedback, ...]
    ):
        """
        Attach cavity feedback to the RF station after initialization.

        Parameters
        ----------
        cavity_feedback
            Cavity feedback to be attached to the RF station.
        """
        # TODO: This can also be list of cavity feedbacks and can also be called multiple times to keep adding CCFBs
        if isinstance(
            cavity_feedback, LocalFeedback
        ):  # TODO: what if a wrong object is given?
            cavity_feedback = (cavity_feedback,)
        for feedback in cavity_feedback:
            if not isinstance(feedback, LocalFeedback):
                raise ValueError("given feedback is not a LocalFeedback")

            feedback.set_parent_rf_station(rf_station=self)

        if self._cavity_feedback is not None:
            raise Warning(
                "Already present cavity feedbacks are being overridden"
            )

        self._cavity_feedback = cavity_feedback

    def calc_synchrotron_tune_single_harmonic(
        self,
        beam: BeamBaseClass,
        phi_s: float | None = None,
        eta_0: float | None = None,
    ):
        """
        Function calculating the turn-by-turn synchrotron tune.

        The calculation assumes a single-harmonic RF system and no intensity
        effects.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        phi_s
            Synchronous phase, in [rad]. Will be calculated if not provided.
        eta_0
            First order slippage factor, in []. Will be calculated if not provided.

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
            beta=beam.reference.beta,
            energy=beam.reference.total_energy,
            phi_s=phi_s,
            harmonic=self.get_main_harmonic(),
            eta_0=eta_0,
        )

        return Q_s0

    def calc_phi_s_single_harmonic(self, beam: BeamBaseClass) -> float:
        """
        Calculate the main harmonic synchronous phase.

        Parameters
        ----------
        beam
            Beam class to interact with this element.

        Returns
        -------
        phi_s_single_harmonic
            Synchronous phase for the current RF parameters, in [rad].
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
            reference_time=float(beam.reference.time),
            particle_type=beam.particle_type,
        )
        reference_energy_change = (
            target_total_energy - beam.reference.total_energy
        )

        assert self.voltage is not None
        assert self.phi_rf is not None
        phi_s = calc_phi_s_single_harmonic(
            charge=beam.particle_type.charge,
            voltage=float(self.get_main_harmonic_voltage()),
            energy_gain=reference_energy_change,
            above_transition=not self._ring.is_below_transition(beam=beam),
        )

        return phi_s

    @property  # as readonly attributes
    def n_rf(self) -> int:
        """
        Number of different rf waves for interaction.

        Returns
        -------
        n_rf
            Number of different rf waves.
        """
        return self._n_rf

    def _update_beam_based_attributes(self, beam: BeamBaseClass) -> None:
        """
        Update internal data based on the tracked beam.

        Parameters
        ----------
        beam
            Beam to update the attributes from.
        """
        self.omega_rf_design = self.calc_omega_rf_design(
            beam_beta=beam.reference.beta,
            closed_orbit_length=self._ring.circumference,
        )

        self._t_rf = (2 * np.pi) / self.omega_rf_design  # TODO: remove
        self._t_rev = self.get_main_harmonic_t_rf() * self.get_main_harmonic()
        try:
            self.phi_s = self.calc_phi_s_single_harmonic(beam=beam)
        except Exception as exc:
            warnings.warn(str(exc), UserWarning, stacklevel=1)
            self.phi_s = np.nan

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        super()._track(beam=beam)

        # set design omega etc. for this turn
        self._update_beam_based_attributes(beam=beam)

        # Correction from cavity loop
        if (
            not isinstance(beam, ProbeBeam)
            and self._cavity_feedback is not None
        ):
            for feedback in self._cavity_feedback:
                if feedback is not None:
                    feedback.track(beam=beam)

        if self._local_wakefield is not None:
            self._local_wakefield.track(beam=beam)

    def _update_delta_phi_rf_from_beam_feedback(self):
        """
        Update the phase slip for the next turn depending on the frequency change from the beam feedback.

        Update the RF phase of all systems for the next turn
        Accumulated phase offset due to beam phase loop or frequency offset.
        """
        assert self.harmonic is not None
        assert self.omega_rf is not None

        phi_increment = (
            2.0 * np.pi * self.harmonic * self.delta_omega_rf / self.omega_rf
        )

        self._dphi_rf_next += phi_increment

    def track_reference(
        self,
        reference: ReferenceCoordinates,
        is_counter_rotating: bool = False,
    ) -> float:
        """
        Update the coordinates of the reference coordinate system.

        Parameters
        ----------
        reference
            The object that holds the reference time [s] and total energy [eV].
        is_counter_rotating
            Whether the beam is counter rotating or not.

        Returns
        -------
        reference_energy_change
            Change of reference energy [eV].
        """
        target_total_energy = self._magnetic_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index
            if not is_counter_rotating
            else len(self._ring.section_lengths) - self.section_index - 1,
            reference_time=reference.time,
            particle_type=reference.particle_type,
        )
        reference_energy_change = target_total_energy - reference.total_energy
        reference.total_energy = target_total_energy
        return reference_energy_change

    @abstractmethod  # pragma: no cover
    def voltage_waveform_tmp(self, ts: NumpyArray):
        """
        Calculate voltage of RF station for current turn.

        Parameters
        ----------
        ts
            Time array, in [s] to calculate voltage.
        """
        pass

    def calc_omega_rf_design(
        self,
        beam_beta: float,
        closed_orbit_length: float,
    ) -> float | NumpyArray:
        """
        Calculate angular frequency of RF station, in [rad/s].

        Parameters
        ----------
        beam_beta
            Beam reference fraction of speed of light (v/c0).
        closed_orbit_length
            Reference synchrotron circumference, in [m].

        Returns
        -------
        omega
            Angular frequency (2 PI f) of RF station, in [rad/s].
        """
        return self.harmonic * float(
            TWOPI_C0 * beam_beta / closed_orbit_length
        )

    def info_string(self, prefix="") -> str:
        """
        Inform that the feedback/wakefield is also executed within the track method.

        Parameters
        ----------
        prefix
            Prefix to add to the output string.

        Returns
        -------
        string
            Information string.
        """
        content = ""
        if self._cavity_feedback is not None:
            for feedback in self._cavity_feedback:
                content += f"{feedback.info_string(prefix=prefix + ' ↓ ')}\n"

        if self._local_wakefield is not None:
            content += (
                f"{self._local_wakefield.info_string(prefix=prefix + ' ↓ ')}\n"
            )
        content += f"{super().info_string(prefix=prefix)}"
        return content


class SingleHarmonicRFStation(RFStationBaseClass):
    r"""
    RF station with only one RF wave for beam interaction.

    The energy change is calculated as:

    .. math::
        dE = \left( n_\text{charge} \cdot V \cdot
        \sin\left(\omega_{\text{rf}} \cdot dt + \phi_{\text{rf}}
        \right) \right) + \Delta E_\text{reference}

    where :math:`\Delta E_\text{reference}` is the change of reference energy.

    Parameters
    ----------
    voltage
        RF station's effective voltage, in [V].
    phi_rf
        RF station's design phase, in [rad].
    harmonic
        RF station's design harmonic [].
    section_index
        Section index to group elements into sections.
    local_wakefield
        Optional wakefield to interact with beam.
    cavity_feedback
        Optional cavity feedback to change cavity parameters.
    beam_feedback
        Optional beam feedback.
    name
        User given name of the element.
    **kwargs
        Additional keyword arguments for MRO of fused elements.

    Attributes
    ----------
    voltage
        RF station's effective voltage, in [V].
    phi_rf
        RF station's design phase, in [rad].
    harmonic
        RF station's design harmonic [].

    Examples
    --------
    Parameters can be scheduled along the simulation execution

    >>> import numpy as np
    >>> from blond import SingleHarmonicRFStation
    >>> rf_station = SingleHarmonicRFStation(...)
    >>> rf_station.schedule(attribute='phi_rf', value=np.array(...), mode="per-turn")
    """

    def __init__(
        self,
        voltage: float | None = None,
        phi_rf: float | None = None,
        harmonic: float | None = None,
        section_index: int = 0,
        local_wakefield: WakeField | None = None,
        cavity_feedback: LocalFeedback
        | tuple[LocalFeedback, ...]
        | None = None,
        beam_feedback: BeamFeedbackBase | None = None,
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
        self.phi_rf_design: float | None = phi_rf
        self.harmonic: float | None = harmonic

    def get_main_harmonic(self) -> float:
        """
        Return the harmonic number of the main harmonic.

        Returns
        -------
        main_harmonic
            Harmonic number of the main harmonic.
        """
        return self.harmonic

    def get_main_harmonic_voltage(self) -> float:
        """
        Return the voltage of the main harmonic, in [V].

        Returns
        -------
        main_harmonic_voltage
            Voltage of the main harmonic, in [V].
        """
        return self.voltage

    def get_main_harmonic_phi_rf(self) -> float:
        """
        Return the phi_rf of the main harmonic, in [rad].

        Returns
        -------
        main_harmonic_phi_rf
            The phi_rf of the main harmonic, in [rad].
        """
        return self.phi_rf_design

    def calc_main_harmonic_omega_rf_design(
        self,
        beam_beta: float,
        closed_orbit_length: float,
    ) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Parameters
        ----------
        beam_beta
            Relativistic beta of the beam.
        closed_orbit_length
            Ring circumference, in [m].

        Returns
        -------
        _main_harmonic_omega_rf
            The omega_rf of the main harmonic, in [rad/s].
        """
        return self.calc_omega_rf_design(
            beam_beta=beam_beta,
            closed_orbit_length=closed_orbit_length,
        )

    def get_main_harmonic_omega_rf(self) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Returns
        -------
        main_harmonic_omega_rf
            The omega_rf of the main harmonic, in [rad/s].
        """
        return self.omega_rf

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        self.delta_phi_rf = np.copy(self._dphi_rf_next)

        super()._track(beam=beam)

        reference = beam.reference
        reference_energy_change = self.track_reference(
            reference, beam.is_counter_rotating
        )

        if beam.common_array_size > 0:
            if self._cavity_feedback is None:
                backend.specials.kick_single_harmonic(
                    dt=beam.read_partial_dt(),
                    dE=beam.write_partial_dE(),
                    voltage=self.voltage,
                    phi_rf=self.phi_rf,
                    omega_rf=self.omega_rf,
                    charge=beam.particle_type.charge,
                    acceleration_kick=-reference_energy_change,  # Mind the minus!
                )
            else:
                gap_voltage = self.calc_gap_voltage()
                backend.specials.kick_induced_voltage(
                    dt=beam.read_partial_dt(),
                    dE=beam.write_partial_dE(),
                    voltage=gap_voltage,
                    bin_centers=self._cavity_feedback[0].profile.hist_x,
                    charge=beam.particle_type.charge,
                    acceleration_kick=-reference_energy_change,  # Mind the minus!
                )

        if self.delta_omega_rf != 0:
            self._update_delta_phi_rf_from_beam_feedback()

    def calc_gap_voltage(self):
        """
        Calculate total gap voltage in the RF station.

        This function calculates the total gap voltage including
        both the beam-induced and generator-induced voltages inside the
        RF cavities of the RF station.

        Returns
        -------
        gap_voltage
            Gap voltage in [V] within the length of the profile.
        """
        x_arr = self._cavity_feedback[0].profile.hist_x

        voltages = self.voltage
        omega_rf = self.omega_rf
        phi_rf = self.phi_rf

        gap_voltage = (
            voltages
            * self._cavity_feedback[0].relative_voltage_correction
            * np.sin(
                omega_rf * x_arr
                + phi_rf
                + self._cavity_feedback[0].phase_correction
            )
        )

        return gap_voltage

    def voltage_waveform_tmp(self, ts: NumpyArray):
        """
        Calculate voltage of RF station for current turn.

        Parameters
        ----------
        ts
            Time array, in [s] to calculate voltage.

        Returns
        -------
        voltage_waveform
            RF station voltage in [V] at time `ts`.

        Notes
        -----
        This function is intended for small `ts` arrays
        and not executed in parallel.
        """
        voltage = self.voltage
        phi_rf = self.phi_rf
        omega_rf = self.omega_rf
        return voltage * np.sin(omega_rf * ts + phi_rf)

    @staticmethod
    def headless(
        section_index: int,
        voltage: float,
        phi_rf: float,
        harmonic: float,
        circumference: float,
        total_energy: float,
        beam_reference_beta: float,
        local_wakefield: WakeField | None = None,
        cavity_feedback: LocalFeedback | None = None,
    ) -> SingleHarmonicRFStation:
        """
        Initialize object without simulation context.

        Parameters
        ----------
        section_index
            Section index to group elements into sections.
        voltage
            RF station's effective voltage in [V].
        phi_rf
            RF station's design phase in [rad].
        harmonic
            RF station's design harmonic [].
        circumference
            Synchrotron circumference in [m].
        total_energy
            Target total energy in [eV].
        beam_reference_beta
            Beam velocity as a fraction of the speed of light [1].
        local_wakefield
            Optional wakefield to interact with beam.
        cavity_feedback
            Optional cavity feedback to change cavity parameters.

        Returns
        -------
        rf_station
            Initialized RF station object.
        """
        from blond.core.beam.base import BeamBaseClass
        from blond.core.ring.ring import Ring
        from blond.core.simulation.simulation import Simulation
        from blond.cycles.magnetic_cycle import ConstantMagneticCycle

        single_harmonic_rf_station = SingleHarmonicRFStation(
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
            voltage=voltage,
            phi_rf=phi_rf,
            harmonic=harmonic,
        )

        ring = Mock(Ring)
        ring.circumference = circumference

        energy_cycle = Mock(ConstantMagneticCycle)
        energy_cycle.get_target_total_energy.return_value = total_energy

        simulation = Mock(Simulation)
        simulation.ring = ring
        simulation.magnetic_cycle = energy_cycle
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0

        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        beam.reference.beta = beam_reference_beta
        single_harmonic_rf_station.on_init_simulation(simulation=simulation)
        single_harmonic_rf_station.on_run_simulation(
            simulation=simulation,
            n_turns=1,
            beam=beam,
        )
        return single_harmonic_rf_station


class MultiHarmonicRFStation(RFStationBaseClass):
    r"""
    RF station with several RF wave for beam interaction.

    The energy change is calculated as:

    .. math::
        dE = \sum_{j} \left( n_\text{charge} \cdot V[j] \cdot
        \sin\left(\omega_{\text{rf}}[j] \cdot dt + \phi_{\text{rf}}[
        j]\right) \right) + \Delta E_\text{reference}

    where :math:`\Delta E_\text{reference}` is the change of reference energy.

    Parameters
    ----------
    voltage
        Cavity's effective voltages (per harmonic) in [V].
    phi_rf
        Cavity's design phases (per harmonic) in [rad].
    harmonic
        Cavity's design harmonics (per harmonic) [].
    n_harmonics
        Number of different RF waves for interaction.
    main_harmonic_idx
        Index of the RF station's main harmonic.
        Used to calculate attributes that rely on only one harmonic.
    section_index
        Section index to group elements into sections.
    local_wakefield
        Optional wakefield to interact with beam.
    cavity_feedback
        Optional cavity feedback to change cavity parameters.
    beam_feedback
        Optional beam feedback.
    name
        User given name of the element.

    Attributes
    ----------
    voltage
        RF station's effective voltages (per harmonic) in [V].
    phi_rf
        RF station's design phases (per harmonic) in [rad].
    harmonic
        RF station's design harmonics (per harmonic) [].

    Examples
    --------
    Parameters can be scheduled along the simulation execution

    >>> from blond import MultiHarmonicRFStation
    >>> rf_station = MultiHarmonicRFStation(...)
    >>> rf_station.schedule(attribute='phi_rf', value=np.array(...), mode="per-turn")
    """

    def __init__(
        self,
        voltage: NumpyArray,
        phi_rf: NumpyArray,
        harmonic: NumpyArray,
        n_harmonics: int,
        main_harmonic_idx: int,
        section_index: int = 0,
        local_wakefield: WakeField | None = None,
        cavity_feedback: LocalFeedback
        | tuple[LocalFeedback, ...]
        | None = None,
        beam_feedback: BeamFeedbackBase | None = None,
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

        self.voltage: NumpyArray | None = (
            np.array(voltage) if (voltage is not None) else None
        )
        self.phi_rf_design: NumpyArray | None = (
            np.array(phi_rf) if (phi_rf is not None) else None
        )
        self.harmonic: NumpyArray | None = (
            np.array(harmonic) if (harmonic is not None) else None
        )
        self.delta_phi_rf: NumpyArray | None = np.zeros(n_harmonics)
        self.delta_omega_rf: NumpyArray | None = np.zeros(n_harmonics)

        for array_name, input_array in (
            ("voltage", voltage),
            ("phi_rf", phi_rf),
            ("harmonic", harmonic),
        ):
            if input_array is not None and len(input_array) != n_harmonics:
                raise ValueError(
                    f"Length of input array must be equal to {n_harmonics=}, "
                    f"but {array_name} had the length {len(input_array)}"
                )

        assert main_harmonic_idx < n_harmonics, (
            f"main_harmonic_index was {main_harmonic_idx}, "
            f"but needs to be smaller than {n_harmonics}"
        )

        self.delta_phi_rf: NumpyArray | None = np.zeros(n_harmonics)
        self.delta_omega_rf: NumpyArray | None = np.zeros(n_harmonics)

        self._t_rf: NumpyArray | None = None
        self._t_rev: float | None = None

        self._dphi_rf_next: NumpyArray | None = np.zeros(n_harmonics)

    def get_main_harmonic(self) -> float:
        """
        Return the harmonic number of the main harmonic.

        Returns
        -------
        main_harmonic
            Harmonic number of the main harmonic.
        """
        return self.harmonic[self.main_harmonic_idx]

    def get_main_harmonic_voltage(self) -> float:
        """
        Return the voltage of the main harmonic, in [V].

        Returns
        -------
        main_harmonic_voltage
            Voltage of the main harmonic, in [V].
        """
        return self.voltage[self.main_harmonic_idx]

    def get_main_harmonic_phi_rf(self) -> float:
        """
        Return the phi_rf of the main harmonic, in [rad].

        Returns
        -------
        main_harmonic_phi_rf
            The phi_rf of the main harmonic, in [rad].
        """
        return self.phi_rf[self.main_harmonic_idx]

    def calc_main_harmonic_omega_rf_design(
        self, beam_beta: float, closed_orbit_length: float
    ) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Parameters
        ----------
        beam_beta
            Relativistic beta of the beam.
        closed_orbit_length
            Ring circumference, in [m].

        Returns
        -------
        main_harmonic_omega_rf
            The omega_rf of the main harmonic, in [rad/s].
        """
        return self.calc_omega_rf_design(
            beam_beta=beam_beta,
            closed_orbit_length=closed_orbit_length,
        )[self.main_harmonic_idx]

    def get_main_harmonic_omega_rf(self) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Returns
        -------
        omega_rf
            The angular frequency of the main harmonic, in [rad/s].
        """
        return self.omega_rf[self.main_harmonic_idx]

    def calc_gap_voltage(self):
        """
        Calculate total gap voltage in the RF station.

        This function calculates the total gap voltage including
        both the beam-induced and generator-induced voltages inside the
        RF cavities of the RF station.

        Returns
        -------
        gap_voltage
            Gap voltage in [V] within the length of the profile.
        """
        n_slices = self._cavity_feedback[0].profile.n_bins
        x_arr = self._cavity_feedback[0].profile.hist_x

        voltages = np.outer(self.voltage, backend.ones(n_slices))
        omega_rf = np.outer(self.omega_rf, backend.ones(n_slices))
        phi_rf = np.outer(self.phi_rf, backend.ones(n_slices))

        gap_voltage = backend.zeros(n_slices)
        for ind, feedback in enumerate(self._cavity_feedback):
            if feedback is not None:
                gap_voltage = (
                    voltages[ind]
                    * feedback.relative_voltage_correction
                    * np.sin(
                        omega_rf[ind] * x_arr
                        + phi_rf[ind]
                        + feedback.phase_correction
                    )
                )
            else:
                gap_voltage = voltages[ind] * np.sin(
                    omega_rf[ind] * x_arr + phi_rf[ind]
                )

        return gap_voltage

    def voltage_waveform_tmp(self, ts: NumpyArray):  # pragma: no cover
        """
        Calculate voltage of cavity for current turn.

        Parameters
        ----------
        ts
            Time array, in [s] to calculate voltage.

        Notes
        -----
        This function is intended for small ts arrays
        and not executed in parallel.
        """
        raise NotImplementedError
        voltage = self.voltage[0] * np.sin(
            self.omega_rf[0] * ts
            + self.phi_rf_design[0]
            + self.delta_phi_rf[0]
        )
        for i in range(1, len(self.voltage)):
            voltage += self.voltage[i] * np.sin(
                self.omega_rf[i] * ts
                + self.phi_rf_design[i]
                + self.delta_phi_rf[i]
            )

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        self.delta_phi_rf = np.copy(self._dphi_rf_next)

        super()._track(beam=beam)

        reference = beam.reference
        reference_energy_change = self.track_reference(
            reference, beam.is_counter_rotating
        )

        if beam.common_array_size > 0:
            if self._cavity_feedback is None:
                backend.specials.kick_multi_harmonic(
                    dt=beam.read_partial_dt(),
                    dE=beam.write_partial_dE(),
                    voltage=self.voltage.astype(backend.float),
                    phi_rf=self.phi_rf.astype(backend.float),
                    omega_rf=self.omega_rf.astype(backend.float),
                    charge=beam.particle_type.charge,
                    n_rf=self.n_rf,
                    acceleration_kick=-reference_energy_change,  # Mind the minus!
                )
            else:
                gap_voltage = self.calc_gap_voltage()
                backend.specials.kick_induced_voltage(
                    dt=beam.read_partial_dt(),
                    dE=beam.write_partial_dE(),
                    voltage=gap_voltage,
                    bin_centers=self._cavity_feedback[0].profile.hist_x,
                    charge=beam.particle_type.charge,
                    acceleration_kick=-reference_energy_change,  # Mind the minus!
                )

        if self.delta_omega_rf[self.main_harmonic_idx] != 0:
            self._update_delta_phi_rf_from_beam_feedback()

    @staticmethod
    def headless(
        section_index: int,
        voltage: NumpyArray,
        phi_rf: NumpyArray,
        harmonic: NumpyArray,
        circumference: float,
        total_energy: float,
        main_harmonic_idx: int,
        beam_reference_beta: float,
        local_wakefield: WakeField | None = None,
        cavity_feedback: LocalFeedback | None = None,
        beam_feedback: BeamFeedbackBase | None = None,
    ) -> MultiHarmonicRFStation:
        """
        Initialize object without simulation context.

        Parameters
        ----------
        section_index
            Section index to group elements into sections.
        voltage
            RF station's effective voltages (per harmonic) in [V].
        phi_rf
            RF station's design phases (per harmonic) in [rad].
        harmonic
            RF station's design harmonics (per harmonic) [].
        circumference
            Synchrotron circumference in [m].
        total_energy
            Target total energy in [eV].
        main_harmonic_idx
            Index of the cavity's main harmonic
            Used to calculate attributes that rely on only one harmonic.
        beam_reference_beta
            Beam reference fraction of speed of light (v/c0) [].
        local_wakefield
            Optional wakefield to interact with beam.
        cavity_feedback
            Optional cavity feedback to change cavity parameters.
        beam_feedback
            Optional beam feedback to change cavity parameters.

        Returns
        -------
        rf_station
            Initialized RF station object.
        """
        from blond.core.beam.base import BeamBaseClass
        from blond.core.ring.ring import Ring
        from blond.core.simulation.simulation import Simulation
        from blond.cycles.magnetic_cycle import ConstantMagneticCycle

        multi_harmonic_rf_station = MultiHarmonicRFStation(
            harmonic=np.array(harmonic, dtype=float),
            voltage=np.array(voltage, dtype=float),
            phi_rf=np.array(phi_rf, dtype=float),
            n_harmonics=len(voltage),
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
            beam_feedback=beam_feedback,
            main_harmonic_idx=main_harmonic_idx,
        )

        ring = Mock(Ring)
        ring.circumference = circumference

        energy_cycle = Mock(ConstantMagneticCycle)
        energy_cycle.get_target_total_energy.return_value = total_energy

        simulation = Mock(Simulation)
        simulation.ring = ring
        simulation.magnetic_cycle = energy_cycle
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0
        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        beam.reference.beta = beam_reference_beta
        multi_harmonic_rf_station.on_init_simulation(simulation=simulation)
        multi_harmonic_rf_station.on_run_simulation(
            simulation=simulation,
            n_turns=1,
            beam=beam,
        )

        multi_harmonic_rf_station._update_beam_based_attributes(beam)
        return multi_harmonic_rf_station
