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

from blond.acc_math.analytic.synchrotron_radiation.synchrotron_radiation_maths import (
    calculate_energy_loss_per_turn,
)
from blond.core.backends.backend import backend
from blond.core.base import (
    AltersReference,
    BeamPhysicsRelevant,
    DynamicParameter,
    Schedulable,
)
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.experimental.physics.feedbacks.beam_feedback import (
    Blond2BeamFeedback,
)

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond import Ring
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.cycles.magnetic_cycle import MagneticCycleBase
    from blond.physics.feedbacks.base import LocalFeedback
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

    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        super().track(beam=beam)
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

    def __init__(
        self,
        n_rf: int,
        section_index: int,
        local_wakefield: WakeField | None,
        cavity_feedback: tuple[LocalFeedback, ...]
        | None,  # FIXME should be tuple or not???
        beam_feedback: Blond2BeamFeedback | None,
        name: str | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ):
        from blond.physics.feedbacks.base import LocalFeedback

        # prevent cyclic import

        super().__init__(
            section_index=section_index,
            name=name,
            **kwargs,  # for MRO of fused elements
        )
        if cavity_feedback is None:
            pass
        else:
            for feedback in cavity_feedback:
                if isinstance(feedback, LocalFeedback):
                    pass  # TODO: fix, currently, one cannot setup the cavity without setting up the RF station first and vice versa
                    # cavity_feedback.set_parent_rf_station(rf_station=self)
                else:
                    raise ValueError(cavity_feedback)

        if beam_feedback is None:
            pass
        elif isinstance(beam_feedback, LocalFeedback):
            beam_feedback.set_parent_rf_station(rf_station=self)
        else:
            raise ValueError(beam_feedback)
        self._n_rf = n_rf
        self._local_wakefield = local_wakefield
        self._cavity_feedback: tuple[LocalFeedback, ...] | None = (
            cavity_feedback
        )
        self._beam_feedback = beam_feedback

        self._magnetic_cycle: MagneticCycleBase | None = None
        self._ring: Ring | None = None

        # TODO MOVE
        self._omega_rf: NumpyArray | float | None = None
        self.delta_omega_rf: NumpyArray | float | None = None
        self._t_rf: NumpyArray | float | None = None
        self._t_rev: float | None = None
        self.voltage: NumpyArray | float | None = None
        self.phi_rf: NumpyArray | float | None = None
        self.harmonic: NumpyArray | float | None = None
        self.phi_s: float | None = None

        self._use_synchrotron_radiation: bool = False

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

        if self._ring.synchrotron_radiation_integrals is not None:
            self._use_synchrotron_radiation = True

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
        pass

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
    def calc_main_harmonic_omega_rf(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> float:
        """
        Calculate the omega_rf of the main harmonic, in [rad/s].

        Parameters
        ----------
        beam_beta
            Relativistic beta of the beam.
        ring_circumference
            Ring circumference, in [m].

        Returns
        -------
        main_harmonic_omega_rf
            The omega_rf of the main harmonic, in [rad/s].
        """
        pass

    @abstractmethod  # pragma: no cover
    def get_main_harmonic_omega_rf_current(self) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Returns
        -------
        main_harmonic_omega_rf_current
            The omega_rf of the main harmonic, in [rad/s].
        """
        pass

    @abstractmethod  # pragma: no cover
    def calc_main_harmonic_t_rf(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> float:
        """
        Calculate the t_rf of the main harmonic.

        Parameters
        ----------
        beam_beta
            Relativistic beta of the beam.
        ring_circumference
            Ring circumference, in [m].

        Returns
        -------
        main_harmonic_t_rf(
            The t_rf of the main harmonic, in [s].
        """
        pass

    @abstractmethod  # pragma: no cover
    def get_main_harmonic_t_rf_current(self) -> float:
        """
        Return the current t_rf of the main harmonic.

        Returns
        -------
        main_harmonic_t_rf_curren
            The t_rf of the main harmonic, in [s].
        """
        pass

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
        if self._use_synchrotron_radiation:
            energy_loss_per_turn = calculate_energy_loss_per_turn(
                energy=target_total_energy,
                synchrotron_radiation_integrals=self._ring.synchrotron_radiation_integrals,
                particle_type=beam.particle_type,
            )  # TODO: check the indexing (using target total_energy or
            # beam.reference_energy?)
            reference_energy_change = (
                target_total_energy
                - beam.reference.total_energy
                + energy_loss_per_turn
            )
        else:
            reference_energy_change = (
                target_total_energy - beam.reference.total_energy
            )

        from blond.acc_math.analytic.hamilton import (
            calc_phi_s_single_harmonic,
        )

        assert self.voltage is not None
        assert self.phi_rf is not None
        phi_s = calc_phi_s_single_harmonic(
            charge=beam.particle_type.charge,
            voltage=float(self.get_main_harmonic_voltage()),
            phase=float(self.get_main_harmonic_phi_rf()),
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

    @abstractmethod  # pragma: no cover
    def _update_beam_based_attributes(self, beam: BeamBaseClass) -> None:
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        super().track(beam=beam)

        # set design omega etc. for this turn
        self._update_beam_based_attributes(beam=beam)

        # TODO incorrect for simulations that start later
        # Determine phase loop correction on RF phase and frequency
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
        if np.any(
            self.delta_omega_rf
        ):  # equivalent to np.any(self.delta_omega_rf != 0)
            assert self.harmonic is not None
            assert self._omega_rf is not None
            assert self.delta_omega_rf is not None
            phi_increment = (
                2.0
                * np.pi
                * self.harmonic[:]
                * self.delta_omega_rf
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
            self._beam_feedback.update_domega_rf(
                beam=beam
            )  # will be applied next turn

        # Correction from cavity loop
        if self._cavity_feedback is not None:
            for feedback in self._cavity_feedback:
                feedback.track(beam=beam)

        if self._local_wakefield is not None:
            self._local_wakefield.track(beam=beam)

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

    @abstractmethod  # pragma: no cover
    def calc_omega(
        self,
        beam_beta: float,
        closed_orbit_length: float,
    ):
        """
        Calculate angular frequency of RF station, in [rad/s].

        Parameters
        ----------
        beam_beta
            Beam reference fraction of speed of light (v/c0).
        closed_orbit_length
            Length of the closed orbit, in [m].

        Returns
        -------
        omega
            Angular frequency (2 PI f) of RF station, in [rad/s].
        """
        pass

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
    voltage
        RF station's effective voltage, in [V].
    phi_rf
        RF station's design phase, in [rad].
    harmonic
        RF station's design harmonic [].
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
        section_index: int = 0,
        local_wakefield: WakeField | None = None,
        cavity_feedback: tuple[LocalFeedback, ...] | None = None,
        beam_feedback: Blond2BeamFeedback | None = None,
        name: str | None = None,
        voltage: float | None = None,
        phi_rf: float | None = None,
        harmonic: float | None = None,
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
        self.phi_rf: float | None = phi_rf
        self.harmonic: float | None = harmonic
        self.delta_phi_rf: float = 0.0
        self.delta_omega_rf: float = 0.0

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
        return self.phi_rf

    def calc_main_harmonic_omega_rf(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Parameters
        ----------
        beam_beta
            Relativistic beta of the beam.
        ring_circumference
            Ring circumference, in [m].

        Returns
        -------
        _main_harmonic_omega_rf
            The omega_rf of the main harmonic, in [rad/s].
        """
        return self.calc_omega(
            beam_beta=beam_beta,
            ring_circumference=ring_circumference,
        )

    def get_main_harmonic_omega_rf_current(self) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Returns
        -------
        main_harmonic_omega_rf_current
            The omega_rf of the main harmonic, in [rad/s].
        """
        return self._omega_rf

    def get_main_harmonic_t_rf_current(
        self,
    ) -> float:
        """
        Return the t_rf of the main harmonic, in [s].

        Returns
        -------
        main_harmonic_t_rf_current
            The t_rf of the main harmonic, in [s].
        """
        return (2 * np.pi) / self.get_main_harmonic_omega_rf_current()

    def calc_main_harmonic_t_rf(
        self, beam_beta: float, ring_circumference: float
    ) -> float:
        """
        Return the t_rf of the main harmonic, in [s].

        Parameters
        ----------
        beam_beta
            Relativistic beta of the beam.
        ring_circumference
            Ring circumference, in [m].

        Returns
        -------
        main_harmonic_t_rf
            The t_rf of the main harmonic, in [s].
        """
        return (2 * np.pi) / self.calc_main_harmonic_omega_rf(
            beam_beta, ring_circumference
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
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
        self._omega_rf = self.calc_omega(
            beam_beta=beam.reference.beta,
            ring_circumference=self._ring.circumference,
        )
        """
        self._t_rf = (2 * np.pi) / self._omega_rf
        self._t_rev = self._t_rf * self.harmonic
        try:
            self.phi_s = self.calc_phi_s_single_harmonic(beam=beam)
        except Exception as exc:
            warnings.warn(str(exc), UserWarning, stacklevel=1)
            self.phi_s = np.nan"""

    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        super().track(beam=beam)
        reference = beam.reference
        reference_energy_change = self.track_reference(
            reference, beam.is_counter_rotating
        )
        if beam.common_array_size > 0:
            backend.specials.kick_single_harmonic(
                dt=beam.read_partial_dt(),
                dE=beam.write_partial_dE(),
                voltage=self.voltage,
                phi_rf=self.phi_rf + self.delta_phi_rf,
                omega_rf=self._omega_rf + self.delta_omega_rf,
                charge=beam.particle_type.charge,  #  FIXME
                acceleration_kick=-reference_energy_change,  # Mind the minus!
            )

    def calc_omega(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> float:
        """
        Calculate angular frequency of RF station, in [rad/s].

        Parameters
        ----------
        beam_beta
            Beam reference fraction of speed of light (v/c0).
        ring_circumference
            Reference synchrotron circumference, in [m].

        Returns
        -------
        oemga
            Angular frequency (2 PI f) of RF station, in [rad/s].
        """
        return self.harmonic * backend.float(
            TWOPI_C0 * beam_beta / ring_circumference
        )

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
        phi_rf = self.phi_rf + self.delta_phi_rf
        omega_rf = self._omega_rf + self.delta_omega_rf
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
        )

        single_harmonic_rf_station.voltage = voltage
        single_harmonic_rf_station.phi_rf = phi_rf
        single_harmonic_rf_station.harmonic = harmonic

        ring = Mock(Ring)
        ring.circumference = circumference

        energy_cycle = Mock(ConstantMagneticCycle)
        energy_cycle.get_target_total_energy.return_value = total_energy

        simulation = Mock(Simulation)
        simulation.ring = ring
        simulation.magnetic_cycle = energy_cycle
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0

        single_harmonic_rf_station.on_init_simulation(simulation=simulation)
        single_harmonic_rf_station.on_run_simulation(
            simulation=simulation,
            n_turns=1,
            beam=Mock(BeamBaseClass),
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
    n_harmonics
        Number of different RF waves for interaction.
    main_harmonic_idx
        Index of the RF station's main harmonic.
        Used to calculate attributes that rely on only one harmonic.
    voltage
        Cavity's effective voltages (per harmonic) in [V].
    phi_rf
        Cavity's design phases (per harmonic) in [rad].
    harmonic
        Cavity's design harmonics (per harmonic) [].
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
        n_harmonics: int,
        main_harmonic_idx: int,
        voltage: NumpyArray | None = None,
        phi_rf: NumpyArray | None = None,
        harmonic: NumpyArray | None = None,
        section_index: int = 0,
        local_wakefield: WakeField | None = None,
        cavity_feedback: tuple[LocalFeedback, ...] | None = None,
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

        self.voltage: NumpyArray | None = voltage
        self.phi_rf: NumpyArray | None = phi_rf
        self.harmonic: NumpyArray | None = harmonic

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

        self.delta_phi_rf: NumpyArray | None = backend.zeros(n_harmonics)
        self.delta_omega_rf: NumpyArray | None = backend.zeros(n_harmonics)

        self._t_rf: NumpyArray | None = None
        self._t_rev: float | None = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
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
        self._omega_rf = self.calc_omega(
            beam_beta=beam.reference.beta,
            ring_circumference=self._ring.circumference,
        )

        self._t_rf = 2 * np.pi / self._omega_rf
        self._t_rev = (
            self.get_main_harmonic_t_rf_current() * self.get_main_harmonic()
        )
        try:
            self.phi_s = self.calc_phi_s_single_harmonic(beam=beam)
        except Exception as exc:
            warnings.warn(str(exc), stacklevel=1)
            self.phi_s = np.nan

    def calc_omega(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> NumpyArray:
        """
        Calculate angular frequency of RF station in [rad/s].

        Parameters
        ----------
        beam_beta
            Beam reference fraction of speed of light (v/c0).
        ring_circumference
            Reference synchrotron circumference, in [m].

        Returns
        -------
        omega
            Angular frequency (2 PI f) of RF station in [rad/s].
        """
        return self.harmonic * (TWOPI_C0 * beam_beta / ring_circumference)

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

    def get_main_harmonic_omega_rf_current(self) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Returns
        -------
        main_harmonic_omega_rf_current
            The omega_rf of the main harmonic, in [rad/s].
        """
        return float(self._omega_rf[self.main_harmonic_idx])

    def calc_main_harmonic_omega_rf(
        self, beam_beta: float, ring_circumference: float
    ) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Parameters
        ----------
        beam_beta
            Relativistic beta of the beam.
        ring_circumference
            Ring circumference, in [m].

        Returns
        -------
        main_harmonic_omega_rf
            The omega_rf of the main harmonic, in [rad/s].
        """
        return float(
            self.calc_omega(
                beam_beta=beam_beta,
                ring_circumference=ring_circumference,
            )[self.main_harmonic_idx]
        )

    def get_main_harmonic_t_rf_current(
        self,
    ) -> float:
        """
        Return the t_rf of the main harmonic, in [s].

        Returns
        -------
        main_harmonic_t_rf_current
            The t_rf of the main harmonic, in [s].
        """
        return (2 * np.pi) / self.get_main_harmonic_omega_rf_current()

    def calc_main_harmonic_t_rf(
        self, beam_beta: float, ring_circumference: float
    ) -> float:
        """
        Return the t_rf of the main harmonic, in [s].

        Parameters
        ----------
        beam_beta
            Relativistic beta of the beam.
        ring_circumference
            Ring circumference, in [m].

        Returns
        -------
        main_harmonic_t_rf
            The t_rf of the main harmonic, in [s].
        """
        return (2 * np.pi) / self.calc_main_harmonic_omega_rf(
            beam_beta, ring_circumference
        )

    def voltage_waveform_tmp(self, ts: NumpyArray):  # pragma: no cover
        """
        Calculate voltage of RF station for current turn.

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

    @staticmethod
    def headless(
        section_index: int,
        voltage: NumpyArray,
        phi_rf: NumpyArray,
        harmonic: NumpyArray,
        circumference: float,
        total_energy: float,
        main_harmonic_idx: int,
        reference_beta: float,
        local_wakefield: WakeField | None = None,
        cavity_feedback: LocalFeedback | None = None,
        beam_feedback: Blond2BeamFeedback | None = None,
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
            Index of the cavity's main harmonic.
            Used to calculate attributes that rely on only one harmonic.
        reference_beta
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
            harmonic=np.array(harmonic, dtype=backend.float),
            voltage=np.array(voltage, dtype=backend.float),
            phi_rf=np.array(phi_rf, dtype=backend.float),
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
        multi_harmonic_rf_station.on_init_simulation(simulation=simulation)
        multi_harmonic_rf_station.on_run_simulation(
            simulation=simulation,
            n_turns=1,
            beam=Mock(BeamBaseClass),
            main_harmonic_idx=main_harmonic_idx,
        )
        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        beam.reference.beta = reference_beta
        multi_harmonic_rf_station._update_beam_based_attributes(beam)
        return multi_harmonic_rf_station

    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        super().track(beam=beam)
        reference = beam.reference
        reference_energy_change = self.track_reference(
            reference, beam.is_counter_rotating
        )
        if beam.common_array_size > 0:
            backend.specials.kick_multi_harmonic(
                dt=beam.read_partial_dt(),
                dE=beam.write_partial_dE(),
                voltage=self.voltage.astype(backend.float),
                phi_rf=(self.phi_rf + self.delta_phi_rf).astype(backend.float),
                omega_rf=(self._omega_rf + self.delta_omega_rf).astype(
                    backend.float
                ),
                charge=beam.particle_type.charge,
                n_rf=self.n_rf,
                acceleration_kick=-reference_energy_change,  # Mind the minus!
            )
