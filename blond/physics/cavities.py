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
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0

from blond.acc_math.analytic.hamilton import (
    calc_phi_s_single_harmonic,
)
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
from blond.core.beam.beams import ProbeBeam
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.core.ring.helpers import requires
from blond.experimental.physics.feedbacks.base import (
    LocalFeedback as LocalFeedbackExp,
)
from blond.experimental.physics.feedbacks.beam_feedback import (
    BeamFeedbackBase,
)
from blond.experimental.physics.kick_pooling import (
    PooledInterpolationKick,
    SupportsPooledInterpolationKickMixIn,
)
from blond.physics.feedbacks.base import LocalFeedback

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

    from blond import Ring
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.cycles.magnetic_cycle import MagneticCycleBase
    from blond.experimental.physics.feedbacks.base import (
        LocalFeedback as LocalFeedbackExp,
    )
    from blond.experimental.physics.feedbacks.beam_feedback import (
        BeamFeedbackBase,
    )
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
        Additional keyword arguments for method
        resolution order of inheriting elements.
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
        if self.schedule_active:
            self.apply_schedules(
                turn_i=self._turn_i.value,
                reference_time=float(beam.reference.time),
            )


class RFStationBaseClass(RFManipulationBaseClass, AltersReference, ABC):
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
        For multi-harmonic cavities this needs to be a list with
        the same length as `n_rf`. Any number of elements in this list can be None.
        For a single-harmonic cavity either a list of length
        one or a LocalFeedback object can be provided.
        See :meth:`attach_cavity_feedback`.
    beam_feedback
        Optional beam feedback.
    name
        User given name of the element.
    delayed_kick
        The common interface to apply the kick later.
        `PooledInterpolationKick.track(...)` must be executed elsewhere.
    delayed_kick_time_axis
        The time axis along which to interpolate the kick.
        This impacts the accuracy and range of the RF kick.
    **kwargs
        Additional keyword arguments for method
        resolution order of inheriting elements.

    Attributes
    ----------
    omega_rf_design
        Design angular frequency relating to the harmonic numbers, in [rad/s].
    delta_omega_rf
        Correction term to omega_rf_design, used by feedbacks, in [rad/s].
    phi_rf_design
        Design angular phase, in [rad].
    delta_phi_rf
        Correction term for phi_rf_design, used by feedbacks, in [rad].
    voltage
        Voltage/s, in [V].
    harmonic
        Harmonic number, relating the rf frequency/ies to the revolution frequency.
    """

    def __init__(
        self,
        n_rf: int,
        section_index: int,
        local_wakefield: WakeField | None,
        cavity_feedback: LocalFeedback
        | LocalFeedbackExp
        | list[LocalFeedback | LocalFeedbackExp | None]
        | None = None,
        beam_feedback: BeamFeedbackBase | None = None,
        name: str | None = None,
        delayed_kick: PooledInterpolationKick | None = None,
        delayed_kick_time_axis: NumpyArray | CupyArray | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ):
        super().__init__(
            section_index=section_index,
            name=name,
            **kwargs,  # for MRO of fused elements
        )

        self._add_intended_schedule(
            "voltage",
            "phi_rf_design",
            "harmonic",
        )

        self._n_rf = n_rf

        self.cavity_feedback_list: list[
            LocalFeedback | LocalFeedbackExp | None
        ] = [None for _ in range(self._n_rf)]

        if cavity_feedback is not None:
            self.attach_cavity_feedback(cavity_feedback=cavity_feedback)

        self._beam_feedback: BeamFeedbackBase | None = (
            None  # set by  `attach_beam_feedback`
        )
        if beam_feedback is not None:
            self.attach_beam_feedback(beam_feedback)

        self._local_wakefield = local_wakefield

        self._magnetic_cycle: MagneticCycleBase | None = None
        self._ring: Ring | None = None

        self.omega_rf_design: NumpyArray | float | None = None
        self.delta_omega_rf: NumpyArray | float | None = None

        self.phi_rf_design: NumpyArray | float | None = None
        self.delta_phi_rf: NumpyArray | float | None = None

        # `_dphi_rf_next` is used to apply
        # the phase shift that was caused in
        # last turn to this turn before beam and
        # cavity feedbacks get updated.
        self._dphi_rf_next: NumpyArray | float | None = None

        self.voltage: NumpyArray | float | None = None
        self.harmonic: NumpyArray | float | None = None

        self._delayed_kick = delayed_kick
        if (
            self._delayed_kick is not None
            and self.cavity_feedback_list is None
        ):
            assert delayed_kick_time_axis is not None
        self._delayed_kick_time_axis = delayed_kick_time_axis

    @property
    def any_feedback_not_none(self) -> bool:
        """
        Check if there is a cavity feedback in the list of feedbacks, which is not None.

        Returns
        -------
        any_feedback_not_none
            If one array element is not None in self._cavity_feedback, return True.
        """
        return any(
            cavity_feedback is not None
            for cavity_feedback in self.cavity_feedback_list
        )

    @property
    def omega_rf(self) -> NumpyArray | float:
        """
        RF angular frequency.

        This might be altered by detuning/feedbacks.

        Returns
        -------
        omega_rf
            Angular rf frequency, potentially with feedback corrections.

        Notes
        -----
        `omega_rf` can not be set, use `omega_rf_design` instead
        """
        return self.omega_rf_design + self.delta_omega_rf

    @omega_rf.setter
    def omega_rf(self, _) -> None:
        raise AttributeError(
            "`omega_rf` can not be set, use `omega_rf_design` instead!"
        )

    @property
    def phi_rf(self) -> NumpyArray | float:
        """
        RF angular phase.

        This might be altered by detuning/feedbacks.

        Returns
        -------
        phi_rf
            Angular rf phase, potentially with feedback corrections.

        Notes
        -----
        `phi_rf` can not be set, use `phi_rf_design` instead!
        """
        return self.phi_rf_design + self.delta_phi_rf

    @phi_rf.setter
    def phi_rf(self, _) -> None:
        raise AttributeError(
            "`phi_rf` can not be set, use `phi_rf_design` instead!"
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
    def get_main_harmonic_omega_rf(self) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Returns
        -------
        main_harmonic_omega_rf
            The omega_rf of the main harmonic, in [rad/s].
        """
        pass

    def _get_gap_voltage_per_harmonic(
        self,
        ts: NumpyArray,
        harmonic_index: int | None = None,
        phase_offsets: NumpyArray | float = 0.0,
        voltage_correction_factors: NumpyArray | float = 1.0,
    ) -> NumpyArray:
        """
        Calculate voltage of RF station for current parameters.

        Parameters
        ----------
        ts
            Time array, in [s] to calculate voltage.
        harmonic_index
            Harmonic index to use, default is None, which will use the full array/float.
        phase_offsets
            Absolute Phase offset array, in [rad/s].
        voltage_correction_factors
            Relative voltage correction factors, in [1].

        Returns
        -------
        gap_voltage_per_harmonic
            RF station voltage in [V] at time `ts`.

        Notes
        -----
        This function is intended for small `ts` arrays
        and not executed in parallel.
        """
        if harmonic_index is None and not isinstance(self.phi_rf, float):
            raise ValueError(
                "If no `harmonic_index` is provided, `phi_rf` needs to be a float."
            )

        phi_rf = (
            self.phi_rf[harmonic_index]
            if isinstance(self, MultiHarmonicRFStation)
            else self.phi_rf
        )
        omega_rf = (
            self.omega_rf[harmonic_index]
            if isinstance(self, MultiHarmonicRFStation)
            else self.omega_rf
        )
        voltage = (
            self.voltage[harmonic_index]
            if isinstance(self, MultiHarmonicRFStation)
            else self.voltage
        )
        gap_voltage = (
            voltage
            * voltage_correction_factors
            * np.sin(omega_rf * ts + phi_rf + phase_offsets)
        )
        return backend.array(gap_voltage, backend.float)

    def calc_main_harmonic_t_rf(
        self, beam_beta: float, ring_circumference: float
    ) -> float:
        """
        Calculate the t_rf of the main harmonic, in [s].

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
        return (2 * np.pi) / self.calc_main_harmonic_omega_rf_design(
            beam_beta, ring_circumference
        )

    def attach_beam_feedback(self, beam_feedback: BeamFeedbackBase):
        """
        Attach beam feedback to the RF station after initialization.

        Parameters
        ----------
        beam_feedback
            Beam feedback to be attached to the RF station.
        """
        from blond.experimental.physics.feedbacks.beam_feedback import (
            BeamFeedbackBase,
        )

        if isinstance(beam_feedback, BeamFeedbackBase):
            self._beam_feedback = beam_feedback
        else:
            raise TypeError(f"{type(beam_feedback)=}")

    def attach_cavity_feedback(  # noqa: PLR0912
        self,
        cavity_feedback: LocalFeedback
        | LocalFeedbackExp
        | list[LocalFeedbackExp | LocalFeedback | None],
        harmonic_index: int | None = None,
    ):
        """
        Attach cavity feedback to the RF station after initialization.

        Parameters
        ----------
        cavity_feedback
            For multi-harmonic cavities this needs to be a list with
            the same length as `n_rf`. Any number of elements in this list can be None.
            For a single-harmonic cavity either a list of length
            one or a LocalFeedback object can be provided.
        harmonic_index
            Harmonic index at which to place the provided feedback.
            This needs to be provided for multiharmonic cavities,
            where a single LocalFeedback is provided.
        """
        from blond.experimental.physics.feedbacks.base import (
            LocalFeedback as LocalFeedbackExp,  # warning on BLonD startup; prevent Experimental
        )

        if isinstance(cavity_feedback, LocalFeedback | LocalFeedbackExp):
            if harmonic_index is None:
                if self._n_rf == 1:
                    harmonic_index = 0
                else:
                    raise ValueError(
                        "If a single feedback is provided, the harmonic_index needs to be provided as well."
                    )

            if harmonic_index > self._n_rf - 1:
                raise ValueError(
                    "Harmonic index must be less than the number of RF stations."
                )

            cavity_feedback.set_parent_rf_station(rf_station=self)  # type: ignore
            self.cavity_feedback_list[harmonic_index] = cavity_feedback

        elif isinstance(cavity_feedback, list):
            if len(cavity_feedback) != self._n_rf:
                raise ValueError(
                    f"Provided list has incorrect length, must be {self._n_rf=} but was {len(cavity_feedback)=}."
                )

            if harmonic_index is not None:
                warnings.warn(
                    "Given harmonic_index will be ignored since a list was provided.",
                    UserWarning,
                    stacklevel=2,
                )

            for feedback in cavity_feedback:
                if isinstance(feedback, LocalFeedback | LocalFeedbackExp):
                    feedback.set_parent_rf_station(rf_station=self)  # type: ignore
                elif feedback is None:
                    pass
                else:
                    raise TypeError(f"{type(feedback)=}")
            if self.any_feedback_not_none:
                warnings.warn(
                    "Already present cavity feedbacks are being overridden.",
                    UserWarning,
                    stacklevel=1,
                )
            self.cavity_feedback_list = list(cavity_feedback)
        else:
            raise TypeError(f"Invalid input type {type(cavity_feedback)=}")

    def calc_synchrotron_tune_main_harmonic(
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
            eta_0 = self._ring.calc_average_eta_0(beam.reference.gamma)

        if phi_s is None:
            phi_s = self.calc_phi_s_main_harmonic(beam)

        from blond.acc_math.analytic.hamilton import (
            calc_synchrotron_tune_single_harmonic,
        )

        Q_s0 = calc_synchrotron_tune_single_harmonic(
            charge=beam.particle_type.charge,
            voltage=self.get_main_harmonic_voltage(),
            beta=beam.reference.beta,
            energy=beam.reference.total_energy,
            phi_s=phi_s,
            harmonic=self.get_main_harmonic(),
            eta_0=eta_0,
        )

        return Q_s0

    def calc_phi_s_main_harmonic(self, beam: BeamBaseClass) -> float:
        """
        Calculate the main harmonic synchronous phase.

        Parameters
        ----------
        beam
            Beam class to interact with this element.

        Returns
        -------
        phi_s_main_harmonic
            Synchronous phase for the current RF parameters, in [rad].
        """
        # TODO rewrite for efficiency
        target_total_energy = self._magnetic_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index
            if not beam.is_counter_rotating
            else len(self._ring.section_lengths) - self.section_index - 1,
            reference_time=float(beam.reference.time),
            particle_type=beam.particle_type,
        )
        if self._ring.radiation_integrals is not None:
            energy_loss_per_turn = calculate_energy_loss_per_turn(
                energy=target_total_energy,
                radiation_integrals=self._ring.radiation_integrals,
                particle_type=beam.particle_type,
            )
            reference_energy_change = (
                target_total_energy
                - beam.reference.total_energy
                + energy_loss_per_turn
            )
        else:
            reference_energy_change = (
                target_total_energy - beam.reference.total_energy
            )

        phi_s = calc_phi_s_single_harmonic(
            charge=beam.particle_type.charge,
            voltage=float(self.get_main_harmonic_voltage()),
            energy_gain=reference_energy_change,
            above_transition=not self._ring.is_below_transition(beam=beam),
        )

        return phi_s

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
            ring_circumference=self._ring.circumference,
        )

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
        if not isinstance(beam, ProbeBeam) and self.any_feedback_not_none:
            for feedback in self.cavity_feedback_list:
                if feedback is not None:
                    feedback.track(beam=beam)

        if self._local_wakefield is not None:
            self._local_wakefield.track(beam=beam)

        if np.any(self.delta_omega_rf != 0):
            self._update_delta_phi_rf_from_beam_feedback()

    def _track_interp(
        self,
        beam: BeamBaseClass,
        reference_energy_change: float,
        time_axis: NumpyArray | CupyArray,
        voltage: NumpyArray | CupyArray,
    ):
        if self._delayed_kick is not None:
            self._delayed_kick.register(
                time_axis=time_axis,
                voltage=voltage - reference_energy_change,
            )
        else:
            backend.specials.kick_interpolated(
                dt=beam.read_partial_dt(),
                dE=beam.write_partial_dE(),
                voltage=backend.array(voltage, dtype=backend.float),
                bin_centers=backend.array(time_axis, dtype=backend.float),
                charge=beam.signed_charge_with_direction(),
                acceleration_kick=-reference_energy_change,  # Mind the minus!
            )

    def _update_delta_phi_rf_from_beam_feedback(self):
        """
        Update the phase slip for the next turn depending on the frequency change from the beam feedback.

        Update the RF phase of all systems for the next turn
        Accumulated phase offset due to beam phase loop or frequency offset.
        """
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

    def calc_omega_rf_design(
        self,
        beam_beta: float,
        ring_circumference: float,
    ) -> float | NumpyArray:
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
        omega
            Angular frequency (2 PI f) of RF station, in [rad/s].
        """
        return self.harmonic * float(TWOPI_C0 * beam_beta / ring_circumference)

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
        if self.any_feedback_not_none:
            for feedback in self.cavity_feedback_list:
                if feedback is not None:
                    content += (
                        f"{feedback.info_string(prefix=prefix + ' ↓ ')}\n"
                    )

        if self._local_wakefield is not None:
            content += (
                f"{self._local_wakefield.info_string(prefix=prefix + ' ↓ ')}\n"
            )
        content += f"{super().info_string(prefix=prefix)}"
        return content


class SingleHarmonicRFStation(
    RFStationBaseClass,
    SupportsPooledInterpolationKickMixIn,
):
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
    delayed_kick
        The common interface to apply the kick later.
        `PooledInterpolationKick.track(...)` must be executed elsewhere.
    delayed_kick_time_axis
        The time axis along which to interpolate the kick.
        This impacts the accuracy and range of the RF kick.
    **kwargs
        Additional keyword arguments for method
        resolution order of inheriting elements.

    Examples
    --------
    Parameters can be scheduled along the simulation execution

    >>> import numpy as np
    >>> from blond import SingleHarmonicRFStation
    >>> rf_station = SingleHarmonicRFStation(...)
    >>> rf_station.schedule(attribute='phi_rf', value=np.array(...))
    """

    voltage: float | None
    phi_rf: float | None
    omega_rf: float | None

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
        delayed_kick: PooledInterpolationKick | None = None,
        delayed_kick_time_axis: NumpyArray | CupyArray | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ):
        super().__init__(
            n_rf=1,
            section_index=section_index,
            local_wakefield=local_wakefield,
            cavity_feedback=cavity_feedback,
            beam_feedback=beam_feedback,
            name=name,
            delayed_kick=delayed_kick,
            delayed_kick_time_axis=delayed_kick_time_axis,
            **kwargs,  # for MRO of fused elements
        )

        self.voltage: float | None = voltage
        self.phi_rf_design: float | None = phi_rf
        self.harmonic: float | None = harmonic

        self.delta_phi_rf: float = 0.0
        self.delta_omega_rf: float = 0.0
        self._dphi_rf_next: float = 0.0

        if self._delayed_kick is not None and self.any_feedback_not_none:
            assert delayed_kick_time_axis is not None, (
                f"Got {delayed_kick_time_axis=}"
            )
        self._delayed_kick_time_axis = delayed_kick_time_axis

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
        if self.any_feedback_not_none:
            warnings.warn(
                "`get_main_harmonic_voltage` returns unperturbed "
                "voltage, even though local feedbacks are active.",
                UserWarning,
                stacklevel=2,
            )
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

    def calc_main_harmonic_omega_rf_design(
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
        return self.calc_omega_rf_design(
            beam_beta=beam_beta,
            ring_circumference=ring_circumference,
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

    def calc_gap_voltage_without_feedbacks(
        self, ts: NumpyArray
    ) -> NumpyArray | CupyArray:
        """
        Calculate total gap voltage in the RF station.

        This function calculates the total gap voltage including
        both the beam-induced and generator-induced voltages inside the
        RF cavities of the RF station.

        Parameters
        ----------
        ts
            Time array at which to evaluate, in [s].

        Returns
        -------
        gap_voltage
            Gap voltage in [V] within the length of the profile.
        """
        gap_voltage = self._get_gap_voltage_per_harmonic(
            ts=ts,
            harmonic_index=None,
        )
        return gap_voltage

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        # Apply phase shift that was caused in last turn
        # to this turn before beam and cavity feedbacks get updated.
        self.delta_phi_rf = deepcopy(self._dphi_rf_next)

        super()._track(beam=beam)

        reference = beam.reference
        reference_energy_change = self.track_reference(
            reference, beam.is_counter_rotating
        )

        if beam.common_array_size > 0:
            if self.any_feedback_not_none:
                voltage = backend.array(
                    self.calc_gap_voltage_with_feedbacks(), dtype=backend.float
                )
                time_axis = self.cavity_feedback_list[0].profile.hist_x
                if self._delayed_kick is not None:
                    if self._delayed_kick_time_axis is not None:
                        warnings.warn(
                            "`delayed_kick_time_axis` is ignored with "
                            "feedbacks. Set to `None` to silence this warning.",
                            UserWarning,
                            stacklevel=1,
                        )
                    self._delayed_kick.register(
                        time_axis=time_axis,
                        voltage=voltage - reference_energy_change,
                    )
                else:
                    self._track_interp(
                        beam=beam,
                        reference_energy_change=reference_energy_change,
                        time_axis=time_axis,
                        voltage=voltage,
                    )
            elif self._delayed_kick is not None:
                assert self._delayed_kick_time_axis is not None

                time_axis = self._delayed_kick_time_axis
                voltage = self.calc_gap_voltage_without_feedbacks(ts=time_axis)
                self._delayed_kick.register(
                    time_axis=time_axis,
                    voltage=voltage - reference_energy_change,
                )
            else:
                self._track_no_interp(
                    beam=beam, reference_energy_change=reference_energy_change
                )

    def _track_no_interp(
        self, beam: BeamBaseClass, reference_energy_change: float
    ):
        """
        Track without interpolation.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        reference_energy_change
            Update of the reference coordinate system, in [eV].
        """
        assert self.voltage is not None
        assert self.phi_rf is not None
        assert self.omega_rf is not None

        backend.specials.kick_single_harmonic(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=self.voltage,
            phi_rf=self.phi_rf,
            omega_rf=self.omega_rf,
            charge=beam.signed_charge_with_direction(),
            acceleration_kick=-reference_energy_change,  # Mind the minus!
        )

    def calc_gap_voltage_with_feedbacks(self):
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
        gap_voltage = self._get_gap_voltage_per_harmonic(
            ts=self.cavity_feedback_list[0].profile.hist_x,
            phase_offsets=self.cavity_feedback_list[0].phase_correction,
            voltage_correction_factors=self.cavity_feedback_list[
                0
            ].relative_voltage_correction,
        )

        return gap_voltage

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
        delayed_kick: PooledInterpolationKick | None = None,
        delayed_kick_time_axis: NumpyArray | CupyArray | None = None,
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
        delayed_kick
            The common interface to apply the kick later.
            `PooledInterpolationKick.track(...)` must be executed elsewhere.
        delayed_kick_time_axis
            The time axis along which to interpolate the kick.
            This impacts the accuracy and range of the RF kick.

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
            delayed_kick=delayed_kick,
            delayed_kick_time_axis=delayed_kick_time_axis,
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


class MultiHarmonicRFStation(
    RFStationBaseClass, SupportsPooledInterpolationKickMixIn
):
    r"""
    RF station with several RF wave for beam interaction.

    The energy change is calculated as:

    .. math::
        dE = \sum_{j} \left( n_\text{charge} \cdot V_j \cdot
        \sin\left(\omega_{\text{rf}, j} \cdot dt + \phi_{\text{rf}, j}\right)
        \right) + \Delta E_\text{reference}

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
    delayed_kick
        The common interface to apply the kick later.
        `PooledInterpolationKick.track(...)` must be executed elsewhere.
    delayed_kick_time_axis
        The time axis along which to interpolate the kick.
        This impacts the accuracy and range of the RF kick.
    **kwargs
        Additional keyword arguments for method
        resolution order of inheriting elements.

    Examples
    --------
    Parameters can be scheduled along the simulation execution

    >>> from blond import MultiHarmonicRFStation
    >>> rf_station = MultiHarmonicRFStation(...)
    >>> rf_station.schedule(attribute='phi_rf', value=np.array(...))
    """

    voltage: NumpyArray | CupyArray | None
    phi_rf: NumpyArray | CupyArray | None
    omega_rf: NumpyArray | CupyArray | None

    def __init__(
        self,
        n_harmonics: int,
        main_harmonic_idx: int,
        voltage: NumpyArray | None = None,
        phi_rf: NumpyArray | None = None,
        harmonic: NumpyArray | None = None,
        section_index: int = 0,
        local_wakefield: WakeField | None = None,
        cavity_feedback: LocalFeedback
        | tuple[LocalFeedback, ...]
        | None = None,
        beam_feedback: BeamFeedbackBase | None = None,
        name: str | None = None,
        delayed_kick: PooledInterpolationKick | None = None,
        delayed_kick_time_axis: NumpyArray | CupyArray | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
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
            delayed_kick=delayed_kick,
            delayed_kick_time_axis=delayed_kick_time_axis,
            **kwargs,  # for MRO of fused elements
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

        self.delta_phi_rf: NumpyArray = np.zeros(n_harmonics)
        self.delta_omega_rf: NumpyArray = np.zeros(n_harmonics)
        self._dphi_rf_next: NumpyArray = np.zeros(n_harmonics)

        if self._delayed_kick is not None and self.any_feedback_not_none:
            assert delayed_kick_time_axis is not None, (
                f"Got {delayed_kick_time_axis=}."
            )
        self._delayed_kick_time_axis = delayed_kick_time_axis

    def get_main_harmonic(self) -> float:
        """
        Return the harmonic number of the main harmonic.

        Returns
        -------
        main_harmonic
            Harmonic number of the main harmonic.
        """
        return self.harmonic[self.main_harmonic_idx]  # type: ignore

    def get_main_harmonic_voltage(self) -> float:
        """
        Return the voltage of the main harmonic, in [V].

        Returns
        -------
        main_harmonic_voltage
            Voltage of the main harmonic, in [V].
        """
        if self.any_feedback_not_none:
            warnings.warn(
                "`get_main_harmonic_voltage` returns unperturbed "
                "voltage, even though local feedbacks are active.",
                UserWarning,
                stacklevel=2,
            )
        return self.voltage[self.main_harmonic_idx]  # type: ignore

    def get_main_harmonic_phi_rf(self) -> float:
        """
        Return the phi_rf of the main harmonic, in [rad].

        Returns
        -------
        main_harmonic_phi_rf
            The phi_rf of the main harmonic, in [rad].
        """
        return self.phi_rf[self.main_harmonic_idx]  # type: ignore

    def calc_main_harmonic_omega_rf_design(
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
        return self.calc_omega_rf_design(  # type: ignore
            beam_beta=beam_beta,
            ring_circumference=ring_circumference,
        )[self.main_harmonic_idx]

    def get_main_harmonic_omega_rf(self) -> float:
        """
        Return the omega_rf of the main harmonic, in [rad/s].

        Returns
        -------
        omega_rf
            The angular frequency of the main harmonic, in [rad/s].
        """
        assert self.omega_rf is not None
        return self.omega_rf[self.main_harmonic_idx]

    def calc_gap_voltage_without_feedbacks(
        self, ts: NumpyArray
    ) -> NumpyArray | CupyArray:
        """
        Calculate total gap voltage in the RF station.

        This function calculates the total gap voltage including
        both the beam-induced and generator-induced voltages inside the
        RF cavities of the RF station.

        Parameters
        ----------
        ts
            Time array at which to evaluate, in [s].

        Returns
        -------
        gap_voltage
            Gap voltage in [V] within the length of the profile.
        """
        gap_voltage = backend.zeros(len(ts))
        for ind in range(self.n_rf):
            gap_voltage += self._get_gap_voltage_per_harmonic(
                ts=ts,
                harmonic_index=ind,
            )
        return gap_voltage

    def calc_gap_voltage_with_feedbacks(self):
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
        gap_voltage = backend.zeros(
            self.cavity_feedback_list[0].profile.n_bins
        )
        for ind, feedback in enumerate(self.cavity_feedback_list):
            if feedback is not None:
                gap_voltage += self._get_gap_voltage_per_harmonic(
                    ts=self.cavity_feedback_list[0].profile.hist_x,
                    harmonic_index=ind,
                    voltage_correction_factors=feedback.relative_voltage_correction,
                    phase_offsets=feedback.phase_correction,
                )
            else:
                gap_voltage += self._get_gap_voltage_per_harmonic(
                    ts=self.cavity_feedback_list[0].profile.hist_x,
                    harmonic_index=ind,
                )

        return gap_voltage

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        # Apply phase shift that was caused in last turn
        # to this turn before beam and cavity feedbacks get updated.
        self.delta_phi_rf = np.copy(self._dphi_rf_next)

        super()._track(beam=beam)

        reference = beam.reference
        reference_energy_change = self.track_reference(
            reference, beam.is_counter_rotating
        )

        if beam.common_array_size > 0:
            if self.any_feedback_not_none:
                voltage = backend.array(
                    self.calc_gap_voltage_with_feedbacks(), dtype=backend.float
                )
                time_axis = self.cavity_feedback_list[0].profile.hist_x
                if self._delayed_kick is not None:
                    if self._delayed_kick_time_axis is not None:
                        warnings.warn(
                            "`delayed_kick_time_axis` is ignored with "
                            "feedbacks. Set to `None` to silence this warning.",
                            UserWarning,
                            stacklevel=1,
                        )
                    self._delayed_kick.register(
                        time_axis=time_axis,
                        voltage=voltage - reference_energy_change,
                    )
                else:
                    self._track_interp(
                        beam=beam,
                        reference_energy_change=reference_energy_change,
                        time_axis=time_axis,
                        voltage=voltage,
                    )
            elif self._delayed_kick is not None:
                assert self._delayed_kick_time_axis is not None

                time_axis = self._delayed_kick_time_axis
                voltage = self.calc_gap_voltage_without_feedbacks(
                    ts=self._delayed_kick_time_axis,
                )

                self._delayed_kick.register(
                    time_axis=time_axis,
                    voltage=voltage - reference_energy_change,
                )
            else:
                self._track_no_interp(
                    beam=beam, reference_energy_change=reference_energy_change
                )

    def _track_no_interp(
        self, beam: BeamBaseClass, reference_energy_change: float
    ):
        """
        Track without interpolation.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        reference_energy_change
            Update of the reference coordinate system, in [eV].
        """
        assert self.voltage is not None
        assert self.phi_rf is not None
        assert self.omega_rf is not None

        backend.specials.kick_multi_harmonic(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=backend.array(self.voltage, dtype=backend.float),
            phi_rf=backend.array(self.phi_rf, dtype=backend.float),
            omega_rf=backend.array(self.omega_rf, dtype=backend.float),
            charge=beam.signed_charge_with_direction(),
            n_rf=self.n_rf,
            acceleration_kick=-reference_energy_change,  # Mind the minus!
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
        beam_reference_beta: float,
        local_wakefield: WakeField | None = None,
        cavity_feedback: LocalFeedback | None = None,
        beam_feedback: BeamFeedbackBase | None = None,
        delayed_kick: PooledInterpolationKick | None = None,
        delayed_kick_time_axis: NumpyArray | CupyArray | None = None,
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
        delayed_kick
            The common interface to apply the kick later.
            `PooledInterpolationKick.track(...)` must be executed elsewhere.
        delayed_kick_time_axis
            The time axis along which to interpolate the kick.
            This impacts the accuracy and range of the RF kick.

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
            delayed_kick=delayed_kick,
            delayed_kick_time_axis=delayed_kick_time_axis,
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
