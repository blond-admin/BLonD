# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Pydantic models of a portable cycle definition.

All quantities are in SI / accelerator units: [s], [eV], [eV/c], [V],
[rad], [m], [T]. Time is measured from the start of the simulation.

All numbers must be finite; `NaN` and infinities are rejected.
"""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import pydantic as pyd
from scipy.interpolate import (
    Akima1DInterpolator,
    PchipInterpolator,
    interp1d,
)

from blond.core.beam.particle_types import ParticleType
from blond.core.ring.ring import Ring
from blond.core.scheduling import ScheduledInterpolation
from blond.cycles.magnetic_cycle import (
    ConstantMagneticCycle,
    MagneticCycleByTime,
)
from blond.generals.interpolators import DerivativeInterpolator
from blond.physics.drifts import DriftExact, DriftSimple
from blond.physics.rf_station import (
    MultiHarmonicRFStation,
    SingleHarmonicRFStation,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Sequence
    from typing import Self

    from numpy.typing import NDArray as NumpyArray

    from blond.core.base import Schedulable
    from blond.cycles.magnetic_cycle import MagneticCycleBase

SCHEMA_VERSION = 1

_INTERPOLATOR_TYPES = {
    "interp1d": interp1d,
    "Akima1DInterpolator": Akima1DInterpolator,
    "PchipInterpolator": PchipInterpolator,
    "DerivativeInterpolator": DerivativeInterpolator,
}


def _interpolator_name(interpolator_type: type) -> str:
    """
    Get the portable name of an interpolator class.

    Parameters
    ----------
    interpolator_type
        Interpolator class.

    Returns
    -------
    name
        Name used in `Programme.interpolator`.

    Raises
    ------
    ValueError
        If the class is not supported.
    """
    for name, known_type in _INTERPOLATOR_TYPES.items():
        if known_type is interpolator_type:
            return name
    raise ValueError(f"Unsupported interpolator {interpolator_type.__name__}")


def _from_array(value: Any) -> Any:
    """
    Convert array-likes (e.g. NumPy or CuPy arrays) to Python lists.

    Parameters
    ----------
    value
        Value to convert.

    Returns
    -------
    converted
        `value.tolist()` if available, otherwise `value`.
    """
    return value.tolist() if hasattr(value, "tolist") else value


def _to_int_if_integer(value: Any) -> Any:
    """
    Convert floats with an integer value, e.g. ``1.0``, to `int`.

    Parameters
    ----------
    value
        Value to convert.

    Returns
    -------
    converted
        `int(value)` for a float with an integer value, otherwise `value`.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _reject_values(
    values: Iterable[float],
    validator: Callable[[float], bool],
    message: str,
) -> None:
    """
    Raise if any value is invalid.

    Parameters
    ----------
    values
        Values to check.
    validator
        Returns True for an invalid value.
    message
        Error message.

    Raises
    ------
    ValueError
        If `validator` is True for any value.
    """
    if any(validator(value) for value in values):
        raise ValueError(message)


def _interpolate_programme(
    programme: Programme, times: NumpyArray
) -> NumpyArray:
    """
    Get the values of a programme on a time axis.

    Parameters
    ----------
    programme
        Programme to evaluate.
    times
        Time axis, in [s].

    Returns
    -------
    values
        The programme's own values if `times` is its time axis, a constant
        for a single point, otherwise its interpolator evaluated at `times`.
    """
    if programme.is_constant:
        values = np.full(len(times), programme.value[0])

    elif np.array_equal(programme.time, times):
        values = np.array(programme.value)

    else:
        interpolator = _INTERPOLATOR_TYPES[programme.interpolator](
            programme.time, programme.value, **programme.interpolator_kwargs
        )
        values = interpolator(times)

    return values


def _to_scheduled_interpolation(
    attribute: str,
    programmes: Sequence[Programme],
    main_idx: int,
) -> ScheduledInterpolation:
    """
    Combine the programmes of an attribute into one schedule.

    Every programme is evaluated with its own interpolator on the union of
    all time axes. The result is interpolated with the interpolator of the
    main programme.

    Parameters
    ----------
    attribute
        Name of the scheduled attribute.
    programmes
        One programme per harmonic or order.
    main_idx
        Index of the main programme in `programmes`.

    Returns
    -------
    scheduled_interpolation
        Schedule returning one value per programme, or a scalar for a single
        programme.

    Raises
    ------
    ValueError
        If several programmes are combined with a `DerivativeInterpolator`.

    Warns
    -----
    UserWarning
        If the programmes use different interpolators.
    """
    main = programmes[main_idx]
    interpolator_type = _INTERPOLATOR_TYPES[main.interpolator]

    # Return early if only one programme
    if len(programmes) == 1:
        return ScheduledInterpolation(
            main.time,
            main.value,
            interpolator_type,
            **main.interpolator_kwargs,
        )

    if len({p.interpolator for p in programmes if not p.is_constant}) > 1:
        warnings.warn(
            f"The programmes of `{attribute}` use different interpolators. "
            f"The interpolator of the main programme (index {main_idx}), "
            f"'{main.interpolator}', is used for all programmes.",
            UserWarning,
            stacklevel=3,
        )

    if interpolator_type is DerivativeInterpolator:
        raise ValueError(
            f"`{attribute}`: 'DerivativeInterpolator' of the main programme "
            "cannot interpolate several programmes at once"
        )

    times = np.unique(
        np.concatenate(
            [p.time for p in programmes if not p.is_constant],
        )
    )
    return ScheduledInterpolation(
        times,
        np.column_stack(
            [_interpolate_programme(p, times) for p in programmes]
        ),
        interpolator_type,
        **{**main.interpolator_kwargs, "axis": 0},
    )


def _read_programmes(
    element: Schedulable,
    attribute: str,
) -> list[Programme]:
    """
    Read the programmes of a schedulable attribute of an element.

    Parameters
    ----------
    element
        A schedulable BLonD element, e.g. an RF station or a drift.
    attribute
        Name of the attribute.

    Returns
    -------
    programmes
        One programme per harmonic or order: single points for an
        unscheduled attribute, otherwise the stored input of its
        `ScheduledInterpolation`.

    Raises
    ------
    ValueError
        If the attribute is not set, or its values are not stored along the
        first axis.
    TypeError
        If the attribute is scheduled by anything but a
        `ScheduledInterpolation`.
    """
    schedule = element.schedules.get(attribute)

    # Early return/error if no schedule
    if schedule is None:
        value = getattr(element, attribute)
        if value is None:
            raise ValueError(
                f"`{attribute}` of the {type(element).__name__} is not set"
            )
        else:
            return [
                Programme(time=(0.0,), value=(v,))
                for v in np.atleast_1d(value)
            ]

    if not isinstance(schedule, ScheduledInterpolation):
        raise TypeError(
            f"`{attribute}` is scheduled by {type(schedule).__name__}, only "
            "`ScheduledInterpolation` is supported"
        )

    kwargs = {**schedule.interpolator_kwargs}
    default_axis = -1 if schedule.interpolator_type is interp1d else 0
    axis = kwargs.pop("axis", default_axis)
    values = schedule.values
    if values.ndim == 1:
        columns = [values]
    elif axis == 0:
        columns = list(values.T)
    else:
        raise ValueError(
            f"`{attribute}`: values must be stored along axis 0, got {axis=}"
        )
    return [
        Programme(
            time=schedule.times,
            value=column,
            interpolator=_interpolator_name(schedule.interpolator_type),
            interpolator_kwargs=kwargs,
        )
        for column in columns
    ]


Float = Annotated[float, pyd.Strict(), pyd.AllowInfNan(False)]
Int = Annotated[int, pyd.Strict()]


class _FrozenModel(pyd.BaseModel):
    """Immutable model that rejects unknown fields."""

    model_config = pyd.ConfigDict(frozen=True, extra="forbid")


class Programme(_FrozenModel):
    """
    Value programmed along time.

    A single point means a constant value.

    Attributes
    ----------
    time
        Times of the programmed points, strictly increasing, in [s].
    value
        Values at `time`, in the unit of the owning field.
    interpolator
        Intended BLonD interpolator between points.
    interpolator_kwargs
        Keyword arguments for `interpolator`, e.g. ``{"method": "makima"}``
        for 'Akima1DInterpolator'.
    """

    time: Annotated[tuple[Float, ...], pyd.BeforeValidator(_from_array)] = (
        pyd.Field(min_length=1)
    )
    value: Annotated[tuple[Float, ...], pyd.BeforeValidator(_from_array)] = (
        pyd.Field(min_length=1)
    )
    interpolator: Literal[
        "interp1d",
        "Akima1DInterpolator",
        "PchipInterpolator",
        "DerivativeInterpolator",
    ] = "interp1d"
    interpolator_kwargs: dict[str, str | bool | Int | Float] = pyd.Field(
        default_factory=dict
    )

    @pyd.model_validator(mode="after")
    def _check_points(self) -> Self:
        """
        Check that `time` and `value` match and `time` increases.

        Returns
        -------
        programme
            The validated programme.

        Raises
        ------
        ValueError
            If the lengths differ or `time` is not strictly increasing.
        """
        if len(self.time) != len(self.value):
            raise ValueError(
                f"`time` and `value` lengths differ: "
                f"{len(self.time)} != {len(self.value)}"
            )

        if not all(t1 > t0 for t0, t1 in itertools.pairwise(self.time)):
            raise ValueError("`time` must be strictly increasing")

        return self

    @pyd.model_validator(mode="after")
    def _check_interpolator_kwargs(self) -> Self:
        """
        Check that `interpolator_kwargs` are accepted by `interpolator`.

        Returns
        -------
        programme
            The validated programme.

        Raises
        ------
        ValueError
            If a keyword is not accepted by `interpolator`.
        """
        accepted = {
            "interp1d": {
                "kind",
                "copy",
                "bounds_error",
                "fill_value",
                "assume_sorted",
            },
            "Akima1DInterpolator": {"method", "extrapolate"},
            "PchipInterpolator": {"extrapolate"},
            "DerivativeInterpolator": {"left", "right"},
        }[self.interpolator]
        unknown = sorted(set(self.interpolator_kwargs) - accepted)
        if unknown:
            raise ValueError(
                f"`interpolator_kwargs` {unknown} not accepted by "
                f"'{self.interpolator}', accepted: {sorted(accepted)}"
            )
        return self

    @property
    def is_constant(self) -> bool:
        """
        Whether the programme is a single point, i.e. constant in time.

        Returns
        -------
        is_constant
            `True` for a single point, `False` for a time-varying programme.
        """
        return len(self.value) == 1


class ParticleDefinition(_FrozenModel):
    """
    Reference particle.

    Attributes
    ----------
    mass
        Rest mass energy, in [eV].
    charge
        Number of elementary charges, in [].
    decay_rate
        Decay rate, in [1/s].
    """

    mass: Float = pyd.Field(gt=0)
    charge: Annotated[Int, pyd.BeforeValidator(_to_int_if_integer)]
    decay_rate: Float = pyd.Field(default=0.0, ge=0)

    @pyd.field_validator("charge")
    @classmethod
    def _check_charge(cls, charge: int) -> int:
        """
        Check that `charge` is non-zero.

        Parameters
        ----------
        charge
            Number of elementary charges, in [].

        Returns
        -------
        charge
            The validated charge.

        Raises
        ------
        ValueError
            If `charge` is zero.
        """
        if charge == 0:
            raise ValueError("`charge` must not be 0")
        return charge

    def to_particle_type(self) -> ParticleType:
        """
        Convert to a BLonD particle type.

        Returns
        -------
        particle_type
            The BLonD particle with the same mass, charge and decay rate.
        """
        return ParticleType(
            mass=self.mass,
            charge=self.charge,
            user_decay_rate=self.decay_rate,
        )

    @classmethod
    def from_particle_type(cls, particle_type: ParticleType) -> Self:
        """
        Create from a BLonD particle type.

        Parameters
        ----------
        particle_type
            BLonD particle to convert.

        Returns
        -------
        particle
            The portable particle definition.

        Raises
        ------
        pydantic.ValidationError
            If the charge of `particle_type` is not an integer.
        """
        return cls(
            mass=particle_type.mass,
            charge=particle_type.charge,
            decay_rate=particle_type.user_decay_rate,
        )


class MagneticProgramme(Programme):
    """
    Magnetic cycle programmed along time.

    Attributes
    ----------
    particle
        Reference particle.
    time
        Times of the programmed points, strictly increasing, in [s].
    value
        Values at `time`, in the unit given by `in_unit`.
    interpolator
        Intended BLonD interpolator between points.
    interpolator_kwargs
        Keyword arguments for `interpolator`.
    in_unit
        Quantity of `value`: 'momentum' [eV/c], 'total energy' [eV],
        'kinetic energy' [eV] or 'bending field' [T].
    bending_radius
        Bending radius, in [m]. Required for 'bending field'.
    """

    particle: ParticleDefinition
    in_unit: Literal[
        "momentum", "total energy", "kinetic energy", "bending field"
    ] = "momentum"
    bending_radius: Float | None = pyd.Field(default=None, gt=0)

    @pyd.model_validator(mode="after")
    def _check_bending_radius(self) -> Self:
        """
        Check that `bending_radius` is given for 'bending field'.

        Returns
        -------
        programme
            The validated programme.

        Raises
        ------
        ValueError
            If `bending_radius` is missing for 'bending field'.
        """
        if self.in_unit == "bending field" and self.bending_radius is None:
            raise ValueError(
                "`bending_radius` is required for 'bending field'."
            )
        return self

    @pyd.model_validator(mode="after")
    def _check_positive(self) -> Self:
        """
        Check that all values are positive.

        Returns
        -------
        programme
            The validated programme.

        Raises
        ------
        ValueError
            If any value is not above zero.
        """
        _reject_values(
            self.value,
            lambda value: value <= 0,
            f"'{self.in_unit}' must be > 0",
        )
        return self

    @pyd.model_validator(mode="after")
    def _check_total_energy(self) -> Self:
        """
        Check that a 'total energy' programme exceeds the rest mass.

        Returns
        -------
        programme
            The validated programme.

        Raises
        ------
        ValueError
            If any total energy is not above `particle.mass`.
        """
        if self.in_unit == "total energy":
            _reject_values(
                self.value,
                lambda energy: energy <= self.particle.mass,
                "'total energy' must exceed `particle.mass`",
            )
        return self

    def to_magnetic_cycle(
        self,
    ) -> ConstantMagneticCycle | MagneticCycleByTime:
        """
        Convert to a BLonD magnetic cycle.

        Returns
        -------
        magnetic_cycle
            A `ConstantMagneticCycle` for a single point, otherwise a
            `MagneticCycleByTime`.
        """
        reference_particle = self.particle.to_particle_type()
        if self.is_constant:
            return ConstantMagneticCycle(
                reference_particle=reference_particle,
                value=self.value[0],
                in_unit=self.in_unit,
                bending_radius=self.bending_radius,
            )

        else:
            return MagneticCycleByTime(
                reference_particle=reference_particle,
                reference_time=np.array(self.time),
                reference_values=np.array(self.value),
                in_unit=self.in_unit,
                bending_radius=self.bending_radius,
                interpolator=_INTERPOLATOR_TYPES[self.interpolator],
                **self.interpolator_kwargs,
            )

    @classmethod
    def from_magnetic_cycle(cls, magnetic_cycle: MagneticCycleBase) -> Self:
        """
        Create from a BLonD magnetic cycle.

        Parameters
        ----------
        magnetic_cycle
            A `ConstantMagneticCycle` or a `MagneticCycleByTime`.

        Returns
        -------
        programme
            The portable magnetic programme.

        Raises
        ------
        TypeError
            If `magnetic_cycle` is of any other type.
        ValueError
            If the interpolator of a `MagneticCycleByTime` is not supported.
        """
        particle = ParticleDefinition.from_particle_type(
            magnetic_cycle.reference_particle
        )
        if isinstance(magnetic_cycle, ConstantMagneticCycle):
            return cls(
                particle=particle,
                time=(0.0,),
                value=(float(magnetic_cycle.value),),
                in_unit=magnetic_cycle.in_unit,
                bending_radius=magnetic_cycle.bending_radius,
            )

        elif isinstance(magnetic_cycle, MagneticCycleByTime):
            return cls(
                particle=particle,
                time=magnetic_cycle.reference_time,
                value=magnetic_cycle.reference_values,
                in_unit=magnetic_cycle.in_unit,
                bending_radius=magnetic_cycle.bending_radius,
                interpolator=_interpolator_name(
                    magnetic_cycle.interpolator_type
                ),
                interpolator_kwargs=magnetic_cycle.interpolator_kwargs,
            )

        else:
            raise TypeError(
                "Only `ConstantMagneticCycle` and `MagneticCycleByTime` are "
                f"supported, got {type(magnetic_cycle).__name__}"
            )


class RFHarmonicDefinition(_FrozenModel):
    """
    Programmes of one RF harmonic.

    Attributes
    ----------
    harmonic
        Harmonic number, in [].
    voltage
        Effective voltage, in [V].
    phi_rf_design
        Design phase, in [rad].
    """

    harmonic: Programme
    voltage: Programme
    phi_rf_design: Programme

    @pyd.model_validator(mode="after")
    def _check_ranges(self) -> Self:
        """
        Check that `harmonic` is positive and `voltage` non-negative.

        Returns
        -------
        harmonic_definition
            The validated harmonic.

        Raises
        ------
        ValueError
            If a value is out of range.
        """
        _reject_values(
            self.harmonic.value, lambda h: h <= 0, "`harmonic` must be > 0"
        )
        _reject_values(
            self.voltage.value, lambda v: v < 0, "`voltage` must be >= 0"
        )
        return self


class RFStationDefinition(_FrozenModel):
    """
    RF station with one or more harmonics.

    Attributes
    ----------
    harmonics
        Programmes per harmonic.
    main_harmonic_idx
        Index of the main harmonic in `harmonics`.
    section_index
        Section of the ring the station belongs to.
    name
        Optional station name.
    """

    harmonics: tuple[RFHarmonicDefinition, ...] = pyd.Field(min_length=1)
    main_harmonic_idx: Int = pyd.Field(default=0, ge=0)
    section_index: Int = pyd.Field(default=0, ge=0)
    name: str | None = None

    @pyd.model_validator(mode="after")
    def _check_main_harmonic_idx(self) -> Self:
        """
        Check that `main_harmonic_idx` indexes `harmonics`.

        Returns
        -------
        station
            The validated station.

        Raises
        ------
        ValueError
            If `main_harmonic_idx` is out of range.
        """
        if self.main_harmonic_idx >= len(self.harmonics):
            raise ValueError(
                f"`main_harmonic_idx`={self.main_harmonic_idx} out of range "
                f"for {len(self.harmonics)} harmonics"
            )
        return self

    def to_rf_station(
        self,
    ) -> SingleHarmonicRFStation | MultiHarmonicRFStation:
        """
        Convert to a BLonD RF station.

        Single-point programmes set the plain value, all others are
        scheduled in time. The programmes of all harmonics of an attribute
        are evaluated on the union of their time axes and scheduled together
        with the interpolator of the main harmonic.

        Returns
        -------
        rf_station
            A `SingleHarmonicRFStation` for one harmonic, otherwise a
            `MultiHarmonicRFStation`.

        Warns
        -----
        UserWarning
            If the harmonics of an attribute use different interpolators.
        """
        is_single = len(self.harmonics) == 1
        constants = {}
        schedules = {}

        for attribute in RFHarmonicDefinition.model_fields:
            programmes = [getattr(h, attribute) for h in self.harmonics]

            if all(programme.is_constant for programme in programmes):
                values = [programme.value[0] for programme in programmes]
                constants[attribute] = (
                    values[0] if is_single else np.array(values)
                )

            else:
                schedules[attribute] = _to_scheduled_interpolation(
                    attribute, programmes, self.main_harmonic_idx
                )

        arguments = {
            "voltage": constants.get("voltage"),
            "phi_rf": constants.get("phi_rf_design"),
            "harmonic": constants.get("harmonic"),
            "section_index": self.section_index,
            "name": self.name,
        }

        if is_single:
            rf_station = SingleHarmonicRFStation(**arguments)

        else:
            rf_station = MultiHarmonicRFStation(
                n_harmonics=len(self.harmonics),
                main_harmonic_idx=self.main_harmonic_idx,
                **arguments,
            )

        for attribute, scheduled in schedules.items():
            rf_station.schedule(attribute, scheduled)

        return rf_station

    @classmethod
    def from_rf_station(
        cls, rf_station: SingleHarmonicRFStation | MultiHarmonicRFStation
    ) -> Self:
        """
        Create from a BLonD RF station.

        Feedbacks, wakefields and kick settings of the station are not
        represented. Harmonics scheduled together share one time axis and
        interpolator.

        Parameters
        ----------
        rf_station
            A `SingleHarmonicRFStation` or `MultiHarmonicRFStation`.

        Returns
        -------
        station
            The portable RF station definition.

        Raises
        ------
        TypeError
            If `rf_station` is of another type, or an attribute is scheduled
            by anything but a `ScheduledInterpolation`.
        ValueError
            If an attribute is neither set nor scheduled.
        """
        if not isinstance(
            rf_station, SingleHarmonicRFStation | MultiHarmonicRFStation
        ):
            raise TypeError(
                "Only `SingleHarmonicRFStation` and `MultiHarmonicRFStation` "
                f"are supported, got {type(rf_station).__name__}"
            )
        programmes = {
            attribute: _read_programmes(rf_station, attribute)
            for attribute in RFHarmonicDefinition.model_fields
        }
        return cls(
            harmonics=tuple(
                RFHarmonicDefinition(
                    **{
                        a: programmes[a][i]
                        for a in RFHarmonicDefinition.model_fields
                    }
                )
                for i in range(rf_station.n_rf)
            ),
            main_harmonic_idx=getattr(rf_station, "main_harmonic_idx", 0),
            section_index=rf_station.section_index,
            name=rf_station.name,
        )


class DriftDefinition(_FrozenModel):
    """
    Longitudinal drift between RF stations.

    Attributes
    ----------
    orbit_length
        Length of the reference orbit, in [m].
    momentum_compaction_factor
        Momentum compaction factor alpha_0, in [].
    higher_order_alpha
        Higher-order momentum compaction factors, one programme per order
        (alpha_1, alpha_2, ...), in []. `None` means no higher orders.
    section_index
        Section of the ring the drift belongs to.
    radiation_integrals
        Synchrotron radiation integrals of the drift. `None` means none.
    name
        Optional drift name.
    """

    orbit_length: Float = pyd.Field(gt=0)
    momentum_compaction_factor: Programme
    higher_order_alpha: tuple[Programme, ...] | None = None
    section_index: Int = pyd.Field(default=0, ge=0)
    radiation_integrals: Annotated[
        tuple[Float, ...] | None, pyd.BeforeValidator(_from_array)
    ] = None
    name: str | None = None

    def to_drift(self) -> DriftSimple | DriftExact:
        """
        Convert to a BLonD drift.

        Single-point programmes set the plain value, all others are
        scheduled in time. The orders of `higher_order_alpha` are evaluated
        on the union of their time axes and scheduled together with the
        interpolator of the first order.

        Returns
        -------
        drift
            A `DriftExact` if `higher_order_alpha` is given, otherwise a
            `DriftSimple`.

        Warns
        -----
        UserWarning
            If the orders of `higher_order_alpha` use different
            interpolators.
        """
        arguments = {
            "orbit_length": self.orbit_length,
            "section_index": self.section_index,
            "name": self.name,
        }
        if self.radiation_integrals is not None:
            arguments["radiation_integrals"] = np.array(
                self.radiation_integrals
            )
        programmes = {
            "momentum_compaction_factor": [self.momentum_compaction_factor]
        }
        if self.higher_order_alpha is not None:
            programmes["higher_order_alpha"] = list(self.higher_order_alpha)

        schedules = {}
        for attribute, attribute_programmes in programmes.items():
            if all(p.is_constant for p in attribute_programmes):
                values = np.array([p.value[0] for p in attribute_programmes])
                arguments[attribute] = (
                    values[0]
                    if attribute == "momentum_compaction_factor"
                    else values
                )
            else:
                schedules[attribute] = _to_scheduled_interpolation(
                    attribute, attribute_programmes, 0
                )

        is_exact = self.higher_order_alpha is not None
        drift_type = DriftExact if is_exact else DriftSimple
        drift = drift_type(**arguments)

        for attribute, scheduled in schedules.items():
            drift.schedule(attribute, scheduled)

        return drift

    @classmethod
    def from_drift(cls, drift: DriftSimple | DriftExact) -> Self:
        """
        Create from a BLonD drift.

        A `DriftExact` without `higher_order_alpha` is indistinguishable from
        a `DriftSimple` and is converted as such. Orders scheduled together
        share one time axis and interpolator.

        Parameters
        ----------
        drift
            A `DriftSimple` or `DriftExact`.

        Returns
        -------
        drift_definition
            The portable drift definition.

        Raises
        ------
        TypeError
            If `drift` is of another type, or an attribute is scheduled by
            anything but a `ScheduledInterpolation`.
        ValueError
            If the momentum compaction factor is neither set nor scheduled.
        """
        if type(drift) not in (DriftSimple, DriftExact):
            raise TypeError(
                "Only `DriftSimple` and `DriftExact` are supported, got "
                f"{type(drift).__name__}"
            )
        higher_order_alpha = None
        if isinstance(drift, DriftExact) and (
            drift.higher_order_alpha is not None
            or "higher_order_alpha" in drift.schedules
        ):
            higher_order_alpha = tuple(
                _read_programmes(drift, "higher_order_alpha")
            )
        radiation_integrals = drift.radiation_integrals
        return cls(
            orbit_length=drift.orbit_length,
            momentum_compaction_factor=_read_programmes(
                drift, "momentum_compaction_factor"
            )[0],
            higher_order_alpha=higher_order_alpha,
            radiation_integrals=(
                None
                if radiation_integrals is None
                else tuple(float(value) for value in radiation_integrals)
            ),
            section_index=drift.section_index,
            name=drift.name,
        )


class CycleDefinition(_FrozenModel):
    """
    Complete portable cycle definition.

    Attributes
    ----------
    circumference
        Ring circumference, in [m].
    magnetic_cycle
        Magnetic programme.
    rf_stations
        RF stations.
    drifts
        Drifts.
    name
        Optional cycle name.
    schema_version
        Version of this schema; other versions are rejected.
    """

    circumference: Float = pyd.Field(gt=0)
    magnetic_cycle: MagneticProgramme
    rf_stations: tuple[RFStationDefinition, ...] = ()
    drifts: tuple[DriftDefinition, ...] = ()
    name: str | None = None
    schema_version: Literal[1] = SCHEMA_VERSION

    @pyd.model_validator(mode="after")
    def _check_one_rf_station_per_section(self) -> Self:
        """
        Check that no section holds several RF stations.

        Returns
        -------
        cycle
            The validated cycle.

        Raises
        ------
        ValueError
            If a section has more than one RF station.
        """
        sections = [station.section_index for station in self.rf_stations]
        repeated = sorted({i for i in sections if sections.count(i) > 1})
        if repeated:
            raise ValueError(
                "At most one RF station per section is allowed, got several "
                f"in sections {repeated}"
            )
        return self

    @pyd.model_validator(mode="after")
    def _check_circumference(self) -> Self:
        """
        Check that the drift orbit lengths add up to the circumference.

        The comparison is the one of `Ring.assert_circumference`. Cycles
        without drifts are not checked.

        Returns
        -------
        cycle
            The validated cycle.

        Raises
        ------
        ValueError
            If the orbit lengths of the drifts do not sum to `circumference`.
        """
        if self.drifts:
            total_length = sum(drift.orbit_length for drift in self.drifts)
            if not np.isclose(total_length, self.circumference, atol=1e-6):
                raise ValueError(
                    f"Drift orbit lengths sum to {total_length} m, but "
                    f"`circumference` is {self.circumference} m"
                )
        return self

    def to_ring(self) -> Ring:
        """
        Convert to a BLonD ring holding the RF stations and drifts.

        Elements are ordered by `section_index`, then by BLonD's natural
        order within a section (RF station before drift). The ring's
        radiation integrals are the sum of those of the drifts, if all
        drifts have them. The magnetic cycle is not part of the ring, use
        `magnetic_cycle.to_magnetic_cycle()`.

        Returns
        -------
        ring
            The ring with all elements added.

        Warns
        -----
        UserWarning
            If the harmonics of an RF station attribute use different
            interpolators.
        """
        drifts = [drift.to_drift() for drift in self.drifts]
        radiation_integrals = None

        if self.drifts and all(
            drift.radiation_integrals is not None for drift in self.drifts
        ):
            radiation_integrals = np.sum(
                [drift.radiation_integrals for drift in self.drifts], axis=0
            )

        ring = Ring(
            circumference=self.circumference,
            radiation_integrals=radiation_integrals,
        )
        ring.add_elements(
            [station.to_rf_station() for station in self.rf_stations] + drifts,
            reorder=True,
        )

        return ring

    @classmethod
    def from_ring(
        cls,
        ring: Ring,
        magnetic_cycle: MagneticCycleBase,
        name: str | None = None,
    ) -> Self:
        """
        Create from a BLonD ring and its magnetic cycle.

        Only RF stations and `DriftSimple` / `DriftExact` drifts are
        represented, in the order of the ring. All other elements are skipped
        with a warning. Radiation integrals of the ring itself are not
        represented, those of the drifts are.

        Parameters
        ----------
        ring
            BLonD ring holding the elements.
        magnetic_cycle
            A `ConstantMagneticCycle` or a `MagneticCycleByTime`.
        name
            Optional cycle name.

        Returns
        -------
        cycle
            The portable cycle definition.

        Raises
        ------
        TypeError
            If `magnetic_cycle` is of another type, or an attribute of an RF
            station or drift is scheduled by anything but a
            `ScheduledInterpolation`.
        ValueError
            If an attribute of an RF station or drift is neither set nor
            scheduled, or its interpolator is not supported.
        pydantic.ValidationError
            If the result is not a valid cycle, e.g. the drift orbit lengths
            do not sum to the circumference.

        Warns
        -----
        UserWarning
            If the ring holds elements that are not represented.
        """
        rf_stations = []
        drifts = []
        skipped = []

        for element in ring.elements.elements:
            if isinstance(
                element, SingleHarmonicRFStation | MultiHarmonicRFStation
            ):
                rf_stations.append(
                    RFStationDefinition.from_rf_station(element)
                )
            elif type(element) in (DriftSimple, DriftExact):
                drifts.append(DriftDefinition.from_drift(element))
            else:
                skipped.append(
                    f"'{getattr(element, 'name', '?')}' "
                    f"({type(element).__name__})"
                )

        if skipped:
            warnings.warn(
                f"Skipped elements that are not represented: "
                f"{', '.join(skipped)}",
                UserWarning,
                stacklevel=2,
            )

        return cls(
            circumference=ring.circumference,
            magnetic_cycle=MagneticProgramme.from_magnetic_cycle(
                magnetic_cycle
            ),
            rf_stations=tuple(rf_stations),
            drifts=tuple(drifts),
            name=name,
        )
