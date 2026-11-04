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
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic as pyd

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable

SCHEMA_VERSION = 1


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
    value: Annotated[
        tuple[Float, ...], pyd.BeforeValidator(_from_array)
    ] = pyd.Field(min_length=1)
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
    def _check_points(self) -> Programme:
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
    def _check_interpolator_kwargs(self) -> Programme:
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


class MagneticProgramme(Programme):
    """
    Magnetic cycle programmed along time.

    Attributes
    ----------
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

    in_unit: Literal[
        "momentum", "total energy", "kinetic energy", "bending field"
    ] = "momentum"
    bending_radius: Float | None = pyd.Field(default=None, gt=0)

    @pyd.model_validator(mode="after")
    def _check_bending_radius(self) -> MagneticProgramme:
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
    def _check_positive(self) -> MagneticProgramme:
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
    def _check_ranges(self) -> RFHarmonicDefinition:
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
    def _check_main_harmonic_idx(self) -> RFStationDefinition:
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
    name
        Optional drift name.
    """

    orbit_length: Float = pyd.Field(gt=0)
    momentum_compaction_factor: Programme
    higher_order_alpha: tuple[Programme, ...] | None = None
    section_index: Int = pyd.Field(default=0, ge=0)
    name: str | None = None


class CycleDefinition(_FrozenModel):
    """
    Complete portable cycle definition.

    Attributes
    ----------
    particle
        Reference particle.
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

    particle: ParticleDefinition
    circumference: Float = pyd.Field(gt=0)
    magnetic_cycle: MagneticProgramme
    rf_stations: tuple[RFStationDefinition, ...] = ()
    drifts: tuple[DriftDefinition, ...] = ()
    name: str | None = None
    schema_version: Literal[1] = SCHEMA_VERSION

    @pyd.model_validator(mode="after")
    def _check_total_energy(self) -> CycleDefinition:
        """
        Check that a 'total energy' programme exceeds the rest mass.

        Returns
        -------
        cycle
            The validated cycle.

        Raises
        ------
        ValueError
            If any total energy is not above `particle.mass`.
        """
        if self.magnetic_cycle.in_unit == "total energy":
            _reject_values(
                self.magnetic_cycle.value,
                lambda energy: energy <= self.particle.mass,
                "'total energy' must exceed `particle.mass`",
            )
        return self
