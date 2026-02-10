# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Several classes to manage and describe the ramp of the magnets.

Notes
-----
The following classes are currently available:
- :class:`~blond.cycles.magnetic_cycles.ConstantMagneticCycle`
- :class:`~blond.cycles.magnetic_cycles.MagneticCyclePerTurn`
- :class:`~blond.cycles.magnetic_cycles.MagneticCyclePerTurnAllRFStations`
- :class:`~blond.cycles.magnetic_cycles.MagneticCycleByTime`

Authors:
Simon Lauber
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0
from scipy.interpolate import interp1d

from blond.acc_math.analytic.simple_math import calc_total_energy
from blond.core.base import HasPropertyCache
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.particle_types import ParticleType, proton
from blond.cycles.base import ProgrammedCycle

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Literal, TypeVar

    from numpy.typing import NDArray as NumpyArray
    from scipy.interpolate import (
        Akima1DInterpolator,
        PchipInterpolator,
    )

    from blond.core.simulation.simulation import Simulation
    from blond.generals.protocols import AnyInterpolator

    FloatOrArray = float | NumpyArray

    T = TypeVar("T")

    SynchronousDataTypes = Literal[
        "momentum",
        "total energy",
        "kinetic energy",
        "bending field",
    ]


class MagneticCycleBase(ProgrammedCycle, HasPropertyCache):
    """
    Programmed magnetic cycle of the synchrotron.

    Parameters
    ----------
    reference_particle
        Type of particles, e.g. protons.
    magnetic_rigidity_init
        Initial magnetic rigidity.
    """

    def __init__(
        self,
        reference_particle: ParticleType,
        magnetic_rigidity_init: float,
    ):
        super().__init__()
        assert isinstance(reference_particle, ParticleType), (
            f"{type(reference_particle)}"
        )
        self._reference_particle: ParticleType = reference_particle

        self._magnetic_rigidity_before_turn_0: float = magnetic_rigidity_init
        self._n_turns_max: None | int = None

    def on_init_simulation(
        self,
        simulation: Simulation,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Additional keyword arguments.
        """
        super().on_init_simulation(simulation=simulation)
        self._n_turns_max = kwargs["n_turns_max"]

        self.invalidate_cache()

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
            Simulation :class:`~blond._cycles_core.beam.beam.Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )
        self.invalidate_cache()

    @property
    def reference_particle(self) -> ParticleType:
        """
        Reference particle type for the magnetic cycle.

        Returns
        -------
        reference_particle
            Reference particle type for the magnetic cycle.
        """
        return self._reference_particle

    @property
    def n_turns(self) -> None | int:
        """
        Number of turns that are defined by this cycle.

        Returns
        -------
        n_turns
            Number of turns that are defined by this cycle.
        """
        return self._n_turns_max

    @abstractmethod  # pragma: no cover
    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
        particle_type: ParticleType,
    ):
        """
        Calculate the total energy [eV] that is foreseen by the magnetic cycle.

        Parameters
        ----------
        turn_i
            Currently turn index.
            (Eventually needed for array accessing).
        section_i
            Currently section index.
            (Eventually needed for array accessing).
        reference_time
            Current reference time.
            (Eventually needed for interpolation).
        particle_type
            Type of particles, e.g. protons.

        Returns
        -------
        reference_total_energy
            The new energy, in [eV].
        """
        pass

    def get_total_energy_init(
        self,
        particle_type: ParticleType,
    ) -> float:
        """
        Compute the initial the total energy [eV] for the initial turn.

        Parameters
        ----------
        particle_type
            Type of particles, e.g. protons.

        Returns
        -------
        reference_total_energy
            The total energy, in [eV].
        """
        total_energy_init = calc_total_energy(
            mass=particle_type.mass,
            momentum=magnetic_rigidity_to_momentum(
                magnetic_rigidity=self._magnetic_rigidity_before_turn_0,
                charge=particle_type.charge,
            ),
        )

        return total_energy_init

    def get_t_rev_init(
        self,
        circumference: float,
        particle_type: ParticleType,
    ) -> float:
        r"""
        Compute the initial revolution period of a reference particle, in [s].

        Parameters
        ----------
        circumference : float
            Reference circumference of the synchrotron, in [m].
        particle_type : ParticleType
            Object containing particle properties (e.g., mass, charge).

        Returns
        -------
        t_rev_init
            Initial revolution period, in [s].

        Notes
        -----
        The revolution period \( T_{\mathrm{rev}} \) is computed as:

        .. math::

            T_{\mathrm{rev}} = \frac{C}{\beta c}

        where:
            - \( C \) is the machine circumference,
            - \( \beta = v / c \) is the normalized velocity,
            - \( c \) is the speed of light.

        The relativistic gamma and beta factors are computed from the total energy of the particle:

        .. math::

            \gamma = \frac{E_{\mathrm{tot}}}{mc^2}, \quad
            \beta = \sqrt{1 - \frac{1}{\gamma^2}}
        """
        reference_total_energy = self.get_total_energy_init(
            particle_type=particle_type,
        )
        reference_gamma = reference_total_energy * particle_type.mass_inv

        reference_beta = np.sqrt(
            1.0 - 1.0 / (reference_gamma * reference_gamma)
        )

        reference_velocity = reference_beta * c0
        return circumference / reference_velocity

    @staticmethod
    @abstractmethod  # pragma: no cover
    def headless(*args, **kwargs):
        """
        Initialize object without simulation context.

        Parameters
        ----------
        *args
            Variable positional arguments.
        **kwargs
            Variable keyword arguments.
        """
        pass

    cached_props = ()

    def invalidate_cache(self):
        """Delete the stored values of functions with @cached_property."""
        super()._invalidate_cache(MagneticCycleBase.cached_props)


class ConstantMagneticCycle(MagneticCycleBase):
    """
    Magnetic cycle for a non-changing magnetic field.

    Parameters
    ----------
    reference_particle
        Type of particles, e.g. protons.
    value
        Constant value of unit `in_unit`.
    in_unit
        - 'momentum' [eV/c], (no conversion is done)
        - 'total energy' [eV],
        - 'kinetic energy' [eV], or
        - 'bending field' [T]
    bending_radius
        To 'bending field' associated bending radius, in [m].
    """

    def __init__(
        self,
        reference_particle: ParticleType,
        value: float,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: float | None = None,
    ):
        self._magnetic_rigidity: float = _to_magnetic_rigidity(
            data=value,
            mass=reference_particle.mass,
            charge=reference_particle.charge,
            convert_from=in_unit,
            bending_radius=(
                bending_radius if in_unit == "bending field" else None
            ),
        )
        super().__init__(
            reference_particle=reference_particle,
            magnetic_rigidity_init=self._magnetic_rigidity,
        )
        self._value = value
        self._in_unit = in_unit
        self._bending_radius = bending_radius

        self._total_energy_cache: dict[int, float] | None = {}

    def on_init_simulation(
        self,
        simulation: Simulation,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Additional keyword arguments.
        """
        super().on_init_simulation(
            simulation=simulation,
            n_turns_max=None,
        )

    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
        particle_type: ParticleType,
    ) -> float:
        """
        Calculate the total energy [eV] that is foreseen by the magnetic cycle.

        Parameters
        ----------
        turn_i
            Currently turn index.
            (Eventually needed for array accessing).
        section_i
            Currently section index.
            (Eventually needed for array accessing).
        reference_time
            Current reference time.
            (Eventually needed for interpolation).
        particle_type
            Type of particles, e.g. protons.

        Returns
        -------
        total_energy
            Total relativistic energy, in [eV].
        """
        # constant because ConstantMagneticCycle
        key = hash(particle_type)
        if key not in self._total_energy_cache:
            self._total_energy_cache[key] = calc_total_energy(
                mass=particle_type.mass,
                momentum=magnetic_rigidity_to_momentum(
                    magnetic_rigidity=self._magnetic_rigidity,
                    charge=particle_type.charge,
                ),
            )
        return self._total_energy_cache[key]

    @staticmethod
    def headless(
        value: float,
        particle_type: ParticleType,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: float | None = None,
    ) -> ConstantMagneticCycle:
        """
        Initialize object without simulation context.

        Parameters
        ----------
        value
            Constant value of unit `in_unit`.
        particle_type
            Type of particles, e.g. protons.
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        bending_radius
            Bending radius, in [m].

        Returns
        -------
        constant_magnetic_cycle
            Initialized ConstantMagneticCycle instance.
        """
        ret = ConstantMagneticCycle(
            value=value,
            in_unit=in_unit,
            reference_particle=proton,
        )
        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.ring.bending_radius = bending_radius

        ret.on_init_simulation(simulation=simulation)
        return ret


class MagneticCyclePerTurn(MagneticCycleBase):
    """
    Magnetic cycle per turn.

    Parameters
    ----------
    reference_particle
        Type of particles, e.g. protons.
    value_init
        Initial value at start of simulation in of unit `in_unit`.
    values_after_turn
        Value after turn in synchrotron in of unit `in_unit`.
    in_unit
        - 'momentum' [eV/c], (no conversion is done)
        - 'total energy' [eV],
        - 'kinetic energy' [eV], or
        - 'bending field' [T]
    bending_radius
        To 'bending field' associated bending radius, in [m].

    Notes
    -----
    Assumes each RF station has the same increment of beam energy.
    """

    def __init__(
        self,
        reference_particle: ParticleType,
        value_init: float,
        values_after_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: float | None = None,
    ):
        magnetic_rigidity_init = _to_magnetic_rigidity(
            data=value_init,
            mass=reference_particle.mass,
            charge=reference_particle.charge,
            convert_from=in_unit,
            bending_radius=(
                bending_radius if in_unit == "bending field" else None
            ),
        )
        super().__init__(
            reference_particle=reference_particle,
            magnetic_rigidity_init=magnetic_rigidity_init,
        )
        self._value_init = value_init

        assert len(values_after_turn.shape) == 1, (
            f"Expected 1D array, but got {values_after_turn.shape}"
        )

        self._values_after_turn = values_after_turn[:]
        self._in_unit = in_unit
        self._bending_radius = bending_radius

        self._magnetic_rigidity: NumpyArray | None = None
        self._momentum_cached: dict[int, NumpyArray] = {}
        self._total_energy_cached: dict[int, NumpyArray] = {}

    def on_init_simulation(
        self,
        simulation: Simulation,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Additional keyword arguments.
        """
        n_rf_stations = simulation.ring.n_rf_stations
        n_turns_max = self._values_after_turn.shape[0]

        magnetic_rigidity_per_turn = _to_magnetic_rigidity(
            data=self._values_after_turn,
            mass=self._reference_particle.mass,
            charge=self._reference_particle.charge,
            convert_from=self._in_unit,
            bending_radius=(
                self._bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        assert n_rf_stations > 0
        shape = (n_rf_stations, n_turns_max)
        _magnetic_rigidity = np.empty(shape)
        # assume that each RF station gives an
        # even part of the kick
        stair_like = np.linspace(
            1 / n_rf_stations, 1, n_rf_stations, endpoint=True
        )
        base = np.concatenate(
            (
                [self._magnetic_rigidity_before_turn_0],
                magnetic_rigidity_per_turn,
            )
        )
        step = np.diff(base)
        for cav_i in range(n_rf_stations):
            _magnetic_rigidity[cav_i, :] = base[:-1] + stair_like[cav_i] * step

        super().on_init_simulation(
            simulation=simulation,
            n_turns_max=n_turns_max,
        )
        self._magnetic_rigidity = _magnetic_rigidity

    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
        particle_type: ParticleType,
    ) -> float:
        """
        Calculate the total energy [eV] that is foreseen by the magnetic cycle.

        Parameters
        ----------
        turn_i
            Currently turn index.
            (Eventually needed for array accessing).
        section_i
            Currently section index.
            (Eventually needed for array accessing).
        reference_time
            Current reference time.
            (Eventually needed for interpolation).
        particle_type
            Type of particles, e.g. protons.

        Returns
        -------
        total_energy
            Total relativistic energy, in [eV].
        """
        key = hash(particle_type)
        if key not in self._momentum_cached:
            self._momentum_cached[key] = magnetic_rigidity_to_momentum(
                magnetic_rigidity=self._magnetic_rigidity[:, :],
                charge=particle_type.charge,
            )
            self._total_energy_cached[key] = calc_total_energy(
                mass=particle_type.mass,
                momentum=self._momentum_cached[key],
            )
        return self._total_energy_cached[key][section_i, int(turn_i)]

    @staticmethod
    def headless(
        reference_particle: ParticleType,
        value_init: float,
        values_after_turn: NumpyArray,
        n_rf_stations: int,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: float | None = None,
    ) -> MagneticCyclePerTurn:
        """
        Initialize object without simulation context.

        Parameters
        ----------
        reference_particle
            Type of particles, e.g. protons.
        value_init
            Initial value at start of simulation in of unit `in_unit`.
        values_after_turn
            Value after turn in Synchrotron in of unit `in_unit`.
        n_rf_stations
            Number of RF stations.
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        bending_radius
            Bending radius, in [m].

        Returns
        -------
        magnetic_cycle_per_turn
            Initialized MagneticCyclePerTurn instance.
        """
        ret = MagneticCyclePerTurn(
            value_init=value_init,
            values_after_turn=values_after_turn,
            in_unit=in_unit,
            reference_particle=reference_particle,
        )

        from blond.core.beam.base import BeamBaseClass
        from blond.core.beam.particle_types import ParticleType
        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        beam.particle_type = Mock(ParticleType)

        simulation.ring.bending_radius = bending_radius
        beam.particle_type = reference_particle
        simulation.ring.n_rf_stations = n_rf_stations
        ret.on_init_simulation(simulation=simulation)
        ret.on_run_simulation(
            simulation=simulation,
            n_turns=len(values_after_turn),
            beam=beam,
        )

        return ret


class MagneticCyclePerTurnAllRFStations(MagneticCycleBase):
    """
    Magnetic program per turn, defined for each RF station.

    Parameters
    ----------
    reference_particle
        Type of particles, e.g. protons.
    value_init
        Initial value at start of simulation in of unit `in_unit`.
    values_after_rf_station_per_turn
        Value after each RF station and each turn in Synchrotron
         in of unit `in_unit`.
    in_unit
        - 'momentum' [eV/c], (no conversion is done)
        - 'total energy' [eV],
        - 'kinetic energy' [eV], or
        - 'bending field' [T]
    bending_radius
        To 'bending field' associated bending radius, in [m].
    """

    def __init__(
        self,
        reference_particle: ParticleType,
        value_init: float,
        values_after_rf_station_per_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: float | None = None,
    ):
        magnetic_rigidity_init = _to_magnetic_rigidity(
            data=value_init,
            mass=reference_particle.mass,
            charge=reference_particle.charge,
            convert_from=in_unit,
            bending_radius=(
                bending_radius if in_unit == "bending field" else None
            ),
        )
        super().__init__(
            reference_particle=reference_particle,
            magnetic_rigidity_init=magnetic_rigidity_init,
        )
        self._value_init = value_init
        self._values_after_rf_station_per_turn = (
            values_after_rf_station_per_turn[:, :]
        )
        self._n_turns_max = self._values_after_rf_station_per_turn.shape[1]
        self._in_unit = in_unit
        self._bending_radius = bending_radius

        self._magnetic_rigidity_after_rf_station_per_turn: (
            NumpyArray | None
        ) = None
        self._momentum_cached: dict[int, NumpyArray] = {}

    def on_init_simulation(
        self,
        simulation: Simulation,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Additional keyword arguments.
        """
        magnetic_rigidity_after_rf_station_per_turn = _to_magnetic_rigidity(
            data=self._values_after_rf_station_per_turn[:, :],
            mass=self._reference_particle.mass,
            charge=self._reference_particle.charge,
            convert_from=self._in_unit,
            bending_radius=(
                self._bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        n_rf_stations = simulation.ring.n_rf_stations
        n_turns_max = magnetic_rigidity_after_rf_station_per_turn.shape[1]
        assert (
            n_rf_stations
            == magnetic_rigidity_after_rf_station_per_turn.shape[0]
        ), (
            f"{n_rf_stations=}, but {magnetic_rigidity_after_rf_station_per_turn.shape=}"
        )

        super().on_init_simulation(
            simulation=simulation,
            n_turns_max=n_turns_max,
            magnetic_rigidity_init=self._magnetic_rigidity_before_turn_0,
        )
        self._magnetic_rigidity_after_rf_station_per_turn = (
            magnetic_rigidity_after_rf_station_per_turn
        )

    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
        particle_type: ParticleType,
    ):
        """
        Calculate the total energy [eV] that is foreseen by the magnetic cycle.

        Parameters
        ----------
        turn_i
            Currently turn index.
            (Eventually needed for array accessing).
        section_i
            Currently section index.
            (Eventually needed for array accessing).
        reference_time
            Current reference time.
            (Eventually needed for interpolation).
        particle_type
            Type of particles, e.g. protons.

        Returns
        -------
        total_energy
            Total relativistic energy, in [eV].
        """
        key = hash(particle_type)
        if key not in self._momentum_cached:
            self._momentum_cached[key] = magnetic_rigidity_to_momentum(
                magnetic_rigidity=self._magnetic_rigidity_after_rf_station_per_turn[
                    :, :
                ],
                charge=particle_type.charge,
            )
        return calc_total_energy(
            mass=particle_type.mass,
            momentum=self._momentum_cached[key][section_i, turn_i],
        )

    @staticmethod
    def headless(
        reference_particle: ParticleType,
        value_init: float,
        values_after_rf_station_per_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: float | None = None,
    ) -> MagneticCyclePerTurnAllRFStations:
        """
        Initialize object without simulation context.

        Parameters
        ----------
        reference_particle
            Type of particles, e.g. protons.
        value_init
            Initial value at start of simulation in of unit `in_unit`.
        values_after_rf_station_per_turn
            Value after each RF Station and each turn in Synchrotron
             in of unit `in_unit`.
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        bending_radius
            Bending radius, in [m].

        Returns
        -------
        magnetic_cycle
            Fully initialized :class:`~blond.cycles.magnetic_cycles.MagneticCyclePerTurnAllRFStations`.
        """
        ret = MagneticCyclePerTurnAllRFStations(
            value_init=value_init,
            values_after_rf_station_per_turn=values_after_rf_station_per_turn,
            in_unit=in_unit,
            reference_particle=reference_particle,
        )
        from blond.core.beam.base import BeamBaseClass
        from blond.core.beam.particle_types import ParticleType
        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        beam.particle_type = Mock(ParticleType)

        simulation.ring.bending_radius = bending_radius
        beam.particle_type = reference_particle
        simulation.ring.n_rf_stations = values_after_rf_station_per_turn.shape[
            0
        ]

        ret.on_init_simulation(simulation=simulation)
        ret.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=values_after_rf_station_per_turn.shape[1],
        )
        return ret


class MagneticCycleByTime(MagneticCycleBase):
    """
    Magnetic cycle defined as B vs. time, interpolated just in time.

    Parameters
    ----------
    reference_particle
        Type of particles, e.g. protons.
    base_time
        Values of time [s].
    base_values
        Values at time in synchrotron in of unit `in_unit`.
    in_unit
        - 'momentum' [eV/c], (no conversion is done)
        - 'total energy' [eV],
        - 'kinetic energy' [eV], or
        - 'bending field' [T]
    bending_radius
        To 'bending field' associated bending radius, in [m].
    interpolator
        Interpolation routine to get time in between the base values.
        Default: `scipy.interpolate.interp1d`.
    **kwargs
        Optional keyword arguments for the interpolator.

    See Also
    --------
    scipy.interpolate.interp1d : 1D interpolator similar to `np.interp`.
    scipy.interpolate.Akima1DInterpolator : Modified Akima Interpolation.
    scipy.interpolate.PchipInterpolator : Piecewise Cubic Hermite Interpolating Polynomial.

    Examples
    --------
    >>> import scipy
    >>> import numpy as np
    >>> from blond import mu_plus, MagneticCycleByTime
    >>> time_per_turn = 953.338 * 2 * np.pi / scipy.constants.c
    >>> n_turns = 17
    >>> energy_ramp = np.linspace(63e9, 313.83e9 * 100, n_turns)
    >>> energy_cycle = MagneticCycleByTime(
    ...     reference_particle=mu_plus,
    ...     base_time=np.linspace(0, 18 * time_per_turn, n_turns),
    ...     base_values=energy_ramp,
    ...     in_unit="momentum",
    ...     interpolator=scipy.interpolate.Akima1DInterpolator,
    ...     method="makima",
    ... )
    """

    def __init__(
        self,
        reference_particle: ParticleType,
        base_time: NumpyArray,
        base_values: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: float | None = None,
        interpolator: type[
            Akima1DInterpolator
            | PchipInterpolator
            | interp1d
            | AnyInterpolator
        ] = interp1d,
        **kwargs,
    ):
        assert not np.any(np.isnan(base_values)), (
            "NaN occurred in `base_values`"
        )
        assert not np.any(np.isnan(base_time)), "NaN occurred in `base_time`"

        base_magnetic_rigidity = _to_magnetic_rigidity(
            data=base_values,
            mass=reference_particle.mass,
            charge=reference_particle.charge,
            convert_from=in_unit,
            bending_radius=(
                bending_radius if in_unit == "bending field" else None
            ),
        )
        self._base_magnetic_rigidity: NumpyArray = base_magnetic_rigidity

        super().__init__(
            reference_particle=reference_particle,
            magnetic_rigidity_init=base_magnetic_rigidity[0],
        )
        self._interpolator = interpolator(
            base_time[:],
            base_magnetic_rigidity[:],
            **kwargs,
        )
        self._base_values = base_values[:]  # only for debugging
        self._in_unit = in_unit  # only for debugging
        self._bending_radius = bending_radius  # only for debugging

    def on_init_simulation(
        self,
        simulation: Simulation,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Additional keyword arguments.
        """
        super().on_init_simulation(
            simulation=simulation,
            n_turns_max=None,
            **kwargs,
        )

    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
        particle_type: ParticleType,
    ):
        """
        Calculate the total energy [eV] that is foreseen by the magnetic cycle.

        Parameters
        ----------
        turn_i
            Currently turn index.
            (Eventually needed for array accessing).
        section_i
            Currently section index.
            (Eventually needed for array accessing).
        reference_time
            Current reference time.
            (Eventually needed for interpolation).
        particle_type
            Type of particles, e.g. protons.

        Returns
        -------
        total_energy
            Total relativistic energy, in [eV].
        """
        magnetic_rigidity = self._interpolator(reference_time)
        assert not np.isnan(magnetic_rigidity)
        assert not np.isinf(magnetic_rigidity)
        return calc_total_energy(
            mass=particle_type.mass,
            momentum=magnetic_rigidity_to_momentum(
                magnetic_rigidity=magnetic_rigidity,
                charge=particle_type.charge,
            ),
        )

    @staticmethod
    def headless(
        reference_particle: ParticleType,
        base_time: NumpyArray,
        base_values: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: float | None = None,
        interpolator: Akima1DInterpolator
        | PchipInterpolator
        | interp1d
        | AnyInterpolator = interp1d,
    ) -> MagneticCycleByTime:
        """
        Initialize object without simulation context.

        Parameters
        ----------
        reference_particle
            Type of particles, e.g. protons.
            Example: For an electron `charge=-1`.
        base_time
            Values of time [s].
        base_values
            Values at time in synchrotron in of unit `in_unit`.
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        bending_radius
            Bending radius, in [m].
        interpolator
                Interpolation routine to get time in between the base values.
                Default: `scipy.interpolate.interp1d`.

        Returns
        -------
        magnetic_cycle_by_time
            Initialized MagneticCycleByTime instance.

        See Also
        --------
        scipy.interpolate.interp1d : 1D interpolator similar to `np.interp`.
        scipy.interpolate.Akima1DInterpolator : Modified Akima Interpolation.
        scipy.interpolate.PchipInterpolator : Piecewise Cubic Hermite Interpolating Polynomial.
        """
        from blond.core.beam.base import BeamBaseClass
        from blond.core.beam.particle_types import ParticleType
        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        beam.particle_type = Mock(ParticleType)

        beam.particle_type = reference_particle
        simulation.ring.bending_radius = bending_radius

        ret = MagneticCycleByTime(
            base_time=base_time,
            base_values=base_values,
            in_unit=in_unit,
            interpolator=interpolator,
            reference_particle=reference_particle,
        )

        ret.on_init_simulation(simulation=simulation)
        ret.on_run_simulation(
            simulation=simulation,
            n_turns=1,
            beam=beam,
        )

        return ret


def _to_magnetic_rigidity(
    data: int | float | NumpyArray,
    mass: float,
    charge: float,
    convert_from: SynchronousDataTypes = "momentum",
    bending_radius: float | None = None,
) -> NumpyArray | float:
    """
    Unit conversion for different input data types.

    Parameters
    ----------
    data
        The data to convert, in units of [eV/c], [eV] or [T].
    mass
        The mass of the particles in [eV/c**2].
    charge
        Particle charge, i.e. number of elementary charges `e`.
        Example: For an electron `charge=-1`.
    convert_from
        What units `data` given in:
        - 'momentum' [eV/c], (no conversion is done)
        - 'total energy' [eV],
        - 'kinetic energy' [eV], or
        - 'bending field' [T] (requires `bending_radius`)
    bending_radius
        Bending radius in [m] in case `convert_from` is 'bending field'.

    Returns
    -------
    momentum
        The data in units of momentum [eV/c].
    """
    if convert_from == "momentum":
        momentum = data
    elif convert_from == "total energy":
        assert np.all(data > mass), (
            f"The total energy is smaller than the rest mass: {np.min(data)} eV"
            f", but must be bigger than {mass:e} eV."
        )
        momentum = np.sqrt(data**2 - mass**2)
    elif convert_from == "kinetic energy":
        momentum = np.sqrt((data + mass) ** 2 - mass**2)
    elif convert_from == "bending field":
        if bending_radius is None:
            raise ValueError(
                "The 'bending_radius' parameter must be provided and cannot be None."
            )
        momentum = data * bending_radius * np.abs(charge) * c0
    else:
        raise ValueError(f"Unrecognized option {convert_from=}")
    magnetic_rigidity = momentum / (np.abs(charge) * c0)
    return magnetic_rigidity


def magnetic_rigidity_to_momentum(
    magnetic_rigidity: float | NumpyArray,
    charge: float,
) -> float | NumpyArray:
    r"""
    Convert magnetic rigidity to momentum.

    Parameters
    ----------
    magnetic_rigidity
        Magnetic rigidity :math:`B \rho`, in [Tm].
    charge
        Particle charge, i.e. number of elementary charges `e`.
        Example: For an electron `charge=-1`.

    Returns
    -------
    momentum
        Relativistic momentum, in [eV/c].

    Notes
    -----
    The momentum is calculated using the relation:

    .. math::

        p = B \rho \cdot |q| \cdot c

    where:
        - :math:`p`  is the momentum,
        - :math:`B \rho` is the magnetic rigidity,
        - :math:`q`  is the particle charge in units of `e`,
        - :math:`c` is the speed of light in vacuum.
    """
    return magnetic_rigidity * np.abs(charge) * c0
