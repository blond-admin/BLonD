from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0

from .base import ProgrammedCycle
from .._core.base import HasPropertyCache

if TYPE_CHECKING:
    from typing import (
        Optional as LateInit,
        TypeVar,
        Optional,
        Literal,
        Union,
    )

    from numpy.typing import NDArray as NumpyArray

    from .._core.simulation.simulation import Simulation

    FloatOrArray = Union[float, NumpyArray]

    T = TypeVar("T")

    SynchronousDataTypes = Literal[
        "momentum", "total energy", "kinetic energy", "bending field"
    ]


def calc_beta(mass: float, momentum: FloatOrArray) -> FloatOrArray:
    """
    Relativistic beta factor (v = beta * c0)

    Parameters
    ----------
    mass : float
        Particle mass in [eV/c²]
    momentum : float or NDArray
        Particle momentum in [eV/c]

    Returns
    -------
    beta : float or NDArray
        Relativistic beta factor (unitless), such that v = beta * c
    """
    return np.sqrt(1 / (1 + (mass / momentum) ** 2))


def calc_gamma(mass: float, momentum: FloatOrArray) -> FloatOrArray:
    """
    Relativistic gamma factor (Lorentz factor)

    Parameters
    ----------
    mass : float
        Particle mass in [eV/c²]
    momentum : float or NDArray
        Particle momentum in [eV/c]

    Returns
    -------
    gamma : float or NDArray
        Lorentz factor (unitless)
    """
    return np.sqrt(1 + (momentum / mass) ** 2)


def calc_total_energy(mass: float, momentum: FloatOrArray) -> FloatOrArray:
    """
    Total relativistic energy of the particle

    Parameters
    ----------
    mass : float
        Particle mass in [eV/c²]
    momentum : float or NDArray
        Particle momentum in [eV/c]

    Returns
    -------
    energy : float or NDArray
        Total relativistic energy in [eV]
    """
    return np.sqrt(momentum**2 + mass**2)


def calc_energy_kin(mass: float, momentum: FloatOrArray) -> FloatOrArray:
    """
    Relativistic kinetic energy of the particle

    Parameters
    ----------
    mass : float
        Particle mass in [eV/c²]
    momentum : float or NDArray
        Particle momentum in [eV/c]

    Returns
    -------
    kinetic_energy : float or NDArray
        Kinetic energy in [eV], defined as total energy - rest energy
    """
    return calc_total_energy(mass, momentum) - mass


class EnergyCycleBase(ProgrammedCycle, HasPropertyCache):
    """Programmed energy program of the synchrotron"""

    def __init__(
        self,
    ):
        super().__init__()
        self._momentum_init: LateInit[float] = None
        self._mass: LateInit[float] = None
        self._n_turns: None | int = None

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        self._momentum_init = kwargs["momentum_init"]
        self._n_turns = kwargs["n_turns"]
        self._mass = simulation.beams[0].particle_type.mass
        self.invalidate_cache()

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        self.invalidate_cache()

    @property
    def n_turns(self):
        """Number of turns that are defined by this cycle"""
        return self._n_turns

    @abstractmethod
    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
    ):
        """
        Calculate the total energy [eV] that is foreseen by the energy cycle

        Parameters
        ----------
        turn_i
            Currently turn index
            (Eventually needed for array accessing)
        section_i
            Currently section index
            (Eventually needed for array accessing)
        reference_time
            Current reference time
            (Eventually needed for interpolation)

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def headless(*args, **kwargs):
        pass

    @cached_property  # as readonly attributes
    def total_energy_init(self):
        energy_init = calc_total_energy(mass=self._mass, momentum=self._momentum_init)
        return energy_init

    cached_props = ("total_energy_init",)

    def invalidate_cache(self):
        super()._invalidate_cache(EnergyCycleBase.cached_props)


class ConstantEnergyCycle(EnergyCycleBase):
    def __init__(
        self,
        value: float,
        in_unit: SynchronousDataTypes = "momentum",
    ):
        super().__init__()
        self._value = value
        self._in_unit = in_unit

        self._momentum: LateInit[float] = None
        self._total_energy: LateInit[float] = None

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        self._momentum = _to_momentum(
            data=self._value,
            mass=simulation.beams[0].particle_type.mass,
            charge=simulation.beams[0].particle_type.charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )

        super().on_init_simulation(
            simulation=simulation,
            momentum_init=self._momentum,
            n_turns=None,
        )
        self._total_energy = calc_total_energy(mass=self._mass, momentum=self._momentum)

    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
    ):
        # constant because ConstantEnergyCycle
        return self._total_energy

    @staticmethod
    def headless(
        value: float,
        mass: float,
        charge: float,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
    ) -> ConstantEnergyCycle:
        ret = ConstantEnergyCycle(
            value=value,
            in_unit=in_unit,
        )
        from .._core.beam.base import BeamBaseClass
        from .._core.beam.particle_types import ParticleType
        from .._core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.ring.bending_radius = bending_radius
        beam = Mock(BeamBaseClass)
        beam.particle_type = Mock(ParticleType)

        beam.particle_type.mass = mass
        beam.particle_type.charge = charge
        simulation.beams = (beam,)
        ret.on_init_simulation(simulation=simulation)
        return ret


class EnergyCyclePerTurn(EnergyCycleBase):
    def __init__(
        self,
        value_init: float,
        values_after_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
    ):
        super().__init__()
        self._value_init = value_init

        assert len(values_after_turn.shape) == 1, (
            f"Expected 1D array, but got {values_after_turn.shape}"
        )

        self._values_after_turn = values_after_turn[:]
        self._n_turns = self._values_after_turn.shape[0]
        self._in_unit = in_unit

        self._momentum: LateInit[NumpyArray] = None

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        mass = simulation.beams[0].particle_type.mass
        charge = simulation.beams[0].particle_type.charge
        n_cavities = simulation.ring.n_cavities

        momentum_init = _to_momentum(
            data=self._value_init,
            mass=mass,
            charge=charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        momentum_per_turn = _to_momentum(
            data=self._values_after_turn,
            mass=mass,
            charge=charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        assert n_cavities > 0
        n_turns = self._n_turns
        shape = (n_cavities, n_turns)
        _momentum = np.empty(shape)
        # assume that each cavity gives an
        # even part of the kick
        stair_like = np.linspace(1 / n_cavities, 1, n_cavities, endpoint=True)
        base = np.concatenate(([momentum_init], momentum_per_turn))
        step = np.diff(base)
        for cav_i in range(n_cavities):
            _momentum[cav_i, :] = base[:-1] + stair_like[cav_i] * step

        super().on_init_simulation(
            simulation=simulation,
            momentum_init=momentum_init,
            momentum=_momentum,
            n_turns=n_turns,
        )
        self._momentum = _momentum

    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
    ):
        return calc_total_energy(
            mass=self._mass,
            momentum=self._momentum[section_i, turn_i],
        )

    @staticmethod
    def headless(
        value_init: float,
        mass: float,
        section_lengths: NumpyArray,
        charge: float,
        values_after_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
    ) -> EnergyCyclePerTurn:
        ret = EnergyCyclePerTurn(
            value_init=value_init,
            values_after_turn=values_after_turn,
            in_unit=in_unit,
        )

        from .._core.simulation.simulation import Simulation
        from .._core.beam.base import BeamBaseClass
        from .._core.beam.particle_types import ParticleType

        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        beam.particle_type = Mock(ParticleType)
        simulation.beams = (beam,)
        simulation.ring.section_lengths = section_lengths

        simulation.ring.bending_radius = bending_radius
        beam.particle_type.mass = mass
        beam.particle_type.charge = charge
        simulation.ring.n_cavities = len(section_lengths)
        ret.on_init_simulation(simulation=simulation)
        ret.on_run_simulation(
            simulation=simulation,
            n_turns=len(values_after_turn),
            turn_i_init=0,
        )

        return ret


class EnergyCyclePerTurnAllCavities(EnergyCycleBase):
    def __init__(
        self,
        value_init: float,
        values_after_cavity_per_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
    ):
        super().__init__()
        self._value_init = value_init
        self._values_after_cavity_per_turn = values_after_cavity_per_turn[:, :]
        self._n_turns = self._values_after_cavity_per_turn.shape[1]
        self._in_unit = in_unit

        self._momentum_after_cavity_per_turn: LateInit[NumpyArray] = None

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        mass = simulation.beams[0].particle_type.mass
        charge = simulation.beams[0].particle_type.charge
        momentum_init = _to_momentum(
            data=self._value_init,
            mass=mass,
            charge=charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        momentum_after_cavity_per_turn = _to_momentum(
            data=self._values_after_cavity_per_turn[:, :],
            mass=mass,
            charge=charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        n_cavities = simulation.ring.n_cavities
        n_turns = momentum_after_cavity_per_turn.shape[1]
        assert n_cavities == momentum_after_cavity_per_turn.shape[0], (
            f"{n_cavities=}, but {momentum_after_cavity_per_turn.shape=}"
        )

        super().on_init_simulation(
            simulation=simulation,
            momentum_init=momentum_init,
            n_turns=n_turns,
        )
        self._momentum_after_cavity_per_turn = momentum_after_cavity_per_turn

    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
    ):
        return calc_total_energy(
            mass=self._mass,
            momentum=self._momentum_after_cavity_per_turn[section_i, turn_i],
        )

    @staticmethod
    def headless(
        section_lengths: NumpyArray,
        mass: float,
        charge: float,
        value_init: float,
        values_after_cavity_per_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
    ) -> EnergyCyclePerTurnAllCavities:
        assert len(section_lengths) == values_after_cavity_per_turn.shape[0]
        ret = EnergyCyclePerTurnAllCavities(
            value_init=value_init,
            values_after_cavity_per_turn=values_after_cavity_per_turn,
            in_unit=in_unit,
        )
        from .._core.simulation.simulation import Simulation
        from .._core.beam.base import BeamBaseClass
        from .._core.beam.particle_types import ParticleType

        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        beam.particle_type = Mock(ParticleType)
        simulation.beams = (beam,)

        simulation.ring.section_lengths = section_lengths
        simulation.ring.bending_radius = bending_radius
        beam.particle_type.mass = mass
        beam.particle_type.charge = charge
        simulation.ring.n_cavities = len(section_lengths)

        ret.on_init_simulation(simulation=simulation)
        ret.on_run_simulation(
            simulation=simulation,
            n_turns=values_after_cavity_per_turn.shape[1],
            turn_i_init=0,
        )

        return ret


class EnergyCycleByTime(EnergyCycleBase):
    def __init__(
        self,
        t0: float,
        base_time: NumpyArray,
        base_values: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        interpolator=np.interp,
    ):
        super().__init__()
        self._t0 = t0
        self._interpolator = interpolator
        self._base_time = base_time[:]
        self._base_values = base_values[:]
        self._in_unit = in_unit

        self._base_momentum: LateInit[NumpyArray] = None

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        mass = simulation.beams[0].particle_type.mass
        charge = simulation.beams[0].particle_type.charge

        base_momentum = _to_momentum(
            data=self._base_values,
            mass=mass,
            charge=charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        self._base_momentum = base_momentum
        _momentum_init = self._interpolator(self._t0, self._base_time, base_momentum)
        super().on_init_simulation(
            simulation=simulation,
            momentum_init=_momentum_init,
            n_turns=None,
        )

    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
    ):
        momentum = self._interpolator(self._t0, self._base_time, self._base_momentum)
        return calc_total_energy(
            mass=self._mass,
            momentum=momentum,
        )

    @staticmethod
    def headless(
        mass: float,
        charge: float,
        t0: float,
        base_time: NumpyArray,
        base_values: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
        interpolator=np.interp,
    ) -> EnergyCycleByTime:
        from .._core.simulation.simulation import Simulation
        from .._core.beam.base import BeamBaseClass
        from .._core.beam.particle_types import ParticleType

        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        beam.particle_type = Mock(ParticleType)
        simulation.beams = (beam,)

        beam.particle_type.mass = mass
        beam.particle_type.charge = charge
        simulation.ring.bending_radius = bending_radius

        ret = EnergyCycleByTime(
            t0=t0,
            base_time=base_time,
            base_values=base_values,
            in_unit=in_unit,
            interpolator=interpolator,
        )

        ret.on_init_simulation(simulation=simulation)
        ret.on_run_simulation(simulation=simulation, n_turns=1, turn_i_init=0)

        return ret


def _to_momentum(
    data: int | float | NumpyArray,
    mass: float,
    charge: float,
    convert_from: SynchronousDataTypes = "momentum",
    bending_radius: Optional[float] = None,
) -> NumpyArray:
    """Unit conversion for different input data types

    Parameters
    ----------
    data
        The data to convert, in units of [eV/c], [eV] or [T]
    mass
        The mass of the particles in [eV/c**2]
    charge
        Particle charge, i.e. number of elementary charges `e`
    convert_from
        What units `data` given in:
        - 'momentum' [eV/c], (no conversion is done)
        - 'total energy' [eV],
        - 'kinetic energy' [eV], or
        - 'bending field' [T] (requires `bending_radius`)
    bending_radius
        Bending radius in [m] in case `convert_from` is 'bending field'

    Returns
    -------
    momentum : float array
        The data in units of momentum [eV/c]

    """

    if convert_from == "momentum":
        momentum = data
    elif convert_from == "total energy":
        momentum = np.sqrt(data**2 - mass**2)
    elif convert_from == "kinetic energy":
        momentum = np.sqrt((data + mass) ** 2 - mass**2)
    elif convert_from == "bending field":
        if bending_radius is None:
            raise ValueError(
                "The 'bending_radius' parameter must be provided and cannot be None."
            )
        momentum = data * bending_radius * charge * c0
    else:
        raise ValueError(f"Unrecognized option {convert_from=}")

    return momentum


def beta_by_momentum(momentum: T, mass: float) -> T:
    return np.sqrt(1 / (1 + (mass / momentum) ** 2))
