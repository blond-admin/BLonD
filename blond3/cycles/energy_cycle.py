from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import numpy as np
from scipy.constants import speed_of_light as c0

from .base import ProgrammedCycle
from .._core.base import HasPropertyCache
from ..physics.cavities import CavityBaseClass
from ..physics.drifts import DriftBaseClass

if TYPE_CHECKING:
    from typing import (
        Optional as LateInit,
        TypeVar,
        Optional,
        Literal,
        Union,
        List,
        Tuple,
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
    def __init__(
        self,
    ):
        super().__init__()
        self._momentum_init: LateInit[float] = None
        self._momentum: LateInit[NumpyArray] = None
        self._section_lengths: LateInit[NumpyArray] = None
        self._mass: LateInit[float] = None

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        self._momentum_init = kwargs["momentum_init"]
        self._momentum = kwargs["momentum"]

        assert self._momentum is not None, f"{self._momentum=}"
        assert len(self._momentum.shape) == 2, f"{self._momentum.shape=}"
        self._mass = simulation.beams[0].particle_type.mass
        self._section_lengths = simulation.ring.section_lengths
        self.invalidate_cache()

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        self.invalidate_cache()

    @staticmethod
    @abstractmethod
    def headless(*args, **kwargs):
        pass

    @property
    def momentum(self):
        return self._momentum

    @cached_property
    def n_turns(self):
        return self._momentum.shape[1]

    @cached_property  # as readonly attributes
    def total_energy(self) -> NumpyArray:
        return calc_total_energy(mass=self._mass, momentum=self._momentum[:, :])

    @cached_property  # as readonly attributes
    def total_energy_init(self):
        energy_init = calc_total_energy(mass=self._mass, momentum=self._momentum_init)
        return energy_init

    # @cached_property  # as readonly attributes
    # def omega_rev(self) -> NumpyArray:
    #    return 2 * np.pi * self.f_rev  # todo

    cached_props = (
        "n_turns",
        "total_energy",
        "total_energy_init",
    )

    def invalidate_cache(self):
        super()._invalidate_cache(EnergyCycleBase.cached_props)


class ConstantEnergyCycle(EnergyCycleBase):
    def __init__(
        self,
        value: float,
        max_turns: int,
        in_unit: SynchronousDataTypes = "momentum",
    ):
        super().__init__()
        self._value = value
        self._n_turns = int(max_turns)
        self._in_unit = in_unit

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        momentum = _to_momentum(
            data=self._value,
            mass=simulation.beams[0].particle_type.mass,
            charge=simulation.beams[0].particle_type.charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending " "field"
                else None
            ),
        )
        _momentum = momentum * np.ones((simulation.ring.n_cavities, self._n_turns))

        super().on_init_simulation(
            simulation=simulation,
            momentum_init=_momentum[0],  # because its constant
            momentum=_momentum,
        )

    @staticmethod
    def headless(
        section_lengths: NumpyArray,
        value: float,
        mass: float,
        charge: float,
        max_turns: int,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
    ):
        ret = ConstantEnergyCycle(
            value=value,
            max_turns=max_turns,
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
        simulation.ring.n_cavities = len(section_lengths)
        simulation.beams = (beam,)
        simulation.ring.section_lengths = section_lengths
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
        self._values_after_turn = values_after_turn[:]
        self._n_turns = self._values_after_turn.shape[0]
        self._in_unit = in_unit

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
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
            simulation=simulation, momentum_init=momentum_init, momentum=_momentum
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
    ):
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

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
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
        assert (
            n_cavities == momentum_after_cavity_per_turn.shape[0]
        ), f"{n_cavities=}, but {momentum_after_cavity_per_turn.shape=}"

        super().on_init_simulation(
            simulation=simulation,
            momentum_init=momentum_init,
            momentum=momentum_after_cavity_per_turn,
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
    ):
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
        max_turns: int,
        base_time: NumpyArray,
        base_values: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        interpolator=np.interp,
    ):
        super().__init__()
        self._t0 = t0
        self._n_turns = int(max_turns)
        self._interpolator = interpolator
        self._base_time = base_time[:]
        self._base_values = base_values[:]
        self._in_unit = in_unit

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        mass = simulation.beams[0].particle_type.mass
        charge = simulation.beams[0].particle_type.charge
        n_cavities = simulation.ring.n_cavities
        fast_execution_order = derive_execution_order(simulation)
        n_turns = self._n_turns

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
        at_time = derive_time(
            fast_execution_order=fast_execution_order,
            interpolate=self._interpolator,
            mass=mass,
            n_sections=n_cavities,
            n_turns=n_turns,
            program_momentum=base_momentum,
            program_time=self._base_time,
            t0=self._t0,
        )
        momentum = np.empty((n_cavities, n_turns))
        for cavity_i in range(n_cavities):
            momentum[cavity_i, :] = self._interpolator(
                at_time[cavity_i, :], self._base_time, base_momentum
            )
        _momentum_init = self._interpolator(self._t0, self._base_time, base_momentum)
        super().on_init_simulation(
            simulation=simulation,
            momentum_init=_momentum_init,
            momentum=momentum,
        )

    @staticmethod
    def headless(
        fast_execution_order: List | Tuple,
        mass: float,
        charge: float,
        t0: float,
        max_turns: int,
        base_time: NumpyArray,
        base_values: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
        interpolator=np.interp,
    ):
        from .._core.simulation.simulation import Simulation
        from .._core.beam.base import BeamBaseClass
        from .._core.beam.particle_types import ParticleType

        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        beam.particle_type = Mock(ParticleType)
        simulation.beams = (beam,)

        simulation.ring.section_lengths = [
            f for f in fast_execution_order if isinstance(f, float)
        ]
        simulation.ring.n_cavities = sum(
            [1 for i in fast_execution_order if isinstance(i, int)]
        )
        beam.particle_type.mass = mass
        beam.particle_type.charge = charge
        simulation.ring.bending_radius = bending_radius

        ret = EnergyCycleByTime(
            t0=t0,
            max_turns=max_turns,
            base_time=base_time,
            base_values=base_values,
            in_unit=in_unit,
            interpolator=interpolator,
        )

        with patch("blond3.cycles.energy_cycle.derive_execution_order") as mock_d:
            mock_d.return_value = fast_execution_order

            ret.on_init_simulation(simulation=simulation)
        ret.on_run_simulation(simulation=simulation, n_turns=max_turns, turn_i_init=0)

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
        The charge of the particles in units of [e]
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


def derive_execution_order(simulation):
    # generate execution order from reading attributes from ring
    # this is only ugly for performance.
    # elements are converted to int/float if cavity/drift
    fast_execution_order = []
    cavity_i = 0
    for elem in simulation.ring.elements.elements:
        if isinstance(elem, DriftBaseClass):
            drift_length = elem.share_of_circumference * simulation.ring.circumference
            fast_execution_order.append(float(drift_length))
        if isinstance(elem, CavityBaseClass):
            fast_execution_order.append(int(cavity_i))
            cavity_i += 1
    fast_execution_order = tuple(fast_execution_order)
    return fast_execution_order


def derive_time(
    fast_execution_order,
    mass,
    n_sections,
    n_turns: int,
    t0: float,
    program_momentum: NumpyArray,
    program_time: NumpyArray,
    interpolate=np.interp,
):
    """Derive time at each station by time-dependant momentum program

    Parameters
    ----------
    fast_execution_order
        List of ints and floats
        int means rf nr i [#]
        float means drift of length [m]
        Must have as many ints as n_sections
    mass
        The mass of the particles in [eV/c**2]
    n_sections
        Number of RF Stations
    n_turns
        Number of turns to derive
    t0
        Initial time to start deriving the times
        Initial momentum is derived from this too.
    program_momentum
        Array of momentum (vs time)
        The input synchronous data [eV/c]
    program_time
        Array of time (vs momentum)
        The input time [s]
    interpolate
        Interpolation method to get values
        in between given `program_time`

    Returns
    -------
    times
        Shape (n_sections, n_turns)

    """
    times = np.empty((n_sections, n_turns))
    t = t0
    momentum = float(
        interpolate(
            x=t,
            xp=program_time[:],
            fp=program_momentum[:],
            left=float(program_momentum[0]),
            right=float(program_momentum[-1]),
        )
    )
    beta = beta_by_momentum(momentum, mass)
    # mini simulation based on the knowledge of which
    # momentum is wanted at which moment in time.
    # From this, one can use x=v*t linear motion
    for turn_i in range(n_turns):
        for element in fast_execution_order:
            if isinstance(element, float):
                drift_length = element
                dt = drift_length / (beta * c0)
                t += dt
            if isinstance(element, int):
                momentum = float(
                    interpolate(
                        x=t,
                        xp=program_time[:],
                        fp=program_momentum[:],
                        left=float(program_momentum[0]),
                        right=float(program_momentum[-1]),
                    )
                )
                beta = beta_by_momentum(momentum, mass)
                cavity_i = element
                times[cavity_i, turn_i] = t
    return times
