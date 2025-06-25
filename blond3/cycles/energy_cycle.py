from __future__ import annotations

from functools import cached_property
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Optional,
    Literal,
)

import numpy as np
from numpy.typing import ArrayLike

from .base import ProgrammedCycle
from .._core.base import HasPropertyCache
from ..physics.cavities import (
    CavityBaseClass,
)

if TYPE_CHECKING:
    from typing import Optional as LateInit

    from numpy.typing import NDArray as NumpyArray
    from .._core.simulation.simulation import Simulation


class EnergyCycleBase(ProgrammedCycle, HasPropertyCache):
    def __init__(
        self,
    ):
        super().__init__()
        self._momentum: LateInit[NumpyArray] = None
        self._section_lengths: LateInit[NumpyArray] = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        assert self._momentum is not None, f"{self._momentum=}"
        assert len(self._momentum.shape) == 2, f"{self._momentum.shape=}"
        self._particle = simulation.beams[0].particle_type
        self._section_lengths = simulation.ring.elements.get_section_circumference_shares()  # TODO
        self.invalidate_cache()

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        self.invalidate_cache()

    @cached_property
    def n_turns(self):
        return self._momentum.shape[1] - 1

    @cached_property  # as readonly attributes
    def beta(self) -> NumpyArray:
        return np.sqrt(1 / (1 + (self._particle.mass / self._momentum[:, :]) ** 2))

    @cached_property  # as readonly attributes
    def gamma(self) -> NumpyArray:
        return np.sqrt(1 + (self._momentum[:, :] / self._particle.mass) ** 2)

    @cached_property  # as readonly attributes
    def energy(self) -> NumpyArray:
        return np.sqrt(self._momentum[:, :] ** 2 + self._particle.mass**2)

    @cached_property  # as readonly attributes
    def kin_energy(self) -> NumpyArray:
        return (
            np.sqrt(self._momentum[:, :] ** 2 + self._particle.mass**2)
            - self._particle.mass
        )

    @cached_property  # as readonly attributes
    def delta_E(self) -> NumpyArray:
        shape2d = self.energy.shape
        flat_diff = np.diff(self.energy.flatten())  # loses one entry
        diff1d = np.concatenate(([0], flat_diff))  # add back one entry
        diff2d = diff1d.reshape(shape2d)
        return diff2d

    @cached_property  # as readonly attributes
    def t_section(self) -> NumpyArray:
        return (
            self._section_lengths[:] * 1 / (self.beta[:, :] * c0)
        )  # todo check matching
        # shapes?!

    @cached_property  # as readonly attributes
    def t_rev(self) -> NumpyArray:
        return np.dot(
            self._section_lengths, 1 / (self.beta * c0)
        )  # todo check matching shapes?!

    @cached_property  # as readonly attributes
    def cycle_time(self) -> NumpyArray:
        shape2d = self.t_section.shape
        c_time = np.cumsum(self.t_section.flatten()).reshape(shape2d)
        return c_time

    @cached_property  # as readonly attributes
    def f_rev(self) -> NumpyArray:
        return 1 / self.t_rev  # TODO

    @cached_property  # as readonly attributes
    def omega_rev(self) -> NumpyArray:
        return 2 * np.pi * self.f_rev  # todo

    props = (
        "n_turns",
        "beta",
        "gamma",
        "energy",
        "kin_energy",
        "delta_E",
        "t_rev",
        "cycle_time",
        "f_rev",
        "omega_rev",
    )

    def invalidate_cache(self):
        super().invalidate_cache(EnergyCycleBase.props)


class ConstantEnergyCycle(EnergyCycleBase):
    def __init__(
        self,
        value: float,
        max_turns: int,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float]=None,
    ):
        super().__init__()
        self._value = value
        self._n_turns = int(max_turns)
        self._in_unit = in_unit
        if self._in_unit == "bending field":
            assert bending_radius is not None
            self._bending_radius = bending_radius
        else:
            self._bending_radius = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        momentum = _to_momentum(
            data=self._value,
            mass=simulation.beams[0].particle_type.mass,
            charge=simulation.beams[0].particle_type.charge,
            convert_from=self._in_unit,
            bending_radius=(
                self._bending_radius if self._in_unit == "bending field" else None
            ),
        )
        n_turns = self._n_turns

        n_cavities = simulation.ring.elements.count(CavityBaseClass)

        shape = (n_cavities, n_turns + 1)  # TODO
        self._momentum = momentum * np.ones(shape)
        super().on_init_simulation(simulation=simulation)


class EnergyCyclePerTurn(EnergyCycleBase):
    def __init__(
        self,
        values_per_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float]=None,
    ):
        super().__init__()
        self._values_per_turn = values_per_turn[:]
        self._n_turns = self._values_per_turn.shape[0] - 1
        self._in_unit = in_unit
        if self._in_unit == "bending field":
            assert bending_radius is not None
            self._bending_radius = bending_radius
        else:
            self._bending_radius = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        momentum_per_turn = _to_momentum(
            data=self._values_per_turn,
            mass=simulation.beams[0].particle_type.mass,
            charge=simulation.beams[0].particle_type.charge,
            convert_from=self._in_unit,
            bending_radius=(
                self._bending_radius if self._in_unit == "bending field" else None
            ),
        )
        n_cavities = simulation.ring.elements.count(CavityBaseClass)
        n_turns = self._n_turns

        shape = (n_cavities, n_turns + 1)
        result = np.empty(shape)
        # assume that each cavity gives an
        # even part of the kick
        stair_like = np.arange(1, n_cavities + 1, dtype=float)
        stair_like /= np.sum(stair_like)
        assert np.sum(stair_like) == 1
        result[:, 0] = float(momentum_per_turn[0])
        for turn_i in range(1, n_turns + 1):
            base = momentum_per_turn[turn_i - 1]
            step = momentum_per_turn[turn_i] - momentum_per_turn[turn_i - 1]
            result[:, turn_i] = base + stair_like * step
        self._momentum = result
        super().on_init_simulation(simulation=simulation)



class EnergyCyclePerTurnAllCavities(EnergyCycleBase):
    def __init__(
        self,
        values_per_cavity_per_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float]=None,
    ):
        super().__init__()
        self._values_per_cavity_per_turn = values_per_cavity_per_turn[:, :]
        self._n_turns = self._values_per_cavity_per_turn.shape[1] - 1
        self._in_unit = in_unit
        if self._in_unit == "bending field":
            assert bending_radius is not None
            self._bending_radius = bending_radius
        else:
            self._bending_radius = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        momentum_per_cavity_per_turn = _to_momentum(
            data=self._values_per_cavity_per_turn[:, :],
            mass=simulation.beams[0].particle_type.mass,
            charge=simulation.beams[0].particle_type.charge,
            convert_from=self._in_unit,
            bending_radius=(
                self._bending_radius if self._in_unit == "bending field" else None
            ),
        )
        n_cavities = simulation.ring.elements.count(CavityBaseClass)

        assert len(n_cavities) == momentum_per_cavity_per_turn.shape[0], (
            f"{len(n_cavities)=}, but {momentum_per_cavity_per_turn.shape=}"
        )
        self._momentum = momentum_per_cavity_per_turn
        super().on_init_simulation(simulation=simulation)


class EnergyCycleByTime(EnergyCycleBase):
    def __init__(
        self,
        t0: float,
        max_turns: int,
        base_time: NumpyArray,
        base_values: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float]=None,
        interpolator=np.interp,
    ):
        super().__init__()
        self._t0 = t0
        self._n_turns = int(max_turns)
        self._interpolator = interpolator
        self._base_time = base_time[:]
        self._base_values = base_values[:]
        self._in_unit = in_unit
        if self._in_unit == "bending field":
            assert bending_radius is not None
            self._bending_radius = bending_radius
        else:
            self._bending_radius = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        base_momentum = _to_momentum(
            data=self._base_values,
            mass=simulation.beams[0].particle_type.mass,
            charge=simulation.beams[0].particle_type.charge,
            convert_from=self._in_unit,
            bending_radius=(
                self._bending_radius if self._in_unit == "bending field" else None
            ),
        )
        n_turns = self._n_turns
        n_cavities = simulation.ring.elements.count(CavityBaseClass)

        at_time = derive_time(
            n_turns=n_turns,
            section_lengths=simulation.ring.elements.get_section_circumference_shares(),  # TODO,
            t0=self._t0,
            program_time=self._base_time,
            program_momentum=base_momentum,
            mass=simulation.beams[0].particle_type.mass,
            interpolate=self._interpolator,
        )
        momentum = np.empty((n_cavities, n_turns + 1))
        momentum[:, 0] = np.interp(at_time[0, 0], self._base_time, base_momentum)
        for turn_i in range(1, n_turns + 1):
            momentum[:, turn_i] = np.interp(
                at_time[:, turn_i], self._base_time, base_momentum
            )
        self._momentum = momentum
        super().on_init_simulation(simulation=simulation)



from scipy.constants import speed_of_light as c0

T = TypeVar("T")

SynchronousDataTypes = Literal[
    "momentum", "total energy", "kinetic energy", "bending field"
]


def _to_momentum(
    data: int | float | NumpyArray,
    mass: float,
    charge: float,
    convert_from: SynchronousDataTypes = "momentum",
    bending_radius: Optional[float] = None,
) -> NumpyArray:
    """Function to convert synchronous data (i.e. energy program of the
    synchrotron) into momentum.

    Parameters
    ----------
    data : float array
        The synchronous data to be converted to momentum
    mass : float or Particle.mass
        The mass of the particles in [eV/c**2]
    charge : int or Particle.charge
        The charge of the particles in units of [e]
    convert_from : str
        Type of input for the synchronous data ; can be 'momentum',
        'total energy', 'kinetic energy' or 'bending field' (last case
        requires bending_radius to be defined)
    bending_radius : float
        Bending radius in [m] in case convert_from is
        'bending field'

    Returns
    -------
    momentum : float array
        The input synchronous_data converted into momentum [eV/c]

    """

    if convert_from == "momentum":
        momentum = data
    elif convert_from == "total energy":
        momentum = np.sqrt(data**2 - mass**2)
    elif convert_from == "kinetic energy":
        momentum = np.sqrt((data + mass) ** 2 - mass**2)
    elif convert_from == "bending field":
        if bending_radius is None:
            # InputDataError
            raise RuntimeError(
                "ERROR in Ring: bending_radius is not "
                "defined and is required to compute "
                "momentum"
            )
        momentum = data * bending_radius * charge * c0
    else:
        # InputDataError
        raise ValueError(f"{convert_from=}")

    return momentum


def beta_by_momentum(momentum: T, mass: float) -> T:
    return np.sqrt(1 / (1 + (mass / momentum) ** 2))


def derive_time(
    n_turns: int,
    section_lengths: ArrayLike[float],
    t0: float,
    program_time: NumpyArray,
    program_momentum: NumpyArray,
    mass: float,
    interpolate=np.interp,
) -> NumpyArray:
    """Derive time at different sections for n_turns, given momentum(time)"""
    times = np.empty((len(section_lengths), n_turns + 1))

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

    all_sections = slice(0, len(section_lengths))
    turn_i = 0
    times[all_sections, turn_i] = t
    del all_sections

    # mini simulation based on the knowledge of which momentum is wanted
    # at which moment in time.
    # From this, one can use x=v*t linear motion
    for turn_i in range(1, n_turns + 1):
        for section_i, drift_length in enumerate(section_lengths):
            dt = drift_length / beta
            t += dt
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
            times[section_i, turn_i] = t

    return times
