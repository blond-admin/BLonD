from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0

from .base import ProgrammedCycle
from .._core.backends.backend import backend
from .._core.base import HasPropertyCache
from .._core.beam.base import BeamBaseClass
from .._core.beam.particle_types import ParticleType, proton
from ..acc_math.analytic.simple_math import calc_total_energy

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional as LateInit, TypeVar, Optional, Literal, Union, Dict

    from numpy.typing import NDArray as NumpyArray

    from .._core.simulation.simulation import Simulation

    FloatOrArray = Union[float, NumpyArray]

    T = TypeVar("T")

    SynchronousDataTypes = Literal[
        "momentum", "total energy", "kinetic energy", "bending field"
    ]


class MagneticCycleBase(ProgrammedCycle, HasPropertyCache):
    """Programmed magnetic cycle of the synchrotron

    Parameters
    ----------
    reference_particle
        Type of particles, e.g. protons
    """

    def __init__(
        self,
        reference_particle: ParticleType,
    ):
        super().__init__()
        assert isinstance(
            reference_particle, ParticleType
        ), f"{type(reference_particle)}"
        self._reference_particle: ParticleType = reference_particle

        self._magnetic_rigidity_before_turn_0: LateInit[float] = None
        self._n_turns_max: None | int = None

    def on_init_simulation(
        self,
        simulation: Simulation,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)
        self._magnetic_rigidity_before_turn_0 = kwargs["magnetic_rigidity_init"]
        self._n_turns_max = kwargs["n_turns_max"]

        self.invalidate_cache()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )
        self.invalidate_cache()

    @property
    def n_turns(self) -> None | int:
        """Number of turns that are defined by this cycle"""
        return self._n_turns_max

    @abstractmethod
    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
        particle_type: ParticleType,
    ):
        """
        Calculate the total energy [eV] that is foreseen by the magnetic cycle

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
        particle_type
            Type of particles, e.g. protons

        Returns
        -------

        """
        pass

    def get_total_energy_init(
        self,
        turn_i_init: int,
        t_init: float,
        particle_type: ParticleType,
    ) -> backend.float:
        index = turn_i_init - 1
        if index < 0:
            total_energy_init = calc_total_energy(
                mass=particle_type.mass,
                momentum=magnetic_rigidity_to_momentum(
                    magnetic_rigidity=self._magnetic_rigidity_before_turn_0,
                    charge=particle_type.charge,
                ),
            )
            new_reference_total_energy = total_energy_init
        else:
            new_reference_total_energy = self.get_target_total_energy(
                turn_i=index,
                section_i=-1,
                reference_time=t_init,
                particle_type=particle_type,
            )
        return new_reference_total_energy

    @staticmethod
    @abstractmethod
    def headless(*args, **kwargs):
        """Initialize object without simulation context"""
        pass

    cached_props = ()

    def invalidate_cache(self):
        """Delete the stored values of functions with @cached_property"""
        super()._invalidate_cache(MagneticCycleBase.cached_props)


class ConstantMagneticCycle(MagneticCycleBase):
    def __init__(
        self,
        reference_particle: ParticleType,
        value: float,
        in_unit: SynchronousDataTypes = "momentum",
    ):
        """
        Magnetic cycle for a non-changing magnetic field

        Parameters
        ----------
        reference_particle
            Type of particles, e.g. protons
        value
            Constant value of unit `in_unit`
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        """
        super().__init__(
            reference_particle=reference_particle,
        )
        self._value = value
        self._in_unit = in_unit

        self._magnetic_rigidity: LateInit[float] = None
        self._total_energy_cache: LateInit[Dict[int, float]] = {}

    def on_init_simulation(
        self,
        simulation: Simulation,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        self._magnetic_rigidity = _to_magnetic_rigidity(
            data=self._value,
            mass=self._reference_particle.mass,
            charge=self._reference_particle.charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending " "field"
                else None
            ),
        )

        super().on_init_simulation(
            simulation=simulation,
            n_turns_max=None,
            magnetic_rigidity_init=self._magnetic_rigidity,
        )

    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
        particle_type: ParticleType,
    ):
        """
        Calculate the total energy [eV] that is foreseen by the magnetic cycle

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
        particle_type
            Type of particles, e.g. protons

        Returns
        -------

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
        bending_radius: Optional[float] = None,
    ) -> ConstantMagneticCycle:
        """
        Initialize object without simulation context

        Parameters
        ----------
        value
            Constant value of unit `in_unit`
        particle_type
            Type of particles, e.g. protons
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        bending_radius
            Bending radius in [m]

        Returns
        -------
            constant_magnetic_cycle

        """
        ret = ConstantMagneticCycle(
            value=value,
            in_unit=in_unit,
            reference_particle=proton,
        )
        from .._core.beam.base import BeamBaseClass
        from .._core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.ring.bending_radius = bending_radius

        ret.on_init_simulation(simulation=simulation)
        return ret


class MagneticCyclePerTurn(MagneticCycleBase):
    def __init__(
        self,
        reference_particle: ParticleType,
        value_init: float,
        values_after_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
    ):
        """
        Magnetic cycle per turn. Assumes each cavity has the same increment
        of beam energy.

        Parameters
        ----------
        reference_particle
            Type of particles, e.g. protons
        value_init
            Initial value at start of simulation in of unit `in_unit`
        values_after_turn
            Value after turn in synchrotron in of unit `in_unit`
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        """
        super().__init__(
            reference_particle=reference_particle,
        )
        self._value_init = value_init

        assert (
            len(values_after_turn.shape) == 1
        ), f"Expected 1D array, but got {values_after_turn.shape}"

        self._values_after_turn = values_after_turn[:]
        self._in_unit = in_unit

        self._magnetic_rigidity: LateInit[NumpyArray] = None
        self._momentum_cached: Dict[int, NumpyArray] = {}

    def on_init_simulation(
        self,
        simulation: Simulation,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        """Lateinit method when `simulation.run` is called

        simulation
            Simulation context manager
        """
        n_cavities = simulation.ring.n_cavities
        n_turns_max = self._values_after_turn.shape[0]

        magnetic_rigidity_init = _to_magnetic_rigidity(
            data=self._value_init,
            mass=self._reference_particle.mass,
            charge=self._reference_particle.charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        magnetic_rigidity_per_turn = _to_magnetic_rigidity(
            data=self._values_after_turn,
            mass=self._reference_particle.mass,
            charge=self._reference_particle.charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        assert n_cavities > 0
        shape = (n_cavities, n_turns_max)
        _magnetic_rigidity = np.empty(shape)
        # assume that each cavity gives an
        # even part of the kick
        stair_like = np.linspace(1 / n_cavities, 1, n_cavities, endpoint=True)
        base = np.concatenate(([magnetic_rigidity_init], magnetic_rigidity_per_turn))
        step = np.diff(base)
        for cav_i in range(n_cavities):
            _magnetic_rigidity[cav_i, :] = base[:-1] + stair_like[cav_i] * step

        super().on_init_simulation(
            simulation=simulation,
            n_turns_max=n_turns_max,
            magnetic_rigidity_init=magnetic_rigidity_init,
        )
        self._magnetic_rigidity = _magnetic_rigidity

    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
        particle_type: ParticleType,
    ):
        """
        Calculate the total energy [eV] that is foreseen by the magnetic cycle

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
        particle_type
            Type of particles, e.g. protons

        Returns
        -------

        """
        key = hash(particle_type)
        if key not in self._momentum_cached:
            self._momentum_cached[key] = magnetic_rigidity_to_momentum(
                magnetic_rigidity=self._magnetic_rigidity[:, :],
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
        values_after_turn: NumpyArray,
        n_cavities: int,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
    ) -> MagneticCyclePerTurn:
        """
        Initialize object without simulation context

        Parameters
        ----------
        reference_particle
            Type of particles, e.g. protons
        value_init
            Initial value at start of simulation in of unit `in_unit`
        values_after_turn
            Value after turn in Synchrotron in of unit `in_unit`
        n_cavities
            Number of cavities
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        bending_radius
            Bending radius in [m]

        Returns
        -------
        magnetic_cycle_per_turn

        """
        ret = MagneticCyclePerTurn(
            value_init=value_init,
            values_after_turn=values_after_turn,
            in_unit=in_unit,
            reference_particle=reference_particle,
        )

        from .._core.simulation.simulation import Simulation
        from .._core.beam.base import BeamBaseClass
        from .._core.beam.particle_types import ParticleType

        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        beam.particle_type = Mock(ParticleType)

        simulation.ring.bending_radius = bending_radius
        beam.particle_type = reference_particle
        simulation.ring.n_cavities = n_cavities
        ret.on_init_simulation(simulation=simulation)
        ret.on_run_simulation(
            simulation=simulation,
            n_turns=len(values_after_turn),
            turn_i_init=0,
            beam=beam,
        )

        return ret


class MagneticCyclePerTurnAllCavities(MagneticCycleBase):
    def __init__(
        self,
        reference_particle: ParticleType,
        value_init: float,
        values_after_cavity_per_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
    ):
        """
        Magnetic program per turn, defined for each cavity

        Parameters
        ----------
        reference_particle
            Type of particles, e.g. protons
        value_init
            Initial value at start of simulation in of unit `in_unit`
        values_after_cavity_per_turn
            Value after each cavity and each turn in Synchrotron
             in of unit `in_unit`
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        """
        super().__init__(
            reference_particle=reference_particle,
        )
        self._value_init = value_init
        self._values_after_cavity_per_turn = values_after_cavity_per_turn[:, :]
        self._n_turns_max = self._values_after_cavity_per_turn.shape[1]
        self._in_unit = in_unit

        self._magnetic_rigidity_after_cavity_per_turn: LateInit[NumpyArray] = None
        self._momentum_cached: Dict[int, NumpyArray] = {}

    def on_init_simulation(
        self,
        simulation: Simulation,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        magnetic_rigidity_init = _to_magnetic_rigidity(
            data=self._value_init,
            mass=self._reference_particle.mass,
            charge=self._reference_particle.charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        magnetic_rigidity_after_cavity_per_turn = _to_magnetic_rigidity(
            data=self._values_after_cavity_per_turn[:, :],
            mass=self._reference_particle.mass,
            charge=self._reference_particle.charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        n_cavities = simulation.ring.n_cavities
        n_turns_max = magnetic_rigidity_after_cavity_per_turn.shape[1]
        assert (
            n_cavities == magnetic_rigidity_after_cavity_per_turn.shape[0]
        ), f"{n_cavities=}, but {magnetic_rigidity_after_cavity_per_turn.shape=}"

        super().on_init_simulation(
            simulation=simulation,
            n_turns_max=n_turns_max,
            magnetic_rigidity_init=magnetic_rigidity_init,
        )
        self._magnetic_rigidity_after_cavity_per_turn = (
            magnetic_rigidity_after_cavity_per_turn
        )

    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
        particle_type: ParticleType,
    ):
        """
        Calculate the total energy [eV] that is foreseen by the magnetic cycle

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
        particle_type
            Type of particles, e.g. protons

        Returns
        -------

        """

        key = hash(particle_type)
        if key not in self._momentum_cached:
            self._momentum_cached[key] = magnetic_rigidity_to_momentum(
                magnetic_rigidity=self._magnetic_rigidity_after_cavity_per_turn[:, :],
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
        values_after_cavity_per_turn: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
    ) -> MagneticCyclePerTurnAllCavities:
        """
        Initialize object without simulation context

        Parameters
        ----------
        reference_particle
            Type of particles, e.g. protons
        value_init
            Initial value at start of simulation in of unit `in_unit`
        values_after_cavity_per_turn
            Value after each cavity and each turn in Synchrotron
             in of unit `in_unit`
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        bending_radius
            Bending radius in [m]

        Returns
        -------

        """
        ret = MagneticCyclePerTurnAllCavities(
            value_init=value_init,
            values_after_cavity_per_turn=values_after_cavity_per_turn,
            in_unit=in_unit,
            reference_particle=reference_particle,
        )
        from .._core.simulation.simulation import Simulation
        from .._core.beam.base import BeamBaseClass
        from .._core.beam.particle_types import ParticleType

        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        beam.particle_type = Mock(ParticleType)

        simulation.ring.bending_radius = bending_radius
        beam.particle_type = reference_particle
        simulation.ring.n_cavities = values_after_cavity_per_turn.shape[0]

        ret.on_init_simulation(simulation=simulation)
        ret.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=values_after_cavity_per_turn.shape[1],
            turn_i_init=0,
        )
        return ret


class MagneticCycleByTime(MagneticCycleBase):
    def __init__(
        self,
        reference_particle: ParticleType,
        base_time: NumpyArray,
        base_values: NumpyArray,
        in_unit: SynchronousDataTypes = "momentum",
        interpolator=np.interp,
    ):
        """
        Magnetic cycle defined as B vs. Time, interpolated just in
        time

        Parameters
        ----------
        reference_particle
            Type of particles, e.g. protons
        base_time
            Values of time [s]
        base_values
            Values at time in synchrotron in of unit `in_unit`
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        interpolator
            Interpolation routine to get time in between the base values
            Default: `numpy.interp`
        """
        super().__init__(
            reference_particle=reference_particle,
        )
        self._interpolator = interpolator
        self._base_time = base_time[:]
        self._base_values = base_values[:]
        self._in_unit = in_unit

        self._base_magnetic_rigidity: LateInit[NumpyArray] = None

    def on_init_simulation(
        self,
        simulation: Simulation,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        base_magnetic_rigidity = _to_magnetic_rigidity(
            data=self._base_values,
            mass=self._reference_particle.mass,
            charge=self._reference_particle.charge,
            convert_from=self._in_unit,
            bending_radius=(
                simulation.ring.bending_radius
                if self._in_unit == "bending field"
                else None
            ),
        )
        self._base_magnetic_rigidity = base_magnetic_rigidity

        magnetic_rigidity_init = base_magnetic_rigidity[0]

        super().on_init_simulation(
            simulation=simulation,
            n_turns_max=None,
            magnetic_rigidity_init=magnetic_rigidity_init,
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
        Calculate the total energy [eV] that is foreseen by the magnetic cycle

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
        particle_type
            Type of particles, e.g. protons

        Returns
        -------

        """
        magnetic_rigidity = self._interpolator(
            reference_time,
            self._base_time,
            self._base_magnetic_rigidity,
        )
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
        bending_radius: Optional[float] = None,
        interpolator=np.interp,
    ) -> MagneticCycleByTime:
        """
        Initialize object without simulation context


        Parameters
        ----------
        reference_particle
            Type of particles, e.g. protons
            Example: For an electron `charge=1`
        base_time
            Values of time [s]
        base_values
            Values at time in synchrotron in of unit `in_unit`
        in_unit
            - 'momentum' [eV/c], (no conversion is done)
            - 'total energy' [eV],
            - 'kinetic energy' [eV], or
            - 'bending field' [T]
        bending_radius
            Bending radius in [m]
        interpolator
            Interpolation routine to get time in between the base values
            Default: `numpy.interp`

        Returns
        -------
        Magnetic_cycle_by_time
        """
        from .._core.simulation.simulation import Simulation
        from .._core.beam.base import BeamBaseClass
        from .._core.beam.particle_types import ParticleType

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
            turn_i_init=0,
            beam=beam,
        )

        return ret


def _to_magnetic_rigidity(
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
        Example: For an electron `charge=1`
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


def magnetic_rigidity_to_momentum(magnetic_rigidity, charge: float):
    return magnetic_rigidity * np.abs(charge) * c0
