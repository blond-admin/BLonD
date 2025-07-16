from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Iterable,
    Tuple,
    Optional,
    Callable,
)

import numpy as np
from numpy.typing import ArrayLike

from .._core.backends.backend import backend
from .._core.base import HasPropertyCache
from .._core.ring.helpers import requires
from ..physics.cavities import (
    CavityBaseClass,
)
from ..physics.drifts import DriftBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional as LateInit

    from numpy.typing import NDArray as NumpyArray
    from .._core.simulation.simulation import Simulation

    from .._core.beam.base import BeamBaseClass


class _ReshapeHandler:
    @staticmethod
    def reshape_any(
        input_data: float
        | int
        | ArrayLike
        | tuple[NumpyArray, NumpyArray]
        | Tuple[Tuple[NumpyArray, NumpyArray], ...],
        n_turns: int,
        n_sections: int,
        interp_time: Optional[NumpyArray] = None,
    ) -> NumpyArray:
        # If single float, expands the value to match the input number of turns
        # and sections
        if isinstance(input_data, float) or isinstance(input_data, int):
            output_data = _ReshapeHandler.reshape_float(
                input_data=input_data,
                n_sections=n_sections,
                n_turns=n_turns,
            )
        # If array/list, compares with the input number of turns and
        # if synchronous_data is a single value converts it into a (n_turns+1)
        # array
        elif isinstance(input_data, np.ndarray) or isinstance(input_data, list):
            output_data = _ReshapeHandler.reshape_iterable(
                input_data=input_data,
                n_sections=n_sections,
                n_turns=n_turns,
            )
        # If tuple, separate time and synchronous data and check data
        elif isinstance(input_data, tuple):
            assert interp_time is not None
            output_data = _ReshapeHandler.reshape_tuple(
                times=input_data[0],
                values=input_data[1],
                interp_time=interp_time,
            )

        else:
            raise TypeError(
                f"{type(input_data)=} is not recognised, must be float, int, "
                f"list, np.array, or tuple."
            )

        return output_data

    @staticmethod
    def reshape_float(
        input_data: float,
        n_sections: Optional[float],
        n_turns: Optional[float],
    ):
        input_data = float(input_data)
        output_data = input_data * np.ones((n_sections, n_turns + 1))
        return output_data

    @staticmethod
    def reshape_iterable(
        input_data: Iterable,
        n_sections: int,
        n_turns: int,
    ):
        input_data = np.array(input_data, ndmin=2, dtype=float)
        output_data = np.zeros((n_sections, n_turns + 1), dtype=float)
        # If the number of points is exactly the same as n_rf, this means
        # that the rf program for each harmonic is constant, reshaping
        # the array so that the size is [n_sections,1] for successful
        # reshaping
        if input_data.size == n_sections:
            input_data = input_data.reshape((n_sections, 1))
        if len(input_data) != n_sections:
            # InputDataError
            raise RuntimeError(
                f"`input_data.shape[0]` must be {n_sections=}, but is "
                f"{input_data.shape=})"
            )
        for index_section in range(len(input_data)):
            if len(input_data[index_section]) == 1:
                output_data[index_section] = input_data[index_section] * np.ones(
                    n_turns + 1
                )

            elif len(input_data[index_section]) == (n_turns + 1):
                output_data[index_section] = np.array(input_data[index_section])

            else:
                # InputDataError
                raise RuntimeError(
                    "ERROR in Ring: The 'input_data'"
                    + f"should have {(n_turns+1)=} entries, but shape is {input_data.shape=}"
                )
        return output_data

    @staticmethod
    def reshape_tuple(
        times: NumpyArray,
        values: NumpyArray,
        interp_time: NumpyArray,
    ):
        assert len(interp_time.shape) == 2
        result = np.empty(interp_time.shape)
        for i in range(interp_time.shape[0]):
            result[i, :] = np.interp(interp_time[i, :], times, values)

        return result


class _EnergyCycleAdapter:
    def __init__(
        self,
        synchronous_data: NumpyArray,
        synchronous_data_type: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
    ):
        self._synchronous_data = synchronous_data
        self._synchronous_data_type = synchronous_data_type

        if self._synchronous_data_type == "bending field":
            assert bending_radius is not None
            self._bending_radius = float(bending_radius)
        else:
            self._bending_radius = None

    def get_momentum_program(self, mass: float, charge: float):
        return _to_momentum(
            synchronous_data=self._synchronous_data[1],
            mass=mass,
            charge=charge,
            convert_from=self._synchronous_data_type,
            bending_radius=self._bending_radius
            if self._synchronous_data_type == "bending field"
            else None,
        )

    def derive_interp_time(
        self,
        n_turns: int,
        section_lengths: ArrayLike[float],
        t0: float,
        mass: float,
        charge: float,
        interpolate=np.interp,
    ) -> NumpyArray | None:
        if isinstance(self._synchronous_data, tuple):
            assert len(self._synchronous_data) == 2
            assert isinstance(self._synchronous_data[0], np.ndarray)
            assert isinstance(self._synchronous_data[1], np.ndarray)

            times = derive_time(
                n_turns=n_turns,
                section_lengths=section_lengths,
                t0=t0,
                program_time=self._synchronous_data[0],
                program_momentum=self.get_momentum_program(mass, charge),
                mass=mass,
                interpolate=interpolate,
            )
        else:
            times = None
        return times


class EnergyCycleBase(ProgrammedCycle, HasPropertyCache):
    def __init__(
        self,
        synchronous_data: float | NumpyArray | Tuple[NumpyArray, NumpyArray],
        synchronous_data_type: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
    ):
        super().__init__()
        self._adapter = _EnergyCycleAdapter(
            synchronous_data=synchronous_data,
            synchronous_data_type=synchronous_data_type,
            bending_radius=bending_radius,
        )

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
        self._invalidate_cache()

    @cached_property
    @abstractmethod  # as readonly attributes
    def n_turns(self):
        pass

    @cached_property  # as readonly attributes
    @abstractmethod  # as readonly attributes
    def beta(self) -> NumpyArray:
        pass

    @cached_property  # as readonly attributes
    @abstractmethod  # as readonly attributes
    def gamma(self) -> NumpyArray:
        pass

    @cached_property  # as readonly attributes
    @abstractmethod  # as readonly attributes
    def energy(self) -> NumpyArray:
        pass

    @cached_property  # as readonly attributes
    @abstractmethod  # as readonly attributes
    def kin_energy(self) -> NumpyArray:
        pass

    @cached_property  # as readonly attributes
    @abstractmethod  # as readonly attributes
    def delta_E(self) -> NumpyArray:
        pass

    @cached_property  # as readonly attributes
    @abstractmethod  # as readonly attributes
    def t_rev(self) -> NumpyArray:
        pass

    @cached_property  # as readonly attributes
    @abstractmethod  # as readonly attributes
    def cycle_time(self) -> NumpyArray:
        pass

    @cached_property  # as readonly attributes
    @abstractmethod  # as readonly attributes
    def f_rev(self) -> NumpyArray:
        pass

    @cached_property  # as readonly attributes
    @abstractmethod  # as readonly attributes
    def omega_rev(self) -> NumpyArray:
        pass

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

    def _invalidate_cache(self):
        super()._invalidate_cache(EnergyCycle.props)


class EnergyCycle(EnergyCycleBase):
    def __init__(
        self,
        synchronous_data: float | NumpyArray,
        synchronous_data_type: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
    ):
        super().__init__(
            synchronous_data=synchronous_data,
            synchronous_data_type=synchronous_data_type,
            bending_radius=bending_radius,
        )
        from blond.input_parameters.ring import Ring as Blond2Ring

        self._ring: LateInit[Blond2Ring] = None

    @staticmethod
    def from_linspace(start, stop, turns, endpoint: bool = True):
        return EnergyCycle(
            synchronous_data=np.linspace(
                start, stop, turns + 1, endpoint=endpoint, dtype=backend.float
            )
        )

    @requires(["Ring"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        from blond.input_parameters.ring import Ring as Blond2Ring

        drifts = simulation.ring.elements.get_elements(DriftBaseClass)
        cavities = simulation.ring.elements.get_elements(CavityBaseClass)
        assert len(drifts) == len(cavities)
        self._ring = Blond2Ring(
            ring_length=[
                (e.share_of_circumference * simulation.ring.circumference)
                for e in drifts
            ],
            synchronous_data=self._adapter._synchronous_data,
            synchronous_data_type=self._adapter._synchronous_data_type,
            n_sections=len(cavities),
            alpha_0=np.nan,
            particle=simulation.beams[0].particle_type,
            n_turns=self.n_turns,
            bending_radius=simulation.ring.bending_radius,
        )
        self._invalidate_cache()
        _data = _ReshapeHandler.reshape_any(
            input_data=self._adapter._synchronous_data,
            n_turns=len(self._adapter._synchronous_data),
            n_sections=len(cavities),
            interp_time=None,
        )
        momentum = _to_momentum(
            synchronous_data=_data,
            mass=simulation.beams[0].particle_type.mass,
            charge=simulation.beams[0].particle_type.charge,
            convert_from=self._adapter._synchronous_data_type,
            bending_radius=self._adapter._bending_radius,
        )
        self._invalidate_cache()

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
        self._invalidate_cache()

    @cached_property
    def n_turns(self):
        return len(self._synchronous_data)  # TODO wrong

    @cached_property  # as readonly attributes
    def beta(self) -> NumpyArray:
        return self._ring.beta.astype(backend.float)

    @cached_property  # as readonly attributes
    def gamma(self) -> NumpyArray:
        return self._ring.gamma.astype(backend.float)

    @cached_property  # as readonly attributes
    def energy(self) -> NumpyArray:
        return self._ring.energy.astype(backend.float)

    @cached_property  # as readonly attributes
    def kin_energy(self) -> NumpyArray:
        return self._ring.kin_energy.astype(backend.float)

    @cached_property  # as readonly attributes
    def delta_E(self) -> NumpyArray:
        return self._ring.delta_E.astype(backend.float)

    @cached_property  # as readonly attributes
    def t_rev(self) -> NumpyArray:
        return self._ring.t_rev.astype(backend.float)

    @cached_property  # as readonly attributes
    def cycle_time(self) -> NumpyArray:
        return self._ring.cycle_time.astype(backend.float)

    @cached_property  # as readonly attributes
    def f_rev(self) -> NumpyArray:
        return self._ring.f_rev.astype(backend.float)

    @cached_property  # as readonly attributes
    def omega_rev(self) -> NumpyArray:
        return self._ring.omega_rev.astype(backend.float)


class InterpolatedEnergyCycle(EnergyCycle):
    def __init__(
        self,
        t_start: float,
        n_turns: int,
        base_time: NumpyArray,
        base_synchronous_data: NumpyArray,
        synchronous_data_type: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
        interpolate: Callable = np.interp,
    ):
        super().__init__(
            synchronous_data=(base_time, base_synchronous_data),
            synchronous_data_type=synchronous_data_type,
            bending_radius=bending_radius,
        )
        self._n_turns = n_turns
        self._t0 = t_start
        self._interpolate = interpolate

    @staticmethod
    def from_linspace(start, stop, turns, endpoint: bool = True):
        raise NotImplementedError("Use EnergyCycle instead!")

    @requires(["Ring"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        from blond.input_parameters.ring import Ring as Blond2Ring

        drifts = simulation.ring.elements.get_elements(DriftBaseClass)
        cavities = simulation.ring.elements.get_elements(CavityBaseClass)
        assert len(drifts) == len(cavities)
        self._ring = Blond2Ring(
            ring_length=[
                (e.share_of_circumference * simulation.ring.circumference)
                for e in drifts
            ],
            synchronous_data=self._adapter._synchronous_data,
            synchronous_data_type=self._adapter._synchronous_data_type,
            n_sections=len(cavities),
            alpha_0=np.nan,
            particle=simulation.beams[0].particle_type,
            n_turns=self.n_turns,
            bending_radius=simulation.ring.bending_radius,
        )
        interp_time = self._adapter.derive_interp_time(
            n_turns=self.n_turns,  # TODO
            section_lengths=[
                (e.share_of_circumference * simulation.ring.circumference)
                for e in drifts
            ],
            t0=self._t0,  # TODO
            mass=simulation.beams[0].particle_type.mass,
            charge=simulation.beams[0].particle_type.charge,
            interpolate=self._interpolate,
        )
        _synchronous_data = _ReshapeHandler.reshape_tuple(
            times=self._adapter._synchronous_data[0],
            values=self._adapter.get_momentum_program(
                mass=simulation.beams[0].particle_type.mass,
                charge=simulation.beams[0].particle_type.charge,
            ),
            interp_time=interp_time,
        )
        # todo steal equations from blond
        self._invalidate_cache()
