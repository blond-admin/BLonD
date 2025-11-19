"""Collection of abstract classes to handle the calculation of wake potentials.

Authors
-------
Simon Lauber
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..._core.backends.backend import backend
from ..._core.base import BeamPhysicsRelevant
from ..._core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

    from ..._core.beam.base import BeamBaseClass
    from ..._core.simulation.simulation import Simulation
    from ..profiles import ProfileBaseClass


class WakeFieldSolver:
    """Abstract class for a solver that generates wake fields based on beam profiles."""

    @abstractmethod  # pragma: no cover
    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        """Lateinit method when :class:`blond.physics.impedances.base.WakeField` is late-initialized.

        Parameters
        ----------
        simulation
            Simulation context manager
        parent_wakefield
            Wakefield that this solver affiliated to
        """
        pass

    @abstractmethod  # pragma: no cover
    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """Calculates the induced voltage based on the beam profile and beam parameters.

        Parameters
        ----------
        beam
            Simulation object of a particle beam

        Returns
        -------
        induced_voltage
            Induced voltage, in [V]
        """
        pass


class WakeFieldSource(ABC):
    """General abstract class for wake fields."""

    def __init__(self, is_dynamic: bool):
        self.is_dynamic = is_dynamic


class TimeDomain(ABC):
    """Indication of a source is defined in time domain."""

    @abstractmethod  # pragma: no cover
    def get_wake_impedance(
        self,
        time: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_fft: int,
    ) -> NumpyArray:
        """Get impedance equivalent to the partial wake in time domain.

        Parameters
        ----------
        time
            Time array to get wake, in [s]
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object
        n_fft
            number of points to be used in the fft

        Returns
        -------
        wake_impedance

        """
        pass


class TimeDomainCounterRotation(ABC):
    """Indication of a source, which has a defined wakefield for the counterrotating case."""

    @abstractmethod  # pragma: no cover
    def get_wake(
        self, time: NumpyArray
    ) -> NumpyArray:  # TODO: this function should be moved to TimeDomain
        """Get wake potential equivalent to the partial wake in time domain.

        Parameters
        ----------
        time : NumpyArray
            time array at which the wake is calculated [V]
        """
        pass

    @abstractmethod  # pragma: no cover
    def get_wake_counter_rotation(self, time: NumpyArray) -> NumpyArray:
        """Get wake potential equivalent to the partial wake in time domain for the counter-rotating case.

        Parameters
        ----------
        time : NumpyArray
            time array at which the wake is calculated, in [s]

        Returns
        -------
        wake_potential: NumpyArray
            potential array, in [V]

        """
        pass

    @abstractmethod  # pragma: no cover
    def get_wake_impedance_counter_rotation(
        self,
        time: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_fft: int,
    ) -> NumpyArray:
        """Get impedance equivalent to the partial wake in time domain for the counter-rotating case.

        Parameters
        ----------
        time
            Time array to get wake, in [s]
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object
        n_fft
            number of points used in the fft

        Returns
        -------
        wake_impedance

        """
        pass


class FreqDomain(ABC):
    """Indication of a source is defined in frequency domain."""

    @abstractmethod  # pragma: no cover
    def get_impedance(
        self,
        freq_x: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> NumpyArray | CupyArray:
        """Return the impedance in the frequency domain.

        Parameters
        ----------
        freq_x
            Frequency axis, in [Hz].
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object

        Returns
        -------
        impedance
            Complex impedance array.
        """
        pass


class ImpedanceBaseClass(BeamPhysicsRelevant):
    """Abstract class on how to calculate induced voltages."""

    def __init__(
        self,
        section_index: int = 0,
        profile: ProfileBaseClass | None = None,
    ):
        super().__init__(section_index=section_index)
        self._profile = profile

    @property  # as readonly attributes
    def profile(self) -> ProfileBaseClass:
        """The reference profile that is causing the wake."""
        return self._profile

    @abstractmethod  # pragma: no cover
    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """Calculates the induced voltage based on the beam profile and beam parameters.

        Parameters
        ----------
        beam
            Simulation object of a particle beam

        Returns
        -------
        induced_voltage
            Induced voltage, in [V]
        """
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called.

        simulation
            Simulation context manager
        beam
            Simulation `Beam` object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        pass

    @requires(
        [
            "BeamPhysicsRelevantElements",  # for .section_index,
        ]
    )
    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        from ..profiles import ProfileBaseClass  # prevent cyclic import

        if self._profile is None:
            profiles = simulation.ring.elements.get_elements(
                ProfileBaseClass, section_i=self.section_index
            )
            assert len(profiles) == 1, (
                f"Found {len(profiles)} profiles in "
                f"{self.section_index=}, but can only handle one. Set the attribute "
                f"`your_impedance.profile` in advance or remove the second "
                f"profile from this group."
            )
            self._profile = profiles[0]
        else:
            pass


class WakeField(ImpedanceBaseClass):
    """Manager class to calculate wake-fields.

    Parameters
    ----------
    sources
        List of sources that cause wake-fields
    solver
        Solver to calculate the induced voltage from the sources
    section_index
        Section index to group elements into sections
    profile
        Object for calculation of beam profiles

    Attributes
    ----------
    sources
        List of sources that cause wake-fields
    solver
        Solver to calculate the induced voltage from the sources
    """

    def __init__(
        self,
        sources: tuple[WakeFieldSource, ...],
        solver: WakeFieldSolver | None,
        section_index: int = 0,
        profile: ProfileBaseClass | None = None,
    ):
        """Manager class to calculate wake-fields.

        Parameters
        ----------
        sources
            List of sources that cause wake-fields
        solver
            Solver to calculate the induced voltage from the sources
        section_index
            Section index to group elements into sections
        profile
            Object for calculation of beam profiles

        """
        super().__init__(section_index=section_index, profile=profile)

        self.solver = solver
        self.sources = sources
        self._induced_voltage = None

    def info_string(self, prefix="") -> str:
        """Inform that the profile is also executed within the track method."""
        content = (
            f"{self.profile.info_string(prefix=prefix + ' ↓ ')}\n"
            f"{super().info_string(prefix=prefix)}"
        )
        return content

    @property
    def induced_voltage(self) -> NumpyArray | CupyArray:
        """Induced voltage in [V] from given beam profile and sources."""
        if self._induced_voltage is None:
            raise AttributeError("Use `calc_induced_voltage` first!")
        return self._induced_voltage

    @requires(["EnergyCycleBase", "BeamBaseClass", "Beam"])  # because
    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)
        assert len(self.sources) > 0, (
            "Provide for at least one `WakeFieldSource`"
        )
        self.solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=self
        )

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """Calculation of induced voltage from all sources.

        Parameters
        ----------
        beam
            Simulation object of a particle beam

        Returns
        -------
        induced_voltage
        """
        self._induced_voltage = self.solver.calc_induced_voltage(beam=beam)[
            : self.profile.n_bins
        ]
        # the induced voltage has to be provided with the backend precision
        # because the track() method below requires it by calling the backend.
        return self.induced_voltage[: self.profile.n_bins]

    def track(self, beam: BeamBaseClass) -> None:
        """Calculate induced voltage and apply this voltage to the beam.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        if self.profile.active:
            self.profile.track(beam=beam)
        induced_voltage = self.calc_induced_voltage(beam=beam)
        assert (induced_voltage).dtype == backend.float
        backend.specials.kick_induced_voltage(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            # TODO improve induced_voltage calculation data type for speedup
            voltage=induced_voltage.astype(backend.float),
            bin_centers=self.profile.hist_x,  # base for induced voltage
            charge=beam.particle_type.charge,
            acceleration_kick=0.0,  # TODO was this ever required??
        )

    @staticmethod
    def headless(
        beam: BeamBaseClass,
        sources: tuple[WakeFieldSource, ...],
        solver: WakeFieldSolver,
        section_index: int = 0,
        profile: ProfileBaseClass | None = None,
    ):
        """Initialize the full class.

        Parameters
        ----------
        sources
            List of sources that cause wake-fields
        solver
            Solver to calculate the induced voltage from the sources
        section_index
            Section index to group elements into sections
        profile
            Object for calculation of beam profiles

        Returns
        -------
        Instance with lateinit methods executed.

        """
        wf = WakeField(
            sources=sources,
            solver=solver,
            section_index=section_index,
            profile=profile,
        )
        from unittest.mock import Mock

        from ..._core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        wf.on_init_simulation(simulation=simulation)
        wf.on_run_simulation(
            simulation=simulation, beam=beam, n_turns=1, turn_i_init=0
        )
        return wf
