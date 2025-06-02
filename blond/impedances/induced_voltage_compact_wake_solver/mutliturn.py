from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy._typing import NDArray

from ..impedance_sources import Resonators
from ...utils import bmath as bm, precision
from ..impedance import TotalInducedVoltageAbstract

if TYPE_CHECKING:
    from typing import Literal
    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray
    from . import InducedVoltageContainer
    from ...beam.profile import Profile
    from ...beam.beam import Beam
    from ...utils.types import DeviceType

    from ...impedances.impedance import (
        _InducedVoltage,
        InducedVoltageFreq,
        InductiveImpedance,
    )

    InducedVoltageTyes = (
        Literal[
            # InducedVoltageTime, # disallowed because resampling of wake in
            # time-domain could lead to a lot of problems
            InducedVoltageFreq,
            InductiveImpedance,
            # InducedVoltageResonator, # use InducedVoltageFreq with
            # Resonatros instead
        ]
        | _InducedVoltage
    )


@dataclass(frozen=True)  # Dont allow modifications
class ProfileRangeMultiTurnEvolution:
    """Dataclass to store how the beam profile evolves for several turns

    Parameters
    ----------
    starts
        `cut_left` of a `Profile` for n turns
    stops
        `cut_right` of a `Profile` for n turns
    t_revs
        `t_rev` of a `Ring` for n turns
        The evolution of the revolution time of a synchrotron

    """

    starts: NDArray
    stops: NDArray
    t_revs: NDArray

    def __post_init__(self):
        # Check that sizes are correct
        n = int(
            np.median(
                (
                    len(self.starts),  # Who knows what is the intended size..
                    len(self.stops),  # Probably it's the median!
                    len(self.t_revs),
                )
            )
        )

        assert len(self.starts.shape) == 1, f"{self.starts.shape=}"
        assert len(self.starts) == n, f"{len(self.starts)=}, but must be {n}"

        assert len(self.stops.shape) == 1, f"{self.stops.shape=}"
        assert len(self.stops) == n, f"{len(self.stops)=}, but must be {n}"

        assert len(self.t_revs.shape) == 1, f"{self.t_revs.shape=}"
        assert len(self.t_revs) == n, f"{len(self.t_revs)=}, but must be {n}"

        assert np.all(self.starts[:] < self.stops[:])
        # Dont allow modifications
        self.starts.flags.writeable = False
        self.stops.flags.writeable = False
        self.t_revs.flags.writeable = False

    @property
    def max_turns(self) -> int:
        return len(self.starts)  # __post_init__ asserts  that all arrays
        # have the same size

    def update_profile(self, profile: Profile, turn_i: int):
        profile.cut_options.cut_left = float(self.starts[turn_i])
        profile.cut_options.cut_right = float(self.stops[turn_i])
        profile.set_slices_parameters()

    def get_mutliturn_profile_limits(self, turn_start: int = 0) -> NDArray:
        """Profile limits in absolute time, fist profile centered

        Parameters
        ----------
        turn_start
            Will ignore all turns less than `turn_start`
            and put the zero to `cut_left` of the n-th turn.
        """

        #  profile                   profile (next turn)
        # [       ]                    [             ]
        # ----|----------------------->
        #     |       One Turn distance
        #   t = 0
        first_center = (self.stops[turn_start] + self.starts[turn_start]) / 2
        time_distance = np.cumsum(self.t_revs[turn_start:]) - self.t_revs[turn_start]

        offsets = time_distance - first_center
        starts_absolute = self.starts[turn_start:] + offsets
        stops_absolute = self.stops[turn_start:] + offsets
        return starts_absolute, stops_absolute


class InducedVoltageCompactWakeMultiTurnSolver(TotalInducedVoltageAbstract):
    """Solver to calculate multi-turn wakes with a single profile and compact convolution

    Parameters
    ----------
    beam
        Class containing the beam properties.
    profile
        Profile object
    profile_evolution
        Helper class to know how the profile boundaries look like for n turns
    induced_voltage_container
        Helper class to contain several InducedVoltage objects
    current_turn
        Initial turn to calculate from

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from blond.impedances.impedance import InducedVoltageFreq
    >>> from blond.impedances.impedance_sources import Resonators
    >>> from blond.impedances.induced_voltage_compact_wake_solver import \
    >>>     InducedVoltageContainer, ProfileRangeMultiTurnEvolution, \
    >>>     InducedVoltageCompactWakeMultiTurnSolver
    >>>
    >>> resonators = Resonators([4.5e6], [200.222e6], [200])
    >>> induced_voltage_freq = InducedVoltageFreq(
    >>>     beam=beam,
    >>>     profile=profile1,
    >>>     impedance_source_list=[resonators],
    >>> )
    >>>
    >>> induced_voltage_container = InducedVoltageContainer()
    >>> induced_voltage_container.add_induced_voltage(induced_voltage_freq)
    >>>
    >>> profile_evolution1 = ProfileRangeMultiTurnEvolution(
    >>>     starts=profile1.cut_left * np.ones(ring.n_turns + 1), # no change
    >>>     stops=profile1.cut_right * np.ones(ring.n_turns + 1), # no change
    >>>     t_revs=ring.t_rev,
    >>> )
    >>> total_induced_voltage = InducedVoltageCompactWakeMultiTurnSolver(
    >>>     beam=beam,
    >>>     profile=profile1,
    >>>     induced_voltage_container=induced_voltage_container,
    >>>     profile_evolution=profile_evolution1,
    >>> )

    """
    def __init__(
        self,
        beam: Beam,
        profile: Profile,
        profile_evolution: ProfileRangeMultiTurnEvolution,
        induced_voltage_container: InducedVoltageContainer,
        current_turn: int = 0,
    ):
        self._beam = beam
        self.profile = profile
        self._profile_evolution = profile_evolution

        self._induced_voltage_container = induced_voltage_container
        self._induced_voltage_container.lock()  # DONT ALLOW ANY CHANGE ANYMORE

        self._entire_multiparticle_wake = np.zeros(  # fixme GPU?
            self._profile_evolution.max_turns * self.profile.n_slices * 2
        )

        self._turn_i = 0  # incremented by self.track()
        self._turn_i_already_calculated: int = -1

        self._n_fft_faster = 1024  # decides whether to use `convolve` or `fftconvolve`


    @property
    def induced_voltage(self):
        """
        Get the induced voltage for the profile.

        Returns
        -------
        induced_voltage : NumpyArray or CupyArray
            The induced voltage corresponding to the reference profile.
        """
        from scipy.constants import elementary_charge as e

        profile = self.profile
        induced_voltage_ = -(
            self._beam.particle.charge * e * self._beam.ratio * profile.wake
        )
        return induced_voltage_

    def to_gpu(self, recursive: bool = True):
        if recursive:
            self.profile.to_gpu(recursive=recursive)
        self._device: DeviceType = "GPU"

    def to_cpu(self, recursive=True):
        if recursive:
            self.profile.to_cpu(recursive=recursive)
        self._device: DeviceType = "CPU"

    def reprocess(self):
        """Recalculates the compressed wake for induced voltage calculation"""
        if self._turn_i != self._turn_i_already_calculated:
            self._compressed_wake_kernel = self._get_compressed_wake_kernel(
                turn_i=self._turn_i
            )

    def track(self):
        """Update induced voltage and apply kick function to beam"""
        self.induced_voltage_sum()
        bm.linear_interp_kick(
            dt=self._beam.dt,
            dE=self._beam.dE,
            voltage=self.induced_voltage,
            bin_centers=self.profile.bin_centers,
            charge=self._beam.particle.charge,
            acceleration_kick=0.0,
        )
        # TODO activate del wake
        # del profile.wake  # NOQA remove attribute, so its not visible
        # from the outside.

    def track_ghosts_particles(self, ghost_beam: Beam):
        """Apply kick function to beam"""

        self.induced_voltage_sum()
        bm.linear_interp_kick(
            dt=ghost_beam.dt,
            dE=ghost_beam.dE,
            voltage=self.induced_voltage,
            bin_centers=self.profile.bin_centers,
            charge=self._beam.particle.charge,  # FIXME is this the correct
            # charge? Should it bhe the one of ghost_beam?
            acceleration_kick=0.0,
        )

    def induced_voltage_sum(self):
        """
        Compute and update the induced voltage for the current turn.

        This method performs:
        - An update of the compressed wake kernel for all remaining turns,
        - A convolution between the wake kernel (remaining turns) and the current
           macro particle distribution (current turn)
        - Accumulation into a wake buffer for remaining turns

        The computed wake for the current turn is stored in
        `self.profile.wake` for internal use.

        """
        assert self._turn_i_already_calculated == (self._turn_i - 1)

        # Update wake kernel on each turn
        self._compressed_wake_kernel = self._get_compressed_wake_kernel(
            turn_i=self._turn_i
        )
        # Size of compressed wake per profile, defined in `_get_compressed_wake_kernel`
        step = 2 * self.profile.n_slices

        if len(self._compressed_wake_kernel) > self._n_fft_faster:
            convolve = bm.fftconvolve
        else:
            convolve = bm.convolve
        # accumulate multi turn wakes into this buffer
        self._entire_multiparticle_wake[:] += convolve(
            self._compressed_wake_kernel[:],
            self.profile.n_macroparticles[:],
            mode="same",
        )

        # Because the wake is `2 * number_of_bins` in the compressed wake
        # `start` and `stop` are shifted to include only the relevant
        # `1 * number_of_bins`.
        start = 0 * step + step // 4
        stop = 1 * step - step // 4
        size = stop - start

        assert size == self.profile.n_slices, f"{size=} == " f"{self.profile.n_slices=}"



        # Read the current turn wake, which finished accumulation
        # Set fake attribute `wake` to be used in `track`. This is
        # private to `TotalInducedVoltageNew`
        self.profile.wake = self._entire_multiparticle_wake[start:stop]

        self._turn_i_already_calculated = self._turn_i
        self._turn_i += 1

        # forget about the current turn, ready buffer for all remaining turns
        self._entire_multiparticle_wake = self._entire_multiparticle_wake[(1 * step) :]

    def _get_compressed_wake_kernel(self, turn_i: int) -> NumpyArray | CupyArray:
        """Calculates the wake kernel at every profile"""
        n_entries = 2 * self.profile.n_slices

        compressed_wake_kernel = bm.empty(
            ((self._profile_evolution.max_turns - turn_i) * n_entries)
        )
        cuts_left, cuts_right = self._profile_evolution.get_mutliturn_profile_limits(
            turn_start=turn_i
        )
        for i in range(len(cuts_left)):
            cut_left, cut_right = cuts_left[i], cuts_right[i]
            sel_tmp = slice(i * n_entries, (i + 1) * n_entries)

            # make the kernel_width 2 * profile_width,
            # so that during convolution, the kernel will reach all profile
            # entries when at left and right end of profile
            width = cut_right - cut_left
            step = width / self.profile.n_slices
            t_start = cut_left - width / 2 + step / 2
            t_stop = cut_right + width / 2 - step / 2

            msg = f"Bins must be even, but {self.profile.n_slices=}"
            assert self.profile.n_slices % 2 == 0, msg
            time_array = np.linspace(t_start, t_stop, n_entries)
            if i == 0:
                # force to have one entry that is t=0
                # Offset must be considered correctly during convolution.
                offset = time_array[len(time_array) // 2 - 1]
            time_array -= offset
            # Sum all wakes (impedances) for a single profile target
            for induced_voltage_object in self._induced_voltage_container:
                induced_voltage_object: InducedVoltageFreq
                for imp in induced_voltage_object.impedance_source_list:
                    if not isinstance(imp, Resonators):
                        raise NotImplementedError("Discuss with the " "developers")
                    wake_tmp = imp.get_nonperiodic_wake(time_array=time_array)
                    compressed_wake_kernel[sel_tmp] += wake_tmp

        return compressed_wake_kernel
