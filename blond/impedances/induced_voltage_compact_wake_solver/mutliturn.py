from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray

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
        _InducedVoltageSolver,
        InducedVoltageFreqDomainSolver,
        InductiveImpedanceShortcutSolver,
    )

    InducedVoltageTyes = (
        Literal[
            # InducedVoltageTime, # disallowed because resampling of wake in
            # time-domain could lead to a lot of problems
            InducedVoltageFreqDomainSolver,
            InductiveImpedanceShortcutSolver,
            # InducedVoltageResonator, # use InducedVoltageFreq with
            # Resonatros insteead
        ]
        | _InducedVoltageSolver
    )


def find_closest(arr: NDArray, target: float):
    arr = np.array(arr)
    idx = np.abs(arr - target).argmin()
    return arr[idx]


@dataclass(frozen=True)  # Dont allow modifications
class ProfileRangeMultiTurnEvolution:
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
        """Profile limits in absolute time, fist profile centered"""

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
    def __init__(
        self,
        beam: Beam,
        profile: Profile,
        profile_evolution: ProfileRangeMultiTurnEvolution,
        induced_voltage_container: InducedVoltageContainer,
        current_turn: int = 0,
    ):
        self._beam = beam
        self._profile = profile
        self._profile_evolution = profile_evolution
        self.current_turn = current_turn

        self._induced_voltage_container = induced_voltage_container
        self._induced_voltage_container.lock()  # DONT ALLOW ANY CHANGE ANYMORE

        self._entire_multiparticle_wake = np.zeros(  # fixme GPU?
            self._profile_evolution.max_turns * self.profile.number_of_bins * 2
        )

        self._turn_i = 0  # incremented by self.track()
        self._turn_i_already_calculated: int = -1

        self._n_fft_faster = 1024  # decides whether to use `convolve` or `fftconvolve`

    @property
    def profile(self):
        # TODO: PRELIMINARY CODE
        return self._profile

    @property
    def induced_voltage(self):
        # TODO: PRELIMINARY CODE
        from scipy.constants import elementary_charge as e

        profile = self.profile
        induced_voltage_ = -(
            self._beam.particle.charge * e * self._beam.ratio * profile.wake
        )
        return induced_voltage_

    def to_gpu(self, recursive: bool = True):
        # TODO: PRELIMINARY CODE

        if recursive:
            self.profile.to_gpu(recursive=recursive)
        self._device: DeviceType = "GPU"

    def to_cpu(self, recursive=True):
        # TODO: PRELIMINARY CODE

        if recursive:
            self.profile.to_cpu(recursive=recursive)
        self._device: DeviceType = "CPU"

    def reprocess(self):
        # TODO: PRELIMINARY CODE
        if self._turn_i != self._turn_i_already_calculated:
            self._compressed_wake_kernel = self._get_compressed_wake_kernel(
                turn_i=self._turn_i
            )

    def induced_voltage_sum(self):
        # TODO: PRELIMINARY CODE
        return self._induced_voltage_sum()

    def track(self):
        """Apply kick function to beam"""

        self._induced_voltage_sum()
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

        self._induced_voltage_sum()
        bm.linear_interp_kick(
            dt=ghost_beam.dt,
            dE=ghost_beam.dE,
            voltage=self.induced_voltage,
            bin_centers=self.profile.bin_centers,
            charge=self._beam.particle.charge,  # FIXME is this the correct
            # charge? Should it bhe the one of ghost_beam?
            acceleration_kick=0.0,
        )

    def _induced_voltage_sum(self):
        assert self._turn_i_already_calculated == (self._turn_i - 1)

        # Update wake kernel on each turn
        self._compressed_wake_kernel = self._get_compressed_wake_kernel(
            turn_i=self._turn_i
        )
        fig_n = plt.gcf().number
        plt.figure("dev")
        plt.plot(self._compressed_wake_kernel)
        plt.figure(fig_n)
        plt.draw()
        plt.pause(1)
        # Size of compressed wake per profile, defined in `_get_compressed_wake_kernel`
        step = 2 * self._profile.number_of_bins

        if len(self._compressed_wake_kernel) > self._n_fft_faster:
            convolve = bm.fftconvolve
        else:
            convolve = bm.convolve

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

        assert size == self.profile.number_of_bins, (
            f"{size=} == " f"{self.profile.number_of_bins=}"
        )

        idxs = np.arange(len(self._entire_multiparticle_wake))
        fig_n = plt.gcf().number
        plt.figure(22)

        plt.subplot(3, 1, 1)
        plt.plot(idxs, self._entire_multiparticle_wake)
        plt.subplot(3, 1, 1)

        plt.plot(
            idxs[start:stop],
            self._entire_multiparticle_wake[start:stop],
            "x",
        )
        plt.legend()
        plt.figure(fig_n)

        # Set fake attribute `wake` to be used in `track`. This is
        # private to `TotalInducedVoltageNew`
        self.profile.wake = self._entire_multiparticle_wake[start:stop]
        self._turn_i_already_calculated = self._turn_i
        self._turn_i += 1
        self._entire_multiparticle_wake = self._entire_multiparticle_wake[(1 * step) :]

    def _get_compressed_wake_kernel(self, turn_i: int) -> NumpyArray | CupyArray:
        """Calculates the wake kernel at every profile"""
        n_entries = 2 * self._profile.number_of_bins

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
            step = width / self._profile.number_of_bins
            t_start = cut_left - width / 2 + step / 2
            t_stop = cut_right + width / 2 - step / 2

            msg = f"Bins must be  even, but {self._profile.number_of_bins=}"
            assert self._profile.number_of_bins % 2 == 0, msg
            time_array = np.linspace(t_start, t_stop, n_entries)
            if i == 0:
                # force to have one entry that is t=0
                # Offset must be considered correctly during convolution.
                offset = time_array[len(time_array) // 2 - 1]
            time_array -= offset
            # Sum all wakes (impedances) for a single profile target
            for induced_voltage_object in self._induced_voltage_container:
                induced_voltage_object: InducedVoltageFreqDomainSolver
                for imp in induced_voltage_object.impedance_source_list:
                    wake_tmp = imp.get_nonperiodic_wake(time_array=time_array)
                    compressed_wake_kernel[sel_tmp] += wake_tmp

        return compressed_wake_kernel
