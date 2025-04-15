from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.utils import bmath as bm, precision

if TYPE_CHECKING:
    from typing import Tuple, Literal
    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray
    from .beam import Beam
    from .profile import Profile
    from ..impedances.impedance import (
        _InducedVoltage,
        InducedVoltageTime,
        InducedVoltageFreq,
        InductiveImpedance,
        InducedVoltageResonator,
    )

    InducedVoltageTyes = (
        Literal[
            InducedVoltageTime,
            InducedVoltageFreq,
            InductiveImpedance,
            InducedVoltageResonator,
        ]
        | _InducedVoltage
    )


class Lockable:
    def __init__(self):
        """Class that can be locked and unlocked"""
        self.__locked = False

    def lock(self):
        self.__locked = True

    def unlock(self):
        self.__locked = True

    @property
    def is_locked(self):
        return self.__locked


class ProfileContainer(Lockable):
    def __init__(self):
        """Helper class to contain several Profile objects"""
        super().__init__()
        self._profiles: Tuple[Profile] = tuple()

        # Memory helper to calculate histogram of several profiles efficiently
        self._total_histogram: NumpyArray | CupyArray = None

    @property
    def bin_width(self):
        if self.n_profiles == 0:
            msg = (
                "`bin_width` is undefined because no profiles are in the "
                "`MultiProfileContainer`, use `add_profile()` first!"
            )
            raise ValueError(msg)
        return self._profiles[0].bin_width

    @property
    def n_profiles(self):
        return len(self._profiles)

    @property
    def number_of_bins(self):
        if self.n_profiles == 0:
            msg = (
                "`number_of_bins` is undefined because no profiles are in the "
                "`MultiProfileContainer`, use `add_profile()` first!"
            )
            raise ValueError(msg)
        return self._profiles[0].number_of_bins

    def to_gpu(self, recursive: bool = True):
        import cupy as cp

        self._total_histogram = cp.array(self._total_histogram)
        for i, profile in enumerate(self._profiles):
            profile.n_macroparticles = self._total_histogram[:, i].view()

    def add_profile(self, profile: Profile):
        assert not self.is_locked
        if self.n_profiles > 0:
            msg = f"{profile.bin_width=}, but must be {self.bin_width}"
            assert np.isclose(profile.bin_width, self.bin_width), msg
            msg = f"{profile.number_of_bins=}, but must be {self.number_of_bins}"
            assert profile.number_of_bins == self.number_of_bins, msg
            for p in self._profiles:
                start = p.cut_left
                stop = p.cut_right
                if not (
                    ((profile.cut_left <= start) & (profile.cut_right <= start))
                    or ((profile.cut_left >= stop) & (profile.cut_right >= stop))
                ):
                    raise ValueError("Profiles are overlapping")
        self._profiles = (*self._profiles, profile)
        self._update_memory()

    def _update_memory(self):
        self._total_histogram = bm.zeros(
            shape=(self.number_of_bins, self.n_profiles),
            dtype=bm.precision.real_t,
            order="F",
        )
        for i, profile in enumerate(self._profiles):
            # TODO WRITE TEST THAT ITS WRITING TO THE CORRECT POSITION IN
            self._total_histogram[:, i] = profile.n_macroparticles
            profile.n_macroparticles = self._total_histogram[:, i].view()

    def track(self):
        #  TODO implement more efficient method using `_total_histogram`
        for profile in self._profiles:
            profile.track()

    def __len__(self):
        return len(self._profiles)

    def __iter__(self):
        for profile in self._profiles:
            yield profile


class InducedVoltageContainer(Lockable):
    def __init__(self):
        """Helper class to contain several InducedVoltage objects"""

        super().__init__()
        self._induced_voltage_objects: Tuple[InducedVoltageTyes] = tuple()

    @property
    def n_objects(self):
        return len(self._induced_voltage_objects)

    def add_induced_voltage(self, induced_voltage: InducedVoltageTyes):
        assert not self.is_locked
        self._induced_voltage_objects = (
            *self._induced_voltage_objects,
            induced_voltage,
        )

    def __len__(self):
        return len(self._induced_voltage_objects)

    def __iter__(self):
        for induced_voltage_object in self._induced_voltage_objects:
            yield induced_voltage_object


class TotalInducedVoltageNew:
    def __init__(
        self,
        beam: Beam,
        profile_container: ProfileContainer,
        induced_voltage_container: InducedVoltageContainer,
        track_update_wake_kernel: bool,
    ):
        """Helper to calculate induced voltage for several profiles

        Parameters
        ----------
        beam
            Class containing the beam properties.
        profile_container
            Helper class to contain several Profile objects
        induced_voltage_container
            Helper class to contain several InducedVoltage objects
        track_update_wake_kernel
            If True, the wake kernel is recalculated for each `track` call.
            This might be necessary, but can lead to low performance.
        """
        self._beam = beam

        self._profile_container = profile_container
        self._profile_container.lock()  # DONT ALLOW ANY CHANGED ANYMORE

        self._induced_voltage_container = induced_voltage_container
        self._induced_voltage_container.lock()  # DONT ALLOW ANY CHANGED ANYMORE

        self._profile_source = self._profile_container._total_histogram[
            :, :
        ].T.flatten()
        self.track_update_wake_kernel = track_update_wake_kernel
        self._compressed_wake_kernel = self._get_compressed_wake_kernel(
            self._profile_container
        )

        self._induced_voltage_amplitude: NumpyArray | CupyArray = None  # TODO
        # self._induced_voltage_time: NumpyArray | CupyArray # TODO

    @property
    def profile(self):
        # TODO: PRELIMINARY CODE
        return self._profile_container._profiles[0]

    @property
    def induced_voltage(self):
        # TODO: PRELIMINARY CODE
        from scipy.constants import elementary_charge as e

        induced_voltage = -(
            self._beam.particle.charge * e * self._beam.ratio * self.profile.wake
        )
        return induced_voltage

    def to_gpu(self, recursive: bool = True):
        # TODO: PRELIMINARY CODE
        import cupy as cp

        self._compressed_wake_kernel = cp.array(self._compressed_wake_kernel)
        if recursive:
            self._profile_container.to_gpu(recursive=recursive)

    def reprocess(self):
        # TODO: PRELIMINARY CODE
        self._compressed_wake_kernel = self._get_compressed_wake_kernel(
            self._profile_container
        )

    def induced_voltage_sum(self):
        # TODO: PRELIMINARY CODE
        return self._induced_voltage_sum()

    def track(self):
        """Apply kick function to beam"""

        self._induced_voltage_sum()
        for profile in self._profile_container:
            profile: Profile
            bm.linear_interp_kick(
                dt=self._beam.dt,
                dE=self._beam.dE,
                voltage=self.induced_voltage,  # NOQA, This is set by `_induced_voltage_sum`
                bin_centers=profile.bin_centers,
                charge=self._beam.particle.charge,
                acceleration_kick=0.0,
            )
            # TODO activate del wake
            # del profile.wake  # NOQA remove attribute, so its not visible
            # from the outside.

    def _induced_voltage_sum(self):
        has_one_profile = self._profile_container.n_profiles == 1

        if not has_one_profile:
            for profile_j, profile_target in enumerate(self._profile_container):
                profile_target: Profile
                # attribute `wake` to be used in `track`. This is private to
                # `TotalInducedVoltageNew`
                profile_target.wake = bm.zeros(profile_target.number_of_bins)

        # Size of compressed wake per profile, defined in `_get_compressed_wake_kernel`
        step = 2 * self._profile_container.number_of_bins

        if self.track_update_wake_kernel:
            # This is potentially a non-performing operation,
            # but would be required if the profiles change their position
            self._compressed_wake_kernel = self._get_compressed_wake_kernel(
                self._profile_container
            )
        if len(self._compressed_wake_kernel) > 2048:
            convolve = bm.fftconvolve
        else:
            convolve = bm.convolve
        for profile_i, profile_source in enumerate(self._profile_container):
            profile_source: Profile

            compressed_wake = convolve(
                self._compressed_wake_kernel,
                profile_source.n_macroparticles[:],
                mode="same",
            )
            DEV_DEBUG = False
            if DEV_DEBUG:
                try: # .get() so it works only on GPU
                    from matplotlib import pyplot as plt
                    fig_n = plt.gcf().number
                    plt.figure(22)
                    plt.clf()
                    plt.subplot(3, 1, 1)
                    plt.title("self._compressed_wake_kernel")
                    plt.plot(self._compressed_wake_kernel.get(), label=f"{profile_i}")
                    plt.subplot(3, 1, 2)
                    plt.title("profile_source.n_macroparticles")
                    plt.plot(profile_source.n_macroparticles.get(), label=f"{profile_i}")
                    plt.subplot(3, 1, 3)
                    plt.plot(compressed_wake.get(), label=f"{profile_i=}")
                    plt.figure(fig_n)
                except AttributeError:
                    pass

            for profile_j, profile_target in enumerate(self._profile_container):
                profile_target: Profile
                # skip waves backwards in time, so that the first profile
                # affects all following profiles, but the last profile affects
                # only itself.
                if profile_j < profile_i:
                    continue

                # Read/write the wakefield from the compressed wake.
                dev_debug = profile_j - profile_i
                # print(f"{profile_j=} {dev_debug=}")
                assert dev_debug >= 0
                # Because the wake is `2 * number_of_bins` in the compressed wake
                # `start` and `stop` are shifted to include only the relevant
                # `1 * number_of_bins`.
                start = (profile_j - profile_i + 0) * step + step // 4
                stop = (profile_j - profile_i + 1) * step - step // 4
                size = stop - start


                # Set fake attribute `wake` to be used in `track`. This is
                # private to `TotalInducedVoltageNew`
                """idxs = np.arange(len(compressed_wake))
                plt.figure(22)

                plt.subplot(3, 1, 3)

                plt.plot(
                    idxs[start:stop],
                    compressed_wake[start:stop],
                    "x",
                    label=f"{profile_i=} {profile_j=}",
                )
                plt.legend()
                plt.figure(fig_n)"""
                if has_one_profile:
                    profile_target.wake = compressed_wake[start:stop]
                else:
                    assert size == len(profile_target.wake), (
                        f"{size} !=" f" {len(profile_target.wake)}"
                    )
                    profile_target.wake += compressed_wake[start:stop]


    def _get_compressed_wake_kernel(
        self, profile_container: ProfileContainer
    ) -> NumpyArray | CupyArray:
        """Calculates the wake kernel at every profile"""
        concat_later = []
        # TODO CONSIDER MTW
        t_min_glob = min(map(lambda p: p.cut_left, self._profile_container))
        first_profile_center = (
            profile_container._profiles[0].cut_left
            + profile_container._profiles[0].cut_right
        ) / 2
        t_offset = first_profile_center
        for profile_dest in profile_container:
            profile_dest: Profile
            width = profile_dest.cut_right - profile_dest.cut_left

            # make the kernel_width 2 * profile_width,
            # so that during convolution, the kernel will reach all profile
            # entries when at left and right end of profile
            t_start = profile_dest.cut_left - width / 2 - t_offset
            t_stop = profile_dest.cut_right + width / 2 - t_offset
            n_entries = 2 * self._profile_container.number_of_bins

            if t_start < 0:
                assert np.isclose(t_start, -t_stop), (
                    "Must be symmetric "
                    f"around 0, "
                    f"but {t_start=}, "
                    f"and {t_stop=}"
                )
                n_entries -= 1
            assert t_start < t_stop, f"{t_start=} {t_stop=}"
            msg = f"Bins must be  even, but {profile_dest.number_of_bins=}"
            assert profile_dest.number_of_bins % 2 == 0, msg

            wake_kernel_at_single_profile = bm.zeros(
                # Factor 2 because `start` and `stop` was increased by `width / 2`
                n_entries,
                dtype=precision.real_t,
            )
            # Sum all wakes (impedances) for a single profile target
            for induced_voltage_object in self._induced_voltage_container:
                wake_kernel_at_single_profile += induced_voltage_object.get_wake_kernel(
                    t_start=t_start,
                    t_stop=t_stop,
                    n=len(wake_kernel_at_single_profile),
                )

            concat_later.append(wake_kernel_at_single_profile)

        for wake_i, wake in enumerate(concat_later[1:]):
            # This script can be further developed to consider different
            # wake kernel sizes.  Must be considered in
            # `_induced_voltage_sum` too.
            msg = (f"Not all wake kernels have the same size: "
                   f"{[len(w) for w in concat_later]}")
            assert len(wake) == (len(concat_later[0])+1), f"{len(wake)} == {(len(concat_later[0]) + 1)}"
        compressed_wake_kernel = bm.concatenate(concat_later, dtype=precision.real_t)

        return compressed_wake_kernel
