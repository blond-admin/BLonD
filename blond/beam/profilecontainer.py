from __future__ import annotations
from typing import TYPE_CHECKING

from scipy.signal import fftconvolve

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

    def add_profile(self, profile: Profile):
        assert not self.is_locked
        if self.n_profiles > 0:
            msg = f"{profile.bin_width=}, but must be {self.bin_width}"
            assert profile.bin_width == self.bin_width, msg
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

        self.track_update_wake_kernel = track_update_wake_kernel
        self._compressed_wake_kernel = self._get_compressed_wake_kernel()

        self._induced_voltage_amplitude: NumpyArray | CupyArray = None  # TODO
        # self._induced_voltage_time: NumpyArray | CupyArray # TODO

    def track(self):
        """Apply kick function to beam"""

        self._induced_voltage_sum()
        for profile in self._profile_container:
            profile: Profile
            bm.linear_interp_kick(
                dt=self._beam.dt,
                dE=self._beam.dE,
                voltage=profile.wake,  # NOQA, This is set by `_induced_voltage_sum`
                bin_centers=profile.bin_centers,
                charge=self._beam.particle.charge,
                acceleration_kick=0.0,
            )
            del profile.wake  # NOQA remove attribute, so its not visible
            # from the outside.

    def _induced_voltage_sum(self):
        for profile_j, profile_dest in enumerate(self._profile_container):
            profile_dest: Profile
            profile_dest.wake = bm.zeros(profile_dest.number_of_bins)

        # Size of compressed wake per profile, defined in `_get_compressed_wake_kernel`
        step = 2 * self._profile_container.number_of_bins

        for profile_i, profile_source in enumerate(self._profile_container):
            profile_source: Profile
            if self.track_update_wake_kernel:
                self._compressed_wake_kernel = self._get_compressed_wake_kernel()

            # skip waves backwards in time, so that the first profile
            # affects all following profiles, but the last profile affects
            # only itself.
            start = profile_i * step
            compressed_wake = fftconvolve(
                profile_source, self._compressed_wake_kernel[start:]
            )

            for profile_j, profile_dest in enumerate(self._profile_container):
                profile_dest: Profile
                # skip waves backward in time (explained above)
                if profile_j < profile_i:
                    continue

                # Read/write the wakefield from the compressed wake.

                # Because the wake is `2 * number_of_bins` in the compressed wake
                # `start` and `stop` are shifted to include only the relevant
                # `1 * number_of_bins`.
                start = (profile_j - profile_i + 0) * step + step / 4
                stop = (profile_j - profile_i + 1) * step - step / 4
                profile_dest.wake += compressed_wake[start:stop]  # Set fake
                # attribute to be used in `track`. This is private to
                # `TotalInducedVoltageNew`

    def _get_compressed_wake_kernel(
        self,
    ) -> NumpyArray | CupyArray:
        """Calculates the wake kernel at every profile"""
        concat_later = []
        # TODO CONSIDER MTW
        for profile_dest in self._profile_container:
            profile_dest: Profile
            width = profile_dest.cut_right - profile_dest.cut_left
            t_start = profile_dest.cut_left - width / 2
            t_stop = profile_dest.cut_right + width / 2

            assert t_start < t_stop, f"{t_start=} {t_stop=}"
            msg = f"Bins must be  even, but {profile_dest.number_of_bins=}"
            assert profile_dest.number_of_bins % 2 == 0, msg

            wake_kernel_at_single_profile = bm.zeros(
                # Factor 2 because `start` and `stop` was increased by `width / 2`
                2 * self._profile_container.number_of_bins,
                dtype=precision.real_t,
            )
            # Sum all wakes (impedances) for a single profile target
            for induced_voltage_object in self._induced_voltage_container:
                wake_kernel_at_single_profile += induced_voltage_object.get_wake(
                    t_start=t_start, t_stop=t_stop, n=len(wake_kernel_at_single_profile)
                )

            concat_later.append(wake_kernel_at_single_profile)

        for wake in concat_later:
            # This script can be further developed to consider different
            # wake kernel sizes.  Must be considered in
            # `_induced_voltage_sum` too.
            msg = "Not all wake kernels have the same size.."
            assert len(wake) == len(concat_later[0]), msg
        compressed_wake_kernel = bm.concatenate(concat_later, dtype=precision.real_t)

        return compressed_wake_kernel
