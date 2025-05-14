from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import numpy as np

from ...utils import bmath as bm

if TYPE_CHECKING:
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

    from ...beam.profile import Profile


class Lockable:
    def __init__(self):
        """Class that can be locked and unlocked"""
        self.__locked = False

    def lock(self):
        self.__locked = True

    def unlock(self):
        self.__locked = False

    @property
    def is_locked(self):
        return self.__locked


class _ProfileContainer(Lockable):
    """Helper class to contain several Profile objects"""

    def __init__(self):
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

    def to_cpu(self, recursive: bool = True):
        import cupy as cp

        self._total_histogram = cp.asnumpy(self._total_histogram)
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


class EquiSpacedProfiles(_ProfileContainer):
    """Helper class to contain several evently spaced Profile objects"""

    def __init__(self):
        super().__init__()

    def add_profile(self, profile: Profile):
        if self.n_profiles >= 2:
            dist_reference = self._profiles[1].center - self._profiles[0].center
            dist_new = profile.center - self._profiles[1].center
            if not np.isclose(dist_reference, dist_new):
                raise ValueError(
                    "Can only accept evenly spaced profiles!"
                    f" The distance of the first two profiles is "
                    f" {dist_reference}, but the distance to the added"
                    f" profile is {dist_new}."
                )
        super().add_profile(profile=profile)
