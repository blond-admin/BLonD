from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

from ...beam.profile import Profile
from ...utils import bmath as bm, precision
from ..impedance import TotalInducedVoltageAbstract

if TYPE_CHECKING:

    from typing import Literal
    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray
    from . import EquiSpacedProfiles, InducedVoltageContainer

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


class InducedVoltageCompactWakeSolver(TotalInducedVoltageAbstract):
    """Solver to calculate induced voltage for several profiles

    Parameters
    ----------
    beam
        Class containing the beam properties.
    equi_spaced_profiles
        Helper class to contain several Profile objects
    induced_voltage_container
        Helper class to contain several InducedVoltage objects
    track_update_wake_kernel
        If True, the wake kernel is recalculated for each `track` call.
        This might be necessary, but can lead to low performance.

    Attributes
    ----------
    track_update_wake_kernel
        If True, the wake kernel is recalculated for each `track` call.
        This might be necessary, but can lead to low performance.
    assume_periodic_wake
        If True, the wake kernel will be symmetric around t=0,
        otherwise it will be only defined for t>0.

    Examples
    --------
    >>> from impedances.induced_voltage_compact_wake_solver import EquiSpacedProfiles, InducedVoltageContainer    >>> from blond.beam.profile import Profile
    >>> from blond.impedances.induced_voltage_compact_wake_solver import (
    >>>     InducedVoltageCompactWakeSolver,
    >>>     )
    >>> from blond.impedances.impedance import (
    >>>     InducedVoltageFreq,
    >>>     InductiveImpedance,
    >>> )
    >>>
    >>> profile1 = Profile(**fill_this)
    >>> inductive_impedance_1 = InductiveImpedance(**fill_this)
    >>> inductive_impedance_2 = InducedVoltageFreq(**fill_this)
    >>> induced_voltage_container = InducedVoltageContainer()
    >>> induced_voltage_container.add_induced_voltage(inductive_impedance_1)
    >>> induced_voltage_container.add_induced_voltage(inductive_impedance_2)
    >>>
    >>> equi_spaced_profiles = EquiSpacedProfiles()
    >>> equi_spaced_profiles.add_profile(profile1)
    >>>
    >>>
    >>> total_induced_voltage_NEW = InducedVoltageCompactWakeSolver(
    >>>     beam=beam,
    >>>     equi_spaced_profiles=equi_spaced_profiles,
    >>>     induced_voltage_container=induced_voltage_container,
    >>>     track_update_wake_kernel=False,
    >>> )
    >>>
    """

    def __init__(
        self,
        beam: Beam,
        equi_spaced_profiles: EquiSpacedProfiles | Profile,
        induced_voltage_container: InducedVoltageContainer,
        track_update_wake_kernel: bool,
    ):
        self._beam = beam
        if isinstance(equi_spaced_profiles, Profile):
            # iImport here to prevent cyclic dependency
            from . import EquiSpacedProfiles
            __profile = equi_spaced_profiles
            equi_spaced_profiles = EquiSpacedProfiles()
            equi_spaced_profiles.add_profile(__profile)
            del __profile

        self._equi_spaced_profiles = equi_spaced_profiles
        self._equi_spaced_profiles.lock()  # DONT ALLOW ANY CHANGE ANYMORE

        self._induced_voltage_container = induced_voltage_container
        self._induced_voltage_container.lock()  # DONT ALLOW ANY CHANGE ANYMORE

        self.track_update_wake_kernel = track_update_wake_kernel
        self.assume_periodic_wake = True
        self._compressed_wake_kernel = self._get_compressed_wake_kernel(
            self._equi_spaced_profiles
        )

        self._induced_voltage_amplitude: NumpyArray | CupyArray = None  # TODO
        # self._induced_voltage_time: NumpyArray | CupyArray # TODO
        self._reference_profile_idx = 0

    @property
    def induced_voltage(self):
        """
        Get the induced voltage for the reference equi-spaced profile.

        This is a preliminary implementation that returns the induced voltage
        for a single profile, currently selected via `self._reference_profile_idx`.

        Returns
        -------
        induced_voltage : NumpyArray or CupyArray
            The induced voltage corresponding to the reference profile.

        """
        return self.get_induced_voltage(profile_i=self._reference_profile_idx)


    def get_induced_voltage(self, profile_i: int) -> NumpyArray | CupyArray:
        """
        Return the induced voltage for a given equi-spaced profile.

        Parameters
        ----------
        profile_i : int
            Index of the equi-spaced profile

        Returns
        -------
        induced_voltage_ : NumpyArray or CupyArray
            The induced voltage computed as:
            `- (particle_charge * elementary_charge * beam_ratio * profile_wake)`.
            The type depends on the backend used (NumPy or CuPy).

        """
        from scipy.constants import elementary_charge as e

        profile = self._equi_spaced_profiles._profiles[profile_i]
        induced_voltage_ = -(
            self._beam.particle.charge * e * self._beam.ratio * profile.wake
        )
        return induced_voltage_

    @property
    def entire_induced_voltage(self) -> NumpyArray | CupyArray:
        """
        Compute the total induced voltage across all equi-spaced profiles.

        This method loops through each equi-spaced profile and computes the induced
        voltage using the formula:

        Between each induced voltage segment, a zero-valued buffer of the same length
        as each segment is inserted.

        Returns
        -------
        induced_voltage : NumpyArray or CupyArray
            Concatenated induced voltage from all profiles, with zero-padding between segments.
            The return type depends on the backend used (NumPy or CuPy).

        Notes
        -----
        This is useful for constructing a full wakefield voltage profile over
        a concatenated domain that includes spacing between individual bunch profiles.
        """
        from scipy.constants import elementary_charge as e

        for i, profile in enumerate(self._equi_spaced_profiles):
            induced_voltage_ = -(
                self._beam.particle.charge * e * self._beam.ratio * profile.wake
            )
            if i == 0:
                induced_voltage = induced_voltage_
            else:
                induced_voltage = np.concatenate(
                    (induced_voltage, np.zeros(len(induced_voltage_)), induced_voltage_)
                )
        return induced_voltage

    def to_gpu(self, recursive: bool = True):
        import cupy as cp

        self._compressed_wake_kernel = cp.array(self._compressed_wake_kernel)
        if recursive:
            self._equi_spaced_profiles.to_gpu(recursive=recursive)
        self._device: DeviceType = "GPU"

    def to_cpu(self, recursive=True):
        import cupy as cp

        self._compressed_wake_kernel = cp.asnumpy(self._compressed_wake_kernel)
        if recursive:
            self._equi_spaced_profiles.to_cpu(recursive=recursive)
        self._device: DeviceType = "CPU"

    def reprocess(self):
        """Recalculates the compressed wake for induced voltage calculation"""
        self._compressed_wake_kernel = self._get_compressed_wake_kernel(
            self._equi_spaced_profiles
        )


    def track(self):
        """Update induced voltage and apply kick function to beam for all profiles"""

        self.induced_voltage_sum()
        for profile_i, profile in enumerate(self._equi_spaced_profiles):
            bm.linear_interp_kick(
                dt=self._beam.dt,
                dE=self._beam.dE,
                voltage=self.get_induced_voltage(profile_i),
                bin_centers=profile.bin_centers,
                charge=self._beam.particle.charge,
                acceleration_kick=0.0,
            )
            # TODO activate del wake
            # del profile.wake  # NOQA remove attribute, so its not visible
            # from the outside.

    def track_ghosts_particles(self, ghost_beam: Beam):
        """Apply kick function to beam"""

        self.induced_voltage_sum()
        for profile_i, profile in enumerate(self._equi_spaced_profiles):
            bm.linear_interp_kick(
                dt=ghost_beam.dt,
                dE=ghost_beam.dE,
                voltage=self.get_induced_voltage(profile_i),
                bin_centers=profile.bin_centers,
                charge=self._beam.particle.charge,  # FIXME is this the correct
                # charge? Should it bhe the one of ghost_beam?
                acceleration_kick=0.0,
            )

    def induced_voltage_sum(self):
        """
        Compute the summed induced voltage for all equi-spaced profiles.

        This method:
        - Initializes wake arrays for each profile (if multiple exist),
        - Optionally updates the compressed wake kernel,
        - Selects between FFT-based and standard convolution,
        - Applies the convolution between the wake kernel and macroparticle distributions.

        The result is stored in the `wake` attribute of each profile for
        internal use (e.g. in tracking calculations).

        Notes
        -----
        - Uses `bm.fftconvolve` or `bm.convolve` depending on kernel size.
        - `self._compressed_wake_kernel` is updated if `self.track_update_wake_kernel` is True.
        """
        has_one_profile = self._equi_spaced_profiles.n_profiles == 1

        if not has_one_profile:
            for profile_j, profile_target in enumerate(self._equi_spaced_profiles):
                profile_target: Profile
                # attribute `wake` to be used in `track`. This is private to
                # `TotalInducedVoltageNew`
                profile_target.wake = bm.zeros(profile_target.n_slices)

        # Size of compressed wake per profile, defined in `_get_compressed_wake_kernel`
        step = 2 * self._equi_spaced_profiles.n_slices
        if self.assume_periodic_wake:
            offset = (self._equi_spaced_profiles.n_profiles - 1) * step
        else:
            offset = 0

        if self.track_update_wake_kernel:
            # This is potentially a non-performing operation,
            # but would be required if the profiles change their position
            self._compressed_wake_kernel = self._get_compressed_wake_kernel(
                self._equi_spaced_profiles
            )

        if len(self._compressed_wake_kernel) > 1024:
            convolve = bm.fftconvolve
        else:
            convolve = bm.convolve

        for profile_i, profile_source in enumerate(self._equi_spaced_profiles):
            profile_source: Profile

            compressed_wake = convolve(
                self._compressed_wake_kernel,
                profile_source.n_macroparticles[:],
                mode="same",
            )

            for profile_j, profile_target in enumerate(self._equi_spaced_profiles):
                profile_target: Profile
                # skip waves backwards in time, so that the first profile
                # affects all following profiles, but the last profile affects
                # only itself.
                if not self.assume_periodic_wake:
                    if profile_j < profile_i:
                        continue

                # Read/write the wakefield from the compressed wake.

                # Because the wake is `2 * number_of_bins` in the compressed wake
                # `start` and `stop` are shifted to include only the relevant
                # `1 * number_of_bins`.
                start = (profile_j - profile_i + 0) * step + step // 4 + offset
                stop = (profile_j - profile_i + 1) * step - step // 4 + offset
                size = stop - start

                # Set fake attribute `wake` to be used in `track`. This is
                # private to `TotalInducedVoltageNew`
                if has_one_profile:
                    profile_target.wake = compressed_wake[start:stop]
                else:
                    assert size == len(profile_target.wake), (
                        f"{size} !=" f" {len(profile_target.wake)}"
                    )
                    # print(f"{profile_j=} {start=} {stop}")
                    profile_target.wake += compressed_wake[start:stop]

    def _get_compressed_wake_kernel(
        self, profile_container: EquiSpacedProfiles
    ) -> NumpyArray | CupyArray:
        """Calculates the wake kernel at every profile in a common array"""
        concat_later = []

        first_profile_center = (
            profile_container._profiles[0].cut_left
            + profile_container._profiles[0].cut_right
        ) / 2
        t_offset = first_profile_center
        for profile_i, profile_dest in enumerate(profile_container):
            profile_dest: Profile
            width = profile_dest.cut_right - profile_dest.cut_left

            # make the kernel_width 2 * profile_width,
            # so that during convolution, the kernel will reach all profile
            # entries when at left and right end of profile
            t_start = profile_dest.cut_left - width / 2 - t_offset
            t_stop = profile_dest.cut_right + width / 2 - t_offset
            n_entries = 2 * self._equi_spaced_profiles.n_slices

            if t_start < 0:
                assert np.isclose(t_start, -t_stop), (
                    "Must be symmetric "
                    f"around 0, "
                    f"but {t_start=}, "
                    f"and {t_stop=}"
                )
            assert t_start < t_stop, f"{t_start=} {t_stop=}"
            msg = f"Bins must be  even, but {profile_dest.n_slices=}"
            assert profile_dest.n_slices % 2 == 0, msg

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
            if self.assume_periodic_wake:
                if profile_i > 0:
                    t_start, t_stop = -t_stop, -t_start  # mirror around 0
                    wake_kernel_at_single_profile = bm.zeros(
                        # Factor 2 because `start` and `stop` was increased by `width / 2`
                        n_entries,
                        dtype=precision.real_t,
                    )
                    # Sum all wakes (impedances) for a single profile target
                    for induced_voltage_object in self._induced_voltage_container:
                        wake_kernel_at_single_profile += (
                            induced_voltage_object.get_wake_kernel(
                                t_start=t_start,
                                t_stop=t_stop,
                                n=len(wake_kernel_at_single_profile),
                            )
                        )

                    concat_later.insert(0, wake_kernel_at_single_profile)

        for wake_i, wake in enumerate(concat_later[:]):
            # This script can be further developed to consider different
            # wake kernel sizes.  Must be considered in
            # `induced_voltage_sum` too.
            msg = (
                f"Not all wake kernels have the same size: "
                f"{[len(w) for w in concat_later]}"
            )
            assert len(wake) == (
                len(concat_later[0])
            ), f"{wake_i=} {len(wake)=} != {(len(concat_later[0]) )}"

        compressed_wake_kernel = bm.concatenate(concat_later, dtype=precision.real_t)

        return compressed_wake_kernel
