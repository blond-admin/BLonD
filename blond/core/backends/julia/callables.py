# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `JuliaCpuSpecials`, `JuliaGpuSpecials` and helper functions.

Both classes are thin wrappers: they validate the incoming arrays with
`assert` (intentionally stripped by ``python -O``), cast the scalars to
the types the Julia entry points declare, and hand the arrays over as
raw ``(pointer, length)`` pairs. Julia never owns the memory.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import Specials
from blond.core.backends.julia.julia_env import (
    julia_cuda_kernels,
    julia_kernels,
)
from blond.core.beam.flags import BeamFlags
from blond.generals.cupy_.no_cupy_import import is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from blond.generals.typing_ import AnyArray

FLOAT = np.float64
COMPLEX = np.complex128

# Julia device objects, created on first use: building one requires a live
# Julia session, which must not be started at import time. A dict is used
# so the lazy initialisation needs no `global` statement.
_device_cache: dict[str, Any] = {}

# Cache of uniformity verdicts for the `bin_centers` arrays passed to the
# dense path of `kick_interpolated`, copied from the CUDA backend:
# `bin_centers` is rebuilt only on profile reconfiguration, so checking it
# once per distinct array avoids a host<->device sync on every turn.
# `weakref.finalize` purges the entry when the array is deallocated, so a
# recycled `id()` can never read a stale verdict.
_MAX_UNIFORMITY_CACHE_SIZE = 64
_bin_centers_uniformity_cache: dict[int, bool] = {}

NON_UNIFORM_BIN_CENTERS_MESSAGE = (
    "bin_centers is not uniformly spaced (looks like "
    "a sparse/multi-island "
    "EquidistantMultiProfile.hist_x). Either pass "
    "this profile's sparse metadata (first_left_cut, "
    "left_cut_distance, cut_width, bins_per_profile, "
    "filling_pattern, bucket_index_to_memory_index), "
    "e.g. via `profile.sparse_kick_metadata`, or use "
    "EquidistantMultiProfile.profiles[i].hist_x for "
    "a single bucket."
)


def _is_uniformly_spaced(bin_centers: AnyArray) -> bool:
    """
    Check (and cache) whether `bin_centers` is uniformly spaced.

    Parameters
    ----------
    bin_centers
        Positions of the voltage samples, in [s]. NumPy or CuPy.

    Returns
    -------
    is_uniform
        True if the spacing is uniform within a relative tolerance.
    """
    key = id(bin_centers)
    cached = _bin_centers_uniformity_cache.get(key)
    if cached is not None:
        return cached

    if is_cupy_array(bin_centers):
        import cupy as array_module  # type: ignore
    else:
        array_module = np

    diffs = array_module.diff(bin_centers)
    is_uniform = bool(
        array_module.allclose(diffs, diffs[0], rtol=1e-6, atol=0.0)
    )

    if len(_bin_centers_uniformity_cache) >= _MAX_UNIFORMITY_CACHE_SIZE:
        _bin_centers_uniformity_cache.clear()
    _bin_centers_uniformity_cache[key] = is_uniform
    weakref.finalize(bin_centers, _bin_centers_uniformity_cache.pop, key, None)
    return is_uniform


class _JuliaSpecialsBase(Specials):
    """Shared wrappers around the `BLonDKernels` Julia entry points.

    Subclasses provide the three hooks `_kernels`, `_device` and `_ptr`
    (plus, on the GPU, `_pre_call`), which is all that separates the
    host from the device backend.
    """

    @classmethod
    def _kernels(cls) -> Any:
        """
        Return the `BLonDKernels` Julia module handle.

        Returns
        -------
        kernels
            Handle of the loaded `BLonDKernels` Julia module.
        """
        raise NotImplementedError

    @classmethod
    def _device(cls) -> Any:
        """
        Return the KernelAbstractions device the kernels run on.

        Returns
        -------
        device
            The Julia device object.
        """
        raise NotImplementedError

    @classmethod
    def _ptr(cls, array: AnyArray) -> int:
        """
        Return the address of the first element of `array`.

        Parameters
        ----------
        array
            A contiguous NumPy or CuPy array.

        Returns
        -------
        pointer
            The raw memory address as an integer.
        """
        raise NotImplementedError

    @classmethod
    def _pre_call(cls) -> None:
        """Synchronize the caller's stream before entering Julia."""

    @classmethod
    def _call(cls, entry_name: str, *args: Any) -> Any:
        """
        Call one `BLonDKernels` entry point on this backend's device.

        Parameters
        ----------
        entry_name
            Name of the Julia entry function, e.g. ``"drift_simple!"``.
        *args
            Arguments following the device, per the calling contract.

        Returns
        -------
        result
            Whatever the Julia entry point returns.
        """
        cls._pre_call()
        return getattr(cls._kernels(), entry_name)(cls._device(), *args)

    @classmethod
    def loss_box(  # NOQA: D102
        cls,
        e_max: float,
        e_min: float,
        t_min: float,
        t_max: float,
        dt: AnyArray,
        dE: AnyArray,
        flags: AnyArray,
    ) -> None:
        cls._assert_device(dt, dE, flags)
        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert flags.dtype == np.int32
        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous
        assert flags.flags.c_contiguous

        cls._call(
            "loss_box!",
            float(e_max),
            float(e_min),
            float(t_min),
            float(t_max),
            cls._ptr(dt),
            cls._ptr(dE),
            cls._ptr(flags),
            int(len(dE)),
            # The Julia entry declares the flag as `Int32`; a plain
            # Python `int` would arrive as `Int64` and miss the method.
            np.int32(BeamFlags.LOST.value),
        )

    @classmethod
    def kick_single_harmonic(  # NOQA: D102
        cls,
        dt: AnyArray,
        dE: AnyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: float,
        acceleration_kick: float,
    ) -> None:
        cls._assert_device(dt, dE)
        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous

        cls._call(
            "kick_single_harmonic!",
            cls._ptr(dt),
            cls._ptr(dE),
            int(len(dE)),
            float(voltage),
            float(omega_rf),
            float(phi_rf),
            float(charge),
            float(acceleration_kick),
        )

    @classmethod
    def kick_multi_harmonic(  # NOQA: D102
        cls,
        dt: AnyArray,
        dE: AnyArray,
        voltage: AnyArray,
        omega_rf: AnyArray,
        phi_rf: AnyArray,
        charge: float,
        n_rf: int,
        acceleration_kick: float,
    ) -> None:
        cls._assert_device(dt, dE, voltage, omega_rf, phi_rf)
        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert voltage.dtype == FLOAT
        assert omega_rf.dtype == FLOAT
        assert phi_rf.dtype == FLOAT
        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous
        assert voltage.flags.c_contiguous
        assert omega_rf.flags.c_contiguous
        assert phi_rf.flags.c_contiguous

        cls._call(
            "kick_multi_harmonic!",
            cls._ptr(dt),
            cls._ptr(dE),
            int(len(dE)),
            cls._ptr(voltage),
            cls._ptr(omega_rf),
            cls._ptr(phi_rf),
            int(n_rf),
            float(charge),
            float(acceleration_kick),
        )

    @classmethod
    def drift_simple(  # NOQA: D102
        cls,
        dt: AnyArray,
        dE: AnyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ) -> None:
        cls._assert_device(dt, dE)
        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous

        cls._call(
            "drift_simple!",
            cls._ptr(dt),
            cls._ptr(dE),
            int(len(dE)),
            float(T),
            float(eta_0),
            float(beta),
            float(energy),
        )

    @classmethod
    def drift_exact(  # NOQA: D102
        cls,
        dt: AnyArray,
        dE: AnyArray,
        T: float,
        alpha_0: float,
        higher_alpha: AnyArray,
        beta: float,
        energy: float,
    ) -> None:
        cls._assert_device(dt, dE, higher_alpha)
        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert higher_alpha.dtype == FLOAT
        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous
        assert higher_alpha.flags.c_contiguous

        cls._call(
            "drift_exact!",
            cls._ptr(dt),
            cls._ptr(dE),
            int(len(dE)),
            float(T),
            float(alpha_0),
            cls._ptr(higher_alpha),
            int(len(higher_alpha)),
            float(beta),
            float(energy),
        )

    @classmethod
    def sum_1d_array(cls, array: AnyArray) -> float:
        """
        Return the sum of a 1d array.

        Parameters
        ----------
        array
            Input array.

        Returns
        -------
        sum_1d_array
            Sum of the array.
        """
        cls._assert_device(array)
        assert array.dtype == FLOAT
        assert array.flags.c_contiguous

        return FLOAT(
            cls._call("sum_1d_array", cls._ptr(array), int(len(array)))
        )

    @classmethod
    def dot_product_1d_array(  # NOQA: D102
        cls, array_1: AnyArray, array_2: AnyArray
    ) -> float:
        cls._assert_device(array_1, array_2)
        assert array_1.dtype == FLOAT
        assert array_2.dtype == FLOAT
        assert array_1.flags.c_contiguous
        assert array_2.flags.c_contiguous
        assert len(array_1) == len(array_2)

        return FLOAT(
            cls._call(
                "dot_product_1d_array",
                cls._ptr(array_1),
                cls._ptr(array_2),
                int(len(array_1)),
            )
        )

    @classmethod
    def histogram(  # NOQA: D102
        cls,
        array_read: AnyArray,
        array_write: AnyArray,
        start: float,
        stop: float,
    ) -> None:
        cls._assert_device(array_read, array_write)
        assert array_read.dtype == FLOAT
        assert array_write.dtype == FLOAT
        assert array_read.flags.c_contiguous
        assert array_write.flags.c_contiguous

        cls._call(
            "histogram!",
            cls._ptr(array_read),
            int(len(array_read)),
            cls._ptr(array_write),
            int(len(array_write)),
            float(start),
            float(stop),
        )

    @classmethod
    def beam_phase(  # NOQA: D102
        cls,
        hist_x: AnyArray,
        hist_y: AnyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> float:
        cls._assert_device(hist_x, hist_y)
        assert hist_x.dtype == FLOAT
        assert hist_y.dtype == FLOAT
        assert hist_x.flags.c_contiguous
        assert hist_y.flags.c_contiguous
        assert len(hist_x) == len(hist_y)

        return FLOAT(
            cls._call(
                "beam_phase",
                cls._ptr(hist_x),
                cls._ptr(hist_y),
                int(len(hist_x)),
                float(alpha),
                float(omega_rf),
                float(phi_rf),
                float(bin_size),
            )
        )

    @classmethod
    def kick_interpolated(  # NOQA: D102
        cls,
        dt: AnyArray,
        dE: AnyArray,
        voltage: AnyArray,
        bin_centers: AnyArray,
        charge: float,
        acceleration_kick: float,
        first_left_cut: float | None = None,
        left_cut_distance: float | None = None,
        cut_width: float | None = None,
        bins_per_profile: int | None = None,
        filling_pattern: AnyArray | None = None,
        bucket_index_to_memory_index: AnyArray | None = None,
    ) -> None:
        cls._assert_device(dt, dE, voltage, bin_centers)
        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert voltage.dtype == FLOAT
        assert bin_centers.dtype == FLOAT
        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous
        assert voltage.flags.c_contiguous
        assert bin_centers.flags.c_contiguous

        n_slices = int(len(bin_centers))

        if first_left_cut is None:
            if n_slices >= 2 and not _is_uniformly_spaced(  # noqa: PLR2004
                bin_centers
            ):
                raise ValueError(NON_UNIFORM_BIN_CENTERS_MESSAGE)

            cls._call(
                "kick_interpolated_dense!",
                cls._ptr(dt),
                cls._ptr(dE),
                int(len(dt)),
                cls._ptr(voltage),
                cls._ptr(bin_centers),
                n_slices,
                float(charge),
                float(acceleration_kick),
            )
            return

        cls._assert_device(filling_pattern, bucket_index_to_memory_index)
        assert filling_pattern.dtype == np.bool_
        assert bucket_index_to_memory_index.dtype == np.int32
        assert filling_pattern.flags.c_contiguous
        assert bucket_index_to_memory_index.flags.c_contiguous

        cls._call(
            "kick_interpolated_sparse!",
            cls._ptr(dt),
            cls._ptr(dE),
            int(len(dt)),
            cls._ptr(voltage),
            cls._ptr(bin_centers),
            n_slices,
            float(charge),
            float(acceleration_kick),
            float(first_left_cut),
            float(left_cut_distance),
            float(cut_width),
            int(bins_per_profile),
            cls._ptr(filling_pattern),
            int(len(filling_pattern)),
            cls._ptr(bucket_index_to_memory_index),
        )

    @classmethod
    def histogram_sparse(  # NOQA: D102
        cls,
        x: AnyArray,
        out: AnyArray,
        first_left_cut: float,
        left_cut_distance: float,
        cut_width: float,
        bins_per_profile: int,
        n_active_profiles: int,
        filling_pattern: AnyArray,
        bucket_index_to_memory_index: AnyArray,
    ) -> None:
        cls._assert_device(
            x, out, filling_pattern, bucket_index_to_memory_index
        )
        assert x.dtype == FLOAT
        assert out.dtype == FLOAT
        assert filling_pattern.dtype == np.bool_
        assert bucket_index_to_memory_index.dtype == np.int32
        assert x.flags.c_contiguous
        assert out.flags.c_contiguous
        assert filling_pattern.flags.c_contiguous
        assert bucket_index_to_memory_index.flags.c_contiguous

        cls._call(
            "histogram_sparse!",
            cls._ptr(x),
            int(len(x)),
            cls._ptr(out),
            int(len(out)),
            float(first_left_cut),
            float(left_cut_distance),
            float(cut_width),
            int(bins_per_profile),
            cls._ptr(filling_pattern),
            int(len(filling_pattern)),
            cls._ptr(bucket_index_to_memory_index),
        )

    @classmethod
    def move_flagged_elements_to_end(  # NOQA: D102
        cls,
        flag: int,
        flags: AnyArray,
        dt: AnyArray,
        dE: AnyArray,
        ids: AnyArray,
    ) -> int:
        cls._assert_device(flags, dt, dE, ids)
        assert flags.dtype == np.int32
        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert ids.dtype == np.int32
        assert flags.flags.c_contiguous
        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous
        assert ids.flags.c_contiguous

        return int(
            cls._call(
                "move_flagged_elements_to_end!",
                np.int32(flag),
                cls._ptr(flags),
                cls._ptr(dt),
                cls._ptr(dE),
                cls._ptr(ids),
                int(len(flags)),
            )
        )

    @classmethod
    def wake_from_pole_residue(  # NOQA: D102
        cls,
        # read
        profile: AnyArray,
        profile_dts: AnyArray,
        poles: AnyArray,
        residues: AnyArray,
        is_counterrotating_beam: bool,
        counterrotating_pole_signs: AnyArray,
        update_on_bin: AnyArray,
        factor: float,
        # write
        states: AnyArray,
        voltage: AnyArray,
        voltage_threaded: AnyArray,
    ) -> None:
        cls._assert_device(
            profile,
            profile_dts,
            poles,
            residues,
            counterrotating_pole_signs,
            update_on_bin,
            states,
            voltage,
        )
        assert profile.dtype == FLOAT
        assert profile_dts.dtype == FLOAT
        assert voltage.dtype == FLOAT
        assert counterrotating_pole_signs.dtype == FLOAT
        assert poles.dtype == COMPLEX
        assert residues.dtype == COMPLEX
        assert states.dtype == COMPLEX
        assert update_on_bin.dtype == np.int32
        assert profile.flags.c_contiguous
        assert profile_dts.flags.c_contiguous
        assert poles.flags.c_contiguous
        assert residues.flags.c_contiguous
        assert counterrotating_pole_signs.flags.c_contiguous
        assert update_on_bin.flags.c_contiguous
        assert states.flags.c_contiguous
        assert voltage.flags.c_contiguous

        n_bins = int(profile.shape[0])
        n_poles = int(poles.shape[0])

        # `states` carries one state per pole plus `t_start` in the last
        # entry, see `Specials.wake_from_pole_residue`.
        assert states.shape[0] == n_poles + 1
        assert residues.shape[0] == n_poles
        assert counterrotating_pole_signs.shape[0] == n_poles
        assert voltage.shape[0] == n_bins

        cls._call(
            "wake_from_pole_residue!",
            cls._ptr(profile),
            n_bins,
            cls._ptr(profile_dts),
            int(profile_dts.shape[0]),
            cls._ptr(poles),
            cls._ptr(residues),
            n_poles,
            bool(is_counterrotating_beam),
            cls._ptr(counterrotating_pole_signs),
            cls._ptr(update_on_bin),
            int(update_on_bin.shape[0]),
            float(factor),
            cls._ptr(states),
            cls._ptr(voltage),
        )

    @classmethod
    def apply_synchrotron_radiation_and_quantum_excitation_energy_kick(  # NOQA: D102
        cls,
        beam_dE: AnyArray,
        energy_lost: float,
        longitudinal_damping_time: float,
        natural_energy_spread: float,
        total_energy: float,
        disable_quantum_excitation: bool = False,
    ) -> None:
        cls._assert_device(beam_dE)
        assert beam_dE.dtype == FLOAT
        assert beam_dE.flags.c_contiguous

        cls._call(
            "apply_synchrotron_radiation!",
            cls._ptr(beam_dE),
            int(len(beam_dE)),
            float(energy_lost),
            float(longitudinal_damping_time),
            float(natural_energy_spread),
            float(total_energy),
            bool(disable_quantum_excitation),
        )

    @classmethod
    def _assert_device(cls, *arrays: AnyArray) -> None:
        """
        Validate that every array lives on this backend's device.

        Parameters
        ----------
        *arrays
            Arrays handed to a kernel wrapper.
        """
        raise NotImplementedError


class JuliaCpuSpecials(_JuliaSpecialsBase):
    """Julia kernels on the KernelAbstractions `CPU()` device."""

    @classmethod
    def _kernels(cls) -> Any:  # NOQA: D102
        return julia_kernels()

    @classmethod
    def _device(cls) -> Any:  # NOQA: D102
        if "host" not in _device_cache:
            _device_cache["host"] = cls._kernels().host_device()
        return _device_cache["host"]

    @classmethod
    def _ptr(cls, array: AnyArray) -> int:  # NOQA: D102
        return int(array.ctypes.data)

    @classmethod
    def _assert_device(cls, *arrays: AnyArray) -> None:  # NOQA: D102
        for array in arrays:
            assert isinstance(array, np.ndarray), (
                f"Requires Numpy array, but got {type(array)}."
            )

    @classmethod
    def get_max_threads(cls) -> int:
        """
        Return the max number of threads this backend's kernels may use.

        Returns
        -------
        max_threads
            Number of Julia threads available to the `CPU()` device.
        """
        return int(cls._kernels().max_threads(cls._device()))

    @classmethod
    def music_track(  # NOQA: D102 inherited from `Specials.music_track`
        cls,
        beam_dt: AnyArray,
        beam_dE: AnyArray,
        induced_voltage: AnyArray,
        parameter_array: AnyArray,
        alpha: float,
        omega_bar: float,
        const: float,
        coeff1: float,
        coeff2: float,
        coeff3: float,
        coeff4: float,
        time_since_last_track: float,
        multiturn: bool,
    ) -> None:
        cls._assert_device(beam_dt, beam_dE, induced_voltage, parameter_array)
        assert beam_dt.dtype == FLOAT
        assert beam_dE.dtype == FLOAT
        assert induced_voltage.dtype == FLOAT
        assert parameter_array.dtype == FLOAT
        assert beam_dt.flags.c_contiguous
        assert beam_dE.flags.c_contiguous
        assert induced_voltage.flags.c_contiguous
        assert parameter_array.flags.c_contiguous

        cls._call(
            "music_track!",
            cls._ptr(beam_dt),
            cls._ptr(beam_dE),
            cls._ptr(induced_voltage),
            cls._ptr(parameter_array),
            int(len(beam_dt)),
            float(alpha),
            float(omega_bar),
            float(const),
            float(coeff1),
            float(coeff2),
            float(coeff3),
            float(coeff4),
            float(time_since_last_track),
            bool(multiturn),
        )


class JuliaGpuSpecials(_JuliaSpecialsBase):
    """Julia kernels on the KernelAbstractions `CUDABackend()` device."""

    @classmethod
    def _kernels(cls) -> Any:  # NOQA: D102
        return julia_cuda_kernels()

    @classmethod
    def _device(cls) -> Any:  # NOQA: D102
        if "cuda" not in _device_cache:
            _device_cache["cuda"] = cls._kernels().cuda_device()
        return _device_cache["cuda"]

    @classmethod
    def _ptr(cls, array: AnyArray) -> int:  # NOQA: D102
        return int(array.data.ptr)

    @classmethod
    def _assert_device(cls, *arrays: AnyArray) -> None:  # NOQA: D102
        for array in arrays:
            assert array.device != "cpu", (
                f"Requires Cupy array, but got {type(array)}."
            )

    @classmethod
    def _pre_call(cls) -> None:
        """Flush CuPy's stream so Julia sees all pending writes."""
        import cupy as cp  # type: ignore

        cp.cuda.get_current_stream().synchronize()

    @classmethod
    def get_max_threads(cls) -> int:
        """
        Return the max number of threads this backend's kernels may use.

        Returns
        -------
        max_threads
            Always 1: the GPU kernels use no per-thread scratch buffers.
        """
        return 1

    @classmethod
    def music_track(  # NOQA: D102 inherited from `Specials.music_track`
        cls,
        beam_dt: AnyArray,
        beam_dE: AnyArray,
        induced_voltage: AnyArray,
        parameter_array: AnyArray,
        alpha: float,
        omega_bar: float,
        const: float,
        coeff1: float,
        coeff2: float,
        coeff3: float,
        coeff4: float,
        time_since_last_track: float,
        multiturn: bool,
    ) -> None:
        raise NotImplementedError(
            "MuSiC is a sequential algorithm and is not implemented on "
            "the `julia_gpu` backend."
        )
