# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `CppSpecials` and helper functions."""

from __future__ import annotations

import ctypes as ct
import os
import sys
import weakref
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import Specials, backend
from blond.core.backends.cpp.compile import add_dll_directory_once
from blond.core.backends.cpp.compiled_dir_handler import cpp_compiled_dir
from blond.generals.compiled_cache import mark_used

if TYPE_CHECKING:  # pragma: no cover
    from ctypes import CDLL

    from numpy.typing import NDArray as NumpyArray

# Upper bound on the number of cached array-pointer entries; the cache is
# cleared wholesale once it grows past this to keep memory bounded.
_PTR_CACHE_MAX_SIZE = 4096


def c_real(
    scalar: float, floattype: type[np.float64]
) -> ct.c_float | ct.c_double:
    """Convert input to default precision."""
    if floattype == np.float32:
        raise TypeError("32-bit float and 64-bit complex have been removed.")
    elif floattype == np.float64:
        return ct.c_double(scalar)
    else:
        raise ValueError(floattype)


def c_real_t(
    floattype: type[np.float64],
) -> type[ct.c_float | ct.c_double]:
    """Get default precision."""
    if floattype == np.float32:
        raise TypeError("32-bit float and 64-bit complex have been removed.")
    elif floattype == np.float64:
        return ct.c_double
    else:
        raise ValueError(floattype)


def reload_cpp_backend(  # NOQA: PLR0915
    floattype: type[np.float64], parallel: bool = True
) -> type[Specials]:
    """
    Load and link the according C++ backend.

    Parameters
    ----------
    floattype
        Float type to compile the backend for.
        32 or 64 bit.
    parallel
        If True, loads the parallel OMP computing backend.

    Returns
    -------
    CppSpecials
        The `CppSpecials` class.

    """
    parallel_suffix = "" if parallel else "_noOMP"

    def load_libblond(precision: str = "double") -> CDLL:
        """
        Locates and initializes the blond compiled library.

        Parameters
        ----------
        precision
            The floating point precision of the calculations.
            Can only be 'double'.
            Default is  "double".
        """
        if precision != "double":
            raise TypeError(
                "Only double precision (64 Bit) callables are "
                f"available, requested precision is {precision}"
            )

        libblond_path_ = os.environ.get("LIBBLOND", None)
        if libblond_path_ is not None and not libblond_path_.strip():
            raise ValueError(
                "LIBBLOND is set but empty; unset it to use the default "
                "library or set it to a valid path."
            )

        folder = os.path.dirname(os.path.abspath(__file__))

        # Same toolchain/CPU-aware directory the compiler writes to.
        basepath = cpp_compiled_dir(folder)
        if "posix" in os.name:
            if libblond_path_ is not None:
                libblond_path = os.path.abspath(libblond_path_)
            else:
                libblond_path = os.path.join(
                    basepath, f"libblond_{precision}{parallel_suffix}.so"
                )
            _LIBBLOND = ct.CDLL(str(libblond_path))
        elif "win" in sys.platform:
            if libblond_path_ is not None:
                libblond_path = os.path.abspath(libblond_path_)
            else:
                libblond_path = os.path.join(
                    basepath, f"libblond_{precision}{parallel_suffix}.dll"
                )

            # Cached: repeated backend loads must not keep appending
            # to the DLL search path (see add_dll_directory_once).
            add_dll_directory_once(os.path.dirname(libblond_path))
            _LIBBLOND = ct.CDLL(str(libblond_path), winmode=0)
        else:
            raise ValueError(
                f"Supporting 'win' and 'posix', not {sys.platform}."
            )

        if libblond_path_ is None:
            # Refresh the LRU stamp on the hashed cache dir we loaded from
            # (skipped when an explicit LIBBLOND path bypassed it).
            mark_used(basepath)

        return _LIBBLOND

    # Validate up front; only ``double``/float64 is supported. These raise
    # TypeError, which is unrelated to the load failures handled below.
    if floattype == np.float32:
        raise TypeError("32-bit float and 64-bit complex have been removed.")
    elif floattype != np.float64:
        raise TypeError(floattype)

    # FileNotFoundError is an OSError subclass; catching OSError covers both.
    try:
        _LIBBLOND = load_libblond(precision="double")
    except OSError:
        from blond.core.backends.cpp.compile import compile_cpp_library

        print(
            "C++ backend was not found.. Trying to compile parallel backend."
        )
        compile_cpp_library()
        try:
            _LIBBLOND = load_libblond(precision="double")
        except OSError as exc:
            raise OSError(
                "`load_libblond` failed. Has the backend been compiled?\n"
                f"{__file__.replace('callables.py', 'compile.py')}:1"  # :1 to
                # make PyCharm automatically link the correct file
            ) from exc

    # Building a ctypes pointer from a numpy array (`x.ctypes.data_as(...)`)
    # constructs a fresh numpy `_ctypes` helper and a new ctypes object on every
    # call -- ~2 us each, which dominates the cost of small/medium kernels once
    # you pass several arrays. In a simulation the same array objects are reused
    # every turn (buffers updated in place; BLonD never resizes kernel arrays),
    # so cache the `c_void_p` keyed by array identity.
    #
    # Safety: a numpy array never relocates its data buffer while alive, so a
    # cached address stays valid for that array's lifetime. We store a `weakref`
    # (not a strong ref, so caching never keeps an array -- or a beam -- alive)
    # and re-validate `ref() is x` on every hit, which makes a recycled `id()`
    # after garbage collection a cache miss rather than a stale pointer.
    _pointer_cache: dict[int, tuple] = {}

    def _get_pointer(x: NumpyArray) -> ct.c_void_p:
        """
        Return a cached ``c_void_p`` to ``x``'s data buffer.

        Parameters
        ----------
        x
            Array whose data buffer address is needed. The cache holds only a
            weak reference to it, so the caller must keep ``x`` alive while
            using the returned pointer.

        Returns
        -------
        ct.c_void_p
            Pointer to ``x``'s data buffer, reused from the cache when ``x``
            has been seen before.
        """
        _id = id(x)
        entry = _pointer_cache.get(_id)
        if entry is not None and entry[0]() is x:
            pointer = entry[1]
        else:
            pointer = ct.c_void_p(
                x.ctypes.data
            )  # int address -> pointer; does not pin x
            if len(_pointer_cache) >= _PTR_CACHE_MAX_SIZE:  # bound cache size
                _pointer_cache.clear()
            _pointer_cache[_id] = (weakref.ref(x), pointer)
        return pointer

    def _get_len(x: NumpyArray) -> ct.c_int:
        """
        Return the length of ``x`` as a ``c_int``.

        Parameters
        ----------
        x
            Array whose length is passed to the C++ kernel.

        Returns
        -------
        ct.c_int
            ``len(x)`` wrapped as a ctypes ``c_int``.
        """
        return ct.c_int(len(x))

    def _is_valid(*pairs: tuple[NumpyArray, type]) -> bool:
        """
        Assert each ``(array, dtype)`` has that dtype and is C-contiguous.

        The C++ kernels are compiled for a fixed dtype and read raw,
        contiguous buffers, so a wrong dtype or a non-contiguous array would
        crash or silently corrupt results. Uses ``assert`` on purpose so
        ``python -O`` strips the checks from the hot path.

        Parameters
        ----------
        *pairs
            One or more ``(array, dtype)`` tuples; each ``array`` must have
            the given ``dtype`` and be C-contiguous.
        """
        valid = True
        for arr, dtype in pairs:
            if (arr.dtype != dtype) or (not arr.flags.c_contiguous):
                valid = False
                break
        return valid

    _LIBBLOND.beam_phase.restype = c_real_t(floattype)
    _LIBBLOND.sum_1d_array.restype = c_real_t(floattype)
    _LIBBLOND.dot_product_1d_array.restype = c_real_t(floattype)
    _LIBBLOND.blond_omp_get_max_threads.restype = ct.c_int
    _LIBBLOND.blond_omp_get_max_threads.argtypes = []

    # The array pointers are cached by the shared `_get_pointer` above; like every
    # other callable here we pass already-typed ctypes objects, so no `argtypes`
    # are needed (measured: setting them adds ~0.5 us of redundant per-arg
    # type-checking).
    # TODO: Refactor this file to float64 only.
    complextype = np.complex64 if floattype == np.float32 else np.complex128

    class CppSpecials(Specials):
        @staticmethod
        def get_max_threads() -> int:
            """
            Return the max number of threads this backend's kernels may use.

            Returns
            -------
            max_threads
                Maximum number of threads this backend's kernels may use.
            """
            return int(_LIBBLOND.blond_omp_get_max_threads())

        @staticmethod
        def beam_phase(
            hist_x: NumpyArray,
            hist_y: NumpyArray,
            alpha: float,
            omega_rf: float,
            phi_rf: float,
            bin_size: float,
        ) -> float:
            assert _is_valid((hist_x, floattype), (hist_y, floattype))

            # Cast Python floats to backend floattype
            alpha = floattype(alpha)
            omega_rf = floattype(omega_rf)
            phi_rf = floattype(phi_rf)
            bin_size = floattype(bin_size)

            # requires setting of _LIBBLOND.beam_phase.restype = c_real_t(floattype) in
            # reload function
            return floattype(
                _LIBBLOND.beam_phase(
                    hist_x.ctypes.data_as(ct.c_void_p),  # bin_centers
                    hist_y.ctypes.data_as(ct.c_void_p),  # profile
                    c_real(alpha, floattype),  # alpha
                    c_real(omega_rf, floattype),  # omega_rf
                    c_real(phi_rf, floattype),  # phi_rf
                    c_real(bin_size, floattype),  # bin_size
                    ct.c_int(len(hist_x)),  # n_bins
                )
            )

        @staticmethod
        def histogram(
            array_read: NumpyArray,
            array_write: NumpyArray,
            start: float,
            stop: float,
        ) -> None:
            assert _is_valid((array_read, floattype), (array_write, floattype))

            # Cast Python floats to backend floattype
            start = floattype(start)
            stop = floattype(stop)

            _LIBBLOND.histogram(
                array_read.ctypes.data_as(ct.c_void_p),
                array_write.ctypes.data_as(ct.c_void_p),
                c_real(start, floattype),
                c_real(stop, floattype),
                ct.c_int(len(array_write)),
                ct.c_int(len(array_read)),
            )

        @staticmethod
        def kick_interpolated(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: NumpyArray,
            bin_centers: NumpyArray,
            charge: float,
            acceleration_kick: float,
            first_left_cut: float | None = None,
            left_cut_distance: float | None = None,
            cut_width: float | None = None,
            bins_per_profile: int | None = None,
            filling_pattern: NumpyArray | None = None,
            bucket_index_to_memory_index: NumpyArray | None = None,
        ) -> None:
            assert _is_valid(
                (dt, floattype),
                (dE, floattype),
                (voltage, floattype),
                (bin_centers, floattype),
            )

            charge = floattype(charge)
            acceleration_kick = floattype(acceleration_kick)

            if first_left_cut is None:
                n_slices = len(bin_centers)
                if n_slices >= 2:  # noqa: PLR2004
                    diffs = np.diff(bin_centers)
                    if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=0.0):
                        raise ValueError(
                            "bin_centers is not uniformly spaced (looks "
                            "like a sparse/multi-island "
                            "EquidistantMultiProfile.hist_x). Either "
                            "pass this profile's sparse metadata "
                            "(first_left_cut, left_cut_distance, "
                            "cut_width, bins_per_profile, "
                            "filling_pattern, "
                            "bucket_index_to_memory_index), e.g. via "
                            "`profile.sparse_kick_metadata`, or use "
                            "EquidistantMultiProfile.profiles[i].hist_x "
                            "for a single bucket."
                        )
                _LIBBLOND.linear_interp_kick(
                    dt.ctypes.data_as(ct.c_void_p),
                    dE.ctypes.data_as(ct.c_void_p),
                    voltage.ctypes.data_as(ct.c_void_p),
                    bin_centers.ctypes.data_as(ct.c_void_p),
                    c_real(charge, floattype),
                    ct.c_int(len(bin_centers)),
                    ct.c_int(len(dt)),
                    c_real(acceleration_kick, floattype),
                )
                return

            assert filling_pattern.dtype == np.bool_
            assert bucket_index_to_memory_index.dtype == np.int32
            assert filling_pattern.flags.c_contiguous
            assert bucket_index_to_memory_index.flags.c_contiguous

            _LIBBLOND.linear_interp_kick_sparse(
                dt.ctypes.data_as(ct.c_void_p),
                dE.ctypes.data_as(ct.c_void_p),
                voltage.ctypes.data_as(ct.c_void_p),
                bin_centers.ctypes.data_as(ct.c_void_p),
                c_real(charge, floattype),
                ct.c_int(len(bin_centers)),
                ct.c_int(len(dt)),
                c_real(acceleration_kick, floattype),
                c_real(floattype(first_left_cut), floattype),
                c_real(floattype(left_cut_distance), floattype),
                c_real(floattype(cut_width), floattype),
                ct.c_int(bins_per_profile),
                ct.c_int(len(filling_pattern)),
                filling_pattern.ctypes.data_as(ct.c_void_p),
                bucket_index_to_memory_index.ctypes.data_as(ct.c_void_p),
            )

        @staticmethod
        def loss_box(
            e_max: float,
            e_min: float,
            t_min: float,
            t_max: float,
            dt: NumpyArray,
            dE: NumpyArray,
            flags: NumpyArray,
        ) -> None:
            assert _is_valid(
                (dt, floattype),
                (dE, floattype),
                (flags, np.int32),
            )

            _LIBBLOND.loss_box(
                c_real(e_max, floattype),
                c_real(e_min, floattype),
                c_real(t_min, floattype),
                c_real(t_max, floattype),
                _get_pointer(dt),
                _get_pointer(dE),
                _get_pointer(flags),
                _get_len(dt),
            )

        @staticmethod
        def kick_single_harmonic(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: float,
            omega_rf: float,
            phi_rf: float,
            charge: float,
            acceleration_kick: float,
        ) -> None:
            assert _is_valid((dt, floattype), (dE, floattype))

            # Cast Python floats to backend floattype
            charge = floattype(charge)
            voltage = floattype(voltage)
            omega_rf = floattype(omega_rf)
            phi_rf = floattype(phi_rf)
            acceleration_kick = floattype(acceleration_kick)

            _LIBBLOND.kick_single_harmonic(
                dt.ctypes.data_as(ct.c_void_p),
                dE.ctypes.data_as(ct.c_void_p),
                c_real(charge, floattype),
                c_real(voltage, floattype),
                c_real(omega_rf, floattype),
                c_real(phi_rf, floattype),
                ct.c_int(len(dt)),
                c_real(acceleration_kick, floattype),
            )

        @staticmethod
        def kick_multi_harmonic(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: NumpyArray,
            omega_rf: NumpyArray,
            phi_rf: NumpyArray,
            charge: float,
            n_rf: int,
            acceleration_kick: float,
        ) -> None:
            assert _is_valid(
                (dt, floattype),
                (dE, floattype),
                (voltage, floattype),
                (omega_rf, floattype),
                (phi_rf, floattype),
            )

            # Cast Python floats to backend floattype
            charge = floattype(charge)
            acceleration_kick = floattype(acceleration_kick)

            _LIBBLOND.kick_multi_harmonic(
                _get_pointer(dt),
                _get_pointer(dE),
                ct.c_int(n_rf),
                c_real(charge, floattype),
                _get_pointer(voltage),
                _get_pointer(omega_rf),
                _get_pointer(phi_rf),
                _get_len(dt),
                c_real(acceleration_kick, floattype),
            )

        @staticmethod
        def sum_1d_array(array: NumpyArray) -> float:
            assert _is_valid((array, floattype))
            # requires setting of _LIBBLOND.sum_1d_array.restype = c_real_t(floattype) in
            # reload function
            return floattype(
                _LIBBLOND.sum_1d_array(_get_pointer(array), _get_len(array))
            )

        @staticmethod
        def dot_product_1d_array(
            array_1: NumpyArray,
            array_2: NumpyArray,
        ) -> float:
            assert _is_valid((array_1, floattype), (array_2, floattype))
            assert len(array_1) == len(array_2)

            # requires setting of _LIBBLOND.dot_product_1d_array.restype = c_real_t(floattype) in
            # reload function
            return floattype(
                _LIBBLOND.dot_product_1d_array(
                    _get_pointer(array_1),
                    _get_pointer(array_2),
                    ct.c_int(len(array_2)),
                )
            )

        @staticmethod
        def drift_simple(
            dt: NumpyArray,
            dE: NumpyArray,
            T: float,
            eta_0: float,
            beta: float,
            energy: float,
        ) -> None:
            assert _is_valid((dt, floattype), (dE, floattype))

            # Cast Python floats to backend floattype
            T = floattype(T)
            eta_0 = floattype(eta_0)
            beta = floattype(beta)
            energy = floattype(energy)

            _LIBBLOND.drift_simple(
                _get_pointer(dt),
                _get_pointer(dE),
                c_real(T, floattype),
                c_real(eta_0, floattype),
                c_real(beta, floattype),
                c_real(energy, floattype),
                _get_len(dt),
            )

        @staticmethod
        def drift_exact(
            dt: NumpyArray,
            dE: NumpyArray,
            T: float,
            alpha_0: float,
            higher_alpha: NumpyArray,
            beta: float,
            energy: float,
        ):
            assert _is_valid(
                (dt, floattype),
                (dE, floattype),
                (higher_alpha, floattype),
            )

            # Cast Python floats to backend floattype
            T = floattype(T)
            beta = floattype(beta)
            energy = floattype(energy)
            alpha_0 = floattype(alpha_0)

            _LIBBLOND.drift_exact(
                _get_pointer(dt),  # real_t *__restrict__ beam_dt
                _get_pointer(dE),  # const real_t *__restrict__ beam_dE
                c_real(T, floattype),  # const real_t T
                c_real(alpha_0, floattype),  # const real_t alpha_zero
                _get_pointer(
                    higher_alpha
                ),  # const real_t *__restrict__ higher_alpha
                _get_len(higher_alpha),  # const int n_alpha
                c_real(beta, floattype),  # const real_t beta
                c_real(energy, floattype),  # const real_t energy
                _get_len(dt),  # const int n_macroparticles
            )

        @staticmethod
        def apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
            beam_dE: NumpyArray,
            energy_lost: float,
            longitudinal_damping_time: float,
            natural_energy_spread: float,
            total_energy: float,
            disable_quantum_excitation: bool = False,
        ) -> None:
            assert beam_dE.dtype == floattype
            assert beam_dE.flags.c_contiguous

            damping_factor = floattype(1.0 - 2.0 / longitudinal_damping_time)
            energy_lost_typed = floattype(energy_lost)

            if disable_quantum_excitation:
                _LIBBLOND.apply_synchrotron_radiation_no_excitation(
                    _get_pointer(beam_dE),
                    c_real(damping_factor, floattype),
                    c_real(energy_lost_typed, floattype),
                    _get_len(beam_dE),
                )
            else:
                noise_scale = floattype(
                    2.0
                    * natural_energy_spread
                    / np.sqrt(longitudinal_damping_time)
                    * total_energy
                )
                _LIBBLOND.apply_synchrotron_radiation_and_quantum_excitation(
                    _get_pointer(beam_dE),
                    c_real(damping_factor, floattype),
                    c_real(energy_lost_typed, floattype),
                    c_real(noise_scale, floattype),
                    _get_len(beam_dE),
                )

        @staticmethod
        def move_flagged_elements_to_end(
            flag: int,
            flags: NumpyArray,  # also purged
            dt: NumpyArray,
            dE: NumpyArray,
            ids: NumpyArray,
        ):
            assert _is_valid(
                (dt, floattype),
                (dE, floattype),
                (flags, np.int32),
                (ids, np.int32),
            )

            n_new = _LIBBLOND.move_flagged_elements_to_end(
                ct.c_int32(np.int32(flag)),
                flags.ctypes.data_as(ct.c_void_p),
                dt.ctypes.data_as(ct.c_void_p),
                dE.ctypes.data_as(ct.c_void_p),
                ids.ctypes.data_as(ct.c_void_p),
                ct.c_int32(len(dt)),  # n_macroparticles
            )
            n_new = int(n_new)
            return n_new

        @staticmethod
        def histogram_sparse(
            x: NumpyArray,
            out: NumpyArray,
            first_left_cut: float,
            left_cut_distance: float,
            cut_width: float,
            bins_per_profile: int,
            n_active_profiles: int,
            filling_pattern: NumpyArray,
            bucket_index_to_memory_index: NumpyArray,
        ) -> None:
            """
            Sparse histogram with strided memory layout (gaps between profiles).

            Parameters
            ----------
            x
                An array, e.g., the particle ``dt`` values.
            out
                Output histogram ``(n_filled_buckets * bins_per_profile)``.
            first_left_cut
                Start of the first histogram.
            left_cut_distance
                Distance between the start of each histogram.
            cut_width
                Distance between left and right edge of the histogram.
            bins_per_profile
                Number of bins per bucket.
            n_active_profiles
                Number of non-empty buckets.
            filling_pattern
                Filling pattern as a boolean array
                where ``True`` means filled bucket.
            bucket_index_to_memory_index
                Maps bucket index to memory index.
                For a ``filling_pattern = [1, 0, 0, 1]``
                ``bucket_index_to_memory_index = [0, 0, 0, 8]`` with
                ``bins_per_profile = 8``.
                Use `_gen_array_bucket_index_to_memory_index` to generate this.
            """
            assert _is_valid(
                (x, floattype),
                (out, floattype),
                (filling_pattern, np.bool),
                (bucket_index_to_memory_index, np.int32),
            )

            _LIBBLOND.histogram_sparse(
                _get_pointer(x),  # input
                _get_pointer(out),  # output
                c_real(first_left_cut, floattype),  # first_left_cut
                c_real(left_cut_distance, floattype),  # left_cut_distance
                c_real(cut_width, floattype),  # cut_width
                ct.c_int(bins_per_profile),  # bins_per_profile
                ct.c_int(n_active_profiles),  # n_profiles
                ct.c_int(len(filling_pattern)),  # n_buckets
                ct.c_int(len(x)),  # n_macroparticles # n_macroparticles
                _get_pointer(filling_pattern),  # filling_pattern
                _get_pointer(
                    bucket_index_to_memory_index
                ),  # bucket_index_to_memory_index
            )

        @staticmethod
        def wake_from_pole_residue(
            # read
            profile: NumpyArray,
            profile_dts: NumpyArray,
            poles: NumpyArray,
            residues: NumpyArray,
            is_counterrotating_beam: bool,
            counterrotating_pole_signs: NumpyArray,
            update_on_bin: NumpyArray,
            factor: float,
            # write
            states: NumpyArray,
            voltage: NumpyArray,
            voltage_threaded: NumpyArray,
        ) -> None:
            """
            Apply poles based on the `profile` to generate `voltage`.

            Parameters
            ----------
            profile
                Beam profile histogram.
            profile_dts
                Base for time step, connected to `update_on_bin`.
            poles
                Complex poles of an equivalent circuit model.
            residues
                Complex residues of an equivalent circuit model.
            is_counterrotating_beam
                If true, the current beam is counter-rotating.
            counterrotating_pole_signs
                Array per pole, -1 if the sign of the impedance is flipped
                for a counter-rotating beam.
            update_on_bin
                Index when to trigger an update of dt. For speedup.
                E.g. For profile no.: `0,0,0,1,1,1,1,2,2,2`
                one needs `update_on_bin = [0,3,7]`.
            factor
                To convert `profile` to current per bin [A].
            states
                Complex state vector, initially ``(0 + 0j)``.
            voltage
                Output voltage, in [V].
            voltage_threaded
                Cached `voltage` array per thread. For speedup.
            """
            assert _is_valid(
                (profile, floattype),
                (profile_dts, floattype),
                (poles, complextype),
                (residues, complextype),
                (counterrotating_pole_signs, floattype),
                (states, complextype),
                (voltage, floattype),
                (voltage_threaded, floattype),
                (update_on_bin, np.int32),
            )

            # Array pointers come from the shared (cached) `_get_pointer`; the
            # changing scalars and the cheap sizes are passed as fresh typed
            # ctypes objects (same convention as every other callable here).
            _LIBBLOND.wake_from_pole_residue(
                _get_pointer(profile),
                _get_pointer(profile_dts),
                _get_pointer(poles),
                _get_pointer(residues),
                ct.c_bool(is_counterrotating_beam),
                _get_pointer(counterrotating_pole_signs),
                _get_pointer(update_on_bin),
                c_real(factor, floattype),
                _get_pointer(states),
                _get_pointer(voltage),
                _get_pointer(voltage_threaded),
                _get_len(profile),  # n_bins
                _get_len(poles),  # n_poles
                ct.c_int(voltage_threaded.shape[0]),  # n_threads
                _get_len(update_on_bin),  # n_updates
                _get_len(profile_dts),  # n_profile_dts
            )

        @staticmethod
        def wake_from_twc_fir(
            # read
            profile: NumpyArray,
            grid_index: NumpyArray,
            r_shunt: NumpyArray,
            a_tilde: NumpyArray,
            omega_r: NumpyArray,
            bin_dt: float,
            factor: float,
            # write
            voltage: NumpyArray,
            voltage_threaded: NumpyArray,
        ) -> None:
            """
            Travelling-wave-cavity wake via a phasor FIR recursion.

            See the ``Specials`` ABC for the full description of the
            algorithm and its lattice-grid convention.

            Parameters
            ----------
            profile
                Beam profile histogram (occupied lattice sites only).
            grid_index
                Lattice site of each profile bin, strictly increasing.
            r_shunt
                Shunt impedance per TWC mode, in [Ohm].
            a_tilde
                Wake support (filling) time per mode, in [s].
            omega_r
                Angular resonant frequency per mode, in [rad/s].
            bin_dt
                Spacing of the underlying equidistant lattice, in [s].
            factor
                To convert `profile` to current per bin [A].
            voltage
                Output voltage, in [V]. Overwritten.
            voltage_threaded
                Cached `voltage` array per thread. For speedup.
            """
            assert profile.dtype == floattype
            assert grid_index.dtype == np.int32
            assert r_shunt.dtype == floattype
            assert a_tilde.dtype == floattype
            assert omega_r.dtype == floattype
            assert voltage.dtype == floattype
            assert voltage_threaded.dtype == floattype

            assert profile.flags.c_contiguous
            assert grid_index.flags.c_contiguous
            assert r_shunt.flags.c_contiguous
            assert a_tilde.flags.c_contiguous
            assert omega_r.flags.c_contiguous
            assert voltage.flags.c_contiguous
            assert voltage_threaded.flags.c_contiguous

            assert len(grid_index) == len(profile)
            assert len(r_shunt) == len(a_tilde)
            assert len(r_shunt) == len(omega_r)

            _LIBBLOND.wake_from_twc_fir(
                _get_pointer(profile),
                _get_pointer(grid_index),
                _get_pointer(r_shunt),
                _get_pointer(a_tilde),
                _get_pointer(omega_r),
                c_real(bin_dt, floattype),
                c_real(factor, floattype),
                _get_pointer(voltage),
                _get_pointer(voltage_threaded),
                ct.c_int(len(profile)),  # n_bins
                ct.c_int(len(r_shunt)),  # n_modes
                ct.c_int(voltage_threaded.shape[0]),  # n_threads
            )

    return CppSpecials


def __getattr__(name: str):
    """
    Provide `CppSpecials` lazily (PEP 562).

    Building `CppSpecials` loads (and potentially compiles) the C++
    library, which must not happen as a side effect of importing this
    module; `set_specials("cpp")` builds it via `reload_cpp_backend`
    anyway.

    Parameters
    ----------
    name
        Name of the requested module attribute.

    Returns
    -------
    attribute
        The lazily created module attribute.
    """
    if name == "CppSpecials":
        cpp_specials = reload_cpp_backend(
            floattype=backend.float,
            parallel=True,
        )
        globals()["CppSpecials"] = cpp_specials
        return cpp_specials
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
