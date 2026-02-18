# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `CppSpecials` and helper functions."""

from __future__ import annotations

import ctypes as ct
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import Specials, backend

if TYPE_CHECKING:  # pragma: no cover
    from ctypes import CDLL

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def c_real(
    scalar: float, floattype: type[np.float32] | type[np.float64]
) -> ct.c_float | ct.c_double:
    """Convert input to default precision."""
    if floattype == np.float32:
        return ct.c_float(scalar)
    elif floattype == np.float64:
        return ct.c_double(scalar)
    else:
        raise ValueError(floattype)


def c_real_t(
    floattype: type[np.float32] | type[np.float64],
) -> type[ct.c_float | ct.c_double]:
    """Get default precision."""
    if floattype == np.float32:
        return ct.c_float
    elif floattype == np.float64:
        return ct.c_double
    else:
        raise ValueError(floattype)


def reload_cpp_backend(  # NOQA: PLR0915
    floattype: type[np.float32] | type[np.float64],
) -> CppSpecials:
    """
    Load and link the according C++ backend.

    Parameters
    ----------
    floattype
        Float type to compile the backend for.
        32 or 64 bit.

    Returns
    -------
    CppSpecials
        The `CppSpecials` class.

    """

    def load_libblond(precision: str = "single") -> CDLL:
        """
        Locates and initializes the blond compiled library.

        Parameters
        ----------
        precision
            The floating point precision of the calculations.
            Can be 'single' or 'double'.
            Default is  "single".
        """
        libblond_path_ = os.environ.get("LIBBLOND", None)

        from blond.generals.hashing_ import hash_in_folder

        folder = os.path.dirname(os.path.abspath(__file__))

        hash_ = hash_in_folder(
            folder=folder,
            extensions=(".py", ".h", ".cpp"),
            recursive=False,
        )
        basepath = os.path.join(folder, "compiled", hash_)
        if "posix" in os.name:
            if libblond_path_:
                libblond_path = os.path.abspath(libblond_path_)
            else:
                libblond_path = os.path.join(
                    basepath, f"libblond_{precision}.so"
                )
            _LIBBLOND = ct.CDLL(str(libblond_path))
        elif "win" in sys.platform:
            if libblond_path_:
                libblond_path = os.path.abspath(libblond_path_)
            else:
                libblond_path = os.path.join(
                    basepath, f"libblond_{precision}.dll"
                )

            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(os.path.dirname(libblond_path))
                _LIBBLOND = ct.CDLL(str(libblond_path), winmode=0)
            else:
                _LIBBLOND = ct.CDLL(str(libblond_path))
        else:
            raise ValueError(
                f"Supporting 'win' and 'posix', not {sys.platform}."
            )

        return _LIBBLOND

    try:
        if floattype == np.float32:
            _LIBBLOND = load_libblond(precision="single")
        elif floattype == np.float64:
            _LIBBLOND = load_libblond(precision="double")
        else:
            raise TypeError(floattype)
    except (OSError, FileNotFoundError) as exc:
        raise OSError(
            "`load_libblond` failed. Has the backend been compiled?\n"
            f"{__file__.replace('callables.py', 'compile.py')}:1"
        ) from exc

    def _getPointer(x: NumpyArray) -> ct.c_void_p:
        return x.ctypes.data_as(ct.c_void_p)

    def _getLen(x: NumpyArray) -> ct.c_int:
        return ct.c_int(len(x))

    _LIBBLOND.beam_phase.restype = c_real_t(floattype)

    class CppSpecials(Specials):
        @staticmethod
        def beam_phase(
            hist_x: NumpyArray,
            hist_y: NumpyArray,
            alpha: float,
            omega_rf: float,
            phi_rf: float,
            bin_size: float,
        ) -> float:
            assert hist_x.dtype == floattype
            assert hist_y.dtype == floattype
            assert hist_x.flags.c_contiguous
            assert hist_y.flags.c_contiguous

            # Cast Python floats to backend floattype
            alpha = floattype(alpha)
            omega_rf = floattype(omega_rf)
            phi_rf = floattype(phi_rf)
            bin_size = floattype(bin_size)

            return _LIBBLOND.beam_phase(
                hist_x.ctypes.data_as(ct.c_void_p),  # bin_centers
                hist_y.ctypes.data_as(ct.c_void_p),  # profile
                c_real(alpha, floattype),  # alpha
                c_real(omega_rf, floattype),  # omega_rf
                c_real(phi_rf, floattype),  # phi_rf
                c_real(bin_size, floattype),  # bin_size
                ct.c_int(len(hist_x)),  # n_bins
            )

        @staticmethod
        def histogram(
            array_read: NumpyArray,
            array_write: NumpyArray,
            start: float,
            stop: float,
        ) -> None:
            assert array_read.dtype == floattype
            assert array_write.dtype == floattype
            assert array_read.flags.c_contiguous
            assert array_write.flags.c_contiguous

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
        def kick_induced_voltage(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: NumpyArray,
            bin_centers: NumpyArray,
            charge: float,
            acceleration_kick: float,
        ) -> None:
            assert dt.dtype == floattype
            assert dE.dtype == floattype
            assert voltage.dtype == floattype
            assert bin_centers.dtype == floattype
            assert dt.flags.c_contiguous
            assert dE.flags.c_contiguous
            assert voltage.flags.c_contiguous
            assert bin_centers.flags.c_contiguous

            # Cast Python floats to backend floattype
            charge = floattype(charge)
            acceleration_kick = floattype(acceleration_kick)

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

        @staticmethod
        def loss_box(
            e_max: float,
            e_min: float,
            t_min: float,
            t_max: float,
            dt: CupyArray,
            dE: CupyArray,
            flags: CupyArray,
        ) -> None:
            _LIBBLOND.loss_box(
                c_real(e_max, floattype),
                c_real(e_min, floattype),
                c_real(t_min, floattype),
                c_real(t_max, floattype),
                _getPointer(dt),
                _getPointer(dE),
                _getPointer(flags),
                _getLen(dt),
            )

        @staticmethod
        def kick_single_harmonic(
            dt: NumpyArray | CupyArray,
            dE: NumpyArray | CupyArray,
            voltage: float,
            omega_rf: float,
            phi_rf: float,
            charge: float,
            acceleration_kick: float,
        ) -> None:
            assert dt.dtype == floattype
            assert dE.dtype == floattype
            assert dt.flags.c_contiguous
            assert dE.flags.c_contiguous

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
            dt: NumpyArray | CupyArray,
            dE: NumpyArray | CupyArray,
            voltage: NumpyArray,
            omega_rf: NumpyArray,
            phi_rf: NumpyArray,
            charge: float,
            n_rf: int,
            acceleration_kick: float,
        ) -> None:
            assert dt.dtype == floattype
            assert dE.dtype == floattype
            assert voltage.dtype == floattype
            assert omega_rf.dtype == floattype
            assert phi_rf.dtype == floattype
            assert dt.flags.c_contiguous
            assert dE.flags.c_contiguous
            assert voltage.flags.c_contiguous
            assert omega_rf.flags.c_contiguous
            assert phi_rf.flags.c_contiguous

            # Cast Python floats to backend floattype
            charge = floattype(charge)
            acceleration_kick = floattype(acceleration_kick)

            _LIBBLOND.kick_multi_harmonic(
                _getPointer(dt),
                _getPointer(dE),
                ct.c_int(n_rf),
                c_real(charge, floattype),
                _getPointer(voltage),
                _getPointer(omega_rf),
                _getPointer(phi_rf),
                _getLen(dt),
                c_real(acceleration_kick, floattype),
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
            assert dt.dtype == floattype
            assert dE.dtype == floattype
            assert dt.flags.c_contiguous
            assert dE.flags.c_contiguous

            # Cast Python floats to backend floattype
            T = floattype(T)
            eta_0 = floattype(eta_0)
            beta = floattype(beta)
            energy = floattype(energy)

            _LIBBLOND.drift_simple(
                _getPointer(dt),
                _getPointer(dE),
                c_real(T, floattype),
                c_real(eta_0, floattype),
                c_real(beta, floattype),
                c_real(energy, floattype),
                _getLen(dt),
            )

        @staticmethod
        def drift_legacy(
            dt: NumpyArray,
            dE: NumpyArray,
            t_rev: float,
            length_ratio: float,
            alpha_order,
            eta_0: float,
            eta_1: float,
            eta_2: float,
            beta: float,
            energy: float,
        ):
            pass

        @staticmethod
        def drift_exact(
            dt: NumpyArray,
            dE: NumpyArray,
            t_rev: float,
            length_ratio: float,
            alpha_0: float,
            alpha_1: float,
            alpha_2: float,
            beta: float,
            energy: float,
        ):
            pass

        @staticmethod
        def move_flagged_elements_to_end(
            flag: int,
            flags: NumpyArray | CupyArray,  # also purged
            dt: NumpyArray | CupyArray,
            dE: NumpyArray | CupyArray,
            ids: NumpyArray | CupyArray,
        ):
            assert dt.dtype == floattype
            assert dE.dtype == floattype
            assert dt.flags.c_contiguous
            assert dE.flags.c_contiguous

            assert flags.dtype == np.int32
            assert ids.dtype == np.int32
            assert flags.flags.c_contiguous
            assert ids.flags.c_contiguous

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
        def sparse_histogram_strided(
            x: NumpyArray,
            out: NumpyArray,
            first_left_cut: float,
            left_cut_distance: float,
            cut_width: float,
            bins_per_profile: int,
            n_profiles: int,
            filling_pattern: NumpyArray,
            bucket_index_to_memory_index: NumpyArray,
        ) -> None:
            """
            Sparse histogram with strided memory layout (gaps between profiles).

            Parameters
            ----------
            x
                An array, e.g., the particle dt values.
            out
                Output histogram (n_filled_buckets * stride).
            first_left_cut
                Start of the first histogram.
            left_cut_distance
                Distance between the start of each histogram.
            cut_width
                Distance between left and right edge of the histogram.
            bins_per_profile
                Number of bins per bucket.
            n_profiles
                Number of non-empty buckets.
            filling_pattern
                Filling pattern as a boolean array
                where ``True`` means filled bucket.
            bucket_index_to_memory_index
                Maps bucket index to memory index.
                For a ``filling_pattern = [1, 0, 0, 1]``
                ``bucket_index_to_memory_index = [8, 8, 8, 16]`` with
                ``bins_per_profile = 8``.
            """
            assert x.dtype == floattype
            assert out.dtype == floattype
            assert filling_pattern.dtype == np.bool
            assert bucket_index_to_memory_index.dtype == np.int32

            assert x.flags.c_contiguous
            assert out.flags.c_contiguous
            assert filling_pattern.flags.c_contiguous
            assert bucket_index_to_memory_index.flags.c_contiguous

            _LIBBLOND.sparse_histogram_strided(
                _getPointer(x),
                _getPointer(out),
                c_real(first_left_cut, floattype),
                c_real(left_cut_distance, floattype),
                c_real(cut_width, floattype),
                ct.c_int(bins_per_profile),
                ct.c_int(n_profiles),
                ct.c_int(len(filling_pattern)),
                ct.c_int(len(x)),  # n_macroparticles
                _getPointer(filling_pattern),
                _getPointer(bucket_index_to_memory_index),
            )

        @staticmethod
        def apply_poles2(
            # read
            profile,
            profile_dts,
            poles,
            residues,
            # write
            states,
            voltage,
            voltage_threaded,
            update_on_bin,
            factor,
        ) -> None:
            complextype = (
                np.complex64 if floattype == np.float32 else np.complex128
            )

            assert profile.dtype == floattype
            assert profile_dts.dtype == floattype
            assert poles.dtype == complextype
            assert residues.dtype == complextype
            assert states.dtype == complextype
            assert voltage.dtype == floattype
            assert voltage_threaded.dtype == floattype
            assert update_on_bin.dtype == np.int32

            assert profile.flags.c_contiguous
            assert profile_dts.flags.c_contiguous
            assert poles.flags.c_contiguous
            assert residues.flags.c_contiguous
            assert states.flags.c_contiguous
            assert voltage.flags.c_contiguous
            assert voltage_threaded.flags.c_contiguous
            assert update_on_bin.flags.c_contiguous

            _LIBBLOND.apply_poles(
                _getPointer(profile),
                _getPointer(profile_dts),
                _getPointer(poles),
                _getPointer(residues),
                _getPointer(states),
                _getPointer(voltage),
                _getPointer(voltage_threaded),
                _getPointer(update_on_bin),
                c_real(factor, floattype),
                ct.c_int(len(profile)),  # n_bins
                ct.c_int(len(poles)),  # n_poles
                ct.c_int(voltage_threaded.shape[0]),  # n_threads
                ct.c_int(len(update_on_bin)),  # n_updates
                ct.c_int(len(profile_dts)),  # n_profile_dts
            )

    return CppSpecials


CppSpecials = reload_cpp_backend(backend.float)
