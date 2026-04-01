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
    except (OSError, FileNotFoundError):
        from blond.core.backends.cpp.compile import compile_cpp_library

        print(
            "C++ backend was not found.. Trying to compile parallel backend."
        )
        compile_cpp_library(parallel=True)
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
                f"{__file__.replace('callables.py', 'compile.py')}:1"  # :1 to
                # make PyCharm automatically link the correct file
            ) from exc

    def _getPointer(x: NumpyArray) -> ct.c_void_p:
        return x.ctypes.data_as(ct.c_void_p)

    def _getLen(x: NumpyArray) -> ct.c_int:
        return ct.c_int(len(x))

    _LIBBLOND.beam_phase.restype = c_real_t(floattype)
    _LIBBLOND.sum_1d_array.restype = c_real_t(floattype)
    _LIBBLOND.dot_product_1d_array.restype = c_real_t(floattype)

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
        def sum_1d_array(array: NumpyArray) -> float:
            assert array.dtype == floattype
            # requires setting of _LIBBLOND.sum_1d_array.restype = c_real_t(floattype) in
            # reload function
            return floattype(
                _LIBBLOND.sum_1d_array(_getPointer(array), _getLen(array))
            )

        @staticmethod
        def dot_product_1d_array(
            array_1: NumpyArray,
            array_2: NumpyArray,
        ) -> float:
            assert array_1.dtype == floattype
            assert array_2.dtype == floattype
            assert len(array_1) == len(array_2)

            # requires setting of _LIBBLOND.dot_product_1d_array.restype = c_real_t(floattype) in
            # reload function
            return floattype(
                _LIBBLOND.dot_product_1d_array(
                    _getPointer(array_1),
                    _getPointer(array_2),
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
        def drift_exact(
            dt: NumpyArray,
            dE: NumpyArray,
            T: float,
            alpha_0: float,
            higher_alpha: NumpyArray,
            beta: float,
            energy: float,
        ):
            assert dt.dtype == floattype
            assert dE.dtype == floattype
            assert higher_alpha.dtype == floattype

            assert dt.flags.c_contiguous
            assert dE.flags.c_contiguous
            assert higher_alpha.flags.c_contiguous

            # Cast Python floats to backend floattype
            T = floattype(T)
            beta = floattype(beta)
            energy = floattype(energy)
            alpha_0 = floattype(alpha_0)

            _LIBBLOND.drift_exact(
                _getPointer(dt),  # real_t *__restrict__ beam_dt
                _getPointer(dE),  # const real_t *__restrict__ beam_dE
                c_real(T, floattype),  # const real_t T
                c_real(alpha_0, floattype),  # const real_t alpha_zero
                _getPointer(
                    higher_alpha
                ),  # const real_t *__restrict__ higher_alpha
                _getLen(higher_alpha),  # const int n_alpha
                c_real(beta, floattype),  # const real_t beta
                c_real(energy, floattype),  # const real_t energy
                _getLen(dt),  # const int n_macroparticles
            )

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

    return CppSpecials


CppSpecials = reload_cpp_backend(backend.float)
