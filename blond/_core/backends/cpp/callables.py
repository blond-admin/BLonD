from __future__ import annotations

import ctypes as ct
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

from ...._core.backends.backend import Specials, backend

if TYPE_CHECKING:  # pragma: no cover
    from ctypes import CDLL

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


class PrecisionClass:
    """Singleton class. Holds information about the floating point precision of the calculations."""

    int_t: type[np.float32 | np.float64]
    real_t: type[np.float32 | np.float64]
    c_real_t: type[ct.c_float | ct.c_double]
    complex_t: type[np.complex64 | np.complex128]

    __instance = None

    def __init__(self, _precision: str = "double") -> None:
        """Constructor.

        Args:
            _precision (str, optional): _description_. Defaults to 'double'.
        """
        PrecisionClass.__instance = self
        self.set(_precision)

    def set(self, _precision: str = "double") -> None:
        """Set the precision to single or double.

        Args:
            _precision (str, optional): _description_. Defaults to 'double'.
        """
        if _precision in ["single", "s", "32", "float32", "float", "f"]:
            self.str = "single"
            self.int_t = np.int32
            self.real_t = np.float32
            self.c_real_t = ct.c_float
            self.complex_t = np.complex64
            self.num = 1
        elif _precision in ["double", "d", "64", "float64"]:
            self.str = "double"
            self.int_t = np.int64
            self.real_t = np.float64
            self.c_real_t = ct.c_double
            self.complex_t = np.complex128
            self.num = 2
        else:
            msg = f"{_precision=} is not recognized, use 'single' or 'double'"
            raise ValueError(msg)


class c_complex128(ct.Structure):
    """128-bit (64+64) Complex number, compatible with std::complex layout."""

    real: ct.c_double
    imag: ct.c_double

    def __init__(self, pycomplex: complex) -> None:
        """Init from Python complex.

        Args:
            pycomplex (_type_): _description_
        """
        # FIXME this seems broken from the type hint side, but is anyway not used
        self.real = pycomplex.real.astype(np.float64, order="C")  # type: ignore
        self.imag = pycomplex.imag.astype(np.float64, order="C")  # type: ignore

    def to_complex(self) -> complex:
        """Convert to Python complex.

        Returns
        -------
            _type_: _description_
        """
        return self.real + 1.0j * self.imag  # type: ignore


class c_complex64(ct.Structure):
    """64-bit (32+32) Complex number, compatible with std::complex layout."""

    _fields_ = [("real", ct.c_float), ("imag", ct.c_float)]

    def __init__(self, pycomplex: complex) -> None:
        """Init from Python complex.

        Args:
            pycomplex (_type_): _description_
        """
        # FIXME this seems broken from the type hint side, but is anyway not used
        self.real = pycomplex.real.astype(np.float32, order="C")  # type: ignore
        self.imag = pycomplex.imag.astype(np.float32, order="C")  # type: ignore

    def to_complex(self) -> complex:
        """Convert to Python complex.

        Returns
        -------
            _type_: _description_
        """
        return self.real + 1.0j * self.imag


def c_int(scalar: int, precision: PrecisionClass) -> ct.c_int32 | ct.c_int64:
    """Convert input to default precision."""
    if precision.num == 1:
        return ct.c_int32(scalar)
    return ct.c_int64(scalar)


def c_real(
    scalar: float, precision: PrecisionClass
) -> ct.c_float | ct.c_double:
    """Convert input to default precision."""
    if precision.num == 1:
        return ct.c_float(scalar)
    return ct.c_double(scalar)


def c_complex(
    scalar: complex, precision: PrecisionClass
) -> c_complex128 | c_complex64:
    """Convert input to default precision."""
    if precision.num == 1:
        return c_complex64(scalar)
    return c_complex128(scalar)


def reload_cpp_backend(
    floattype: type[np.float32] | type[np.float64],
) -> CppSpecials:
    if floattype == np.float32:
        # By default, use double precision
        precision = PrecisionClass("single")
    elif floattype == np.float64:
        # By default, use double precision
        precision = PrecisionClass("double")
    else:
        raise TypeError(floattype)

    def load_libblond(precision: str = "single") -> CDLL:
        """Locates and initializes the blond compiled library.

        Parameters
        ----------
        precision
            The floating point precision of the calculations.
            Can be 'single' or 'double'.
            Default is  "single".
        """
        libblond_path_ = os.environ.get("LIBBLOND", None)

        from ...._generals._hashing import hash_in_folder

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

    if floattype == np.float32:
        _LIBBLOND = load_libblond(precision="single")
    elif floattype == np.float64:
        _LIBBLOND = load_libblond(precision="double")

    else:
        raise TypeError(floattype)

    def _getPointer(x: NumpyArray) -> ct.c_void_p:
        return x.ctypes.data_as(ct.c_void_p)

    def _getLen(x: NumpyArray) -> ct.c_int:
        return ct.c_int(len(x))

    _LIBBLOND.beam_phase.restype = precision.c_real_t

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
            return _LIBBLOND.beam_phase(
                hist_x.ctypes.data_as(ct.c_void_p),  # bin_centers
                hist_y.ctypes.data_as(ct.c_void_p),  # profile
                c_real(alpha, precision),  # alpha
                c_real(omega_rf, precision),  # omega_rf
                c_real(phi_rf, precision),  # phi_rf
                c_real(bin_size, precision),  # bin_size
                ct.c_int(len(hist_x)),  # n_bins
            )

        @staticmethod
        def histogram(
            array_read: NumpyArray,
            array_write: NumpyArray,
            start: np.float32 | np.float64,
            stop: np.float32 | np.float64,
        ) -> None:
            _LIBBLOND.histogram(
                array_read.ctypes.data_as(ct.c_void_p),
                array_write.ctypes.data_as(ct.c_void_p),
                c_real(start, precision),
                c_real(stop, precision),
                ct.c_int(len(array_write)),
                ct.c_int(len(array_read)),
            )

        @staticmethod
        def kick_induced_voltage(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: NumpyArray,
            bin_centers: NumpyArray,
            charge: np.float32 | np.float64,
            acceleration_kick: np.float32 | np.float64,
        ) -> None:
            _LIBBLOND.linear_interp_kick(
                dt.ctypes.data_as(ct.c_void_p),
                dE.ctypes.data_as(ct.c_void_p),
                voltage.ctypes.data_as(ct.c_void_p),
                bin_centers.ctypes.data_as(ct.c_void_p),
                c_real(charge, precision),
                ct.c_int(len(bin_centers)),
                ct.c_int(len(dt)),
                c_real(acceleration_kick, precision),
            )

        @staticmethod
        def loss_box(
            top: float, bottom: float, left: float, right: float
        ) -> None:
            pass

        @staticmethod
        def kick_single_harmonic(
            dt: NumpyArray | CupyArray,
            dE: NumpyArray | CupyArray,
            voltage: float,
            omega_rf: float,
            phi_rf: float,
            charge: np.float32 | np.float64,
            acceleration_kick: np.float32 | np.float64,
        ) -> None:
            _LIBBLOND.kick_single_harmonic(
                dt.ctypes.data_as(ct.c_void_p),
                dE.ctypes.data_as(ct.c_void_p),
                c_real(charge, precision),
                c_real(voltage, precision),
                c_real(omega_rf, precision),
                c_real(phi_rf, precision),
                ct.c_int(len(dt)),
                c_real(acceleration_kick, precision),
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
            _LIBBLOND.kick_multi_harmonic(
                _getPointer(dt),
                _getPointer(dE),
                ct.c_int(n_rf),
                c_real(charge, precision),
                _getPointer(voltage),
                _getPointer(omega_rf),
                _getPointer(phi_rf),
                _getLen(dt),
                c_real(acceleration_kick, precision),
            )

        @staticmethod
        def drift_simple(
            dt: NumpyArray,
            dE: NumpyArray,
            T: np.float32 | np.float64,
            eta_0: np.float32 | np.float64,
            beta: np.float32 | np.float64,
            energy: np.float32 | np.float64,
        ) -> None:
            _LIBBLOND.drift_simple(
                _getPointer(dt),
                _getPointer(dE),
                c_real(T, precision),
                c_real(eta_0, precision),
                c_real(beta, precision),
                c_real(energy, precision),
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
        def purge4(
            flag: np.int32,
            flags: NumpyArray | CupyArray,  # also purged
            dt: NumpyArray | CupyArray,
            dE: NumpyArray | CupyArray,
            ids: NumpyArray | CupyArray,
        ):
            n_new = _LIBBLOND.purge4(
                ct.c_int32(flag),
                flags.ctypes.data_as(ct.c_void_p),
                dt.ctypes.data_as(ct.c_void_p),
                dE.ctypes.data_as(ct.c_void_p),
                ids.ctypes.data_as(ct.c_void_p),
                ct.c_int(len(dt)),  # n_macroparticles
            )
            n_new = int(n_new)
            return n_new

    return CppSpecials


CppSpecials = reload_cpp_backend(backend.float)
