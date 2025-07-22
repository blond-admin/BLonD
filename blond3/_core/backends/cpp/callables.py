from __future__ import annotations

import ctypes as ct
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

from blond3._core.backends.backend import Specials

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray


class PrecisionClass:
    """Singleton class. Holds information about the floating point precision of the calculations."""

    __instance = None

    def __init__(self, _precision: str = "double"):
        """Constructor

        Args:
            _precision (str, optional): _description_. Defaults to 'double'.
        """
        if PrecisionClass.__instance is not None:
            return

        PrecisionClass.__instance = self
        self.set(_precision)

    def set(self, _precision: str = "double"):
        """Set the precision to single or double.

        Args:
            _precision (str, optional): _description_. Defaults to 'double'.
        """
        if _precision in ["single", "s", "32", "float32", "float", "f"]:
            self.str = "single"
            self.real_t = np.float32
            self.c_real_t = ct.c_float
            self.complex_t = np.complex64
            self.num = 1
        elif _precision in ["double", "d", "64", "float64"]:
            self.str = "double"
            self.real_t = np.float64
            self.c_real_t = ct.c_double
            self.complex_t = np.complex128
            self.num = 2
        else:
            msg = f"{_precision=} is not recognized, use 'single' or 'double'"
            raise ValueError(msg)


class c_complex128(ct.Structure):
    """128-bit (64+64) Complex number, compatible with std::complex layout"""

    _fields_ = [("real", ct.c_double), ("imag", ct.c_double)]

    def __init__(self, pycomplex: NumpyArray):
        """Init from Python complex

        Args:
            pycomplex (_type_): _description_
        """
        self.real = pycomplex.real.astype(np.float64, order="C")
        self.imag = pycomplex.imag.astype(np.float64, order="C")

    def to_complex(self):
        """Convert to Python complex

        Returns:
            _type_: _description_
        """
        return self.real + (1.0j) * self.imag


class c_complex64(ct.Structure):
    """64-bit (32+32) Complex number, compatible with std::complex layout"""

    _fields_ = [("real", ct.c_float), ("imag", ct.c_float)]

    def __init__(self, pycomplex: NumpyArray):
        """Init from Python complex

        Args:
            pycomplex (_type_): _description_
        """
        self.real = pycomplex.real.astype(np.float32, order="C")
        self.imag = pycomplex.imag.astype(np.float32, order="C")

    def to_complex(self):
        """Convert to Python complex

        Returns:
            _type_: _description_
        """
        return self.real + (1.0j) * self.imag


def c_real(scalar: float) -> ct.c_float | ct.c_double:
    """Convert input to default precision."""
    if precision.num == 1:
        return ct.c_float(scalar)
    return ct.c_double(scalar)


def c_complex(scalar: complex):
    """Convert input to default precision."""
    if precision.num == 1:
        return c_complex64(scalar)
    return c_complex128(scalar)


# By default, use double precision
precision = PrecisionClass("double")


def load_libblond(precision: str = "single"):
    """Locates and initializes the blond compiled library
    @param precision: The floating point precision of the calculations. Can be 'single' or 'double'.
    """
    libblond_path = os.environ.get("LIBBLOND", None)
    path = os.path.realpath(__file__)
    basepath = os.sep.join(path.split(os.sep)[:-1])
    if "posix" in os.name:
        if libblond_path:
            libblond_path = os.path.abspath(libblond_path)
        else:
            libblond_path = os.path.join(basepath, f"libblond_{precision}.so")
        _LIBBLOND = ct.CDLL(libblond_path)
    elif "win" in sys.platform:
        if libblond_path:
            libblond_path = os.path.abspath(libblond_path)
        else:
            libblond_path = os.path.join(basepath, f"libblond_{precision}.dll")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.path.dirname(libblond_path))
            _LIBBLOND = ct.CDLL(libblond_path, winmode=0)
        else:
            _LIBBLOND = ct.CDLL(libblond_path)
    else:
        print("YOU DO NOT HAVE A WINDOWS OR UNIX OPERATING SYSTEM. ABORTING.")
        sys.exit()

    return _LIBBLOND


_LIBBLOND = load_libblond(precision="double")


def _getPointer(x: NumpyArray) -> ct.c_void_p:
    return x.ctypes.data_as(ct.c_void_p)


def _getLen(x: NumpyArray) -> ct.c_int:
    return ct.c_int(len(x))


class CppSpecials(Specials):
    @staticmethod
    def histogram(
        array_read: NumpyArray, array_write: NumpyArray, start: float, stop: float
    ):
        _LIBBLOND.histogram(
            array_read.ctypes.data_as(ct.c_void_p),
            array_write.ctypes.data_as(ct.c_void_p),
            c_real(start),
            c_real(stop),
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
    ):
        _LIBBLOND.linear_interp_kick(
            dt.ctypes.data_as(ct.c_void_p),
            dE.ctypes.data_as(ct.c_void_p),
            voltage.ctypes.data_as(ct.c_void_p),
            bin_centers.ctypes.data_as(ct.c_void_p),
            c_real(charge),
            ct.c_int(len(bin_centers)),
            ct.c_int(len(dt)),
            c_real(acceleration_kick),
        )

    @staticmethod
    def loss_box(self, a, b, c, d) -> None:
        pass

    @staticmethod
    def kick_single_harmonic(
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: float,
        acceleration_kick: float,
    ):
        _LIBBLOND.kick_single_harmonic(
            dt.ctypes.data_as(ct.c_void_p),
            dE.ctypes.data_as(ct.c_void_p),
            c_real(charge),
            c_real(voltage),
            c_real(omega_rf),
            c_real(phi_rf),
            ct.c_int(len(dt)),
            c_real(acceleration_kick),
        )
        return

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
    ):
        _LIBBLOND.kick_multi_harmonic(
            _getPointer(dt),
            _getPointer(dE),
            ct.c_int(n_rf),
            c_real(charge),
            _getPointer(voltage),
            _getPointer(omega_rf),
            _getPointer(phi_rf),
            _getLen(dt),
            c_real(acceleration_kick),
        )
        return

    @staticmethod
    def drift_simple(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ):
        _LIBBLOND.drift_simple(
            _getPointer(dt),
            _getPointer(dE),
            c_real(T),
            c_real(eta_0),
            c_real(beta),
            c_real(energy),
            _getLen(dt),
        )
        return

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
