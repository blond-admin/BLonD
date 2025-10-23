"""
Basic methods and objects related to the computational core.

@author: Konstantinos Iliakis
@date: 25.05.2023
"""

from __future__ import annotations
import ctypes as ct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


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

from .bmath_backends import BlondMathBackend

bmath = BlondMathBackend()  # this line controls static type hints of bmath
bmath.use_cpu()  # this line changes the backend to the most suitable one
