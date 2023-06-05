'''
Basic methods and objects related to the computational core.

@author: Konstantinos Iliakis
@date: 25.05.2023
'''

import ctypes as ct
import numpy as np


class PrecisionClass:
    """Singleton class. Holds information about the floating point precision of the calculations.
    """
    __instance = None

    def __init__(self, _precision='double'):
        """Constructor

        Args:
            _precision (str, optional): _description_. Defaults to 'double'.
        """
        if PrecisionClass.__instance is not None:
            return

        PrecisionClass.__instance = self
        self.set(_precision)

    def set(self, _precision='double'):
        """Set the precision to single or double.

        Args:
            _precision (str, optional): _description_. Defaults to 'double'.
        """
        if _precision in ['single', 's', '32', 'float32', 'float', 'f']:
            self.str = 'float32'
            self.real_t = np.float32
            self.c_real_t = ct.c_float
            self.complex_t = np.complex64
            self.num = 1
        elif _precision in ['double', 'd', '64', 'float64']:
            self.str = 'float64'
            self.real_t = np.float64
            self.c_real_t = ct.c_double
            self.complex_t = np.complex128
            self.num = 2


class c_complex128(ct.Structure):
    """128-bit (64+64) Complex number, compatible with std::complex layout

    Args:
        ct (_type_): _description_

    Returns:
        _type_: _description_
    """
    _fields_ = [("real", ct.c_double), ("imag", ct.c_double)]

    def __init__(self, pycomplex):
        """Init from Python complex

        Args:
            pycomplex (_type_): _description_
        """
        self.real = pycomplex.real.astype(np.float64, order='C')
        self.imag = pycomplex.imag.astype(np.float64, order='C')

    def to_complex(self):
        """Convert to Python complex

        Returns:
            _type_: _description_
        """
        return self.real + (1.j) * self.imag


class c_complex64(ct.Structure):
    """64-bit (32+32) Complex number, compatible with std::complex layout

    Args:
        ct (_type_): _description_

    Returns:
        _type_: _description_
    """
    _fields_ = [("real", ct.c_float), ("imag", ct.c_float)]

    def __init__(self, pycomplex):
        """Init from Python complex

        Args:
            pycomplex (_type_): _description_
        """
        self.real = pycomplex.real.astype(np.float32, order='C')
        self.imag = pycomplex.imag.astype(np.float32, order='C')

    def to_complex(self):
        """Convert to Python complex

        Returns:
            _type_: _description_
        """
        return self.real + (1.j) * self.imag


def c_real(scalar):
    """Convert input to default precision.

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    if precision.num == 1:
        return ct.c_float(scalar)
    return ct.c_double(scalar)


def c_complex(scalar):
    """Convert input to default precision.

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    if precision.num == 1:
        return c_complex64(scalar)
    return c_complex128(scalar)


# By default use double precision
precision = PrecisionClass('double')
