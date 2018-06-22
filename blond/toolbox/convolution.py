# from ..setup_cpp import libblond
from .. import libblond

import ctypes
import numpy as np


def convolution(signal, kernel):
    signalLen = len(signal)
    kernelLen = len(kernel)
    result = np.empty(signalLen + kernelLen - 1)
    libblond.convolution(signal.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(signalLen),
                       kernel.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(kernelLen),
                       result.ctypes.data_as(ctypes.c_void_p))
    return result
