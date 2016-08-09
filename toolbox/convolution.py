from setup_cpp import libfib
import ctypes
import numpy as np


def convolution(signal, kernel):
    size = len(signal) + len(kernel) - 1
    result = np.ascontiguousarray(np.empty(size))
    libfib.convolution(signal.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(len(signal)),
                       kernel.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(len(kernel)),
                       result.ctypes.data_as(ctypes.c_void_p))
    return result
