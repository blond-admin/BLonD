'''
BLonD math wrapper functions

@author Konstantinos Iliakis
@date 20.10.2017
'''

import ctypes as ct
import numpy as np
import os
from .. import libblond as __lib

def __getPointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def __getLen(x):
    return ct.c_int(len(x))


def convolve(signal, kernel, mode='full', result=None):
    if mode != 'full':
        # ConvolutionError
        raise RuntimeError('[convolve] Only full mode is supported')
    if result is None:
        result = np.empty(len(signal) + len(kernel) - 1, dtype=float)
    __lib.convolution(__getPointer(signal), __getLen(signal),
                      __getPointer(kernel), __getLen(kernel),
                      __getPointer(result))
    return result

# Similar to np.where with a condition of more_than < x < less_than
# You need to define at least one of more_than, less_than
# @return: a bool array, size equal to the input,
#           True: element satisfied the cond, False: otherwise


def where(x, more_than=None, less_than=None, result=None):
    if result is None:
        result = np.empty(len(x), dtype=np.bool)
    if more_than is None and less_than is not None:
        __lib.where_less_than(__getPointer(x), __getLen(x),
                              ct.c_double(less_than),
                              __getPointer(result))
    elif more_than is not None and less_than is None:
        __lib.where_more_than(__getPointer(x), __getLen(x),
                              ct.c_double(more_than),
                              __getPointer(result))

    elif more_than is not None and less_than is not None:
        __lib.where_more_less_than(__getPointer(x), __getLen(x),
                                   ct.c_double(more_than),
                                   ct.c_double(less_than),
                                   __getPointer(result))

    else:
        raise RuntimeError(
            '[bmath:where] You need to define at least one of more_than, less_than')
    return result


def mean(x):
    __lib.mean.restype = ct.c_double
    return __lib.mean(__getPointer(x), __getLen(x))


def std(x):
    __lib.stdev.restype = ct.c_double
    return __lib.stdev(__getPointer(x), __getLen(x))


def sin(x, result=None):
    if isinstance(x, np.ndarray) and isinstance(x[0], np.float64):
        if result is None:
            result = np.empty(len(x), dtype=float)
        __lib.fast_sinv(__getPointer(x), __getLen(x), __getPointer(result))
        return result
    elif isinstance(x, float) or isinstance(x, int):
        __lib.fast_sin.restype = ct.c_double
        return __lib.fast_sin(ct.c_double(x))
    else:
        # TypeError
        raise RuntimeError('[sin] The type %s is not supported', type(x))


def cos(x, result=None):
    if isinstance(x, np.ndarray) and isinstance(x[0], np.float64):
        if result is None:
            result = np.empty(len(x), dtype=float)
        __lib.fast_cosv(__getPointer(x), __getLen(x), __getPointer(result))
        return result
    elif isinstance(x, float) or isinstance(x, int):
        __lib.fast_cos.restype = ct.c_double
        return __lib.fast_cos(ct.c_double(x))
    else:
        # TypeError
        raise RuntimeError('[cos] The type %s is not supported', type(x))


def exp(x, result=None):
    if isinstance(x, np.ndarray) and isinstance(x[0], np.float64):
        if result is None:
            result = np.empty(len(x), dtype=float)
        __lib.fast_expv(__getPointer(x), __getLen(x), __getPointer(result))
        return result
    elif isinstance(x, float) or isinstance(x, int):
        __lib.fast_exp.restype = ct.c_double
        return __lib.fast_exp(ct.c_double(x))
    else:
        # TypeError
        raise RuntimeError('[exp] The type %s is not supported', type(x))


def interp(x, xp, yp, left=None, right=None, result=None):
    if not left:
        left = yp[0]
    if not right:
        right = yp[-1]
    if result is None:
        result = np.empty(len(x), dtype=float)
    __lib.interp(__getPointer(x), __getLen(x),
                 __getPointer(xp), __getLen(xp),
                 __getPointer(yp),
                 ct.c_double(left),
                 ct.c_double(right),
                 __getPointer(result))
    return result


def cumtrapz(y, x=None, dx=1.0, initial=None, result=None):
    if x is not None:
        # IntegrationError
        raise RuntimeError('[cumtrapz] x attribute is not yet supported')
    if initial:
        if result is None:
            result = np.empty(len(y), dtype=float)
        __lib.cumtrapz_w_initial(__getPointer(y),
                                 ct.c_double(dx), ct.c_double(initial),
                                 __getLen(y), __getPointer(result))
    else:
        if result is None:
            result = np.empty(len(y)-1, dtype=float)
        __lib.cumtrapz_wo_initial(__getPointer(y), ct.c_double(dx),
                                  __getLen(y), __getPointer(result))
    return result


def trapz(y, x=None, dx=1.0):
    if x is None:
        __lib.trapz_const_delta.restype = ct.c_double
        return __lib.trapz_const_delta(__getPointer(y), ct.c_double(dx),
                                       __getLen(y))
    else:
        __lib.trapz_var_delta.restype = ct.c_double
        return __lib.trapz_var_delta(__getPointer(y), __getPointer(x),
                                     __getLen(y))


def argmin(x):
    __lib.min_idx.restype = ct.c_int
    return __lib.min_idx(__getPointer(x), __getLen(x))


def argmax(x):
    __lib.max_idx.restype = ct.c_int
    return __lib.max_idx(__getPointer(x), __getLen(x))


def linspace(start, stop, num=50, retstep=False, result=None):
    if result is None:
        result = np.empty(num, dtype=float)
    __lib.linspace(ct.c_double(start), ct.c_double(stop),
                   ct.c_int(num), __getPointer(result))
    if retstep:
        return result, 1. * (stop-start) / (num-1)
    else:
        return result


def arange(start, stop, step, dtype=float, result=None):
    size = int(np.ceil((stop-start)/step))
    if result is None:
        result = np.empty(size, dtype=dtype)
    if dtype == float:
        __lib.arange_double(ct.c_double(start), ct.c_double(stop),
                            ct.c_double(step), __getPointer(result))
    elif dtype == int:
        __lib.arange_int(ct.c_int(start), ct.c_int(stop),
                         ct.c_int(step), __getPointer(result))

    return result


def sum(x):
    __lib.sum.restype = ct.c_double
    return __lib.sum(__getPointer(x), __getLen(x))


def sort(x, reverse=False):
    if x.dtype == 'int32':
        __lib.sort_int(__getPointer(x), __getLen(x), ct.c_bool(reverse))
    elif x.dtype == 'float64':
        __lib.sort_double(__getPointer(x), __getLen(x), ct.c_bool(reverse))
    elif x.dtype == 'int64':
        __lib.sort_longint(__getPointer(x), __getLen(x), ct.c_bool(reverse))
    else:
        # SortError
        raise RuntimeError('[sort] Datatype %s not supported' % x.dtype)
    return x


def rfft(a, n=0, result=None):
    if (n == 0) and (result == None):
        result = np.empty(len(a)//2 + 1, dtype=np.complex128)
    elif (n != 0) and (result == None):
        result = np.empty(n//2 + 1, dtype=np.complex128)

    __lib.rfft(__getPointer(a),
               __getLen(a),
               __getPointer(result),
               ct.c_int(int(n)),
               ct.c_int(int(os.environ.get('OMP_NUM_THREADS', 1))))

    return result


def irfft(a, n=0, result=None):

    if (n == 0) and (result == None):
        result = np.empty(2*(len(a)-1), dtype=np.float64)
    elif (n != 0) and (result == None):
        result = np.empty(n, dtype=np.float64)

    __lib.irfft(__getPointer(a),
                __getLen(a),
                __getPointer(result),
                ct.c_int(int(n)),
                ct.c_int(int(os.environ.get('OMP_NUM_THREADS', 1))))
    return result


def rfftfreq(n, d=1.0, result=None):
    if d == 0:
        raise ZeroDivisionError('d must be non-zero')
    if result is None:
        result = np.empty(n//2 + 1, dtype=np.float64)

    __lib.rfftfreq(ct.c_int(n),
                   __getPointer(result),
                   ct.c_double(d))
    return result
