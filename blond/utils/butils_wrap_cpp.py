'''
BLonD math wrapper functions

@author Konstantinos Iliakis
@date 20.10.2017
'''

import ctypes as ct
import os

import numpy as np

from .. import LIBBLOND as __lib
from . import c_complex64, c_complex128, c_real, precision


def __getPointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def __getLen(x):
    return ct.c_int(len(x))


# Similar to np.where with a condition of more_than < x < less_than
# You need to define at least one of more_than, less_than
# @return: a bool array, size equal to the input,
#           True: element satisfied the cond, False: otherwise
def where_cpp(x, more_than=None, less_than=None, result=None):
    if result is None:
        result = np.empty_like(x, dtype=bool)
    if more_than is None and less_than is not None:
        __lib.where_less_than(__getPointer(x), x.size,
                              ct.c_double(less_than),
                              __getPointer(result))
    elif more_than is not None and less_than is None:
        __lib.where_more_than(__getPointer(x), x.size,
                              ct.c_double(more_than),
                              __getPointer(result))

    elif more_than is not None and less_than is not None:
        __lib.where_more_less_than(__getPointer(x), x.size,
                                   ct.c_double(more_than),
                                   ct.c_double(less_than),
                                   __getPointer(result))

    else:
        raise RuntimeError(
            '[bmath:where] You need to define at least one of more_than, less_than')
    return result


def add_cpp(a, b, result=None, inplace=False):
    if(len(a) != len(b)):
        raise ValueError(
            'operands could not be broadcast together with shapes ',
            a.shape, b.shape)
    if a.dtype != b.dtype:
        raise TypeError(
            'given arrays not of the same type ', a.dtype(), b.dtype())

    if (result is None) and (not inplace):
        result = np.empty_like(a, order='C')

    if (a.dtype == 'int32'):
        if inplace:
            __lib.add_int_vector_inplace(__getPointer(a), __getPointer(b),
                                         __getLen(a))
        else:
            __lib.add_int_vector(__getPointer(a), __getPointer(b),
                                 __getLen(a), __getPointer(result))
    elif (a.dtype == 'int64'):
        if inplace:
            __lib.add_longint_vector_inplace(__getPointer(a), __getPointer(b),
                                             __getLen(a))
        else:
            __lib.add_longint_vector(__getPointer(a), __getPointer(b),
                                     __getLen(a), __getPointer(result))

    elif (a.dtype == 'float64'):
        if inplace:
            __lib.add_double_vector_inplace(__getPointer(a), __getPointer(b),
                                            __getLen(a))
        else:
            __lib.add_double_vector(__getPointer(a), __getPointer(b),
                                    __getLen(a), __getPointer(result))
    elif (a.dtype == 'float32'):
        if inplace:
            __lib.add_float_vector_inplace(__getPointer(a), __getPointer(b),
                                           __getLen(a))
        else:
            __lib.add_float_vector(__getPointer(a), __getPointer(b),
                                   __getLen(a), __getPointer(result))

    elif (a.dtype == 'uint16'):
        if inplace:
            __lib.add_uint16_vector_inplace(__getPointer(a), __getPointer(b),
                                            __getLen(a))
        else:
            __lib.add_uint16_vector(__getPointer(a), __getPointer(b),
                                    __getLen(a), __getPointer(result))
    elif (a.dtype == 'uint32'):
        if inplace:
            __lib.add_uint32_vector_inplace(__getPointer(a), __getPointer(b),
                                            __getLen(a))
        else:
            __lib.add_uint32_vector(__getPointer(a), __getPointer(b),
                                    __getLen(a), __getPointer(result))

    else:
        raise TypeError('type ', a.dtype, ' is not supported')

    return result


def mul_cpp(a, b, result=None):
    if(type(a) == np.ndarray and type(b) != np.ndarray):
        if result is None:
            result = np.empty_like(a, order='C')

        if (a.dtype == 'int32'):
            __lib.scalar_mul_int32(__getPointer(a), ct.c_int32(np.int32(b)),
                                   __getLen(a), __getPointer(result))
        elif (a.dtype == 'int64'):
            __lib.scalar_mul_int64(__getPointer(a), ct.c_int64(np.int64(b)),
                                   __getLen(a), __getPointer(result))
        elif (a.dtype == 'float32'):
            __lib.scalar_mul_float32(__getPointer(a), ct.c_float(np.float32(b)),
                                     __getLen(a), __getPointer(result))
        elif (a.dtype == 'float64'):
            __lib.scalar_mul_float64(__getPointer(a), ct.c_double(np.float64(b)),
                                     __getLen(a), __getPointer(result))
        elif (a.dtype == 'complex64'):
            __lib.scalar_mul_compex64(__getPointer(a), c_complex64(np.complex64(b)),
                                      __getLen(a), __getPointer(result))
        elif (a.dtype == 'complex128'):
            __lib.scalar_mul_complex128(__getPointer(a), c_complex128(np.complex128(b)),
                                        __getLen(a), __getPointer(result))
        else:
            raise TypeError('type ', a.dtype, ' is not supported')

    elif(type(b) == np.ndarray and type(a) != np.ndarray):
        return mul_cpp(b, a, result)
    elif(type(a) == np.ndarray and type(b) == np.ndarray):
        if result is None:
            result = np.empty_like(a, order='C')

        if (a.dtype == 'int32'):
            __lib.vector_mul_int32(__getPointer(a), __getPointer(b),
                                   __getLen(a), __getPointer(result))
        elif (a.dtype == 'int64'):
            __lib.vector_mul_int64(__getPointer(a), __getPointer(b),
                                   __getLen(a), __getPointer(result))
        elif (a.dtype == 'float32'):
            __lib.vector_mul_float32(__getPointer(a), __getPointer(b),
                                     __getLen(a), __getPointer(result))
        elif (a.dtype == 'float64'):
            __lib.vector_mul_float64(__getPointer(a), __getPointer(b),
                                     __getLen(a), __getPointer(result))
        elif (a.dtype == 'complex64'):
            __lib.vector_mul_compex64(__getPointer(a), __getPointer(b),
                                      __getLen(a), __getPointer(result))
        elif (a.dtype == 'complex128'):
            __lib.vector_mul_complex128(__getPointer(a), __getPointer(b),
                                        __getLen(a), __getPointer(result))
        else:
            raise TypeError('type ', a.dtype, ' is not supported')
    else:
        raise TypeError(
            'types {} and {} are not supported'.format(type(a), type(b)))
    return result


def argmin_cpp(x):
    __lib.min_idx.restype = ct.c_int
    return __lib.min_idx(__getPointer(x), __getLen(x))


def argmax_cpp(x):
    __lib.max_idx.restype = ct.c_int
    return __lib.max_idx(__getPointer(x), __getLen(x))


def linspace_cpp(start, stop, num=50, retstep=False, result=None):
    if result is None:
        result = np.empty(num, dtype=float)
    __lib.linspace(c_real(start), c_real(stop),
                   ct.c_int(num), __getPointer(result))
    if retstep:
        return result, 1. * (stop - start) / (num - 1)
    else:
        return result


def arange_cpp(start, stop, step, dtype=float, result=None):
    size = int(np.ceil((stop - start) / step))
    if result is None:
        result = np.empty(size, dtype=dtype)
    if dtype == float:
        __lib.arange_double(c_real(start), c_real(stop),
                            c_real(step), __getPointer(result))
    elif dtype == int:
        __lib.arange_int(ct.c_int(start), ct.c_int(stop),
                         ct.c_int(step), __getPointer(result))

    return result


def sum_cpp(x):
    __lib.sum.restype = ct.c_double
    return __lib.sum(__getPointer(x), __getLen(x))


def sort_cpp(x, reverse=False):
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


def mean_cpp(x):
    if isinstance(x[0], np.float32):
        __lib.meanf.restype = ct.c_float
        return __lib.meanf(__getPointer(x), __getLen(x))
    elif isinstance(x[0], np.float64):
        __lib.mean.restype = ct.c_double
        return __lib.mean(__getPointer(x), __getLen(x))


def std_cpp(x):
    if isinstance(x[0], np.float32):
        __lib.stdevf.restype = ct.c_float
        return __lib.stdevf(__getPointer(x), __getLen(x))
    elif isinstance(x[0], np.float64):
        __lib.stdev.restype = ct.c_double
        return __lib.stdev(__getPointer(x), __getLen(x))


def sin_cpp(x, result=None):
    if isinstance(x, np.ndarray) and isinstance(x[0], np.float64):
        if result is None:
            result = np.empty(len(x), dtype=np.float64, order='C')
        __lib.fast_sinv(__getPointer(x), __getLen(x), __getPointer(result))
        return result
    elif isinstance(x, np.ndarray) and isinstance(x[0], np.float32):
        if result is None:
            result = np.empty(len(x), dtype=np.float32, order='C')
        __lib.fast_sinvf(__getPointer(x), __getLen(x), __getPointer(result))
        return result
    elif isinstance(x, float) or isinstance(x, np.float32) or isinstance(x, int):
        __lib.fast_sin.restype = ct.c_double
        return __lib.fast_sin(ct.c_double(x))
    else:
        # TypeError
        raise RuntimeError('[sin] The type %s is not supported' % type(x))


def cos_cpp(x, result=None):
    if isinstance(x, np.ndarray) and isinstance(x[0], np.float64):
        if result is None:
            result = np.empty(len(x), dtype=np.float64, order='C')
        __lib.fast_cosv(__getPointer(x), __getLen(x), __getPointer(result))
        return result
    elif isinstance(x, np.ndarray) and isinstance(x[0], np.float32):
        if result is None:
            result = np.empty(len(x), dtype=np.float32, order='C')
        __lib.fast_cosvf(__getPointer(x), __getLen(x), __getPointer(result))
        return result
    elif isinstance(x, float) or isinstance(x, np.float32) or isinstance(x, int):
        __lib.fast_cos.restype = ct.c_double
        return __lib.fast_cos(ct.c_double(x))
    else:
        # TypeError
        raise RuntimeError('[cos] The type %s is not supported' % type(x))


def exp_cpp(x, result=None):
    if isinstance(x, np.ndarray) and isinstance(x[0], np.float64):
        if result is None:
            result = np.empty(len(x), dtype=np.float64, order='C')
        __lib.fast_expv(__getPointer(x), __getLen(x), __getPointer(result))
        return result
    elif isinstance(x, np.ndarray) and isinstance(x[0], np.float32):
        if result is None:
            result = np.empty(len(x), dtype=np.float32, order='C')
        __lib.fast_expvf(__getPointer(x), __getLen(x), __getPointer(result))
        return result
    elif isinstance(x, float) or isinstance(x, np.float32) or isinstance(x, int):
        __lib.fast_exp.restype = ct.c_double
        return __lib.fast_exp(ct.c_double(x))
    else:
        # TypeError
        raise RuntimeError('[exp] The type %s is not supported' % type(x))


def interp_cpp(x, xp, yp, left=None, right=None, result=None):
    x = x.astype(dtype=precision.real_t, order='C', copy=False)
    xp = xp.astype(dtype=precision.real_t, order='C', copy=False)
    yp = yp.astype(dtype=precision.real_t, order='C', copy=False)

    if not left:
        left = yp[0]
    if not right:
        right = yp[-1]
    if result is None:
        result = np.empty(len(x), dtype=precision.real_t, order='C')

    if precision.num == 1:
        __lib.interpf(__getPointer(x), __getLen(x),
                      __getPointer(xp), __getLen(xp),
                      __getPointer(yp),
                      c_real(left),
                      c_real(right),
                      __getPointer(result))
    else:
        __lib.interp(__getPointer(x), __getLen(x),
                     __getPointer(xp), __getLen(xp),
                     __getPointer(yp),
                     c_real(left),
                     c_real(right),
                     __getPointer(result))

    return result


def interp_const_space(x, xp, yp, left=None, right=None, result=None):
    x = x.astype(dtype=precision.real_t, order='C', copy=False)
    xp = xp.astype(dtype=precision.real_t, order='C', copy=False)
    yp = yp.astype(dtype=precision.real_t, order='C', copy=False)

    if not left:
        left = yp[0]
    if not right:
        right = yp[-1]
    if result is None:
        result = np.empty(len(x), dtype=precision.real_t, order='C')

    if precision.num == 1:
        __lib.interp_const_spacef(__getPointer(x), __getLen(x),
                                  __getPointer(xp), __getLen(xp),
                                  __getPointer(yp),
                                  c_real(left),
                                  c_real(right),
                                  __getPointer(result))
    else:
        __lib.interp_const_space(__getPointer(x), __getLen(x),
                                 __getPointer(xp), __getLen(xp),
                                 __getPointer(yp),
                                 c_real(left),
                                 c_real(right),
                                 __getPointer(result))

    return result


def rfft(a, n=0, result=None):
    a = a.astype(dtype=precision.real_t, order='C', copy=False)
    if (n == 0) and (result is None):
        result = np.empty(len(a) // 2 + 1, dtype=precision.complex_t, order='C')
    elif (n != 0) and (result is None):
        result = np.empty(n // 2 + 1, dtype=precision.complex_t, order='C')

    if precision.num == 1:
        __lib.rfftf(__getPointer(a),
                    __getLen(a),
                    __getPointer(result),
                    ct.c_int(int(n)),
                    ct.c_int(int(os.environ.get('OMP_NUM_THREADS', 1))))
    else:
        __lib.rfft(__getPointer(a),
                   __getLen(a),
                   __getPointer(result),
                   ct.c_int(int(n)),
                   ct.c_int(int(os.environ.get('OMP_NUM_THREADS', 1))))

    return result


def irfft(a, n=0, result=None):
    a = a.astype(dtype=precision.complex_t, order='C', copy=False)

    if (n == 0) and (result is None):
        result = np.empty(2 * (len(a) - 1), dtype=precision.real_t, order='C')
    elif (n != 0) and (result is None):
        result = np.empty(n, dtype=precision.real_t, order='C')

    if precision.num == 1:
        __lib.irfftf(__getPointer(a),
                     __getLen(a),
                     __getPointer(result),
                     ct.c_int(int(n)),
                     ct.c_int(int(os.environ.get('OMP_NUM_THREADS', 1))))
    else:
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
        result = np.empty(n // 2 + 1, dtype=precision.real_t)

    if precision.num == 1:
        __lib.rfftfreqf(ct.c_int(n),
                        __getPointer(result),
                        c_real(d))
    else:
        __lib.rfftfreq(ct.c_int(n),
                       __getPointer(result),
                       c_real(d))

    return result


def irfft_packed(signal, fftsize=0, result=None):

    n0 = len(signal[0])
    howmany = len(signal)

    signal = np.ascontiguousarray(np.reshape(
        signal, -1), dtype=precision.complex_t)

    if (fftsize == 0) and (result is None):
        result = np.empty(howmany * 2 * (n0 - 1), dtype=precision.real_t)
    elif (fftsize != 0) and (result is None):
        result = np.empty(howmany * fftsize, dtype=precision.real_t)

    if precision.num == 1:
        __lib.irfft_packedf(__getPointer(signal),
                            ct.c_int(n0),
                            ct.c_int(howmany),
                            __getPointer(result),
                            ct.c_int(int(fftsize)),
                            ct.c_int(int(os.environ.get('OMP_NUM_THREADS', 1))))
    else:
        __lib.irfft_packed(__getPointer(signal),
                           ct.c_int(n0),
                           ct.c_int(howmany),
                           __getPointer(result),
                           ct.c_int(int(fftsize)),
                           ct.c_int(int(os.environ.get('OMP_NUM_THREADS', 1))))

    result = np.reshape(result, (howmany, -1))

    return result


def cumtrapz(y, x=None, dx=1.0, initial=None, result=None):
    if x is not None:
        # IntegrationError
        raise RuntimeError('[cumtrapz] x attribute is not yet supported')
    if initial:
        if result is None:
            result = np.empty(len(y), dtype=float)
        __lib.cumtrapz_w_initial(__getPointer(y),
                                 c_real(dx), c_real(initial),
                                 __getLen(y), __getPointer(result))
    else:
        if result is None:
            result = np.empty(len(y) - 1, dtype=float)
        __lib.cumtrapz_wo_initial(__getPointer(y), c_real(dx),
                                  __getLen(y), __getPointer(result))
    return result


def trapz_cpp(y, x=None, dx=1.0):
    if x is None:
        __lib.trapz_const_delta.restype = ct.c_double
        return __lib.trapz_const_delta(__getPointer(y), c_real(dx),
                                       __getLen(y))
    else:
        __lib.trapz_var_delta.restype = ct.c_double
        return __lib.trapz_var_delta(__getPointer(y), __getPointer(x),
                                     __getLen(y))


def beam_phase(bin_centers, profile, alpha, omegarf, phirf, bin_size):
    bin_centers = bin_centers.astype(dtype=precision.real_t, order='C',
                                     copy=False)
    profile = profile.astype(dtype=precision.real_t, order='C', copy=False)

    if precision.num == 1:
        __lib.beam_phasef.restype = ct.c_float
        coeff = __lib.beam_phasef(__getPointer(bin_centers),
                                  __getPointer(profile),
                                  c_real(alpha),
                                  c_real(omegarf),
                                  c_real(phirf),
                                  c_real(bin_size),
                                  __getLen(profile))

    else:
        __lib.beam_phase.restype = ct.c_double
        coeff = __lib.beam_phase(__getPointer(bin_centers),
                                 __getPointer(profile),
                                 c_real(alpha),
                                 c_real(omegarf),
                                 c_real(phirf),
                                 c_real(bin_size),
                                 __getLen(profile))
    return coeff


def beam_phase_fast(bin_centers, profile, omegarf, phirf, bin_size):
    bin_centers = bin_centers.astype(dtype=precision.real_t, order='C',
                                     copy=False)
    profile = profile.astype(dtype=precision.real_t, order='C', copy=False)

    __lib.beam_phase_fast.restype = ct.c_double
    coeff = __lib.beam_phase_fast(__getPointer(bin_centers),
                                  __getPointer(profile),
                                  c_real(omegarf),
                                  c_real(phirf),
                                  c_real(bin_size),
                                  __getLen(profile))
    return coeff


def rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers):

    bin_centers = bin_centers.astype(
        dtype=precision.real_t, order='C', copy=False)
    voltages = voltages.astype(dtype=precision.real_t, order='C', copy=False)
    omega_rf = omega_rf.astype(dtype=precision.real_t, order='C', copy=False)
    phi_rf = phi_rf.astype(dtype=precision.real_t, order='C', copy=False)

    rf_voltage = np.zeros(len(bin_centers), dtype=precision.real_t, order='C')

    if precision.num == 1:
        __lib.rf_volt_compf(__getPointer(voltages),
                            __getPointer(omega_rf),
                            __getPointer(phi_rf),
                            __getPointer(bin_centers),
                            __getLen(voltages),
                            __getLen(rf_voltage),
                            __getPointer(rf_voltage))
    else:
        __lib.rf_volt_comp(__getPointer(voltages),
                           __getPointer(omega_rf),
                           __getPointer(phi_rf),
                           __getPointer(bin_centers),
                           __getLen(voltages),
                           __getLen(rf_voltage),
                           __getPointer(rf_voltage))

    return rf_voltage


def kick(dt, dE, voltage, omega_rf, phi_rf, charge, n_rf, acceleration_kick):
    assert isinstance(dt[0], precision.real_t)
    assert isinstance(dE[0], precision.real_t)

    # dt = dt.astype(dtype=precision.real_t, order='C', copy=False)
    # dE = dE.astype(dtype=precision.real_t, order='C', copy=False)
    voltage_kick = charge * \
        voltage.astype(dtype=precision.real_t, order='C', copy=False)
    omegarf_kick = omega_rf.astype(
        dtype=precision.real_t, order='C', copy=False)
    phirf_kick = phi_rf.astype(dtype=precision.real_t, order='C', copy=False)
    if precision.num == 1:
        __lib.kickf(__getPointer(dt),
                    __getPointer(dE),
                    ct.c_int(n_rf),
                    __getPointer(voltage_kick),
                    __getPointer(omegarf_kick),
                    __getPointer(phirf_kick),
                    __getLen(dt),
                    c_real(acceleration_kick))
    else:
        __lib.kick(__getPointer(dt),
                   __getPointer(dE),
                   ct.c_int(n_rf),
                   __getPointer(voltage_kick),
                   __getPointer(omegarf_kick),
                   __getPointer(phirf_kick),
                   __getLen(dt),
                   c_real(acceleration_kick))


def drift(dt, dE, solver, t_rev, length_ratio, alpha_order, eta_0,
          eta_1, eta_2, alpha_0, alpha_1, alpha_2, beta, energy):
    assert isinstance(dt[0], precision.real_t)
    assert isinstance(dE[0], precision.real_t)

    # dt = dt.astype(dtype=precision.real_t, order='C', copy=False)
    # dE = dE.astype(dtype=precision.real_t, order='C', copy=False)
    if precision.num == 1:
        __lib.driftf(__getPointer(dt),
                     __getPointer(dE),
                     ct.c_char_p(solver),
                     c_real(t_rev),
                     c_real(length_ratio),
                     c_real(alpha_order),
                     c_real(eta_0),
                     c_real(eta_1),
                     c_real(eta_2),
                     c_real(alpha_0),
                     c_real(alpha_1),
                     c_real(alpha_2),
                     c_real(beta),
                     c_real(energy),
                     __getLen(dt))
    else:
        __lib.drift(__getPointer(dt),
                    __getPointer(dE),
                    ct.c_char_p(solver),
                    c_real(t_rev),
                    c_real(length_ratio),
                    c_real(alpha_order),
                    c_real(eta_0),
                    c_real(eta_1),
                    c_real(eta_2),
                    c_real(alpha_0),
                    c_real(alpha_1),
                    c_real(alpha_2),
                    c_real(beta),
                    c_real(energy),
                    __getLen(dt))


def linear_interp_kick(dt, dE, voltage,
                       bin_centers, charge,
                       acceleration_kick):

    assert isinstance(dt[0], precision.real_t)
    assert isinstance(dE[0], precision.real_t)
    assert isinstance(voltage[0], precision.real_t)
    assert isinstance(bin_centers[0], precision.real_t)

    # dt = dt.astype(dtype=precision.real_t, order='C', copy=False)
    # dE = dE.astype(dtype=precision.real_t, order='C', copy=False)
    # voltage = voltage.astype(dtype=precision.real_t, order='C', copy=False)
    # bin_centers = bin_centers.astype(
    # dtype=precision.real_t, order='C', copy=False)

    if precision.num == 1:
        __lib.linear_interp_kickf(__getPointer(dt),
                                  __getPointer(dE),
                                  __getPointer(voltage),
                                  __getPointer(bin_centers),
                                  c_real(charge),
                                  __getLen(bin_centers),
                                  __getLen(dt),
                                  c_real(acceleration_kick))
    else:
        __lib.linear_interp_kick(__getPointer(dt),
                                 __getPointer(dE),
                                 __getPointer(voltage),
                                 __getPointer(bin_centers),
                                 c_real(charge),
                                 __getLen(bin_centers),
                                 __getLen(dt),
                                 c_real(acceleration_kick))


def linear_interp_kick_n_drift(dt, dE, total_voltage, bin_centers, charge, acc_kick,
                               solver, t_rev, length_ratio, alpha_order, eta_0, eta_1,
                               eta_2, beta, energy):
    assert isinstance(dt[0], precision.real_t)
    assert isinstance(dE[0], precision.real_t)
    assert isinstance(total_voltage[0], precision.real_t)
    assert isinstance(bin_centers[0], precision.real_t)

    # dt = dt.astype(dtype=precision.real_t, order='C', copy=False)
    # dE = dE.astype(dtype=precision.real_t, order='C', copy=False)
    # total_voltage = total_voltage.astype(
    #     dtype=precision.real_t, order='C', copy=False)
    # bin_centers = bin_centers.astype(
    #     dtype=precision.real_t, order='C', copy=False)
    if precision.num == 1:
        __lib.linear_interp_kick_n_driftf(__getPointer(dt),
                                          __getPointer(dE),
                                          __getPointer(total_voltage),
                                          __getPointer(bin_centers),
                                          __getLen(bin_centers),
                                          __getLen(dt),
                                          c_real(acc_kick),
                                          ct.c_char_p(solver),
                                          c_real(t_rev),
                                          c_real(length_ratio),
                                          c_real(alpha_order),
                                          c_real(eta_0),
                                          c_real(eta_1),
                                          c_real(eta_2),
                                          c_real(beta),
                                          c_real(energy),
                                          c_real(charge))
    else:
        __lib.linear_interp_kick_n_drift(__getPointer(dt),
                                         __getPointer(dE),
                                         __getPointer(total_voltage),
                                         __getPointer(bin_centers),
                                         __getLen(bin_centers),
                                         __getLen(dt),
                                         c_real(acc_kick),
                                         ct.c_char_p(solver),
                                         c_real(t_rev),
                                         c_real(length_ratio),
                                         c_real(alpha_order),
                                         c_real(eta_0),
                                         c_real(eta_1),
                                         c_real(eta_2),
                                         c_real(beta),
                                         c_real(energy),
                                         c_real(charge))


def slice_beam(dt, profile, cut_left, cut_right):
    assert isinstance(dt[0], precision.real_t)
    assert isinstance(profile[0], precision.real_t)

    # dt = dt.astype(dtype=precision.real_t, order='C', copy=False)
    # profile = profile.astype(dtype=precision.real_t, order='C', copy=False)

    if precision.num == 1:
        __lib.histogramf(__getPointer(dt),
                         __getPointer(profile),
                         c_real(cut_left),
                         c_real(cut_right),
                         __getLen(profile),
                         __getLen(dt))
    else:
        __lib.histogram(__getPointer(dt),
                        __getPointer(profile),
                        c_real(cut_left),
                        c_real(cut_right),
                        __getLen(profile),
                        __getLen(dt))


def slice_smooth(dt, profile, cut_left, cut_right):
    assert isinstance(dt[0], precision.real_t)
    assert isinstance(profile[0], precision.real_t)

    # dt = dt.astype(dtype=precision.real_t, order='C', copy=False)
    # profile = profile.astype(dtype=precision.real_t, order='C', copy=False)

    if precision.num == 1:
        __lib.smooth_histogramf(__getPointer(dt),
                                __getPointer(profile),
                                c_real(cut_left),
                                c_real(cut_right),
                                __getLen(profile),
                                __getLen(dt))
    else:
        __lib.smooth_histogram(__getPointer(dt),
                               __getPointer(profile),
                               c_real(cut_left),
                               c_real(cut_right),
                               __getLen(profile),
                               __getLen(dt))


def sparse_histogram(dt, profile, cut_left, cut_right, bunch_indexes, n_slices_bucket):
    assert isinstance(dt[0], precision.real_t)
    assert isinstance(profile[0][0], precision.real_t)

    if precision.num == 1:
        __lib.sparse_histogramf(__getPointer(dt),
                                __getPointer(profile),
                                __getPointer(cut_left),
                                __getPointer(cut_right),
                                __getPointer(bunch_indexes),
                                ct.c_int(n_slices_bucket),
                                __getLen(cut_left),
                                __getLen(dt))
    else:
        __lib.sparse_histogram(__getPointer(dt),
                               __getPointer(profile),
                               __getPointer(cut_left),
                               __getPointer(cut_right),
                               __getPointer(bunch_indexes),
                               ct.c_int(n_slices_bucket),
                               __getLen(cut_left),
                               __getLen(dt))


def music_track(dt, dE, induced_voltage, array_parameters,
                alpha, omega_bar,
                const, coeff1, coeff2, coeff3, coeff4):
    assert isinstance(dt[0], precision.real_t)
    assert isinstance(dE[0], precision.real_t)
    assert isinstance(induced_voltage[0], precision.real_t)
    assert isinstance(array_parameters[0], precision.real_t)

    # beam.dt = beam.dt.astype(
    #     dtype=precision.real_t, order='C', copy=False)
    # beam.dE = beam.dE.astype(
    #     dtype=precision.real_t, order='C', copy=False)
    # induced_voltage = induced_voltage.astype(
    #     dtype=precision.real_t, order='C', copy=False)
    # array_parameters = array_parameters.astype(
    #     dtype=precision.real_t, order='C', copy=False)

    if precision.num == 1:
        __lib.music_trackf(__getPointer(dt),
                           __getPointer(dE),
                           __getPointer(induced_voltage),
                           __getPointer(array_parameters),
                           __getLen(dt),
                           c_real(alpha),
                           c_real(omega_bar),
                           c_real(const),
                           c_real(coeff1),
                           c_real(coeff2),
                           c_real(coeff3),
                           c_real(coeff4))
    else:
        __lib.music_track(__getPointer(dt),
                          __getPointer(dE),
                          __getPointer(induced_voltage),
                          __getPointer(array_parameters),
                          __getLen(dt),
                          c_real(alpha),
                          c_real(omega_bar),
                          c_real(const),
                          c_real(coeff1),
                          c_real(coeff2),
                          c_real(coeff3),
                          c_real(coeff4))


def music_track_multiturn(dt, dE, induced_voltage, array_parameters,
                          alpha, omega_bar,
                          const, coeff1, coeff2, coeff3, coeff4):
    assert isinstance(dt[0], precision.real_t)
    assert isinstance(dE[0], precision.real_t)
    assert isinstance(induced_voltage[0], precision.real_t)
    assert isinstance(array_parameters[0], precision.real_t)

    # beam.dt = beam.dt.astype(
    #     dtype=precision.real_t, order='C', copy=False)
    # beam.dE = beam.dE.astype(
    #     dtype=precision.real_t, order='C', copy=False)
    # induced_voltage = induced_voltage.astype(
    #     dtype=precision.real_t, order='C', copy=False)
    # array_parameters = array_parameters.astype(
    #     dtype=precision.real_t, order='C', copy=False)

    if precision.num == 1:
        __lib.music_track_multiturnf(__getPointer(dt),
                                     __getPointer(dE),
                                     __getPointer(induced_voltage),
                                     __getPointer(array_parameters),
                                     __getLen(dt),
                                     c_real(alpha),
                                     c_real(omega_bar),
                                     c_real(const),
                                     c_real(coeff1),
                                     c_real(coeff2),
                                     c_real(coeff3),
                                     c_real(coeff4))
    else:
        __lib.music_track_multiturn(__getPointer(dt),
                                    __getPointer(dE),
                                    __getPointer(induced_voltage),
                                    __getPointer(array_parameters),
                                    __getLen(dt),
                                    c_real(alpha),
                                    c_real(omega_bar),
                                    c_real(const),
                                    c_real(coeff1),
                                    c_real(coeff2),
                                    c_real(coeff3),
                                    c_real(coeff4))


def synchrotron_radiation(dE, U0, n_kicks, tau_z):
    assert isinstance(dE[0], precision.real_t)
    # dE = dE.astype(dtype=precision.real_t, order='C', copy=False)

    if precision.num == 1:
        __lib.synchrotron_radiationf(
            __getPointer(dE),
            c_real(U0 / n_kicks),
            __getLen(dE),
            c_real(tau_z * n_kicks),
            ct.c_int(n_kicks))
    else:
        __lib.synchrotron_radiation(
            __getPointer(dE),
            c_real(U0 / n_kicks),
            __getLen(dE),
            c_real(tau_z * n_kicks),
            ct.c_int(n_kicks))


def synchrotron_radiation_full(dE, U0, n_kicks, tau_z, sigma_dE, energy):
    assert isinstance(dE[0], precision.real_t)

    # dE = dE.astype(dtype=precision.real_t, order='C', copy=False)

    if precision.num == 1:
        __lib.synchrotron_radiation_fullf(
            __getPointer(dE),
            c_real(U0 / n_kicks),
            __getLen(dE),
            c_real(sigma_dE),
            c_real(tau_z * n_kicks),
            c_real(energy),
            ct.c_int(n_kicks))
    else:
        __lib.synchrotron_radiation_full(
            __getPointer(dE),
            c_real(U0 / n_kicks),
            __getLen(dE),
            c_real(sigma_dE),
            c_real(tau_z * n_kicks),
            c_real(energy),
            ct.c_int(n_kicks))


def set_random_seed(seed):
    __lib.set_random_seed(ct.c_int(seed))


def fast_resonator(R_S, Q, frequency_array, frequency_R, impedance=None):
    R_S = R_S.astype(dtype=precision.real_t, order='C', copy=False)
    Q = Q.astype(dtype=precision.real_t, order='C', copy=False)
    frequency_array = frequency_array.astype(
        dtype=precision.real_t, order='C', copy=False)
    frequency_R = frequency_R.astype(
        dtype=precision.real_t, order='C', copy=False)

    realImp = np.zeros(len(frequency_array), dtype=precision.real_t)
    imagImp = np.zeros(len(frequency_array), dtype=precision.real_t)

    if precision.num == 1:
        __lib.fast_resonator_real_imagf(
            __getPointer(realImp),
            __getPointer(imagImp),
            __getPointer(frequency_array),
            __getPointer(R_S),
            __getPointer(Q),
            __getPointer(frequency_R),
            __getLen(R_S),
            __getLen(frequency_array))
    else:
        __lib.fast_resonator_real_imag(
            __getPointer(realImp),
            __getPointer(imagImp),
            __getPointer(frequency_array),
            __getPointer(R_S),
            __getPointer(Q),
            __getPointer(frequency_R),
            __getLen(R_S),
            __getLen(frequency_array))

    impedance = realImp + 1j * imagImp
    return impedance


def distribution_from_tomoscope(dt, dE, probDistr, seed, profLen,
                                cutoff, x0, y0, dtBin, dEBin):

    __lib.generate_distribution(__getPointer(dt),
                                __getPointer(dE),
                                __getPointer(probDistr),
                                ct.c_uint(seed), ct.c_uint(profLen),
                                ct.c_double(cutoff), ct.c_double(x0),
                                ct.c_double(y0), ct.c_double(dtBin),
                                ct.c_double(dEBin), __getLen(dt))


# def mean(x):
#     __lib.mean.restype = ct.c_double
#     return __lib.mean(__getPointer(x), __getLen(x))


# def std(x):
#     __lib.stdev.restype = ct.c_double
#     return __lib.stdev(__getPointer(x), __getLen(x))


# def sin(x, result=None):
#     if isinstance(x, np.ndarray) and isinstance(x[0], np.float64):
#         if result is None:
#             result = np.empty(len(x), dtype=float)
#         __lib.fast_sinv(__getPointer(x), __getLen(x), __getPointer(result))
#         return result
#     elif isinstance(x, float) or isinstance(x, int):
#         __lib.fast_sin.restype = ct.c_double
#         return __lib.fast_sin(ct.c_double(x))
#     else:
#         # TypeError
#         raise RuntimeError('[sin] The type %s is not supported', type(x))


# def cos(x, result=None):
#     if isinstance(x, np.ndarray) and isinstance(x[0], np.float64):
#         if result is None:
#             result = np.empty(len(x), dtype=float)
#         __lib.fast_cosv(__getPointer(x), __getLen(x), __getPointer(result))
#         return result
#     elif isinstance(x, float) or isinstance(x, int):
#         __lib.fast_cos.restype = ct.c_double
#         return __lib.fast_cos(ct.c_double(x))
#     else:
#         # TypeError
#         raise RuntimeError('[cos] The type %s is not supported', type(x))


# def exp(x, result=None):
#     if isinstance(x, np.ndarray) and isinstance(x[0], np.float64):
#         if result is None:
#             result = np.empty(len(x), dtype=float)
#         __lib.fast_expv(__getPointer(x), __getLen(x), __getPointer(result))
#         return result
#     elif isinstance(x, float) or isinstance(x, int):
#         __lib.fast_exp.restype = ct.c_double
#         return __lib.fast_exp(ct.c_double(x))
#     else:
#         # TypeError
#         raise RuntimeError('[exp] The type %s is not supported', type(x))


# def interp(x, xp, yp, left=None, right=None, result=None):
#     if not left:
#         left = yp[0]
#     if not right:
#         right = yp[-1]
#     if result is None:
#         result = np.empty(len(x), dtype=float)
#     __lib.interp(__getPointer(x), __getLen(x),
#                  __getPointer(xp), __getLen(xp),
#                  __getPointer(yp),
#                  ct.c_double(left),
#                  ct.c_double(right),
#                  __getPointer(result))
#     return result


# def interp_const_space(x, xp, yp, left=None, right=None, result=None):
#     if not left:
#         left = yp[0]
#     if not right:
#         right = yp[-1]
#     if result is None:
#         result = np.empty_like(x, order='C')

#     __lib.interp_const_space(__getPointer(x), __getLen(x),
#                              __getPointer(xp), __getLen(xp),
#                              __getPointer(yp),
#                              ct.c_double(left),
#                              ct.c_double(right),
#                              __getPointer(result))
#     return result


# def cumtrapz(y, x=None, dx=1.0, initial=None, result=None):
#     if x is not None:
#         # IntegrationError
#         raise RuntimeError('[cumtrapz] x attribute is not yet supported')
#     if initial:
#         if result is None:
#             result = np.empty(len(y), dtype=float)
#         __lib.cumtrapz_w_initial(__getPointer(y),
#                                  ct.c_double(dx), ct.c_double(initial),
#                                  __getLen(y), __getPointer(result))
#     else:
#         if result is None:
#             result = np.empty(len(y)-1, dtype=float)
#         __lib.cumtrapz_wo_initial(__getPointer(y), ct.c_double(dx),
#                                   __getLen(y), __getPointer(result))
#     return result


# def trapz(y, x=None, dx=1.0):
#     if x is None:
#         __lib.trapz_const_delta.restype = ct.c_double
#         return __lib.trapz_const_delta(__getPointer(y), ct.c_double(dx),
#                                        __getLen(y))
#     else:
#         __lib.trapz_var_delta.restype = ct.c_double
#         return __lib.trapz_var_delta(__getPointer(y), __getPointer(x),
#                                      __getLen(y))


# def argmin(x):
#     __lib.min_idx.restype = ct.c_int
#     return __lib.min_idx(__getPointer(x), __getLen(x))


# def argmax(x):
#     __lib.max_idx.restype = ct.c_int
#     return __lib.max_idx(__getPointer(x), __getLen(x))


# def linspace(start, stop, num=50, retstep=False, result=None):
#     if result is None:
#         result = np.empty(num, dtype=float)
#     __lib.linspace(ct.c_double(start), ct.c_double(stop),
#                    ct.c_int(num), __getPointer(result))
#     if retstep:
#         return result, 1. * (stop-start) / (num-1)
#     else:
#         return result


# def arange(start, stop, step, dtype=float, result=None):
#     size = int(np.ceil((stop-start)/step))
#     if result is None:
#         result = np.empty(size, dtype=dtype)
#     if dtype == float:
#         __lib.arange_double(ct.c_double(start), ct.c_double(stop),
#                             ct.c_double(step), __getPointer(result))
#     elif dtype == int:
#         __lib.arange_int(ct.c_int(start), ct.c_int(stop),
#                          ct.c_int(step), __getPointer(result))

#     return result


# def sum(x):
#     __lib.sum.restype = ct.c_double
#     return __lib.sum(__getPointer(x), __getLen(x))


# def sort(x, reverse=False):
#     if x.dtype == 'int32':
#         __lib.sort_int(__getPointer(x), __getLen(x), ct.c_bool(reverse))
#     elif x.dtype == 'float64':
#         __lib.sort_double(__getPointer(x), __getLen(x), ct.c_bool(reverse))
#     elif x.dtype == 'int64':
#         __lib.sort_longint(__getPointer(x), __getLen(x), ct.c_bool(reverse))
#     else:
#         # SortError
#         raise RuntimeError('[sort] Datatype %s not supported' % x.dtype)
#     return x


# def rfft(a, n=0, result=None):
#     if (n == 0) and (result is None):
#         result = np.empty(len(a)//2 + 1, dtype=np.complex128)
#     elif (n != 0) and (result is None):
#         result = np.empty(n//2 + 1, dtype=np.complex128)

#     __lib.rfft(__getPointer(a),
#                __getLen(a),
#                __getPointer(result),
#                ct.c_int(int(n)),
#                ct.c_int(int(os.environ.get('OMP_NUM_THREADS', 1))))

#     return result


# def irfft(a, n=0, result=None):

#     if (n == 0) and (result is None):
#         result = np.empty(2*(len(a)-1), dtype=np.float64)
#     elif (n != 0) and (result is None):
#         result = np.empty(n, dtype=np.float64)

#     __lib.irfft(__getPointer(a),
#                 __getLen(a),
#                 __getPointer(result),
#                 ct.c_int(int(n)),
#                 ct.c_int(int(os.environ.get('OMP_NUM_THREADS', 1))))
#     return result


# def rfftfreq(n, d=1.0, result=None):
#     if d == 0:
#         raise ZeroDivisionError('d must be non-zero')
#     if result is None:
#         result = np.empty(n//2 + 1, dtype=np.float64)

#     __lib.rfftfreq(ct.c_int(n),
#                    __getPointer(result),
#                    ct.c_double(d))
#     return result


# def irfft_packed(signal, fftsize=0, result=None):

#     n0 = len(signal[0])
#     howmany = len(signal)

#     signal = np.ascontiguousarray(np.reshape(signal, -1))

#     if (fftsize == 0) and (result is None):
#         result = np.empty(howmany * 2*(n0-1), dtype=np.float64)
#     elif (fftsize != 0) and (result is None):
#         result = np.empty(howmany * fftsize, dtype=np.float64)

#     __lib.irfft_packed(__getPointer(signal),
#                        n0,
#                        howmany,
#                        __getPointer(result),
#                        ct.c_int(int(fftsize)),
#                        ct.c_int(int(os.environ.get('OMP_NUM_THREADS', 1))))

#     result = np.reshape(result, (howmany, -1))

#     return result
