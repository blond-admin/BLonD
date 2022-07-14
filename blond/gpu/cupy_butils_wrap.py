import os

import numpy as np
import cupy as cp
import cupy.fft as fft

from ..gpu import grid_size, block_size
from ..gpu.cupy_array import get_gpuarray
from ..utils import bmath as bm

central_mod = bm.gpuDev().mod
basedir = os.path.dirname(os.path.realpath(__file__)) + "/cuda_kernels/"


ElementwiseKernel = cp.ElementwiseKernel
ReductionKernel = cp.ReductionKernel


cugradient = central_mod.get_function("cugradient")

# custom_gpu_trapz = central_mod.get_function("gpu_trapz_custom")


# beam_feedback

triple_kernel = central_mod.get_function("gpu_beam_fb_track_other")

first_kernel_x = ElementwiseKernel(
    f"raw {bm.precision.str} harmonic,  {bm.precision.str} domega_rf, int32 size, int32 counter",
    f"raw {bm.precision.str} omega_rf",
    "omega_rf[i*size +counter] += domega_rf * harmonic[i*size + counter] / harmonic[counter]",
    "first_kernel_x")

second_kernel_x = ElementwiseKernel(
    f"raw {bm.precision.str} *harmonic, raw {bm.precision.str} omega_rf, raw {bm.precision.str} omega_rf_d, int32 size, int32 counter, {bm.precision.str} pi",
    f"raw {bm.precision.str} dphi_rf",
    "dphi_rf[i] +=  2.0*pi*harmonic[size*i+counter]*(omega_rf[size*i+counter]-omega_rf_d[size*i+counter])/omega_rf_d[size*i+counter]",
    "second_kernel_x")

third_kernel_x = ElementwiseKernel(
    f"raw {bm.precision.str} y, int32 size_0, int32 counter",
    f"raw {bm.precision.str} x",
    "x[i*size_0 + counter] += y[i]",
    "third_kernel_x")

indexing_double = ElementwiseKernel(
    f"raw {bm.precision.str} in, int32 *ind",
    f"raw {bm.precision.str} out",
    "out[i] = in[ind[i]]",
    "indexing_double")

indexing_int = ElementwiseKernel(
    f"raw int32 in, raw int32 ind",
    f"raw {bm.precision.str} out",
    "out[i] = in[ind[i]]",
    "indexing_int")

sincos_mul_add = ElementwiseKernel(
    f"raw {bm.precision.str} ar, {bm.precision.str} a, {bm.precision.str} b, raw {bm.precision.str} s, raw {bm.precision.str} c",
    '',
    "sincos(a*ar[i]+b, &s[i], &c[i])",
    "sincos_mul_add")

sincos_mul_add_2 = ElementwiseKernel(
    f"raw {bm.precision.str} ar, {bm.precision.str} a, {bm.precision.str} b, raw {bm.precision.str} c",
    f"raw {bm.precision.str} s",
    "s[i] = cos(a*ar[i]+b-pi/2); c[i] = cos(a*ar[i]+b)",
    "sincos_mul_add_2")

# gpu_trapz = ReductionKernel(
#     in_params = f"raw {bm.precision.str} *y, {bm.precision.str} x, int32 sz",
#     out_params = f"{bm.precision.str} z",
#     map_expr = "(i<sz-1) ? x*(y[i]+y[i+1])/2.0 : 0.0",
#     reduce_expr = "a+b",
#     post_map_expr = "z = a",
#     identity="0",
#     name="gpu_trapz"
# )


# The following functions are being used for the methods of the tracker

def first_kernel_tracker(phi_rf, x, phi_noise, length, turn, limit = None):
    index = len(phi_rf) if limit is None else limit
    for i in range(index):
        phi_rf[length*i + turn] += x * phi_noise[length*i + turn]

def second_kernel_tracker(phi_rf, omega_rf, phi_mod0, phi_mod1, size, turn, limit = None):
    index = len(phi_rf) if limit is None else limit
    for i in range(index):
        phi_rf[i*size+turn] += phi_mod0[i*size+turn] 
        omega_rf[i*size+turn] += phi_mod1[i*size+turn]

def copy_column(x, y, size, column):
    for i in range(len(x)):
        x[i] = y[i*size + column]


rf_voltage_calculation_kernel = copy_column


def cavityFB_case(rf_voltage, voltage, omega_rf, phi_rf, bin_centers, 
                    V_corr, phi_corr, size, column):
    from cupy import sin 
    for i in range(len(rf_voltage)):
        rf_voltage[i] = voltage[0] * V_corr * sin(omega_rf[0] * bin_centers[i]+phi_rf[0]+phi_corr)


gpu_rf_voltage_calc_mem_ops = central_mod.get_function("gpu_rf_voltage_calc_mem_ops")

cuinterp = central_mod.get_function("cuinterp")

# The following methods are being used during the beam_phase


bm_phase_exp_times_scalar = ElementwiseKernel(
    f"raw {bm.precision.str} b, {bm.precision.str} c, raw int32 d",
    f"raw {bm.precision.str} a",
    "a[i] = exp(c*b[i])*d[i]",
    "bm_phase_exp_times_scalar")

bm_sin_cos = ElementwiseKernel(
    f"raw {bm.precision.str} b, raw {bm.precision.str} c",
    f"raw {bm.precision.str} a",
    "sincos(a[i],&b[i], &c[i])",
    "bm_sin_cos")



# ffts dicts, a cache for the plans of the ffts

plans_dict = {}
inverse_plans_dict = {}


def gpu_rfft(dev_a, n=0, result=None, caller_id=None):
    if n == 0 and result is None:
        n = dev_a.size
    elif n != 0 and result is None:
        pass
    if caller_id is None:
        result = cp.empty(n // 2 + 1, bm.precision.complex_t)
    else:
        result = get_gpuarray(n // 2 + 1, bm.precision.complex_t)
    out_size = n // 2 + 1
    in_size = dev_a.size

    if n == in_size:
        dev_in = get_gpuarray(n, bm.precision.real_t)
        dev_in = dev_a.astype(dev_in.dtype)
    else:
        dev_in = get_gpuarray(n, bm.precision.real_t, zero_fills=True)
        if n < in_size:
            dev_in = dev_a[:n].astype(dev_in.dtype)
        else:
             dev_in[:in_size] = dev_a.astype(dev_in.dtype)
    result = fft.rfft(dev_in)
    return result


def gpu_irfft(dev_a, n=0, result=None, caller_id=None):
    if n == 0 and result is None:
        n = 2 * (dev_a.size - 1)
    elif n != 0 and result is None:
        pass

    if caller_id is None:
        result = cp.empty(n, dtype=bm.precision.real_t)
    else:
        result = get_gpuarray(n, bm.precision.real_t)

    out_size = n
    in_size = dev_a.size

    if out_size == 0:
        out_size = 2 * (in_size - 1)
    n = out_size // 2 + 1

    if n == in_size:
        dev_in = dev_a
    else:
        dev_in = get_gpuarray(n, bm.precision.complex_t, zero_fills=True)
        if n < in_size:
            dev_in = dev_a[:n]
        else:
            dev_in[:in_size] = dev_a

    result = fft.irfft(dev_in)
    return result


# def gpu_rfftfreq(n, d=1.0, result=None):
#     factor = 1 / (d * n)
#     result = factor * cp.asnumpy(cp.arange(0, n // 2 + 1, dtype=bm.precision.real_t))
#     return result


def gpu_convolve(signal, kernel, mode='full', result=None):
    if mode != 'full':
        # ConvolutionError
        raise RuntimeError('[convolve] Only full mode is supported')
    if result is None:
        result = np.empty(len(signal) + len(kernel) - 1, dtype=bm.precision.real_t)
    real_size = len(signal) + len(kernel) - 1
    complex_size = real_size // 2 + 1
    result1 = np.empty(complex_size, dtype=bm.precision.complex_t)
    result2 = np.empty(complex_size, dtype=bm.precision.complex_t)
    result1 = gpu_rfft(signal, result=result1)
    result2 = gpu_rfft(kernel, result=result2)
    result2 = result1 * result2
    result = gpu_irfft(result2.get(), result=result).get()
    return result


def gpu_interp(dev_x, dev_xp, dev_yp, left=0.12345, right=0.12345, caller_id=None):
    if caller_id is None:
        dev_res = get_gpuarray(dev_x.size, bm.precision.real_t)
    else:
        dev_res = cp.empty(dev_x.size, bm.precision.real_t)
    cuinterp(args = (dev_x, np.int32(dev_x.size),
             dev_xp, np.int32(dev_xp.size),
             dev_yp, dev_res,
             bm.precision.real_t(left), bm.precision.real_t(right)),
             block=block_size, grid=grid_size)
    return dev_res
