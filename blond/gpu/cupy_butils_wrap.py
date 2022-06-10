import os

import numpy as np
import cupy as cp
import pycuda.elementwise as elw
import pycuda.reduction as red
from pycuda.tools import ScalarArg
#from skcuda import fft
import cupyx.scipy.fft as fft

from ..gpu import grid_size, block_size
from ..gpu.cupy_cache import get_gpuarray
from ..utils import bmath as bm

central_mod = bm.getMod()
basedir = os.path.dirname(os.path.realpath(__file__)) + "/cuda_kernels/"


# Since we use compiled versions of functions, we override
# some functions of pycuda to use the cubin we have created


'''def custom_get_elwise_range_module(arguments, operation,
                                   name="kernel", keep=False, options=None,
                                   preamble="", loop_prep="", after_loop=""):
    return central_mod


def custom_get_elwise_no_range_module(arguments, operation,
                                      name="kernel", keep=False, options=None,
                                      preamble="", loop_prep="", after_loop=""):
    return central_mod


def custom_get_elwise_kernel_and_types(arguments, operation,
                                       name="kernel", keep=False, options=None, use_range=False, **kwargs):
    if isinstance(arguments, str):
        from pycuda.tools import parse_c_arg
        arguments = [parse_c_arg(arg) for arg in arguments.split(",")]

    if use_range:
        arguments.extend([
            ScalarArg(np.intp, "start"),
            ScalarArg(np.intp, "stop"),
            ScalarArg(np.intp, "step"),
        ])
    else:
        arguments.append(ScalarArg(np.uintp, "n"))

    if use_range:
        module_builder = custom_get_elwise_range_module
    else:
        module_builder = custom_get_elwise_no_range_module

    mod = module_builder(arguments, operation, name,
                         keep, options, **kwargs)

    func = mod.get_function(name + use_range * "_range")
    func.prepare("".join(arg.struct_char for arg in arguments))

    return mod, func, arguments


elw.get_elwise_range_module = custom_get_elwise_range_module
elw.get_elwise_module = custom_get_elwise_no_range_module
elw.get_elwise_kernel_and_types = custom_get_elwise_kernel_and_types


def get_reduction_module(*args, **kwargs):
    return central_mod


red.get_reduction_module = get_reduction_module'''


ElementwiseKernel = cp.ElementwiseKernel
ReductionKernel = cp.ReductionKernel


cugradient = central_mod.get_function("cugradient")

custom_gpu_trapz = central_mod.get_function("gpu_trapz_custom")


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

'''gpu_trapz = ReductionKernel(reduce_type = bm.precision.real_t, identity="0", reduce_expr="a+b",
                            args=f"{bm.precision.str} *y, {bm.precision.str} x, int32 sz",
                            map_expr="(i<sz-1) ? x*(y[i]+y[i+1])/2.0 : 0.0",
                            name="gpu_trapz")'''


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
    f"raw {bm.precision.str} a, raw {bm.precision.str} b, raw {bm.precision.str} c",
    '',
    "sincos(a[i],&b[i], &c[i])",
    "bm_sin_cos")



# ffts dicts, a cache for the plans of the ffts

plans_dict = {}
inverse_plans_dict = {}

# scale kernels

scale_int = ElementwiseKernel(
    "int32 a",
    "raw int32 b",
    "b[i] /= a ",
    "scale_kernel_int")

scale_double = ElementwiseKernel(
    f"{bm.precision.str} a",
    f"raw {bm.precision.str} b",
    "b[i] /= a ",
    "scale_kernel_double")

scale_float = ElementwiseKernel(
    f"{bm.precision.str} a",
    f"raw {bm.precision.str} b",
    "b[i] /= a ",
    "scale_kernel_float")


def _get_scale_kernel(dtype):
    if dtype == np.float64:
        return scale_double
    elif dtype == np.float32:
        return scale_float
    elif dtype in [np.int, np.int32]:
        return scale_int


#fft._get_scale_kernel = _get_scale_kernel


def find_plan(arr, my_size):
    if my_size not in plans_dict:
        plans_dict[my_size] = fft.get_fft_plan(a=arr, shape=my_size)
    return plans_dict[my_size]


def inverse_find_plan(arr, size):
    if size not in inverse_plans_dict:
        inverse_plans_dict[size] = fft.get_fft_plan(a=arr, shape=size)
    return inverse_plans_dict[size]


def gpu_rfft(dev_a, n=0, result=None, caller_id=None):
    if n == 0 and result is None:
        n = dev_a.size
    elif n != 0 and result is None:
        pass
    if caller_id is None:
        result = cp.empty(n // 2 + 1, bm.precision.complex_t)
    else:
        result = get_gpuarray((n // 2 + 1, bm.precision.complex_t, 0, 'rfft'), zero_fills=False)
    out_size = n // 2 + 1
    in_size = dev_a.size

    if n == in_size:
        dev_in = get_gpuarray((n, bm.precision.real_t, 0, 'rfft'))
        dev_in = dev_a.astype(dev_in.dtype)
    else:
        dev_in = get_gpuarray((n, bm.precision.real_t, 0, 'rfft'), zero_fills=True)
        if n < in_size:
            dev_in = dev_a[:n].astype(dev_in.dtype)
        else:
             dev_in[:in_size] = dev_a.astype(dev_in.dtype)
    plan = find_plan(dev_in, dev_in.shape)
    result = fft.fft(dev_in, plan = plan)
    return result


def gpu_irfft(dev_a, n=0, result=None, caller_id=None):
    if n == 0 and result is None:
        n = 2 * (dev_a.size - 1)
    elif n != 0 and result is None:
        pass

    if caller_id is None:
        result = cp.empty(n, dtype=bm.precision.real_t)
    else:
        key = (n, bm.precision.real_t, caller_id, 'irfft')
        result = get_gpuarray(key)

    out_size = n
    in_size = dev_a.size

    if out_size == 0:
        out_size = 2 * (in_size - 1)
    n = out_size // 2 + 1

    if n == in_size:
        dev_in = dev_a
    else:
        dev_in = get_gpuarray((n, bm.precision.complex_t, 0, 'irfft'), zero_fills=True)
        if n < in_size:
            dev_in = dev_a[:n]
        else:
            dev_in[:in_size] = dev_a

    inverse_plan = inverse_find_plan(dev_in, out_size)
    result = fft.ifft(dev_in, plan = inverse_plan)
    return result


def gpu_rfftfreq(n, d=1.0, result=None):
    factor = 1 / (d * n)
    result = factor * cp.asnumpy(cp.arange(0, n // 2 + 1, dtype=bm.precision.real_t))
    return result


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
        dev_res = get_gpuarray((dev_x.size, bm.precision.real_t, caller_id, 'interp'))
    else:
        dev_res = cp.empty(dev_x.size, bm.precision.real_t)
    cuinterp(args = (dev_x, np.int32(dev_x.size),
             dev_xp, np.int32(dev_xp.size),
             dev_yp, dev_res,
             bm.precision.real_t(left), bm.precision.real_t(right)),
             block=block_size, grid=grid_size)
    return dev_res
