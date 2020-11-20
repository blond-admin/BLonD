import os

import numpy as np
import pycuda.elementwise as elw
import pycuda.reduction as red
from pycuda import gpuarray
from pycuda.tools import ScalarArg
from skcuda import fft

from ..gpu import grid_size, block_size
from ..gpu.gpu_cache import get_gpuarray
from ..utils import bmath as bm

central_mod = bm.getMod()
basedir = os.path.dirname(os.path.realpath(__file__)) + "/cuda_kernels/"


# Since we use compiled versions of functions, we override
# some functions of pycuda to use the cubin we have created


def custom_get_elwise_range_module(arguments, operation,
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


red.get_reduction_module = get_reduction_module


ElementwiseKernel = elw.ElementwiseKernel
ReductionKernel = red.ReductionKernel

gpu_copy_i2d = ElementwiseKernel(
    f"{bm.precision.str} *x, int *y",
    f"x[i] = ({bm.precision.str}) y[i]*1.0",
    "gpu_copy_i2d")

gpu_copy_d2d = ElementwiseKernel(
    f"{bm.precision.str} *x,{bm.precision.str} *y",
    "x[i] = y[i]",
    "gpu_copy_d2d")

gpu_complex_copy = ElementwiseKernel(
    f"pycuda::complex<{bm.precision.str}> *x, pycuda::complex<{bm.precision.str}> *y",
    "x[i] = y[i]",
    "gpu_complex_copy",
    preamble="#include <pycuda-complex.hpp>")

stdKernel = ReductionKernel(bm.precision.real_t, neutral="0",
                            reduce_expr="a+b", map_expr="(y[i]!=0)*(x[i]-m)*(x[i]-m)",
                            arguments=f"{bm.precision.str} *x, {bm.precision.str} *y, {bm.precision.str} m",
                            name="stdKernel")

sum_non_zeros = ReductionKernel(bm.precision.real_t, neutral="0",
                                reduce_expr="a+b", map_expr="(x[i]!=0)",
                                arguments=f"{bm.precision.str} *x",
                                name="sum_non_zeros")

mean_non_zeros = ReductionKernel(bm.precision.real_t, neutral="0",
                                 reduce_expr="a+b", map_expr="(id[i]!=0)*x[i]",
                                 arguments=f"{bm.precision.str} *x, {bm.precision.str} *id",
                                 name="mean_non_zeros")

cugradient = central_mod.get_function("cugradient")

custom_gpu_trapz = central_mod.get_function("gpu_trapz_custom")

gpu_diff = ElementwiseKernel(f"int *a, {bm.precision.str} *b, {bm.precision.str} c",
                             "b[i] = (a[i+1]-a[i])/c", "gpu_diff")

set_zero_float = ElementwiseKernel(
    f"float *x",
    "x[i] = 0",
    "set_zero_float")

set_zero_double = ElementwiseKernel(
    f"double *x",
    "x[i] = 0",
    "set_zero_double")

if bm.precision.num == 1:
    set_zero_real = set_zero_float
else:
    set_zero_real = set_zero_double

set_zero_int = ElementwiseKernel(
    "int *x",
    "x[i] = 0",
    "set_zero_int")

set_zero_complex64 = ElementwiseKernel(
    f"pycuda::complex<float> *x",
    "x[i] = 0",
    "set_zero_complex",
    preamble="#include <pycuda-complex.hpp>")

set_zero_complex128 = ElementwiseKernel(
    f"pycuda::complex<double> *x",
    "x[i] = 0",
    "set_zero_complex128",
    preamble="#include <pycuda-complex.hpp>")

increase_by_value = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} a",
    "x[i] += a",
    "increase_by_value")

add_array = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y",
    "x[i] += y[i]",
    "add_array")

complex_mul = ElementwiseKernel(
    f"pycuda::complex<{bm.precision.str}> *x, pycuda::complex<{bm.precision.str}> *y, pycuda::complex<{bm.precision.str}> *z",
    "z[i] = x[i] * y[i]",
    "complex_mul",
    preamble="#include <pycuda-complex.hpp>")

gpu_mul = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y, {bm.precision.str} a",
    "x[i] = a*y[i]",
    "gpu_mul")

# beam_feedback
gpu_copy_one = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y, int ind",
    "x[i] = y[ind]",
    "gpu_copy_one")

triple_kernel = central_mod.get_function("gpu_beam_fb_track_other")

first_kernel_x = ElementwiseKernel(
    f"{bm.precision.str} *omega_rf, {bm.precision.str} *harmonic,  {bm.precision.str} domega_rf, int size, int counter",
    "omega_rf[i*size +counter] += domega_rf * harmonic[i*size + counter] / harmonic[counter]",
    "first_kernel_x")

second_kernel_x = ElementwiseKernel(
    f"{bm.precision.str} *dphi_rf, {bm.precision.str} *harmonic, {bm.precision.str} *omega_rf, {bm.precision.str} *omega_rf_d, int size, int counter, {bm.precision.str} pi",
    "dphi_rf[i] +=  2.0*pi*harmonic[size*i+counter]*(omega_rf[size*i+counter]-omega_rf_d[size*i+counter])/omega_rf_d[size*i+counter]",
    "second_kernel_x")

third_kernel_x = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y, int size_0, int counter",
    "x[i*size_0 + counter] += y[i]",
    "third_kernel_x")

indexing_double = ElementwiseKernel(
    f"{bm.precision.str} *out, {bm.precision.str} *in, int *ind",
    "out[i] = in[ind[i]]",
    "indexing_double")

indexing_int = ElementwiseKernel(
    f"{bm.precision.str} *out, int *in, int *ind",
    "out[i] = in[ind[i]]",
    "indexing_int")

sincos_mul_add = ElementwiseKernel(
    f"{bm.precision.str} *ar, {bm.precision.str} a, {bm.precision.str} b, {bm.precision.str} *s, {bm.precision.str} *c",
    "sincos(a*ar[i]+b, &s[i], &c[i])",
    "sincos_mul_add")

sincos_mul_add_2 = ElementwiseKernel(
    f"{bm.precision.str} *ar, {bm.precision.str} a, {bm.precision.str} b, {bm.precision.str} *s, {bm.precision.str} *c",
    "s[i] = cos(a*ar[i]+b-pi/2); c[i] = cos(a*ar[i]+b)",
    "sincos_mul_add_2")

gpu_trapz = ReductionKernel(bm.precision.real_t, neutral="0", reduce_expr="a+b",
                            arguments=f"{bm.precision.str} *y, {bm.precision.str} x, int sz",
                            map_expr="(i<sz-1) ? x*(y[i]+y[i+1])/2.0 : 0.0",
                            name="gpu_trapz")

mul_d = ElementwiseKernel(
    f"{bm.precision.str} *a1, {bm.precision.str} *a2",
    "a1[i] *= a2[i]",
    "mul_d")

# The following functions are being used for the methods of the tracker

add_kernel = ElementwiseKernel(
    f"{bm.precision.str} *a, {bm.precision.str} *b, {bm.precision.str} *c",
    "a[i]=b[i]+c[i]",
    "add_kernel")

first_kernel_tracker = ElementwiseKernel(
    f"{bm.precision.str} *phi_rf, {bm.precision.str} x, {bm.precision.str} *phi_noise, int len, int turn",
    "phi_rf[len*i + turn] += x * phi_noise[len*i + turn]",
    "first_kernel_tracker")

second_kernel_tracker = ElementwiseKernel(
    f"{bm.precision.str} *phi_rf, {bm.precision.str} *omega_rf, {bm.precision.str} *phi_mod0, {bm.precision.str} *phi_mod1, int size, int turn",
    "phi_rf[i*size+turn] += phi_mod0[i*size+turn]; omega_rf[i*size+turn] += phi_mod1[i*size+turn]",
    "second_kernel_tracker")

copy_column = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y, int size, int column",
    "x[i] = y[i*size + column]",
    "copy_column")

rf_voltage_calculation_kernel = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y, int size, int column",
    "x[i] = y[i*size + column]",
    "rf_voltage_calculation_kernel")

cavityFB_case = ElementwiseKernel(
    f"{bm.precision.str} *rf_voltage, {bm.precision.str} *voltage, {bm.precision.str} *omega_rf, {bm.precision.str} *phi_rf," +
    f"{bm.precision.str} *bin_centers, {bm.precision.str} V_corr, {bm.precision.str} phi_corr," +
    "int size, int column",
    "rf_voltage[i] = voltage[0] * V_corr * sin(omega_rf[0] * bin_centers[i]+phi_rf[0]+phi_corr)",
    "cavityFB_case")

gpu_rf_voltage_calc_mem_ops = central_mod.get_function("gpu_rf_voltage_calc_mem_ops")

cuinterp = central_mod.get_function("cuinterp")

# The following methods are being used during the beam_phase


bm_phase_exp_times_scalar = ElementwiseKernel(
    f"{bm.precision.str} *a, {bm.precision.str} *b, {bm.precision.str} c, int *d",
    "a[i] = exp(c*b[i])*d[i]",
    "bm_phase_exp_times_scalar")

bm_phase_mul_add = ElementwiseKernel(
    f"{bm.precision.str} *a, {bm.precision.str} b, {bm.precision.str} *c, {bm.precision.str} d",
    "a[i] = b*c[i] + d",
    "bm_phase_mul_add")

bm_sin_cos = ElementwiseKernel(
    f"{bm.precision.str} *a, {bm.precision.str} *b, {bm.precision.str} *c",
    "sincos(a[i],&b[i], &c[i])",
    "bm_sin_cos")

d_multiply = ElementwiseKernel(
    f"{bm.precision.str} *a, {bm.precision.str} *b",
    "a[i] *= b[i]",
    "d_multiply")

d_multscalar = ElementwiseKernel(
    f"{bm.precision.str} *a, {bm.precision.str} *b, {bm.precision.str} c",
    "a[i] = c*b[i]",
    "d_multscalar")

# ffts dicts, a cache for the plans of the ffts

plans_dict = {}
inverse_plans_dict = {}

# scale kernels

scale_int = ElementwiseKernel(
    "int a, int *b",
    "b[i] /= a ",
    "scale_kernel_int")

scale_double = ElementwiseKernel(
    f"{bm.precision.str} a, {bm.precision.str} *b",
    "b[i] /= a ",
    "scale_kernel_double")

scale_float = ElementwiseKernel(
    f"{bm.precision.str} a, {bm.precision.str} *b",
    "b[i] /= a ",
    "scale_kernel_float")


def _get_scale_kernel(dtype):
    if dtype == np.float64:
        return scale_double
    elif dtype == np.float32:
        return scale_float
    elif dtype in [np.int, np.int32]:
        return scale_int


fft._get_scale_kernel = _get_scale_kernel


def find_plan(my_size):
    if my_size not in plans_dict:
        plans_dict[my_size] = fft.Plan(my_size, bm.precision.real_t, bm.precision.complex_t)
    return plans_dict[my_size]


def inverse_find_plan(size):
    if size not in inverse_plans_dict:
        inverse_plans_dict[size] = fft.Plan(
            size, in_dtype=bm.precision.complex_t, out_dtype=bm.precision.real_t)
    return inverse_plans_dict[size]


def gpu_rfft(dev_a, n=0, result=None, caller_id=None):
    if n == 0 and result == None:
        n = dev_a.size
    elif (n != 0) and (result == None):
        pass
    if caller_id is None:
        result = gpuarray.empty(n // 2 + 1, bm.precision.complex_t)
    else:
        result = get_gpuarray((n // 2 + 1, bm.precision.complex_t, 0, 'rfft'), zero_fills=True)
    out_size = n // 2 + 1
    in_size = dev_a.size

    if dev_a.dtype == np.int32:
        gpu_copy = gpu_copy_i2d
    else:
        gpu_copy = gpu_copy_d2d

    if n == in_size:
        dev_in = get_gpuarray((n, bm.precision.real_t, 0, 'rfft'))
        gpu_copy(dev_in, dev_a, slice=slice(0, n))
    else:
        dev_in = get_gpuarray((n, bm.precision.real_t, 0, 'rfft'), zero_fills=True)
        if n < in_size:
            gpu_copy(dev_in, dev_a, slice=slice(0, n))
        else:
            gpu_copy(dev_in, dev_a, slice=slice(0, in_size))
    plan = find_plan(dev_in.shape)
    fft.fft(dev_in, result, plan)
    return result


def gpu_irfft(dev_a, n=0, result=None, caller_id=None):
    if (n == 0) and (result == None):
        n = 2 * (dev_a.size - 1)
    elif (n != 0) and (result == None):
        pass

    if caller_id is None:
        result = gpuarray.empty(n, dtype=bm.precision.real_t)
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
        dev_in = get_gpuarray((n, bm.precision.complex_t, 0, 'irfft'))
        if n < in_size:
            gpu_complex_copy(dev_in, dev_a, slice=slice(0, n))
        else:
            gpu_complex_copy(dev_in, dev_a, slice=slice(0, n))

    inverse_plan = inverse_find_plan(out_size)
    fft.ifft(dev_in, result, inverse_plan, scale=True)
    return result


def gpu_rfftfreq(n, d=1.0, result=None):
    factor = 1 / (d * n)
    result = factor * gpuarray.arange(0, n // 2 + 1, dtype=bm.precision.real_t).get()
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
        dev_res = gpuarray.empty(dev_x.size, bm.precision.real_t)
    cuinterp(dev_x, np.int32(dev_x.size),
             dev_xp, np.int32(dev_xp.size),
             dev_yp, dev_res,
             bm.precision.real_t(left), bm.precision.real_t(right),
             block=block_size, grid=grid_size)
    return dev_res
