'''
BLonD math and physics core functions

@author Stefan Hegglin, Konstantinos Iliakis, Panagiotis Tsapatsaris, Georgios Typaldos
@date 20.10.2017
'''

# from functools import wraps
import os

import numpy as np
from ..utils import butils_wrap_cpp as _cpp
from ..utils import butils_wrap_python as _py
from . import precision
from . import gpu_dev

# Global variables
__exec_mode = 'single_node'
# Other modes: multi_node

def use_cpp():
    '''
    Replace all python functions by there equivalent in cpp
    '''

    # dictionary storing the CPP versions of the most compute intensive functions #
    CPP_func_dict = {
        'rfft': np.fft.rfft,
        'irfft': np.fft.irfft,
        'rfftfreq': np.fft.rfftfreq,

        'kick': _cpp.kick,
        'rf_volt_comp': _cpp.rf_volt_comp,
        'drift': _cpp.drift,
        'slice': _cpp.slice,
        'slice_smooth': _cpp.slice_smooth,
        'linear_interp_kick': _cpp.linear_interp_kick,
        'synchrotron_radiation': _cpp.synchrotron_radiation,
        'synchrotron_radiation_full': _cpp.synchrotron_radiation_full,
        'music_track': _cpp.music_track,
        'music_track_multiturn': _cpp.music_track_multiturn,
        'fast_resonator': _cpp.fast_resonator,
        'beam_phase': _cpp.beam_phase,
        'beam_phase_fast': _cpp.beam_phase_fast,
        'sparse_histogram': _cpp.sparse_histogram,
        'distribution_from_tomoscope': _cpp.distribution_from_tomoscope,
        'set_random_seed': _cpp.set_random_seed,

        'sin_cpp': _cpp.sin_cpp,
        'cos_cpp': _cpp.cos_cpp,
        'exp_cpp': _cpp.exp_cpp,
        'mean_cpp': _cpp.mean_cpp,
        'std_cpp': _cpp.std_cpp,
        'where_cpp': _cpp.where_cpp,
        'interp_cpp': np.interp,
        # 'interp_cpp': _cpp.interp_cpp,
        # 'interp_const_space': _cpp.interp_const_space,
        'interp_const_space': np.interp,
        'cumtrapz': _cpp.cumtrapz,
        'trapz_cpp': _cpp.trapz_cpp,
        'linspace_cpp': _cpp.linspace_cpp,
        'argmin_cpp': _cpp.argmin_cpp,
        'argmax_cpp': _cpp.argmax_cpp,
        'convolve': _cpp.convolve,
        'arange_cpp': _cpp.arange_cpp,
        'sum_cpp': _cpp.sum_cpp,
        'sort_cpp': _cpp.sort_cpp,
        'add_cpp': _cpp.add_cpp,
        'mul_cpp': _cpp.mul_cpp,

        'device': 'CPU_CPP'
    }

    # add numpy functions in the dictionary
    for fname in dir(np):
        if callable(getattr(np, fname)) and (fname not in CPP_func_dict) \
                and (fname[0] != '_'):
            CPP_func_dict[fname] = getattr(np, fname)

    # add basic numpy modules to dictionary as they are not callable
    CPP_func_dict['random'] = getattr(np, 'random')
    CPP_func_dict['fft'] = getattr(np, 'fft')

    __update_active_dict(CPP_func_dict)


def use_py():
    '''
    Replace all python functions by there equivalent in python
    '''
    # dictionary storing the Python-only versions of the most compute intensive functions #
    PY_func_dict = {
        'rfft': np.fft.rfft,
        'irfft': np.fft.irfft,
        'rfftfreq': np.fft.rfftfreq,

        'kick': _py.kick,
        'rf_volt_comp': _py.rf_volt_comp,
        'drift': _py.drift,
        'slice': _py.slice,
        'slice_smooth': _py.slice_smooth,
        'linear_interp_kick': _py.linear_interp_kick,
        'synchrotron_radiation': _py.synchrotron_radiation,
        'synchrotron_radiation_full': _py.synchrotron_radiation_full,
        'music_track': _py.music_track,
        'music_track_multiturn': _py.music_track_multiturn,
        'fast_resonator': _py.fast_resonator,
        'beam_phase': _py.beam_phase,
        'beam_phase_fast': _py.beam_phase_fast,
        'sparse_histogram': _py.sparse_histogram,
        'distribution_from_tomoscope': _py.distribution_from_tomoscope,
        'set_random_seed': _py.set_random_seed,

        'device': 'CPU_PY'
    }

    # add numpy functions in the dictionary
    for fname in dir(np):
        if callable(getattr(np, fname)) and (fname not in PY_func_dict) \
                and (fname[0] != '_'):

            PY_func_dict[fname] = getattr(np, fname)

    # add basic numpy modules to dictionary as they are not callable
    PY_func_dict['random'] = getattr(np, 'random')
    PY_func_dict['fft'] = getattr(np, 'fft')

    # Update the global functions
    __update_active_dict(PY_func_dict)


def use_cpu():
    '''
    If not library is found, use the python implementations
    '''
    from .. import libblond as __lib
    if __lib is None:
        use_py()
    else:
        use_cpp()


def use_mpi():
    '''
    Replace some bm functions with MPI implementations
    '''
    global __exec_mode

    MPI_func_dict = {}
    globals().update(MPI_func_dict)
    __exec_mode = 'multi_node'


def mpiMode():
    global __exec_mode
    return __exec_mode == 'multi_node'


def use_fftw():
    '''
    Replace the existing rfft and irfft implementations
    with the ones coming from _cpp.
    '''

    FFTW_func_dict = {
        'rfft': _cpp.rfft,
        'irfft': _cpp.irfft,
        'rfftfreq': _cpp.rfftfreq
    }
    globals().update(FFTW_func_dict)


# precision can be single or double
def use_precision(_precision='double'):
    global precision
    precision.set(_precision)
    # utils.precision = utils.PrecisionClass(_precision)
    # precision = PrecisionClass(_precision)
    # Make sure that the precision object in __init__.py is also updated
    # from . import precision as _precision
    # _precision = precision
    # utils.precision = PrecisionClass(_precision)
    # precision = _cpp.precision


def __update_active_dict(new_dict):
    '''
    Update the currently active dictionary. Removes the keys of the currently
    active dictionary from globals() and spills the keys
    from new_dict to globals()
    Args:
        new_dict A dictionary which contents will be spilled to globals()
    '''
    if not hasattr(__update_active_dict, 'active_dict'):
        __update_active_dict.active_dict = new_dict

    # delete all old implementations/references from globals()
    for key in __update_active_dict.active_dict.keys():
        if key in globals():
            del globals()[key]
    # for key in globals().keys():
    #     if key in __update_active_dict.active_dict.keys():
    #         del globals()[key]
    # add the new active dict to the globals()
    globals().update(new_dict)
    __update_active_dict.active_dict = new_dict


# GPU Related Utilities
def gpuDev():
    return gpu_dev


def use_gpu(gpu_id=0):

    if gpu_id < 0:
        return

    global gpu_dev
    from . import GPUDev
    if gpu_dev is None:
        gpu_dev = GPUDev(gpu_id)

        print(''.join(['#']*10) +
              ' Using GPU: id {}, name {}, Compute Capability {} '.format(
            gpu_dev.id, gpu_dev.name, gpu_dev.dev.compute_capability)
            + ''.join(['#']*10) + '\n', flush=True)

    from ..gpu import butils_wrap_cupy as _cupy
    import cupy as cp

    GPU_func_dict = {
        'rfft': cp.fft.rfft,
        'irfft': cp.fft.irfft,
        'rfftfreq': cp.fft.rfftfreq,
        'convolve': cp.convolve,
        # 'convolve': _cupy.convolve,
        'beam_phase': _cupy.beam_phase,
        'beam_phase_fast': _cupy.beam_phase_fast,
        'kick': _cupy.kick,
        'rf_volt_comp': _cupy.rf_volt_comp,
        'drift': _cupy.drift,
        'linear_interp_kick': _cupy.linear_interp_kick,
        'LIKick_n_drift': _cupy.linear_interp_kick_drift,
        'synchrotron_radiation': _cupy.synchrotron_radiation,
        'synchrotron_radiation_full': _cupy.synchrotron_radiation_full,
        'slice': _cupy.slice,
        # 'interp_const_space': _cupy.interp,
        'interp_const_space': cp.interp,
        'device': 'GPU'
    }
    # add cupy functions in the dictionary
    for fname in dir(cp):
        if callable(getattr(cp, fname)) and (fname not in GPU_func_dict):
            GPU_func_dict[fname] = getattr(cp, fname)
    __update_active_dict(GPU_func_dict)

    # add basic cupy modules to dictionary as they are not callable
    GPU_func_dict['random'] = getattr(cp, 'random')
    GPU_func_dict['fft'] = getattr(cp, 'fft')


###############################################################################
# By default use the CPU backend (python-only or C++)
use_cpu()
###############################################################################
