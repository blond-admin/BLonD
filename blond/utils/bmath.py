'''
BLonD math and physics core functions

@author Stefan Hegglin, Konstantinos Iliakis, Panagiotis Tsapatsaris, Georgios Typaldos
@date 20.10.2017
'''

import numpy as np

from ..utils import butils_wrap_cpp as _cpp
from ..utils import butils_wrap_python as _py
from . import precision


def use_cpp():
    '''
    Replace all python functions by there equivalent in cpp
    '''
    print('---------- Using the C++ computational backend ----------')
    # dictionary storing the CPP versions of the most compute intensive functions #
    cpp_func_dict = {
        'rfft': np.fft.rfft,
        'irfft': np.fft.irfft,
        'rfftfreq': np.fft.rfftfreq,

        'kick': _cpp.kick,
        'rf_volt_comp': _cpp.rf_volt_comp,
        'drift': _cpp.drift,
        'slice_beam': _cpp.slice_beam,
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
        'interp_cpp': _cpp.interp_cpp,
        'interp_const_space': _cpp.interp_const_space,
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
        if callable(getattr(np, fname)) and (fname not in cpp_func_dict) \
                and (fname[0] != '_'):
            cpp_func_dict[fname] = getattr(np, fname)

    # add basic numpy modules to dictionary as they are not callable
    cpp_func_dict['random'] = getattr(np, 'random')
    cpp_func_dict['fft'] = getattr(np, 'fft')

    __update_active_dict(cpp_func_dict)


def use_py():
    '''
    Replace all python functions by there equivalent in python
    '''
    print('---------- Using the Python computational backend ----------')

    # dictionary storing the Python-only versions of the most compute intensive functions #
    py_func_dict = {
        'rfft': np.fft.rfft,
        'irfft': np.fft.irfft,
        'rfftfreq': np.fft.rfftfreq,

        'kick': _py.kick,
        'rf_volt_comp': _py.rf_volt_comp,
        'drift': _py.drift,
        'slice_beam': _py.slice_beam,
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
        if callable(getattr(np, fname)) and (fname not in py_func_dict) \
                and (fname[0] != '_'):

            py_func_dict[fname] = getattr(np, fname)

    # add basic numpy modules to dictionary as they are not callable
    py_func_dict['random'] = getattr(np, 'random')
    py_func_dict['fft'] = getattr(np, 'fft')

    # Update the global functions
    __update_active_dict(py_func_dict)


def use_cpu():
    '''
    If not library is found, use the python implementations
    '''
    from .. import LIBBLOND as __lib
    if __lib is None:
        use_py()
    else:
        use_cpp()


def use_mpi():
    '''
    Replace some bm functions with MPI implementations
    '''

    mpi_func_dict = {
        'device': 'CPU_MPI'
    }
    globals().update(mpi_func_dict)


def in_mpi():
    """Check if we are currently in MPI mode

    Returns:
        bool: True if in MPI mode
    """
    return globals()['device'] == 'CPU_MPI'


def use_fftw():
    '''
    Replace the existing rfft and irfft implementations
    with the ones coming from _cpp.
    '''
    print('---------- Using the FFTW FFT library ----------')

    fftw_func_dict = {
        'rfft': _cpp.rfft,
        'irfft': _cpp.irfft,
        'rfftfreq': _cpp.rfftfreq
    }
    globals().update(fftw_func_dict)


# precision can be single or double
def use_precision(_precision='double'):
    """Change the precision used in caclulations.

    Args:
        _precision (str, optional): Can be either 'single' or 'double'. Defaults to 'double'.
    """
    print(f'---------- Using {_precision} precision numeric datatypes ----------')
    precision.set(_precision)


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
def get_gpu_device():
    """Get the GPU device object

    Returns:
        _type_: _description_
    """
    from ..gpu import GPU_DEV
    return GPU_DEV


def use_gpu(gpu_id=0):
    """Use the GPU device to perform the calculations.

    Args:
        gpu_id (int, optional): The device id. Defaults to 0.
    """
    if gpu_id < 0:
        return

    from ..gpu import GPU_DEV

    GPU_DEV.set(gpu_id)

    import cupy as cp

    from ..gpu import butils_wrap_cupy as _cupy

    gpu_func_dict = {
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
        # 'LIKick_n_drift': _cupy.linear_interp_kick_drift,
        'synchrotron_radiation': _cupy.synchrotron_radiation,
        'synchrotron_radiation_full': _cupy.synchrotron_radiation_full,
        'slice_beam': _cupy.slice_beam,
        # 'interp_const_space': _cupy.interp,
        'interp_const_space': cp.interp,
        'device': 'GPU'
    }
    # add cupy functions in the dictionary
    for fname in dir(cp):
        if callable(getattr(cp, fname)) and (fname not in gpu_func_dict):
            gpu_func_dict[fname] = getattr(cp, fname)
    __update_active_dict(gpu_func_dict)

    # add basic cupy modules to dictionary as they are not callable
    gpu_func_dict['random'] = getattr(cp, 'random')
    gpu_func_dict['fft'] = getattr(cp, 'fft')

    print('---------- Using the GPU computational backend ----------')
    print(f'---------- GPU Device: id {GPU_DEV.id}, name {GPU_DEV.name}, Compute Capability {GPU_DEV.dev.compute_capability} ----------',
           flush=True)


###############################################################################
# By default use the CPU backend (python-only or C++)
use_cpu()
###############################################################################
