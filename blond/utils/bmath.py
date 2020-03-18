'''
BLonD math and physics core functions

@author Stefan Hegglin, Konstantinos Iliakis
@date 20.10.2017
'''
# from functools import wraps
import numpy as np
from ..utils import butils_wrap
from ..utils import bphysics_wrap
from numpy import fft

__exec_mode = 'single_node'
# Other modes: multi_node

# dictionary storing the CPU versions of the desired functions #
_CPU_func_dict = {
    'rfft': fft.rfft,
    'irfft': fft.irfft,
    'rfftfreq': fft.rfftfreq,
    'irfft_packed': butils_wrap.irfft_packed,
    'sin': butils_wrap.sin,
    'cos': butils_wrap.cos,
    'exp': butils_wrap.exp,
    'mean': butils_wrap.mean,
    'std': butils_wrap.std,
    'where': butils_wrap.where,
    'interp': butils_wrap.interp,
    'interp_const_space': butils_wrap.interp_const_space,
    'cumtrapz': butils_wrap.cumtrapz,
    'trapz': butils_wrap.trapz,
    'linspace': butils_wrap.linspace,
    'argmin': butils_wrap.argmin,
    'argmax': butils_wrap.argmax,
    'convolve': butils_wrap.convolve,
    'arange': butils_wrap.arange,
    'sum': butils_wrap.sum,
    'sort': butils_wrap.sort,
    'add': butils_wrap.add,
    'mul': butils_wrap.mul,
    'beam_phase': bphysics_wrap.beam_phase,
    'fast_resonator': bphysics_wrap.fast_resonator,
    'kick': bphysics_wrap.kick,
    'rf_volt_comp': bphysics_wrap.rf_volt_comp,
    'drift': bphysics_wrap.drift,
    'linear_interp_kick': bphysics_wrap.linear_interp_kick,
    'LIKick_n_drift': bphysics_wrap.linear_interp_kick_n_drift,
    'synchrotron_radiation': bphysics_wrap.synchrotron_radiation,
    'synchrotron_radiation_full': bphysics_wrap.synchrotron_radiation_full,
    # 'linear_interp_time_translation': bphysics_wrap.linear_interp_time_translation,
    'slice': bphysics_wrap.slice,
    'slice_smooth': bphysics_wrap.slice_smooth,
    'music_track': bphysics_wrap.music_track,
    'music_track_multiturn': bphysics_wrap.music_track_multiturn,
    'diff': np.diff,
    'cumsum': np.cumsum,
    'cumprod': np.cumprod,
    'gradient': np.gradient,
    'sqrt': np.sqrt,
    'device': 'CPU'
}

_FFTW_func_dict = {
    'rfft': butils_wrap.rfft,
    'irfft': butils_wrap.irfft,
    'rfftfreq': butils_wrap.rfftfreq
}

_MPI_func_dict = {

}


def use_mpi():
    '''
    Replace some bm functions with MPI implementations
    '''
    global __exec_mode
    globals().update(_MPI_func_dict)
    __exec_mode = 'multi_node'


def mpiMode():
    global __exec_mode
    return __exec_mode == 'multi_node'


def use_fftw():
    '''
    Replace the existing rfft and irfft implementations
    with the ones coming from butils_wrap.
    '''
    globals().update(_FFTW_func_dict)


def update_active_dict(new_dict):
    '''
    Update the currently active dictionary. Removes the keys of the currently
    active dictionary from globals() and spills the keys
    from new_dict to globals()
    Args:
        new_dict A dictionary which contents will be spilled to globals()
    '''
    if not hasattr(update_active_dict, 'active_dict'):
        update_active_dict.active_dict = new_dict

    # delete all old implementations/references from globals()
    for key in update_active_dict.active_dict.keys():
        if key in globals():
            del globals()[key]
    # for key in globals().keys():
    #     if key in update_active_dict.active_dict.keys():
    #         del globals()[key]
    # add the new active dict to the globals()
    globals().update(new_dict)
    update_active_dict.active_dict = new_dict


################################################################################
update_active_dict(_CPU_func_dict)
################################################################################

# print ('Available functions on GPU:\n' + str(_CPU_numpy_func_dict.keys()))
# print ('Available functions on CPU:\n' + str(_GPU_func_dict.keys()))
