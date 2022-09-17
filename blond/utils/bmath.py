'''
BLonD math and physics core functions

@author Stefan Hegglin, Konstantinos Iliakis, Panagiotis Tsapatsaris
@date 20.10.2017
'''
# from functools import wraps
import os

import numpy as np
from ..utils import butils_wrap
# from ..utils import bphysics_wrap

precision = butils_wrap.precision
__exec_mode = 'single_node'
# Other modes: multi_node

__gpu_dev = None

# dictionary storing the CPU versions of the desired functions #
_CPU_func_dict = {
    'rfft': np.fft.rfft,
    'irfft': np.fft.irfft,
    'rfftfreq': np.fft.rfftfreq,
    'irfft_packed': butils_wrap.irfft_packed,
    'sin_cpp': butils_wrap.sin_cpp,
    'cos_cpp': butils_wrap.cos_cpp,
    'exp_cpp': butils_wrap.exp_cpp,
    'mean_cpp': butils_wrap.mean_cpp,
    'std_cpp': butils_wrap.std_cpp,
    'where_cpp': butils_wrap.where_cpp,
    'interp_cpp': np.interp,
    # 'interp_cpp': butils_wrap.interp_cpp,
    # 'interp_const_space': butils_wrap.interp_const_space,
    'interp_const_space': np.interp,
    'cumtrapz': butils_wrap.cumtrapz,
    'trapz_cpp': butils_wrap.trapz_cpp,
    'linspace_cpp': butils_wrap.linspace_cpp,
    'argmin_cpp': butils_wrap.argmin_cpp,
    'argmax_cpp': butils_wrap.argmax_cpp,
    'convolve': butils_wrap.convolve,
    'arange_cpp': butils_wrap.arange_cpp,
    'sum_cpp': butils_wrap.sum_cpp,
    'sort_cpp': butils_wrap.sort_cpp,
    'add_cpp': butils_wrap.add_cpp,
    'mul_cpp': butils_wrap.mul_cpp,
    'beam_phase': butils_wrap.beam_phase,
    'beam_phase_fast': butils_wrap.beam_phase_fast,
    'fast_resonator': butils_wrap.fast_resonator,
    'kick': butils_wrap.kick,
    'rf_volt_comp': butils_wrap.rf_volt_comp,
    'drift': butils_wrap.drift,
    'linear_interp_kick': butils_wrap.linear_interp_kick,
    'LIKick_n_drift': butils_wrap.linear_interp_kick_n_drift,
    'synchrotron_radiation': butils_wrap.synchrotron_radiation,
    'synchrotron_radiation_full': butils_wrap.synchrotron_radiation_full,
    'set_random_seed': butils_wrap.set_random_seed,
    'sparse_histogram': butils_wrap.sparse_histogram,
    # 'linear_interp_time_translation': butils_wrap.linear_interp_time_translation,
    'slice': butils_wrap.slice,
    'slice_smooth': butils_wrap.slice_smooth,
    'music_track': butils_wrap.music_track,
    'music_track_multiturn': butils_wrap.music_track_multiturn,
    'device': 'CPU'
}

# add numpy functions in the dictionary
for fname in dir(np):
    if callable(getattr(np, fname)) and (fname not in _CPU_func_dict):
        _CPU_func_dict[fname] = getattr(np, fname)

# add basic numpy modules to dictionary as they are not callable
_CPU_func_dict['random'] = getattr(np, 'random')
_CPU_func_dict['fft'] = getattr(np, 'fft')

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


# precision can be single or double
def use_precision(_precision='double'):
    global precision
    butils_wrap.precision = butils_wrap.Precision(_precision)
    precision = butils_wrap.precision


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


###############################################################################
update_active_dict(_CPU_func_dict)
###############################################################################


# GPU Related Utilities

def gpuDev():
    return __gpu_dev


class GPUDev:
    __instance = None

    def __init__(self, _gpu_num=0):
        if GPUDev.__instance is not None:
            return
            # raise Exception("The GPUDev class is a singleton!")
        else:
            GPUDev.__instance = self

        import cupy as cp
        self.id = _gpu_num
        self.dev = cp.cuda.Device(self.id)
        self.dev.use()

        self.name = cp.cuda.runtime.getDeviceProperties(self.dev)['name']
        self.attributes = self.dev.attributes
        self.properties = cp.cuda.runtime.getDeviceProperties(self.dev)

        # set the default grid and block sizes
        default_blocks = 2 * self.attributes['MultiProcessorCount']
        default_threads = self.attributes['MaxThreadsPerBlock']
        blocks = int(os.environ.get('GPU_BLOCKS', default_blocks))
        threads = int(os.environ.get('GPU_THREADS', default_threads))
        self.grid_size = (blocks, 1, 1)
        self.block_size = (threads, 1, 1)

        this_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
        comp_capability = self.dev.compute_capability

        if precision.num == 1:
            self.mod = cp.RawModule(path=os.path.join(
                this_dir, f'../gpu/cuda_kernels/kernels_single_sm_{comp_capability}.cubin'))
        else:
            self.mod = cp.RawModule(path=os.path.join(
                this_dir, f'../gpu/cuda_kernels/kernels_double_sm_{comp_capability}.cubin'))

    def report_attributes(self):
        # Saves into a file all the device attributes
        with open(f'{self.name}-attributes.txt', 'w') as f:
            for k, v in self.attributes.items():
                f.write(f"{k}:{v}\n")

    def func(self, name):
        return self.mod.get_function(name)

    def __del__(self):
        update_active_dict(_CPU_func_dict)


def use_gpu(gpu_id=0):

    if gpu_id < 0:
        return

    global __gpu_dev
    if __gpu_dev is None:
        __gpu_dev = GPUDev(gpu_id)

        print(''.join(['#']*10) +
              ' Using GPU: id {}, name {}, Compute Capability {} '.format(
            __gpu_dev.id, __gpu_dev.name, __gpu_dev.dev.compute_capability)
            + ''.join(['#']*10) + '\n', flush=True)

    from ..gpu import cupy_butils_wrap
    import cupy as cp

    _GPU_func_dict = {
        # 'rfft': cupy_butils_wrap.rfft,
        # 'irfft': cupy_butils_wrap.irfft,
        'rfft': cp.fft.rfft,
        'irfft': cp.fft.irfft,
        'rfftfreq': cp.fft.rfftfreq,
        'convolve': cp.convolve,
        # 'convolve': cupy_butils_wrap.convolve,
        'beam_phase': cupy_butils_wrap.beam_phase,
        'beam_phase_fast': cupy_butils_wrap.beam_phase_fast,
        'kick': cupy_butils_wrap.kick,
        'rf_volt_comp': cupy_butils_wrap.rf_volt_comp,
        'drift': cupy_butils_wrap.drift,
        'linear_interp_kick': cupy_butils_wrap.linear_interp_kick,
        'LIKick_n_drift': cupy_butils_wrap.linear_interp_kick_drift,
        'synchrotron_radiation': cupy_butils_wrap.synchrotron_radiation,
        'synchrotron_radiation_full': cupy_butils_wrap.synchrotron_radiation_full,
        'slice': cupy_butils_wrap.slice,
        # 'interp_const_space': cupy_butils_wrap.interp,
        'interp_const_space': cp.interp,
        'device': 'GPU'
    }
    # add cupy functions in the dictionary
    for fname in dir(cp):
        if callable(getattr(cp, fname)) and (fname not in _GPU_func_dict):
            _GPU_func_dict[fname] = getattr(cp, fname)
    update_active_dict(_GPU_func_dict)

    # add basic cupy modules to dictionary as they are not callable
    _GPU_func_dict['random'] = getattr(cp, 'random')
    _GPU_func_dict['fft'] = getattr(cp, 'fft')


def use_cpu():
    update_active_dict(_CPU_func_dict)
