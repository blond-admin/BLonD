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

# dictionary storing the CPU versions of the desired functions #
_CPU_func_dict = {
    'rfft': np.fft.rfft,
    'irfft': np.fft.irfft,
    'rfftfreq': np.fft.rfftfreq,
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
    'beam_phase': butils_wrap.beam_phase,
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


################################################################################
update_active_dict(_CPU_func_dict)
################################################################################


# GPU Related Utilities
def gpuMode():
    return globals()['device'] == 'GPU'


def gpuId():
    return __gpu_dev.id


def gpuDev():
    return __gpu_dev.dev


def gpuCtx():
    return __gpu_dev.ctx


def getMod():
    return __gpu_dev.my_mod()


class GPUDev:
    __instance = None

    def __init__(self, _gpu_num=0):
        if GPUDev.__instance != None:
            raise Exception("The GPUDev class is a singleton!")
        else:
            GPUDev.__instance = self
        import cupy as cp
        import cupy_backends.cuda.api.driver as driver
        from pycuda import driver as drv
        #drv.init()
        self.id = _gpu_num
        self.dev =cp.cuda.Device(self.id)
        self.dev.use()
        #self.ctx = driver.ctxCreate(self.dev)
        this_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

        if precision.num == 1:
            self.mod = cp.RawModule(path = os.path.join(
                this_dir, '../gpu/cuda_kernels/kernels_single.cubin'))
        else:
            self.mod = cp.RawModule(path = os.path.join(
                this_dir, '../gpu/cuda_kernels/kernels_double.cubin'))

    def report_attributes(self):
        # Saves into a file all the device attributes
        with open(f'{self.dev.name()}-attributes.txt', 'w') as f:
            for k, v in self.dev.get_attributes().items():
                f.write(f"{k}:{v}\n")

    def func(self, name):
        return self.mod.get_function(name)

    def __del__(self):
        #import cupy_backends.cuda.api.driver as driver
        #gpuDev().use()
        #driver.ctxDestroy(self.ctx)
        self.ctx = None
        update_active_dict(_CPU_func_dict)

    def my_mod(self):
        return self.mod


def use_gpu(gpu_id=0):

    if gpu_id < 0:
        return

    global __gpu_dev
    __gpu_dev = GPUDev(gpu_id)
    from ..gpu import cupy_physics_wrap
    from ..gpu import cupy_butils_wrap
    import cupy as cp
    # now we have to add use_gpu methods to our objects
    from ..gpu import gpu_activation
    dev_name = cp.cuda.runtime.getDeviceProperties(gpuDev())['name']
    print(''.join(['#']*20) +
          ' Using GPU: id {}, name {}, Compute Capability {} '.format(
              gpuId(), dev_name, gpuDev().compute_capability)
          + ''.join(['#']*20), flush=True)
    
    globals()['device'] = 'GPU'

    _GPU_func_dict = {
        'rfft': cupy_butils_wrap.gpu_rfft,
        'irfft': cupy_butils_wrap.gpu_irfft,
        'convolve': cupy_butils_wrap.gpu_convolve,
        'beam_phase': cupy_physics_wrap.gpu_beam_phase,
        'kick': cupy_physics_wrap.gpu_kick,
        'rf_volt_comp': cupy_physics_wrap.gpu_rf_volt_comp,
        'drift': cupy_physics_wrap.gpu_drift,
        'linear_interp_kick': cupy_physics_wrap.gpu_linear_interp_kick,
        'LIKick_n_drift': cupy_physics_wrap.gpu_linear_interp_kick_drift,
        'synchrotron_radiation': cupy_physics_wrap.gpu_synchrotron_radiation,
        'synchrotron_radiation_full': cupy_physics_wrap.gpu_synchrotron_radiation_full,
        # 'linear_interp_time_translation': butils_wrap.linear_interp_time_translation,
        'slice': cupy_physics_wrap.gpu_slice,
        'slice_smooth': butils_wrap.slice_smooth,
        # 'rfftfreq': gpu_butils_wrap.gpu_rfftfreq,
        'rfftfreq': np.fft.rfftfreq,
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
        'arange': butils_wrap.arange,
        'sum': butils_wrap.sum,
        'sort': butils_wrap.sort,
        'add': butils_wrap.add,
        'mul': butils_wrap.mul,
        'fast_resonator': butils_wrap.fast_resonator,
        'music_track': butils_wrap.music_track,
        'music_track_multiturn': butils_wrap.music_track_multiturn,
        'diff': np.diff,
        'cumsum': np.cumsum,
        'cumprod': np.cumprod,
        'gradient': np.gradient,
        'sqrt': np.sqrt,
        'device': 'GPU'
    }
    update_active_dict(_GPU_func_dict)


# print ('Available functions on GPU:\n' + str(_CPU_numpy_func_dict.keys()))
# print ('Available functions on CPU:\n' + str(_GPU_func_dict.keys()))
