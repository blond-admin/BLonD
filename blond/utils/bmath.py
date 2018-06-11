'''
BLonD math and physics core functions

@author Stefan Hegglin, Konstantinos Iliakis
@date 20.10.2017
'''
# from functools import wraps
import numpy as np
from ..utils import butils_wrap
from ..utils import bphysics_wrap

# dictionary storing the CPU versions of the desired functions #
_CPU_func_dict = {
    'sin': butils_wrap.sin,
    'cos': butils_wrap.cos,
    'exp': butils_wrap.exp,
    'mean': butils_wrap.mean,
    'std': butils_wrap.std,
    'interp': butils_wrap.interp,
    'cumtrapz': butils_wrap.cumtrapz,
    'trapz': butils_wrap.trapz,
    'linspace': butils_wrap.linspace,
    'argmin': butils_wrap.argmin,
    'argmax': butils_wrap.argmax,
    'convolve': butils_wrap.convolve,
    'arange': butils_wrap.arange,
    'sum': butils_wrap.sum,
    'sort': butils_wrap.sort,
    'kick': bphysics_wrap.kick,
    'rf_volt_comp': bphysics_wrap.rf_volt_comp,
    'drift': bphysics_wrap.drift,
    'linear_interp_kick': bphysics_wrap.linear_interp_kick,
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
    for key in globals().keys():
        if key in update_active_dict.active_dict.keys():
            del globals()[key]
    # add the new active dict to the globals()
    globals().update(new_dict)
    update_active_dict.active_dict = new_dict


################################################################################
update_active_dict(_CPU_func_dict)
################################################################################

# print ('Available functions on GPU:\n' + str(_CPU_numpy_func_dict.keys()))
# print ('Available functions on CPU:\n' + str(_GPU_func_dict.keys()))
