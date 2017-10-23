'''
BLonD math functions

@author Stefan Hegglin, Konstantinos Iliakis
@date 20.10.2017
'''
import numpy as np
# from functools import wraps
from utils import blondmath_wrap as cpp

#### dictionary storing the CPU versions of the desired functions ####
_CPU_func_dict = {
    'sin': cpp.sin,
    'cos': cpp.cos,
    'exp': cpp.exp,
    'mean': cpp.mean,
    'std': cpp.std,
    'interp': cpp.interp,
    'cumtrapz': cpp.cumtrapz,
    'trapz': cpp.trapz,
    'linspace': cpp.linspace,
    'argmin': cpp.argmin,
    'argmax': cpp.argmax,
    'convolve': cpp.convolve,
    'arange': cpp.arange,
    'sum': cpp.sum,
    'diff': np.diff,
    'cumsum': np.cumsum,
    'sort': np.sort,
    'cumprod': np.cumprod,
    'gradient': np.gradient,
    'sqrt': np.sqrt,
    # 'floor': np.floor,
    # 'take': np.take,
    # 'seq': lambda stop: np.arange(stop, dtype=np.int32),
    # 'zeros': np.zeros,
    # 'empty': np.empty,
    # 'empty_like': np.empty_like,
    # 'ones': np.ones,
    # 'all': np.all,
    # 'any': np.any,
    # 'indexify': lambda array: array.astype(np.int32),
    # 'abs': np.abs,
    # 'sign': np.sign,
    # 'allclose': np.allclose,
    # 'put': np.put,
    # 'atleast_1d': np.atleast_1d,
    # 'almost_zero': lambda array, *args, **kwargs: np.allclose(array, 0, *args, **kwargs),
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
