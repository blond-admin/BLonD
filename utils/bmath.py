'''
BLonD math functions

@author Stefan Hegglin, Konstantinos Iliakis
@date 20.10.2017
'''
import numpy as np
from functools import wraps
import blondmath_wrap.py

#### dictionary storing the CPU versions of the desired functions ####
_CPU_func_dict = {
    'sin': np.sin,
    'cos': np.cos,
    'exp': np.exp,
    'mean': np.mean,
    'std': cp.std,
    'min_idx': np.min,
    'max_idx': np.max,
    'diff': np.diff,
    'floor': np.floor,
    'argsort': np.argsort,
    'take': np.take,
    'convolve': np.convolve,
    'seq': lambda stop: np.arange(stop, dtype=np.int32),
    'arange': wraps(np.arange)(
        lambda start, stop, step, nslices=None, dtype=np.float64:
            np.arange(start, stop, step, dtype)
    ),
    'zeros': np.zeros,
    'empty': np.empty,
    'empty_like': np.empty_like,
    'ones': np.ones,
    'device': 'CPU',
    'sum': np.sum,
    'cumsum': np.cumsum,
    'all': np.all,
    'any': np.any,
    'indexify': lambda array: array.astype(np.int32),
    'abs': np.abs,
    'sign': np.sign,
    'sqrt': np.sqrt,
    'allclose': np.allclose,
    'put': np.put,
    'atleast_1d': np.atleast_1d,
    'almost_zero': lambda array, *args, **kwargs: np.allclose(array, 0, *args, **kwargs)
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
