import numpy as np
from pycuda import gpuarray


# we have to define this method, to use the cubin file instead of jit compile
def fill(self, value):
    from ..gpu import gpu_butils_wrap as gpu_utils
    if self.dtype in [np.int, np.int32]:
        gpu_utils.set_zero_int(self)
    elif self.dtype in [np.float, np.float64]:
        gpu_utils.set_zero_double(self)
    elif self.dtype in [np.float32]:
        gpu_utils.set_zero_float(self)
    elif self.dtype in [np.complex64]:
        gpu_utils.set_zero_complex64(self)
    elif self.dtype in [np.complex128]:
        gpu_utils.set_zero_complex128(self)
    else:
        print(f'[cucache::fill] invalid data type: {self.dtype}')
        exit(-1)


gpuarray.GPUArray.fill = fill


class GpuarrayCache:
    """ this class is a software implemented cache for our gpuarrays,
    in order to avoid unnecessary memory allocations in the gpu"""

    def __init__(self):
        self.gpuarray_dict = {}
        self.enabled = False

    def add_array(self, key):

        self.gpuarray_dict[key] = gpuarray.empty(key[0], dtype=key[1])

    def get_array(self, key, zero_fills):
        if self.enabled:
            if key not in self.gpuarray_dict:
                self.add_array(key)
            else:
                if zero_fills:
                    self.gpuarray_dict[key].fill(0)
            return self.gpuarray_dict[key]
        else:
            to_ret = gpuarray.empty(key[0], dtype=key[1])
            to_ret.fill(0)
            return to_ret

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


gpu_cache = GpuarrayCache()


def get_gpuarray(key, zero_fills=False):
    return gpu_cache.get_array(key, zero_fills=zero_fills)


def enable_cache():
    gpu_cache.enable()


def disable_cache():
    gpu_cache.disable()
