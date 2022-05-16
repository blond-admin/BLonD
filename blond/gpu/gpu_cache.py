import numpy as np
from pycuda import gpuarray
import pycuda

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

dtype_to_bytes = {
    np.int32: 4,
    np.float64: 8,
    np.float32: 4,
    np.complex128: 16,
    np.complex64: 8
}
gpuarray.GPUArray.fill = fill


class GpuarrayCache:
    """ this class is a software implemented cache for our gpuarrays,
    in order to avoid unnecessary memory allocations in the gpu"""

    def __init__(self, capacity=-1):
        self.gpuarray_dict = {}
        self.enabled = False
        self.i = 0
        self.size = 0
        self.capacity = capacity

    def add_array(self, key):
        self.i += 1
        self.check_size(key)
        while(True):
            try:
                self.gpuarray_dict[key] = gpuarray.empty(key[0], dtype=key[1])
                break
            except pycuda._driver.MemoryError:
                self.size = 0
                self.gpuarray_dict = {}
        self.size += key[0] * dtype_to_bytes[key[1]]

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

    def check_size(self, key):
        if self.capacity==-1:
            return
        else:
            if (self.size+key[0]*dtype_to_bytes[key[1]] > self.capacity):
                print("Freeing cache {}".format(self.i))
                self.gpuarray_dict = {}
                self.size=0

    def enable(self, capacity=-1):
        self.capacity=capacity
        self.enabled = True

    def disable(self):
        self.enabled = False

    

gpu_cache = GpuarrayCache()


def get_gpuarray(key, zero_fills=False):
    return gpu_cache.get_array(key, zero_fills=zero_fills)


def enable_cache(capacity=-1):
    gpu_cache.enable(capacity=capacity)


def disable_cache():
    gpu_cache.disable()

def get_size():
    return gpu_cache.size
