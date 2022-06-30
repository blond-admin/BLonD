import numpy as np
import cupy as cp
from ..utils import bmath as bm

class CGA:
    def __init__(self, input_array=None, shape=None, dtype=None):

        if input_array is None and shape is None and dtype is None:
            self.dtype = bm.precision.real_t
            self._cpu_array = np.array([], dtype=self.dtype)
            self._dev_array = cp.array([], dtype=self.dtype)
        else:
            self.dtype = input_array.dtype if dtype is None else dtype
            self.dev = cp.cuda.Device(0)
            if input_array is None:
                self.sp = shape
                gpu_shape = np.prod(shape) # flatten
                self._cpu_array = np.ndarray(shape, dtype=self.dtype)
                self._dev_array = cp.ndarray(gpu_shape, dtype=self.dtype)
            else:
                self.sp = input_array.shape
                self._cpu_array = np.array(input_array, dtype=self.dtype)
                self._dev_array = cp.array(input_array.flatten(), dtype=self.dtype)
        
        self.cpu_valid = True
        self.gpu_valid = True


    def invalidate_cpu(self):
        # Must be called after _dev_array change through indexing
        self.cpu_valid = False

    def invalidate_gpu(self):
        # Must be called after _cpu_array change through indexing
        self.gpu_valid = False

    @property
    def my_array(self):
        self.cpu_validate()
        return self._cpu_array

    @my_array.setter
    def my_array(self, new_array):
        if self._cpu_array.shape == new_array.shape and new_array.dtype == self._cpu_array.dtype:
                self._cpu_array[:] = new_array 
        else:
            self.dtype = new_array.dtype
            self.sp = new_array.shape
            self._cpu_array = np.array(new_array, dtype=self.dtype)
            self._dev_array = cp.array(new_array.flatten(), dtype=self.dtype)

        self.cpu_valid = True
        self.gpu_valid = False

    @property
    def dev_my_array(self):
        self.gpu_validate()
        return self._dev_array

    @dev_my_array.setter
    def dev_my_array(self, new_array):
        if self._dev_array.shape == new_array.shape and new_array.dtype == self._dev_array.dtype:
                self._dev_array[:] = new_array
        else:
            self.dtype = new_array.dtype
            self.sp = new_array.shape
            self._cpu_array = np.array(new_array.get(), dtype=self.dtype)
            self._dev_array = cp.array(new_array, dtype=self.dtype)

        self.cpu_valid = False
        self.gpu_valid = True


    def cpu_validate(self):
        if not self.cpu_valid:
            dummy = self._dev_array.get().reshape(self.sp).astype(self.dtype)
            self._cpu_array.__setitem__(slice(None, None, None), dummy)
            self.cpu_valid = True

    def gpu_validate(self):
        if not self.gpu_valid:
            self._dev_array.set(self._cpu_array.flatten())
            self.gpu_valid = True