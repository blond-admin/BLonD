import numpy as np
import cupy as cp
#from ..utils import bmath as bm
import blond.utils.bmath as bm

class MyGpuarray(cp.ndarray):

    def __init__(self, input_array, dtype):
        self.array = cp.array(input_array, dtype = dtype)
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    @property
    def data(self):
        if self.parent is not None:
            self.parent.gpu_validate()
        return self.array
    
    @data.setter
    def data(self, key, value):
        self.parent.gpu_validate()
        self.array[key] = value
        self.parent.cpu_valid = False


class MyCpuarray(np.ndarray):

    def __new__(cls, input_array, dtype1=None, dtype2=None):
        if input_array is None:
            input_array = np.array([], dtype=bm.precision.real_t)

        obj = np.asarray(input_array).view(cls)
        if dtype1 is None:
            obj.dtype1 = input_array.dtype
        else:
            obj.dtype1 = dtype1
        if dtype2 is None:
            obj.dtype2 = input_array.dtype
        else:
            obj.dtype2 = dtype2
        obj.__class__ = MyCpuarray
        obj.cpu_valid = True
        obj.gpu_valid = False
        obj.sp = input_array.shape

        obj.dev_class = MyGpuarray(input_array.flatten(), obj.dtype2)
        obj.dev_array = obj.dev_class.data
        obj.dev_class.set_parent(obj)
        obj.gpu_valid = True

        return obj

    def cpu_validate(self):
        if hasattr(self, 'dev_array') and (not hasattr(self, "cpu_valid") or not self.cpu_valid):
        # if not self.cpu_valid:
                self.cpu_valid = True
                dummy = self.dev_array.get().reshape(self.sp).astype(self.dtype1)
                super().__setitem__(slice(None, None, None), dummy)
                self.cpu_valid = True

    def gpu_validate(self):
        if not self.gpu_valid:
            self.dev_array.set(np.array(self.flatten(), dtype = self.dtype2))
            self.gpu_valid = True

    def __setitem__(self, key, value):
        self.cpu_validate()
        # we either update the base array or this one
        if hasattr(self.base, 'gpu_valid'):
            self.base.gpu_valid = False
        else:
            self.gpu_valid = False
        super(MyCpuarray, self).__setitem__(key, value)

class CGA:
    def __init__(self, input_array, dtype1=None, dtype2=None):
        self.array_obj = MyCpuarray(input_array, dtype1=dtype1, dtype2=dtype2)
        self._dev_array = self.array_obj.dev_array

    def invalidate_cpu(self):
        self.array_obj.cpu_valid = False

    def invalidate_gpu(self):
        self.array_obj.gpu_valid = False

    @property
    def my_array(self):
        self.array_obj.cpu_validate()
        return self.array_obj

    @my_array.setter
    def my_array(self, value):
        if self.array_obj.size != 0 and value.dtype == self.array_obj.dtype1 and\
                self.array_obj.shape == value.shape:
            super(MyCpuarray, self.array_obj).__setitem__(slice(None, None, None), value)
        else:
            self.array_obj = MyCpuarray(value)

        self.array_obj.gpu_valid = False
        self.array_obj.cpu_valid = True

    @property
    def dev_my_array(self):
        self.array_obj.gpu_validate()
        return self.array_obj.dev_array

    @dev_my_array.setter
    def dev_my_array(self, value):
        if self.array_obj.dev_array.size != 0 and value.dtype == self.array_obj.dtype2 and\
                self.array_obj.dev_array.shape == value.shape:
            self.array_obj.dev_array[:] = value
        else:
            self.array_obj = MyCpuarray(value.get())
            self.array_obj.dev_class = MyGpuarray(value.get(), value.get().dtype)
            self.array_obj.dev_array = self.array_obj.dev_class.data
            self.array_obj.dev_class.set_parent(self.array_obj)
            self.array_obj.gpu_valid = True
            self.array_obj.cpu_valid = False

        self.array_obj.cpu_valid = False
        self.array_obj.gpu_valid = True