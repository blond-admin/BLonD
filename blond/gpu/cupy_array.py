import numpy as np
import cupy as cp
import ctypes
from ..utils import bmath as bm



class MyGpuarray(cp.ndarray):

    def __new__(cls, shape, dtype, parent, ptr=None):
        obj = super().__new__(cls, shape=shape, dtype=dtype, memptr=ptr)
        obj.parent = parent # gain access to CGA functions, flags
        return obj

    def __getitem__(self, key):
        if hasattr(self.parent, 'gpu_valid'):
            self.parent.gpu_validate()

        return super(MyGpuarray, self).__getitem__(key)

    def __setitem__(self, key, value):
        if hasattr(self.parent, 'sync_needed'):
            self.parent.sync_needed = True
        elif hasattr(self.parent, 'gpu_valid'):
            self.parent.gpu_validate()
            self.parent.cpu_valid = False 

        super(MyGpuarray, self).__setitem__(key, value)


class MyCpuarray(np.ndarray):

    def __new__(cls, shape, parent, dtype=None, ptr=None):
        if ptr is None:
            obj = np.ndarray(shape=shape, dtype=dtype).view(cls)
        else:
            obj = np.ctypeslib.as_array(ptr, shape).view(cls)
        obj.parent = parent
        obj.dev = bm.gpuDev()
        return obj

    def __getitem__(self, key):
        if hasattr(self.parent, 'sync_needed') and (self.parent.sync_needed):
            self.dev.synchronize()
            self.parent.sync_needed = False
        elif hasattr(self.parent, 'cpu_valid'):
            self.parent.cpu_validate()

        return super(MyCpuarray, self).__getitem__(key)

    def __setitem__(self, key, value):
        if hasattr(self.parent, 'sync_needed') and (self.parent.sync_needed):
            self.dev.synchronize()
            self.parent.sync_needed = False
        elif hasattr(self.parent, 'cpu_valid'):
            self.parent.cpu_validate()
            self.parent.gpu_valid = False
        super(MyCpuarray, self).__setitem__(key, value)


class CGA:
    def __init__(self, input_array=None, shape=None, dtype=None):

        if input_array is None and shape is None: # init with None object
            self.dtype = bm.precision.real_t
            self._cpu_array = np.array([], dtype=self.dtype)
            self._dev_array = cp.array([], dtype=self.dtype)
        else:  
            self.dtype = input_array.dtype if dtype is None else dtype
            if input_array is not None: # provided input_array
                if input_array.shape == (0,): # init with empty array 
                    self._cpu_array = np.array([], dtype=self.dtype)
                    self._dev_array = cp.array([], dtype=self.dtype)
                else:
                    self.create_arrays(input_array, self.dtype)
            else:  # provided shape, dtype  
                self.create_arrays(shape, self.dtype)
                
    @property
    def my_array(self):
        if hasattr(self, 'cpu_valid'):
            self.cpu_validate()
        return self._cpu_array

    @my_array.setter
    def my_array(self, new_array):
        if self._cpu_array.shape == new_array.shape and new_array.dtype == self.dtype:
                self._cpu_array[:] = new_array   
        else:
            self.dtype = new_array.dtype
            self.create_arrays(new_array, self.dtype)

    @property
    def dev_my_array(self):
        if hasattr(self, 'gpu_valid'):
            self.gpu_validate()
        return self._dev_array

    @dev_my_array.setter
    def dev_my_array(self, new_array):
        if self._dev_array.shape == new_array.shape and new_array.dtype == self.dtype:
                self._dev_array[:] = new_array   
        else:
            self.dtype = new_array.dtype
            self.create_arrays(new_array.get(), self.dtype)


    def create_arrays(self, input_obj, dtype):
        # input_obj can either be shape tuple or numpy array_like object
        sp = input_obj.shape if hasattr(input_obj,'shape') else input_obj

        if np.issubdtype(self.dtype, np.complexfloating):
            # unified memory pointer to np_array with complex dtype
            # can't be initiated as ctypeslib doesn't support it
            self.cpu_valid = self.gpu_valid = True
            self._cpu_array = MyCpuarray(shape=sp, dtype=dtype, parent=self)
            self._dev_array = MyGpuarray(shape=sp, dtype=dtype, parent=self)
            if hasattr(input_obj,'shape'):
                self._cpu_array[:] = input_obj
                self._dev_array[:] = cp.array(input_obj)
        else:
            self.sync_needed = False
            self._cpu_array, self._dev_array = self.unified_arrays(shape=sp, dtype=dtype)
            if hasattr(input_obj,'shape'):
                self._cpu_array[:] = input_obj

        
    def unified_arrays(self, shape, dtype):
        arr_size = np.prod(shape) 
        alloc_size = arr_size * np.dtype(dtype).itemsize
        # malloc alloc_size bytes of unified memory
        ptr = cp.cuda.malloc_managed(alloc_size)

        # convert dt to ctype_float
        ctp = np.ctypeslib.as_ctypes_type(np.dtype(dtype))
        # cast pointers value to ctype POINTER of c_float  
        c_ptr = ctypes.cast(ptr.ptr,ctypes.POINTER(ctp))
        # create numpy array from POINTER
        np_arr = MyCpuarray(shape=shape, parent=self, ptr=c_ptr)
        gpu_shape = (np.prod(shape),) # flatten array
        cp_arr = MyGpuarray(shape=gpu_shape, dtype=dtype, parent=self, ptr=ptr)
        return np_arr, cp_arr

    def cpu_validate(self):
        if not self.cpu_valid:
            self.cpu_valid = True
            dummy = self._dev_array.get().reshape(self._cpu_array.shape).astype(self.dtype)
            self._cpu_array[:] = dummy

    def gpu_validate(self):
        if not self.gpu_valid:
            self.gpu_valid = True
            self._dev_array.set(self._cpu_array.flatten())




def get_gpuarray(shape, dtype, zero_fills=False):
    # cupy by default keeps the allocated memory of an array
    # in the default memory pool when it goes out of scope
    array = cp.empty(shape, dtype)
    if zero_fills:
        array.fill(0)
    return array






### TESTING ###
'''input_array = np.array([1,2,3,4])
print("Testing CGA with input_array and dtype:float64")
x = CGA(input_array,dtype=np.float64)
assert type(x.my_array) is MyCpuarray
assert type(x.dev_my_array) is MyGpuarray

assert issubclass(type(x.my_array), np.ndarray)
assert issubclass(type(x.dev_my_array), cp.ndarray)

x.dev_my_array[2] = 8.2
assert x.my_array[2] == 8.2

x.my_array[0] = 6.4
assert x.dev_my_array[0] == 6.4
print("Testing completed successfully")

print('--'*20)

print("Testing CGA with input_array and dtype:complex64")
x = CGA(input_array,dtype=np.complex64)
assert type(x.my_array) is MyCpuarray
assert type(x.dev_my_array) is MyGpuarray

assert issubclass(type(x.my_array), np.ndarray)
assert issubclass(type(x.dev_my_array), cp.ndarray)

x.dev_my_array[2] = 8.2 + 2j
assert x.my_array[2] == np.complex64(8.2 + 2j)

x.my_array[0] = 6.4 - 4.2j
assert x.dev_my_array[0] == np.complex64(6.4 - 4.2j)
print("Testing completed successfully")

print('--'*20)

print("Testing CGA with shape and dtype:float64")
x = CGA(shape=(4,),dtype=np.float64)
assert type(x.my_array) is MyCpuarray
assert type(x.dev_my_array) is MyGpuarray

assert issubclass(type(x.my_array), np.ndarray)
assert issubclass(type(x.dev_my_array), cp.ndarray)

x.my_array = np.array([1,2,3,4],dtype=np.float64)

x.dev_my_array[2] = 8.2
assert x.my_array[2] == 8.2

x.my_array[0] = 6.4
assert x.dev_my_array[0] == 6.4
print("Testing completed successfully")

print('--'*20)

print("Testing CGA with shape and dtype:complex128 and setting dev_array")
x = CGA(shape=(4,),dtype=np.complex64)
assert type(x.my_array) is MyCpuarray
assert type(x.dev_my_array) is MyGpuarray

assert issubclass(type(x.my_array), np.ndarray)
assert issubclass(type(x.dev_my_array), cp.ndarray)

x.dev_my_array = cp.array(np.array([1,2,3,4,5]), dtype=np.complex128)

x.dev_my_array[2] = 8.2 + 2j
assert x.my_array[2] == np.complex64(8.2 + 2j)

x.my_array[0] = 6.4 - 4.2j
assert x.dev_my_array[0] == np.complex64(6.4 - 4.2j)
print("Testing completed successfully")'''