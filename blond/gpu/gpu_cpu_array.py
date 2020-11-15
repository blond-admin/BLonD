import numpy as np
from pycuda import gpuarray
from ..utils import bmath as bm


try:
    from pyprof import timing
except ImportError:
    from ..utils import profile_mock as timing


class MyGpuarray(gpuarray.GPUArray):

    def set_parent(self, parent):
        self.parent = parent

    def __setitem__(self, key, value):
        self.parent.gpu_validate()
        super().__setitem__(key, value)
        self.parent.cpu_valid = False
        return self

    def __getitem__(self, key):
        self.parent.gpu_validate()
        return super(MyGpuarray, self).__getitem__(key)


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

        obj.dev_array = gpuarray.to_gpu(input_array.flatten().astype(obj.dtype2))
        obj.dev_array.__class__ = MyGpuarray
        obj.dev_array.set_parent(obj)
        obj.gpu_valid = True

        return obj

    def cpu_validate(self):
        if hasattr(self, 'dev_array') and (not hasattr(self, "cpu_valid") or not self.cpu_valid):
                self.cpu_valid = True
                dummy = self.dev_array.get().reshape(self.sp).astype(self.dtype1)
                super().__setitem__(slice(None, None, None), dummy)
                self.cpu_valid = True

    def gpu_validate(self):
        if not self.gpu_valid:
            self.dev_array.set(gpuarray.to_gpu(self.flatten().astype(self.dtype2)))
            self.gpu_valid = True

    def __setitem__(self, key, value):
        self.cpu_validate()
        # we either update the base array or this one
        if hasattr(self.base, 'gpu_valid'):
            self.base.gpu_valid = False
        else:
            self.gpu_valid = False
        super(MyCpuarray, self).__setitem__(key, value)

    def __getitem__(self, key):
        self.cpu_validate()
        temp = super(MyCpuarray, self).__getitem__(key)
        return temp


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
            self.array_obj.dev_array = value
            self.array_obj.dev_array.__class__ = MyGpuarray
            self.array_obj.dev_array.set_parent(self.array_obj)
            self.array_obj.gpu_valid = True
            self.array_obj.cpu_valid = False

        self.array_obj.cpu_valid = False
        self.array_obj.gpu_valid = True


# # To test this implementation
#
# class ExampleClass:
#     def __init__(self, bin_centers):
#         self.bin_centers_obj = CGA(bin_centers)
#
#     @property
#     def bin_centers(self):
#         return self.bin_centers_obj.my_array
#
#     @bin_centers.setter
#     def bin_centers(self, value):
#         self.bin_centers_obj.my_array[:] = value
#
#     @property
#     def dev_bin_centers(self):
#         return self.bin_centers_obj.dev_my_array
#
#     @dev_bin_centers.setter
#     def dev_bin_centers(self, value):
#         self.bin_centers_obj.dev_my_array = value
#
#
# # Testing with a 2d array
#
# C = ExampleClass(np.array([[1, 2, 3, 4],
#                            [5, 6, 7, 8]]).astype(np.float64))
# C.bin_centers[0][0] = 5
# print(C.bin_centers)
# print(C.dev_bin_centers)
#
# print("....................")
# C.dev_bin_centers = gpuarray.zeros(8, np.float64)
# print(C.bin_centers)
# print(C.dev_bin_centers)
#
# print("....................")
# C.bin_centers[0][:3] = np.array([1, 2, 3])
# print(C.bin_centers)
# print(C.dev_bin_centers)
#
# print("....................")
# C.bin_centers *= C.bin_centers + C.bin_centers
# print(C.bin_centers)
# print(C.dev_bin_centers)
#
# # Testing with a 2d array
#
# D = ExampleClass(np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float64))
# D.bin_centers[0] = 5
# print(D.bin_centers)
# print(D.dev_bin_centers)
#
# print("....................")
# D.dev_bin_centers = gpuarray.zeros(8, np.float64)
# print(D.bin_centers)
# print(D.dev_bin_centers)
#
# print("....................")
# D.bin_centers[:3] = np.array([1, 2, 3])
# print(D.bin_centers)
# print(D.dev_bin_centers)
#
# print("....................")
# D.bin_centers *= D.bin_centers + D.bin_centers
# print(D.bin_centers)
# print(D.dev_bin_centers)
