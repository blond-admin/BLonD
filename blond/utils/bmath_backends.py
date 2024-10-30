"""File to deal with dependencies to numpy, cupy, c++, numba, etc."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
    from ..gpu import GPU_DEV


class MasterBackend:
    """
    Listing of all attributes that must be declared for different backends
    """

    def __init__(self):
        from blond.utils.butils_wrap_cpp import precision
        self.precision = precision

        self.device = None

        # listing definitions that shall be declared !
        # _verify_backend() will check for declaration
        # and raise exception if not declared
        self.diff = None
        self.uint = None
        self.empty = None
        self.sign = None
        self.vectorize = None
        self.hstack = None
        self.arctan = None
        self.arcsin = None
        self.sqrt = None
        self.trapz = None
        self.reshape = None
        self.exp = None
        self.int32 = None
        self.delete = None
        self.unique = None
        self.abs = None
        self.zeros = None
        self.floor = None
        self.resize = None
        self.take = None
        self.round = None
        self.fabs = None
        self.float64 = None
        self.meshgrid = None
        self.mean = None
        self.append = None
        self.argmax = None
        self.dot = None
        self.double = None
        self.sum = None
        self.cumsum = None
        self.cos = None
        self.concatenate = None
        self.full = None
        self.asarray = None
        self.ones = None
        self.uint32 = None
        self.histogram = None
        self.unwrap = None
        self.ceil = None
        self.interp = None
        self.std = None
        self.empty_like = None
        self.min = None
        self.max = None
        self.cosh = None
        self.linspace = None
        self.array = None
        self.gradient = None
        self.isnan = None
        self.hamming = None
        self.clip = None
        self.where = None
        self.isinf = None
        self.sin = None
        self.zeros_like = None
        self.arange = None
        self.loadtxt = None
        self.convolve = None
        self.fliplr = None
        self.log = None
        self.heaviside = None
        self.absolute = None
        self.float32 = None
        self.ascontiguousarray = None
        self.complex64 = None
        self.copy = None
        self.poly1d = None
        self.invert = None
        self.all = None
        self.int64 = None
        self.count_nonzero = None
        self.nonzero = None
        self.log10 = None
        self.real = None
        self.argsort = None
        self.complex128 = None
        self.irfft = None
        # self.cumtrapz = None # not available in cupy
        self.convolve = None
        self.rfft = None
        self.rfftfreq = None
        # full packages
        self.random = None
        self.fft = None

        # extra definitions
        self.slice_beam = None
        self.drift = None
        self.sparse_histogram = None
        self.distribution_from_tomoscope = None
        self.set_random_seed = None
        self.synchrotron_radiation_full = None
        self.kick = None
        self.fast_resonator = None
        self.beam_phase_fast = None
        self.rf_volt_comp = None
        self.slice_smooth = None
        self.linear_interp_kick = None
        self.music_track_multiturn = None
        self.music_track = None
        self.synchrotron_radiation = None
        self.beam_phase = None

        # self.interp_const_bin = None
        # self.mean_cpp = None # todo required??
        # self.sum_cpp = None # todo required??
        # self.arange_cpp = None # todo required??
        # self.sort_cpp = None # todo required??
        # self.random_normal = None # todo required
        # self.mul_cpp = None # todo required??
        # self.trapz_cpp = None # todo required??
        # self.argmax_cpp = None # todo required??
        # self.std_cpp = None # todo required??
        # self.argmin_cpp = None # todo required??
        # self.where_cpp = None # todo required??
        # self.exp_cpp = None # todo required??
        # self.sin_cpp = None # todo required??
        # self.cos_cpp = None # todo required??
        # self.add_cpp = None # todo required??
        # self.linspace_cpp = None # todo required??
        # self.interp_cpp = None # todo required??
        # self.interp_const_space = None # todo required??

    def verify_backend(self):
        """Checks if all attributes from init are declared correctly

        Raises
        ------
        NotImplementedError
            If some attribute is None
        AttributeError
            If an attribute was declared by a subclass, that is not declared
            in MasterBackend
        """

        master = MasterBackend()
        master_attributes = master.__dict__.keys()
        for key, value in self.__dict__.items():
            if value is None:
                raise NotImplementedError(
                    f"Missing attribute 'self.{key}': "
                    f"Please declare attribute in '{type(self).__name__}'"
                )
            if key not in master_attributes:
                raise AttributeError(
                    f"Attribute 'self.{key}' is not foreseen: "
                    f"Please declare attribute in 'BackendMaster' and all of "
                    f"its subclasses !"

                )

    def use_cpp(self):
        """
        Replace some dependencies by there equivalent in C++
        """

        # hacky way to replace all methods from __init__
        self.__dict__ = CppBackend().__dict__

    def use_numba(self):
        """
        Replace some dependencies by there equivalent in Numba
        """
        # hacky way to replace all methods from __init__
        warnings.warn("""It is recommended to update bmath""")
        self.__dict__ = NumbaBackend().__dict__

    def use_py(self):
        """
        Replace some dependencies by there equivalent in Numpy/Python
        """
        # hacky way to replace all methods from __init__
        self.__dict__ = PyBackend().__dict__

    def use_cpu(self):
        """Try using one of the CPU backends

        Notes
        -----
        Order of priority:
            1. CppBackend
            2. NumbaBackend
            3. PyBackend"""
        success = False
        for backend_class in (CppBackend, NumbaBackend, PyBackend):
            try:
                backend = backend_class()
                self.__dict__ = backend.__dict__
                success = True
                break
            except Exception as exc:
                print(f"Couldn't set {backend_class.__name__}: {exc}")
        assert success, "Could not set any CPU backend."

    def use_mpi(self):
        """Sets the device to CPU_MPI"""
        self.device = "CPU_MPI"  # todo which backend??

    def in_mpi(self) -> bool:
        """Checks if the device is CPU_MPI"""
        return self.device == "CPU_MPI"

    def use_fftw(self):
        """Overwrites rfft, irfft, and rfftfreq by C++ functions"""
        from blond.utils import butils_wrap_cpp as _cpp

        self.rfft = _cpp.rfft
        self.irfft = _cpp.irfft
        self.rfftfreq = _cpp.rfftfreq

    def use_precision(self,
                      _precision: Literal['single', 'double'] = 'double'):
        """Change the precision used in calculations.

        Parameters
        ----------
        _precision (str, optional):
            Can be either 'single' or 'double'.  Defaults to 'double'.
        """

        from blond.utils import butils_wrap_cpp as _cpp
        self.precision.set(_precision)

        try:
            from ..gpu import GPU_DEV
            GPU_DEV.load_library(_precision)
        except Exception as e:
            from warnings import warn
            warn(f"The GPU backend is not available:\n{e}", UserWarning)
        _cpp.load_libblond(_precision)

    def __update_active_dict(self):
        # outdated method from bmath
        raise NotImplementedError()

    def get_gpu_device(self) -> GPU_DEV:
        """Get the GPU device object

        Returns:
            GPU_DEV: TODO
        """
        from ..gpu import GPU_DEV
        return GPU_DEV

    def use_gpu(self, gpu_id: int = 0):
        """Use the GPU device to perform the calculations.

        Parameters
        ----------
            gpu_id int:
                The device id. Defaults to 0.
        """

        if gpu_id < 0:
            warnings.warn(f"Invalid {gpu_id=}, must positive number!")
            return

        from ..gpu import GPU_DEV

        GPU_DEV.set(gpu_id)

        self.__dict__ = GpuBackend().__dict__

    def report_backend(self):
        """Prints the current backend"""
        print(f'Using the {self.device}')


class __NumpyBackend(MasterBackend):
    def __init__(self):
        """All numpy  function definitions. Accelerator science definitions
        missing"""
        super().__init__()

        import numpy as np
        import scipy
        from packaging.version import Version

        self.arctan = np.arctan
        self.ones = np.ones
        self.float64 = np.float64
        self.sign = np.sign
        self.sum = np.sum
        self.complex64 = np.complex64
        self.clip = np.clip
        self.interp = np.interp
        self.round = np.round
        self.asarray = np.asarray
        self.unique = np.unique
        self.absolute = np.absolute
        self.empty_like = np.empty_like
        self.sqrt = np.sqrt
        self.abs = np.abs
        self.nonzero = np.nonzero
        self.delete = np.delete
        self.reshape = np.reshape
        self.fliplr = np.fliplr
        self.loadtxt = np.loadtxt
        self.dot = np.dot
        self.argmax = np.argmax
        self.zeros = np.zeros
        self.exp = np.exp
        self.cos = np.cos
        self.max = np.max
        self.floor = np.floor
        self.int64 = np.int64
        self.sin = np.sin
        self.mean = np.mean
        self.resize = np.resize
        self.fabs = np.fabs
        self.heaviside = np.heaviside
        self.log10 = np.log10
        self.hstack = np.hstack
        self.float32 = np.float32
        self.array = np.array
        self.histogram = np.histogram
        self.append = np.append
        self.arange = np.arange
        self.poly1d = np.poly1d
        self.count_nonzero = np.count_nonzero
        self.arcsin = np.arcsin
        self.copy = np.copy
        self.real = np.real
        self.meshgrid = np.meshgrid
        self.std = np.std
        self.empty = np.empty
        self.cumsum = np.cumsum
        self.min = np.min
        self.uint32 = np.uint32
        self.uint = np.uint
        self.invert = np.invert
        self.int32 = np.int32
        self.double = np.double
        self.trapz = np.trapz
        # self.cumtrapz = cumtrapz # not available in cupy
        self.zeros_like = np.zeros_like
        self.full = np.full
        self.concatenate = np.concatenate
        self.argsort = np.argsort
        self.ascontiguousarray = np.ascontiguousarray
        self.ceil = np.ceil
        self.gradient = np.gradient
        self.where = np.where
        self.take = np.take
        self.hamming = np.hamming
        self.log = np.log
        self.complex128 = np.complex128
        self.all = np.all
        self.isnan = np.isnan
        self.linspace = np.linspace
        self.vectorize = np.vectorize
        self.cosh = np.cosh
        self.convolve = np.convolve
        self.diff = np.diff
        self.unwrap = np.unwrap
        self.isinf = np.isinf

        self.random = np.random
        self.fft = np.fft
        self.rfft = np.fft.rfft
        self.irfft = np.fft.irfft
        self.rfftfreq = np.fft.rfftfreq


class __CupyBackend(MasterBackend):
    def __init__(self):
        """All cupy  function definitions. Accelerator science definitions
        missing"""
        super().__init__()

        import cupy as cp
        # self.cumtrapz = None # not available in cupy..
        self.float32 = cp.float32
        self.int32 = cp.int32
        self.all = cp.all
        self.hamming = cp.hamming
        self.empty = cp.empty
        self.fabs = cp.fabs
        self.double = cp.double
        self.absolute = cp.absolute
        self.count_nonzero = cp.count_nonzero
        self.round = cp.round
        self.isnan = cp.isnan
        self.poly1d = cp.poly1d
        self.unwrap = cp.unwrap
        self.meshgrid = cp.meshgrid
        self.interp = cp.interp
        self.histogram = cp.histogram
        self.cumsum = cp.cumsum
        self.gradient = cp.gradient
        self.nonzero = cp.nonzero
        self.take = cp.take
        self.hstack = cp.hstack
        self.ones = cp.ones
        self.cos = cp.cos
        self.arctan = cp.arctan
        self.where = cp.where
        self.uint = cp.uint
        self.min = cp.min
        self.full = cp.full
        self.std = cp.std
        self.abs = cp.abs
        self.loadtxt = cp.loadtxt
        self.array = cp.array
        self.sqrt = cp.sqrt
        self.complex128 = cp.complex128
        self.int64 = cp.int64
        self.floor = cp.floor
        self.append = cp.append
        self.invert = cp.invert
        self.arange = cp.arange
        self.dot = cp.dot
        self.zeros_like = cp.zeros_like
        self.concatenate = cp.concatenate
        self.empty_like = cp.empty_like
        self.exp = cp.exp
        self.heaviside = cp.heaviside
        self.ascontiguousarray = cp.ascontiguousarray
        self.float64 = cp.float64
        self.zeros = cp.zeros
        self.complex64 = cp.complex64
        self.sum = cp.sum
        self.asarray = cp.asarray
        self.fliplr = cp.fliplr
        self.delete = cp.delete
        self.ceil = cp.ceil
        self.sin = cp.sin
        self.log10 = cp.log10
        self.diff = cp.diff
        self.uint32 = cp.uint32
        self.unique = cp.unique
        self.isinf = cp.isinf
        self.resize = cp.resize
        self.copy = cp.copy
        self.convolve = cp.convolve
        self.sign = cp.sign
        self.argsort = cp.argsort
        self.argmax = cp.argmax
        self.max = cp.max
        self.mean = cp.mean
        self.real = cp.real
        self.linspace = cp.linspace
        self.log = cp.log
        self.clip = cp.clip
        self.vectorize = cp.vectorize
        self.arcsin = cp.arcsin
        self.trapz = cp.trapz
        self.reshape = cp.reshape
        self.cosh = cp.cosh

        self.random = cp.random
        self.fft = cp.fft

        self.rfft = cp.fft.rfft
        self.irfft = cp.fft.irfft
        self.rfftfreq = cp.fft.rfftfreq


class CppBackend(__NumpyBackend):
    def __init__(self):
        """Mostly numpy backend, with some declarations from
        blond.utils.butils_wrap_cpp"""
        super().__init__()

        self.device = "CPU_CPP"

        from blond.utils import butils_wrap_cpp as _cpp

        self.kick = _cpp.kick
        self.rf_volt_comp = _cpp.rf_volt_comp
        self.drift = _cpp.drift
        self.slice_beam = _cpp.slice_beam
        self.slice_smooth = _cpp.slice_smooth
        self.linear_interp_kick = _cpp.linear_interp_kick
        self.synchrotron_radiation = _cpp.synchrotron_radiation
        self.synchrotron_radiation_full = _cpp.synchrotron_radiation_full
        self.music_track = _cpp.music_track
        self.music_track_multiturn = _cpp.music_track_multiturn
        self.fast_resonator = _cpp.fast_resonator
        self.beam_phase = _cpp.beam_phase
        self.beam_phase_fast = _cpp.beam_phase_fast
        self.sparse_histogram = _cpp.sparse_histogram
        self.distribution_from_tomoscope = _cpp.distribution_from_tomoscope
        #self.set_random_seed = _cpp.set_random_seed # fixme

        # elf.sin_cpp = _cpp.sin_cpp # todo add?
        # elf.cos_cpp = _cpp.cos_cpp # todo add?
        # elf.exp_cpp = _cpp.exp_cpp # todo add?
        # elf.mean_cpp = _cpp.mean_cpp # todo add?
        # elf.std_cpp = _cpp.std_cpp # todo add?
        # elf.where_cpp = _cpp.where_cpp # todo add?
        # elf.interp_cpp = _cpp.interp_cpp # todo add?
        # self.interp_const_space = _cpp.interp_const_space  # todo add?
        # self.interp_const_bin = _cpp.interp_const_bin
        # self.cumtrapz = _cpp.cumtrapz # not available in cupy
        # elf.trapz_cpp = _cpp.trapz_cpp # todo add?
        # elf.linspace_cpp = _cpp.linspace_cpp # todo add?
        # elf.argmin_cpp = _cpp.argmin_cpp # todo add?
        # elf.argmax_cpp = _cpp.argmax_cpp # todo add?
        self.convolve = _cpp.convolve
        # elf.arange_cpp = _cpp.arange_cpp # todo add?
        # elf.sum_cpp = _cpp.sum_cpp # todo add?
        # elf.sort_cpp = _cpp.sort_cpp # todo add?
        # elf.add_cpp = _cpp.add_cpp # todo add?
        # elf.mul_cpp = _cpp.mul_cpp # todo add?
        # self.random_normal = _cpp.random_normal # todo required


class NumbaBackend(__NumpyBackend):
    def __init__(self):
        """Mostly numpy backend, with some declarations from
        blond.utils.butils_wrap_numba"""
        super().__init__()

        self.device = "CPU_NU"

        from blond.utils import butils_wrap_numba as _nu
        import numpy as np
        self.rfft = np.fft.rfft
        self.irfft = np.fft.irfft
        self.rfftfreq = np.fft.rfftfreq

        self.kick = _nu.kick
        self.rf_volt_comp = _nu.rf_volt_comp
        self.drift = _nu.drift
        self.slice_beam = _nu.slice_beam
        self.slice_smooth = _nu.slice_smooth
        self.linear_interp_kick = _nu.linear_interp_kick
        self.synchrotron_radiation = _nu.synchrotron_radiation
        self.synchrotron_radiation_full = _nu.synchrotron_radiation_full
        self.music_track = _nu.music_track
        self.music_track_multiturn = _nu.music_track_multiturn
        self.fast_resonator = _nu.fast_resonator
        self.beam_phase = _nu.beam_phase
        self.beam_phase_fast = _nu.beam_phase_fast
        self.sparse_histogram = _nu.sparse_histogram
        self.distribution_from_tomoscope = _nu.distribution_from_tomoscope


class PyBackend(__NumpyBackend):
    """Mostly numpy backend, with some declarations from
    blond.utils.butils_wrap_python"""

    def __init__(self):
        super().__init__()

        self.device = "CPU_PY"

        from blond.utils import butils_wrap_python as _py

        self.kick = _py.kick
        self.rf_volt_comp = _py.rf_volt_comp
        self.drift = _py.drift
        self.slice_beam = _py.slice_beam
        self.slice_smooth = _py.slice_smooth
        self.linear_interp_kick = _py.linear_interp_kick
        self.synchrotron_radiation = _py.synchrotron_radiation
        self.synchrotron_radiation_full = _py.synchrotron_radiation_full
        self.music_track = _py.music_track
        self.music_track_multiturn = _py.music_track_multiturn
        self.fast_resonator = _py.fast_resonator
        self.beam_phase = _py.beam_phase
        self.beam_phase_fast = _py.beam_phase_fast
        self.sparse_histogram = _py.sparse_histogram
        self.distribution_from_tomoscope = _py.distribution_from_tomoscope
        self.set_random_seed = _py.set_random_seed

def not_implemented():
    """
    Used as a placeholder for development
    """
    from warnings import warn
    warn("This method is a placeholder and must be replaced by an actual method!", DeprecationWarning)
    raise NotImplementedError("Contact BLonD developers !")

class GpuBackend(__CupyBackend):
    def __init__(self):
        super().__init__()
        self.device = 'GPU'

        from blond.gpu import butils_wrap_cupy
        import cupy as cp

        self.rfft = cp.fft.rfft
        self.irfft = cp.fft.irfft
        self.rfftfreq = cp.fft.rfftfreq
        self.convolve = cp.convolve
        # 'convolve' = butils_wrap_cupy.convolve
        self.beam_phase = butils_wrap_cupy.beam_phase
        self.beam_phase_fast = butils_wrap_cupy.beam_phase_fast
        self.kick = butils_wrap_cupy.kick
        self.rf_volt_comp = butils_wrap_cupy.rf_volt_comp
        self.drift = butils_wrap_cupy.drift
        self.linear_interp_kick = butils_wrap_cupy.linear_interp_kick
        # 'LIKick_n_drift' = butils_wrap_cupy.linear_interp_kick_drift
        self.synchrotron_radiation = butils_wrap_cupy.synchrotron_radiation
        self.synchrotron_radiation_full = butils_wrap_cupy.synchrotron_radiation_full
        self.slice_beam = butils_wrap_cupy.slice_beam
        # 'interp_const_space' = butils_wrap_cupy.interp
        # self.interp_const_space = cp.interp  # todo add?

        self.sparse_histogram = not_implemented # todo implement !
        self.distribution_from_tomoscope = not_implemented # todo implement !
        self.fast_resonator = not_implemented # todo implement !
        self.rf_volt_comp = not_implemented # todo implement !
        self.slice_smooth = not_implemented # todo implement !
        self.music_track_multiturn = not_implemented # todo implement !
        self.music_track = not_implemented # todo implement !

__check_backends = [CppBackend(), NumbaBackend(), PyBackend()]
try:
    import cupy

    __check_backends.append(GpuBackend())
except ImportError:
    pass

for __my_backend in __check_backends:
    __my_backend.verify_backend()


# this line controls static type hints of bmath
# static type hints are done to PyBackend
class AnyBackend(PyBackend):
    def __init__(self):
        """Initialized as PyBackend (Numpy based)

        Notes
        -----
        Might be changed to other backand using use_xxx() method
        """
        super().__init__()
