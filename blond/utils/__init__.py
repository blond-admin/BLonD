import ctypes as ct
import numpy as np

class PrecisionClass:
    __instance = None

    def __init__(self, precision='double'):
        if PrecisionClass.__instance is not None:
            return
        else:
            PrecisionClass.__instance = self

        self.set(precision)

    def set(self, precision='double'):
        if precision in ['single', 's', '32', 'float32', 'float', 'f']:
            self.str = 'float32'
            self.real_t = np.float32
            self.c_real_t = ct.c_float
            self.complex_t = np.complex64
            self.num = 1
        elif precision in ['double', 'd', '64', 'float64']:
            self.str = 'float64'
            self.real_t = np.float64
            self.c_real_t = ct.c_double
            self.complex_t = np.complex128
            self.num = 2



class c_complex128(ct.Structure):
    # Complex number, compatible with std::complex layout
    _fields_ = [("real", ct.c_double), ("imag", ct.c_double)]

    def __init__(self, pycomplex):
        # Init from Python complex
        self.real = pycomplex.real.astype(np.float64, order='C')
        self.imag = pycomplex.imag.astype(np.float64, order='C')

    def to_complex(self):
        # Convert to Python complex
        return self.real + (1.j) * self.imag


class c_complex64(ct.Structure):
    # Complex number, compatible with std::complex layout
    _fields_ = [("real", ct.c_float), ("imag", ct.c_float)]

    def __init__(self, pycomplex):
        # Init from Python complex
        self.real = pycomplex.real.astype(np.float32, order='C')
        self.imag = pycomplex.imag.astype(np.float32, order='C')

    def to_complex(self):
        # Convert to Python complex
        return self.real + (1.j) * self.imag

def c_real(x):
    global precision
    if precision.num == 1:
        return ct.c_float(x)
    else:
        return ct.c_double(x)

def c_complex(x):
    global precision
    if precision.num == 1:
        return c_complex64(x)
    else:
        return c_complex128(x)


class GPUDev:
    __instance = None

    def __init__(self, _gpu_num=0):
        if GPUDev.__instance is not None:
            return
            # raise Exception("The GPUDev class is a singleton!")
        else:
            GPUDev.__instance = self

        import cupy as cp
        self.id = _gpu_num
        self.dev = cp.cuda.Device(self.id)
        self.dev.use()

        self.name = cp.cuda.runtime.getDeviceProperties(self.dev)['name']
        self.attributes = self.dev.attributes
        self.properties = cp.cuda.runtime.getDeviceProperties(self.dev)

        # set the default grid and block sizes
        default_blocks = 2 * self.attributes['MultiProcessorCount']
        default_threads = self.attributes['MaxThreadsPerBlock']
        blocks = int(os.environ.get('GPU_BLOCKS', default_blocks))
        threads = int(os.environ.get('GPU_THREADS', default_threads))
        self.grid_size = (blocks, 1, 1)
        self.block_size = (threads, 1, 1)

        this_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
        comp_capability = self.dev.compute_capability

        if precision.num == 1:
            self.mod = cp.RawModule(path=os.path.join(
                this_dir, f'../gpu/cuda_kernels/kernels_single_sm_{comp_capability}.cubin'))
        else:
            self.mod = cp.RawModule(path=os.path.join(
                this_dir, f'../gpu/cuda_kernels/kernels_double_sm_{comp_capability}.cubin'))

    def report_attributes(self):
        # Saves into a file all the device attributes
        with open(f'{self.name}-attributes.txt', 'w') as f:
            for k, v in self.attributes.items():
                f.write(f"{k}:{v}\n")

    def func(self, name):
        return self.mod.get_function(name)

    def __del__(self):
        from .bmath import use_cpu
        use_cpu()


# By default use double precision
precision = PrecisionClass('double')
gpu_dev = None