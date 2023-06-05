'''
Basic methods and objects related to the GPU computational core.

@author: Konstantinos Iliakis
@date: 25.05.2023
'''

import os

from ..utils import precision


class GPUDev:
    """GPU Device object, singleton class

    Returns:
        _type_: _description_
    """
    __instance = None

    def __init__(self):
        """Initializes the GPU device object.

        Args:
            _gpu_num (int, optional): The device id. Defaults to 0.
        """
        if GPUDev.__instance is not None:
            return
        GPUDev.__instance = self
        self.id = -1
        self.dev = None
        self.name = ''
        self.attributes = {}
        self.attributes = {}
        self.grid_size = 0
        self.block_size = 0
        self.mod = None

    def set(self, _gpu_num=0):
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
        """ Stores in file device attributes
        """
        # Saves into a file all the device attributes
        with open(f'{self.name}-attributes.txt', 'w') as file:
            for key, value in self.attributes.items():
                file.write(f"{key}:{value}\n")

    def func(self, name):
        """Get kernel from kernel module

        Args:
            name (string): Kernel name

        Returns:
            _type_: _description_
        """
        return self.mod.get_function(name)


# Initialize empty GPU object
GPU_DEV = GPUDev()
