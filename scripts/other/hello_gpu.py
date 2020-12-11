from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as drv
import pycuda.tools
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule
from pycuda import gpuarray

import pycuda.autoinit

devdata = pycuda.tools.DeviceData()

occ = pycuda.tools.OccupancyRecord(devdata, 1024, shared_mem=0, registers=28)
print(occ.occupancy)