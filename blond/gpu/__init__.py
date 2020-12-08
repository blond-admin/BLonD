from ..utils import bmath as bm
import os 

default_blocks = 2*bm.gpuDev().MULTIPROCESSOR_COUNT
default_threads = bm.gpuDev().MAX_THREADS_PER_BLOCK
# default_blocks = 256
# default_threads = 1024

blocks = int(os.environ.get('BLOCKS', default_blocks))
threads = int(os.environ.get('THREADS', default_threads))
try:
    grid_size = (blocks, 1, 1)
    block_size = (threads, 1, 1)
except:
    print("error")