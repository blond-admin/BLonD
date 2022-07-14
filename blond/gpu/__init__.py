# from ..utils import bmath as bm
# import os 

# __default_blocks = 2*bm.gpuDev().attributes['MultiProcessorCount']
# __default_threads = bm.gpuDev().attributes['MaxThreadsPerBlock']

# __blocks = int(os.environ.get('GPU_BLOCKS', __default_blocks))
# __threads = int(os.environ.get('GPU_THREADS', __default_threads))
# try:
#     grid_size = (__blocks, 1, 1)
#     block_size = (__threads, 1, 1)
# except:
#     print("Error when initializing the block and grid sizes")
