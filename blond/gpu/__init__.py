from ..utils import bmath as bm
import os 
blocks = int(os.environ.get('BLOCKS', 30))
threads = int(os.environ.get('THREADS', 1024))
try:
    grid_size = (blocks, 1, 1)
    block_size = (threads, 1, 1)
except:
    print("error")