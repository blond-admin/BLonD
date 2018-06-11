import os
import ctypes

path = os.path.realpath(__file__)
parent_path = os.sep.join(path.split(os.sep)[:-1])

if ('posix' in os.name):
    libblond = ctypes.CDLL(os.path.join(parent_path, 'cpp_routines/libblond.so'))
elif ('win' in sys.platform):
    libblond = ctypes.CDLL(os.path.join(parent_path, 'cpp_routines\\libblond.dll'))
else:
    print('YOU DO NOT HAVE A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING...')
    sys.exit()