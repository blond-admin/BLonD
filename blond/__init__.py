import os
import ctypes
import sys

path = os.path.realpath(__file__)
basepath = os.sep.join(path.split(os.sep)[:-1])

try:
    if ('posix' in os.name):
        libblond = ctypes.CDLL(os.path.join(
            basepath, 'cpp_routines/libblond.so'))
    elif ('win' in sys.platform):
        libblond = ctypes.CDLL(os.path.join(
            basepath, 'cpp_routines\\libblond.dll'))
    else:
        print('YOU DO NOT HAVE A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING...')
        sys.exit()
except OSError as e:
    print("""
        Warning: The compiled blond library was not found.
        You can safely ignore this warning if you are in 
        the process of compiling the library.""")
