import os
import ctypes
import sys

path = os.path.realpath(__file__)
basepath = os.sep.join(path.split(os.sep)[:-1])
libblond = None
libblond_path = os.environ.get('LIBBLOND', None)
try:
    if 'posix' in os.name:
        if libblond_path:
            libblond_path = os.path.abspath(libblond_path)
        else:
            libblond_path = os.path.join(basepath, 'cpp_routines/libblond.so')
        libblond = ctypes.CDLL(libblond_path)
    elif 'win' in sys.platform:
        if libblond_path:
            libblond_path = os.path.abspath(libblond_path)
        else:
            libblond_path = os.path.join(basepath, 'cpp_routines/libblond.dll')
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(os.path.dirname(libblond_path))
            libblond = ctypes.CDLL(libblond_path, winmode=0)
        else:
            libblond = ctypes.CDLL(libblond_path)
    else:
        print('YOU DO NOT HAVE A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING...')
        sys.exit()
except OSError as e:
    print("""
        Warning: The compiled blond library was not found.
        You can safely ignore this warning if you are in 
        the process of compiling the library.""")

