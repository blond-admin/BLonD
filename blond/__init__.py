'''
**__init__.py: Locate and load the compiled blond library.**

:Authors: Konstantinos Iliakis**
'''
import ctypes
import os
import sys

LIBBLOND = None

def init():
    '''Locates and initializes the blond compiled library
    '''
    global LIBBLOND
    libblond_path = os.environ.get('LIBBLOND', None)
    path = os.path.realpath(__file__)
    basepath = os.sep.join(path.split(os.sep)[:-1])
    try:
        if 'posix' in os.name:
            if libblond_path:
                libblond_path = os.path.abspath(libblond_path)
            else:
                libblond_path = os.path.join(basepath, 'cpp_routines/libblond.so')
            LIBBLOND = ctypes.CDLL(libblond_path)
        elif 'win' in sys.platform:
            if libblond_path:
                libblond_path = os.path.abspath(libblond_path)
            else:
                libblond_path = os.path.join(basepath, 'cpp_routines/libblond.dll')
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(os.path.dirname(libblond_path))
                LIBBLOND = ctypes.CDLL(libblond_path, winmode=0)
            else:
                LIBBLOND = ctypes.CDLL(libblond_path)
        else:
            print('YOU DO NOT HAVE A WINDOWS OR UNIX OPERATING SYSTEM. ABORTING.')
            sys.exit()
    except OSError:
        # Silently pass. The python backend will be used.
        pass
        # print("""
        #     Warning: The compiled blond library was not found.
        #     The python routines will be used instead.
        #     """)

init()
