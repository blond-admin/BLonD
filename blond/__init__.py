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
                libblond_path = os.path.join(
                    basepath, 'cpp_routines/libblond.so')
            LIBBLOND = ctypes.CDLL(libblond_path)
        elif 'win' in sys.platform:
            if libblond_path:
                libblond_path = os.path.abspath(libblond_path)
            else:
                libblond_path = os.path.join(
                    basepath, 'cpp_routines/libblond.dll')
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


def test():
    """A simple test to verify that blond has been correctly installed.
    """
    try:
        # Test that all the blond modules exist
        # Verify all the dependencies are also installed
        import __future__
        from packaging import version
        import platform
        import h5py
        import matplotlib
        import mpmath
        import numpy
        import scipy

        assert version.parse(matplotlib.__version__) >= version.parse('3.7'), f'matplotlib version {matplotlib.__version__} does not meet minimum required version 3.7'
        assert version.parse(platform.python_version()) >= version.parse('3.8'), f'python version {platform.python_version()} does not meet minimum required version 3.8'
        assert version.parse(numpy.__version__) >= version.parse('1.20'), 'numpy version {numpy.__version__} does not meet minimum required version 1.20'

        from blond import (beam, impedances, input_parameters, llrf, monitors,
                           plots, synchrotron_radiation, toolbox, trackers,
                           utils)
        # This should report the backend that is used
        from blond.utils import bmath
        print('\nBLonD installed successfully!')
        return 0
    except Exception as exception:
        print('\nBLonD installation verification failed: ', exception)
        return -1
