
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''

@author: Danilo Quartullo, Konstantinos Iliakis
'''

# MAKE SURE YOU HAVE GCC 4.8.1 OR LATER VERSIONS ON YOUR SYSTEM LINKED TO YOUR
# SYSTEM PATH.IF YOU ARE ON CERN-LXPLUS YOU CAN TYPE FROM CONSOLE
# source /afs/cern.ch/sw/lcg/contrib/gcc/4.8.1/x86_64-slc6/setup.sh
# TO GET GCC 4.8.1 64 BIT. IN GENERAL IT IS ADVISED TO USE PYTHON 64 BIT PLUS
# GCC 64 BIT.

from __future__ import print_function
import os
import sys
import subprocess
import ctypes
import argparse

path = os.path.realpath(__file__)
basepath = os.sep.join(path.split(os.sep)[:-1])

parser = argparse.ArgumentParser(description='Run python setup_cpp.py to'
                                 ' compile the cpp routines needed from BLonD')

parser.add_argument('-p', '--parallel',
                    default=False, action='store_true',
                    help='Produce Multi-threaded code. Use the environment'
                    ' variable OMP_NUM_THREADS=xx to control the number of'
                    ' threads that will be used.'
                    ' Default: Serial code')

parser.add_argument('-b', '--boost', type=str, nargs='?', const='',
                    help='Use boost library to speedup synchrotron radiation'
                    ' routines. If the installation path of boost differs'
                    ' from the default, you have to pass it as an argument.'
                    ' Default: Boost will not be used')

parser.add_argument('-c', '--compiler', type=str, default='g++',
                    help='C++ compiler that will be used to compile the'
                    ' source files. Default: g++')

parser.add_argument('--with-fftw', action='store_true',
                    help='Use the FFTs from FFTW3.')

parser.add_argument('--with-fftw-threads', action='store_true',
                    help='Use the multi-threaded FFTs from FFTW3.')

parser.add_argument('--with-fftw-omp', action='store_true',
                    help='Use the OMP FFTs from FFTW3.')

parser.add_argument('--with-fftw-lib', type=str,
                    help='Path to the FFTW3 library (.so, .dll).')

parser.add_argument('--with-fftw-header', type=str,
                    help='Path to the FFTW3 header files.')

parser.add_argument('--flags', type=str, default='',
                    help='Additional compile flags.')

parser.add_argument('--libs', type=str, default='',
                    help='Any extra libraries needed to compile')

# Additional libs needed to compile the blond library
libs = []


# EXAMPLE FLAGS: -Ofast -std=c++11 -fopt-info-vec -march=native
#                -mfma4 -fopenmp -ftree-vectorizer-verbose=1
cflags = ['-O3', '-ffast-math', '-std=c++11', '-shared']

cpp_files = [
    os.path.join(basepath, 'cpp_routines/kick.cpp'),
    os.path.join(basepath, 'cpp_routines/drift.cpp'),
    os.path.join(basepath, 'cpp_routines/linear_interp_kick.cpp'),
    os.path.join(basepath, 'cpp_routines/histogram.cpp'),
    os.path.join(basepath, 'cpp_routines/music_track.cpp'),
    os.path.join(basepath, 'cpp_routines/blondmath.cpp'),
    os.path.join(basepath, 'cpp_routines/fast_resonator.cpp'),
    os.path.join(basepath, 'cpp_routines/beam_phase.cpp'),
    os.path.join(basepath, 'cpp_routines/fft.cpp'),
    os.path.join(basepath, 'toolbox/tomoscope.cpp'),
    os.path.join(basepath, 'synchrotron_radiation/synchrotron_radiation.cpp'),
    os.path.join(basepath, 'beam/sparse_histogram.cpp'),
]


if (__name__ == "__main__"):
    args = parser.parse_args()
    boost_path = None
    with_fftw = args.with_fftw or args.with_fftw_threads or args.with_fftw_omp or \
        (args.with_fftw_lib is not None) or (args.with_fftw_header is not None)
    if(args.boost is not None):
        if(args.boost):
            boost_path = os.path.abspath(args.boost)
        else:
            boost_path = ''
        cflags += ['-I', boost_path, '-DBOOST']
    compiler = args.compiler

    if (args.libs):
        libs = args.libs.split()

    if (args.parallel):
        cflags += ['-fopenmp', '-DPARALLEL', '-D_GLIBCXX_PARALLEL']

    if (args.flags):
        cflags += args.flags.split()

    if with_fftw:
        cflags += ['-DUSEFFTW3']
        if args.with_fftw_lib is not None:
            libs += ['-L', args.with_fftw_lib]
        if args.with_fftw_header is not None:
            cflags += ['-I', args.with_fftw_header]
        if 'win' in sys.platform:
            libs += ['-lfftw3-3']
        else:
            libs += ['-lfftw3']
            if args.with_fftw_omp:
                cflags += ['-DFFTW3PARALLEL']
                libs += ['-lfftw3_omp']
            elif args.with_fftw_threads:
                cflags += ['-DFFTW3PARALLEL']
                libs += ['-lfftw3_threads']

    if ('posix' in os.name):
        cflags += ['-fPIC']
        libname = os.path.join(basepath, 'cpp_routines/libblond.so')
    elif ('win' in sys.platform):
        libname = os.path.join(basepath, 'cpp_routines/libblond.dll')
    else:
        print(
            'YOU ARE NOT USING A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING...')
        sys.exit(-1)

    command = [compiler] + cflags + ['-o', libname] + cpp_files + libs

    print('Enable Multi-threaded code: ', args.parallel)
    print('Using boost: ', args.boost is not None)
    if args.boost is not None:
        print('Boost installation path: ', boost_path)
    print('With FFTW3: ', with_fftw)
    if with_fftw:
        print('Parallel FFTW3:', args.with_fftw_threads or args.with_fftw_omp)
    if args.with_fftw_lib or args.with_fftw_header:
        print('FFTW3 Library path: ', args.with_fftw_lib)
        print('FFTW3 Headers path: ', args.with_fftw_header)
    print('C++ Compiler: ', compiler)
    print('Compiler flags: ', ' '.join(cflags))
    print('Extra libraries: ', ' '.join(libs))
    subprocess.call([compiler, '--version'])

    try:
        os.remove(libname)
    except OSError as e:
        pass

    subprocess.call(command)

    try:
        libblond = ctypes.CDLL(libname)
        print('\nThe blond library has been successfully compiled.')
    except Exception as e:
        print('\nCompilation failed.')
        print(e)
