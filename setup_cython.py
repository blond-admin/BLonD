
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
@author: Danilo Quartullo
'''
# LAUNCH THIS SCRIPT WITH "python setup_cython.py build_ext --inplace"
# IF YOU HAVE A VERSION 64 BIT ON WINDOWS ADD THE FLAG "-D MS_WIN64"

import numpy as np
import os
import sys
import subprocess
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [Extension("cython_routines.kick", ["cython_routines/kick.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                extra_compile_args=['-Ofast', '-ftree-vectorize'],
                extra_link_args=['-Ofast', '-ftree-vectorize'])]

setup(ext_modules=cythonize(extensions))

if "lin" in sys.platform:
    subprocess.Popen("rm -rf build", shell = True, executable = "/bin/bash")
    subprocess.Popen("rm -rf cython_routines/*.c", shell = True, executable = "/bin/bash")
elif "win" in sys.platform:
    os.system('rd /s/q '+ os.getcwd() +'\\build')
    os.system('del /s/q '+ os.getcwd() +'\\cython_routines\\*.c')
else:
    print "You have not a Windows or Linux operating system. Aborting..."
    sys.exit()


