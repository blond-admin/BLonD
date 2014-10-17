'''
@author: Danilo Quartullo
'''
# MAKE SURE YOU HAVE GCC 4.8.1 OR LATER VERSIONS ON YOUR SYSTEM LINKED TO YOUR SYSTEM PATH
# IF YOU ARE ON CERN-LXPLUS YOU CAN TYPE FROM CONSOLE source /afs/cern.ch/sw/lcg/contrib/gcc/4.8.1/x86_64-slc6/setup_cython.sh

import os
import sys
import subprocess
import ctypes

# EXAMPLE FLAGS: -Ofast -std=c++11 -ftree-vectorize -mfma4
flags = '-Ofast -std=c++11 -ftree-vectorize'
list_cpp_files = 'cpp_routines/histogram.cpp cpp_routines/kick.cpp'

if "lin" in sys.platform:
    subprocess.Popen("g++ -o cpp_routines/result.so -shared " + flags + " -fPIC " + list_cpp_files, shell = True, executable = "/bin/bash")
elif "win" in sys.platform:
    x = os.getcwd()
    os.system('g++ -o '+ x +'\\cpp_routines\\result.dll -shared ' + flags + ' ' + x + '\\' + list_cpp_files)
else:
    print "You have not a Windows or Linux operating system. Aborting..."
    sys.exit()

path = os.path.realpath(__file__)
parent_path = os.sep.join(path.split(os.sep)[:-1])
if "lin" in sys.platform:
    libfib=ctypes.CDLL(parent_path+'/cpp_routines/result.so')
elif "win" in sys.platform:
    libfib=ctypes.CDLL(parent_path+'\\cpp_routines\\result.dll')
else:
    sys.exit()

print "Setup file run successfully!"

    