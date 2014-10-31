
# Copyright 2014 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''

@author: Danilo Quartullo
'''

# MAKE SURE YOU HAVE GCC 4.8.1 OR LATER VERSIONS ON YOUR SYSTEM LINKED TO YOUR
# SYSTEM PATH.IF YOU ARE ON CERN-LXPLUS YOU CAN TYPE FROM CONSOLE 
# source /afs/cern.ch/sw/lcg/contrib/gcc/4.8.1/x86_64-slc6/setup.sh
# TO GET GCC 4.8.1 64 BIT. IN GENERAL IT IS ADVISED TO USE PYTHON 64 BIT PLUS 
# GCC 64 BIT.

import os
import sys
import subprocess
import ctypes

# CHOOSE THE FLAG THAT YOU WANT
# EXAMPLE FLAGS: -Ofast -std=c++11 -ftree-vectorizer-verbose=1
#                -mfma4 -fopenmp
flags = '-Ofast -std=c++11 -ftree-vectorizer-verbose=1'

# CHOOSE THE cpp FILES THAT YOU WANT TO COMPILE
list_cpp_files = 'cpp_routines/histogram.cpp cpp_routines/kicks.cpp cpp_routines/drift_simple.cpp'

# DON'T TOUCH THE CODE FROM HERE TILL THE END OF THIS SCRIPT!
if __name__ == "__main__":
    
    if "lin" in sys.platform:
        subprocess.Popen("rm -rf cpp_routines/*.so", shell = True, executable = "/bin/bash")
        x = os.getcwd()
        os.system('g++ -o '+ x +'/cpp_routines/result.so -shared ' + flags + ' -fPIC ' + x + '/' + list_cpp_files)
        print ""
        print ""
        print "IF THE COMPILATION IS CORRECT A FILE NAMED result.so SHOULD APPEAR IN THE cpp_routines FOLDER." 
        print "OTHERWISE YOU HAVE TO CORRECT THE ERRORS AND COMPILE AGAIN."
        sys.exit()
    
    elif "win" in sys.platform:
        os.system('del /s/q '+ os.getcwd() +'\\cpp_routines\\*.dll')
        x = os.getcwd()
        os.system('g++ -o '+ x +'\\cpp_routines\\result.dll -shared ' + flags + ' ' + x + '\\' + list_cpp_files)
        print ""
        print ""
        print "IF THE COMPILATION IS CORRECT A FILE NAMED result.dll SHOULD APPEAR IN THE cpp_routines FOLDER." 
        print "OTHERWISE YOU HAVE TO CORRECT THE ERRORS AND COMPILE AGAIN."
        sys.exit()
    
    else:
        print "YOU DO NOT HAVE A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING..."
        sys.exit()

path = os.path.realpath(__file__)
parent_path = os.sep.join(path.split(os.sep)[:-1])

if "lin" in sys.platform:
    libfib=ctypes.CDLL(parent_path+'/cpp_routines/result.so')
elif "win" in sys.platform:
    libfib=ctypes.CDLL(parent_path+'\\cpp_routines\\result.dll')
else:
    print "YOU DO NOT HAVE A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING..."
    sys.exit()


    

