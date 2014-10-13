# MAKE SURE YOU HAVE GCC 4.8.1 OR LATER VERSIONS ON YOUR SYSTEM LINKED TO YOUR SYSTEM PATH
# IF YOU ARE ON CERN-LXPLUS YOU CAN LUNCH TYPE FROM CONSOLE source /afs/cern.ch/sw/lcg/contrib/gcc/4.8.1/x86_64-slc6/setup.sh
# THE FOLDER CPP_ROUTINES MUST BE LINKED ALSO TO YOUR SYSTEM PATH

import os
import sys
import subprocess

flags = '-Ofast -std=c++11 -ftree-vectorizer-verbose=1'

if "lin" in sys.platform:
    subprocess.Popen("g++ -o cpp_routines/histogram.so -shared " + flags + " -fPIC cpp_routines/histogram.cpp", shell = True, executable = "/bin/bash")
elif "win" in sys.platform:
    os.system('g++ -o '+ os.getcwd() +'\\cpp_routines\\histogram.dll -shared ' + flags + ' ' + os.getcwd() +'\\cpp_routines\\histogram.cpp')
else:
    print "You have not a Windows or Linux operating system. Aborting..."
    sys.exit()

