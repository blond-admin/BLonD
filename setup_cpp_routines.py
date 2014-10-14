# MAKE SURE YOU HAVE GCC 4.8.1 OR LATER VERSIONS ON YOUR SYSTEM LINKED TO YOUR SYSTEM PATH
# IF YOU ARE ON CERN-LXPLUS YOU CAN TYPE FROM CONSOLE source /afs/cern.ch/sw/lcg/contrib/gcc/4.8.1/x86_64-slc6/setup.sh

import os
import sys
import subprocess

# EXAMPLE FLAGS: -Ofast -std=c++11 -ftree-vectorize
flags = '-Ofast -std=c++11 -ftree-vectorize'

if "lin" in sys.platform:
    subprocess.Popen("g++ -o cpp_routines/histogram.so -shared " + flags + " -fPIC cpp_routines/histogram.cpp", shell = True, executable = "/bin/bash")
    x = os.getcwd() + '/cpp_routines'
    if x in os.getenv('LD_LIBRARY_PATH'):
        pass
    else:
        subprocess.Popen("echo 'export LD_LIBRARY_PATH="+x+":$LD_LIBRARY_PATH'  >> ~/.bashrc", shell = True, executable = "/bin/bash")
elif "win" in sys.platform:
    os.system('g++ -o '+ os.getcwd() +'\\cpp_routines\\histogram.dll -shared ' + flags + ' ' + os.getcwd() +'\\cpp_routines\\histogram.cpp')
    x = os.getcwd() + '\\cpp_routines'
    if x in os.getenv('PATH'):
        pass
    else:
        os.system('setx Path "'+x+';%Path%"')
        print 'Path added! Restart the environment!'
else:
    print "You have not a Windows or Linux operating system. Aborting..."
    sys.exit()
    