import numpy as np

import os
import sys
import subprocess
import cython_gsl

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# Remember that you have to launch this setup.py script from console with the exact following syntax "python setup.py cleanall build_ext --inplace"

args = sys.argv[1:]

# Make a `cleanall` rule to get rid of previously created cython files

if "cleanall" in args:
    print "Deleting cython files..."
    if "lin" in sys.platform:
        subprocess.Popen("rm -rf build", shell = True, executable = "/bin/bash")
        subprocess.Popen("rm -rf *.c", shell = True, executable = "/bin/bash")
        subprocess.Popen("rm -rf *.so", shell = True, executable = "/bin/bash")
        sys.argv[1] = "clean"
    elif "win" in sys.platform:
        os.system('rd /s/q '+ os.getcwd() +'\\build')
        os.system('del /s/q '+ os.getcwd() +'\\*.c')
        os.system('del /s/q '+ os.getcwd() +'\\*.html')
        os.system('del /s/q '+ os.getcwd() +'\\*.pyd')
        sys.argv[1] = "clean"
    else:
        print "You have not a Windows or Linux operating system. Aborting..."
        sys.exit()

# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext") + 1, "--inplace")

# Only build for 64-bit target
# os.environ['ARCHFLAGS'] = "-arch x86_64"

# Set up extension and build
cy_ext = [
        # Extension("beams.bunch",
        #           ["beams/bunch.pyx"],
        #          include_dirs=[np.get_include()],
        #          #extra_compile_args=["-g"],
        #          #extra_link_args=["-g"],
        #          libraries=["m"],
        #          library_dirs=[],
        #          ),
        Extension("solvers.grid_functions",
                 ["solvers/grid_functions.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 extra_compile_args=["-fopenmp"],     # If your PC is Windows-based you can use the TDM-GCC compiler which supports OpenMP
                 extra_link_args=["-fopenmp"],
                 ),
        Extension("cobra_functions.stats",
                 ["cobra_functions/stats.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 ),
        Extension("cobra_functions.random",
                 ["cobra_functions/random.pyx"],
                 include_dirs=[np.get_include(), cython_gsl.get_cython_include_dir()],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 library_dirs=[], libraries=["gsl", "gslcblas"],
                 ),
        Extension("solvers.compute_potential_fgreenm2m",
                 ["solvers/compute_potential_fgreenm2m.pyx"],
                  include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 ),
        Extension("cobra_functions.interp1d",
                 ["cobra_functions/interp1d.pyx"],
                  include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 )
          ]

    # include_dirs = [cython_gsl.get_include()],
    # cmdclass = {'build_ext': build_ext},
    # ext_modules = [Extension("my_cython_script",
    #                          ["src/my_cython_script.pyx"],
    #                          libraries=cython_gsl.get_libraries(),
    #                          library_dirs=[cython_gsl.get_library_dir()],
    #                          include_dirs=[cython_gsl.get_cython_include_dir()])]
    
cy_ext_options = {"compiler_directives": {"profile": True}, "annotate": True}

setup(cmdclass={'build_ext': build_ext},
      ext_modules=cythonize(cy_ext, **cy_ext_options),)

# setup(
#     name="libBunch",
#     ext_modules=cythonize(extensions),
# )
