import sys
import os
import ctypes
from blond._version import __version__
import distutils
from setuptools import setup, find_packages
from distutils.ccompiler import new_compiler
from distutils import log, dir_util

# from Cython.Distutils import build_ext

log.set_verbosity(log.DEBUG)  # Set DEBUG level

compiler = 'g++'
build_dir = 'blond/build'
cflags = ['-Ofast', '-std=c++11', '-fPIC', '-fopenmp']
lflags = ['-fopenmp']

libs = {
    'blond': {'cfiles': [
        # 'blond/cpp_routines/mean_std_whereint.cpp',
        'blond/cpp_routines/kick.cpp',
        'blond/cpp_routines/drift.cpp',
        'blond/cpp_routines/linear_interp_kick.cpp',
        'blond/toolbox/tomoscope.cpp',
        # 'blond/cpp_routines/convolution.cpp',
        'blond/cpp_routines/music_track.cpp',
        'blond/cpp_routines/fast_resonator.cpp',
        'blond/beam/sparse_histogram.cpp',
        'blond/synchrotron_radiation/synchrotron_radiation.cpp',
        'blond/cpp_routines/blondmath.cpp'],
        'cflags': cflags, 'lflags': lflags, 'output_dir': build_dir}

    # 'syncrad': {'cfiles': ['BLonD/synchrotron_radiation/synchrotron_radiation.cpp'],
    #             'cflags': cflags, 'lflags': lflags, 'output_dir': build_dir},

    # 'blondmath': {'cfiles': ['blond/cpp_routines/blondmath.cpp'],
    # 'cflags': cflags, 'lflags': lflags, 'output_dir': build_dir},

    # 'blondphysics': {'cfiles': ['BLonD/cpp_routines/mean_std_whereint.cpp',
    #                             'BLonD/cpp_routines/kick.cpp',
    #                             'BLonD/cpp_routines/drift.cpp',
    #                             'BLonD/cpp_routines/linear_interp_kick.cpp',
    #                             'BLonD/toolbox/tomoscope.cpp',
    #                             'BLonD/cpp_routines/convolution.cpp',
    #                             'BLonD/cpp_routines/music_track.cpp',
    #                             'BLonD/cpp_routines/fast_resonator.cpp',
    #                             'BLonD/beam/sparse_histogram.cpp',
    #                             'BLonD/synchrotron_radiation/synchrotron_radiation.cpp'],
    #                  'cflags': cflags, 'lflags': lflags,
    #                  'output_dir': build_dir}
}


class BuildCommand(distutils.cmd.Command):
    """A custom command to run compile all C/C++ source files."""

    description = 'Build all the shared libraries'
    user_options = [
        # The format is (long option, short option, description).
        # ('build-rcfile=', None, 'path to build config file'),
    ]

    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        # self.build_rcfile = ''
        pass

    def finalize_options(self):
        """Post-process options."""
        pass
        # if self.build_rcfile:
        #     assert os.path.exists(
        #         self.build_rcfile), ('Build config file %s does not exist.' % self.build_rcfile)

    def run(self):
        """Run command."""
        cc = new_compiler(compiler='unix', verbose=0, dry_run=0, force=0)
        cc.set_executables(compiler=compiler, compiler_so=compiler,
                           compiler_cxx=compiler,
                           linker_so=[compiler, '-shared'])
        for lib, values in libs.items():
            objects = cc.object_filenames(values['cfiles'],
                                          output_dir=values['output_dir'])
            libname = cc.library_filename(lib, lib_type='shared',
                                          output_dir=values['output_dir'])
            cc.compile(values['cfiles'], output_dir=values['output_dir'],
                       extra_preargs=values['cflags'])
            cc.link(lib, objects, libname, extra_preargs=values['lflags'])
            command = 'lib'+lib + '=ctypes.CDLL("'+libname+'")'
            exec('global lib' + lib)
            exec(command)


# if __name__ == "__main__":
#     args = sys.argv[1:]
#     if 'cleanall' in args:
#         dir_util.remove_tree(build_dir)
#         sys.argv = [sys.argv[0], 'clean']
    # else:
    #     build()

setup(name='blond',
      version=__version__,
      description='CERN code for the simulation of longitudinal beam dynamics in synchrotrons.',
      keywords='Beam Longitudinal Dynamics Synchrotrons CERN',
      author='Helga Timko',
      author_email='helga.timko@cern.ch',
      maintainer='Konstantinos Iliakis',
      maintainer_email='konstantinos.iliakis@cern.ch',
      long_description=open('README.md').read(),
      url='https://github.com/blond-admin/BLonD',
      # cmdclass={'build_ext': BuildCommand},
      # package_dir={'': 'BLonD'},
      # packages=['BLonD/beam/', 'BLonD/impedances/', ''],
      packages=find_packages(
          exclude=['__doc', '__BENCHMARKS', '__EXAMPLES', 'unittests']),
      include_package_data=True,
      setup_requires=['numpy',
                      'scipy',
                      'h5py',
                      'matplotlib'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent"]
      )
