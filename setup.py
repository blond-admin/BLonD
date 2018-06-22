import sys
import os
import subprocess
from blond._version import __version__
import distutils
from shutil import rmtree
# from distutils.command.install import install as _install
from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info as _egg_info
from distutils.command.clean import clean as _clean

class Compile(distutils.cmd.Command):
    """Compile all C/C++ source files."""

    description = 'Compile the shared libraries'
    user_options = [
        ('openmp', 'o', 'Enable Multi-threaded code'),
        ('compiler=', 'c', 'Specify the compiler type'),
        ('boost=', None, 'Compile with the Boost Library')]


    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        self.openmp = None
        self.boost = None
        self.compiler = None

    def finalize_options(self):
        """Post-process options."""
        # print('Parallel: ', self.openmp)
        # print('Boost: ', self.boost)
        # print('Compiler: ', self.compiler)
        pass

    def run(self):
        """Run command."""
        cmd = ['python', 'blond/compile.py']
        if self.openmp:
            cmd.append('-p')
        if self.boost:
            cmd += ['-b', self.boost]
        if self.compiler:
            cmd += ['-c', self.compiler]

        subprocess.call(cmd)




class Compile_and_Egg_Info(_egg_info):
    description = 'Compile the C++ sources before installing'
    user_options = _egg_info.user_options
    user_options += [
        ('parallel', 'p', 'Enable Multi-threaded code'),
        ('compiler=', 'c', 'Specify the compiler type'),
        ('boost=', None, 'Compile with the Boost Library')]

    def initialize_options(self):
        _egg_info.initialize_options(self)
        self.parallel = None
        self.boost = None
        self.compiler = None


    def finalize_options(self):
        """Post-process options."""
        _egg_info.finalize_options(self)

    def run(self):
        cmd = ['python', 'blond/compile.py']
        if self.parallel:
            cmd.append('-p')
        if self.boost:
            cmd += ['-b', self.boost]
        if self.compiler:
            cmd += ['-c', self.compiler]

        subprocess.call(cmd)
        # self.run_command('compile')
        return _egg_info.run(self)


class CleanAll(_clean):

    def run(self):
        rmtree('build', ignore_errors=True)
        rmtree('dist', ignore_errors=True)
        rmtree('blond.egg-info', ignore_errors=True)
        return _clean.run(self)




# class Compile_and_Install(_install):
#     description = 'Compile the C++ sources before installing'
#     user_options = _install.user_options
#     # user_options += [
#     #     ('openmp', 'o', 'Enable Multi-threaded code'),
#     #     ('compiler=', 'c', 'Specify the compiler type'),
#     #     ('boost=', None, 'Compile with the Boost Library')]

#     # def initialize_options(self):
#     #     _install.initialize_options(self)
#     #     self.openmp = None
#     #     self.boost = None
#     #     pass

#     # def finalize_options(self):
#     #     """Post-process options."""
#     #     _install.finalize_options(self)
#     #     pass

#     def run(self):
#         # cmd = ['python', 'blond/compile.py']
#         # if self.openmp:
#         #     cmd.append('-p')
#         # if self.boost:
#         #     cmd += ['-b', self.boost]
#         # if self.compiler:
#         #     cmd += ['-c', self.compiler]

#         # subprocess.call(cmd)
#         self.run_command('compile')
#         return _install.run(self)


class Test(distutils.cmd.Command):
    """Run the unittests."""

    description = 'Run the unittests'
    user_options = [
        # The format is (long option, short option, description).
        ('testdir=', 't', 'Directory to collect the unittests'),
    ]

    def initialize_options(self):
        """Set default values for options."""
        self.testdir = 'unittests'
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        subprocess.call(['pytest', '-v', self.testdir])


class PEP8(distutils.cmd.Command):
    """Make the code to PEP8 compliant."""

    description = 'Make the code PEP8 compliant'
    user_options = [
        # The format is (long option, short option, description).
        ('git', 'g', 'Check only the commited files'),
    ]

    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        self.git = False
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        cmd = ['python', 'blond/sanity_check.py', '--pep8']
        if self.git is not False:
            cmd.append('git')
        subprocess.call(cmd)


class Docs(distutils.cmd.Command):
    """Compile the Docs to html"""

    description = 'Compile the Docs to html'
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        subprocess.call(['python', 'blond/sanity_check.py', '--docs'])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        """ Means no arguments were passed """
        sys.argv.append('compile')
    # if ('install' in sys.argv) and ('compile' not in sys.argv):
    #     """ To install, we need first to compile """
    #     sys.argv = [sys.argv[0]] + ['compile'] + sys.argv[1:]


setup(name='blond',
      version=__version__,
      description='CERN code for simulating longitudinal beam dynamics in synchrotrons.',
      keywords='Beam Longitudinal Dynamics Synchrotrons CERN',
      author='Helga Timko et al.',
      author_email='helga.timko@cern.ch',
      maintainer='Konstantinos Iliakis',
      maintainer_email='konstantinos.iliakis@cern.ch',
      long_description=open('README.rst').read(),
      # long_description_content_type='text/markdown',
      url='https://github.com/blond-admin/BLonD',
      download_url='https://github.com/blond-admin/BLonD/archive/v'+__version__+'.tar.gz',
      cmdclass={
          # 'build_py': Compile_and_Build,
          # 'install': Compile_and_Install,
          'egg_info': Compile_and_Egg_Info,
          'compile': Compile,
          'clean': CleanAll,
          'test': Test,
          'pep8': PEP8,
          'docs': Docs},
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
      # test_suite="unittests"
      )
