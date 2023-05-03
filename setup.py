"""
setup.py for test-project.

For reference see
https://packaging.python.org/guides/distributing-packages-using-setuptools/

"""
from pathlib import Path
from setuptools import setup, find_packages
# from blond._version import __version__
from setuptools.command.egg_info import egg_info as _egg_info
import distutils
import subprocess
import os

HERE = Path(__file__).parent.absolute()
with (HERE / 'README.md').open('rt', encoding='utf-8') as fh:
    LONG_DESCRIPTION = fh.read().strip()


REQUIREMENTS: dict = {
    'core': ['numpy',
             'scipy',
             'h5py',
             'matplotlib',
             'mpmath'
             ],
    'test': [
        'pytest',
    ],
    'dev': [
        # 'requirement-for-development-purposes-only',
    ],
    'doc': [
        'sphinx',
        'acc-py-sphinx',
    ],
}


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

        subprocess.run(cmd, check=True, env=os.environ.copy())
                # self.run_command('compile')
        return _egg_info.run(self)

class CompileLibrary(distutils.cmd.Command):
    """Compile all C/C++ source files."""

    description = 'Compile the shared libraries. Use blond/compile.py for advanced options.'
    user_options = [
        ('parallel', 'p', 'Enable Multi-threaded code'),
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
        subprocess.run(cmd, check=True, env=os.environ.copy())


setup(
    name='blond',
    # version=__version__,
    description='CERN code for simulating longitudinal beam dynamics in synchrotrons.',
    author='Helga Timko et al.',
    author_email='helga.timko@cern.ch',
    maintainer='Konstantinos Iliakis',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://gitlab.cern.ch/blond/BLonD',
    # download_url='https://github.com/blond-admin/BLonD/archive/v'+__version__+'.tar.gz',
    packages=find_packages(
        exclude=['__doc', '__BENCHMARKS', '__EXAMPLES', 'unittests']),
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    install_requires=REQUIREMENTS['core'],
    extras_require={
        **REQUIREMENTS,
        # The 'dev' extra is the union of 'test' and 'doc', with an option
        # to have explicit development dependencies listed.
        'dev': [req
                for extra in ['dev', 'test', 'doc']
                for req in REQUIREMENTS.get(extra, [])],
        # The 'all' extra is the union of all requirements.
        'all': [req for reqs in REQUIREMENTS.values() for req in reqs],
    },
    zip_safe=False,
    include_package_data=True,
    cmdclass={
        'egg_info': Compile_and_Egg_Info,
        'compile': CompileLibrary,
    },
)
