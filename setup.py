"""
setup.py for test-project.

For reference see
https://packaging.python.org/guides/distributing-packages-using-setuptools/

"""
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info as _egg_info
import subprocess
import os
import sys

HERE = Path(__file__).parent.absolute()
with (HERE / 'README.md').open('rt', encoding='utf-8') as fh:
    LONG_DESCRIPTION = fh.read().strip()


REQUIREMENTS = {
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
        'sphinx-rtd-theme',
        'sphinxcontrib-napoleon',
        'sphinx-autopackagesummary',
        'pyqt5',
    ],
}

class Compile_and_Egg_Info(_egg_info):
    
    description = _egg_info.description + '\nCompile the C++ sources before package installation.'

    def initialize_options(self):
        _egg_info.initialize_options(self)

    def finalize_options(self):
        """Post-process options."""
        _egg_info.finalize_options(self)

    def run(self):
        cmd = [sys.executable, 'blond/compile.py']
        try:
            subprocess.run(cmd, check=True, env=os.environ.copy())
        except Exception as e:
            print('Compilation failed with: ', e)
            print('Failing back to the python-only backend.')
        return _egg_info.run(self)


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
    packages=find_packages(
        exclude=['__doc', '__BENCHMARKS', '__EXAMPLES', 'unittests']),
    python_requires='>=3.6',
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
    },
)
