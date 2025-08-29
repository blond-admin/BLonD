<div align="center">
<img src="BLonD2_centered.png" alt="drawing" width="300"/>
</div>

[![Pipeline Status](https://gitlab.cern.ch/blond/BLonD/badges/master/pipeline.svg)](https://gitlab.cern.ch/blond/BLonD/-/commits/master) [![Coverage Report](https://gitlab.cern.ch/blond/BLonD/badges/master/coverage.svg)](https://gitlab.cern.ch/blond/BLonD/-/commits/master) [![Latest Release](https://gitlab.cern.ch/blond/BLonD/-/badges/release.svg)](https://gitlab.cern.ch/blond/BLonD/-/releases) [![PyPi](https://img.shields.io/pypi/v/blond.svg)](https://pypi.org/project/blond/) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org) [![Documentation Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://blond-code.docs.cern.ch/)


# Beam Longitudinal Dynamics Code (BLonD)

# Copyright Notice

*Copyright 2019 CERN. This software is distributed under the terms of the
GNU General Public Licence version 3 (GPL Version 3), copied verbatim in
the file LICENCE.txt. In applying this licence, CERN does not waive the
privileges and immunities granted to it by virtue of its status as an
Intergovernmental Organization or submit itself to any jurisdiction.*

# Description

CERN code for the simulation of longitudinal beam dynamics in
synchrotrons.

# Useful Links

Repository: <https://gitlab.cern.ch/blond/BLonD>

Documentation: <https://blond-code.docs.cern.ch/>

Project website: <http://blond.web.cern.ch>

BLonD example project: https://gitlab.cern.ch/blond/blond-simulation-template

# Installation

## Dependencies

1. Python 3.10 or above (python venv is recommended).  
2. (Optional) For better performance, a C++ (e.g. `gcc`, `icc`, `clang`, etc.) compiler with `C++11` support.

### (Optional) C++ compiler installation instructions

#### Windows

1.  Download the latest _mingw-w64_ using this link:
    <https://winlibs.com/#download-release>
    (if you download _mingw-w64_ from another source, it might cause problems due to an altered folder structure)
2.  Extract the downloaded _zip_/_7-zip_ under e.g. `C:\`. You should now
    see a directory `C:\mingw64`.
3.  Add `C:\mingw64\bin` in the PATH Environment Variable. [Here you can
    see how to do this in Windows
    XP/Vista/7/8/10/11](https://www.computerhope.com/issues/ch000549.htm).
4.  To validate the correct setup of gcc, open a command prompt and
    type: `gcc --version`. The first output line should contain the gcc
    version you just installed.

#### Linux
Use your distribution's package manager to install the compiler of your choice. BLonD has been tested with: `gcc` (recommended), `icc`, and `clang`.

## Installation Steps

### Installing BLonD from PyPI.

-   Use the `pip` package manager and simply run:
    ```bash
    pip install blond
    ```

### Installing BLonD manually (advanced users/ developers).

1.  Clone the repository (with `git`) or download and extract it.
2.  (Optional) From within the BLonD directory run:
    ```bash
    python blond/compile.py
    ```
    or from anywhere (after installing BLonD)
    
    ```bash
    blond-compile # executes /BLonD/blond/compile.py
    ```

    See the complete list of optional command line arguments with:
    ```bash
    blond-compile --help
    ```
3. Then install BLonD in edit mode with: 
    ```bash
    pip install -e .
    ```

## Confirm proper installation

-   A quick way to confirm the successful installation is to run:
    ``` bash
    python -c "from blond import test; test()"
    ```

-   If you installed BLonD manually, you can in addition run the unittests with `pytest`. The `pytest` package has to be installed first with `pip`. :
    ``` bash
    pip install pytest
    pytest -v unittests
    ```
    Note that running all the unit-tests might take more than 20 minutes, depending on your system.

-   You may also run some of the example main files found in the `__EXAMPLES` directory:
    ``` bash
    python __EXAMPLES/main_files/EX_01_Acceleration.py
    python __EXAMPLES/main_files/EX_02_Main_long_ps_booster.py
    etc..
    ```

# Performance Optimizations
BLonD contains three computational backends, sorted in order of better performance:
1. `C++` backend (Supports multi-threading and vectorization)
2. [`Numba` backend](https://numba.pydata.org) (Supports multi-threading and vectorization)
3. `Python`-only backend (No multi-threading or vectorization)

The performance order also defines the order in which the backends will be used. If the `C++` blond library has been compiled, then the `C++` backend will be used. Otherwise, if the `numba` package is installed, the numba backend will be used. Finally, if neither condition is met, the `python`-only backend will be used.

To use the `Numba` backend, you simply need to install the numba package with `pip`:
```bash
pip install numba
```

To use the `C++` backend, follow the instructions provided in the section *Installing BLonD manually*.

In addition, you may want to:
* Use the multithreaded blond `C++` backend:
    ``` bash
    blond-compile --parallel
    ```

* Enable processor specific compiler optimizations:
    ``` bash
    blond-compile --parallel --optimize
    ```

* If you are test-case is calling the synchrotron radiation tracking method, you can accelerate it by using the Boost library. To do so you have to:  
    1.  Download Boost: <https://www.boost.org/>. Let's say the version
        you downloaded is boost_1\_70.
    2.  Extract it, let's say in `/user/path/to/boost_1_70`.
    3.  Pass the boost installation path when compiling BLonD:
        ``` bash
        blond-compile --boost=/user/path/to/boost_1_7_70
        ```

* Check the following section about the FFTW3 library.

* All the above can be combined, i.e.:
    ```bash
    blond-compile --parallel --optimize --boost=...
    ```

## Changing the floating point number datatype

By default, BLonD uses double precision calculations (float64). To change to single precision for faster calculations, in the beginning of your main file you will have to add the following code lines:
```python
from blond.utils import bmath as bm
bm.use_precision('single') 
```


## Use the FFTW3 library for the FFTs

So far only the `rfft()`, `irfft()` and `fftfreq()` routines are
supported. `fft_convolve()` to be added soon.

-   Windows:

    1.  Download and unzip the pre-compiled FFTW3 library. Link:
        <ftp://ftp.fftw.org/pub/fftw/fftw-3.3.5-dll64.zip>
    2.  Copy the `libfftw3-3.dll` under your python's distribution
        directory.
    3.  Run the `blond-compile` with the flag `--with-fftw`.
    4.  If the FFTW3 library is not installed in one of the default
        directories, use the `--with-fftw-lib` and `--with-fftw-header`
        to point to the directories where the shared library and header
        files are stored.
    5.  To use the supported routines, you need to call the function
        `use_fftw()` from `bmath.py`:

        ``` python
        from blond.utils import bmath as bm
        bm.use_fftw()
        ...
        bm.rfft(...)
        bm.irfft(...)
        bm.rfftfreq(...)
        ```

-   Linux:

    1.  Download and compile the FFTW3 library. Link:
        <http://www.fftw.org/fftw-3.3.8.tar.gz>
    2.  Run the `blond-compile` with the flag: `--with-fftw`.
    3.  If the FFTW3 library is not installed in one of the default
        directories, use the `--with-fftw-lib` and `--with-fftw-header`
        to point to the directories where the shared library and header
        files are stored.
    4.  Optionally, you can enable one of the flags `--with-fftw-omp` or
        `--with-fftw-threads` to use the multithreaded FFTs.
    5.  To use the supported routines, you need to call the function
        `use_fftw()` from `bmath.py`:

        ``` python
        from blond.utils import bmath as bm
        bm.use_fftw()
        ...
        bm.rfft(...)
        bm.irfft(...)
        bm.rfftfreq(...)
        ```

# Using the multi-node (MPI) implementation

## Set-up Instructions

-   Add the following lines in your \~/.bashrc, then source it:

    ``` bash
    # Environment variables definitions
    export LD_LIBRARY_PATH="$HOME/install/lib"
    export INSTALL_DIR="$HOME/install"
    export BLONDHOME="$HOME/git/BLonD"

    # User aliases
    alias mysqueue="squeue -u $USER"
    alias myscancel="scancel -u $USER"
    alias mywatch="watch -n 30 'squeue -u $USER'"

    # Module loads
    module load compiler/gcc7
    module load mpi/mvapich2/2.3
    ```

-   Download and install anaconda3:

    ``` bash
    cd ~
    mkdir -p ~/downloads
    cd downloads
    wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
    bash Anaconda3-2018.12-Linux-x86_64.sh -b -p $HOME/install/anaconda3
    ```

-   Download and install fftw3 (with the appropriate flags):

    ``` bash
    cd ~
    mkdir -p ~/downloads
    cd downloads
    wget http://www.fftw.org/fftw-3.3.10.tar.gz
    tar -xzvf fftw-3.3.10.tar.gz
    cd fftw-3.3.10
    ./configure --prefix=$HOME/install/ --enable-openmp --enable-single --enable-avx --enable-avx2 --enable-fma --with-our-malloc --disable-fortran --enable-shared
    make -j4
    make install
    ./configure --prefix=$HOME/install/ --enable-openmp --enable-avx --enable-avx2 --enable-fma --with-our-malloc --disable-fortran --enable-shared
    make -j4
    make install
    ```

-   install mpi4py with pip:

    ``` bash
    pip install mpi4py
    ```

-   clone this repo, compile the library and link with fftw3_omp

    ``` bash
    cd ~
    mkdir -p git
    cd git
    git clone --branch=master https://github.com/blond-admin/BLonD.git
    cd BLonD
    blond-compile -p --with-fftw --with-fftw-threads --with-fftw-lib=$INSTALL_DIR/lib --with-fftw-header=$INSTALL_DIR/include
    ```

-   adjust your main file as needed (described bellow).

-   example scripts to set up and run a parameter scan in the HPC Slurm
    cluster: <https://cernbox.cern.ch/index.php/s/shqtotwyn4rm8ws>

## Changes required in the main file for MPI

1.  These statements in the beginning of the script:

    ``` python
    from blond.utils import bmath as bm
    from blond.utils.mpi_config import WORKER, mpiprint
    bm.use_mpi()  
    ```

2.  After having initialized the beam and preferably just before the
    start of the main loop:

    ``` python
    # This line splits the beam coordinates equally among the workers.
    beam.split()
    ```

3.  If there is code block that you want it to be executed by a single
    worker only, you need to surround it with this if condition:

    ``` python
    if WORKER.is_master:
        foo()
        ...
    ```

4.  If you need to re-assemble the whole beam back to the master worker
    you need to run:

    ``` python
    beam.gather()
    ```

5.  Finally, in the end of the simulation main loop, you can terminate
    all workers except from the master with:

    ``` python
    WORKER.finalize()
    ```

6.  To run your script, you need to pass it to **mpirun** or
    **mpiexec**. To spawn P MPI processes run:

    ``` bash
    mpirun -n P python main_file.py
    ```

7.  For more examples have a look at the \_\_EXAMPLES/mpi_main_files/
    directory.

# Using the GPU backend

## Setup Instructions

1.  Install **CUDA**: <https://developer.nvidia.com/cuda-downloads>
2.  Install the **CuPy** library:
    <https://docs.cupy.dev/en/stable/install.html>
3.  To verify your installation open a python terminal and execute the
    following script

    ``` python
    import cupy as cp 
    import numpy as np 
    a = cp.array(np.zeros(1000,np.float64)) 
    ```

    To compile the CUDA files execute blond/compile.py and add the flag
    --gpu. The Compute Capability of your GPU will be automatically
    detected:

    ``` bash
    blond-compiley --gpu 
    ```

## Changes required in the main file for GPU

1.  Right before your main loop you need to add:

    ``` python
    from blond.utils import bmath as bm
    # change some of the basic functions to their GPU equivalent
    bm.use_gpu()
    ```

2.  Also for every object you are using in your main loop that is in the
    following list:

    | GPU objects             |
    |-------------------------|
    | Beam                    |
    | Profile                 |
    | RingAndRFTracker        |
    | TotalInducedVoltage     |
    | InducedVoltageTime      |
    | InducedVoltageFreq      |
    | InductiveImpedance      |
    | InducedVoltageResonator |
    | RFStation               |
    | BeamFeedback            |

    you need to call their `to_gpu()` method. The following is a typical
    example from the \_\_EXAMPLES/gpu_main_files/EX_01_Acceleration.py
    main file.

    ``` python
    # Define Objects
    beam = Beam(ring, N_p, N_b)
    profile = Profile(beam, CutOptions(n_slices=100), 
                FitOptions(fit_option='gaussian'))
    # Initialize gpu
    beam.to_gpu()
    profile.to_gpu()
    ```

    If an object of this list has a reference inside a different one you
    don't need to use the `to_gpu()` for the referenced object. In the
    previous example, we don't need to call `beam.to_gpu()` since `beam` is
    referenced inside the `profile`. The same would apply in a
    `TotalInducedVoltage` object and the objects in its
    `induced_voltage_list`.

# Contributing to BLonD
We welcome contributions from the beam physics community to enhance the capabilities and features of BLonD.

For contribution as developer:
1. Create an [GitLab issue](https://gitlab.cern.ch/blond/BLonD/-/issues) and describe what you want to improve, fix, adapt, etc.
2. Create a branch from your issue by clicking the upper right blue button on your issue page.
3. Checkout your branch with your programming suite (or with the terminal: `git checkout https://gitlab.cern.ch/blond/BLonD/-/issues/YOUR-BRANCH`).
4. Commit and push your changes to your branch until satisfaction.
5. Create a merge request in GitLab from `YOUR-BRANCH` to `develop`.
6. Your code will be reviewed and finally be merged.

# Developers

-   Simon Albright (simon.albright (at) cern.ch)
-   Simon Lauber (simon.fabian.lauber (at) cern.ch)
-   Theodoros Argyropoulos (theodoros.argyropoulos (at) cern.ch)
-   Konstantinos Iliakis (konstantinos.iliakis (at) cern.ch)
-   Ivan Karpov (ivan.karpov (at) cern.ch)
-   Alexandre Lasheen (alexandre.lasheen (at) cern.ch)
-   Juan Esteban Muller (JuanF.EstebanMuller (at) ess.eu)
-   Danilo Quartullo (danilo.quartullo (at) cern.ch)
-   Joel Repond (joel (at) repond.ch)
-   Markus Schwarz (markus.schwarz (at) kit.edu)
-   Helga Timko (Helga.Timko (at) cern.ch)

# Structure

-   the folder \_\_EXAMPLES contains several main files which show how
    to use the principal features of the code;
-   the \_\_doc folder contains the source files for the documentation
    on-line;
-   the various packages which constitute the code are under the blond
    directory;
-   blond/compile.py is needed to compile all the C++ files present in
    the project; this file should be run once before launching any
    simulation. The compiler C++ GCC (at least version 4.8) is
    necessary.
-   WARNINGS.txt contains useful information related to code usage.
