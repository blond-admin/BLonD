# Code Name

BLonD (Beam Longitudinal Dynamics)

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

Repository: <https://github.com/blond-admin/BLonD>

Documentation: <http://blond-admin.github.io/BLonD/>

Project website: <http://blond.web.cern.ch>

# Install

## Requirements

1.  A gcc compiler with C++11 support (version greater than 4.8.4).
2.  An Anaconda distribution (Python 3 recommended).
3.  That's all!

## Windows GCC Installation Instructions

1.  Download the latest mingw-w64 using this link:
    <https://winlibs.com/#download-release>
2.  Extract the downloaded zip/7-zip under e.g. `C:\`. You should now
    see a directory `C:\mingw64`.
3.  Add `C:\mingw64\bin` in the PATH Environment Variable. [Here you can
    see how to do this in Windows
    XP/Vista/7/8/10/11](https://www.computerhope.com/issues/ch000549.htm).
4.  To validate the correct setup of gcc, open a command prompt and
    type: `gcc --version`. The first output line should contain the gcc
    version you just installed.

## Install Steps

### Installing BLonD as a python package.

-   Using the pip package manager:

    ``` bash
    $ pip install blond
    ```

-   If this fails try to:

    1.  Clone the repository from gitlab or download and extract the zip
        from <https://gitlab.cern.ch/blond/BLonD/-/archive/master/BLonD-master.zip>
    2.  Go to the downloaded BLonD directory and run:

        ``` bash
        $ python setup.py install
        ```

-   If it still fails, go to the BLonD directory and run:

    1.  
        ``` bash
        $ python setup.py compile
        ```

    2.  Then you have to set the PYTHONPATH variable to point to the
        BLonD installation path.

-   In the occasion that it continues to fail, please open a new gitlab issue.

### For advanced users or developers.

1.  You are advised to install git in your system.
2.  Clone the repository or download and extract it.
3.  From within the BLonD directory run:
    ``` bash
    $ python blond/compile.py
    ```
4.  Adjust the PYTHONPATH to contain the path to the cloned repository.

## Confirm proper installation

-   Run the unittests with pytest (may need to be installed first with
    pip install pytest):

    ``` bash
    $ pytest -v unittests
    ```

-   Try to run some of the main files found in the examples:

    ``` bash
    $ python __EXAMPLES/main_files/EX_01_Acceleration.py
    $ python __EXAMPLES/main_files/EX_02_Main_long_ps_booster.py
    $ etc..
    ```

## Performace Optimizations

There are some easy ways to reduce the execution time of your
simulation:

1.  Use the multi-threaded C library. To use it you have to add the -p
    flag when compiling the C library:
    ``` bash
    $ python blond/compile.py --parallel
    ```

2.  Enable processor specific compiler optimizations:
    ``` bash
    $ python blond/compile.py --flags='-march=native'
    ```

3.  If you are test-case is calling the synchrotron radiation tracking method, you can accelerate it by using the Boost library. To do so you have to:  
    1.  Download Boost: <https://www.boost.org/>. Let's say the version
        you downloaded is boost_1\_70.
    2.  Extract it, let's say in `/user/path/to/boost_1_70`.
    3.  Pass the boost installation path when compiling BLonD:
        ``` bash
        $ python blond/compile.py --boost=/user/path/to/boost_1_7_70
        ```

4.  Check the following section about the FFTW3 library.

5.  *All the above can be combined.*

## Changing the floating point precision (32 bit floats or 64 bit floats)

-   By default BLonD uses double precision calculations (float64). To
    change to single precision, for faster calculations, in the
    beginning of your mainfile you will have to add the code lines:
    ``` python
    from blond.utils import bmath as bm
    bm.use_precision('single') 
    ```

-   No other modifications are needed.

## Use the FFTW3 library for the FFTs

So far only the `rfft()`, `irfft()` and `fftfreq()` routines are
supported. `fft_convolve()` to be added soon.

-   Windows:

    1.  Download and unzip the pre-compiled FFTW3 library. Link:
        <ftp://ftp.fftw.org/pub/fftw/fftw-3.3.5-dll64.zip>
    2.  Copy the `libfftw3-3.dll` under your python's distribution
        directory.
    3.  Run the `blond/compile.py` with the flag `--with-fftw`.
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
    2.  Run the `blond/compile.py` with the flag: `--with-fftw`.
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
    python blond/compile.py -p --with-fftw --with-fftw-threads --with-fftw-lib=$INSTALL_DIR/lib --with-fftw-header=$INSTALL_DIR/include
    ```

-   adjust your main file as needed (described bellow).

-   example scripts to setup and run a parameter scan in the HPC Slurm
    cluster: <https://cernbox.cern.ch/index.php/s/shqtotwyn4rm8ws>

## Changes required in the main file for MPI

1.  This statements in the beginning of the script:

    ``` python
    from blond.utils import bmath as bm
    from blond.utils.mpi_config import worker, mpiprint
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
    if worker.isMaster:
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
    worker.finalize()
    ```

6.  To run your script, you need to pass it to **mpirun** or
    **mpiexec**. To spawn P MPI processes run:

    ``` bash
    $ mpirun -n P python main_file.py
    ```

7.  For more examples have a look at the \_\_EXAMPLES/mpi_main_files/
    directory.

# Using the GPU Implementation

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

    To compile the Cuda files execute blond/compile.py and add the flag
    --gpu. The Compute Capability of your GPU will be automatically
    detected:

``` bash
$ python blond/compile.py --gpu 
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
mainfile.

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

# Developers

-   Simon Albright (simon.albright (at) cern.ch)
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
