<div align="center">
<img src="BLonD2_centered.png" alt="drawing" width="300"/>
</div>

[![Pipeline Status](https://gitlab.cern.ch/blond/BLonD/badges/blonder/pipeline.svg)](https://gitlab.cern.ch/blond/BLonD/-/commits/blonder) [![Coverage Report](https://gitlab.cern.ch/blond/BLonD/badges/blonder/coverage.svg)](https://gitlab.cern.ch/blond/BLonD/-/commits/blonder) [![Latest Release](https://gitlab.cern.ch/blond/BLonD/-/badges/release.svg)](https://gitlab.cern.ch/blond/BLonD/-/releases) [![PyPi](https://img.shields.io/pypi/v/blond.svg)](https://pypi.org/project/blond/) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org) [![Documentation Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://blond-code.docs.cern.ch/)


# Developer Guide for BLonD
> A guide on how to maintain and extend BLonD 3

To ensure consistent code quality and releases,
a full installation and test, with optional deployment, is done using the [GitLab Continuous Integration (CI) Pipeline](.gitlab-ci.yml).
All relevant commands can be found there.

Code that is not mature enough to be inside the standard codebase should be developed in the folder [blond/experimental](blond/experimental), this folder is excluded from test coverage and pre-commit hooks.

## Project Structure

<!-- Automatically created using `dev_tools/create_tables.py` -->
```
blond/                        BLonD beam dynamics software.
├── acc_math/                 Analytical equations.
├──── analytic/               Analytical equations for theoretic descriptions.
├──── empiric/                Analytical equations for empirical observations.
├── beam_preparation/         Classes to setup the beam coordinates according to a :class:`~blond.core.simulation.simulation.Simulation`.
├── cycles/                   Module to manage and describe the ramp of the magnets and other cycles.
├──── noise_generators/       Module for noise generators.
├── examples/                 Overview of BLonD input files as a starting point for new simulations..
├── experimental/             Untested/unstable code that might be changed in the future.
├──── acc_math/               Helpers for math to deal with the output of simulations.
├────── empiric/              Helpers for math to deal with the output of simulations.
├──── beam_preparation/       Classes to setup the beam coordinates according to a :class:`~blond.core.simulation.simulation.Simulation`.
├──── physics/                Implementations to handle different beam physics processes,
├────── feedbacks/            Module to manage and describe the longitudinal feedbacks.
├──────── accelerators/       Feedback implementations for specific accelerators.
├────────── lhc/              Functions to define the CERN Large Hadron Collider feedback systems.
├────────── sps/              Utility functions to define feedbacks for the CERN synchrotrons.
├────────── psb/              Functions to define the CERN Proton Synchrotron Booster feedback systems.
├──── cycles/                 Module to manage and describe the ramp of the magnets and other cycles.
├────── noise_generators/     Collection of functions to generate noise.
├── handle_results/           Helper functions and detailed implementations to define :class:`blond.handle_results.observables.Observables`.
├── legacy/                   Access point for the legacy blond version, use ``from blond.legacy import blond2``.
├── physics/                  Implementations to handle different beam physics processes, like RF-Stations.
├──── impedances/             Module to handle the interaction of impedance sources with the beam.
├──── feedbacks/              Module to manage and describe the longitudinal feedbacks.
├── testing/                  Utilities for testing of BLonD.
├── specifics/                Utility functions for specific accelerators.
├──── cern/                   Utility functions for CERN synchrotrons.
├────── lhc/                  Utility functions for the CERN Large Hadron Collider.
├────── ps/                   Utility functions for the CERN Proton Synchrotron.
├────── psb/                  Utility functions for the CERN Proton Synchrotron Booster.
├────── sps/                  Utility functions for the CERN Super Proton Synchrotron.
├──── muon_collider/          Helper scripts for the muon collider.
├── interfaces/               Managing access to other (optional) beam physics software, like XSuite.
├──── xsuite/                 Glue code for XSuite.
├────── beam_preparation/     Classes to setup the beam coordinates according to a :class:`~blond.core.simulation.simulation.Simulation`.
├────── physics/              Beam physics classes for interfacing XSuite.
├── performance_blond3/       Testing the performance of BLonD.
├──── backends/               Testing the performance of the BLonD backends.
├── generals/                 Function definitions that are useful outside the beam physics context.
├──── cupy/                   Scripts that are useful to work with Cupy.
├──── distributed/            Helper module to work with CPU/GPU arrays distributed via MPI.
├── core/                     Core functionalities that define BLonD and its runtime.
├──── backends/               All helper functions and implementations for the numeric backends of BLonD.
├────── cpp/                  Holds `CppSpecials` and helper functions.
├────── cuda/                 Holds `CduaSpecials` and helper functions.
├────── fortran/              Holds `FortranSpecials` and helper functions.
├────── numba/                Holds `NumbaSpecials` and helper functions.
├────── python/               Holds `PythonSpecials` and helper functions.
├────── mpi_distributed/      Functions to interface with MPI distributed arrays.
├──── beam/                   Core classes and routines related to the Beam objects.
├──── ring/                   Methods related to the `Ring` class.
├──── simulation/             Definitions related to assembling a `Simulation`.
├────── execution_models/     Different implementations of main-loops, for example for counter-rotation.
├──── reference_clock/        Helper class that holds the reference to the beam coordinate system.
```


---

## Dependencies

Ensure the following tools are installed:

* [Python 3.10+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/)
* [Pre-Commit](https://pre-commit.com/)

**Optional (for C++ extensions / GPU support):**

* **Linux:**

  * [GCC (recommended)](https://gcc.gnu.org/install/)
  * `icc` or `clang` as alternatives
* **Windows:**

  * [mingw-w64](https://winlibs.com/#download-release)
* **GPU Support:**

  * [CUDA Compiler Driver (NVCC)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)

---

## Getting Started
> Automatically done in GitLab CI Pipeline

### 1. Clone the Repository

```bash
git clone https://gitlab.cern.ch/blond/BLonD/
cd blond
git checkout blonder  # Current development branch for BLonD3
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Development Dependencies

For CPU-only development:

```bash
pip install --editable .[dev]
```

For GPU-enabled development:

```bash
pip install --editable .[dev, gpu]
```

### 4. Set Up Pre-Commit Hooks

```bash
pre-commit install
```

### 4. Compiling Native Backends (Optional)

> **Note:** These steps are automatically executed in the GitLab CI pipeline prior to running tests.
> You only need to perform them manually if you are developing or testing locally.

To compile the available native backends, use the following commands:

```bash
blond-compile-cpp --parallel   # Compile the C++ backend
```
```bash
blond-compile-cuda             # Compile the CUDA backend
```
```bash
blond-compile-fortran          # Compile the Fortran backend
```

Once compiled, the corresponding backends will be available for use within your simulation environment.

To activate a specific backend (for example, the C++ backend), you can use the following Python code:

```python
from blond import backend

backend.set_specials(mode="cpp")  # Activate the C++ backend
```


---


## Running Tests
> Automatically done in GitLab CI Pipeline
```bash
python3 -m pytest -v unittests/
```

BLonD provides for marked tests with [PyTest](https://docs.pytest.org/en/stable/how-to/mark.html) via `@pytest.mark.xxx`.
Following markers are used

- 'backend_mutation'
- 'cupy'

Those tests can be excluded for running the tests with the `pytest -m` flag.
```bash
export BLOND_BACKEND_MODE=cuda
export BLOND_BACKEND_BITS=32
python3 -m pytest -m "not backend_mutation"  -v unittests/
```

The tests with distributed computing (MPI) can be executed via
```bash
export MPLBACKEND=Agg  # Prevent matplotlib deadlock
mpirun -n 2 python3 -m pytest -v unittests/ -m "mpi"
```

---

## Linting & Code Formatting

All code linting and formatting is managed via [pre-commit hooks](.pre-commit-config.yaml).

To run hooks on staged files:

```bash
pre-commit run
```

To run hooks on **all files**:

```bash
pre-commit run --all-files
```
An optional check of the code can be done using the command
```bash
ruff check
```

---


## Documentation
> Automatically done in GitLab CI Pipeline

To build the documentation locally:

```bash
python -m pip install .[doc]
python3 -m sphinx build -b html -W -D html_theme=sphinx_rtd_theme -D html_theme_options.navigation_depth=5 --keep-going docs docs/_build/html
```

Built files appear in `docs/_build/html/`.

Then, [index.html](docs/_build/html/index.html) can be opened with a web browser

## Contributing

1. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement your feature **along with unit tests**.

   * Follow the same folder structure in `/unittests` as in `blond/`.

3. Run tests to ensure nothing is broken.

4. Push your changes:
   * [GitLab CI Pipeline](.gitlab-ci.yml) will automatically run all tests online.

5. Create a Merge Request (MR):

   * Clearly explain your changes.
   * MR view shows:

     * Pipeline status (pass/fail).
     * Untested lines (highlighted in red).

       * Avoid committing untested code unless necessary.
       * For experimental/unverified code, use [`blond/experimental`](blond/experimental/), which is excluded from coverage reports.

---

## Release Process
> [!WARNING]
> As long as BLonD 3 is not the main BLonD Version, it will not be available on PyPi and the docuemntation website.

> Automatically done in GitLab CI Pipeline

The [GitLab CI Pipeline](.gitlab-ci.yml) is configured for an automatic release process.
- Uploads **BLonD** from `master` to [PyPi](https://pypi.org/project/blond/)  if a new tag is created (see [BLonD Tags](https://gitlab.cern.ch/blond/BLonD/-/tags))
- Build/updates the **documentation** hosted at [BLonD Documentation Website](https://blond-code.docs.cern.ch/)
  - The linking between the GitLab project and the website can be adjusted in the [GitLab project settings](https://gitlab.cern.ch/blond/BLonD/pages#domains-settings)



---

## CI/CD

The project uses GitLab CI/CD to automate testing, building, and deployment.

* The full pipeline configuration is defined in the root-level **[`.gitlab-ci.yml`](.gitlab-ci.yml)** file.
  Please review it before modifying or extending any part of the CI workflow.

* The Docker images used by the various pipeline stages are maintained in the
  **[GitLab CI Docker project](https://gitlab.cern.ch/blond/developer-tools/gitlab-ci-docker)**.
  If your contribution requires changes to these images, please open a merge request in that repository as well.


---
