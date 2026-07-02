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
├────── synchrotron_radiation/A collection of analytic equations required for synchrotron radiation.
├──── empiric/                Analytical equations for empirical observations.
├── beam_preparation/         Classes to setup the beam coordinates according to a :class:`~blond.core.simulation.simulation.Simulation`.
├── convenience/              Convenience functions to interact with BLonD.
├── core/                     Core functionalities that define BLonD and its runtime.
├──── backends/               All helper functions and implementations for the numeric backends of BLonD.
├────── cpp/                  Holds `CppSpecials` and helper functions.
├────── cuda/                 Holds `CudaSpecials` and helper functions.
├────── mpi_distributed/      Functions to interface with MPI distributed arrays.
├────── numba/                Holds `NumbaSpecials` and helper functions.
├────── python/               Holds `PythonSpecials` and helper functions.
├──── beam/                   Core classes and routines related to the Beam objects.
├──── reference_clock/        Helper class that holds the reference to the beam coordinate system.
├──── ring/                   Methods related to the `Ring` class.
├──── simulation/             Definitions related to assembling a `Simulation`.
├────── execution_models/     Different implementations of main-loops, for example for counter-rotation.
├── cycles/                   Module to manage and describe the ramp of the magnets and other cycles.
├──── noise_generators/       Module for noise generators.
├── examples/                 Overview of BLonD input files as a starting point for new simulations..
├──── notebooks/              Helpful jupyter notebooks that are presented on the website.
├──── scripts/                Overview of BLonD input files as a starting point for new simulations..
├── experimental/             Untested/unstable code that might be changed in the future.
├──── acc_math/               Helpers for math to deal with the output of simulations.
├────── empiric/              Helpers for math to deal with the output of simulations.
├──── beam_preparation/       Classes to setup the beam coordinates according to a :class:`~blond.core.simulation.simulation.Simulation`.
├────── semi_empiric_matcher_extensions/
├──────── line_density/       Helper implementations to fit a line density with `SemiEmpiricMatcher`.
├──── cycles/                 Module to manage and describe the ramp of the magnets and other cycles.
├────── noise_generators/     Collection of functions to generate noise.
├──── physics/                Implementations to handle different beam physics processes,
├────── feedbacks/            Module to manage and describe the longitudinal feedbacks.
├──────── accelerators/       Feedback implementations for specific accelerators.
├────────── lhc/              Functions to define the CERN Large Hadron Collider feedback systems.
├────────── psb/              Functions to define the CERN Proton Synchrotron Booster feedback systems.
├────────── sps/              Utility functions to define feedbacks for the CERN synchrotrons.
├── generals/                 Function definitions that are useful outside the beam physics context.
├──── cupy/                   Scripts that are useful to work with Cupy.
├──── distributed/            Helper module to work with CPU/GPU arrays distributed via MPI.
├── handle_results/           Helper functions and detailed implementations to define :class:`blond.handle_results.observables.ObservablesBaseClass`.
├── interfaces/               Managing access to other (optional) beam physics software, like XSuite.
├──── xsuite/                 Glue code for XSuite.
├────── beam_preparation/     Classes to setup the beam coordinates according to a :class:`~blond.core.simulation.simulation.Simulation`.
├────── physics/              Beam physics classes for interfacing XSuite.
├── legacy/                   Access point for the legacy blond version, use ``from blond.legacy import blond2``.
├── physics/                  Implementations to handle different beam physics processes, like RF-Stations.
├──── feedbacks/              Module to manage and describe the longitudinal feedbacks.
├──── impedances/             Module to handle the interaction of impedance sources with the beam.
├──── synchrotron_radiation/  Implementations to simulate the effect of synchrotron radiation.
├── specifics/                Utility functions for specific accelerators.
├──── cern/                   Utility functions for CERN synchrotrons.
├────── lhc/                  Utility functions for the CERN Large Hadron Collider.
├────── ps/                   Utility functions for the CERN Proton Synchrotron.
├────── psb/                  Utility functions for the CERN Proton Synchrotron Booster.
├────── sps/                  Utility functions for the CERN Super Proton Synchrotron.
├──── fccee/                  Accelerator specifics for the future circular collider.
├──── muon_collider/          Helper scripts for the muon collider.
├── testing/                  Utilities for testing of BLonD.
├── utilities/                Module contains various utilities used throughout the library.
├──── separatrix/             Package which contains utilities for working with separatrix.
```

**Where to start reading:**

* [`blond/examples/scripts/minimum_working_example.py`](blond/examples/scripts/minimum_working_example.py) — smallest end-to-end simulation.
* [`blond/core/simulation/simulation.py`](blond/core/simulation/simulation.py) — assembles `Ring`, `MagneticCycle`, beams, and observables; drives the main loop.
* [`blond/core/ring/ring.py`](blond/core/ring/ring.py) and [`blond/physics/`](blond/physics/) — physics elements added to a ring.
* [`blond/examples/scripts/`](blond/examples/scripts/) — `EX_01_*` … `EX_13_*` cover progressively richer setups.

---

## Dependencies

Ensure the following tools are installed:

* [Python 3.10+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/)
* [Pre-Commit](https://pre-commit.com/)

**Optional (for C++ extensions / GPU / MPI support):**

* **Linux:**

  * [GCC (recommended)](https://gcc.gnu.org/install/)
  * `icc` or `clang` as alternatives
* **Windows:**

  * [mingw-w64](https://winlibs.com/#download-release)
* **GPU Support:**

  * [CUDA Compiler Driver (NVCC)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
* **MPI Support** (required to build `mpi4py`):

  * Linux: `libopenmpi-dev` / `openmpi` (or your distribution's equivalent)
  * macOS: `brew install open-mpi`
  * Windows: [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)

---

## Getting Started
> Automatically done in GitLab CI Pipeline

### 1. Clone the Repository

```bash
git clone https://gitlab.cern.ch/blond/BLonD/
cd BLonD
git checkout blonder  # Current development branch for BLonD3
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate           # Linux / macOS
# .venv\Scripts\activate            # Windows (PowerShell or cmd)
```

### 3. Install Development Dependencies

For CPU-only development:

```bash
pip install --editable ".[dev]"
```

For GPU-enabled development both CUDA12 and CUDA 13 are available:

For CUDA12:
```bash
pip install --editable ".[dev, gpu_cuda12]"
```

For CUDA13:
```bash
pip install --editable ".[dev, gpu_cuda13]"
```

The convenience extras `all_no_cuda`, `all_cuda12`, and `all_cuda13` bundle
`dev`, `doc`, `xsuite`, and `mpi` (and the matching GPU package), e.g.:

```bash
pip install --editable ".[all_cuda12]"
```

For XSuite interop only (e.g., RF-bucket matching via XSuite), add the `xsuite` extra:

```bash
pip install --editable ".[dev, xsuite]"
```

After installation, verify your setup by running a minimal slice of the test suite:

```bash
python3 -m pytest -v tests/unittests/core/ring/
```

### 4. Set Up Pre-Commit Hooks

```bash
pre-commit install
```

### 5. Compiling Native Backends (Optional)

> **Note:** These steps are automatically executed in the GitLab CI pipeline prior to running tests.
> You only need to perform them manually if you are developing or testing locally.

To compile the available native backends, use the following commands:

```bash
blond-compile-cpp --parallel   # Compile the C++ backend
```
```bash
blond-compile-cuda             # Compile the CUDA backend
```

Once compiled, the corresponding backends will be available for use within your simulation environment.

To activate a specific backend (for example, the C++ backend), you can use the following Python code:

```python
from blond import setup_backend

setup_backend("cpp")  # Activate the C++ backend
```


---


## Running Tests
> Automatically done in GitLab CI Pipeline
```bash
python3 -m pytest -v tests/unittests/ --randomly-seed=$CI_PIPELINE_ID
```

The random seed is displayed online in the output terminal of the CI pipeline.
Replace '$CI_PIPELINE_ID' by the actual pipeline number when executing tests on a local machine.

BLonD provides for marked tests with [PyTest](https://docs.pytest.org/en/stable/how-to/mark.html) via `@pytest.mark.xxx`.
Apply a marker when your test depends on global/backend state or external runtimes:

- `backend_mutation` — test changes the active backend as a side effect (e.g.,
  switches numerical specials). Skip these when running against a fixed backend.
- `cupy` — test requires CuPy / a CUDA-capable GPU.
- `mpi` — test must be launched under `mpirun` (uses MPI communication).

Those tests can be excluded for running the tests with the `pytest -m` flag.
```bash
export BLOND_BACKEND_MODE=cuda
export BLOND_BACKEND_BITS=64
python3 -m pytest -m "not backend_mutation"  -v tests/unittests/
```

When modifying backend code, set `BLOND_FORCE_TEST_ALL_BACKENDS=True` to make
backend-aware tests fan out across every available backend instead of only the
one selected by `BLOND_BACKEND_MODE`:

```bash
export BLOND_FORCE_TEST_ALL_BACKENDS=True
python3 -m pytest -v tests/unittests/
```

The tests with distributed computing (MPI) can be executed via
```bash
export MPLBACKEND=Agg  # Prevent matplotlib deadlock
mpirun -n 2 python3 -m pytest -v tests/unittests/ -m "mpi"
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

**Docstring style.** Public functions, classes, and modules use the
[NumPy docstring convention](https://numpydoc.readthedocs.io/en/latest/format.html);
docstrings are validated by `numpydoc` (configured in `pyproject.toml`).

**Copyright header.** Every new file under `blond/` (excluding `blond/legacy/`)
must carry the copyright header from
[`dev_tools/copyright_notice.txt`](dev_tools/copyright_notice.txt) — pre-commit
will reject missing headers. To apply the header to all new files in bulk:

```bash
python3 dev_tools/copy_copyright_to_all_files.py
```

---


## Documentation
> Automatically done in GitLab CI Pipeline

To build the documentation locally:

```bash
python -m pip install .[doc]
cd docs && bash create_docs.sh
```

See [`docs/create_docs.sh`](docs/create_docs.sh) for the full build steps.
Built files appear in `docs/_build/html/`.

Then, [index.html](docs/_build/html/index.html) can be opened with a web browser

## Contributing

> **Reporting bugs or asking questions:** open an issue at
> [BLonD Issues](https://gitlab.cern.ch/blond/BLonD/-/issues). When an issue is
> linked from a branch, reference it in the branch name (e.g.
> `blonder_feature/249-...`).

1. Create a feature branch off `blonder`. Branch naming follows the pattern:

   ```text
   blonder_feature/<issue-or-topic>     # new functionality
   blonder_bugfix/<issue-or-topic>      # bug fixes
   ```

   ```bash
   git checkout -b blonder_feature/your-feature-name
   ```

2. Implement your feature **along with unit tests**.

   * Follow the same folder structure in `tests/unittests` as in `blond/`.

3. Run tests to ensure nothing is broken.

4. Push your changes:
   * [GitLab CI Pipeline](.gitlab-ci.yml) will automatically run all tests online.

5. Create a Merge Request (MR) targeting `blonder`:

   * Clearly explain your changes.
   * MR view shows:

     * Pipeline status (pass/fail).
     * Untested lines (highlighted in red).

       * Avoid committing untested code unless necessary.
       * For experimental/unverified code, use [`blond/experimental`](blond/experimental/), which is excluded from coverage reports.

---

## Guidelines for the use of AI

When contributing code with the assistance of coding agents, the following guidelines should be considered:
1. AI is a tool, not a collaborator, you are entirely responsible for what it does
2. Be sure that you understand the code enough to extend or modify it yourself
3. Know why the chosen solution is the right one, at least subjectively
4. Make sure the documentation is clear, concise, and cites relevant sources

---

## Release Process
> [!WARNING]
> As long as BLonD 3 is not the main BLonD Version, it will not be available on PyPi and the documentation website.

> Automatically done in GitLab CI Pipeline

The [GitLab CI Pipeline](.gitlab-ci.yml) is configured for an automatic release process.
- Uploads **BLonD** to [PyPi](https://pypi.org/project/blond/) whenever a new tag is pushed (see [BLonD Tags](https://gitlab.cern.ch/blond/BLonD/-/tags))
- Build/updates the **documentation** hosted at [BLonD Documentation Website](https://blond-code.docs.cern.ch/)
  - The linking between the GitLab project and the website can be adjusted in the [GitLab project settings](https://gitlab.cern.ch/blond/BLonD/pages#domains-settings)

**Cutting a release (manual steps).** Versioning is driven by Git tags via
[`setuptools_scm`](https://setuptools-scm.readthedocs.io/) — no version field
is edited by hand. To release:

1. Ensure `blonder` is green in CI and that all release-blocking issues are closed.
2. Tag the release commit following [PEP 440](https://peps.python.org/pep-0440/),
   e.g. `git tag v3.0.0` (use `vMAJOR.MINOR.PATCH`).
3. Push the tag: `git push origin v3.0.0`. The `release_sdist_*` jobs (triggered
   by `.on_tag` in `.gitlab-ci.yml`) then publish to PyPi.



---

## CI/CD

The project uses GitLab CI/CD to automate testing, building, and deployment.

* The full pipeline configuration is defined in the root-level **[`.gitlab-ci.yml`](.gitlab-ci.yml)** file.
  Please review it before modifying or extending any part of the CI workflow.

* The Docker images used by the various pipeline stages are maintained in the
  **[GitLab CI Docker project](https://gitlab.cern.ch/blond/developer-tools/gitlab-ci-docker)**.
  If your contribution requires changes to these images, please open a merge request in that repository as well.


---
