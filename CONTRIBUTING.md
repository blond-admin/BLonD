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

```
blond/
├── __doc/                    # Sphinx documentation
├── blond/                    # Core Python package
├──── experimental/           # Untested/unstable code
├──── legacy/                 # The recent version of BLonD 2
├── integrationtests/         # Long-running integration tests
├── legacy/                   # Legacy scripts from BLonD 2
├── unittests/                # Fast-running unit tests
├── .gitlab-ci.yml            # GitLab CI configuration
├── .pre-commit-config.yaml   # Pre-commit hook definitions
├── MANIFEST.in               # Package include/exclude rules
├── pyproject.toml            # Project configuration for pip
└── setup.py                  # Build script for installation
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

### 4. Compile Native Backends

> Automatically done in GitLab CI Pipeline before testing.

```bash
blond-compile-cpp     # Compile the C++ backend
```
```bash
blond-compile-cuda    # Compile the CUDA backend
```
```bash
blond-compile-fortran # Compile the Fortran backend
```

---


## Running Tests
> Automatically done in GitLab CI Pipeline
```bash
python3 -m pytest -v unittests/
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
python3 -m sphinx build -b html -W -D html_theme=sphinx_rtd_theme -D html_theme_options.navigation_depth=5 --keep-going __doc __doc/_build/html
```

Built files appear in `__doc/_build/html/`.

Then, [index.html](__doc/_build/html/index.html) can be opened with a web browser

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
> NOTE: As long as BLonD 3 is not the main BLonD Version, it will not be available on PyPi and the docuemntation website.

> Automatically done in GitLab CI Pipeline

The [GitLab CI Pipeline](.gitlab-ci.yml) is configured for an automatic release process.
- Uploads **BLonD** from `master` to [PyPi](https://pypi.org/project/blond/)  if a new tag is created (see [BLonD Tags](https://gitlab.cern.ch/blond/BLonD/-/tags))
- Build/updates the **documentation** hosted at [BLonD Documentation Website](https://blond-code.docs.cern.ch/)
  - The linking between the GitLab project and the website can be adjusted in the [GitLab project settings](https://gitlab.cern.ch/blond/BLonD/pages#domains-settings)
