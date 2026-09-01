---
name: blond-dev
description: Use when developing in the BLonD3 / BLonD codebase (gitlab.cern.ch/blond/BLonD) — adding or changing backend kernels, installing dev extras, running the test suite, building the Sphinx docs, or debugging pre-commit / doc-build failures.
---

# Developing in BLonD3

CERN Python code for simulating longitudinal beam dynamics in synchrotrons
(macroparticle tracking through RF systems, magnetic ramps, and collective
effects). Active dev branch is `blonder` (NOT `develop`/`master`).
Python ≥3.10, line length 79.

**Stay critical — the code can be wrong.** BLonD is under active development and
still has bugs. Don't assume existing code (or its comments/docstrings) is correct
just because it's there. If something looks off while you work — a suspicious
formula, a mislabelled variable, an inconsistent assumption — **say so explicitly**
and flag it, rather than silently building on it or "fixing" it to match. Surface
the doubt; don't paper over it.

**Layout** (full tree + descriptions in `CONTRIBUTING.md`):
- `blond/core/` — runtime: `simulation/` (assembles elements and drives the
  per-turn main loop), `ring/`, `beam/`, and `backends/` (numeric kernels).
- `blond/physics/` — RF stations, impedances, synchrotron radiation.
- `blond/acc_math/`, `cycles/`, `beam_preparation/`, `handle_results/` —
  analytic math, energy ramps, beam generation, observables.
- `blond/examples/scripts/` — `EX_01…EX_28`, the canonical usage patterns;
  `EX_01_Minimum_working_example.py` is the smallest end-to-end run. Read these
  first when unsure how an API is meant to be used.
- `blond/experimental/` — unstable code (excluded from coverage & pre-commit).
- `tests/unittests/` mirrors the `blond/` tree. The public API is whatever
  `blond/__init__.py` exports.

**Source of truth:** the repo-root `CONTRIBUTING.md` is the full human dev guide
(install/test/docs/release); `.gitlab-ci.yml` is the authoritative list of the commands
CI actually runs. Read `CONTRIBUTING.md` first — this skill captures only the
**non-obvious** parts and gotchas not spelled out there.

**Skills in this repo** (`.agents/skills/`):
- `blond-dev/` — *this* skill: developing/maintaining BLonD (install, test, backends, CI gotchas).
- `blond-assistant/` — *using* BLonD: writing simulation input files and the public API
  (`Ring`, RF stations, `Beam`, `MagneticCycle`, `WakeField`, …). Has a full
  `references/api_reference.md`. Consult it when writing or debugging a simulation script
  rather than the framework itself.

## Install (editable, with extras)

Extras are defined in `pyproject.toml` `[project.optional-dependencies]`:

| Goal | Command |
|------|---------|
| CPU dev | `pip install -e ".[dev]"` |
| GPU (CUDA 12 / 13) | `pip install -e ".[dev,gpu_cuda12]"` (or `gpu_cuda13`) |
| Docs | `pip install -e ".[doc]"` |
| XSuite interop | `pip install -e ".[dev,xsuite]"` |
| Everything | `".[all_no_cuda]"` / `".[all_cuda12]"` / `".[all_cuda13]"` |

`gpu_cuda12` vs `gpu_cuda13` must match the installed CUDA toolkit. After install,
`pre-commit install`. Native backends are optional: `blond-compile-cpp --parallel`,
`blond-compile-cuda` (CI does this before tests).

## Test

```bash
python -m pytest -v tests/unittests/
```

Backend-relevant env vars and markers:
- `BLOND_BACKEND_MODE` (`numba`/`cpp`/`cuda`/`python`); `BLOND_BACKEND_BITS` (the env var
  currently validates to `64` only — see precision note below).
- `BLOND_FORCE_TEST_ALL_BACKENDS=True` — fan a backend-aware test out over **every**
  available backend instead of just the selected one. **Set this whenever you touch
  backend code.**
- Markers (`pyproject.toml`): `backend_mutation`, `cupy`, `mpi`, `integration`.
  Exclude with `-m "not backend_mutation"`. MPI tests run under `mpirun -n 2 … -m "mpi"`.
- `pytest-randomly` randomizes order; reproduce a failure with `--randomly-seed=<N>`.
- **Tests run in random order *and* `backend_mutation` tests flip the global
  active backend (`set_specials`) mid-run.** So both the tests and the BLonD
  code they exercise must be **backend-agnostic**: never assume which backend is
  active, and restore any backend you change in teardown. An order- or
  backend-dependent test that passes on one seed will fail on another — if a
  failure only reproduces under some seeds, suspect leaked global state, not a
  flaky test.

## Backend conventions

A numeric kernel exists once **per backend** under
`blond/core/backends/<name>/callables.py` — `NumbaSpecials`, `CppSpecials`,
`CudaSpecials`, `PythonSpecials`, all subclasses of the `Specials` ABC in
`blond/core/backends/backend.py`. Activate one with `backend.set_specials("cpp")`
(`numba`/`cpp`/`cuda`/`python`). Backend-aware kernel tests live in
`tests/unittests/core/backends/test_backend.py`, looping over `special_modes` and
comparing each backend to the Python reference.

- **`backend` is a backend-agnostic drop-in for `np.`/`cp.` — prefer it over importing
  NumPy or CuPy directly.** The active backend object (`from blond import backend`, or
  `from blond.core.backends.backend import backend`) re-exports the array API under the
  same names: `backend.array`, `backend.zeros`, `backend.empty`, `backend.ones`,
  `backend.zeros_like`, `backend.arange`, `backend.linspace`, `backend.sin`, `backend.cos`,
  `backend.sqrt`, `backend.interp`, `backend.fft`, `backend.histogram`, `backend.random`,
  `backend.sum`, `backend.mean`, … plus the dtype/constants `backend.float`,
  `backend.complex`, `backend.pi`, `backend.twopi`. On a NumPy backend each maps to `np.*`;
  on a CuPy backend to `cp.*` — so writing `backend.zeros(n)` instead of `np.zeros(n)` makes
  the array land on the device the active backend uses (host or GPU) with **no `is_cupy`
  branching and no top-level `import cupy`**. In framework code that creates or operates on
  backend arrays, reach for `backend.<fn>` first; fall back to a literal `np.`/`cp.` only
  for the rare op the backend doesn't re-export (and then branch via `is_cupy_array`). This
  is also why you read precision from `backend.float`, not `np.float64` (see below).
- **Backend parity is mandatory.** Adding or changing a kernel means updating it in
  **all four** backends *and* the `Specials` ABC signature — not just the one you run
  locally. A kernel present in only some backends fails under
  `BLOND_FORCE_TEST_ALL_BACKENDS=True`. The `python` backend is the readable reference
  implementation; mirror its behaviour exactly in `numba`/`cpp`/`cuda`.
- **Arrays may be NumPy *or* CuPy — handle both.** Backend arrays are *not* guaranteed to
  be NumPy. The conversion rules:
  - **Use `copy_to_cpu(arr)`, never `arr.get()` directly.**
    `from blond.generals.cupy_.no_cupy_import import copy_to_cpu` returns a host copy for
    any backend (`.get()` for CuPy, `.copy()` for NumPy); calling `.get()` yourself crashes
    on a NumPy array.
  - **Never call `np.array(arr)` / `np.asarray(arr)` on a backend array without converting
    first.** On a CuPy array these silently mis-handle device memory (or raise) instead of
    giving you the data — convert via `copy_to_cpu(arr)` first.
  - Avoid other NumPy-only calls and bare `.copy()` on data that may live on the GPU; use
    `is_cupy_array(arr)` to branch when you must.
  - `copy_to_cpu` is the standard way tests (`blond/testing/backend_testing.py`),
    observations, and examples pull results off the GPU to compare against the CPU
    reference — reach for it whenever a test or readout needs concrete host numbers.
- **Don't hardcode 64-bit precision — and feed kernels the right dtype yourself.** The
  shipped backends (`Numpy64Bit`, `Cupy64Bit`) and the `BLOND_BACKEND_BITS` env var are
  64-bit, but the kernels are written *precision-generically* — they branch on
  `float32`/`complex64`. Read the float width from `backend.float` (and `backend.complex`)
  instead of assuming `np.float64`, build arrays at that precision
  (`backend.array(x, dtype=backend.float)`), and compare with tolerances (`rtol`/`atol`)
  rather than bit-exact equality. **There is no universal precision-coercion safety net:**
  the `numba` backend's `enforce_precision` decorator only casts stray *scalar* Python
  `float` arguments — it does not fix array dtypes, and the **`cpp` and `cuda` backends do
  no coercion at all.** They `assert` the incoming array dtype (which `python -O` strips —
  see the `assert` note below) and otherwise feed a wrong-precision array straight into a
  kernel compiled for a fixed type, which crashes or silently corrupts. The caller is
  responsible for passing correctly-typed arrays.
- **`assert` is intentional validation.** Backend wrappers validate dtype/contiguity
  with `assert` so `python -O` strips them from hot loops. Never propose
  `assert → raise`. Follow the same pattern for new wrapper validation.
- **Performance > defensive checks in hot paths.** The per-particle kernels are the
  innermost loop of the whole simulation (called every turn, per macroparticle), so they
  deliberately omit guards: `beam_phase` has no zero-profile guard, `fast_sin` keeps a
  short polynomial rather than chasing accuracy. Don't add "safety" checks, branches, or
  allocations inside these loops — push validation out to the wrapper (where `assert`
  lives) or do it once before the loop.
- **No host↔device transfers in the hot loop.** A `copy_to_cpu`/`.get()` (or a stray
  `np.asarray` that forces a sync) inside the per-turn tracking loop drags data back and
  forth across the PCIe bus every turn and destroys GPU performance. Keep beam/profile
  arrays resident on the device for the whole run; only pull to host for occasional
  observations/readouts, outside the inner loop. Physics correctness comes first, but
  a "correct" kernel that round-trips through host memory each turn is still a bug.
- **Many machines have no GPU — degrade gracefully, never hard-require CuPy.** The
  default/CI path is CPU (`numba`/`cpp`/`python`); CUDA is optional. Import CuPy through
  the `blond.generals.cupy_.no_cupy_import` shims (`copy_to_cpu`, `is_cupy_array`) which
  work whether or not CuPy is installed — never `import cupy` at module top level in code
  that must load CPU-only. Code and tests must run end-to-end with no GPU present; gate
  GPU-only tests behind the `cupy`/`cuda` markers so they skip cleanly instead of erroring.

## Coding conventions

- **Speaking names win — resist the physics-code pull toward one-letter variables.**
  Accelerator code tempts you to write `v`, `e`, `n`, `iv` for everything; don't.
  Use an intention-revealing name (`voltage`, `energy`, `n_macroparticles`,
  `induced_voltage`) so the reader knows the quantity *and* (via the SI rule
  below) its unit. The **only** terse names that belong here are the ones that *are* the
  established physics symbol for the quantity — `phi_s`, `eta`, `beta`, `gamma`,
  `alpha` (momentum compaction). A single letter is fine when it's the
  textbook symbol; it's not fine as a lazy abbreviation of a word.
- **Match the name already used for a quantity — don't invent a synonym.** The same value
  should have one name across the codebase; a second name for it is a bug waiting to
  happen (the `phi_rf_design` vs `phi_rf` split below is exactly how much a name can
  matter). Grep for how a quantity is already named in the module before naming a new
  variable.
- **The 79-char limit is not a licence for cryptic names.** If a good name makes the line
  too long, break the line — don't crush the name. Readability of the name beats saving a
  wrap.
- **Keep functions small and single-purpose; push complexity out of the hot loop.** Prefer
  a clearly-named helper over an inline block that needs a comment to explain it — but note
  the hot-path exception under *Backend conventions* (per-turn kernels deliberately stay
  flat and guard-free for performance; put the extracted validation in the wrapper, not the
  kernel).
- **Units are SI / accelerator-physics standard, not normalized:** voltages in
  **volts**, energies in **eV**, momenta in **eV/c**, times in **seconds**, lengths in
  **metres**, angles/phases in **radians**. Mixing these up is the most common physics bug.
- **Naming gotchas:** `phi_rf_design` is the design phase the RF controller targets;
  `phi_rf` is the actual/instantaneous phase — don't conflate them. `run_simulation` takes
  `callbacks=` (plural). `n_macroparticles` may be passed as a float (e.g. `1e6`).
- **`phi_s` is *only* the single-RF synchronous phase — not "the stable fixed point".**
  It is the analytic synchronous phase of a single-harmonic RF bucket. It is *not*
  interchangeable with the bucket's stable fixed point(s): with multiple harmonics or a
  non-trivial potential there can be one *or several* stable fixed points located
  elsewhere, and `phi_s` does not give them. Conflating the two was a real bug source in
  BLonD 2 — don't reintroduce that assumption.
- For the full list of simulation-authoring conventions and mistakes, see the
  `blond-assistant` skill (`.agents/skills/blond-assistant/`).

## MR / TDD workflow

> [!IMPORTANT]
> **NEVER run `git commit` before pre-commit passes. This is the single most
> important step in this section — do not skip it, ever.**
>
> 1. **First commit in a fresh checkout (CI/cloud agent, new clone): run
>    `pre-commit install` once.** Without it the git hook is absent and
>    `git commit` will NOT run the hooks — so the gate silently does nothing.
> 2. Before *every* commit, run `pre-commit run --all-files` (or
>    `pre-commit run --files <changed>`) and read the output.
> 3. Commit **only** once it reports all-green. If a hook auto-fixed files and
>    aborted, `git add` the changes and go back to step 2 — repeat until clean.
>
> The same hooks gate CI, so skipping the local run doesn't avoid the work — it
> just turns a 10-second local fix into a failed pipeline. A commit made without
> a passing pre-commit run is a mistake to be corrected, not a shortcut.

One GitLab MR per item, each on its own branch off `blonder`
(`blonder_feature/<topic>` or `blonder_bugfix/<topic>`); the user usually
**pre-creates the branch** — check `git branch --show-current` before making one.

- **Strict TDD with visible RED:** write the failing test, run it, show it failing,
  *then* implement. (User explicitly requires seeing RED.)
- Tests mirror the `blond/` tree under `tests/unittests/`.
- **Every test class must inherit from `unittest.TestCase` and use its
  assertions (`assertEqual`, `assertTrue`, `assertRaises`, …) — never a bare
  `assert` statement.** This is a different `assert` than the one in *Backend
  conventions* above: that note is about production wrapper code, where a
  stripped-under-`-O` `assert` is the intended fast-path guard. In test code
  bare `assert` is a bug risk, not a convention — it gives no diagnostic on
  failure (no expected-vs-actual) and is *also* silently stripped under
  `python -O`, which can turn a failing test into a silent pass. Write
  `class TestFoo(unittest.TestCase):` with `test_*` methods, not
  module-level `def test_...():` functions with bare `assert`.
- **Pre-commit before every `git commit`** — see the callout above; this is not
  optional.
- Commit messages: past tense ("Fixed …", "Added …"), body explains *why*.
- **MR descriptions: brief and informative — aim for the middle.** A single line
  is too little (don't just restate the title); a long, exhaustive write-up is too
  much and annoying to read. Target a short summary of *what* changed and *why*,
  plus anything a reviewer genuinely needs (breaking changes, follow-ups) — a few
  sentences or a handful of bullets, not paragraphs.
- When working a review backlog, tick items in `REVIEW_TODO.md` (repo root, untracked)
  with branch + commit hash.

## Repo conventions

- **Unstable code → `blond/experimental/`.** That folder is excluded from coverage *and*
  pre-commit, so it's where work-in-progress lives until it can pass the full gate. Don't
  put half-finished code in the main tree.
- **Leave `legacy/` and `blond/legacy/` alone.** These hold the old BLonD 2 code (reached
  via `from blond.legacy import blond2`) and are excluded from ruff / pre-commit / coverage
  / docs. Don't read, lint, refactor, or "fix" them unless the task is specifically about
  legacy code — they follow different conventions on purpose.
- **Never hand-edit generated files.** `blond/_version.py` is produced by `setuptools_scm`
  from git tags. The root `CLAUDE.md` / `AGENTS.md` are generated from this skill (see
  the banner in those files) — edit `.agents/skills/blond-dev/SKILL.md`, not the copies.
- **Public API lives in `blond/__init__.py`.** Anything exported there is the supported
  top-level API and shows up in the docs.

## CI gates (what blocks an MR)

`.gitlab-ci.yml` is authoritative, but an MR is rejected if it:
- **Decreases test coverage.** CI runs the suite under `--cov` and publishes the
  line-rate to GitLab; the project's MR rule fails on any drop. New code needs tests —
  budget for them, don't bolt them on after.
- **Fails pre-commit.** The hooks below run in CI too (ruff, isort, copyright,
  numpydoc, …); a hook that fails locally fails the pipeline.
- **Fails the doc build.** `sphinx-build … -W` treats warnings as errors (see below).

**Docs are Sphinx; docstrings are NumPy style.** Public-API docstrings follow the
[NumPy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html)
and are enforced by `numpydoc-validation`; the HTML docs are built with Sphinx
(`cd docs && bash create_docs.sh`). Write Parameters/Returns/Raises sections in
NumPy format or both the hook and the doc build will reject the MR.

## Common problems

**Pre-commit fails / blocks the commit.** (You should be hitting this from the
proactive `pre-commit run` in *MR / TDD workflow* above — i.e. *before* you
commit, not from a surprise at `git commit` time. Same fixes apply either way.)
- `no-commit-to-branch` blocks direct commits to `blonder`, `develop`, `master` —
  you must be on a feature branch.
- Several hooks auto-fix (isort, `ruff-format`, `ruff-check --fix`, pyupgrade,
  trailing-whitespace). They modify files and the commit aborts — **re-stage the
  changed files and commit again.**
- `check copyright` (`dev_tools/precommit_check_copyright.py`) rejects new `blond/`
  files missing the header from `dev_tools/copyright_notice.txt`. Bulk-apply:
  `python dev_tools/copy_copyright_to_all_files.py`.
- `numpydoc-validation` enforces NumPy-style docstrings on public API (config in
  `pyproject.toml`; `callables.py` files are excluded there).

**Doc build fails (`cd docs && bash create_docs.sh`).**
- `sphinx-build -b html . ./_build/html -W` uses **`-W`: warnings are errors.** A
  single broken cross-reference / bad docstring fails the whole build. Fix the
  warning; don't try to suppress it.
- `create_doc_blond_main_objects.py` **crashes by design** if a class you exported in
  `blond/__init__.py` isn't in its `ASSIGNED_CATEGORIES` dict. So whenever you add a new
  top-level export, also add the class name to `ASSIGNED_CATEGORIES` mapped to a
  `Categories` value. The script runs `print_unlinked_classes` and names exactly what's
  missing.

**Ruff/line length.** Line length is 79 everywhere (ruff + isort `--line-length 79`);
`E501` itself is ignored but `ruff-format` will still reflow.
