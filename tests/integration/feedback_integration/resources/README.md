# BLonD 2 reference files

The `*_blond2.npz` files in this directory are **BLonD 2 reference files**: outputs of
BLonD 2 (`blond.legacy.blond2`), frozen so the tests next to this directory
can compare BLonD 3 against them without running BLonD 2.

| BLonD 2 reference file | Written by |
|-------------|------------|
| `lhc_beam_control_single_bunch_injection_blond2.npz` | `test_lhc_beam_control_single_bunch_injection.py::TestSingleBunchInjectionWithPhaseLoop::setUpClass` |

- **Don't edit these files by hand.** Regenerate them.
- A test fails with `FileNotFoundError` when its BLonD 2 reference file is missing.
  That is intentional. Don't work around it; rewrite the file.
- **To rewrite:** set `REWRITE_BLOND2_REFERENCE_FILE = True` in the test module and
  run it in the pinned environment from `tests/blond2-reference-requirements.txt`
  (the header of that file has the exact commands). Then set the flag back to
  `False` and commit the `.npz`.
- **Debugging a mismatch:** each file stores the environment that produced
  it (`pip_list`, Python version, platform, git commit, date) as JSON under
  the `blond2_reference_environment` key:

  ```bash
  python -c "import numpy as np, sys; print(np.load(sys.argv[1])['blond2_reference_environment'])" <file.npz>
  ```

The long-term aim is to stop running BLonD 2 anywhere in the test suite, so
ideally these files never need rewriting.
