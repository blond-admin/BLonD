Here are some helper scripts to autogenerate empty testcases.
1. Run pytest to generate `coverage.json` file
     - `--cov=blond --cov-report=term --cov-report=json`
2. Run `make_test_stubs.py` to automatically generate testcases.
3. Rework the generated files.