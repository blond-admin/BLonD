"""Run ruff's missing-docstring rules on private folders and files, too.

ruff's ``D1xx`` rules treat everything inside a ``_private`` package or
module as private and skip it. In BLonD a leading underscore on a folder
or file marks it as internal -- it is not meant to switch off the
docstring checks. So this hook hands such files to ruff under the same
path with the underscores removed. How ruff treats the code itself
(``_private`` function names, ``noqa`` comments, overloads, ...) stays
untouched. Files with a public path are skipped, the ruff hook covers
them already.
"""

import json
import subprocess
import sys
from pathlib import PurePath


def public_path(path: str) -> str:
    """Remove the leading underscore of every private folder and file.

    Parameters
    ----------
    path
        Path of a Python file.

    Returns
    -------
    public_path
        The path with ``_name`` turned into ``name``; dunder names like
        ``__init__.py`` are kept.
    """
    parts = [
        part.lstrip("_")
        if part.startswith("_") and not part.startswith("__")
        else part
        for part in PurePath(path).parts
    ]
    return PurePath(*parts).as_posix()


def main(paths: list[str]) -> int:
    """Check the files in private paths with ruff's ``D1xx`` rules.

    Parameters
    ----------
    paths
        Python files to check, as passed by pre-commit.

    Returns
    -------
    exit_code
        0 if no docstring is missing, else 1.
    """
    exit_code = 0
    for path in paths:
        checked_as = public_path(path)
        if checked_as == PurePath(path).as_posix():
            continue
        with open(path, encoding="utf-8") as file:
            source = file.read()
        result = subprocess.run(
            [
                "ruff",
                "check",
                "--select=D1",
                "--no-fix",
                "--output-format=json",
                f"--stdin-filename={checked_as}",
                "-",
            ],
            input=source,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(f"ruff failed on {path}:\n{result.stderr}")
        for violation in json.loads(result.stdout):
            location = violation["location"]
            print(
                f"{path}:{location['row']}:{location['column']}: "
                f"{violation['code']} {violation['message']}"
            )
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
