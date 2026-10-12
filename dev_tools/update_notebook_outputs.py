"""Regenerate the executed ``.ipynb`` next to each example marimo notebook.

The example notebooks in ``blond/examples/notebooks/`` are marimo ``.py``
files; that is the source developers edit. The docs render them through
nbsphinx, which needs an ``.ipynb`` with outputs. Those ``.ipynb`` files are
committed and only change when a developer runs this script, so the docs
build never executes the notebooks.

The outputs are therefore as old as the last run of this script: rerun it
after changing a notebook, or after a BLonD change that should show up in the
example outputs.

Usage::

    python dev_tools/update_notebook_outputs.py [NOTEBOOK.py ...]

Without arguments, every notebook in ``blond/examples/notebooks/`` is
exported.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "blond" / "examples" / "notebooks"


def export_notebook(notebook: Path) -> None:
    """
    Execute a marimo notebook and write it as ``.ipynb`` with outputs.

    Parameters
    ----------
    notebook
        Path of the marimo ``.py`` notebook. The ``.ipynb`` is written
        next to it, with the same name.

    Raises
    ------
    RuntimeError
        If the export contains ``<marimo-tex>`` tags, which only marimo's
        own frontend renders, not MathJax in the docs.
    """
    target = notebook.with_suffix(".ipynb")
    environment = dict(os.environ, MPLBACKEND="Agg")
    subprocess.run(
        [sys.executable, "-m", "marimo", "export", "ipynb", str(notebook),
         "-o", str(target), "--include-outputs", "--force"],
        check=True,
        env=environment,
    )  # fmt: skip
    if "marimo-tex" in target.read_text(encoding="utf-8"):
        raise RuntimeError(
            f"{target} contains <marimo-tex> math, which the docs cannot "
            "render. Write the math in a static mo.md() cell instead."
        )


def main(notebooks: list[str]) -> None:
    """
    Export the given notebooks, or all example notebooks.

    Parameters
    ----------
    notebooks
        Paths of marimo notebooks. Empty means every notebook in
        ``blond/examples/notebooks/``.
    """
    if notebooks:
        paths = [Path(notebook).resolve() for notebook in notebooks]
    else:
        paths = sorted(
            path
            for path in NOTEBOOK_DIR.glob("*.py")
            if path.name != "__init__.py"
        )
    for path in paths:
        print(f"Exporting {path.relative_to(ROOT)}", flush=True)
        export_notebook(path)


if __name__ == "__main__":
    main(sys.argv[1:])
