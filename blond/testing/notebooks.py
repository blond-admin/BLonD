# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper tools to test Jupyter-Notebooks."""

import nbformat


def ipynb_to_py(notebook_path: str, output_path: str) -> None:
    """
    Convert a jupyter notebook to a python file.

    Parameters
    ----------
    notebook_path
        Path of the jupyter notebook to convert.
    output_path
        Path of the Python file to be written.
    """
    assert notebook_path.endswith("ipynb"), f"{notebook_path=}"
    assert output_path.endswith("py"), f"{output_path=}"
    # Load the notebook
    with open(notebook_path, encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    # Collect all `from __future__` imports from code cells — they must appear
    # at the top of the generated file (Python syntax requirement), but users
    # may legitimately repeat them in later cells so each section works when
    # run in isolation inside Jupyter.
    future_imports: set[str] = set()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for line in cell.source.splitlines():
                stripped = line.strip()
                if stripped.startswith("from __future__"):
                    future_imports.add(stripped)

    # Open the output Python file
    with open(output_path, "w", encoding="utf-8") as python_file:
        # Write hoisted __future__ imports first
        for stmt in sorted(future_imports):
            python_file.write(stmt + "\n")
        if future_imports:
            python_file.write("\n")

        # Loop through each cell in the notebook
        for cell in notebook.cells:
            # Check if the cell is a code cell
            if cell.cell_type == "code":
                # Strip any `from __future__` lines — already written at top
                source_lines = [
                    line
                    for line in cell.source.splitlines(keepends=True)
                    if not line.strip().startswith("from __future__")
                ]
                source = "".join(source_lines).strip("\n")
                if not source:
                    continue
                # Write the code to the Python file
                python_file.write(
                    f"# Cell {cell.execution_count}:\n"
                )  # Optionally add a header with the cell's execution count
                python_file.write(source)
                python_file.write("\n\n")  # Add a newline between cells
