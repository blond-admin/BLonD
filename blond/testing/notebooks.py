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

    # Open the output Python file
    with open(output_path, "w", encoding="utf-8") as python_file:
        # Loop through each cell in the notebook
        for cell in notebook.cells:
            # Check if the cell is a code cell
            if cell.cell_type == "code":
                # Write the code to the Python file
                python_file.write(
                    f"# Cell {cell.execution_count}:\n"
                )  # Optionally add a header with the cell's execution count
                python_file.write(cell.source)
                python_file.write("\n\n")  # Add a newline between cells
