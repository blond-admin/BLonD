# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Generate `folder_overview.md` to be copied elsewhere."""

import ast
import os
import warnings
from pathlib import Path

skipfolders = [
    "legacy/blond2/",  #  Would be another full project, bloating the output.
]


def extract_docstring(file_path: str) -> str:
    """
    Extract the docstring from the __init__.py file.

    Parameters
    ----------
    file_path
        Target to extract the top-level docstring from.

    Returns
    -------
    docstring
        The first line of the docstring
    """
    with open(file_path, encoding="utf-8") as f:
        file_content = f.read()
    # Parse the Python file and extract the docstring of the first module
    tree = ast.parse(file_content)
    docstring = ast.get_docstring(tree)
    if docstring is None:
        warnings.warn(
            f"\n{file_path}:1 is missing a docstring.",
            UserWarning,
            stacklevel=3,
        )
        return ""
    return docstring.split("\n")[0]


def generate_file_description_table(root_dir: os.PathLike) -> list[str]:
    """
    Recursively traverse directories and generate an overview with folder structure.

    Parameters
    ----------
    root_dir
        The location where to start generating the overview.

    Returns
    -------
    content
        The overview for each folder as a string.

    """
    content: list[str] = []

    for root, _, files in os.walk(root_dir):
        skip = False
        for f in skipfolders:
            if f in root:
                skip = True
        if skip:
            continue
        try:
            indent = root[root.index("blond/") :].count("/")
        except ValueError:
            indent = 0

        if "__init__.py" in files:
            _add_line(indent, content, root)

    return content


def _add_line(indent: int, content_write: list[str], root: str) -> None:
    """
    Add line to `markdown_overview`.

    Parameters
    ----------
    indent
        The current indent, representing the depth in the directory tree.
    content_write
        The list of strings.
    root
        The root of the current file.
    """
    # Get the folder name and path of __init__.py
    folder_name = os.path.basename(root)
    init_path = os.path.join(root, "__init__.py")
    # Extract the docstring
    docstring = extract_docstring(init_path)
    # Markdown entry for the folder and docstring
    indent_string = "" if indent == 0 else f"├──{2 * (indent) * '─'} "
    folder_string = f"{indent_string}{folder_name}/"
    max_spaces = 30
    spaces = max_spaces - len(folder_string)
    content_write.append(f"{folder_string}{spaces * ' '}{docstring}")


def main():
    """Generate `folder_overview.md`."""
    target = "folder_overview.md"
    this_dir = Path(__file__).parent
    BLOND_ROOT = (this_dir / "../blond/").resolve()
    # Set the root directory (change this to the starting point)
    root_directory = BLOND_ROOT
    # Generate the markdown overview
    head = (
        "<!-- Automatically created using "
        "`dev_tools/create_tables.py` -->\n```\n"
    )
    tail = "\n```\n"
    markdown = (
        head
        + ("\n").join(generate_file_description_table(root_directory))
        + tail
    )
    # Print or save the markdown output
    print(markdown)
    with open(BLOND_ROOT.parent / target, "w", encoding="utf-8") as f:
        f.write(markdown)


if __name__ == "__main__":
    main()
