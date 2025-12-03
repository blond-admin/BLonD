# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Functions that help working with files.

Authors
-------
Simon Lauber
"""

from __future__ import annotations

from pathlib import Path


def get_files_with_extensions(
    folder: str | Path,
    extensions: tuple[str, ...],
    recursive: bool = True,
) -> list[Path]:
    """
    Retrieves all files in a folder that match a set of file extensions.

    Parameters
    ----------
    folder
        The path to the folder to search.
    extensions
        File extensions to match (e.g., {'.txt', '.md'}).
    recursive
        Whether to search subdirectories recursively. Defaults to True.

    Returns
    -------
    files
        A list of Path objects for files that match the given extensions.
    """
    folder = Path(folder)
    files = folder.rglob("*") if recursive else folder.glob("*")

    return [f for f in files if f.is_file() and f.suffix in extensions]
