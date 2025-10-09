from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass


def get_files_with_extensions(
    folder: str | Path,
    extensions: tuple[str, ...],
    recursive: bool = True,
) -> list[Path]:
    """Retrieves all files in a folder that match a set of file extensions.

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
