from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


def hash_files(file_paths: list[str]) -> str:
    """Computes a SHA-256 hash from the contents of a list of files.

    Each file is read in binary mode and processed in chunks.
    The file paths are sorted to ensure consistent ordering,
    and their names are optionally included in the hash to
    avoid collisions from identical file content.

    Parameters
    ----------
    file_paths
        A list of file paths to include in the hash.

    Returns:
    -------
    hash_
        The resulting SHA-256 hexadecimal digest.
    """
    file_paths = sorted(file_paths)

    hasher = hashlib.sha256()

    for file_path in file_paths:
        hasher.update(
            file_path.encode("utf-8")
        )  # Include file name for uniqueness
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

    return hasher.hexdigest()


def hash_in_folder(
    folder: str | Path,
    extensions: set[str],
    recursive: bool = True,
):
    """Load file contents of all files in folder and generate hash from it.

    Parameters
    ----------
    folder
        The path to the folder to search.
    extensions
        A set of file extensions to match (e.g., {'.txt', '.md'}).
    recursive
        Whether to search subdirectories recursively. Defaults to True.

    Returns:
    -------
    hash_
        The resulting SHA-256 hexadecimal digest.
    """
    from ._files import get_files_with_extensions

    files = get_files_with_extensions(
        folder=folder,
        extensions=extensions,
        recursive=recursive,
    )
    _hash = hash_files([str(f) for f in files])
    return _hash
