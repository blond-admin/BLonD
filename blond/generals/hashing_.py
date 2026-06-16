# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Functions that help with hashing."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence
    from pathlib import Path


def hash_files(file_paths: list[str], base_folder: str | None = None) -> str:
    """
    Compute a SHA-256 hash from the contents of a list of files.

    Each file is read in binary mode and processed in chunks. The file paths
    are sorted to ensure consistent ordering, and their names are included in
    the hash to avoid collisions from identical file content.

    Parameters
    ----------
    file_paths
        A list of file paths to include in the hash.
    base_folder
        If given, the name folded into the digest is each file's path
        *relative to* this folder (with ``/`` separators), instead of its
        absolute path. The hash is then independent of where the tree is
        checked out -- a CI build path and a local clone produce the same
        digest -- while files are still read from their actual location.
        Without it, the absolute path leaks into the hash.

    Returns
    -------
    hash_
        The resulting SHA-256 hexadecimal digest.
    """
    file_paths = sorted(file_paths)
    hasher = hashlib.sha256()

    for file_path in file_paths:
        if base_folder is not None:
            # Path relative to the folder, with forward slashes, so the digest
            # depends on layout/content but not on the absolute checkout path.
            name = str(os.path.relpath(file_path, base_folder)).replace(
                os.sep, "/"
            )
        else:
            name = file_path
        hasher.update(name.encode("utf-8"))  # Include file name for uniqueness
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

    return hasher.hexdigest()


def hash_in_folder(
    folder: str | Path,
    extensions: tuple[str, ...],
    recursive: bool = True,
):
    """
    Load file contents of all files in folder and generate hash from it.

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
    hash_
        The resulting SHA-256 hexadecimal digest.

    Notes
    -----
    The digest depends only on the file *contents* and their names relative
    to ``folder`` -- not on the absolute location of ``folder`` -- so it is
    stable across checkout paths (e.g. a CI runner vs a local clone).
    """
    from blond.generals.files_ import get_files_with_extensions

    files = get_files_with_extensions(
        folder=folder,
        extensions=extensions,
        recursive=recursive,
    )
    paths = [str(f) for f in files]
    base = str(folder)
    if platform.system() == "Windows":  # case-insensitive filesystem
        paths = [p.lower() for p in paths]
        base = base.lower()
    _hash = hash_files(paths, base_folder=base)
    return _hash


def hash_build_target(
    folder: str | Path,
    extensions: tuple[str, ...],
    *,
    probe_commands: Sequence[Sequence[str]] = (),
    extra: Sequence[str] = (),
    recursive: bool = False,
) -> str:
    """
    Hash the sources of a backend together with its build environment.

    The returned digest identifies not only the source files (those in
    ``folder`` matching ``extensions``) but everything *external* to the
    source tree that determines whether a produced binary is safe to run on
    a given machine:

    * the stdout/stderr of each command in ``probe_commands`` -- e.g.
      ``[compiler, "--version"]`` for the toolchain identity, or a
      ``[compiler, "-march=native", "-dM", "-E", "-xc", "-"]`` macro dump for
      the exact instruction set ``-march=native`` enables on this host, or
      ``[nvcc, "--version"]`` for the CUDA toolkit version;
    * any ``extra`` strings -- e.g. caller-supplied compiler flags or the GPU
      compute capability.

    A different compiler version, target ISA/CPU, CUDA version or flag set
    therefore yields a different digest, hence a different
    ``compiled/<digest>/`` directory and a rebuild -- never an unsafe reuse of
    a binary built for an incompatible machine. This is what makes a shared
    (e.g. CI-cached) ``compiled/`` directory safe across heterogeneous
    runners.

    Probe commands are run with empty stdin; a probe that cannot be executed
    contributes its error text instead of raising. That text is still
    environment-specific, so it remains safe: at worst it forces a rebuild.

    Parameters
    ----------
    folder
        The path to the folder whose sources are hashed.
    extensions
        Source file extensions to match (e.g. ``(".py", ".cpp", ".h")``).
    probe_commands
        Commands whose combined output fingerprints the toolchain.
    extra
        Additional strings folded into the digest verbatim.
    recursive
        Whether to search subdirectories recursively. Defaults to False.

    Returns
    -------
    hash_
        The resulting SHA-256 hexadecimal digest.
    """
    hasher = hashlib.sha256()
    # Start from the source digest (content + relative names; see
    # `hash_in_folder` -- deliberately location-independent).
    hasher.update(
        hash_in_folder(
            folder=folder, extensions=extensions, recursive=recursive
        ).encode("utf-8")
    )
    for command in probe_commands:
        # Domain-separate each field with a tag that cannot occur in the
        # data itself (NUL bytes never appear in our source digest, version
        # banners or flag strings). Without this, concatenating two fields
        # could collide -- e.g. ("ab", "c") and ("a", "bc") would otherwise
        # hash identically -- which would let distinct toolchains share a key.
        hasher.update(b"\x00probe\x00")
        try:
            result = subprocess.run(
                list(command),
                # Empty stdin: probes like `gcc -dM -E -xc -` read from
                # stdin; feeding "" makes them emit the predefined macros
                # (the target ISA) and return promptly instead of blocking.
                input=b"",
                capture_output=True,
                check=False,
                # Fold both streams: some tools print their version/target
                # info to stderr rather than stdout.
                timeout=60,
            )
            hasher.update(result.stdout)
            hasher.update(result.stderr)
        except Exception as error:
            # A probe we cannot run (compiler absent, timed out, ...) must
            # not crash the build. We fold the error text instead: it is
            # still environment-specific and deterministic, so the worst
            # case is a cache miss (rebuild) -- never a wrong/unsafe reuse.
            hasher.update(repr(error).encode("utf-8"))
    for item in extra:
        # Same domain-separation rationale as the probe tag above.
        hasher.update(b"\x00extra\x00")
        hasher.update(item.encode("utf-8"))
    return hasher.hexdigest()
