# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Read and write BLonD simulation results as HDF5 files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from blond._version import __version__

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from os import PathLike
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
"""Schema version of the results file written by this BLonD version."""

FILE_SUFFIX = ".h5"

ATTR_FORMAT_VERSION = "blond_results_format_version"
ATTR_BLOND_VERSION = "blond_version"
ATTR_CREATED = "created"
ATTR_OBSERVABLE_CLASS = "observable_class"
ATTR_WRITE_IDX = "write_idx"


@dataclass
class GroupPayload:
    """
    In-memory image of one observable group of a results file.

    This is the unit a migration step operates on. It carries the
    group's datasets *and* its attributes, plus the group name, so that
    a migration can act on one observable only (for example renaming a
    dataset in ``BunchStatistics`` alone) and can rewrite group-level
    attributes such as ``observable_class``.
    """

    datasets: dict[str, tuple[NumpyArray, dict[str, Any]]] = field(
        default_factory=dict
    )
    """Dataset name -> (array, dataset attributes as a plain ``dict``)."""

    attrs: dict[str, Any] = field(default_factory=dict)
    """The group-level attributes as a plain ``dict``."""

    group_name: str = ""
    """Name of the group in the results file, without the leading path."""


MIGRATIONS: dict[int, Callable[[GroupPayload], GroupPayload]] = {}
"""Upgraders keyed by the version they migrate *from*, to that version + 1."""


class ResultsFormatError(Exception):
    """Raised when a results file cannot be interpreted safely."""


def results_filepath(stem: str | PathLike) -> Path:
    """
    Return the results file path for ``stem``.

    Parameters
    ----------
    stem : str or os.PathLike
        The path to the results file, with or without the
        ``.h5`` suffix.

    Returns
    -------
    pathlib.Path
        ``stem`` with the ``.h5`` suffix appended, unless it is
        already present.
    """
    path = Path(stem)
    if path.suffix == FILE_SUFFIX:
        return path
    return path.with_name(path.name + FILE_SUFFIX)


def create_results_file(
    stem: str | PathLike,
    overwrite: bool = True,
) -> h5py.File:
    """
    Create a new results file and stamp its root attributes.

    Parameters
    ----------
    stem : str or os.PathLike
        The path to the results file, with or without the
        ``.h5`` suffix.
    overwrite : bool, optional
        If ``True`` (default), overwrite an existing file at the
        target path. If ``False``, raise ``FileExistsError`` when
        the file already exists.

    Returns
    -------
    h5py.File
        The newly created, open results file, stamped with the
        current format version, BLonD version, and creation time.
    """
    filepath = results_filepath(stem)
    file = h5py.File(filepath, "w" if overwrite else "w-")
    try:
        file.attrs[ATTR_FORMAT_VERSION] = FORMAT_VERSION
        file.attrs[ATTR_BLOND_VERSION] = __version__
        file.attrs[ATTR_CREATED] = datetime.now(timezone.utc).isoformat()
    except Exception:
        file.close()
        raise
    return file


def open_results_file(stem: str | PathLike) -> h5py.File:
    """
    Open an existing results file for reading.

    Parameters
    ----------
    stem : str or os.PathLike
        The path to the results file, with or without the
        ``.h5`` suffix.

    Returns
    -------
    h5py.File
        The open results file.

    Raises
    ------
    FileNotFoundError
        If no file exists at the target path.
    ResultsFormatError
        If the file has no format version attribute, or was
        written with a format version newer than this BLonD
        installation understands.
    """
    filepath = results_filepath(stem)
    if not filepath.is_file():
        raise FileNotFoundError(f"No results file at {filepath}.")
    file = h5py.File(filepath, "r")
    try:
        file_format_version = read_format_version(file)
    except Exception:
        file.close()
        raise
    if file_format_version > FORMAT_VERSION:
        file.close()
        raise ResultsFormatError(
            f"{filepath} was written with results format version"
            f" {file_format_version}, but this BLonD ({__version__})"
            f" understands at most version {FORMAT_VERSION}."
            f" Upgrade BLonD to read this file."
        )
    return file


def read_format_version(file: h5py.File) -> int:
    """
    Read the results format version stamped on ``file``.

    Parameters
    ----------
    file : h5py.File
        An open results file.

    Returns
    -------
    int
        The format version stamped in the file's root attributes.

    Raises
    ------
    ResultsFormatError
        If ``file`` has no format version attribute, meaning it is
        not a BLonD results file.
    """
    try:
        return int(file.attrs[ATTR_FORMAT_VERSION])
    except KeyError:
        raise ResultsFormatError(
            f"{file.filename} has no {ATTR_FORMAT_VERSION!r} attribute"
            f" and is not a BLonD results file."
        ) from None


def read_group_payload(group: h5py.Group) -> GroupPayload:
    """
    Read every dataset and attribute of ``group`` into memory.

    Parameters
    ----------
    group : h5py.Group
        The group whose datasets and attributes to read.

    Returns
    -------
    GroupPayload
        The group's datasets, its group-level attributes, and its name.
    """
    return GroupPayload(
        datasets={
            name: (dataset[()], dict(dataset.attrs))
            for name, dataset in group.items()
        },
        attrs=dict(group.attrs),
        group_name=group.name.rsplit("/", 1)[-1],
    )


def migrate_payload(
    payload: GroupPayload,
    from_version: int,
    to_version: int = FORMAT_VERSION,
    migrations: (
        dict[int, Callable[[GroupPayload], GroupPayload]] | None
    ) = None,
) -> GroupPayload:
    """
    Migrate a group payload from one format version to another.

    Parameters
    ----------
    payload : GroupPayload
        The payload to migrate, as returned by
        `read_group_payload`.
    from_version : int
        The format version ``payload`` was written with.
    to_version : int, optional
        The format version to migrate ``payload`` to. Defaults to
        `FORMAT_VERSION`.
    migrations : dict of int to callable, optional
        Upgraders keyed by the version they migrate from. Defaults
        to `MIGRATIONS`.

    Returns
    -------
    GroupPayload
        The migrated payload.

    Raises
    ------
    ResultsFormatError
        If ``from_version`` is newer than ``to_version``, or if no
        migration is registered for a version in the chain from
        ``from_version`` to ``to_version``.
    """
    if migrations is None:
        migrations = MIGRATIONS
    if from_version > to_version:
        raise ResultsFormatError(
            f"The results were written with results format version"
            f" {from_version}, but this BLonD ({__version__})"
            f" understands at most version {to_version}."
            f" Upgrade BLonD to read this file."
        )
    for version in range(from_version, to_version):
        if version not in migrations:
            raise ResultsFormatError(
                f"No migration registered from results format version"
                f" {version} to {version + 1}."
            )
        logger.info(f"Migrating results payload {version} -> {version + 1}.")
        payload = migrations[version](payload)
    return payload
