# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Store the state of a `BeamBaseClass` as an HDF5 file.

What a beam consists of is defined once, by
`blond.core.beam.base.BeamBaseClass.to_dict`; this module only maps such a
dictionary onto HDF5 and back. Nested dictionaries become groups, arrays
become datasets and everything else becomes an attribute, so the file layout
follows the dictionary.

The file carries a `BEAM_SCHEMA_VERSION` attribute. Files written by older
BLonD versions are upgraded on load by the chained migrations in
`blond.core.beam.migrations`, so the layout may only change together with a
version bump. ``tests/unittests/core/beam/test_serialization.py`` enforces
that by fingerprinting the layout of a written file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np

from blond._version import __version__
from blond.core.beam.migrations import migrate_raw_beam

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike
    from typing import Any

    from blond.core.beam.base import BeamBaseClass

    RawBeam = dict[str, Any]

#: Version of the on-disk beam layout written by this BLonD version. Bump
#: whenever the layout changes and register a migration in
#: `blond.core.beam.migrations`.
BEAM_SCHEMA_VERSION = 1


def save_beam(beam: BeamBaseClass, path: str | PathLike) -> None:
    """
    Write a beam to an HDF5 file.

    Particle coordinates are copied to host memory, so that a file written
    on a GPU node can be read on a machine without a GPU.

    Parameters
    ----------
    beam
        Beam to write. It must be set up and not distributed.
    path
        Destination file path. An existing file is overwritten.

    See Also
    --------
    load_beam : Read a beam back from an HDF5 file.
    blond.core.beam.base.BeamBaseClass.to_dict : Defines what is written.
    """
    state = beam.to_dict()
    # Provenance of the file rather than state of the beam: it names the
    # BLonD release that wrote the file, which `schema_version` does not.
    state["blond_version"] = __version__

    with h5py.File(path, "w") as file:
        _write_group(file, state)


def load_beam(path: str | PathLike) -> BeamBaseClass:
    """
    Read a beam from an HDF5 file written by `save_beam`.

    Files of an older schema version are migrated to the current layout
    before the beam is rebuilt. Particle coordinates are placed on the
    active backend, so a file written on a CPU can be loaded on a GPU.

    Parameters
    ----------
    path
        Source file path.

    Returns
    -------
    beam
        The restored beam.

    Raises
    ------
    ValueError
        If the file is not a BLonD beam file, or if it holds a beam class
        this BLonD version does not know.

    See Also
    --------
    save_beam : Write a beam to an HDF5 file.
    blond.core.beam.base.BeamBaseClass.from_dict : Rebuilds the beam.
    """
    # Imported here because `blond.core.beam.base` reaches this module
    # through `BeamBaseClass.save`.
    from blond.core.beam.base import BeamBaseClass

    state = _read_raw(path)
    state = migrate_raw_beam(state, target_version=BEAM_SCHEMA_VERSION)
    return BeamBaseClass.from_dict(state)


def _write_group(group: h5py.Group, state: RawBeam) -> None:
    """
    Write a state dictionary into an HDF5 group.

    Parameters
    ----------
    group
        Group to write into.
    state
        Values to write. Dictionaries become subgroups, arrays become
        datasets, everything else becomes an attribute.
    """
    for name, value in state.items():
        if isinstance(value, dict):
            _write_group(group.create_group(name), value)
        elif isinstance(value, np.ndarray):
            group.create_dataset(
                name,
                data=value,
                # Chunked storage, which compression requires, is not
                # possible for the empty arrays of an `EmptyBeam`.
                compression="gzip" if value.size > 0 else None,
            )
        else:
            group.attrs[name] = _as_attribute(name, value)


def _as_attribute(name: str, value: Any) -> Any:
    """
    Convert a value to what is stored as an HDF5 attribute.

    Parameters
    ----------
    name
        Name of the value, used in the error message.
    value
        Value to convert.

    Returns
    -------
    attribute
        The value as written to file. ``None`` becomes an empty attribute,
        which is how HDF5 writes a null value and keeps the file layout
        independent of whether the value is set.

    Raises
    ------
    TypeError
        If the value has a type that cannot be written to HDF5.
    """
    if value is None:
        return h5py.Empty(np.float64)
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return np.bool_(value)
    if isinstance(value, (int, np.integer)):
        return np.int64(value)
    if isinstance(value, (float, np.floating)):
        return np.float64(value)
    raise TypeError(
        f"Cannot write {name!r} of type {type(value).__name__} to HDF5."
    )


def _read_raw(path: str | PathLike) -> RawBeam:
    """
    Read a beam file into a nested dictionary, without interpreting it.

    The reader walks whatever the file contains instead of assuming the
    current layout, so that files of older schema versions can be read and
    handed to the migrations.

    Parameters
    ----------
    path
        Source file path.

    Returns
    -------
    state
        The beam state as it is stored in the file.

    Raises
    ------
    ValueError
        If the file carries no schema version and hence was not written by
        `save_beam`.
    """
    with h5py.File(path, "r") as file:
        state = _read_group(file)

    if "schema_version" not in state:
        raise ValueError(
            f"{path} is not a BLonD beam file: it has no `schema_version` "
            "attribute."
        )
    state["schema_version"] = int(state["schema_version"])
    return state


def _read_group(group: h5py.Group) -> RawBeam:
    """
    Read one HDF5 group into a dictionary, recursing into subgroups.

    Parameters
    ----------
    group
        Group to read.

    Returns
    -------
    state
        The attributes and datasets of the group, plus one nested dictionary
        per subgroup.
    """
    state: RawBeam = {
        name: _from_attribute(value) for name, value in group.attrs.items()
    }
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            state[name] = _read_group(item)
        else:
            state[name] = item[()]
    return state


def _from_attribute(value: Any) -> Any:
    """
    Convert an HDF5 attribute back to a plain Python value.

    Parameters
    ----------
    value
        Attribute as read from file.

    Returns
    -------
    value
        An empty attribute becomes ``None``, a byte string becomes `str`,
        anything else is returned unchanged.
    """
    if isinstance(value, h5py.Empty):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
