# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Save and load `BeamBaseClass` objects as HDF5 files.

The file carries a `BEAM_SCHEMA_VERSION` attribute. Files written by older
BLonD versions are upgraded on load by the chained migrations in
`blond.core.beam.migrations`, so the layout below may only change together
with a version bump. ``tests/unittests/core/beam/test_serialization.py``
enforces that by fingerprinting the layout of a written file.

Layout::

    /                      schema_version, blond_class, intensity,
                           is_counter_rotating
    /particle_type         mass, charge, user_decay_rate
    /reference             time, has_total_energy, total_energy
    /particles             dt, dE, flags, ids  (datasets)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np

from blond.core.backends.backend import backend
from blond.core.beam.migrations import migrate_raw_beam
from blond.generals.cupy_.no_cupy_import import copy_to_cpu

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

    Particle coordinates are copied to host memory first, so that a file
    written on a GPU node can be read on a machine without a GPU.

    Parameters
    ----------
    beam
        Beam to write. It must be set up and not distributed.
    path
        Destination file path. An existing file is overwritten.

    Raises
    ------
    ValueError
        If the beam has no particle arrays yet, i.e. ``setup_beam`` was
        never called.
    NotImplementedError
        If the beam is distributed over several MPI ranks.

    See Also
    --------
    load_beam : Read a beam back from an HDF5 file.
    """
    if beam.is_distributed:
        raise NotImplementedError(
            "Saving a distributed beam is not supported; gather the beam on "
            "a single rank first."
        )
    if not beam.is_set_up():
        raise ValueError(
            "The beam is not set up, there is nothing to save. Call "
            "`setup_beam(...)` first."
        )

    total_energy = beam.reference._total_energy

    with h5py.File(path, "w") as file:
        file.attrs["schema_version"] = BEAM_SCHEMA_VERSION
        file.attrs["blond_class"] = type(beam).__name__
        file.attrs["intensity"] = np.int64(beam.intensity)
        file.attrs["is_counter_rotating"] = np.bool_(beam.is_counter_rotating)

        particle_type = file.create_group("particle_type")
        particle_type.attrs["mass"] = np.float64(beam.particle_type.mass)
        particle_type.attrs["charge"] = np.float64(beam.particle_type.charge)
        particle_type.attrs["user_decay_rate"] = np.float64(
            beam.particle_type.user_decay_rate
        )

        reference = file.create_group("reference")
        reference.attrs["time"] = np.float64(beam.reference.time)
        # HDF5 has no null value, so the placeholder below is only
        # meaningful when `has_total_energy` is set.
        reference.attrs["has_total_energy"] = np.bool_(
            total_energy is not None
        )
        reference.attrs["total_energy"] = np.float64(
            0.0 if total_energy is None else total_energy
        )

        particles = file.create_group("particles")
        arrays = {
            "dt": beam.read_partial_dt(),
            "dE": beam.read_partial_dE(),
            "flags": beam.read_partial_flags(),
            "ids": beam.read_partial_ids(),
        }
        for name, array in arrays.items():
            data = copy_to_cpu(array)
            particles.create_dataset(
                name,
                data=data,
                # Chunked storage, which compression requires, is not
                # possible for the empty arrays of an `EmptyBeam`.
                compression="gzip" if data.size > 0 else None,
            )


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
    """
    raw = _read_raw(path)
    raw = migrate_raw_beam(raw, target_version=BEAM_SCHEMA_VERSION)
    return _beam_from_raw(raw)


def _read_raw(path: str | PathLike) -> RawBeam:
    """
    Read a beam file into a nested mapping, without interpreting it.

    The reader walks whatever the file contains instead of assuming the
    current layout, so that files of older schema versions can be read and
    handed to the migrations.

    Parameters
    ----------
    path
        Source file path.

    Returns
    -------
    raw
        Root attributes, with one nested mapping per group.

    Raises
    ------
    ValueError
        If the file carries no schema version and hence was not written by
        `save_beam`.
    """
    with h5py.File(path, "r") as file:
        raw = _read_group(file)

    if "schema_version" not in raw:
        raise ValueError(
            f"{path} is not a BLonD beam file: it has no `schema_version` "
            "attribute."
        )
    raw["schema_version"] = int(raw["schema_version"])
    return raw


def _read_group(group: h5py.Group) -> RawBeam:
    """
    Read one HDF5 group into a mapping, recursing into subgroups.

    Parameters
    ----------
    group
        Group to read.

    Returns
    -------
    contents
        The attributes and datasets of the group, plus one nested mapping
        per subgroup.
    """
    contents: RawBeam = dict(group.attrs)
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            contents[name] = _read_group(item)
        else:
            contents[name] = item[()]
    return contents


def _beam_from_raw(raw: RawBeam) -> BeamBaseClass:
    """
    Rebuild a beam from a raw mapping at the current schema version.

    Parameters
    ----------
    raw
        Mapping at `BEAM_SCHEMA_VERSION`.

    Returns
    -------
    beam
        The restored beam.

    Raises
    ------
    ValueError
        If the mapping refers to a beam class unknown to this BLonD version.
    """
    # Imported here because `blond.core.beam.beams` reaches this module
    # through `BeamBaseClass.load`.
    from blond.core.beam.base import BeamBaseClass
    from blond.core.beam.beams import Beam, EmptyBeam, ProbeBeam
    from blond.core.beam.particle_types import ParticleType

    beam_classes = {cls.__name__: cls for cls in (Beam, ProbeBeam, EmptyBeam)}
    class_name = str(raw["blond_class"])
    if class_name not in beam_classes:
        raise ValueError(
            f"The beam file holds a beam of class {class_name!r}, which this "
            f"BLonD version does not know. Known classes: "
            f"{sorted(beam_classes)}."
        )
    beam_class = beam_classes[class_name]

    particle_type = ParticleType(
        mass=float(raw["particle_type"]["mass"]),
        charge=float(raw["particle_type"]["charge"]),
        user_decay_rate=float(raw["particle_type"]["user_decay_rate"]),
    )

    # The subclasses of `Beam` only differ in how their `__init__` builds the
    # particle coordinates; here those come from the file, so only the common
    # base state is initialized.
    beam = beam_class.__new__(beam_class)
    BeamBaseClass.__init__(
        beam,
        intensity=int(raw["intensity"]),
        particle_type=particle_type,
        is_counter_rotating=bool(raw["is_counter_rotating"]),
        is_distributed=False,
    )

    reference = raw["reference"]
    total_energy = (
        float(reference["total_energy"])
        if bool(reference["has_total_energy"])
        else None
    )
    particles = raw["particles"]
    beam.setup_beam(
        dt=backend.array(particles["dt"], dtype=backend.float),
        dE=backend.array(particles["dE"], dtype=backend.float),
        flags=backend.array(particles["flags"], dtype=np.int32),
        ids=backend.array(particles["ids"], dtype=np.int32),
        reference_time=float(reference["time"]),
        reference_total_energy=total_energy,
    )
    return beam
