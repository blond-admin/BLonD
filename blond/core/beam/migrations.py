# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Migrations between on-disk schema versions of saved beams.

A beam file written by an older BLonD version is upgraded to the current
layout by chaining single-step migrations, each of which converts the raw
mapping read from the file from version ``N`` to version ``N + 1``.

To add a migration after changing the on-disk layout:

1. Bump ``BEAM_SCHEMA_VERSION`` in `blond.core.beam.serialization`.
2. Register a function here that rewrites the raw mapping of the previous
   version into the new one::

       @register_migration(from_version=1)
       def _v1_to_v2(raw: dict) -> dict:
           raw["particles"]["weights"] = np.ones_like(raw["particles"]["dt"])
           raw["schema_version"] = 2
           return raw

3. Follow the instructions in
   ``tests/unittests/core/beam/test_serialization.py`` to record the new
   fingerprint and regenerate the golden fixtures.

A migration receives a *copy* of the raw mapping and may mutate it freely.
It must set ``schema_version`` to the version it produces.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from typing import Any

    RawBeam = dict[str, Any]
    Migration = Callable[[RawBeam], RawBeam]


class UnsupportedSchemaVersionError(ValueError):
    """The schema version of a beam file cannot be handled."""


class MissingMigrationError(ValueError):
    """No migration is registered for a schema version that needs one."""


#: Registry of the migrations shipped with this BLonD version, mapping the
#: source version ``N`` to the function producing version ``N + 1``.
MIGRATIONS: dict[int, Migration] = {}


def register_migration(from_version: int) -> Callable[[Migration], Migration]:
    """
    Register a ``from_version -> from_version + 1`` beam migration.

    Parameters
    ----------
    from_version
        Schema version the decorated function accepts.

    Returns
    -------
    decorator
        Decorator registering the migration in `MIGRATIONS`.

    Raises
    ------
    ValueError
        If a migration is already registered for ``from_version``.
    """

    def decorator(migration: Migration) -> Migration:
        if from_version in MIGRATIONS:
            raise ValueError(
                f"A beam migration from version {from_version} is already "
                f"registered: {MIGRATIONS[from_version]!r}."
            )
        MIGRATIONS[from_version] = migration
        return migration

    return decorator


def migrate_raw_beam(
    raw: RawBeam,
    target_version: int,
    registry: dict[int, Migration] | None = None,
) -> RawBeam:
    """
    Upgrade a raw beam mapping to ``target_version``.

    Parameters
    ----------
    raw
        Mapping as read from a beam file. It is not modified; migrations
        operate on a deep copy.
    target_version
        Schema version to migrate to, usually
        ``blond.core.beam.serialization.BEAM_SCHEMA_VERSION``.
    registry
        Migrations to use. Defaults to the shipped `MIGRATIONS`; passing an
        explicit registry is mainly useful for testing.

    Returns
    -------
    raw
        The mapping at ``target_version``.

    Raises
    ------
    UnsupportedSchemaVersionError
        If the file was written by a newer BLonD version, or if its schema
        version is not a positive integer.
    MissingMigrationError
        If no migration is registered for an intermediate version.
    RuntimeError
        If a migration does not advance ``schema_version``.
    """
    if registry is None:
        registry = MIGRATIONS

    version = raw["schema_version"]
    if version < 1:
        raise UnsupportedSchemaVersionError(
            f"Invalid beam schema version {version!r}; versions start at 1."
        )
    if version > target_version:
        raise UnsupportedSchemaVersionError(
            f"The beam file uses schema version {version}, which is newer "
            f"than the version {target_version} supported by this BLonD "
            "installation. Please update BLonD to read it."
        )

    raw = copy.deepcopy(raw)
    while version < target_version:
        try:
            migration = registry[version]
        except KeyError:
            raise MissingMigrationError(
                f"No migration registered from beam schema version "
                f"{version} to {version + 1}. Register one in "
                "blond/core/beam/migrations.py."
            ) from None

        raw = migration(raw)
        new_version = raw["schema_version"]
        if new_version <= version:
            raise RuntimeError(
                f"The beam migration {migration!r} did not advance "
                f"schema_version beyond {version}; it returned "
                f"{new_version!r}."
            )
        version = new_version

    return raw
