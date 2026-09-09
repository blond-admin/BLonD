import unittest

import pytest

from blond.core.beam.migrations import (
    MIGRATIONS,
    MissingMigrationError,
    UnsupportedSchemaVersionError,
    migrate_raw_beam,
)
from blond.core.beam.serialization import BEAM_SCHEMA_VERSION


def _dummy_raw(schema_version: int) -> dict:
    """Minimal raw beam mapping, as read back from an HDF5 file."""
    return {"schema_version": schema_version, "payload": []}


class TestMigrationChain(unittest.TestCase):
    """Chaining of ``v(N) -> v(N+1)`` migrations on the raw mapping."""

    def test_applies_migrations_in_order(self):
        def one_to_two(raw: dict) -> dict:
            raw = dict(raw)
            raw["payload"] = [*raw["payload"], "1->2"]
            raw["schema_version"] = 2
            return raw

        def two_to_three(raw: dict) -> dict:
            raw = dict(raw)
            raw["payload"] = [*raw["payload"], "2->3"]
            raw["schema_version"] = 3
            return raw

        registry = {1: one_to_two, 2: two_to_three}

        migrated = migrate_raw_beam(
            _dummy_raw(1), target_version=3, registry=registry
        )

        self.assertEqual(migrated["payload"], ["1->2", "2->3"])
        self.assertEqual(migrated["schema_version"], 3)

    def test_does_not_mutate_input(self):
        def one_to_two(raw: dict) -> dict:
            raw = dict(raw)
            raw["payload"] = [*raw["payload"], "1->2"]
            raw["schema_version"] = 2
            return raw

        raw = _dummy_raw(1)
        migrate_raw_beam(raw, target_version=2, registry={1: one_to_two})

        self.assertEqual(raw, _dummy_raw(1))

    def test_current_version_is_returned_unchanged(self):
        raw = _dummy_raw(3)

        migrated = migrate_raw_beam(raw, target_version=3, registry={})

        self.assertEqual(migrated, raw)

    def test_missing_migration_raises(self):
        def one_to_two(raw: dict) -> dict:
            raw = dict(raw)
            raw["schema_version"] = 2
            return raw

        with pytest.raises(MissingMigrationError, match="version 2 to 3"):
            migrate_raw_beam(
                _dummy_raw(1), target_version=3, registry={1: one_to_two}
            )

    def test_file_newer_than_code_raises(self):
        with pytest.raises(UnsupportedSchemaVersionError, match="newer"):
            migrate_raw_beam(_dummy_raw(5), target_version=3, registry={})

    def test_migration_that_does_not_bump_version_raises(self):
        def broken(raw: dict) -> dict:
            return dict(raw)  # forgot to set schema_version

        with pytest.raises(RuntimeError, match="schema_version"):
            migrate_raw_beam(
                _dummy_raw(1), target_version=2, registry={1: broken}
            )

    def test_unknown_version_raises(self):
        with pytest.raises(UnsupportedSchemaVersionError):
            migrate_raw_beam(
                _dummy_raw(0), target_version=3, registry={1: dict}
            )


class TestShippedMigrationRegistry(unittest.TestCase):
    """The registry that is actually used by ``load_beam``."""

    def test_registry_is_contiguous_up_to_current_version(self):
        """Every version below the current one must be migratable."""
        expected = set(range(1, BEAM_SCHEMA_VERSION))
        self.assertEqual(
            set(MIGRATIONS),
            expected,
            "A schema version was bumped without registering the "
            "corresponding migration in blond/core/beam/migrations.py.",
        )


if __name__ == "__main__":
    unittest.main()
