"""Regenerate the golden beam fixture of the current schema version.

The golden fixtures are frozen beam files, one per released schema version,
that ``TestGoldenFixtures`` loads through the migration chain. They are the
only test that reads a file this code did not just write, and therefore the
only one that can catch a broken migration.

Run this after bumping ``BEAM_SCHEMA_VERSION``::

    python tests/unittests/core/beam/fixtures/generate_beam_fixtures.py

and commit the new ``beam_v<N>.h5``. Never regenerate the fixture of an
older version: it describes files that already exist on other people's
disks, and rewriting it hides exactly the incompatibility it should catch.

The fixture is written with the 64-bit NumPy backend so it is portable and
reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from blond import backend
from blond.core.backends.backend import Numpy64Bit
from blond.core.beam.serialization import BEAM_SCHEMA_VERSION

# The fixture must be built by the very same code the tests compare against.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_serialization import build_reference_beam  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Write the golden fixture of the current schema version."""
    if not isinstance(backend, Numpy64Bit):
        raise SystemExit(
            "The golden beam fixture must be generated with the 64-bit "
            f"NumPy backend, got {type(backend).__name__}. Unset "
            "BLOND_BACKEND_MODE / BLOND_BACKEND_BITS and try again."
        )
    assert backend.float == np.float64, f"{backend.float=}"

    path = FIXTURES_DIR / f"beam_v{BEAM_SCHEMA_VERSION}.h5"
    if path.exists():
        raise SystemExit(
            f"{path.name} already exists. Regenerating the fixture of an "
            "already released schema version would hide incompatibilities; "
            "bump BEAM_SCHEMA_VERSION instead, or delete the file "
            "deliberately if it was never committed."
        )

    build_reference_beam().save(path)
    print(f"Wrote golden beam fixture of schema version {path}")


if __name__ == "__main__":
    main()
