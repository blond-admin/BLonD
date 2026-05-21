"""Regenerate the committed golden ``Beam`` fixture.

The fixture is a *static*, version-controlled ``Beam.save()`` archive
(``golden_beam.npz``) loaded by ``TestBeamGoldenFixture``. Unlike the
in-process round-trip test, it freezes a file on disk so that any
backward-incompatible change to the ``Beam`` save/load pipeline is caught.

Run this script only when you intentionally change the on-disk beam format and
bump ``blond.core.beam.base._BEAM_SCHEMA_VERSION``::

    python tests/unittests/core/beam/fixtures/generate_golden_beam_fixture.py

Then commit the regenerated ``golden_beam.npz`` alongside the schema bump.

Generate with the default 64-bit NumPy backend so the fixture is portable and
bit-for-bit reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from blond import backend
from blond.core.backends.backend import NumpyBackend
from blond.core.beam.base import _BEAM_SCHEMA_VERSION

# Imported from the test module so the fixture is built with the exact same
# construction code the tests use.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_beams import build_reference_beam  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parent / "golden_beam.npz"


def main() -> None:
    """Build the reference beam and write the golden ``.npz`` fixture."""
    if not isinstance(backend, NumpyBackend) or backend.float != np.float64:
        raise SystemExit(
            "Golden beam fixture must be generated with the default 64-bit "
            f"NumPy backend, got {type(backend).__name__} / {backend.float}. "
            "Unset BLOND_BACKEND_MODE / GPU settings and re-run."
        )

    beam = build_reference_beam()
    # ``Beam.save`` appends ``.npz`` automatically; strip it so we don't get
    # ``golden_beam.npz.npz``.
    beam.save(FIXTURE_PATH.with_suffix(""))

    print(
        f"Wrote golden beam fixture at schema version "
        f"{_BEAM_SCHEMA_VERSION}: {FIXTURE_PATH}"
    )


if __name__ == "__main__":
    main()
