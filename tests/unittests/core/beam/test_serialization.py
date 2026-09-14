import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pytest

from blond import Beam, backend, proton
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.beams import EmptyBeam, ProbeBeam
from blond.core.beam.flags import BeamFlags
from blond.core.beam.migrations import UnsupportedSchemaVersionError
from blond.core.beam.serialization import (
    BEAM_SCHEMA_VERSION,
    load_beam,
    save_beam,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Fingerprint of the on-disk layout of every schema version ever released.
#
# The fingerprint covers the names, nesting and dtypes of everything a saved
# beam contains, so renaming a dataset or an attribute, adding or removing
# one, or changing a dtype fails ``TestSchemaFingerprint``. That is
# deliberate: it forces the developer to
#
# 1. bump ``BEAM_SCHEMA_VERSION``,
# 2. register a migration from the previous version,
# 3. add the new fingerprint below (never edit an existing entry - the old
#    entries describe files that already exist on other people's disks),
# 4. regenerate the golden fixture for the new version.
EXPECTED_LAYOUT_FINGERPRINTS = {
    1: "a5665a0cc5b92349cf9a90382b92c02ce1a6c5110eb94093576ee400ea36d323",
}


# Paths whose dtype follows the precision of the active backend instead of
# being fixed by the format. That their precision is right is checked by
# ``test_particle_arrays_are_written_at_backend_precision`` instead.
BACKEND_FLOAT_PATHS = frozenset({"particles/dt", "particles/dE"})


def _dtype_name(path: str, value) -> str:
    """Name the on-disk dtype of one attribute or dataset."""
    if path in BACKEND_FLOAT_PATHS:
        return "backend_float"
    if isinstance(value, (str, bytes)):
        return "str"
    if isinstance(value, h5py.Empty):
        # A null value, e.g. an unset reference energy. HDF5 expresses that
        # in the dataspace, not the datatype, so it is the value that is
        # absent while the layout stays the same.
        return value.dtype.name
    return np.asarray(value).dtype.name


def layout_fingerprint(path: Path) -> str:
    """Hash names, nesting and dtypes of everything in a beam file."""
    lines: list[str] = []

    def walk(group: h5py.Group, prefix: str) -> None:
        for name, value in sorted(group.attrs.items()):
            full = f"{prefix}{name}"
            lines.append(f"{full}|attribute|{_dtype_name(full, value)}")
        for name, item in sorted(group.items()):
            full = f"{prefix}{name}"
            if isinstance(item, h5py.Group):
                walk(item, f"{full}/")
            else:
                lines.append(f"{full}|dataset|{_dtype_name(full, item[()])}")

    with h5py.File(path, "r") as file:
        walk(file, "")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build_reference_beam() -> Beam:
    """Build a deterministic ``Beam`` used across the serialization tests."""
    beam = Beam(
        intensity=1e12,
        particle_type=proton,
        is_counter_rotating=True,
    )
    rng = np.random.default_rng(0)
    dt = rng.normal(0.0, 1e-10, 64)
    dE = rng.normal(0.0, 1e6, 64)
    flags = np.full(64, BeamFlags.ACTIVE.value, dtype=np.int32)
    flags[:3] = BeamFlags.LOST.value
    beam.setup_beam(
        dt=backend.array(dt, dtype=backend.float),
        dE=backend.array(dE, dtype=backend.float),
        flags=backend.array(flags, dtype=np.int32),
        reference_time=1.5,
        reference_total_energy=450e9,
    )
    return beam


def assert_beams_equal(
    expected: Beam, restored: Beam, rtol: float = 0.0
) -> None:
    """Assert that two beams carry the same state."""
    assert type(expected) is type(restored)
    assert expected.intensity == restored.intensity
    assert expected.is_counter_rotating == restored.is_counter_rotating
    assert expected.is_distributed == restored.is_distributed
    assert expected.particle_type == restored.particle_type
    assert expected.reference.time == restored.reference.time
    assert expected.reference._total_energy == restored.reference._total_energy
    for name in ("dt", "dE", "flags", "ids"):
        np.testing.assert_allclose(
            copy_to_cpu(getattr(expected, f"read_partial_{name}")()),
            copy_to_cpu(getattr(restored, f"read_partial_{name}")()),
            rtol=rtol,
            err_msg=f"mismatch in {name}",
        )


class TestBeamDict(unittest.TestCase):
    """``to_dict`` / ``from_dict``, the single definition of beam state."""

    def test_roundtrip_through_a_dict(self):
        original = build_reference_beam()
        restored = BeamBaseClass.from_dict(original.to_dict())
        assert_beams_equal(original, restored)

    def test_dict_identifies_class_and_schema_version(self):
        dct = build_reference_beam().to_dict()
        self.assertEqual(dct["__class__"], "Beam")
        self.assertEqual(dct["schema_version"], BEAM_SCHEMA_VERSION)

    def test_dict_holds_host_arrays(self):
        """The arrays must be usable without the backend that made them."""
        dct = build_reference_beam().to_dict()
        for name in ("dt", "dE", "flags", "ids"):
            self.assertIsInstance(dct["particles"][name], np.ndarray)

    def test_dict_represents_a_missing_total_energy_as_none(self):
        beam = Beam(intensity=1e10, particle_type=proton)
        beam.setup_beam(
            dt=backend.array([1.0], dtype=backend.float),
            dE=backend.array([2.0], dtype=backend.float),
        )
        self.assertIsNone(beam.to_dict()["reference"]["total_energy"])

    def test_dict_of_a_subclass_roundtrips_as_that_subclass(self):
        original = ProbeBeam(particle_type=proton, dt=np.array([1e-9]))
        restored = BeamBaseClass.from_dict(original.to_dict())
        assert_beams_equal(original, restored)

    def test_from_dict_rejects_an_unknown_class(self):
        dct = build_reference_beam().to_dict()
        dct["__class__"] = "SomeFutureBeam"
        with pytest.raises(ValueError, match="SomeFutureBeam"):
            BeamBaseClass.from_dict(dct)

    def test_to_dict_of_a_beam_that_is_not_set_up_raises(self):
        beam = Beam(intensity=1e10, particle_type=proton)
        with pytest.raises(ValueError, match="not set up"):
            beam.to_dict()

    def test_to_dict_of_a_distributed_beam_raises(self):
        beam = build_reference_beam()
        beam._is_distributed = True
        with pytest.raises(NotImplementedError, match="distributed"):
            beam.to_dict()


class TestSaveLoadRoundtrip(unittest.TestCase):
    """``save_beam`` followed by ``load_beam`` must preserve beam state."""

    def test_roundtrip_preserves_state(self):
        original = build_reference_beam()
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "beam.h5"
            save_beam(original, path)
            restored = load_beam(path)
        assert_beams_equal(original, restored)

    def test_beam_method_roundtrip(self):
        original = build_reference_beam()
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "beam.h5"
            original.save(path)
            restored = Beam.load(path)
        assert_beams_equal(original, restored)

    def test_roundtrip_without_reference_total_energy(self):
        beam = Beam(intensity=1e10, particle_type=proton)
        beam.setup_beam(
            dt=backend.array([1.0, 2.0], dtype=backend.float),
            dE=backend.array([3.0, 4.0], dtype=backend.float),
        )
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "beam.h5"
            beam.save(path)
            restored = Beam.load(path)
        assert restored.reference._total_energy is None
        assert_beams_equal(beam, restored)

    def test_roundtrip_of_beam_subclasses(self):
        beams = (
            ProbeBeam(particle_type=proton, dt=np.array([1e-9, 2e-9])),
            EmptyBeam(particle_type=proton, reference_total_energy=26e9),
        )
        for original in beams:
            with self.subTest(beam=type(original).__name__):
                with TemporaryDirectory() as tmp_dir:
                    path = Path(tmp_dir) / "beam.h5"
                    original.save(path)
                    restored = Beam.load(path)
                assert_beams_equal(original, restored)

    def test_ids_survive_the_roundtrip(self):
        """Ids are not renumbered on load, they are read from the file."""
        original = build_reference_beam()
        original._ids.array_local[:] = original._ids.array_local[::-1]
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "beam.h5"
            original.save(path)
            restored = Beam.load(path)
        np.testing.assert_array_equal(
            copy_to_cpu(original.read_partial_ids()),
            copy_to_cpu(restored.read_partial_ids()),
        )

    def test_written_file_carries_current_schema_version(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "beam.h5"
            build_reference_beam().save(path)
            with h5py.File(path, "r") as file:
                self.assertEqual(
                    file.attrs["schema_version"], BEAM_SCHEMA_VERSION
                )
                self.assertEqual(file.attrs["__class__"], "Beam")


class TestSaveLoadErrors(unittest.TestCase):
    """Loudly reject what cannot be (de)serialized correctly."""

    def test_saving_a_beam_that_is_not_set_up_raises(self):
        beam = Beam(intensity=1e10, particle_type=proton)
        with TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="not set up"):
                beam.save(Path(tmp_dir) / "beam.h5")

    def test_saving_a_distributed_beam_raises(self):
        beam = build_reference_beam()
        beam._is_distributed = True
        with TemporaryDirectory() as tmp_dir:
            with pytest.raises(NotImplementedError, match="distributed"):
                beam.save(Path(tmp_dir) / "beam.h5")

    def test_loading_a_newer_schema_version_raises(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "beam.h5"
            build_reference_beam().save(path)
            with h5py.File(path, "r+") as file:
                file.attrs["schema_version"] = BEAM_SCHEMA_VERSION + 1
            with pytest.raises(UnsupportedSchemaVersionError, match="newer"):
                load_beam(path)

    def test_loading_an_unknown_beam_class_raises(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "beam.h5"
            build_reference_beam().save(path)
            with h5py.File(path, "r+") as file:
                file.attrs["__class__"] = "SomeFutureBeam"
            with pytest.raises(ValueError, match="SomeFutureBeam"):
                load_beam(path)

    def test_loading_a_file_without_schema_version_raises(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "not_a_beam.h5"
            with h5py.File(path, "w") as file:
                file.create_dataset("something", data=[1, 2, 3])
            with pytest.raises(ValueError, match="not a BLonD beam file"):
                load_beam(path)


class TestSchemaFingerprint(unittest.TestCase):
    """Force a schema-version bump whenever the on-disk layout changes."""

    def test_fingerprint_of_written_file_is_unchanged(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "beam.h5"
            build_reference_beam().save(path)
            fingerprint = layout_fingerprint(path)

        to_record = f'    {BEAM_SCHEMA_VERSION}: "{fingerprint}",'
        expected = EXPECTED_LAYOUT_FINGERPRINTS.get(BEAM_SCHEMA_VERSION)
        if expected is None:
            self.fail(
                f"No fingerprint recorded for schema version "
                f"{BEAM_SCHEMA_VERSION}. Add it to "
                f"EXPECTED_LAYOUT_FINGERPRINTS in this file:\n{to_record}"
            )
        self.assertEqual(
            fingerprint,
            expected,
            "The on-disk beam layout changed (a name, a dtype or the "
            "structure). Files written by older BLonD versions can no "
            "longer be read as-is, so you must:\n"
            "  1. bump BEAM_SCHEMA_VERSION in "
            "blond/core/beam/serialization.py,\n"
            "  2. register a migration in blond/core/beam/migrations.py,\n"
            "  3. record the new fingerprint below the existing entries "
            f"(do not edit those):\n{to_record}\n"
            "  4. regenerate the golden fixture with "
            "tests/unittests/core/beam/fixtures/generate_beam_fixtures.py.",
        )

    def test_particle_arrays_are_written_at_backend_precision(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "beam.h5"
            build_reference_beam().save(path)
            with h5py.File(path, "r") as file:
                for name in ("dt", "dE"):
                    self.assertEqual(
                        file["particles"][name].dtype,
                        backend.float,
                        f"{name} must be written at backend precision",
                    )
                for name in ("flags", "ids"):
                    self.assertEqual(file["particles"][name].dtype, np.int32)

    def test_layout_does_not_depend_on_an_unset_total_energy(self):
        """A null value must not change the shape of the file.

        Writing ``None`` as an absent attribute instead of an empty one
        would make the layout, and hence the fingerprint, depend on the
        data rather than on the format.
        """
        beam = build_reference_beam()
        beam.reference._total_energy = None
        with TemporaryDirectory() as tmp_dir:
            without = Path(tmp_dir) / "without.h5"
            beam.save(without)
            fingerprint_without = layout_fingerprint(without)

        self.assertEqual(
            fingerprint_without,
            EXPECTED_LAYOUT_FINGERPRINTS[BEAM_SCHEMA_VERSION],
        )

    def test_no_fingerprint_is_recorded_for_a_future_version(self):
        self.assertLessEqual(
            max(EXPECTED_LAYOUT_FINGERPRINTS),
            BEAM_SCHEMA_VERSION,
            "A fingerprint is recorded for a schema version that "
            "BEAM_SCHEMA_VERSION does not know about.",
        )


class TestGoldenFixtures(unittest.TestCase):
    """Every historic on-disk version must still load through migrations."""

    def test_a_golden_fixture_exists_for_every_released_version(self):
        found = {
            int(path.stem.removeprefix("beam_v"))
            for path in FIXTURES_DIR.glob("beam_v*.h5")
        }
        self.assertEqual(
            found,
            set(range(1, BEAM_SCHEMA_VERSION + 1)),
            "Missing golden beam fixture(s). Regenerate with "
            "tests/unittests/core/beam/fixtures/generate_beam_fixtures.py.",
        )

    def test_golden_fixtures_load_into_the_reference_beam(self):
        expected = build_reference_beam()
        for path in sorted(FIXTURES_DIR.glob("beam_v*.h5")):
            with self.subTest(fixture=path.name):
                # Fixtures are written at 64 bit; a 32 bit backend loses
                # precision on load, hence the tolerance.
                assert_beams_equal(expected, load_beam(path), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
