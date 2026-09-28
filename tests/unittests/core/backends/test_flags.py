import ast
import unittest
from pathlib import Path

import blond.core.backends
from blond.testing.backend_testing import BLonDTestCase

BACKENDS_DIR = Path(blond.core.backends.__file__).parent


def _imported_modules(source_file: Path) -> set[str]:
    """Return every module name imported anywhere in `source_file`."""
    tree = ast.parse(source_file.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


class TestBeamFlagsLocation(BLonDTestCase):
    def test_beam_flags_values(self):
        from blond.core.backends.flags import BeamFlags

        self.assertEqual(BeamFlags.LOST.value, -500)
        self.assertEqual(BeamFlags.ACTIVE.value, 1)

    def test_beam_module_reexports_same_class(self):
        from blond.core.backends.flags import BeamFlags as backend_flags
        from blond.core.beam.flags import BeamFlags as beam_flags

        self.assertIs(beam_flags, backend_flags)

    def test_backends_do_not_import_core_beam(self):
        """`core.backends` must not depend on `core.beam`.

        `core.beam` imports `core.backends`, so the reverse direction
        would make the two packages mutually dependent.
        """
        for source_file in sorted(BACKENDS_DIR.rglob("*.py")):
            with self.subTest(file=str(source_file.relative_to(BACKENDS_DIR))):
                offending = {
                    module
                    for module in _imported_modules(source_file)
                    if module == "blond.core.beam"
                    or module.startswith("blond.core.beam.")
                }
                self.assertEqual(offending, set())


if __name__ == "__main__":
    unittest.main()
