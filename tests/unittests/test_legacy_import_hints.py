# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import importlib
import importlib.machinery
import subprocess
import sys
import unittest
from unittest import mock

import blond
from blond.legacy import DynamicLegacyFinder, legacy_getattr


class TestDynamicLegacyFinder(unittest.TestCase):
    """Importing a BLonD 2 module path that no longer exists in BLonD 3
    must fail with a hint pointing to ``blond.legacy.blond2``.
    """

    def test_moved_module_raises_import_error_with_hint(self):
        with self.assertRaises(ImportError) as context:
            importlib.import_module("blond.beam")
        self.assertIn("blond.legacy.blond2.beam", str(context.exception))

    def test_moved_submodule_raises_import_error_with_hint(self):
        with self.assertRaises(ImportError) as context:
            importlib.import_module("blond.beam.beam")
        self.assertIn("blond.legacy.blond2.beam", str(context.exception))

    def test_unknown_module_still_raises_module_not_found(self):
        with self.assertRaises(ModuleNotFoundError) as context:
            importlib.import_module("blond.this_module_does_not_exist")
        self.assertNotIn("legacy", str(context.exception))

    def test_existing_blond3_module_is_unaffected(self):
        module = importlib.import_module("blond.core.beam")
        self.assertEqual(module.__name__, "blond.core.beam")

    def test_legacy_module_itself_is_unaffected(self):
        module = importlib.import_module("blond.legacy.blond2.beam.beam")
        self.assertTrue(hasattr(module, "Beam"))

    def test_interfaces_subtree_is_never_redirected(self):
        # blond.interfaces exists in both versions; too rare to handle.
        finder = DynamicLegacyFinder()
        spec = finder.find_spec("blond.interfaces.rf_noise_cpp.wrap_rf_noise")
        self.assertIsNone(spec)


class TestLegacyGetattr(unittest.TestCase):
    """Accessing a BLonD 2 name on a BLonD 3 module that still exists
    must fail with a hint pointing to the module in ``blond.legacy.blond2``.
    """

    def test_missing_top_level_name_raises_import_error_with_hint(self):
        with self.assertRaises(ImportError) as context:
            getattr(blond, "bigaussian")
        self.assertIn(
            "from blond.legacy.blond2.beam.distributions import bigaussian",
            str(context.exception),
        )

    def test_from_import_of_missing_top_level_name_raises_import_error(self):
        with self.assertRaises(ImportError) as context:
            exec("from blond import bigaussian", {})
        self.assertIn("blond.legacy.blond2", str(context.exception))

    def test_unknown_name_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            getattr(blond, "this_name_does_not_exist_anywhere")

    def test_private_names_raise_attribute_error_without_lookup(self):
        with self.assertRaises(AttributeError):
            getattr(blond, "__wrapped__")

    def test_existing_top_level_name_is_unaffected(self):
        self.assertTrue(hasattr(blond, "Beam"))

    def test_legacy_getattr_never_redirects_interfaces(self):
        # BlondElement exists in blond.legacy.blond2.interfaces.xsuite.
        with self.assertRaises(AttributeError):
            legacy_getattr("blond.interfaces", "BlondElement")

    def test_legacy_getattr_on_module_without_legacy_twin(self):
        with self.assertRaises(AttributeError):
            legacy_getattr("blond.core.beam", "bigaussian")


class TestHooksAreInertForBlond3(unittest.TestCase):
    """The hooks must never act on ordinary BLonD 3 imports."""

    def test_exported_names_never_call_the_legacy_hook(self):
        with mock.patch.object(blond, "legacy_getattr") as hook:
            for name in blond.__all__:
                getattr(blond, name)
        hook.assert_not_called()

    def test_finder_runs_after_the_standard_path_finder(self):
        # PathFinder is registered as a class, ours as an instance.
        legacy_positions = [
            index
            for index, finder in enumerate(sys.meta_path)
            if isinstance(finder, DynamicLegacyFinder)
        ]
        self.assertEqual(len(legacy_positions), 1)
        self.assertGreater(
            legacy_positions[0],
            sys.meta_path.index(importlib.machinery.PathFinder),
        )

    def test_importing_blond_does_not_import_blond2(self):
        # Fresh interpreter: other tests may already have imported v2.
        code = (
            "import sys, blond\n"
            "from blond import *\n"
            "print('blond.legacy.blond2' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        # BLonD prints a backend banner when BLOND_BACKEND_MODE is set,
        # so only the last stdout line carries the answer.
        last_line = result.stdout.strip().splitlines()[-1]
        self.assertEqual(last_line, "False")
