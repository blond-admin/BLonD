# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import ast
import unittest
from pathlib import Path

import blond
from blond.testing.backend_testing import BLonDTestCase

EXAMPLES_DIR = Path(blond.__file__).parent / "examples"

# Submodules that examples may import from directly, because their
# content is deliberately not part of the public API.
NON_PUBLIC_PREFIXES = ("blond.experimental", "blond.testing")


class TestPublicApi(BLonDTestCase):
    """The names a user script needs should come from ``blond`` itself."""

    def test_every_name_in_all_is_importable(self):
        missing = [name for name in blond.__all__ if not hasattr(blond, name)]
        self.assertEqual(missing, [])

    def test_user_facing_objects_are_exported(self):
        expected = {
            "Coasting",
            "ContinuousMultiTurnTimeDomainSolver",
            "DriftExact",
            "DynamicProfileConstCutoff",
            "FilamentationMatcher",
            "ImpedanceTableTime",
            "MultiPassResonatorSolver",
            "ParticleType",
            "ScheduledArray",
            "ScheduledFunctional",
            "ScheduledInterpolation",
            "SingleTurnResonatorConvolutionSolver",
            "StaticMultiProfileObservation",
            "SynchrotronRadiationMaster",
            "TravelingWaveCavity",
            "VariNoise",
        }
        self.assertEqual(sorted(expected - set(blond.__all__)), [])

    def test_examples_import_public_names_from_facade(self):
        public_names = set(blond.__all__)
        offenders = []
        for path in sorted(EXAMPLES_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if not node.module.startswith("blond.") or (
                    node.module.startswith(NON_PUBLIC_PREFIXES)
                ):
                    continue
                offenders += [
                    f"{path.name}: from {node.module} import {alias.name}"
                    for alias in node.names
                    if alias.name in public_names
                ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
