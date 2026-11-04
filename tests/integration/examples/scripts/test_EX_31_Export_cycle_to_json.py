import contextlib
import io
import unittest

import pytest

from blond.testing.backend_testing import BLonDTestCase

pytest.importorskip("pydantic")


class TestEX_31_Export_cycle_to_json(BLonDTestCase):
    def test_printed_json_is_a_valid_cycle(self):
        from blond.examples.scripts import EX_31_Export_cycle_to_json
        from blond.interfaces.portable import CycleDefinition

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            EX_31_Export_cycle_to_json.main()

        cycle = CycleDefinition.model_validate_json(stdout.getvalue())
        self.assertEqual(cycle.name, "LHC")


if __name__ == "__main__":
    unittest.main()
