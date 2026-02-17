import unittest
from enum import Enum
from pathlib import Path

import numpy as np

from blond.handle_results.helpers import callers_relative_path
from blond.physics.impedances.readers import (
    ExampleImpedanceReader2,
    ModesExampleReader2,
)
from blond.physics.impedances.sources import ImpedanceTableFreq


class TestExampleReader2(unittest.TestCase):
    def test_reader(self):
        reader = ExampleImpedanceReader2(mode=ModesExampleReader2.SHORTED)
        freq_table_short = ImpedanceTableFreq.from_file(
            Path(
                callers_relative_path(
                    "../../../blond/examples/resources/EX_02_Finemet.txt",
                    stacklevel=1,
                )
            ),
            reader,
        )
        shorted_imp_freq_y = freq_table_short._freq_y

        reader = ExampleImpedanceReader2(mode=ModesExampleReader2.CLOSED_LOOP)
        freq_table_short = ImpedanceTableFreq.from_file(
            Path(
                callers_relative_path(
                    "../../../blond/examples/resources/EX_02_Finemet.txt",
                    stacklevel=1,
                )
            ),
            reader,
        )
        assert np.sum(shorted_imp_freq_y) != np.sum(freq_table_short._freq_y)

        reader = ExampleImpedanceReader2(mode=ModesExampleReader2.OPEN_LOOP)
        freq_table_open = ImpedanceTableFreq.from_file(
            Path(
                callers_relative_path(
                    "../../../blond/examples/resources/EX_02_Finemet.txt",
                    stacklevel=1,
                )
            ),
            reader,
        )
        assert np.sum(shorted_imp_freq_y) != np.sum(freq_table_open._freq_y)

        with self.assertRaises(NameError):

            class fake_enum(Enum):
                NOT_IN_THE_MODE_TODAY = "not_in_the_mode_today"

            reader = ExampleImpedanceReader2(
                mode=fake_enum.NOT_IN_THE_MODE_TODAY
            )
            _ = (
                ImpedanceTableFreq.from_file(
                    callers_relative_path(
                        "../../../blond/examples/resources/EX_02_Finemet.txt",
                        stacklevel=1,
                    ),
                    reader,
                ),
            )
