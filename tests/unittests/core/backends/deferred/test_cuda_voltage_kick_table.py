import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.testing.backend_testing import BLonDTestCase, skip_if_no_cupy


@pytest.mark.cupy
@pytest.mark.backend_mutation
class TestCudaVoltageKickTable(BLonDTestCase):
    def tearDown(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")

    @skip_if_no_cupy
    def test_matches_cpp_table(self) -> None:
        bin_centers = np.linspace(-1e-9, 1e-9, 50)
        voltage = np.sin(np.linspace(0, 3, 50)) * 1e3
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")
        expected = backend.specials._build_voltage_kick_table(
            voltage=voltage,
            bin_centers=bin_centers,
            charge=2.0,
            acceleration_kick=0.5,
        )
        from blond.core.backends.backend import Cupy64Bit

        backend.change_backend(Cupy64Bit)
        table = backend.specials._build_voltage_kick_table(
            voltage=backend.array(voltage),
            bin_centers=backend.array(bin_centers),
            charge=2.0,
            acceleration_kick=0.5,
        )
        np.testing.assert_allclose(copy_to_cpu(table), expected, rtol=1e-13)
