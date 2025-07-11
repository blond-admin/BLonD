import unittest

import numpy as np

from blond3._core.backends.backend import backend, Numpy64Bit

backend.change_backend(Numpy64Bit)
from blond3._core.backends.python.callables import PythonSpecials


class TestPythonSpecials(unittest.TestCase):
    def test_kick_induced_voltage(self):
        kick_induced_voltage = PythonSpecials().kick_induced_voltage

        dt = np.linspace(-50, 50, 20)
        dE = np.zeros_like(dt)
        bin_centers = np.linspace(-40, 40, 20)
        voltage = bin_centers**2
        charge = 10.0
        acceleration_kick = 0
        kick_induced_voltage(
            dt=dt,
            dE=dE,
            voltage=voltage,
            bin_centers=bin_centers,
            charge=charge,
            acceleration_kick=acceleration_kick,
        )
        expected = (
            charge * np.interp(dt[:], bin_centers, voltage, left=0, right=0)
            + acceleration_kick
        )
        np.testing.assert_allclose(dE, expected)
