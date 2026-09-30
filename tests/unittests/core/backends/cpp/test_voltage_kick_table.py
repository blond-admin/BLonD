import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.testing.backend_testing import BLonDTestCase


@pytest.mark.backend_mutation
class TestVoltageKickTable(BLonDTestCase):
    def setUp(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

    def tearDown(self) -> None:
        backend.set_specials("python")

    def test_table_layout(self) -> None:
        bin_centers = np.linspace(-1e-9, 1e-9, 5)
        voltage = np.array([1.0, 3.0, 2.0, -1.0, 0.5])
        charge, acceleration_kick = 2.0, 0.25
        table = backend.specials._build_voltage_kick_table(
            voltage=voltage,
            bin_centers=bin_centers,
            charge=charge,
            acceleration_kick=acceleration_kick,
        )
        inverse_bin_width = 4 / (bin_centers[-1] - bin_centers[0])
        slope = charge * np.diff(voltage) * inverse_bin_width
        offset = charge * voltage[:-1] - bin_centers[:-1] * slope
        offset += acceleration_kick
        expected = np.concatenate(
            [
                [bin_centers[0], inverse_bin_width],
                np.column_stack([slope, offset]).ravel(),
            ]
        )
        np.testing.assert_allclose(table, expected, rtol=1e-14)

    def test_multi_harmonic_beyond_32_matches_python(self) -> None:
        # The eager cpp kick now packs 32 harmonics per Args; more must
        # still be applied, with the acceleration kick once.
        rng = np.random.default_rng(0)
        n_rf = 40
        voltage = rng.uniform(1e3, 2e3, n_rf)
        omega_rf = rng.uniform(1e7, 3e7, n_rf)
        phi_rf = rng.uniform(0, 1, n_rf)
        dt = rng.uniform(-1e-8, 1e-8, 1000)
        dE_cpp = np.zeros(1000)
        backend.specials.kick_multi_harmonic(
            dt=dt,
            dE=dE_cpp,
            voltage=voltage,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            charge=1.0,
            n_rf=n_rf,
            acceleration_kick=5.0,
        )
        expected = (
            np.sum(voltage * np.sin(omega_rf * dt[:, None] + phi_rf), axis=1)
            + 5.0
        )
        np.testing.assert_allclose(dE_cpp, expected, rtol=1e-9, atol=1e-6)

    def test_eager_kick_interpolated_spans_threads(self) -> None:
        # The eager kernel builds the table and applies it in one OpenMP
        # region, so every thread must see the whole table before reading
        # it. A beam of many granules and a table of many bins give each
        # thread both table rows and particles to work on; n_slices=2 is
        # the smallest table (one pair, built by a single thread).
        rng = np.random.default_rng(1)
        n_macroparticles = 200_003
        for n_slices in (2, 3, 1000):
            with self.subTest(n_slices=n_slices):
                bin_centers = np.linspace(-1e-9, 1e-9, n_slices)
                voltage = rng.uniform(-1e6, 1e6, n_slices)
                dt = rng.uniform(-1.5e-9, 1.5e-9, n_macroparticles)
                dE = rng.normal(0, 1e6, n_macroparticles)
                charge, acceleration_kick = 2.0, 0.25
                expected = dE + acceleration_kick
                in_range = (dt >= bin_centers[0]) & (dt < bin_centers[-1])
                expected[in_range] += charge * np.interp(
                    dt[in_range], bin_centers, voltage
                )
                backend.specials.kick_interpolated(
                    dt=dt,
                    dE=dE,
                    voltage=voltage,
                    bin_centers=bin_centers,
                    charge=charge,
                    acceleration_kick=acceleration_kick,
                )
                np.testing.assert_allclose(dE, expected, rtol=0, atol=1e-6)
