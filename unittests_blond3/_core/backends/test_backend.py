import unittest

import numpy as np

from blond3._core.backends.backend import (
    Numpy64Bit,
    Numpy32Bit,
    Cupy32Bit,
    Cupy64Bit,
    CupyBackend,
    NumpyBackend,
    backend,
)


class TestBackendBaseClass(unittest.TestCase):
    def setUp(self):
        self.backend_base_class = Numpy32Bit()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_change_backend(self):
        self.backend_base_class.change_backend(new_backend=Numpy64Bit)
        self.assertEqual(self.backend_base_class.float, np.float64)
        self.assertEqual(self.backend_base_class.int, np.int64)
        self.assertEqual(self.backend_base_class.complex, np.complex128)

    def test_set_specials(self):
        self.backend_base_class.set_specials(mode="numba")


@unittest.skip
class TestCupy32Bit(unittest.TestCase):
    def setUp(self):
        self.cupy32_bit = Cupy32Bit()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp


@unittest.skip
class TestCupy64Bit(unittest.TestCase):
    def setUp(self):
        self.cupy64_bit = Cupy64Bit()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp


@unittest.skip
class TestCupyBackend(unittest.TestCase):
    def setUp(self):
        self.cupy_backend = CupyBackend(float_=np.float32, int_=np.float64)

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_set_specials(self):
        self.cupy_backend.set_specials(mode="cuda")


class TestNumpy64Bit(unittest.TestCase):
    def setUp(self):
        self.numpy64_bit = Numpy64Bit()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp


class TestNumpyBackend(unittest.TestCase):
    def setUp(self):
        self.numpy_backend = NumpyBackend(
            float_=np.float32, int_=np.int32, complex_=np.complex64
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_set_specials(self):
        self.numpy_backend.set_specials(mode="python")


class TestSpecials(unittest.TestCase):
    def setUp(self):
        self.special_modes = (
            "python",
            "cpp",
            "numba",
            # "cuda", # todo implement
            "fortran"
        )

    def _setUp(self, dtype, special_mode):
        if special_mode in (
            "python",
            "cpp",
            "numba",
            "fortran",
        ):
            if dtype == np.float32:
                backend.change_backend(Numpy32Bit)
            else:
                backend.change_backend(Numpy64Bit)
        elif special_mode in ("cuda",):
            if dtype == np.float32:
                backend.change_backend(Cupy32Bit)
            else:
                backend.change_backend(Cupy64Bit)
        else:
            raise ValueError(special_mode)

        backend.set_specials(special_mode)

        self.dt = backend.linspace(1e-9, 10e-9, 10, dtype=backend.float)
        self.dE = backend.linspace(1e9, 10e9, 10, dtype=backend.float)
        self.t_rev = backend.float(10)
        self.length_ratio = backend.float(0.5)
        self.alpha_0 = backend.float(1.0)
        self.alpha_1 = backend.float(1.0)
        self.alpha_2 = backend.float(1.0)
        self.beta = backend.float(0.9)
        self.energy = backend.float(10)
        self.alpha_order = backend.int(0.3)
        self.eta_0 = backend.float(0.3)
        self.eta_1 = backend.float(0.3)
        self.eta_2 = backend.float(0.3)
        self.voltage_single_harmonic = backend.float(1e3)
        self.omega_rf_single_harmonic = backend.float(2 * np.pi * 400e3)
        self.phi_rf_single_harmonic = backend.float(0.3)

        self.voltages = backend.linspace(1e6, 5e6, 3, dtype=backend.float)
        self.omegas = backend.linspace(200e6, 400e6, 3, dtype=backend.float)
        self.phis = backend.linspace(0, 2*np.pi, 3, dtype=backend.float)

        self.charge = backend.float(1)
        self.acceleration_kick = backend.float(-1)
        if backend.float == np.float32:
            self.rtol = 1e-6
        elif backend.float == np.float64:
            self.rtol = 1e-12
        else:
            raise ValueError(backend.float)

    def test___init__(self):
        pass
    @unittest.skip
    def test_drift_exact(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                self._setUp(dtype=dtype, special_mode=special)
                backend.specials.drift_exact(
                    dt=self.dt,
                    dE=self.dE,
                    t_rev=self.t_rev,
                    length_ratio=self.length_ratio,
                    alpha_0=self.alpha_0,
                    alpha_1=self.alpha_1,
                    alpha_2=self.alpha_2,
                    beta=self.beta,
                    energy=self.energy,
                )
                result = self.dt
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(result, result_python, rtol=self.rtol)
    @unittest.skip
    def test_drift_legacy(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                self._setUp(dtype=dtype, special_mode=special)
                backend.specials.drift_legacy(
                    dt=self.dt,
                    dE=self.dE,
                    T=self.t_rev * self.length_ratio,
                    alpha_order=self.alpha_order,
                    eta_0=self.eta_0,
                    eta_1=self.eta_1,
                    eta_2=self.eta_2,
                    beta=self.beta,
                    energy=self.energy,
                )
                result = self.dt

                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(result, result_python, rtol=self.rtol)

    def test_drift_simple(self):
        for dtype in (np.float64,): #  (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                self._setUp(dtype=dtype, special_mode=special)
                backend.specials.drift_simple(
                    dt=self.dt,
                    dE=self.dE,
                    T=self.t_rev * self.length_ratio,
                    eta_0=self.eta_0,
                    beta=self.beta,
                    energy=self.energy,
                )
                result = self.dt
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(result, result_python, rtol=self.rtol)

    def test_kick_multi_harmonic(self):
        for dtype in (np.float64,): #  (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                self._setUp(dtype=dtype, special_mode=special)
                backend.specials.kick_multi_harmonic(
                    dt=self.dt,
                    dE=self.dE,
                    voltage=self.voltages,
                    omega_rf=self.omegas,
                    phi_rf=self.phis,
                    charge=self.charge,
                    n_rf=len(self.voltages),
                    acceleration_kick=self.acceleration_kick,
                )
                result = self.dE
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(result, result_python, rtol=self.rtol)

    def test_kick_single_harmonic(self):
        for dtype in (np.float64,): # (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                self._setUp(dtype=dtype, special_mode=special)
                backend.specials.kick_single_harmonic(
                    dt=self.dt,
                    dE=self.dE,
                    voltage=self.voltage_single_harmonic,
                    omega_rf=self.omega_rf_single_harmonic,
                    phi_rf=self.phi_rf_single_harmonic,
                    charge=self.charge,
                    acceleration_kick=self.acceleration_kick,
                )
                result = self.dE
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(result, result_python, rtol=self.rtol)

    @unittest.skip
    def test_loss_box(self):
        # TODO: implement test for `loss_box`
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                self._setUp(dtype=dtype, special_mode=special)
                backend.specials.loss_box(a=None, b=None, c=None, d=None)
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(result, result_python, rtol=self.rtol)


if __name__ == "__main__":
    unittest.main()
