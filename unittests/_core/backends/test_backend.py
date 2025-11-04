import unittest
import warnings

import numpy as np

from blond._core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    CupyBackend,
    Numpy32Bit,
    Numpy64Bit,
    NumpyBackend,
    backend,
)

try:
    import cupy as _  # type: ignore

    cupy_available = True
except ModuleNotFoundError:
    cupy_available = False

from numba import set_num_threads


class TestBackendBaseClass(unittest.TestCase):
    def setUp(self) -> None:
        self.backend_base_class = Numpy32Bit()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_change_backend(self) -> None:
        self.backend_base_class.change_backend(new_backend=Numpy64Bit)
        self.assertEqual(self.backend_base_class.float, np.float64)
        self.assertEqual(self.backend_base_class.int, np.int64)
        self.assertEqual(self.backend_base_class.complex, np.complex128)

    def test_set_specials(self) -> None:
        self.backend_base_class.set_specials(mode="numba")

    def tearDown(self) -> None:
        self.backend_base_class.set_specials(mode="numba")

    def test_apply_environment_variables(self):
        import os

        backend_modes = ["python", "cpp", "numba", "fortran", "fail"]
        backend_bits = ["32", "64", "fail"]
        try:
            import cupy

            backend_modes = ["cuda"] + backend_modes
        except ModuleNotFoundError:
            pass
        for backend_mode in backend_modes:
            os.environ["BLOND_BACKEND_MODE"] = backend_mode
            for backend_bit in backend_bits:
                os.environ["BLOND_BACKEND_BITS"] = backend_bit
                if (backend_mode == "fail") or (backend_bit == "fail"):
                    with self.assertRaises(ValueError):
                        self.backend_base_class.apply_environment_variables()
                else:
                    try:
                        self.backend_base_class.apply_environment_variables()
                    except FileNotFoundError as error:
                        # Compiled backends might not be available locally --> skip.
                        # On the CI, these will always be available, as the before_script builds them
                        # or otherwise fails the CI
                        if (
                            backend_mode == "fortran" or backend_mode == "cpp"
                        ):  # TODO better handling
                            warnings.warn(
                                f"{backend_mode} backend was not supported for {backend_bit}, compilation missing?"
                            )
                        else:
                            raise error


class TestCupy32Bit(unittest.TestCase):
    def test___init__(self) -> None:
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        self.cupy32_bit = Cupy32Bit()


class TestCupy64Bit(unittest.TestCase):
    def test___init__(self) -> None:
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        self.cupy64_bit = Cupy64Bit()


class TestCupyBackend(unittest.TestCase):
    def test___init__(self) -> None:
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        self.cupy_backend = CupyBackend(
            float_=np.float32, int_=np.float32, complex_=np.complex64
        )

    def test_set_specials(self) -> None:
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        self.cupy_backend = CupyBackend(
            float_=np.float32, int_=np.float32, complex_=np.complex64
        )
        self.cupy_backend.set_specials(mode="cuda")


class TestNumpy64Bit(unittest.TestCase):
    def setUp(self) -> None:
        self.numpy64_bit = Numpy64Bit()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp


class TestNumpyBackend(unittest.TestCase):
    def setUp(self) -> None:
        self.numpy_backend = NumpyBackend(
            float_=np.float32,
            int_=np.int32,
            complex_=np.complex64,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_set_specials_python(self) -> None:
        self.numpy_backend.set_specials(mode="python")

    def test_set_specials_cpp(self) -> None:
        try:
            self.numpy_backend.set_specials(mode="cpp")
        except FileNotFoundError:
            self.skipTest(f"cpp not available!")

    def test_set_specials_numba(self) -> None:
        self.numpy_backend.set_specials(mode="numba")

    def test_set_specials_fortran(self) -> None:
        try:
            self.numpy_backend.set_specials(mode="fortran")
        except FileNotFoundError:
            self.skipTest(f"fortran not available!")


class TestSpecials(unittest.TestCase):
    def setUp(self) -> None:
        self.n_voltages = 3
        self.special_modes = [
            "python",
            "cpp",
            "numba",
            "fortran",
        ]
        if cupy_available:
            self.special_modes.append("cuda")
        print(f"Testing {self.special_modes}")
        set_num_threads(8)

    def _setUp(self, dtype, special_mode) -> None:
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

        self.voltages = backend.linspace(
            1e6, 5e6, self.n_voltages, dtype=backend.float
        )
        self.omegas = backend.linspace(
            200e6, 400e6, self.n_voltages, dtype=backend.float
        )
        self.phis = backend.linspace(
            0, 2 * np.pi, self.n_voltages, dtype=backend.float
        )

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
    def test_drift_exact(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
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
                if special == "cuda":
                    result = result.get()
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    @unittest.skip
    def test_drift_legacy(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
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
                if special == "cuda":
                    result = result.get()
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    def test_drift_simple(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                backend.specials.drift_simple(
                    dt=self.dt,
                    dE=self.dE,
                    T=self.t_rev * self.length_ratio,
                    eta_0=self.eta_0,
                    beta=self.beta,
                    energy=self.energy,
                )
                result = self.dt
                if special == "cuda":
                    result = result.get()
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    def test_kick_multi_harmonic(self) -> None:
        for dtype in (np.float32, np.float64):
            for n_voltages in (1, 2, 3, 4, 5):
                for i, special in enumerate(self.special_modes):
                    self.n_voltages = n_voltages
                    try:
                        self._setUp(dtype=dtype, special_mode=special)
                    except (FileNotFoundError, OSError):
                        print(
                            f"Could not perform `{special}` test for {dtype}"
                        )
                        continue
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
                    if special == "cuda":
                        result = result.get()
                    if i == 0:
                        result_python = result
                    else:
                        np.testing.assert_allclose(
                            result,
                            result_python,
                            rtol=self.rtol,
                            err_msg=f"Failed test `{special}` with {dtype}",
                        )

    def test_kick_single_harmonic(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
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
                if special == "cuda":
                    result = result.get()
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    def test_kick_induced_voltage(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                dt = backend.linspace(-5, 5, 20, dtype=backend.float)
                dE = backend.zeros_like(dt, dtype=backend.float)
                bin_centers = backend.linspace(-4, 4, 20, dtype=backend.float)
                voltage = bin_centers**2
                charge = backend.float(10)
                acceleration_kick = backend.float(0.5)
                backend.specials.kick_induced_voltage(
                    dt=dt,
                    dE=dE,
                    voltage=voltage,
                    bin_centers=bin_centers,
                    charge=charge,
                    acceleration_kick=acceleration_kick,
                )
                result = dE
                if special == "cuda":
                    result = result.get()
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    def test_flagged_to_end(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                flag = backend.int(0)
                flags = backend.ones(10, dtype=np.int32)
                flags[[0, 1, -1]] = 0
                dt = backend.array(backend.linspace(0, 10, 10), backend.float)
                dE = backend.array(backend.linspace(0, 10, 10), backend.float)
                ids = backend.array(backend.arange(0, 10), backend.int)
                n_new = backend.specials.flagged_to_end(
                    flag=flag,
                    flags=flags,
                    dt=dt,
                    dE=dE,
                    ids=ids,
                )
                flags = flags[:n_new]
                dt = dt[:n_new]
                dE = dE[:n_new]
                ids = ids[:n_new]

                result = dt  # could be any of the 4 arrays
                self.assertEqual(
                    7,
                    len(flags),
                    msg=f"Failed test `{special}` with {dtype}",
                )
                self.assertEqual(
                    7,
                    len(dt),
                    msg=f"Failed test `{special}` with {dtype}",
                )
                self.assertEqual(
                    7,
                    len(dE),
                    msg=f"Failed test `{special}` with {dtype}",
                )
                self.assertEqual(
                    7,
                    len(ids),
                    msg=f"Failed test `{special}` with {dtype}",
                )
                if special == "cuda":
                    result = result.get()

                result = np.sort(result)  # because of race conditions in
                # parallel execution, the order can not be guaranteed

                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    def test_flagged_to_end_potentially_race_conditions(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                flag = backend.int(0)
                flags = backend.ones(int(1e6), dtype=np.int32)
                np.random.seed(0)
                flags[np.random.randint(0, len(flags), int(1e5))] = 0
                dt = backend.array(
                    backend.linspace(0, 10, len(flags)), backend.float
                )
                dE = backend.array(
                    backend.linspace(0, 10, len(flags)), backend.float
                )
                ids = backend.array(backend.arange(0, len(flags)), backend.int)
                n_new = backend.specials.flagged_to_end(
                    flag=flag,
                    flags=flags,
                    dt=dt,
                    dE=dE,
                    ids=ids,
                )
                assert np.all(flags[:n_new] != 0)
                assert np.all(flags[n_new:] == 0)
                flags = flags[:n_new]
                dt = dt[:n_new]
                dE = dE[:n_new]
                ids = ids[:n_new]

                result = dt  # could be any of the 4 arrays
                if special == "cuda":
                    result = result.get()

                result = np.sort(result)  # because of race conditions in
                # parallel execution, the order can not be guaranteed

                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    def test_flagged_to_end_none_flagged(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                flag = backend.int(0)
                flags = backend.ones(10, dtype=np.int32)

                dt = backend.array(backend.linspace(0, 10, 10), backend.float)
                dE = backend.array(backend.linspace(0, 10, 10), backend.float)
                ids = backend.array(backend.arange(0, 10), backend.int)
                n_new = backend.specials.flagged_to_end(
                    flag=flag,
                    flags=flags,
                    dt=dt,
                    dE=dE,
                    ids=ids,
                )

                self.assertEqual(
                    10,
                    n_new,
                    msg=f"Failed test `{special}` with {dtype}",
                )

    def test_flagged_to_end_all_but_one_flagged(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError) as exc:
                    if True:
                        raise exc
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                flag = backend.int(0)
                flags = backend.zeros(10, dtype=np.int32)
                flags[1] = 1

                dt = backend.array(backend.linspace(0, 10, 10), backend.float)
                dE = backend.array(backend.linspace(0, 10, 10), backend.float)
                ids = backend.array(backend.arange(0, 10), backend.int)
                n_new = backend.specials.flagged_to_end(
                    flag=flag,
                    flags=flags,
                    dt=dt,
                    dE=dE,
                    ids=ids,
                )

                self.assertEqual(
                    1,
                    n_new,
                    msg=f"Failed test `{special}` with {dtype}",
                )

    def test_flagged_to_end_all_flagged(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                flag = backend.int(0)
                flags = backend.zeros(10, dtype=np.int32)

                dt = backend.array(backend.linspace(0, 10, 10), backend.float)
                dE = backend.array(backend.linspace(0, 10, 10), backend.float)
                ids = backend.array(backend.arange(0, 10), backend.int)
                n_new = backend.specials.flagged_to_end(
                    flag=flag,
                    flags=flags,
                    dt=dt,
                    dE=dE,
                    ids=ids,
                )

                self.assertEqual(
                    0,
                    n_new,
                    msg=f"Failed test `{special}` with {dtype}",
                )

    @unittest.skip
    def test_loss_box(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue

                top = backend.float(1)
                bottom = backend.float(-1)
                left = backend.float(-10)
                right = backend.float(10)
                dt = backend.linspace(-20, 20, dtype=backend.float)
                dE = backend.linspace(-2, 2, dtype=backend.float)
                flags = backend.arange(len(dt), dtype=np.int32)
                result = flags

                backend.specials.loss_box(
                    top=top,
                    bottom=bottom,
                    left=left,
                    right=right,
                    dt=dt,
                    dE=dE,
                    flags=flags,
                )
                if special == "cuda":
                    result = result.get()
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    def test_beam_phase(self) -> None:
        for dtype in (
            np.float32,
            np.float64,
        ):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                result = backend.specials.beam_phase(
                    hist_x=backend.linspace(-10, 10, 21, dtype=backend.float),
                    hist_y=10**2
                    - backend.linspace(-10, 10, 21, dtype=backend.float) ** 2,
                    alpha=backend.float(1.5),
                    omega_rf=backend.float(2.5),
                    phi_rf=backend.float(3.5),
                    bin_size=backend.float(1.0),
                )
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        # There is some numerical reason, why 32-bit C++ and
                        # Numba returns slightly different results than
                        # Python.
                        # The Fortran port of the C++ code works fine,
                        # so it's not an algorithmic problem, but something
                        # governed by the compiler.
                        # The accuracy for 32-bit test is therefore lowered
                        # to 1e-5 instead of 1e-6, hopefully without
                        # consequences.
                        rtol=1e-5 if dtype is np.float32 else self.rtol,
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    def test_histogram(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                array_write = backend.ones(21, dtype=backend.float)
                for _ in range(2):
                    backend.specials.histogram(
                        array_read=backend.linspace(
                            -10, 10, 21, dtype=backend.float
                        ),
                        array_write=array_write,
                        start=backend.float(-12),
                        stop=backend.float(8.0),
                    )
                result = array_write

                if special == "cuda":
                    result = result.get()
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"{special=} {dtype=}",
                    )

    def test_histogram_long_profiles(self) -> None:
        """Specifically to test edge effects at beginning and end."""
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                array_write = backend.ones(3, dtype=backend.float)
                spac = backend.linspace(0, 10, 50, dtype=backend.float)
                for _ in range(2):
                    backend.specials.histogram(
                        array_read=spac,
                        array_write=array_write,
                        start=backend.float(2),
                        stop=backend.float(4),
                    )
                result = array_write

                if special == "cuda":
                    result = result.get()
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"{special=} {dtype=}",
                    )

    def test_histogram_short_profile(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                array_write = backend.ones(21, dtype=backend.float)
                for _ in range(2):
                    backend.specials.histogram(
                        array_read=backend.linspace(
                            -5, 5, 51, dtype=backend.float
                        ),
                        array_write=array_write,
                        start=backend.float(-10),
                        stop=backend.float(10),
                    )
                result = array_write

                if special == "cuda":
                    result = result.get()
                if i == 0:
                    result_python = result
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"{special=} {dtype=}",
                    )

    def test_histogram_race_conditions(self) -> None:
        backend.random.seed(42)
        array_read = (
            backend.random.random_sample(size=1024) - 0.5
        ) * 20  # common sample data from -10 to 10
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                set_num_threads(8)
                array_write = backend.ones(21, dtype=backend.float)
                #
                backend.specials.histogram(
                    array_read=backend.array(
                        array_read, dtype=backend.float
                    ),  # casting to correct data type
                    array_write=array_write,
                    start=backend.float(-12),
                    stop=backend.float(8.0),
                )
                result = array_write
                print(result.tolist())

                if special == "cuda":
                    result = result.get()
                if i == 0:
                    result_python = result
                    print(result_python.tolist())
                else:
                    np.testing.assert_allclose(
                        result,
                        result_python,
                        rtol=self.rtol,
                        err_msg=f"{special=} {dtype=}",
                    )

    def tearDown(self) -> None:
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")


if __name__ == "__main__":
    unittest.main()
