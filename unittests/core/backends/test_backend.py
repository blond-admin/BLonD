import unittest
import warnings

import numpy as np
import pytest

from blond.core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    CupyBackend,
    Numpy32Bit,
    Numpy64Bit,
    NumpyBackend,
    backend,
    default,
)
from blond.core.backends.numba.callables import recompile_numba_backend
from blond.generals.exceptions_ import ArrayCastingError
from blond.testing.backend_testing import (
    multi_backend_testcase,
    skip_if_no_cupy,
)

try:
    import cupy as cp  # type: ignore

    cupy_available = True
except ModuleNotFoundError:
    cupy_available = False

from numba import set_num_threads

from blond.testing.helpers import allclose_tolerances


class TestBackendBaseClass(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        backend.change_backend(default)
        backend.set_specials("numba")

    def setUp(self) -> None:
        self.backend_base_class = Numpy32Bit()

    @pytest.mark.backend_mutation
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @pytest.mark.backend_mutation
    def test_change_backend(self) -> None:
        self.backend_base_class.change_backend(new_backend=Numpy64Bit)
        self.assertEqual(self.backend_base_class.float, np.float64)
        self.assertEqual(self.backend_base_class.complex, np.complex128)

    @pytest.mark.backend_mutation
    def test_set_specials(self) -> None:
        self.backend_base_class.set_specials(mode="numba")

    def tearDown(self) -> None:
        self.backend_base_class.set_specials(mode="numba")

    @pytest.mark.backend_mutation
    def test_apply_environment_variables(self):
        import os

        backend_modes = ["python", "cpp", "cpp_single_core", "numba", "fail"]
        backend_bits = ["32", "64", "fail"]
        try:
            import cupy

            backend_modes = ["cuda"] + backend_modes
        except ModuleNotFoundError:
            pass
        print(f"{backend_modes=}")
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
                        if backend_mode == "cpp":  # TODO better handling
                            warnings.warn(
                                f"{backend_mode} backend was not supported for {backend_bit}, compilation missing?"
                            )
                        else:
                            raise error

    @pytest.mark.backend_mutation
    def test__finalize(self):
        some_backend = Numpy32Bit()
        some_backend.array = None
        with self.assertRaises(AttributeError):
            some_backend._finalize()

    @pytest.mark.backend_mutation
    def test_change_backend(self):
        some_backend = Numpy32Bit()
        some_backend.change_backend(some_backend)  # shouldnt do anything

    @pytest.mark.backend_mutation
    def test_temporary_specials_mode(self):
        backend_org = type(backend)
        backend.change_backend(Numpy64Bit)
        specials_org = (
            backend.specials_mode
        )  # prevent side effect on other tests

        backend.set_specials("numba")
        with backend.temporary_specials_mode(mode="python"):
            self.assertEqual(backend.specials_mode, "python")
        self.assertEqual(backend.specials_mode, "numba")

        backend.set_specials(mode=specials_org)  # prevent side effect on tests
        backend.change_backend(backend_org)


class TestCupy32Bit(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test___init__(self) -> None:
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        self.cupy32_bit = Cupy32Bit()


class TestCupy64Bit(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test___init__(self) -> None:
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        self.cupy64_bit = Cupy64Bit()


class TestCupyBackend(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test___init__(self) -> None:
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        self.cupy_backend = CupyBackend(
            float_=np.float32, complex_=np.complex64
        )

    @pytest.mark.backend_mutation
    def test_set_specials(self) -> None:
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        self.cupy_backend = CupyBackend(
            float_=np.float32, complex_=np.complex64
        )
        self.cupy_backend.set_specials(mode="cuda")

    @pytest.mark.backend_mutation
    def test_set_specials_fails(self):
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        self.cupy_backend = CupyBackend(
            float_=np.float32, complex_=np.complex64
        )
        with self.assertRaises(ValueError):
            self.cupy_backend.set_specials("doesnt exist")


class TestNumpy64Bit(unittest.TestCase):
    def setUp(self) -> None:
        self.numpy64_bit = Numpy64Bit()

    @pytest.mark.backend_mutation
    def test___init__(self):
        pass  # calls __init__ in  self.setUp


class TestNumpyBackend(unittest.TestCase):
    def setUp(self) -> None:
        self.numpy_backend = NumpyBackend(
            float_=np.float32,
            complex_=np.complex64,
        )

    @pytest.mark.backend_mutation
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @pytest.mark.backend_mutation
    def test_set_specials_python(self) -> None:
        self.numpy_backend.set_specials(mode="python")

    @pytest.mark.backend_mutation
    def test_set_specials_cpp(self) -> None:
        try:
            self.numpy_backend.set_specials(mode="cpp")
        except FileNotFoundError:
            self.skipTest("cpp not available!")

    @pytest.mark.backend_mutation
    def test_set_specials_cpp(self) -> None:
        try:
            self.numpy_backend.set_specials(mode="cpp_single_core")
        except FileNotFoundError:
            self.skipTest("cpp_single_core not available!")

    @pytest.mark.backend_mutation
    def test_set_specials_numba(self) -> None:
        self.numpy_backend.set_specials(mode="numba")

    @pytest.mark.backend_mutation
    def test_set_specials_fails(self):
        with self.assertRaises(ValueError):
            self.numpy_backend.set_specials("doesnt exist")


class TestSpecials(unittest.TestCase):
    def setUp(self) -> None:
        self.n_voltages = 3
        self.special_modes = [
            "python",
            "cpp",
            "cpp_single_core",
            "numba",
        ]
        if cupy_available:
            self.special_modes.append("cuda")
        set_num_threads(8)

    @pytest.mark.backend_mutation
    def _setUp(self, dtype, special_mode) -> None:
        if special_mode in (
            "python",
            "cpp",
            "cpp_single_core",
            "numba",
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
        self.alpha_order = np.int32(3)
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

    @pytest.mark.backend_mutation
    def test___init__(self):
        pass

    @pytest.mark.backend_mutation
    def test_drift_exact(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                for _ in range(2):
                    backend.specials.drift_exact(
                        dt=self.dt,
                        dE=self.dE,
                        T=self.t_rev * self.length_ratio,
                        alpha_0=self.alpha_0,
                        higher_alpha=backend.array([1.0, 2.0], dtype=dtype),
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
    @pytest.mark.backend_mutation
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

    @pytest.mark.backend_mutation
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
                        rtol=1e-5 if dtype == np.float32 else 1e-12,
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    @pytest.mark.backend_mutation
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

    @pytest.mark.backend_mutation
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
                        rtol=1e-5 if dtype == np.float32 else 1e-12,
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    @pytest.mark.backend_mutation
    def test_kick_interpolated(self) -> None:
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
                backend.specials.kick_interpolated(
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

    @pytest.mark.backend_mutation
    def test_kick_interpolated_edges(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                dt = backend.linspace(-5, 5, 20, dtype=backend.float)
                dE = backend.zeros_like(dt, dtype=backend.float)
                bin_centers = dt.copy()
                voltage = bin_centers**2
                charge = float(10)
                acceleration_kick = float(0.5)
                backend.specials.kick_interpolated(
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

    @pytest.mark.backend_mutation
    def test_kick_interpolated_bug(self) -> None:
        kwargs = {
            "dt": [
                0.0,
                0.010101010101010102,
                0.020202020202020204,
                0.030303030303030304,
                0.04040404040404041,
                0.05050505050505051,
                0.06060606060606061,
                0.07070707070707072,
                0.08080808080808081,
                0.09090909090909091,
                0.10101010101010102,
                0.11111111111111112,
                0.12121212121212122,
                0.13131313131313133,
                0.14141414141414144,
                0.15151515151515152,
                0.16161616161616163,
                0.17171717171717174,
                0.18181818181818182,
                0.19191919191919193,
                0.20202020202020204,
                0.21212121212121213,
                0.22222222222222224,
                0.23232323232323235,
                0.24242424242424243,
                0.25252525252525254,
                0.26262626262626265,
                0.27272727272727276,
                0.2828282828282829,
                0.29292929292929293,
                0.30303030303030304,
                0.31313131313131315,
                0.32323232323232326,
                0.33333333333333337,
                0.3434343434343435,
                0.3535353535353536,
                0.36363636363636365,
                0.37373737373737376,
                0.38383838383838387,
                0.393939393939394,
                0.4040404040404041,
                0.4141414141414142,
                0.42424242424242425,
                0.43434343434343436,
                0.4444444444444445,
                0.4545454545454546,
                0.4646464646464647,
                0.4747474747474748,
                0.48484848484848486,
                0.494949494949495,
                0.5050505050505051,
                0.5151515151515152,
                0.5252525252525253,
                0.5353535353535354,
                0.5454545454545455,
                0.5555555555555556,
                0.5656565656565657,
                0.5757575757575758,
                0.5858585858585859,
                0.595959595959596,
                0.6060606060606061,
                0.6161616161616162,
                0.6262626262626263,
                0.6363636363636365,
                0.6464646464646465,
                0.6565656565656566,
                0.6666666666666667,
                0.6767676767676768,
                0.686868686868687,
                0.696969696969697,
                0.7070707070707072,
                0.7171717171717172,
                0.7272727272727273,
                0.7373737373737375,
                0.7474747474747475,
                0.7575757575757577,
                0.7676767676767677,
                0.7777777777777778,
                0.787878787878788,
                0.797979797979798,
                0.8080808080808082,
                0.8181818181818182,
                0.8282828282828284,
                0.8383838383838385,
                0.8484848484848485,
                0.8585858585858587,
                0.8686868686868687,
                0.8787878787878789,
                0.888888888888889,
                0.8989898989898991,
                0.9090909090909092,
                0.9191919191919192,
                0.9292929292929294,
                0.9393939393939394,
                0.9494949494949496,
                0.9595959595959597,
                0.9696969696969697,
                0.9797979797979799,
                0.98989898989899,
                1.0,
            ],
            "dE": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "voltage": [
                2326.6520833917625,
                -381.49412393699384,
                -1811.6982505072594,
                2640.2141767755493,
                -1637.9477901076682,
                -159.37787531615425,
                1149.1163917576641,
                -565.5699765539746,
                -872.0219800580135,
                1628.2292715811743,
                -745.7741612715321,
                -1260.8425707345555,
                2803.016119909949,
                -2570.5531671828944,
                666.7240796267399,
                1461.954154969815,
                -2243.9523135264876,
                1295.7788621295351,
                303.17035737205947,
                -1001.6621868820863,
                159.8838862167488,
                1386.794652591038,
                -2076.0801670752503,
                1022.9266222754347,
                1152.5560058584956,
                -2806.2167627899316,
                2652.539924949482,
                -853.9280264564891,
                -1126.4962871892715,
                1779.1637632779316,
                -820.5037382971319,
                -616.7362734410475,
                1029.8394301084795,
                94.49929975021905,
                -1802.320561992251,
                2498.5829891112953,
                -1358.4271521480184,
                -910.5628214537546,
                2628.0251683283986,
                -2546.0356688858687,
                887.0717704607081,
                879.9438566838447,
                -1324.3630138484484,
                278.80167651420993,
                1058.788261011891,
                -1226.194075827776,
                -169.7459630768501,
                2061.2744624193347,
                -2820.592752968017,
                1674.3117354618485,
                600.3346225116304,
                -2308.6599105305363,
                2258.0947909726938,
                -738.4735491712512,
                -779.4889956600757,
                955.019449298428,
                250.72078851610888,
                -1563.518719002466,
                1551.0307161171581,
                71.58091221168718,
                -2134.5310402345467,
                2984.1812884652136,
                -1895.4041553418774,
                -299.5504192213387,
                1913.1062030216408,
                -1828.148865259808,
                413.97757593601204,
                854.3250828518638,
                -729.6923978543226,
                -691.926472860837,
                2052.659864835712,
                -1939.7335345977115,
                161.83655443382634,
                2026.4606531545433,
                -2959.2973842530278,
                1963.2173098189578,
                83.72177637576453,
                -1519.129454680069,
                1320.795547700394,
                48.04508098213665,
                -1100.2337895847263,
                679.2836092605421,
                985.0885770465097,
                -2449.8410323027592,
                2314.679699151558,
                -467.0098682169019,
                -1774.0611446218006,
                2749.2992459322018,
                -1846.7847729995692,
                -12.218535419808518,
                1202.940427297991,
                -813.9575441049037,
                -583.6844923053268,
                1480.3275213527909,
                -801.385846780807,
                -1097.8102224141207,
                2694.6002656246537,
                -2599.5520767793446,
                766.868717749934,
                1440.0848398393546,
            ],
            "bin_centers": [
                0.0,
                0.010101010101010102,
                0.020202020202020204,
                0.030303030303030304,
                0.04040404040404041,
                0.05050505050505051,
                0.06060606060606061,
                0.07070707070707072,
                0.08080808080808081,
                0.09090909090909091,
                0.10101010101010102,
                0.11111111111111112,
                0.12121212121212122,
                0.13131313131313133,
                0.14141414141414144,
                0.15151515151515152,
                0.16161616161616163,
                0.17171717171717174,
                0.18181818181818182,
                0.19191919191919193,
                0.20202020202020204,
                0.21212121212121213,
                0.22222222222222224,
                0.23232323232323235,
                0.24242424242424243,
                0.25252525252525254,
                0.26262626262626265,
                0.27272727272727276,
                0.2828282828282829,
                0.29292929292929293,
                0.30303030303030304,
                0.31313131313131315,
                0.32323232323232326,
                0.33333333333333337,
                0.3434343434343435,
                0.3535353535353536,
                0.36363636363636365,
                0.37373737373737376,
                0.38383838383838387,
                0.393939393939394,
                0.4040404040404041,
                0.4141414141414142,
                0.42424242424242425,
                0.43434343434343436,
                0.4444444444444445,
                0.4545454545454546,
                0.4646464646464647,
                0.4747474747474748,
                0.48484848484848486,
                0.494949494949495,
                0.5050505050505051,
                0.5151515151515152,
                0.5252525252525253,
                0.5353535353535354,
                0.5454545454545455,
                0.5555555555555556,
                0.5656565656565657,
                0.5757575757575758,
                0.5858585858585859,
                0.595959595959596,
                0.6060606060606061,
                0.6161616161616162,
                0.6262626262626263,
                0.6363636363636365,
                0.6464646464646465,
                0.6565656565656566,
                0.6666666666666667,
                0.6767676767676768,
                0.686868686868687,
                0.696969696969697,
                0.7070707070707072,
                0.7171717171717172,
                0.7272727272727273,
                0.7373737373737375,
                0.7474747474747475,
                0.7575757575757577,
                0.7676767676767677,
                0.7777777777777778,
                0.787878787878788,
                0.797979797979798,
                0.8080808080808082,
                0.8181818181818182,
                0.8282828282828284,
                0.8383838383838385,
                0.8484848484848485,
                0.8585858585858587,
                0.8686868686868687,
                0.8787878787878789,
                0.888888888888889,
                0.8989898989898991,
                0.9090909090909092,
                0.9191919191919192,
                0.9292929292929294,
                0.9393939393939394,
                0.9494949494949496,
                0.9595959595959597,
                0.9696969696969697,
                0.9797979797979799,
                0.98989898989899,
                1.0,
            ],
            "charge": 82.0,
            "acceleration_kick": 0.0,
        }
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                dt = backend.array(
                    kwargs["dt"],
                    dtype=backend.float,
                )
                dE = backend.array(
                    kwargs["dE"],
                    dtype=backend.float,
                )
                bin_centers = backend.array(
                    kwargs["bin_centers"],
                    dtype=backend.float,
                )
                voltage = backend.array(
                    kwargs["voltage"],
                    dtype=backend.float,
                )
                charge = kwargs["charge"]
                acceleration_kick = kwargs["acceleration_kick"]
                backend.specials.kick_interpolated(
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
                        **allclose_tolerances(result_python, 1e-3),
                        # FIXME
                        #  this tolerance is so low because of the GPU
                        #  backend. Reason unknown for now.
                        err_msg=f"Failed test `{special}` with {dtype}",
                    )

    @pytest.mark.backend_mutation
    def test_move_flagged_elements_to_end(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                flag = 0
                flags = backend.ones(10, dtype=np.int32)
                flags[[0, 1, -1]] = 0
                dt = backend.array(backend.linspace(0, 10, 10), backend.float)
                dE = backend.array(backend.linspace(0, 10, 10), backend.float)
                ids = backend.array(backend.arange(0, 10), np.int32)
                n_new = backend.specials.move_flagged_elements_to_end(
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
                self.assertTrue(np.all(flags == np.ones_like(flags)))
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

    @pytest.mark.backend_mutation
    def test_move_flagged_elements_to_end_potentially_race_conditions(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                flag = 0
                flags = backend.ones(int(1e6), dtype=np.int32)
                np.random.seed(0)
                flags[np.random.randint(0, len(flags), int(1e5))] = 0
                dt = backend.array(
                    backend.linspace(0, 10, len(flags)), backend.float
                )
                dE = backend.array(
                    backend.linspace(0, 10, len(flags)), backend.float
                )
                ids = backend.array(backend.arange(0, len(flags)), np.int32)
                n_new = backend.specials.move_flagged_elements_to_end(
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

    @pytest.mark.backend_mutation
    def test_move_flagged_elements_to_end_none_flagged(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                flag = 0
                flags = backend.ones(10, dtype=np.int32)

                dt = backend.array(backend.linspace(0, 10, 10), backend.float)
                dE = backend.array(backend.linspace(0, 10, 10), backend.float)
                ids = backend.array(backend.arange(0, 10), np.int32)
                n_new = backend.specials.move_flagged_elements_to_end(
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

    @pytest.mark.backend_mutation
    def test_move_flagged_elements_to_end_all_but_one_flagged(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                flag = 0
                flags = backend.zeros(10, dtype=np.int32)
                flags[1] = 1

                dt = backend.array(backend.linspace(0, 10, 10), backend.float)
                dE = backend.array(backend.linspace(0, 10, 10), backend.float)
                ids = backend.array(backend.arange(0, 10), np.int32)
                n_new = backend.specials.move_flagged_elements_to_end(
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

    @pytest.mark.backend_mutation
    def test_move_flagged_elements_to_end_all_flagged(self):
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                flag = 0
                flags = backend.zeros(10, dtype=np.int32)

                dt = backend.array(backend.linspace(0, 10, 10), backend.float)
                dE = backend.array(backend.linspace(0, 10, 10), backend.float)
                ids = backend.array(backend.arange(0, 10), np.int32)
                n_new = backend.specials.move_flagged_elements_to_end(
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

    @pytest.mark.backend_mutation
    def test_loss_box(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue

                e_max = backend.float(1)
                e_min = backend.float(-1)
                t_min = backend.float(-10)
                t_max = backend.float(10)
                dt = backend.linspace(-20, 20, dtype=backend.float)
                dE = backend.linspace(-2, 2, dtype=backend.float)
                flags = backend.arange(len(dt), dtype=np.int32)
                result = flags

                backend.specials.loss_box(
                    e_max=e_max,
                    e_min=e_min,
                    t_min=t_min,
                    t_max=t_max,
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

    @pytest.mark.backend_mutation
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

    @pytest.mark.backend_mutation
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

    @pytest.mark.backend_mutation
    def test_histogram_sparse(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                bins_per_profile = 3
                n_profiles = 3
                array_write = backend.ones(
                    bins_per_profile * n_profiles, dtype=backend.float
                )
                filling_pattern = backend.array([1, 0, 1, 0, 1, 0], dtype=bool)
                bucket_index_to_memory_index = backend.array(
                    [0, 0, 3, 3, 6, 6],
                    dtype=np.int32,
                )
                for _ in range(2):
                    backend.specials.histogram_sparse(
                        x=backend.linspace(-10, 10, 21, dtype=backend.float),
                        out=array_write,
                        first_left_cut=-12,
                        left_cut_distance=8,
                        cut_width=4,
                        bins_per_profile=bins_per_profile,
                        n_active_profiles=n_profiles,
                        filling_pattern=filling_pattern,
                        bucket_index_to_memory_index=bucket_index_to_memory_index,
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

    @pytest.mark.backend_mutation
    def test_histogram_sparse_left_edged(self) -> None:
        for dtype in (np.float32, np.float64):
            for i, special in enumerate(self.special_modes):
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                bins_per_profile = 4
                n_profiles = 3
                array_write = backend.ones(
                    bins_per_profile * n_profiles, dtype=backend.float
                )
                filling_pattern = backend.array([1, 0, 1, 0, 1, 0], dtype=bool)
                bucket_index_to_memory_index = backend.array(
                    [0, 0, 4, 4, 8, 8],
                    dtype=np.int32,
                )
                particles_x = []
                # mark all left and right edges
                for left_edge in (-12, -12 + 2 * 8, -12 + 4 * 8):
                    for _ in range(2):
                        particles_x.append(left_edge)
                for right_edge in (-12 + 4, -12 + 2 * 8 + 4, -12 + 4 * 8 + 4):
                    for _ in range(1):
                        particles_x.append(right_edge)
                particles_x = backend.array(particles_x, backend.float)
                for _ in range(2):
                    backend.specials.histogram_sparse(
                        x=particles_x,
                        out=array_write,
                        first_left_cut=-12,
                        left_cut_distance=8,
                        cut_width=4,
                        bins_per_profile=bins_per_profile,
                        n_active_profiles=n_profiles,
                        filling_pattern=filling_pattern,
                        bucket_index_to_memory_index=bucket_index_to_memory_index,
                    )
                print(backend.specials_mode, array_write)
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

    @pytest.mark.backend_mutation
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

    @pytest.mark.backend_mutation
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

    @pytest.mark.backend_mutation
    def test_histogram_race_conditions(self) -> None:
        backend.random.seed(np.uint(42))
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

    @multi_backend_testcase("Numpy32Bit", "Numpy64Bit")
    def test_cast_float_arr_np_only(self):
        target = backend.array([1, 2, 3], dtype=backend.float)

        for in_type in (tuple, list, np.array):
            cast = backend.cast_arr_float_if_needed(in_type(target))
            self.assertTrue(cast.dtype == backend.float)
            self.assertIsInstance(cast, backend.ndarray)
            np.testing.assert_array_equal(cast, target)

        for in_dtype in (
            np.int32,
            np.int64,
            np.float64,
            np.complex64,
            np.complex128,
        ):
            cast = backend.cast_arr_float_if_needed(target.astype(in_dtype))
            self.assertTrue(cast.dtype == backend.float)
            self.assertIsInstance(cast, backend.ndarray)
            np.testing.assert_array_equal(cast, target)

        unchanged = backend.cast_arr_float_if_needed(target)
        self.assertTrue(target is unchanged)

    @skip_if_no_cupy
    @multi_backend_testcase
    def test_cast_float_arr_full(self):
        for in_type in (tuple, list, np.array, cp.array):
            # Recreate the target for each loop, avoids issues with
            # transferring back and forth between cupy and numpy.
            target = backend.array([1, 2, 3], dtype=backend.float)

            if backend.ndarray is cp.ndarray:
                to_cast = in_type(target.get())
            else:
                to_cast = in_type(target)

            cast = backend.cast_arr_float_if_needed(to_cast)
            self.assertTrue(cast.dtype == backend.float)
            self.assertIsInstance(cast, backend.ndarray)
            if isinstance(backend, CupyBackend):
                cast = cast.get()
                target = target.get()
            np.testing.assert_array_equal(cast, target)
        for in_dtype in (
            np.int32,
            np.int64,
            np.float64,
            np.complex64,
            np.complex128,
        ):
            # Recreate the target for each loop, avoids issues with
            # transferring back and forth between cupy and numpy.
            target = backend.array([1, 2, 3], dtype=backend.float)
            to_cast = target.astype(in_dtype)
            cast = backend.cast_arr_float_if_needed(to_cast)
            self.assertTrue(cast.dtype == backend.float)
            self.assertIsInstance(cast, backend.ndarray)

            if isinstance(backend, CupyBackend):
                cast = cast.get()
                target = target.get()

            np.testing.assert_array_equal(cast, target)

        target = backend.array([1, 2, 3], dtype=backend.float)
        unchanged = backend.cast_arr_float_if_needed(target)
        self.assertTrue(target is unchanged)

    @multi_backend_testcase("Numpy32Bit", "Numpy64Bit")
    def test_cast_complex_arr_np_only(self):
        target = backend.array([1, 2, 3], dtype=backend.complex)
        for in_type in (tuple, list, np.array):
            cast = backend.cast_arr_complex_if_needed(in_type(target))
            self.assertTrue(cast.dtype == backend.complex)
            self.assertIsInstance(cast, backend.ndarray)
            np.testing.assert_array_equal(cast, target)

        for in_dtype in (
            np.int32,
            np.int64,
            np.float64,
            np.complex64,
            np.complex128,
        ):
            cast = backend.cast_arr_complex_if_needed(target.astype(in_dtype))
            self.assertTrue(cast.dtype == backend.complex)
            self.assertIsInstance(cast, backend.ndarray)
            np.testing.assert_array_equal(cast, target)

        target = backend.array([1, 2, 3], dtype=backend.complex)
        unchanged = backend.cast_arr_complex_if_needed(target)
        self.assertTrue(target is unchanged)

    @skip_if_no_cupy
    @multi_backend_testcase
    def test_cast_complex_arr_full(self):
        for in_type in (tuple, list, np.array, cp.array):
            # Recreate the target for each loop, avoids issues with
            # transferring back and forth between cupy and numpy.
            target = backend.array([1, 2, 3], dtype=backend.complex)

            if backend.ndarray is cp.ndarray:
                to_cast = in_type(target.get())
            else:
                to_cast = in_type(target)
            cast = backend.cast_arr_complex_if_needed(to_cast)
            self.assertTrue(cast.dtype == backend.complex)
            self.assertIsInstance(cast, backend.ndarray)
            if isinstance(backend, CupyBackend):
                cast = cast.get()
                target = target.get()
            np.testing.assert_array_equal(cast, target)
        for in_dtype in (
            np.int32,
            np.int64,
            np.float64,
            np.complex64,
            np.complex128,
        ):
            # Recreate the target for each loop, avoids issues with
            # transferring back and forth between cupy and numpy.
            target = backend.array([1, 2, 3], dtype=backend.complex)
            # Manually discard imaginary to prevent exception
            # Needed for cupy array backends
            to_cast = target.real.astype(in_dtype)
            cast = backend.cast_arr_complex_if_needed(to_cast)
            self.assertTrue(cast.dtype == backend.complex)
            self.assertIsInstance(cast, backend.ndarray)
            if isinstance(backend, CupyBackend):
                cast = cast.get()
                target = target.get()

            np.testing.assert_array_equal(cast, target)

        target = backend.array([1, 2, 3], dtype=backend.complex)
        unchanged = backend.cast_arr_complex_if_needed(target)
        self.assertTrue(target is unchanged)

    @multi_backend_testcase
    def test_cast_exceptions(self):
        with self.assertRaises(ArrayCastingError):
            backend.cast_arr_float_if_needed(["a", "b", "c"])

        with self.assertRaises(ArrayCastingError):
            backend.cast_arr_float_if_needed({1, 2, 3})

        with self.assertRaises(ArrayCastingError):
            backend.cast_arr_float_if_needed([[1, 2], 3])

    def tearDown(self) -> None:
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")

    def test_import(self):
        from blond.core.backends import backend  # see if import works


class TestNumbaCompilation(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_raising_of_error(self) -> None:
        with self.assertRaises(TypeError):
            recompile_numba_backend(floattype=np.float16)


if __name__ == "__main__":
    unittest.main()
