import os
import subprocess
import sys
import unittest
import warnings

import numpy as np
import pytest

from blond import copy_to_cpu
from blond.core.backends.backend import (
    Cupy64Bit,
    CupyBackend,
    Numpy64Bit,
    NumpyBackend,
    backend,
)
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

backend_org = backend.__class__
backend_specials_mode_org = backend.specials_mode


class TestBackendBaseClass(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        backend.change_backend(backend_org)
        backend.set_specials(backend_specials_mode_org)

    def setUp(self) -> None:
        self.backend_base_class = Numpy64Bit()

    @pytest.mark.backend_mutation
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @pytest.mark.backend_mutation
    def test_autoselect_backend(self) -> None:
        self.backend_base_class.autoselect_backend()

    @pytest.mark.backend_mutation
    def test_change_backend(self) -> None:
        self.backend_base_class.change_backend(new_backend=Numpy64Bit)
        self.assertEqual(self.backend_base_class.float, np.float64)
        self.assertEqual(self.backend_base_class.complex, np.complex128)

    @pytest.mark.backend_mutation
    def test_set_specials(self) -> None:
        self.backend_base_class.set_specials(mode="numba")

    def tearDown(self) -> None:
        self.backend_base_class.change_backend(Numpy64Bit)
        self.backend_base_class.set_specials(mode="cpp")

    @pytest.mark.backend_mutation
    def test_apply_environment_variables(self):
        import os

        backend_modes = ["python", "cpp", "cpp_single_core", "numba", "fail"]
        backend_bits = ["64", "fail"]
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
        some_backend = Numpy64Bit()
        some_backend.array = None
        with self.assertRaises(AttributeError):
            some_backend._finalize()

    @pytest.mark.backend_mutation
    def test_change_backend(self):
        some_backend = Numpy64Bit()
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

    @pytest.mark.backend_mutation
    def test_apply_environment_variables_error_names_env_var(self):
        import os

        # Save the original value so the try/finally can restore the process
        # environment exactly as it was, leaking no state into other tests.
        # mode_org is None when the var was unset, so we must distinguish
        # "delete it again" from "put the old value back".
        mode_org = os.environ.get("BLOND_BACKEND_MODE")
        os.environ["BLOND_BACKEND_MODE"] = "doesnt_exist"
        try:
            with self.assertRaisesRegex(ValueError, "BLOND_BACKEND_MODE"):
                self.backend_base_class.apply_environment_variables()
        finally:
            if mode_org is None:
                del os.environ["BLOND_BACKEND_MODE"]
            else:
                os.environ["BLOND_BACKEND_MODE"] = mode_org

    @pytest.mark.backend_mutation
    def test_setup_backend_cpp_single_core(self):
        from blond.core.backends.helpers import setup_backend

        setup_backend("cpp_single_core")
        self.assertEqual(backend.specials_mode, "cpp_single_core")


def _run_python(code: str) -> "subprocess.CompletedProcess[str]":
    """Run a code snippet in a fresh interpreter without BLOND env vars."""
    env = os.environ.copy()
    for key in ("BLOND_BACKEND_MODE", "BLOND_BACKEND_BITS"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


class TestImportSideEffects(unittest.TestCase):
    """Importing the backend must not print, probe, or compile anything."""

    def test_import_has_no_stdout_side_effects(self):
        result = _run_python("import blond.core.backends.backend")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "",
            msg=f"import must not print, got: {result.stdout!r}",
        )

    def test_available_backends_is_lazy(self):
        result = _run_python(
            "import blond.core.backends.backend as b;"
            "print('AVAILABLE_BACKENDS' in vars(b));"
            "print('Numpy64Bit' in b.AVAILABLE_BACKENDS);"
            "print('AVAILABLE_BACKENDS' in vars(b))"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(
            result.stdout.split(),
            ["False", "True", "True"],
            msg="backends must only be probed on first access",
        )

    def test_cpp_specials_is_lazy(self):
        result = _run_python(
            "import blond.core.backends.cpp.callables as c;"
            "print('CppSpecials' in vars(c));"
            "print(c.CppSpecials.__name__)"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(
            result.stdout.split(),
            ["False", "CppSpecials"],
            msg="the C++ library must only be loaded on first access",
        )


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
            float_=np.float64, complex_=np.complex128
        )

    @pytest.mark.backend_mutation
    def test_set_specials(self) -> None:
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        self.cupy_backend = CupyBackend(
            float_=np.float64, complex_=np.complex128
        )
        self.cupy_backend.set_specials(mode="cuda")

    @pytest.mark.backend_mutation
    def test_set_specials_fails(self):
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        self.cupy_backend = CupyBackend(
            float_=np.float64, complex_=np.complex128
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
            float_=np.float64,
            complex_=np.complex128,
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
        self.original_backend = type(backend)
        self.original_backend_specials_mode = backend.specials_mode

    def tearDown(self) -> None:
        backend.change_backend(self.original_backend)
        backend.set_specials(self.original_backend_specials_mode)

    @pytest.mark.backend_mutation
    def _setUp(self, dtype, special_mode) -> None:
        if special_mode in (
            "python",
            "cpp",
            "cpp_single_core",
            "numba",
        ):
            if dtype == np.float32:
                raise TypeError("32 Bit backends have been removed")
            else:
                backend.change_backend(Numpy64Bit)
        elif special_mode in ("cuda",):
            if dtype == np.float32:
                raise TypeError("32 Bit backends have been removed")
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
            raise TypeError("32 bit backends have been removed.")
        elif backend.float == np.float64:
            self.rtol = 1e-12
        else:
            raise ValueError(backend.float)

    @pytest.mark.backend_mutation
    def test___init__(self):
        pass

    @pytest.mark.backend_mutation
    def test_drift_exact(self) -> None:
        dtype = np.float64
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
        dtype = np.float64
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
        dtype = np.float64
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
                if backend.float == np.float32:
                    raise TypeError("32 bit backends have been removed.")

                np.testing.assert_allclose(
                    result,
                    result_python,
                    rtol=1e-12,
                    err_msg=f"Failed test `{special}` with {dtype}",
                )

    @pytest.mark.backend_mutation
    def test_kick_multi_harmonic(self) -> None:
        dtype = np.float64
        for n_voltages in (1, 2, 3, 4, 5):
            for i, special in enumerate(self.special_modes):
                self.n_voltages = n_voltages
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
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
    def test_apply_synchrotron_radiation_and_quantum_excitation_energy_kick_zero_macroparticles(
        self,
    ) -> None:
        """Empty beam must be a no-op (no errors, no allocation surprises)."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            beam_dE = backend.zeros(0, dtype=backend.float)
            backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
                beam_dE=beam_dE,
                energy_lost=13e6,
                longitudinal_damping_time=14955,
                natural_energy_spread=1e-3,
                total_energy=20e9,
                disable_quantum_excitation=False,
            )
            self.assertEqual(
                beam_dE.shape,
                (0,),
                msg=f"Failed `{special}` with {dtype}",
            )

    @pytest.mark.backend_mutation
    def test_apply_synchrotron_radiation_and_quantum_excitation_energy_kick_noise_statistics(
        self,
    ) -> None:
        """With QE on, mean ≈ damping result, std ≈ noise_scale ($1\\sigma$ check)."""
        # Use a large beam so the sample stats converge tightly.
        dtype = np.float64
        n_macroparticles = 200_000
        initial_dE = 20e9
        energy_lost = 13e6
        longitudinal_damping_time = 14955.0
        natural_energy_spread = 1e-3
        total_energy = 20e9
        expected_mean = (
            1.0 - 2.0 / longitudinal_damping_time
        ) * initial_dE - energy_lost
        expected_std = (
            2.0
            * natural_energy_spread
            / np.sqrt(longitudinal_damping_time)
            * total_energy
        )
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            beam_dE = backend.array(
                initial_dE * np.ones(n_macroparticles, dtype=dtype),
                dtype=backend.float,
            )
            backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
                beam_dE=beam_dE,
                energy_lost=energy_lost,
                longitudinal_damping_time=longitudinal_damping_time,
                natural_energy_spread=natural_energy_spread,
                total_energy=total_energy,
                disable_quantum_excitation=False,
            )
            dE_after_kick = copy_to_cpu(beam_dE)
            sample_mean = float(dE_after_kick.mean())
            sample_std = float(dE_after_kick.std())
            # 5σ confidence at n=200k → mean error tol ~ 5·σ/√n ≈ 5·1.6e6/√2e5 ≈ 1.8e4
            # Use a looser tol to keep flakes negligible across backends/RNGs.
            self.assertAlmostEqual(
                sample_mean,
                expected_mean,
                delta=max(1e-4 * abs(expected_mean), 5e4),
                msg=(
                    f"`{special}`: sample mean {sample_mean:.4e} far from "
                    f"expected {expected_mean:.4e}"
                ),
            )
            self.assertAlmostEqual(
                sample_std / expected_std,
                1.0,
                delta=0.02,  # within 2% of true σ
                msg=(
                    f"`{special}`: sample std {sample_std:.4e} vs expected "
                    f"{expected_std:.4e}"
                ),
            )

    @pytest.mark.backend_mutation
    def test_apply_synchrotron_radiation_and_quantum_excitation_energy_kick_disable_qe_is_noiseless(
        self,
    ) -> None:
        """When QE is disabled, two consecutive calls must give the same delta."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            kick_kwargs = dict(
                energy_lost=13e6,
                longitudinal_damping_time=14955,
                natural_energy_spread=1e-3,
                total_energy=20e9,
                disable_quantum_excitation=True,
            )
            beam_dE_first_call = backend.array(
                20e9 * np.ones(1000, dtype=dtype), dtype=backend.float
            )
            beam_dE_second_call = backend.array(
                20e9 * np.ones(1000, dtype=dtype), dtype=backend.float
            )
            backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
                beam_dE=beam_dE_first_call, **kick_kwargs
            )
            backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
                beam_dE=beam_dE_second_call, **kick_kwargs
            )
            dE_first_call = copy_to_cpu(beam_dE_first_call)
            dE_second_call = copy_to_cpu(beam_dE_second_call)
            np.testing.assert_array_equal(
                dE_first_call,
                dE_second_call,
                err_msg=f"`{special}` produced non-deterministic output",
            )

    @pytest.mark.backend_mutation
    def test_apply_synchrotron_radiation_and_quantum_excitation_energy_kick_qe_adds_variance(
        self,
    ) -> None:
        """With QE on, two calls with same scalar inputs must differ (noise)."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            kick_kwargs = dict(
                energy_lost=13e6,
                longitudinal_damping_time=14955,
                natural_energy_spread=1e-3,
                total_energy=20e9,
                disable_quantum_excitation=False,
            )
            beam_dE_first_call = backend.array(
                20e9 * np.ones(1000, dtype=dtype), dtype=backend.float
            )
            beam_dE_second_call = backend.array(
                20e9 * np.ones(1000, dtype=dtype), dtype=backend.float
            )
            backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
                beam_dE=beam_dE_first_call, **kick_kwargs
            )
            backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
                beam_dE=beam_dE_second_call, **kick_kwargs
            )
            dE_first_call = copy_to_cpu(beam_dE_first_call)
            dE_second_call = copy_to_cpu(beam_dE_second_call)
            self.assertFalse(
                np.array_equal(dE_first_call, dE_second_call),
                msg=(
                    f"`{special}`: two QE-enabled calls returned identical "
                    f"output — noise term not active"
                ),
            )

    @pytest.mark.backend_mutation
    def test_apply_synchrotron_radiation_and_quantum_excitation_energy_kick_deterministic(
        self,
    ) -> None:
        """Disable QE → result is exactly ``(1 - 2/τ) * beam_dE - energy_lost``."""
        dtype = np.float64
        energy_lost = 13e6
        longitudinal_damping_time = 14955.0
        total_energy = 20e9
        initial_dE = 20e9
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            beam_dE = backend.array(
                initial_dE * np.ones(1000, dtype=dtype), dtype=backend.float
            )
            backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
                beam_dE=beam_dE,
                energy_lost=energy_lost,
                longitudinal_damping_time=longitudinal_damping_time,
                natural_energy_spread=1e-3,
                total_energy=total_energy,
                disable_quantum_excitation=True,
            )
            expected_dE = (
                1.0 - 2.0 / longitudinal_damping_time
            ) * initial_dE - energy_lost
            dE_after_kick = copy_to_cpu(beam_dE)
            np.testing.assert_allclose(
                np.asarray(dE_after_kick),
                expected_dE * np.ones(1000, dtype=dtype),
                rtol=self.rtol,
                err_msg=f"Failed test `{special}` with {dtype}",
            )

    @pytest.mark.backend_mutation
    def test_apply_synchrotron_radiation_and_quantum_excitation_energy_kick_inplace(
        self,
    ) -> None:
        """The kick must mutate ``beam_dE`` in place — same object, not a copy."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            beam_dE = backend.array(
                20e9 * np.ones(100, dtype=dtype), dtype=backend.float
            )
            id_before = id(beam_dE)
            backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
                beam_dE=beam_dE,
                energy_lost=13e6,
                longitudinal_damping_time=14955,
                natural_energy_spread=1e-3,
                total_energy=20e9,
                disable_quantum_excitation=False,
            )
            self.assertEqual(
                id_before,
                id(beam_dE),
                msg=f"Inplace contract violated for `{special}`",
            )
            # Value must have actually changed (damping + noise).
            dE_after_kick = copy_to_cpu(beam_dE)
            self.assertFalse(
                np.allclose(
                    np.asarray(dE_after_kick),
                    20e9 * np.ones(100, dtype=dtype),
                ),
                msg=f"`{special}` did not modify beam_dE",
            )

    @pytest.mark.backend_mutation
    def test_kick_single_harmonic(self) -> None:
        dtype = np.float64
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
                if backend.float == np.float32:
                    raise TypeError("32 bit backends have been removed.")

                np.testing.assert_allclose(
                    result,
                    result_python,
                    rtol=1e-12,
                    err_msg=f"Failed test `{special}` with {dtype}",
                )

    @pytest.mark.backend_mutation
    def test_kick_interpolated(self) -> None:
        dtype = np.float64
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
        dtype = np.float64
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
    def test_kick_interpolated_far_outside_window(self) -> None:
        """Particles far outside the window must not receive any kick.

        The C++ kernel converted ``floor(...)`` of the bin index to
        ``unsigned``, which is undefined behaviour for negative values: on
        x86 it happens to produce a huge value that is skipped, but e.g. on
        ARM the conversion saturates to 0 and such particles would wrongly
        receive the kick of bin 0.
        """
        dtype = np.float64
        dt_np = np.array(
            [-1e30, -1e12, -4.5, 0.0, 4.5, 1e12, 1e30], dtype=dtype
        )
        in_range = np.zeros_like(dt_np, dtype=bool)
        in_range[3] = True  # only dt = 0.0 is inside bin_centers [-4, 4]
        for i, special in enumerate(self.special_modes):
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            dt = backend.array(dt_np, dtype=backend.float)
            dE = backend.zeros_like(dt, dtype=backend.float)
            bin_centers = backend.linspace(-4, 4, 20, dtype=backend.float)
            voltage = bin_centers**2
            backend.specials.kick_interpolated(
                dt=dt,
                dE=dE,
                voltage=voltage,
                bin_centers=bin_centers,
                charge=backend.float(10),
                acceleration_kick=backend.float(0.5),
            )
            result = dE
            if special == "cuda":
                result = result.get()
            result = np.asarray(result)
            np.testing.assert_array_equal(
                result[~in_range],
                0.0,
                err_msg=(
                    f"out-of-window particles must not be kicked, "
                    f"{special=} {dtype=}"
                ),
            )
            self.assertNotEqual(
                result[3],
                0.0,
                msg=f"in-window particle must be kicked, {special=}",
            )
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
    def test_histogram_extreme_outliers(self) -> None:
        """Histogram must ignore values of extreme magnitude.

        Bin indices of such values overflow ``int``; the conversion is
        undefined behaviour in C++ and must not be relied on. Also pins
        the edge semantics: ``== start`` is counted in the first bin,
        ``== stop`` in the last bin.
        """
        dtype = np.float64
        values_np = np.array(
            [-1e30, -1e12, -12.0, 0.0, 8.0, 1e12, 1e30], dtype=dtype
        )
        n_bins = 21
        expected = np.zeros(n_bins, dtype=dtype)
        expected[0] += 1  # -12.0 == start
        expected[int((0.0 - -12.0) / 20.0 * n_bins)] += 1  # 0.0
        expected[-1] += 1  # 8.0 == stop
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            array_write = backend.ones(n_bins, dtype=backend.float)
            backend.specials.histogram(
                array_read=backend.array(values_np, dtype=backend.float),
                array_write=array_write,
                start=backend.float(-12),
                stop=backend.float(8.0),
            )
            result = array_write
            if special == "cuda":
                result = result.get()
            np.testing.assert_array_equal(
                np.asarray(result),
                expected,
                err_msg=f"{special=} {dtype=}",
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
        dtype = np.float64
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
                    **allclose_tolerances(result_python, 1e-6),
                    # FIXME
                    #  this tolerance is so low because of the GPU
                    #  backend. Reason unknown for now.
                    err_msg=f"Failed test `{special}` with {dtype}",
                )

    @pytest.mark.backend_mutation
    def test_move_flagged_elements_to_end(self):
        dtype = np.float64
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
            self.assertEqual(n_new, 10 - 3)
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
                result_n_python = n_new
            else:
                self.assertEqual(
                    n_new,
                    result_n_python,
                    msg=f"Failed test `{special}` with {dtype}",
                )

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
        dtype = np.float64
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
        dtype = np.float64
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
        dtype = np.float64
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
        dtype = np.float64
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
        dtype = np.float64
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
        dtype = np.float64
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
                if backend.float == np.float32:
                    raise TypeError("32 bit backends have been removed.")
                np.testing.assert_allclose(
                    result,
                    result_python,
                    rtol=self.rtol,
                    err_msg=f"Failed test `{special}` with {dtype}",
                )

    @pytest.mark.backend_mutation
    def test_histogram(self) -> None:
        dtype = np.float64
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
        dtype = np.float64
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
        dtype = np.float64
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
    def test_histogram_sparse_outside_edges(self) -> None:
        """Particles slightly outside the profile windows must not be counted.

        Regression test: the numba backend truncated negative float bin
        indices toward zero (``int(-0.5) == 0``), so a particle up to one
        bin width left of ``first_left_cut`` was counted into bin 0 of the
        first profile.
        """
        dtype = np.float64
        bins_per_profile = 4  # bin width = cut_width / 4 = 1
        first_left_cut = -12
        left_cut_distance = 8
        cut_width = 4
        # filled buckets 0, 2, 4 -> windows [-12,-8], [4,8], [20,24]
        filling_pattern_np = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
        bucket_index_to_memory_index_np = np.array(
            [0, 0, 4, 4, 8, 8], dtype=np.int32
        )
        particles_np = np.array(
            [
                # outside any window, must never be counted
                -12.5,  # < one bin left of first cut (numba truncation bug)
                -15.9,  # < one bucket distance left of first cut
                -20.5,  # more than one bucket distance left of first cut
                8.5,  # just right of window [4,8], in a gap
                24.5,  # just right of last window [20,24]
                # inside windows, must be counted
                -11.5,  # profile 0, bin 0
                5.5,  # profile 1, bin 1
                23.9,  # profile 2, bin 3
            ],
            dtype=dtype,
        )
        expected = np.zeros(12, dtype=dtype)
        expected[0] = 1  # -11.5
        expected[4 + 1] = 1  # 5.5
        expected[8 + 3] = 1  # 23.9

        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            particles_x = backend.array(particles_np, dtype=backend.float)
            filling_pattern = backend.array(filling_pattern_np, dtype=bool)
            bucket_index_to_memory_index = backend.array(
                bucket_index_to_memory_index_np, dtype=np.int32
            )
            # initialised to ones to verify the output is zeroed
            array_write = backend.ones(12, dtype=backend.float)
            for _ in range(2):
                backend.specials.histogram_sparse(
                    x=particles_x,
                    out=array_write,
                    first_left_cut=first_left_cut,
                    left_cut_distance=left_cut_distance,
                    cut_width=cut_width,
                    bins_per_profile=bins_per_profile,
                    n_active_profiles=3,
                    filling_pattern=filling_pattern,
                    bucket_index_to_memory_index=bucket_index_to_memory_index,
                )
            result = array_write
            if special == "cuda":
                result = result.get()
            np.testing.assert_array_equal(
                result,
                expected,
                err_msg=f"{special=} {dtype=}",
            )

    @pytest.mark.backend_mutation
    def test_histogram_long_profiles(self) -> None:
        """Specifically to test edge effects at beginning and end."""
        dtype = np.float64
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
        dtype = np.float64
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
        np.random.seed(np.uint(42))
        array_read = (
            np.random.random_sample(size=1024) - 0.5
        ) * 20  # common sample data from -10 to 10
        dtype = np.float64
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

    def _run_wake_from_pole_residue(
        self,
        update_on_bin_np: np.ndarray,
        n_calls: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run `wake_from_pole_residue` on the active backend.

        Returns
        -------
        voltage, states
            Output voltage and pole states as numpy arrays.
        """
        n = 16
        rng = np.random.default_rng(42)
        profile_np = rng.standard_normal(n)
        centers_np = np.linspace(0.0, 1e-9, n)
        bin_dt = centers_np[1] - centers_np[0]
        poles_np = np.array(
            [-1e8 + 2 * np.pi * 1e9j, -2e8 + 2 * np.pi * 1.5e9j],
            dtype=complex,
        )
        residues_np = np.array([1.0 + 0.5j, 0.7 - 0.2j], dtype=complex)

        profile = backend.array(profile_np, dtype=backend.float)
        centers = backend.array(centers_np, dtype=backend.float)
        poles = backend.array(poles_np, dtype=backend.complex)
        residues = backend.array(residues_np, dtype=backend.complex)
        states = backend.zeros(len(poles_np) + 1, dtype=backend.complex)
        # non-zero state so decay handling is observable in the output
        states[0] = 0.3 + 0.1j
        states[-1] = centers_np[0] - bin_dt
        voltage = backend.zeros(n, dtype=backend.float)
        for _ in range(n_calls):
            backend.specials.wake_from_pole_residue(
                profile=profile,
                profile_dts=centers,
                poles=poles,
                residues=residues,
                is_counterrotating_beam=False,
                counterrotating_pole_signs=backend.ones_like(
                    poles, dtype=backend.float
                ),
                update_on_bin=backend.array(update_on_bin_np, dtype=np.int32),
                factor=1.0,
                states=states,
                voltage=voltage,
                voltage_threaded=backend.zeros(
                    (backend.specials.get_max_threads(), n),
                    dtype=backend.float,
                ),
            )
        if backend.is_gpu:
            return voltage.get(), states.get()
        return np.asarray(voltage), np.asarray(states)

    def _assert_wake_matches_python(
        self, update_on_bin_np: np.ndarray, n_calls: int = 1
    ) -> None:
        dtype = np.float64
        for i, special in enumerate(self.special_modes):
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            voltage, states = self._run_wake_from_pole_residue(
                update_on_bin_np=update_on_bin_np,
                n_calls=n_calls,
            )
            if i == 0:
                voltage_python = voltage
                states_python = states
            else:
                np.testing.assert_allclose(
                    voltage,
                    voltage_python,
                    rtol=1e-10,
                    err_msg=f"{special=} {dtype=}",
                )
                np.testing.assert_allclose(
                    states,
                    states_python,
                    rtol=1e-10,
                    err_msg=f"{special=} {dtype=}",
                )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue(self) -> None:
        """All backends must match python, incl. states over two calls."""
        self._assert_wake_matches_python(
            update_on_bin_np=np.array([0], dtype=np.int32),
            n_calls=2,
        )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_update_not_on_first_bin(self) -> None:
        """`update_on_bin[0] != 0` must behave like python (`decay = 0`).

        Before the first update bin, ``state *= decay`` with ``decay = 0``
        zeroes the state. The numba kernel did not initialise ``decay`` and
        relied on implicit zero-initialisation of maybe-undefined variables.
        """
        self._assert_wake_matches_python(
            update_on_bin_np=np.array([2], dtype=np.int32),
        )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_empty_update_on_bin(self) -> None:
        """Empty `update_on_bin` must not read out of bounds.

        The C++ and CUDA kernels treat an empty array as "never update"
        (``decay`` stays 0). The python reference raised ``IndexError`` and
        the numba kernel read ``update_on_bin[0]`` out of bounds.
        """
        self._assert_wake_matches_python(
            update_on_bin_np=np.array([], dtype=np.int32),
        )

    @pytest.mark.backend_mutation
    def test_histogram_weighted_uniform_weights_equals_unweighted(
        self,
    ) -> None:
        """With all weights == 1, weighted and unweighted histograms must agree."""
        for dtype in (np.float64,):
            for special in self.special_modes:
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                array_read = backend.linspace(-5, 5, 51, dtype=backend.float)
                weights = backend.ones(51, dtype=backend.float)
                array_write_unweighted = backend.zeros(21, dtype=backend.float)
                array_write_weighted = backend.zeros(21, dtype=backend.float)
                backend.specials.histogram(
                    array_read=array_read,
                    array_write=array_write_unweighted,
                    start=backend.float(-10),
                    stop=backend.float(10),
                )
                print(f"{special=} {dtype=}")
                backend.specials.histogram_weighted(
                    array_read=array_read,
                    array_write=array_write_weighted,
                    weights=weights,
                    start=backend.float(-10),
                    stop=backend.float(10),
                )
                print(f"{special=} {dtype=} sucessful")
                if special == "cuda":
                    array_write_unweighted = array_write_unweighted.get()
                    array_write_weighted = array_write_weighted.get()
                np.testing.assert_allclose(
                    array_write_weighted,
                    array_write_unweighted,
                    rtol=self.rtol,
                    err_msg=f"{special=} {dtype=}",
                )

    @pytest.mark.backend_mutation
    def test_histogram_weighted_correctness(self) -> None:
        """Weighted histogram values match numpy's reference implementation."""
        np.random.seed(42)
        array_read_np = np.random.uniform(-5, 5, 200)
        weights_np = np.random.uniform(0.5, 1.5, 200)
        # Reference computed at float64 precision
        expected_f64, _ = np.histogram(
            array_read_np,
            bins=20,
            range=(-6.0, 6.0),
            weights=weights_np,
        )
        for dtype in (np.float64,):
            for special in self.special_modes:
                try:
                    self._setUp(dtype=dtype, special_mode=special)
                except (FileNotFoundError, OSError):
                    print(f"Could not perform `{special}` test for {dtype}")
                    continue
                array_write = backend.zeros(20, dtype=backend.float)
                backend.specials.histogram_weighted(
                    array_read=backend.array(
                        array_read_np, dtype=backend.float
                    ),
                    array_write=array_write,
                    weights=backend.array(weights_np, dtype=backend.float),
                    start=backend.float(-6.0),
                    stop=backend.float(6.0),
                )
                result = array_write
                if special == "cuda":
                    result = result.get()
                np.testing.assert_allclose(
                    result,
                    expected_f64,
                    # float32 accumulates rounding error; 1e-5 matches the
                    # tolerance used for other float32 backend comparisons
                    rtol=1e-5 if dtype == np.float32 else 1e-10,
                    err_msg=f"{special=} {dtype=}",
                )

    @multi_backend_testcase("Numpy64Bit")
    @pytest.mark.backend_mutation
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
    @pytest.mark.backend_mutation
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

    @multi_backend_testcase("Numpy64Bit")
    @pytest.mark.backend_mutation
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
    @pytest.mark.backend_mutation
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

    @pytest.mark.backend_mutation
    def test_sum_1d_array(self) -> None:
        dtype = np.float64
        x = np.random.rand(10_000).astype(dtype)
        reference_sum = np.sum(x)
        for i, special in enumerate(self.special_modes):
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            np.testing.assert_allclose(
                copy_to_cpu(backend.specials.sum_1d_array(backend.array(x))),
                reference_sum,
                # Cumulative error is different with and without reduction, causing problems with single-core-cpp.
                rtol=self.rtol,
                err_msg=f"{special=} {dtype=}",
            )

    @pytest.mark.backend_mutation
    def test_dot_product_1d_array(self) -> None:
        dtype = np.float64
        x = np.random.rand(10_000).astype(dtype)
        y = np.random.rand(10_000).astype(dtype)
        reference_dot = np.dot(x, y)
        for i, special in enumerate(self.special_modes):
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            backend_result = backend.specials.dot_product_1d_array(
                backend.array(x),
                backend.array(y),
            )
            backend_result = copy_to_cpu(backend_result)

            if backend.float == np.float32:
                raise TypeError("32 bit backends have been removed.")

            np.testing.assert_allclose(
                backend_result,
                reference_dot,
                # Cumulative error is different with and without reduction, causing problems with single-core-cpp.
                rtol=self.rtol,
                err_msg=f"{special=} {dtype=}",
            )
            self.assertTrue(backend_result.dtype == dtype)

    @pytest.mark.backend_mutation
    def test_drift_exact_zero_macroparticles(self) -> None:
        """`drift_exact` must be a no-op (no errors) on empty dt/dE arrays."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            dt = backend.zeros(0, dtype=backend.float)
            dE = backend.zeros(0, dtype=backend.float)
            backend.specials.drift_exact(
                dt=dt,
                dE=dE,
                T=self.t_rev * self.length_ratio,
                alpha_0=self.alpha_0,
                higher_alpha=backend.array([1.0, 2.0], dtype=dtype),
                beta=self.beta,
                energy=self.energy,
            )
            self.assertEqual(
                dt.shape, (0,), msg=f"Failed `{special}` with {dtype}"
            )
            self.assertEqual(
                dE.shape, (0,), msg=f"Failed `{special}` with {dtype}"
            )

    @pytest.mark.backend_mutation
    def test_drift_simple_zero_macroparticles(self) -> None:
        """`drift_simple` must be a no-op (no errors) on empty dt/dE arrays."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            dt = backend.zeros(0, dtype=backend.float)
            dE = backend.zeros(0, dtype=backend.float)
            backend.specials.drift_simple(
                dt=dt,
                dE=dE,
                T=self.t_rev * self.length_ratio,
                eta_0=self.eta_0,
                beta=self.beta,
                energy=self.energy,
            )
            self.assertEqual(
                dt.shape, (0,), msg=f"Failed `{special}` with {dtype}"
            )
            self.assertEqual(
                dE.shape, (0,), msg=f"Failed `{special}` with {dtype}"
            )

    @pytest.mark.backend_mutation
    def test_kick_single_harmonic_zero_macroparticles(self) -> None:
        """`kick_single_harmonic` must be a no-op on empty dt/dE arrays."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            dt = backend.zeros(0, dtype=backend.float)
            dE = backend.zeros(0, dtype=backend.float)
            backend.specials.kick_single_harmonic(
                dt=dt,
                dE=dE,
                voltage=self.voltage_single_harmonic,
                omega_rf=self.omega_rf_single_harmonic,
                phi_rf=self.phi_rf_single_harmonic,
                charge=self.charge,
                acceleration_kick=self.acceleration_kick,
            )
            self.assertEqual(
                dt.shape, (0,), msg=f"Failed `{special}` with {dtype}"
            )
            self.assertEqual(
                dE.shape, (0,), msg=f"Failed `{special}` with {dtype}"
            )

    @pytest.mark.backend_mutation
    def test_kick_multi_harmonic_zero_macroparticles(self) -> None:
        """`kick_multi_harmonic` must be a no-op on empty dt/dE arrays."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            dt = backend.zeros(0, dtype=backend.float)
            dE = backend.zeros(0, dtype=backend.float)
            backend.specials.kick_multi_harmonic(
                dt=dt,
                dE=dE,
                voltage=self.voltages,
                omega_rf=self.omegas,
                phi_rf=self.phis,
                charge=self.charge,
                n_rf=len(self.voltages),
                acceleration_kick=self.acceleration_kick,
            )
            self.assertEqual(
                dt.shape, (0,), msg=f"Failed `{special}` with {dtype}"
            )
            self.assertEqual(
                dE.shape, (0,), msg=f"Failed `{special}` with {dtype}"
            )

    @pytest.mark.backend_mutation
    def test_kick_multi_harmonic_zero_n_rf(self) -> None:
        """`n_rf=0` with empty rf arrays: only `acceleration_kick` survives."""
        dtype = np.float64
        # Reference result via the python backend (i = 0); compare other modes against it.
        for i, special in enumerate(self.special_modes):
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            dt = backend.linspace(1e-9, 10e-9, 10, dtype=backend.float)
            dE = backend.zeros(10, dtype=backend.float)
            empty_voltage = backend.zeros(0, dtype=backend.float)
            empty_omega = backend.zeros(0, dtype=backend.float)
            empty_phi = backend.zeros(0, dtype=backend.float)
            backend.specials.kick_multi_harmonic(
                dt=dt,
                dE=dE,
                voltage=empty_voltage,
                omega_rf=empty_omega,
                phi_rf=empty_phi,
                charge=self.charge,
                n_rf=0,
                acceleration_kick=self.acceleration_kick,
            )
            result = dE
            if special == "cuda":
                result = result.get()
            # Without any rf harmonic, every particle receives exactly
            # `acceleration_kick`.
            expected = np.full(
                10, float(self.acceleration_kick), dtype=np.float64
            )
            np.testing.assert_allclose(
                np.asarray(result),
                expected,
                rtol=self.rtol,
                err_msg=f"Failed test `{special}` with {dtype}",
            )
            if i == 0:
                result_python = np.asarray(result).copy()
            else:
                np.testing.assert_allclose(
                    np.asarray(result),
                    result_python,
                    rtol=self.rtol,
                    err_msg=f"Failed test `{special}` with {dtype}",
                )

    @pytest.mark.backend_mutation
    def test_kick_interpolated_zero_macroparticles(self) -> None:
        """`kick_interpolated` must be a no-op on empty dt/dE arrays."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            dt = backend.zeros(0, dtype=backend.float)
            dE = backend.zeros(0, dtype=backend.float)
            bin_centers = backend.linspace(-4, 4, 20, dtype=backend.float)
            voltage = bin_centers**2
            backend.specials.kick_interpolated(
                dt=dt,
                dE=dE,
                voltage=voltage,
                bin_centers=bin_centers,
                charge=backend.float(10),
                acceleration_kick=backend.float(0.5),
            )
            self.assertEqual(
                dt.shape, (0,), msg=f"Failed `{special}` with {dtype}"
            )
            self.assertEqual(
                dE.shape, (0,), msg=f"Failed `{special}` with {dtype}"
            )

    @pytest.mark.backend_mutation
    def test_loss_box_zero_macroparticles(self) -> None:
        """`loss_box` must be a no-op on empty dt/dE/flags arrays."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            dt = backend.zeros(0, dtype=backend.float)
            dE = backend.zeros(0, dtype=backend.float)
            flags = backend.zeros(0, dtype=np.int32)
            backend.specials.loss_box(
                e_max=backend.float(1),
                e_min=backend.float(-1),
                t_min=backend.float(-10),
                t_max=backend.float(10),
                dt=dt,
                dE=dE,
                flags=flags,
            )
            self.assertEqual(
                flags.shape, (0,), msg=f"Failed `{special}` with {dtype}"
            )

    @pytest.mark.backend_mutation
    def test_move_flagged_elements_to_end_post_loop_branches(self) -> None:
        """Exercise both post-loop return paths of `move_flagged_elements_to_end`.

        The C++ implementation ends with::

            if (i < n_macroparticles && flags[i] == flag) {
                return i;     // boundary element matches `flag`
            }
            return i + 1;     // boundary element does NOT match `flag`

        Both branches must produce the correct partition count, where the
        return value equals the number of leading non-flagged elements
        (equivalently, the index of the first flagged element).

        Scenarios are tuples of (flags, expected_n_new, branch) with `flag=0`:
        - ([0])          n=1, all flagged     → `return i`   → 0
        - ([1])          n=1, all non-flagged → `return i+1` → 1
        - ([0, 0, 0])    all flagged          → `return i`   → 0
        - ([1, 1, 1])    all non-flagged      → `return i+1` → 3
        - ([1, 1, 0])    boundary at end      → `return i`   → 2
        - ([1, 0, 1, 0]) interleaved          → `return i`   → 2
        """
        scenarios = [
            ([0], 0, "return i"),
            ([1], 1, "return i+1"),
            ([0, 0, 0], 0, "return i"),
            ([1, 1, 1], 3, "return i+1"),
            ([1, 1, 0], 2, "return i"),
            ([1, 0, 1, 0], 2, "return i"),
        ]
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            for flags_init, expected_n_new, branch in scenarios:
                n = len(flags_init)
                flags = backend.array(flags_init, dtype=np.int32)
                dt = backend.array(
                    backend.linspace(0, 1, n), dtype=backend.float
                )
                dE = backend.array(
                    backend.linspace(0, 1, n), dtype=backend.float
                )
                ids = backend.arange(0, n, dtype=np.int32)
                n_new = int(
                    backend.specials.move_flagged_elements_to_end(
                        flag=0,
                        flags=flags,
                        dt=dt,
                        dE=dE,
                        ids=ids,
                    )
                )
                self.assertEqual(
                    expected_n_new,
                    n_new,
                    msg=(
                        f"Failed `{special}` with {dtype}: "
                        f"flags={flags_init}, expected branch `{branch}` "
                        f"→ {expected_n_new}, got {n_new}"
                    ),
                )
                # Post-condition: leading `n_new` entries are non-flagged,
                # trailing entries are flagged — what the return value means.
                flags_after = (
                    flags.get() if special == "cuda" else np.asarray(flags)
                )
                self.assertTrue(
                    bool(np.all(flags_after[:n_new] != 0)),
                    msg=(
                        f"Leading slice not fully unflagged for `{special}` "
                        f"with flags_init={flags_init}: {flags_after}"
                    ),
                )
                self.assertTrue(
                    bool(np.all(flags_after[n_new:] == 0)),
                    msg=(
                        f"Trailing slice not fully flagged for `{special}` "
                        f"with flags_init={flags_init}: {flags_after}"
                    ),
                )

    @pytest.mark.backend_mutation
    def test_move_flagged_elements_to_end_zero_macroparticles(self) -> None:
        """`move_flagged_elements_to_end` on empty arrays must return 0.

        Collects results across all special modes before asserting so the
        failure report names every backend that mishandles the empty case
        rather than aborting on the first mismatch.
        """
        dtype = np.float64
        offenders: list[str] = []
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            flags = backend.zeros(0, dtype=np.int32)
            dt = backend.zeros(0, dtype=backend.float)
            dE = backend.zeros(0, dtype=backend.float)
            ids = backend.zeros(0, dtype=np.int32)
            n_new = backend.specials.move_flagged_elements_to_end(
                flag=0,
                flags=flags,
                dt=dt,
                dE=dE,
                ids=ids,
            )
            if int(n_new) != 0:
                offenders.append(f"{special} returned n_new={int(n_new)}")
        self.assertEqual(
            offenders,
            [],
            msg=(
                "move_flagged_elements_to_end must return 0 on empty arrays; "
                f"these backends did not: {offenders}"
            ),
        )

    @pytest.mark.backend_mutation
    def test_sum_1d_array_zero_macroparticles(self) -> None:
        """`sum_1d_array` of an empty array must equal 0."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            empty = backend.zeros(0, dtype=backend.float)
            result = copy_to_cpu(backend.specials.sum_1d_array(empty))
            np.testing.assert_allclose(
                float(result),
                0.0,
                atol=0.0,
                err_msg=f"Failed test `{special}` with {dtype}",
            )

    @pytest.mark.backend_mutation
    def test_dot_product_1d_array_zero_macroparticles(self) -> None:
        """`dot_product_1d_array` of two empty arrays must equal 0."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            empty_a = backend.zeros(0, dtype=backend.float)
            empty_b = backend.zeros(0, dtype=backend.float)
            result = copy_to_cpu(
                backend.specials.dot_product_1d_array(empty_a, empty_b)
            )
            np.testing.assert_allclose(
                float(result),
                0.0,
                atol=0.0,
                err_msg=f"Failed test `{special}` with {dtype}",
            )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue(self) -> None:
        """Cross-backend parity for `wake_from_pole_residue` voltage output.

        Scoped to float64: numba's kernel signature is hard-coded to
        ``complex128``, and the real caller in ``solvers.py`` always
        allocates ``np.zeros(.., complex)`` — i.e. complex128 — which makes
        ``float64`` the only precision all backends consistently accept.
        """
        import numba as _nb

        n_bins = 64
        n_poles = 3
        dt_val = 1e-9

        # Reference inputs; each backend builds its own arrays from these.
        profile_np = np.sin(np.linspace(0, 3 * np.pi, n_bins)) ** 2
        profile_dts_np = np.linspace(0, n_bins * dt_val, n_bins + 1)
        # Stable poles (Re < 0); decay magnitude per bin exp(Re*dt) in (0, 1).
        poles_np = np.array(
            [-1e8 + 1e9j, -2e8 + 5e8j, -3e8 + 2e9j],
            dtype=np.complex128,
        )
        residues_np = np.array(
            [1.0 + 0.5j, 0.5 - 1.0j, 0.3 + 0.7j],
            dtype=np.complex128,
        )
        update_on_bin_np = np.array([0], dtype=np.int32)

        dtype = np.float64
        for i, special in enumerate(self.special_modes):
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue

            profile = backend.array(profile_np, dtype=backend.float)
            profile_dts = backend.array(profile_dts_np, dtype=backend.float)
            poles = backend.array(poles_np, dtype=np.complex128)
            residues = backend.array(residues_np, dtype=np.complex128)
            cr_flags = backend.ones(n_poles, dtype=backend.float)
            states = backend.zeros(n_poles + 1, dtype=np.complex128)
            voltage = backend.zeros(n_bins, dtype=backend.float)
            voltage_threaded = backend.zeros(
                (_nb.get_num_threads(), n_bins), dtype=backend.float
            )
            update_on_bin = backend.array(update_on_bin_np, dtype=np.int32)

            backend.specials.wake_from_pole_residue(
                profile=profile,
                profile_dts=profile_dts,
                poles=poles,
                residues=residues,
                is_counterrotating_beam=False,
                counterrotating_pole_signs=cr_flags,
                states=states,
                voltage=voltage,
                voltage_threaded=voltage_threaded,
                update_on_bin=update_on_bin,
                factor=backend.float(1.0),
            )

            result = np.asarray(copy_to_cpu(voltage))

            if i == 0:
                result_reference = result
            else:
                np.testing.assert_allclose(
                    result,
                    result_reference,
                    rtol=1e-10,
                    err_msg=f"Failed test `{special}` with {dtype}",
                )

            result2 = np.asarray(copy_to_cpu(states))

            if i == 0:
                result2_reference = result2
            else:
                np.testing.assert_allclose(
                    result2,
                    result2_reference,
                    rtol=1e-10,
                    err_msg=f"Failed test `{special}` with {dtype}",
                )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_charge_counterrotation(self) -> None:
        """Cross-backend parity for `wake_from_pole_residue` voltage output.

        Scoped to float64: numba's kernel signature is hard-coded to
        ``complex128``, and the real caller in ``solvers.py`` always
        allocates ``np.zeros(.., complex)`` — i.e. complex128 — which makes
        ``float64`` the only precision all backends consistently accept.
        """
        import numba as _nb

        for charge in (-1, 1):
            for is_counterrotating_beam in (False, True):
                for cr_flags_sign in (-1, 1):
                    n_bins = 64
                    n_poles = 3
                    dt_val = 1e-9

                    # Reference inputs; each backend builds its own arrays from these.
                    profile_np = np.sin(np.linspace(0, 3 * np.pi, n_bins)) ** 2
                    profile_dts_np = np.linspace(
                        0, n_bins * dt_val, n_bins + 1
                    )
                    # Stable poles (Re < 0); decay magnitude per bin exp(Re*dt) in (0, 1).
                    poles_np = np.array(
                        [-1e8 + 1e9j, -2e8 + 5e8j, -3e8 + 2e9j],
                        dtype=np.complex128,
                    )
                    residues_np = np.array(
                        [1.0 + 0.5j, 0.5 - 1.0j, 0.3 + 0.7j],
                        dtype=np.complex128,
                    )
                    update_on_bin_np = np.array([0], dtype=np.int32)

                    dtype = np.float64
                    for i, special in enumerate(self.special_modes):
                        try:
                            self._setUp(dtype=dtype, special_mode=special)
                        except (FileNotFoundError, OSError):
                            print(
                                f"Could not perform `{special}` test for {dtype}"
                            )
                            continue

                        profile = backend.array(
                            profile_np, dtype=backend.float
                        )
                        profile_dts = backend.array(
                            profile_dts_np, dtype=backend.float
                        )
                        poles = backend.array(poles_np, dtype=np.complex128)
                        residues = backend.array(
                            residues_np, dtype=np.complex128
                        )
                        cr_flags = backend.ones(n_poles, dtype=backend.float)
                        cr_flags[-1] *= cr_flags_sign
                        states = backend.zeros(
                            n_poles + 1, dtype=np.complex128
                        )
                        voltage = backend.zeros(n_bins, dtype=backend.float)
                        voltage_threaded = backend.zeros(
                            (_nb.get_num_threads(), n_bins),
                            dtype=backend.float,
                        )
                        update_on_bin = backend.array(
                            update_on_bin_np, dtype=np.int32
                        )

                        backend.specials.wake_from_pole_residue(
                            profile=profile,
                            profile_dts=profile_dts,
                            poles=poles,
                            residues=residues,
                            is_counterrotating_beam=is_counterrotating_beam,
                            counterrotating_pole_signs=cr_flags,
                            states=states,
                            voltage=voltage,
                            voltage_threaded=voltage_threaded,
                            update_on_bin=update_on_bin,
                            factor=backend.float(charge * 1.0),
                        )

                        result = np.asarray(copy_to_cpu(voltage))

                        if i == 0:
                            result_reference = result
                        else:
                            np.testing.assert_allclose(
                                result,
                                result_reference,
                                rtol=1e-10,
                                err_msg=f"Failed test `{special}` with {dtype}",
                            )
                        result2 = np.asarray(copy_to_cpu(states))

                        if i == 0:
                            result2_reference = result2
                        else:
                            np.testing.assert_allclose(
                                result2,
                                result2_reference,
                                rtol=1e-10,
                                err_msg=f"Failed test `{special}` with {dtype}",
                            )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_cr_flip_invariance(self) -> None:
        """Voltage is invariant under ``cr_pole_flip`` sign flips.

        For a flipped pole the internal state picks up an overall ``-1``
        by induction, but the output voltage multiplies the state by that
        same ``cr_flip`` — the two sign flips cancel in ``Re(res * state)``.
        Starting from zero state, the per-backend voltage must therefore
        be identical with and without flipped poles.
        """
        import numba as _nb

        n_bins = 64
        n_poles = 3
        dt_val = 1e-9

        profile_np = np.sin(np.linspace(0, 3 * np.pi, n_bins)) ** 2
        profile_dts_np = np.linspace(0, n_bins * dt_val, n_bins + 1)
        poles_np = np.array(
            [-1e8 + 1e9j, -2e8 + 5e8j, -3e8 + 2e9j],
            dtype=np.complex128,
        )
        residues_np = np.array(
            [1.0 + 0.5j, 0.5 - 1.0j, 0.3 + 0.7j],
            dtype=np.complex128,
        )
        update_on_bin_np = np.array([0], dtype=np.int32)
        flipped_signs_np = np.array([1.0, -1.0, 1.0])

        def _run(flag: bool, flags_np: np.ndarray) -> np.ndarray:
            profile = backend.array(profile_np, dtype=backend.float)
            profile_dts = backend.array(profile_dts_np, dtype=backend.float)
            poles = backend.array(poles_np, dtype=np.complex128)
            residues = backend.array(residues_np, dtype=np.complex128)
            cr_flags = backend.array(flags_np, dtype=backend.float)
            states = backend.zeros(n_poles + 1, dtype=np.complex128)
            voltage = backend.zeros(n_bins, dtype=backend.float)
            voltage_threaded = backend.zeros(
                (_nb.get_num_threads(), n_bins), dtype=backend.float
            )
            update_on_bin = backend.array(update_on_bin_np, dtype=np.int32)

            # Positional args: see note in `test_wake_from_pole_residue`.
            backend.specials.wake_from_pole_residue(
                profile,
                profile_dts,
                poles,
                residues,
                flag,
                cr_flags,
                update_on_bin,
                backend.float(1.0),
                states,
                voltage,
                voltage_threaded,
            )
            return np.asarray(copy_to_cpu(voltage)).copy()

        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue

            voltage_baseline = _run(flag=False, flags_np=np.ones(n_poles))
            voltage_flipped = _run(flag=True, flags_np=flipped_signs_np)

            np.testing.assert_allclose(
                voltage_flipped,
                voltage_baseline,
                rtol=1e-10,
                err_msg=(
                    "cr_pole_flip must leave voltage invariant "
                    f"(`{special}` with {dtype})"
                ),
            )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_multiple_dt_updates(self) -> None:
        """Cross-backend parity with several ``update_on_bin`` entries.

        Exercises the dt-update branches that a single-bucket profile
        (``update_on_bin = [0]``) never reaches: the dt jump at a non-zero
        bin, and advancing ``i_update`` onto a further update bin. The
        profile is two concatenated sub-profiles with a time gap between
        them, so the jump at the boundary is physically meaningful. Scoped
        to float64 for the same reason as `test_wake_from_pole_residue`.
        """
        import numba as _nb

        n_bins = 64
        n_poles = 3
        dt_val = 1e-9
        boundary = n_bins // 2

        profile_np = np.sin(np.linspace(0, 3 * np.pi, n_bins)) ** 2
        # Second sub-profile (bins >= `boundary`) is shifted later in time,
        # creating a discontinuity that the ``bin_i != 0`` dt-jump branch
        # must absorb.
        profile_dts_np = np.linspace(0, n_bins * dt_val, n_bins + 1)
        profile_dts_np[boundary:] += 10 * dt_val
        poles_np = np.array(
            [-1e8 + 1e9j, -2e8 + 5e8j, -3e8 + 2e9j],
            dtype=np.complex128,
        )
        residues_np = np.array(
            [1.0 + 0.5j, 0.5 - 1.0j, 0.3 + 0.7j],
            dtype=np.complex128,
        )
        # `[0, boundary]`: the update at bin 0 advances ``i_update`` to the
        # second entry (covers ``i_update < len(update_on_bin)``); the
        # update at `boundary` then takes the ``bin_i != 0`` jump branch.
        update_on_bin_np = np.array([0, boundary], dtype=np.int32)

        result_reference = None
        states_reference = None
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue

            profile = backend.array(profile_np, dtype=backend.float)
            profile_dts = backend.array(profile_dts_np, dtype=backend.float)
            poles = backend.array(poles_np, dtype=np.complex128)
            residues = backend.array(residues_np, dtype=np.complex128)
            cr_flags = backend.ones(n_poles, dtype=backend.float)
            states = backend.zeros(n_poles + 1, dtype=np.complex128)
            voltage = backend.zeros(n_bins, dtype=backend.float)
            voltage_threaded = backend.zeros(
                (_nb.get_num_threads(), n_bins), dtype=backend.float
            )
            update_on_bin = backend.array(update_on_bin_np, dtype=np.int32)

            backend.specials.wake_from_pole_residue(
                profile=profile,
                profile_dts=profile_dts,
                poles=poles,
                residues=residues,
                is_counterrotating_beam=False,
                counterrotating_pole_signs=cr_flags,
                update_on_bin=update_on_bin,
                factor=backend.float(1.0),
                states=states,
                voltage=voltage,
                voltage_threaded=voltage_threaded,
            )

            result = np.asarray(copy_to_cpu(voltage))
            states_result = np.asarray(copy_to_cpu(states))

            if result_reference is None:
                result_reference = result
                states_reference = states_result
            else:
                np.testing.assert_allclose(
                    result,
                    result_reference,
                    rtol=1e-10,
                    err_msg=f"Failed test `{special}` with {dtype}",
                )
                np.testing.assert_allclose(
                    states_result,
                    states_reference,
                    rtol=1e-10,
                    err_msg=f"Failed test `{special}` with {dtype}",
                )

    @multi_backend_testcase
    @pytest.mark.backend_mutation
    def test_cast_exceptions(self):
        with self.assertRaises(ArrayCastingError):
            backend.cast_arr_float_if_needed(["a", "b", "c"])

        with self.assertRaises(ArrayCastingError):
            backend.cast_arr_float_if_needed({1, 2, 3})

        with self.assertRaises(ArrayCastingError):
            backend.cast_arr_float_if_needed([[1, 2], 3])

    def test_import(self):
        pass


if __name__ == "__main__":
    unittest.main()
