from unittest import mock

import numpy as np
import pytest
from numba import config as numba_config
from numba import set_num_threads

from blond import copy_to_cpu
from blond.core.backends.backend import Cupy64Bit, Numpy64Bit, backend
from blond.testing.backend_testing import (
    FORCE_ALL_BACKENDS,
    BLonDTestCase,
    cuda_backend_usable,
)

try:
    import cupy  # type: ignore # noqa: F401

    # CuPy can import while the CUDA backend cannot (e.g. no cuBLAS);
    # skip the CUDA cases then, unless all backends are forced.
    cupy_available = FORCE_ALL_BACKENDS or cuda_backend_usable()
except ModuleNotFoundError:
    cupy_available = False

#: Several threads, so the per-thread scratch of the kernels is exercised;
#: clamped to numba's maximum as in `test_backend.py`.
N_TEST_THREADS = min(8, numba_config.NUMBA_NUM_THREADS)


class TestWakeFromPoleResidue(BLonDTestCase):
    """`Specials.wake_from_pole_residue` on every backend."""

    def setUp(self) -> None:
        self.special_modes = [
            "python",
            "cpp",
            "cpp_single_core",
            "numba",
        ]
        if cupy_available:
            self.special_modes.append("cuda")
        set_num_threads(N_TEST_THREADS)
        self.original_backend = type(backend)
        self.original_backend_specials_mode = backend.specials_mode

    def tearDown(self) -> None:
        backend.change_backend(self.original_backend)
        backend.set_specials(self.original_backend_specials_mode)

    def _setUp(self, dtype, special_mode) -> None:
        if dtype != np.float64:
            raise TypeError("32 Bit backends have been removed")
        if special_mode == "cuda":
            backend.change_backend(Cupy64Bit)
        elif special_mode in ("python", "cpp", "cpp_single_core", "numba"):
            backend.change_backend(Numpy64Bit)
        else:
            raise ValueError(special_mode)
        backend.set_specials(special_mode)

    @staticmethod
    def _unrolled_pole_voltage(
        profile_time,
        profile,
        carried_charge,
        carried_is_counterrotating,
        state_lag_dt,
        carried_lag_dt,
        poles,
        residues,
        is_counterrotating_beam,
        signs,
        factor,
        bin_dt,
        states_in,
    ):
        """The kernel's contract written out as a double sum, on the host.

        Bin ``i`` reads the state at ``T_i = t_i - 2 bin_dt``. The state
        holds the incoming state (referenced ``state_lag_dt`` before
        ``T_0``), every profile bin but the last whose time is at or before
        ``T_i``, and -- for every bin but the first -- the carried charge,
        emitted ``carried_lag_dt`` before ``t_0``. The returned state is
        referenced one bin before the last bin and holds all of these.
        """
        n_bins = len(profile)
        t_0 = profile_time[0]
        read_times = profile_time - t_0 - 2.0 * bin_dt
        bin_times = profile_time - t_0
        t_carried = -carried_lag_dt
        tolerance = 1e-6 * bin_dt
        voltage = np.zeros(n_bins)
        states_out = np.zeros(len(poles), dtype=np.complex128)
        for pole_i, (pole, residue) in enumerate(zip(poles, residues)):
            pair = 1.0 if pole.imag == 0 else 2.0
            flip = (
                -1.0
                if (is_counterrotating_beam and signs[pole_i] == -1)
                else 1.0
            )
            carried_flip = (
                -1.0
                if (carried_is_counterrotating and signs[pole_i] == -1)
                else 1.0
            )

            def state_at(
                clock,
                with_carried,
                pole=pole,
                flip=flip,
                pair=pair,
                carried_flip=carried_flip,
                pole_i=pole_i,
            ):
                state = states_in[pole_i] * np.exp(
                    pole * (clock + 2.0 * bin_dt + state_lag_dt)
                )
                for j in range(n_bins - 1):
                    if bin_times[j] <= clock + tolerance:
                        state += (
                            flip
                            * pair
                            * factor
                            * profile[j]
                            * np.exp(pole * (clock - bin_times[j]))
                        )
                if with_carried:
                    state += (
                        carried_flip
                        * pair
                        * carried_charge
                        * np.exp(pole * (clock - t_carried))
                    )
                return state

            for i in range(n_bins):
                state = state_at(read_times[i], with_carried=i > 0)
                voltage[i] += flip * float(np.real(residue * state))
            states_out[pole_i] = state_at(
                bin_times[-1] - bin_dt, with_carried=True
            )
        return voltage, states_out

    def _run_pole_kernel(
        self,
        profile_time,
        profile,
        carried_charge,
        carried_is_counterrotating,
        state_lag_dt,
        carried_lag_dt,
        poles,
        residues,
        is_counterrotating_beam,
        signs,
        factor,
        bin_dt,
        states,
        extra_scratch_rows=0,
    ):
        """Run the active backend's kernel; `states` is a backend array, updated.

        On CUDA, the GEMM far field is checked against the kernel as well.
        """
        states_in = states.copy()
        n_bins = len(profile)
        voltage = backend.zeros(n_bins, dtype=backend.float)
        voltage_threaded = backend.zeros(
            (backend.specials.get_max_threads() + extra_scratch_rows, n_bins),
            dtype=backend.float,
        )
        backend.specials.wake_from_pole_residue(
            profile_time=backend.array(profile_time, dtype=backend.float),
            profile=backend.array(profile, dtype=backend.float),
            carried_charge=backend.array(
                [carried_charge], dtype=backend.float
            ),
            carried_is_counterrotating=carried_is_counterrotating,
            state_lag_dt=state_lag_dt,
            carried_lag_dt=carried_lag_dt,
            poles=backend.array(poles, dtype=backend.complex),
            residues=backend.array(residues, dtype=backend.complex),
            is_counterrotating_beam=is_counterrotating_beam,
            counterrotating_pole_signs=backend.array(
                signs, dtype=backend.float
            ),
            factor=factor,
            bin_dt=bin_dt,
            states=states,
            voltage=voltage,
            voltage_threaded=voltage_threaded,
        )
        if backend.specials_mode == "cuda":
            self._check_pole_gemm_against_kernel(
                profile_time,
                profile,
                carried_charge,
                carried_is_counterrotating,
                state_lag_dt,
                carried_lag_dt,
                poles,
                residues,
                is_counterrotating_beam,
                signs,
                factor,
                bin_dt,
                states_in,
                states,
                voltage,
            )
        return copy_to_cpu(voltage)

    def _check_pole_gemm_against_kernel(
        self,
        profile_time,
        profile,
        carried_charge,
        carried_is_counterrotating,
        state_lag_dt,
        carried_lag_dt,
        poles,
        residues,
        is_counterrotating_beam,
        signs,
        factor,
        bin_dt,
        states_in,
        kernel_states,
        kernel_voltage,
    ):
        """The CUDA GEMM far field gives the kernel's voltage and states.

        Below `_MIN_GEMM_POLES` poles and on runs shorter than 16 bins no
        plan is built in production; both limits are lifted here, so every
        kernel test checks the GEMM too.
        """
        from blond.core.backends.cuda.multipole import pole_gemm

        offsets = np.rint(
            (np.asarray(profile_time) - profile_time[0]) / bin_dt
        ).astype(np.int64)
        with (
            mock.patch.object(pole_gemm, "_MIN_GEMM_POLES", 0),
            mock.patch.object(
                pole_gemm,
                "_GEMM_TILE_LENGTHS",
                (128, 64, 32, 16, 8, 4, 2, 1),
            ),
        ):
            plan = backend.specials.no_parity.pole_tile_gemm_plan(
                bin_offsets=offsets,
                poles=backend.array(poles, dtype=backend.complex),
                residues=backend.array(residues, dtype=backend.complex),
                counterrotating_pole_signs=backend.array(
                    signs, dtype=backend.float
                ),
                bin_dt=bin_dt,
            )
        if len(poles) == 0:
            # cuBLAS rejects empty GEMMs; the kernel handles no poles
            self.assertIsNone(plan)
            return
        self.assertIsNotNone(plan)
        states = states_in.copy()
        voltage = backend.zeros(len(profile), dtype=backend.float) + np.nan
        backend.specials.no_parity.wake_from_pole_residue_gemm(
            plan=plan,
            profile=backend.array(profile, dtype=backend.float),
            carried_charge=backend.array(
                [carried_charge], dtype=backend.float
            ),
            carried_is_counterrotating=carried_is_counterrotating,
            state_lag_dt=state_lag_dt,
            carried_lag_dt=carried_lag_dt,
            is_counterrotating_beam=is_counterrotating_beam,
            factor=factor,
            states=states,
            voltage=voltage,
        )
        # float64 knows the phase Im(p) lag only to ~eps of it, and the two
        # split it into different products
        phase_limit = 1e-15 * float(
            np.max(np.abs(np.imag(poles))) * max(state_lag_dt, carried_lag_dt)
        )
        expected = copy_to_cpu(kernel_voltage)
        np.testing.assert_allclose(
            copy_to_cpu(voltage),
            expected,
            rtol=1e-9,
            atol=(1e-9 + phase_limit)
            * float(np.max(np.abs(expected), initial=0.0)),
            err_msg="GEMM against the CUDA kernel",
        )
        expected_states = copy_to_cpu(kernel_states)
        np.testing.assert_allclose(
            copy_to_cpu(states),
            expected_states,
            rtol=1e-9,
            atol=(1e-9 + phase_limit)
            * float(np.max(np.abs(expected_states), initial=0.0)),
            err_msg="GEMM states against the CUDA kernel",
        )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_matches_unrolled_sum(self) -> None:
        """The recursion equals the unrolled double sum, on every backend.

        A profile with a hole in its axis, a non-zero incoming state, and a
        carried charge that is due before the first read-out (lag of two
        bins and more) or between the first two (lag below two bins).
        """
        bin_dt = 1e-9
        t_0 = 3.0e-4  # the axis does not start at zero
        profile_time = t_0 + bin_dt * np.array([0, 1, 2, 3, 8, 9, 10, 11.0])
        profile = np.array([0.5, 3.0, 1.5, 0.0, 2.0, 0.5, 1.0, 0.7])
        poles_np = np.array(
            [-1e8 + 2e9j, -3e8 + 0.0j, -2e8 + 5e8j], dtype=np.complex128
        )
        residues_np = np.array(
            [1.0 + 0.5j, 2.0 + 0.0j, 0.3 - 0.7j], dtype=np.complex128
        )
        states_in = np.array([0.2 - 0.1j, 0.4 + 0.0j, -0.3 + 0.2j])
        signs = np.ones(len(poles_np))
        factor = 1.7
        carried_charge = 2.3

        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            for carried_lag_bins in (1.0, 1.3, 2.0, 2.5, 4.0):
                state_lag_dt = (carried_lag_bins - 1.0) * bin_dt
                carried_lag_dt = carried_lag_bins * bin_dt
                expected_voltage, expected_states = (
                    self._unrolled_pole_voltage(
                        profile_time,
                        profile,
                        carried_charge,
                        False,
                        state_lag_dt,
                        carried_lag_dt,
                        poles_np,
                        residues_np,
                        False,
                        signs,
                        factor,
                        bin_dt,
                        states_in,
                    )
                )
                states = backend.array(states_in, dtype=backend.complex)
                voltage = self._run_pole_kernel(
                    profile_time,
                    profile,
                    carried_charge,
                    False,
                    state_lag_dt,
                    carried_lag_dt,
                    poles_np,
                    residues_np,
                    False,
                    signs,
                    factor,
                    bin_dt,
                    states,
                )
                # The axis sits at 3e-4 s, where float64 resolves a bin of
                # 1e-9 s to ~5e-11 only; the kernel steps by exact bins,
                # the reference by the rounded times.
                scale = float(np.max(np.abs(expected_voltage)))
                np.testing.assert_allclose(
                    voltage,
                    expected_voltage,
                    rtol=1e-9,
                    atol=1e-9 * scale,
                    err_msg=f"{special=} {carried_lag_bins=}",
                )
                np.testing.assert_allclose(
                    copy_to_cpu(states),
                    expected_states,
                    rtol=1e-9,
                    atol=1e-9 * float(np.max(np.abs(expected_states))),
                    err_msg=f"states {special=} {carried_lag_bins=}",
                )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_persists_state_across_calls(self) -> None:
        """Splitting a profile over two calls equals one long call.

        The second call receives the state one bin before the first call's
        last bin, that bin's charge as the carried charge, and the gap
        between the two calls as the lags -- the hand-over the solver does.
        """
        bin_dt = 1e-9
        profile = np.array([0.0, 0.0, 3.0, 1.5, 0.0, 2.0, 0.5, 1.0])
        poles_np = np.array([-1e8 + 2e9j, -3e8 + 0.0j], dtype=np.complex128)
        residues_np = np.array([1.0 + 0.5j, 2.0 + 0.0j], dtype=np.complex128)
        signs = np.ones(2)
        factor = 1.7

        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            # dense axis, and an axis with a hole exactly at the split
            for gap_bins in (1, 4):
                profile_time = bin_dt * np.concatenate(
                    (np.arange(4), np.arange(4) + 3 + gap_bins)
                )
                states = backend.zeros(2, dtype=backend.complex)
                voltage_single = self._run_pole_kernel(
                    profile_time,
                    profile,
                    0.0,
                    False,
                    0.0,
                    2 * bin_dt,
                    poles_np,
                    residues_np,
                    False,
                    signs,
                    factor,
                    bin_dt,
                    states,
                )
                states = backend.zeros(2, dtype=backend.complex)
                voltage_a = self._run_pole_kernel(
                    profile_time[:4],
                    profile[:4],
                    0.0,
                    False,
                    0.0,
                    2 * bin_dt,
                    poles_np,
                    residues_np,
                    False,
                    signs,
                    factor,
                    bin_dt,
                    states,
                )
                gap_dt = gap_bins * bin_dt
                voltage_b = self._run_pole_kernel(
                    profile_time[4:],
                    profile[4:],
                    factor * profile[3],
                    False,
                    gap_dt - bin_dt,
                    gap_dt,
                    poles_np,
                    residues_np,
                    False,
                    signs,
                    factor,
                    bin_dt,
                    states,
                )
                # The kernel never puts the carried charge into the first
                # bin's far field -- the caller adds it in closed form. For
                # a gap of two bins or more that closed form is the same
                # exponential the single call used, so add it here.
                if gap_bins >= 2:
                    for pole, residue in zip(poles_np, residues_np):
                        pair = 1.0 if pole.imag == 0 else 2.0
                        voltage_b[0] += float(
                            np.real(
                                residue
                                * pair
                                * factor
                                * profile[3]
                                * np.exp(pole * (gap_dt - 2 * bin_dt))
                            )
                        )
                scale = float(np.max(np.abs(voltage_single)))
                np.testing.assert_allclose(
                    voltage_a,
                    voltage_single[:4],
                    rtol=1e-12,
                    atol=1e-12 * scale,
                    err_msg=f"head {special=} {gap_bins=}",
                )
                np.testing.assert_allclose(
                    voltage_b,
                    voltage_single[4:],
                    rtol=1e-12,
                    atol=1e-12 * scale,
                    err_msg=f"tail {special=} {gap_bins=}",
                )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_zero_poles(self) -> None:
        """A model without poles gives zero voltage and does not crash."""
        n_bins = 16
        bin_dt = 1e-9
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            states = backend.zeros(0, dtype=backend.complex)
            voltage = self._run_pole_kernel(
                np.arange(n_bins) * bin_dt,
                np.sin(np.linspace(0, 3 * np.pi, n_bins)) ** 2,
                0.0,
                False,
                0.0,
                2 * bin_dt,
                np.zeros(0, dtype=np.complex128),
                np.zeros(0, dtype=np.complex128),
                False,
                np.zeros(0),
                1.0,
                bin_dt,
                states,
            )
            np.testing.assert_array_equal(
                voltage,
                np.zeros(n_bins),
                err_msg=f"`{special}` gave a voltage without poles",
            )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_many_mixed_poles(self) -> None:
        """Many poles of mixed kind equal the unrolled sum, on every backend.

        Backends may process poles in groups (SIMD blocks, threads); pole
        counts that are not a multiple of any group size, real and complex
        poles and flipped and unflipped poles side by side in one group,
        and an axis with a hole (steps longer than a bin) must still give
        every pole its own recursion.
        """
        bin_dt = 1e-9
        profile_time = bin_dt * np.array([0, 1, 2, 3, 4, 9, 10, 11, 12.0])
        profile = np.array([0.5, 3.0, 1.5, 0.0, 2.0, 0.5, 1.0, 0.7, 0.2])
        factor = 1.7
        carried_charge = 2.3
        state_lag_dt, carried_lag_dt = 0.3 * bin_dt, 1.3 * bin_dt
        rng = np.random.default_rng(42)

        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            for n_poles in (1, 5, 17, 40, 103, 130):
                is_real = np.arange(n_poles) % 4 == 1
                poles_np = -rng.uniform(5e7, 5e8, n_poles) + 1j * np.where(
                    is_real, 0.0, rng.uniform(1e8, 3e9, n_poles)
                )
                residues_np = rng.normal(size=n_poles) + 1j * np.where(
                    is_real, 0.0, rng.normal(size=n_poles)
                )
                states_in = rng.normal(size=n_poles) + 1j * np.where(
                    is_real, 0.0, rng.normal(size=n_poles)
                )
                signs = np.where(np.arange(n_poles) % 3 == 0, -1.0, 1.0)
                for carried_is_counterrotating in (False, True):
                    expected_voltage, expected_states = (
                        self._unrolled_pole_voltage(
                            profile_time,
                            profile,
                            carried_charge,
                            carried_is_counterrotating,
                            state_lag_dt,
                            carried_lag_dt,
                            poles_np,
                            residues_np,
                            True,
                            signs,
                            factor,
                            bin_dt,
                            states_in,
                        )
                    )
                    states = backend.array(states_in, dtype=backend.complex)
                    voltage = self._run_pole_kernel(
                        profile_time,
                        profile,
                        carried_charge,
                        carried_is_counterrotating,
                        state_lag_dt,
                        carried_lag_dt,
                        poles_np,
                        residues_np,
                        True,
                        signs,
                        factor,
                        bin_dt,
                        states,
                    )
                    msg = (
                        f"{special=} {n_poles=} {carried_is_counterrotating=}"
                    )
                    scale = float(np.max(np.abs(expected_voltage)))
                    np.testing.assert_allclose(
                        voltage,
                        expected_voltage,
                        rtol=1e-9,
                        atol=1e-9 * scale,
                        err_msg=msg,
                    )
                    np.testing.assert_allclose(
                        copy_to_cpu(states),
                        expected_states,
                        rtol=1e-9,
                        atol=1e-9 * float(np.max(np.abs(expected_states))),
                        err_msg=f"states {msg}",
                    )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_long_profile_matches_python(self) -> None:
        """A long profile with holes gives the reference result everywhere.

        Long enough that a backend may split the bins into chunks that run
        in parallel and are stitched together afterwards: the stitching
        must reproduce the serial recursion, its hand-over state included.
        """
        bin_dt = 1e-9
        n_bins = 1200
        index = np.arange(n_bins)
        # a hole of 7 bins after every 50 bins (a sparse, multi-bunch axis)
        profile_time = 3.0e-4 + bin_dt * (index + 7 * (index // 50))
        profile = 1.0 + np.sin(index / 37.0) ** 2
        factor = 1.7
        carried_charge = 2.3
        rng = np.random.default_rng(7)
        cases = []
        for n_poles in (3, 70):
            is_real = np.arange(n_poles) % 4 == 1
            poles_np = -rng.uniform(5e6, 5e8, n_poles) + 1j * np.where(
                is_real, 0.0, rng.uniform(1e8, 3e9, n_poles)
            )
            residues_np = rng.normal(size=n_poles) + 1j * np.where(
                is_real, 0.0, rng.normal(size=n_poles)
            )
            states_in = rng.normal(size=n_poles) + 1j * np.where(
                is_real, 0.0, rng.normal(size=n_poles)
            )
            signs = np.where(np.arange(n_poles) % 3 == 0, -1.0, 1.0)
            for carried_lag_bins in (1.3, 2.5):
                cases.append(
                    (
                        poles_np,
                        residues_np,
                        states_in,
                        signs,
                        carried_lag_bins,
                    )
                )

        def run_all():
            results = []
            for poles_np, residues_np, states_in, signs, lag_bins in cases:
                states = backend.array(states_in, dtype=backend.complex)
                voltage = self._run_pole_kernel(
                    profile_time,
                    profile,
                    carried_charge,
                    True,
                    (lag_bins - 1.0) * bin_dt,
                    lag_bins * bin_dt,
                    poles_np,
                    residues_np,
                    True,
                    signs,
                    factor,
                    bin_dt,
                    states,
                )
                results.append((voltage, copy_to_cpu(states)))
            return results

        self._setUp(dtype=np.float64, special_mode="python")
        expected = run_all()
        for special in self.special_modes:
            if special == "python":
                continue
            try:
                self._setUp(dtype=np.float64, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test")
                continue
            results = run_all()
            for case_i, ((voltage, states), (exp_v, exp_s)) in enumerate(
                zip(results, expected, strict=True)
            ):
                msg = f"{special=} {case_i=}"
                np.testing.assert_allclose(
                    voltage,
                    exp_v,
                    rtol=1e-10,
                    atol=1e-10 * float(np.max(np.abs(exp_v))),
                    err_msg=msg,
                )
                np.testing.assert_allclose(
                    states,
                    exp_s,
                    rtol=1e-10,
                    atol=1e-10 * float(np.max(np.abs(exp_s))),
                    err_msg=f"states {msg}",
                )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_long_lag_matches_python(self) -> None:
        """A lag of many RF periods gives the reference result everywhere.

        A slowly decaying GHz pole stepped over ~0.1 s has a phase of
        ~3e9 rad, beyond the range of a fast sine with an int32 quadrant.
        A float64 phase that large is only known to ~eps * phase, so
        backends that split it differently agree only to that.
        """
        bin_dt = 2.5e-11
        n_bins = 8
        profile_time = np.arange(n_bins) * bin_dt
        profile = np.linspace(0.5, 1.5, n_bins)
        poles_np = np.array([-1.0 + 1j * (2 * np.pi * 5e9 + 0.123)])
        residues_np = np.array([1.0 + 0.3j])
        states_in = np.array([1.0 + 0.5j])
        signs = np.ones(1)

        def run(lag_dt):
            states = backend.array(states_in, dtype=backend.complex)
            voltage = self._run_pole_kernel(
                profile_time,
                profile,
                0.7,
                False,
                lag_dt,
                lag_dt + 0.5 * bin_dt,
                poles_np,
                residues_np,
                False,
                signs,
                1.0,
                bin_dt,
                states,
            )
            return voltage, copy_to_cpu(states)

        lags = (1e-2, 1e-1)
        self._setUp(dtype=np.float64, special_mode="python")
        expected = [run(lag_dt) for lag_dt in lags]
        for special in self.special_modes:
            if special == "python":
                continue
            try:
                self._setUp(dtype=np.float64, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test")
                continue
            for lag_dt, (exp_v, exp_s) in zip(lags, expected, strict=True):
                voltage, states = run(lag_dt)
                msg = f"{special=} {lag_dt=}"
                phase_tolerance = 1e-15 * abs(poles_np[0].imag) * lag_dt
                np.testing.assert_allclose(
                    voltage,
                    exp_v,
                    rtol=0,
                    atol=phase_tolerance * float(np.max(np.abs(exp_v))),
                    err_msg=msg,
                )
                np.testing.assert_allclose(
                    states,
                    exp_s,
                    rtol=0,
                    atol=phase_tolerance * float(np.max(np.abs(exp_s))),
                    err_msg=f"states {msg}",
                )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_accepts_extra_scratch_rows(self) -> None:
        """More scratch rows than threads, as the contract allows, are fine.

        The solver sizes the scratch once, for the backend active at its
        first call; a later backend with fewer threads must still run.
        """
        n_bins = 16
        bin_dt = 1e-9
        args = (
            np.arange(n_bins) * bin_dt,
            np.sin(np.linspace(0, 3 * np.pi, n_bins)) ** 2,
            0.0,
            False,
            0.0,
            2 * bin_dt,
            np.array([-1e8 + 2e9j, -3e8 + 0.0j]),
            np.array([1.0 + 0.5j, 2.0 + 0.0j]),
            False,
            np.ones(2),
            1.0,
            bin_dt,
        )
        for special in self.special_modes:
            try:
                self._setUp(dtype=np.float64, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test")
                continue
            expected = self._run_pole_kernel(
                *args, backend.zeros(2, dtype=backend.complex)
            )
            voltage = self._run_pole_kernel(
                *args,
                backend.zeros(2, dtype=backend.complex),
                extra_scratch_rows=3,
            )
            np.testing.assert_allclose(
                voltage, expected, rtol=1e-14, err_msg=f"{special=}"
            )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_extreme_magnitudes(self) -> None:
        """Finite and accurate for the magnitudes a vector fit produces.

        Twenty fast-decaying poles with |Re p| up to 1.4e7 and residues up
        to 1e13 (a real fit of a physical impedance), plus one real pole,
        against the unrolled sum.
        """
        poles_np = np.array(
            [
                -2623831.59946355 + 1.39099415e09j,
                -3324980.21266537 + 1.37641478e09j,
                -2627142.66632887 + 1.12342679e09j,
                -4057802.76251799 + 1.36173266e09j,
                -3332053.6956318 + 1.13800877e09j,
                -4859598.51825629 + 1.34699927e09j,
                -4069084.51675692 + 1.15269216e09j,
                -5769309.76869008 + 1.33225090e09j,
                -4876008.93418745 + 1.16742610e09j,
                -5792260.22147667 + 1.18217408e09j,
                -6839639.82479881 + 1.31752969e09j,
                -6871261.85241204 + 1.19689324e09j,
                -8150077.00435694 + 1.30291029e09j,
                -8193661.34867393 + 1.21150693e09j,
                -9814345.82669513 + 1.28856816e09j,
                -9873890.22060673 + 1.22583392e09j,
                -11873023.777707 + 1.27495791e09j,
                -11943110.44842772 + 1.23940612e09j,
                -13657970.95550037 + 1.26278082e09j,
                -13693795.03125995 + 1.25153131e09j,
                -1.4e7 + 0.0j,
            ],
            dtype=np.complex128,
        )
        rng = np.random.default_rng(7)
        residues_np = (
            1e13
            * (rng.random(len(poles_np)) - 0.5)
            * (1.0 + 1j * (rng.random(len(poles_np)) - 0.5))
        )
        residues_np[-1] = 1e13 + 0.0j
        bin_dt = 1e-10
        profile_time = bin_dt * np.concatenate(
            (np.arange(12), np.arange(12) + 20)
        )
        profile = rng.random(len(profile_time))
        signs = np.ones(len(poles_np))
        factor = 1.6e-19 * 1e11
        states_in = np.zeros(len(poles_np), dtype=np.complex128)

        expected_voltage, _ = self._unrolled_pole_voltage(
            profile_time,
            profile,
            0.4,
            False,
            0.5 * bin_dt,
            1.5 * bin_dt,
            poles_np,
            residues_np,
            False,
            signs,
            factor,
            bin_dt,
            states_in,
        )
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            states = backend.array(states_in, dtype=backend.complex)
            voltage = self._run_pole_kernel(
                profile_time,
                profile,
                0.4,
                False,
                0.5 * bin_dt,
                1.5 * bin_dt,
                poles_np,
                residues_np,
                False,
                signs,
                factor,
                bin_dt,
                states,
            )
            self.assertTrue(np.all(np.isfinite(voltage)), special)
            np.testing.assert_allclose(
                voltage,
                expected_voltage,
                rtol=1e-10,
                atol=1e-10 * float(np.max(np.abs(expected_voltage))),
                err_msg=f"{special=}",
            )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_counter_rotation_flips(self) -> None:
        """A beam's own wake is unchanged by counter-rotation; another's flips.

        The flip is applied to both the injected charge and the amplitude
        read back, so a counter-rotating beam alone sees the co-rotating
        voltage. A charge carried from a call of the *other* beam keeps that
        beam's flip, and a bin observing it sees the product of both.
        """
        bin_dt = 1e-9
        profile_time = bin_dt * np.arange(6)
        profile = np.array([0.0, 3.0, 1.5, 0.0, 2.0, 0.5])
        poles_np = np.array(
            [-1e8 + 1e9j, -2e8 + 5e8j, -3e8 + 2e9j], dtype=np.complex128
        )
        residues_np = np.array(
            [1.0 + 0.5j, 0.5 - 1.0j, 0.3 + 0.7j], dtype=np.complex128
        )
        signs = np.array([1.0, -1.0, 1.0])
        factor = 1.7
        carried_charge = 1.1
        state_lag_dt, carried_lag_dt = 0.5 * bin_dt, 1.5 * bin_dt

        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue

            def run(is_cr, carried_is_cr, signs_=signs):
                states = backend.zeros(len(poles_np), dtype=backend.complex)
                return self._run_pole_kernel(
                    profile_time,
                    profile,
                    carried_charge,
                    carried_is_cr,
                    state_lag_dt,
                    carried_lag_dt,
                    poles_np,
                    residues_np,
                    is_cr,
                    signs_,
                    factor,
                    bin_dt,
                    states,
                )

            baseline = run(False, False)
            # the same beam, counter-rotating, in both calls: invariant
            np.testing.assert_allclose(
                run(True, True),
                baseline,
                rtol=1e-12,
                err_msg=f"own wake must not flip ({special=})",
            )
            # the carried charge came from the other beam: its pole with a
            # sign of -1 contributes with the opposite sign
            expected, _ = self._unrolled_pole_voltage(
                profile_time,
                profile,
                carried_charge,
                True,
                state_lag_dt,
                carried_lag_dt,
                poles_np,
                residues_np,
                False,
                signs,
                factor,
                bin_dt,
                np.zeros(len(poles_np), dtype=np.complex128),
            )
            crossed = run(False, True)
            np.testing.assert_allclose(
                crossed,
                expected,
                rtol=1e-12,
                atol=1e-12 * float(np.max(np.abs(expected))),
                err_msg=f"carried flip ({special=})",
            )
            self.assertFalse(
                np.allclose(crossed, baseline, rtol=1e-6),
                f"a flipped pole must change the carried contribution ({special=})",
            )

    @pytest.mark.backend_mutation
    def test_wake_from_pole_residue_flip_invariance_across_calls(
        self,
    ) -> None:
        """The counter-rotation flip cancels only for the beam that injected it.

        For a pole with a sign of -1 the state a counter-rotating beam
        injects picks up an overall -1, and the same beam's read-out
        multiplies by that -1 again, so the two cancel within one call
        (`test_wake_from_pole_residue_counter_rotation_flips`). Across
        the call boundary the flip is already baked into the persisted
        state and the carried charge: a beam that is counter-rotating in
        both calls must still see the co-rotating voltage, while a flip
        applied in the injecting call only leaves a trace in the
        persisted state that the later, unflipped call cannot undo.
        """
        bin_dt = 1e-9
        profile = np.array([0.0, 0.0, 3.0, 1.5, 0.0, 2.0, 0.5, 1.0])
        poles_np = np.array(
            [-1e8 + 1e9j, -2e8 + 5e8j, -3e8 + 2e9j], dtype=np.complex128
        )
        residues_np = np.array(
            [1.0 + 0.5j, 0.5 - 1.0j, 0.3 + 0.7j], dtype=np.complex128
        )
        signs = np.array([1.0, -1.0, 1.0])
        factor = 1.7

        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            # dense axis, and an axis with a hole exactly at the split
            for gap_bins in (1, 4):
                profile_time = bin_dt * np.concatenate(
                    (np.arange(4), np.arange(4) + 3 + gap_bins)
                )
                gap_dt = gap_bins * bin_dt

                def run_two_calls(
                    first_is_cr,
                    second_is_cr,
                    profile_time=profile_time,
                    gap_dt=gap_dt,
                ):
                    # the hand-over the solver does: the second call
                    # receives the state one bin before the first call's
                    # last bin, that bin's charge as the carried charge
                    # with the first call's rotation, and the gap between
                    # the two calls as the lags
                    states = backend.zeros(
                        len(poles_np), dtype=backend.complex
                    )
                    voltage_a = self._run_pole_kernel(
                        profile_time[:4],
                        profile[:4],
                        0.0,
                        False,
                        0.0,
                        2 * bin_dt,
                        poles_np,
                        residues_np,
                        first_is_cr,
                        signs,
                        factor,
                        bin_dt,
                        states,
                    )
                    voltage_b = self._run_pole_kernel(
                        profile_time[4:],
                        profile[4:],
                        factor * profile[3],
                        first_is_cr,
                        gap_dt - bin_dt,
                        gap_dt,
                        poles_np,
                        residues_np,
                        second_is_cr,
                        signs,
                        factor,
                        bin_dt,
                        states,
                    )
                    return np.concatenate((voltage_a, voltage_b))

                baseline = run_two_calls(False, False)
                scale = float(np.max(np.abs(baseline)))
                np.testing.assert_allclose(
                    run_two_calls(True, True),
                    baseline,
                    rtol=1e-12,
                    atol=1e-12 * scale,
                    err_msg=(
                        "own wake must not flip across calls "
                        f"({special=} {gap_bins=})"
                    ),
                )
                self.assertFalse(
                    np.allclose(
                        run_two_calls(True, False),
                        baseline,
                        rtol=1e-6,
                        atol=1e-6 * scale,
                    ),
                    "a flip in the injecting call only must leave a trace "
                    f"in the persisted state ({special=} {gap_bins=})",
                )
