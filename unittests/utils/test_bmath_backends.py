import unittest
from copy import deepcopy

import numpy as np

from blond.utils.bmath_backends import NumbaBackend, CppBackend, GpuBackend
from blond.utils.bmath_backends import PyBackend

py_backend = PyBackend()


def cast_arrays(other: tuple | dict, tested_backend):
    """Converts arrays to the array type of the backend

    Notes
    -----
    At the time of writing, this is done to cast numpy to cupy arrays.
    Might change when adding more backends.

    """
    if isinstance(other, dict):
        kwargs_other = other
        for key, item in kwargs_other.items():
            if hasattr(item, "dtype"):
                kwargs_other[key] = tested_backend.array(item)
        return kwargs_other
    elif isinstance(other, tuple):
        args_other = list(other)
        for i in range(len(args_other)):
            item = args_other[i]
            if hasattr(item, "dtype"):
                args_other[i] = tested_backend.array(item)
        return tuple(args_other)
    else:
        raise TypeError(str(type(other)))


class TestSameResult(unittest.TestCase):
    def setUp(self):
        # set up backends that are compared against python backend
        try:
            import cupy

            self.tested_backends = (NumbaBackend(), CppBackend(), GpuBackend())

        except ImportError:
            self.tested_backends = (
                NumbaBackend(),
                CppBackend(),
            )

    def test_rfft(self):
        args = (np.random.randn(100),)
        expected_result = py_backend.rfft(*deepcopy(args))
        for tested_backend in self.tested_backends:
            other_result = tested_backend.rfft(
                *cast_arrays(deepcopy(args), tested_backend)
            )
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_irfft(self):
        args = (np.random.randn(100),)

        expected_result = py_backend.irfft(*deepcopy(args))
        for tested_backend in self.tested_backends:
            other_result = tested_backend.irfft(
                *cast_arrays(deepcopy(args), tested_backend)
            )
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_rfftfreq(self):
        args = (100, 0.1)
        kwargs = dict()

        expected_result = py_backend.rfftfreq(
            *deepcopy(args), **deepcopy(kwargs)
        )
        for tested_backend in self.tested_backends:
            other_result = tested_backend.rfftfreq(
                *deepcopy(args),
                **cast_arrays(deepcopy(kwargs), tested_backend),
            )
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_convolve(self):
        args = (np.random.randn(100), np.random.randn(10))

        expected_result = py_backend.convolve(*deepcopy(args))
        for tested_backend in self.tested_backends:
            other_result = tested_backend.convolve(
                *cast_arrays(deepcopy(args), tested_backend)
            )
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_beam_phase(self):
        n_particles = 10
        n_slices = 5
        omega_rf = np.random.rand()
        phi_rf = np.random.rand()
        alpha = np.random.rand()
        dt = np.random.normal(1e-5, 1e-7, n_particles)
        profile, edges = np.histogram(dt, bins=n_slices)
        profile = profile.astype(dtype=float)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size / 2
        kwargs = dict(
            bin_centers=bin_centers,
            profile=profile,
            alpha=alpha,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            bin_size=bin_size,
        )

        expected_result = py_backend.beam_phase(**deepcopy(kwargs))
        for tested_backend in self.tested_backends:
            kwargs_other = cast_arrays(deepcopy(kwargs), tested_backend)
            other_result = tested_backend.beam_phase(**kwargs_other)
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_beam_phase_fast(self):
        args = ()
        n_particles = 10
        n_slices = 5
        omega_rf = np.random.rand()
        phi_rf = np.random.rand()
        dt = np.random.normal(1e-5, 1e-7, n_particles)
        profile, edges = np.histogram(dt, bins=n_slices)
        profile = profile.astype(dtype=float)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size / 2
        kwargs = dict(
            bin_centers=bin_centers,
            profile=profile,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            bin_size=bin_size,
        )

        expected_result = py_backend.beam_phase_fast(
            *deepcopy(args), **deepcopy(kwargs)
        )
        for tested_backend in self.tested_backends:
            other_result = tested_backend.beam_phase_fast(
                *deepcopy(args),
                **cast_arrays(deepcopy(kwargs), tested_backend),
            )
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_kick(self):
        args = ()

        n_particles = 10
        n_rf = 5
        dE = np.random.normal(loc=0, scale=1e7, size=n_particles)
        dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)

        charge = 1.0
        acceleration_kick = 1e3 * np.random.rand()
        voltage = np.random.randn(n_rf)
        omega_rf = np.random.randn(n_rf)
        phi_rf = np.random.randn(n_rf)
        kwargs = dict(
            dt=dt,
            dE=dE,
            voltage=voltage,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            charge=charge,
            n_rf=n_rf,
            acceleration_kick=acceleration_kick,
        )
        args_py, kwargs_py = deepcopy(args), deepcopy(kwargs)
        py_backend.kick(*args_py, **kwargs_py)
        expected_result = kwargs_py["dE"]
        for tested_backend in self.tested_backends:
            args_other, kwargs_other = (
                deepcopy(args),
                cast_arrays(deepcopy(kwargs), tested_backend),
            )
            tested_backend.kick(*args_other, **kwargs_other)
            other_result = kwargs_other["dE"]
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_rf_volt_comp(self):
        args = ()
        n_slices = 10
        kwargs = dict(
            voltages=np.random.randn(n_slices),
            omega_rf=np.random.randn(n_slices),
            phi_rf=np.random.randn(n_slices),
            bin_centers=np.linspace(1e-5, 1e-6, n_slices),
        )

        expected_result = py_backend.rf_volt_comp(
            *deepcopy(args), **deepcopy(kwargs)
        )
        for tested_backend in self.tested_backends:
            other_result = tested_backend.rf_volt_comp(
                *deepcopy(args),
                **cast_arrays(deepcopy(kwargs), tested_backend),
            )
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_drift(self):
        for solver in (
            "simple",
            "legacy",
            "exact",
        ):
            for alpha_order in range(3):
                n_particles = 10
                kwargs = dict(
                    dE=np.random.normal(loc=0, scale=1e7, size=n_particles),
                    dt=np.random.normal(
                        loc=1e-5, scale=1e-7, size=n_particles
                    ),
                    t_rev=np.random.rand(),
                    length_ratio=np.random.uniform(),
                    eta_0=np.random.rand(),
                    eta_1=np.random.rand(),
                    eta_2=np.random.rand(),
                    alpha_0=np.random.rand(),
                    alpha_1=np.random.rand(),
                    alpha_2=np.random.rand(),
                    beta=np.random.rand(),
                    energy=np.random.rand(),
                    solver=solver,
                    alpha_order=alpha_order,
                )

                kwargs_py = deepcopy(kwargs)
                py_backend.drift(**kwargs_py)
                expected_result = kwargs_py["dt"]
                for tested_backend in self.tested_backends:
                    kwargs_other = cast_arrays(
                        deepcopy(kwargs), tested_backend
                    )
                    tested_backend.drift(**kwargs_other)
                    other_result = kwargs_other["dt"]
                    self.assertTrue(
                        np.allclose(expected_result, other_result),
                        f"Failed with {tested_backend}",
                    )

    def test_linear_interp_kick(self):
        args = ()
        n_particles = 10
        n_slices = 5
        dE = np.random.normal(loc=0, scale=1e7, size=n_particles)

        dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        _, edges = np.histogram(dt, bins=n_slices)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size / 2

        charge = 1.0
        acceleration_kick = 1e3 * np.random.rand()
        voltage = np.random.randn(n_slices)
        kwargs = dict(
            dt=dt,
            dE=dE,
            voltage=voltage,
            bin_centers=bin_centers,
            charge=charge,
            acceleration_kick=acceleration_kick,
        )

        kwargs_py = deepcopy(kwargs)
        py_backend.linear_interp_kick(*deepcopy(args), **kwargs_py)
        expected_result = kwargs_py["dE"]
        for tested_backend in self.tested_backends:
            kwargs_other = cast_arrays(deepcopy(kwargs), tested_backend)
            tested_backend.linear_interp_kick(*deepcopy(args), **kwargs_other)
            other_result = kwargs_other["dE"]
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_synchrotron_radiation(self):
        kwargs = dict(
            dE=np.random.uniform(-1e8, 1e8, 10),
            U0=np.random.rand(),
            n_kicks=10,
            tau_z=np.random.rand(),
        )

        kwargs_py = deepcopy(kwargs)
        py_backend.synchrotron_radiation(**kwargs_py)
        expected_result = kwargs_py["dE"]
        for tested_backend in self.tested_backends:
            kwargs_other = cast_arrays(deepcopy(kwargs), tested_backend)
            tested_backend.synchrotron_radiation(**kwargs_other)
            other_result = kwargs_other["dE"]
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_synchrotron_radiation_full(self):
        dE = np.random.uniform(-1e8, 1e8, 10)
        kwargs = dict(
            dE=dE,
            U0=np.random.rand(),
            n_kicks=10,
            tau_z=np.random.rand(),
            sigma_dE=dE.std().item(),
            energy=0.0,
        )
        kwargs_py = deepcopy(kwargs)
        py_backend.synchrotron_radiation_full(**kwargs_py)
        expected_result = kwargs_py["dE"]
        for tested_backend in self.tested_backends:
            kwargs_other = cast_arrays(deepcopy(kwargs), tested_backend)
            tested_backend.synchrotron_radiation_full(**kwargs_other)
            other_result = kwargs_other["dE"]
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_slice_beam(self):
        n_particles, n_slices, cut_left, cut_right = 10, 5, -0.1, +0.1
        dt = np.random.normal(loc=1e-5, scale=1e-6, size=n_particles)

        max_dt = dt.max().item()
        min_dt = dt.min().item()
        cut_left = (1 + cut_left) * min_dt
        cut_right = (1 - cut_right) * max_dt
        profile = np.empty(n_slices, dtype=np.float64)
        kwargs = dict(
            dt=dt, profile=profile, cut_left=cut_left, cut_right=cut_right
        )

        kwargs_py = deepcopy(kwargs)
        py_backend.slice_beam(**kwargs_py)
        expected_result = kwargs_py["profile"]
        for tested_backend in self.tested_backends:
            kwargs_other = cast_arrays(deepcopy(kwargs), tested_backend)
            tested_backend.slice_beam(**kwargs_other)
            other_result = kwargs_other["profile"]
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_sparse_histogram(self):
        args = ()

        kwargs = dict(
            dt=np.ones(10),
            profile=np.zeros((5, 5)),
            cut_left=-0.1 * np.ones(5),
            cut_right=0.1 * np.ones(5),
            bunch_indexes=np.ones(10),
            n_slices_bucket=2,
        )

        kwargs_py = deepcopy(kwargs)
        py_backend.sparse_histogram(*deepcopy(args), **kwargs_py)
        expected_result = kwargs_py["profile"]
        for tested_backend in self.tested_backends:
            if isinstance(tested_backend, GpuBackend):
                continue
            kwargs_other = deepcopy(kwargs)
            tested_backend.sparse_histogram(*deepcopy(args), **kwargs_other)
            other_result = kwargs_other["profile"]
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    @unittest.skip("Might be legacy")  # todo resolve questions
    def test_distribution_from_tomoscope(self):
        args = ()
        kwargs = dict(
            dt=np.empty(10),
            dE=np.empty(10),
            probDistr=np.random.rand(10),
            seed=10,
            profLen=10,
            cutoff=5,
            x0=1,
            y0=2,
            dtBin=0.1,
            dEBin=0.1,
        )

        kwargs_py = deepcopy(kwargs)
        py_backend.distribution_from_tomoscope(*deepcopy(args), **kwargs_py)
        expected_result_dE = kwargs_py["dE"]
        expected_result_dt = kwargs_py["dt"]
        for tested_backend in self.tested_backends:
            kwargs_other = deepcopy(kwargs)
            tested_backend.distribution_from_tomoscope(
                *deepcopy(args), **kwargs_other
            )
            other_result_dE = kwargs_other["dE"]
            other_result_dt = kwargs_other["dt"]
            self.assertTrue(
                np.allclose(expected_result_dE, other_result_dE),
                f"Failed with {tested_backend}",
            )
            self.assertTrue(
                np.allclose(expected_result_dt, other_result_dt),
                f"Failed with {tested_backend}",
            )

    def test_fast_resonator(self):
        args = ()
        size = 10
        R_S = np.random.uniform(size=size)
        Q = np.random.uniform(low=0.5, high=1, size=size)
        frequency_array = np.linspace(1, 100, num=size)
        frequency_R = np.linspace(100, 1000, num=size)
        kwargs = dict(
            R_S=R_S,
            Q=Q,
            frequency_array=frequency_array,
            frequency_R=frequency_R,
        )

        expected_result = py_backend.fast_resonator(
            *deepcopy(args), **deepcopy(kwargs)
        )
        for tested_backend in self.tested_backends:
            if isinstance(tested_backend, GpuBackend):
                continue
            other_result = tested_backend.fast_resonator(
                *deepcopy(args),
                **cast_arrays(deepcopy(kwargs), tested_backend),
            )
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_slice_smooth(self):
        args = ()
        n_particles = 10
        n_slices = 5

        kwargs = dict(
            dt=np.random.normal(loc=1e-5, scale=1e-6, size=n_particles),
            profile=np.empty(n_slices, dtype=float),
            cut_left=-0.1,
            cut_right=0.1,
        )

        kwargs_py = deepcopy(kwargs)
        py_backend.slice_smooth(*deepcopy(args), **kwargs_py)
        expected_result = kwargs_py["profile"]
        for tested_backend in self.tested_backends:
            if isinstance(tested_backend, GpuBackend):
                continue
            kwargs_other = deepcopy(kwargs)
            tested_backend.slice_smooth(*deepcopy(args), **kwargs_other)
            other_result = kwargs_other["profile"]
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    def test_music_track_multiturn(self):
        np.random.seed(0)
        args = ()
        kwargs = dict(
            dt=np.arange(20, dtype=float),
            dE=np.arange(20, dtype=float),
            induced_voltage=np.arange(20, dtype=float),
            array_parameters=np.empty(4),
            alpha=0.1,
            omega_bar=1,
            const=2,
            coeff1=3,
            coeff2=4,
            coeff3=5,
            coeff4=6,
        )

        kwargs_py = deepcopy(kwargs)
        py_backend.music_track_multiturn(*deepcopy(args), **kwargs_py)
        expected_result = kwargs_py["array_parameters"]
        for tested_backend in self.tested_backends:
            if isinstance(tested_backend, GpuBackend):
                continue
            kwargs_other = deepcopy(kwargs)
            tested_backend.music_track_multiturn(
                *deepcopy(args), **kwargs_other
            )
            other_result = kwargs_other["array_parameters"]
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}, expected {expected_result}, but got {other_result}",
            )

    def test_music_track(self):
        args = ()
        kwargs = dict(
            dt=np.random.randn(20),
            dE=np.random.randn(20),
            induced_voltage=np.random.randn(20),
            array_parameters=np.empty(4),
            alpha=0.1,
            omega_bar=1,
            const=2,
            coeff1=3,
            coeff2=4,
            coeff3=5,
            coeff4=6,
        )

        kwargs_py = deepcopy(kwargs)
        py_backend.music_track(*deepcopy(args), **kwargs_py)
        expected_result = kwargs_py["array_parameters"]
        for tested_backend in self.tested_backends:
            if isinstance(tested_backend, GpuBackend):
                continue
            kwargs_other = deepcopy(kwargs)
            tested_backend.music_track(*deepcopy(args), **kwargs_other)
            other_result = kwargs_other["array_parameters"]
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )

    @unittest.skip("Needs to be fixed")  # TODO
    def test_set_random_seed(self):
        args = ()
        kwargs = dict(seed=5)

        py_backend.set_random_seed(*deepcopy(args), **deepcopy(kwargs))
        expected_result = py_backend.random.rand()
        for tested_backend in self.tested_backends:
            tested_backend.set_random_seed(
                *deepcopy(args),
                **cast_arrays(deepcopy(kwargs), tested_backend),
            )
            other_result = tested_backend.random.rand()
            self.assertTrue(
                np.allclose(expected_result, other_result),
                f"Failed with {tested_backend}",
            )


if __name__ == "__main__":
    unittest.main()
