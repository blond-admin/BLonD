import unittest
from copy import deepcopy

import numpy as np
import pytest

from blond.utils import bmath as bm


class TestCompareWraps:
    # Run before every test
    def setup_method(self):
        pass

    # Run after every test
    def teardown_method(self):
        bm.use_cpu()

    @pytest.mark.parametrize(
        "precision, bm_use_other",
        [
            ("single", bm.use_numba),
            ("double", bm.use_numba),
            ("single", bm.use_cpp),
            ("double", bm.use_cpp),
            ("single", bm.use_gpu),
            ("double", bm.use_gpu),
        ],
    )
    def test_drift_simple(self, precision, bm_use_other):
        self._test_drift(precision, bm_use_other, solver="simple")

    @pytest.mark.parametrize(
        "precision, bm_use_other",
        [
            ("single", bm.use_numba),
            ("double", bm.use_numba),
            ("single", bm.use_cpp),
            ("double", bm.use_cpp),
            ("single", bm.use_gpu),
            ("double", bm.use_gpu),
        ],
    )
    def test_drift_legacy(self, precision, bm_use_other):
        self._test_drift(precision, bm_use_other, solver="legacy")

    @pytest.mark.parametrize(
        "precision, bm_use_other",
        [
            ("single", bm.use_numba),
            ("double", bm.use_numba),
            ("single", bm.use_cpp),
            ("double", bm.use_cpp),
            ("single", bm.use_gpu),
            ("double", bm.use_gpu),
        ],
    )
    def test_drift_legacy(self, precision, bm_use_other):
        self._test_drift(precision, bm_use_other, solver="legacy")

    def _test_drift(self, precision, bm_use_other, solver):
        n_particles = 12

        bm.use_py()
        bm.use_precision(_precision=precision)

        _dE = np.random.normal(loc=0, scale=1e7, size=n_particles).astype(
            bm.precision.real_t
        )[:]
        _dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles).astype(
            bm.precision.real_t
        )[:]

        t_rev = np.random.rand()
        length_ratio = np.random.uniform()
        eta_0, eta_1, eta_2 = np.random.randn(3)
        alpha_0, alpha_1, alpha_2 = np.random.randn(3)
        beta = np.random.rand()
        energy = np.random.rand()
        alpha_order = 2
        bm.use_py()
        bm.use_precision(_precision=precision)
        dE = bm.array(deepcopy(_dE), dtype=bm.precision.real_t)
        dt = bm.array(deepcopy(_dt), dtype=bm.precision.real_t)
        bm.drift(
            dt,
            dE,
            solver,
            t_rev,
            length_ratio,
            alpha_order,
            eta_0,
            eta_1,
            eta_2,
            alpha_0,
            alpha_1,
            alpha_2,
            beta,
            energy,
        )

        try:
            bm_use_other()
        except ModuleNotFoundError as exc:
            unittest.skip(str(exc))
        bm.use_precision(_precision=precision)

        dE2 = bm.array(deepcopy(_dE), dtype=bm.precision.real_t)
        dt2 = bm.array(deepcopy(_dt), dtype=bm.precision.real_t)
        bm.drift(
            dt2,
            dE2,
            solver,
            t_rev,
            length_ratio,
            alpha_order,
            eta_0,
            eta_1,
            eta_2,
            alpha_0,
            alpha_1,
            alpha_2,
            beta,
            energy,
        )

        try:
            dt2 = np.array(dt2)
            dE2 = np.array(dE2)
        except TypeError:
            dt2 = dt2.get()
            dE2 = dE2.get()
        if precision == "single":
            rtol = 1e-5
        elif precision == "double":
            rtol = 1e-8
        np.testing.assert_allclose(dt2, dt, rtol=rtol)
        np.testing.assert_allclose(dE2, dE, rtol=rtol)

    @pytest.mark.parametrize(
        "precision, bm_use_other",
        [
            ("single", bm.use_numba),
            ("double", bm.use_numba),
            ("single", bm.use_cpp),
            ("double", bm.use_cpp),
            ("single", bm.use_gpu),
            ("double", bm.use_gpu),
        ],
    )
    def test_kick(self, precision, bm_use_other):
        n_particles = 12

        bm.use_py()
        bm.use_precision(_precision=precision)

        _dE = np.random.normal(loc=0, scale=1e7, size=n_particles).astype(
            bm.precision.real_t
        )[:]
        _dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles).astype(
            bm.precision.real_t
        )[:]

        n_rf = 3
        charge = 1.5
        acceleration_kick = 1e3 * np.random.rand()
        voltage = np.random.randn(n_rf)[:].astype(bm.precision.real_t)
        omega_rf = np.random.randn(n_rf)[:].astype(bm.precision.real_t)
        phi_rf = np.random.randn(n_rf)[:].astype(bm.precision.real_t)

        bm.use_py()
        bm.use_precision(_precision=precision)
        dE = bm.array(deepcopy(_dE), dtype=bm.precision.real_t)
        dt = bm.array(deepcopy(_dt), dtype=bm.precision.real_t)
        bm.kick(
            dt,
            dE,
            bm.array(voltage, dtype=bm.precision.real_t),
            bm.array(omega_rf, dtype=bm.precision.real_t),
            bm.array(phi_rf, dtype=bm.precision.real_t),
            charge,
            n_rf,
            acceleration_kick,
        )

        try:
            bm_use_other()
        except ModuleNotFoundError as exc:
            unittest.skip(str(exc))
        bm.use_precision(_precision=precision)

        dE2 = bm.array(deepcopy(_dE), dtype=bm.precision.real_t)
        dt2 = bm.array(deepcopy(_dt), dtype=bm.precision.real_t)
        bm.kick(
            dt2,
            dE2,
            bm.array(voltage, dtype=bm.precision.real_t),
            bm.array(omega_rf, dtype=bm.precision.real_t),
            bm.array(phi_rf, dtype=bm.precision.real_t),
            charge,
            n_rf,
            acceleration_kick,
        )

        try:
            dt2 = np.array(dt2)
            dE2 = np.array(dE2)
        except TypeError:
            dt2 = dt2.get()
            dE2 = dE2.get()
        if precision == "single":
            rtol = 1e-5
        elif precision == "double":
            rtol = 1e-8
        np.testing.assert_allclose(dt2, dt, rtol=rtol)
        np.testing.assert_allclose(dE2, dE, rtol=rtol)

    @pytest.mark.parametrize(
        "precision, bm_use_other",
        [
            ("single", bm.use_numba),
            ("double", bm.use_numba),
            ("single", bm.use_cpp),
            ("double", bm.use_cpp),
            ("single", bm.use_gpu),
            ("double", bm.use_gpu),
        ],
    )
    def test_slice(self, precision, bm_use_other):
        n_particles = 12

        bm.use_py()
        bm.use_precision(_precision=precision)

        _dE = np.random.normal(loc=0, scale=1e7, size=n_particles).astype(
            bm.precision.real_t
        )[:]
        _dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles).astype(
            bm.precision.real_t
        )[:]
        cut_left = np.min(_dt) - 1e-8
        cut_right = np.max(_dt) + 1e-8

        _profile = np.empty(64).astype(bm.precision.real_t)
        _weights = 1e3 * np.random.rand(len(_dt))
        bm.use_py()
        bm.use_precision(_precision=precision)
        dE = bm.array(deepcopy(_dE), dtype=bm.precision.real_t)
        dt = bm.array(deepcopy(_dt), dtype=bm.precision.real_t)
        profile = bm.array(deepcopy(_profile), dtype=bm.precision.real_t)
        weights = bm.array(deepcopy(_weights), dtype=np.int32)

        bm.slice_beam(
            dt=dt,
            profile=profile,
            cut_left=cut_left,
            cut_right=cut_right,
            weights=weights,
        )

        try:
            bm_use_other()
        except ModuleNotFoundError as exc:
            unittest.skip(str(exc))
        bm.use_precision(_precision=precision)

        dE2 = bm.array(deepcopy(_dE), dtype=bm.precision.real_t)
        dt2 = bm.array(deepcopy(_dt), dtype=bm.precision.real_t)
        profile2 = bm.array(deepcopy(_profile), dtype=bm.precision.real_t)
        weights2 = bm.array(deepcopy(_weights), dtype=np.int32)

        bm.slice_beam(
            dt=dt2,
            profile=profile2,
            cut_left=cut_left,
            cut_right=cut_right,
            weights=weights2,
        )

        try:
            dt2 = np.array(dt2)
            dE2 = np.array(dE2)
        except TypeError:
            dt2 = dt2.get()
            dE2 = dE2.get()
        if precision == "single":
            rtol = 1e-5
        elif precision == "double":
            rtol = 1e-8
        np.testing.assert_allclose(dt2, dt, rtol=rtol)
        np.testing.assert_allclose(dE2, dE, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
