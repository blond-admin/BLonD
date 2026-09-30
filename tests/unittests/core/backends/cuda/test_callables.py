# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Tests the eager CUDA wrappers in ``blond/core/backends/cuda/callables.py``.

They call `CudaSpecials` directly on CuPy arrays, without switching the
global backend.
"""

import unittest

import numpy as np
import pytest

from blond.core.backends.python.callables import PythonSpecials
from blond.testing.backend_testing import BLonDTestCase

try:
    import cupy as cp  # type: ignore
    from cupy.cuda import memory_hook  # type: ignore

    _HAS_CUPY = True
except ModuleNotFoundError:
    _HAS_CUPY = False

# Beyond the parameter-space capacity (`DRIFT_EXACT_MAX_INLINE_ALPHA`).
_ALPHA_ORDERS = tuple(range(11))


def _drift_exact_arguments(n_alpha):
    """Particles and parameters of a `drift_exact` call on the host."""
    return {
        "dt": np.linspace(1e-9, 10e-9, 1000),
        "dE": np.linspace(-1e8, 1e8, 1000),
        "T": 2e-5,
        "alpha_0": 3e-3,
        "higher_alpha": np.array(
            [1e-3 * (1.0 + 0.5 * k) for k in range(n_alpha)]
        ),
        "beta": 0.9,
        "energy": 10e9,
    }


# The drift cancels to a few ulp of ``T``, not of the result: see
# `test_drift_exact_accuracy_at_large_dE` in test_backend.py.
_ATOL = 8 * _drift_exact_arguments(0)["T"] * np.finfo(np.float64).eps


@pytest.mark.cupy
@unittest.skipUnless(_HAS_CUPY, "Requires CuPy")
class TestCudaDriftExact(BLonDTestCase):
    """Eager `CudaSpecials.drift_exact`."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from blond.core.backends.cuda.callables import CudaSpecials

        cls.specials = CudaSpecials

    def _run_cuda(self, arguments, higher_alpha=None):
        """Run the CUDA kernel on a device copy of `arguments`."""
        dt = cp.asarray(arguments["dt"])
        dE = cp.asarray(arguments["dE"])
        self.specials.drift_exact(
            dt=dt,
            dE=dE,
            T=arguments["T"],
            alpha_0=arguments["alpha_0"],
            higher_alpha=(
                arguments["higher_alpha"]
                if higher_alpha is None
                else higher_alpha
            ),
            beta=arguments["beta"],
            energy=arguments["energy"],
        )
        return dt.get()

    def _run_python(self, arguments):
        """Run the Python reference on a copy of `arguments`."""
        dt = arguments["dt"].copy()
        PythonSpecials.drift_exact(
            dt=dt,
            dE=arguments["dE"].copy(),
            T=arguments["T"],
            alpha_0=arguments["alpha_0"],
            higher_alpha=arguments["higher_alpha"],
            beta=arguments["beta"],
            energy=arguments["energy"],
        )
        return dt

    def test_host_coefficients_need_no_device_memory(self):
        """Host coefficients are passed by value, not copied to the GPU.

        A per-call ``cp.asarray`` of the coefficients allocates device
        memory and makes a synchronous pageable copy on every turn. Every
        such copy goes through CuPy's memory pool, so counting the pool's
        allocations during the call catches it.
        """
        for n_alpha in range(9):
            with self.subTest(n_alpha=n_alpha):
                arguments = _drift_exact_arguments(n_alpha)
                dt = cp.asarray(arguments["dt"])
                dE = cp.asarray(arguments["dE"])
                counter = _MallocCounter()
                with counter:
                    self.specials.drift_exact(
                        dt=dt,
                        dE=dE,
                        T=arguments["T"],
                        alpha_0=arguments["alpha_0"],
                        higher_alpha=arguments["higher_alpha"],
                        beta=arguments["beta"],
                        energy=arguments["energy"],
                    )
                self.assertEqual(counter.n_mallocs, 0)

    def test_host_coefficients_match_python(self):
        """Every number of host coefficients, inline and beyond, agrees."""
        for n_alpha in _ALPHA_ORDERS:
            with self.subTest(n_alpha=n_alpha):
                arguments = _drift_exact_arguments(n_alpha)
                np.testing.assert_allclose(
                    self._run_cuda(arguments),
                    self._run_python(arguments),
                    rtol=1e-12,
                    atol=_ATOL,
                )

    def test_device_coefficients_match_python(self):
        """The device-array compatibility path agrees too."""
        for n_alpha in _ALPHA_ORDERS:
            with self.subTest(n_alpha=n_alpha):
                arguments = _drift_exact_arguments(n_alpha)
                np.testing.assert_allclose(
                    self._run_cuda(
                        arguments, cp.asarray(arguments["higher_alpha"])
                    ),
                    self._run_python(arguments),
                    rtol=1e-12,
                    atol=_ATOL,
                )


if _HAS_CUPY:

    class _MallocCounter(memory_hook.MemoryHook):
        """Counts the allocations of CuPy's memory pool."""

        name = "MallocCounter"

        def __init__(self):
            self.n_mallocs = 0

        def malloc_preprocess(self, **kwargs):
            self.n_mallocs += 1
