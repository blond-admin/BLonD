import sys
import unittest
from unittest.mock import Mock, patch

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt

from blond.generals.cupy_.no_cupy_import import AllowPlotting, is_cupy_array
from blond.testing.backend_testing import BLonDTestCase


class _FakeCupyArray:
    # Stand-in for cupy.ndarray on CPU-only machines: explicit .get()
    # works, implicit conversion raises exactly like cupy, and
    # is_cupy_array() recognizes it through the duck-typed .device check.
    def __init__(self, data):
        self._data = np.asarray(data, dtype=float)
        self.device = "cuda:0"

    def get(self):
        return self._data.copy()

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        # cupy indexing returns cupy arrays, never numpy ones; staying
        # "on device" here is what forces the conversion through
        # __array__ like on a real GPU.
        result = self._data[item]
        if isinstance(result, np.ndarray):
            return _FakeCupyArray(result)
        return result

    def __array__(self, *args, **kwargs):
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. "
            "Please use `.get()` to construct a NumPy array explicitly."
        )


class TestAllowPlottingWithoutCupy(BLonDTestCase):
    # Regression tests for the EX_06 cuda64 CI failure: matplotlib
    # converts line data via np.asanyarray (cbook._to_unmasked_float_array),
    # which AllowPlotting must cover in addition to np.asarray/np.array.
    # The fake device array makes this testable without a GPU.

    def setUp(self):
        self.fake = _FakeCupyArray([1.0, 2.0, 3.0])

    def test_fake_is_recognized_as_device_array(self):
        self.assertTrue(is_cupy_array(self.fake))

    def test_asanyarray_inside_allow_plotting(self):
        with AllowPlotting():
            result = np.asanyarray(self.fake, float)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_asarray_inside_allow_plotting(self):
        with AllowPlotting():
            result = np.asarray(self.fake)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_array_inside_allow_plotting(self):
        with AllowPlotting():
            result = np.array(self.fake)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_plot_device_arrays_inside_allow_plotting(self):
        # End-to-end mirror of the EX_06 crash:
        # plt.plot(ts, potential_well) with device arrays.
        matplotlib.use("Agg")
        ys = _FakeCupyArray([4.0, 5.0, 6.0])
        plt.figure()
        with AllowPlotting():
            plt.plot(self.fake, ys)
        plt.close()

    def test_no_override_leaks_outside_context(self):
        with AllowPlotting():
            np.asanyarray(self.fake, float)
        with self.assertRaisesRegex(TypeError, "Implicit conversion"):
            np.asanyarray(self.fake, float)
        with self.assertRaisesRegex(TypeError, "Implicit conversion"):
            np.asarray(self.fake)


class TestFunctions(BLonDTestCase):
    @pytest.mark.cupy
    @pytest.mark.backend_mutation
    def test_allow_plotting(self) -> None:
        try:
            import cupy as cp  # type: ignore
        except ImportError as exc:
            self.skipTest(str(exc))
        from blond import AllowPlotting

        # demo of AllowPlotting
        array = cp.array([1, 2, 23])
        array2 = cp.array([1, 2, 25])
        plt.figure()
        from blond.core.backends.backend import Cupy64Bit, backend

        backend_org = type(backend)
        backend.change_backend(Cupy64Bit)
        with AllowPlotting():
            # would crash without AllowPlotting
            # TypeError: Implicit conversion to a NumPy array is not allowed. Please use `.get()` to construct a NumPy array explicitly.
            plt.plot(array)
            plt.plot(array2)
        with self.assertRaisesRegex(
            TypeError, "Implicit conversion to a NumPy array is not allowed."
        ):
            plt.plot(array)  # should not work outside AllowPlotting

        backend.change_backend(backend_org)
        plt.close()

    @pytest.mark.cupy
    def test_scatter(self):
        try:
            import cupy as cp  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))

        from matplotlib import pyplot as plt

        import blond.generals.cupy_.no_cupy_import as no_cupy

        y = cp.ones(12)

        with no_cupy.AllowPlotting():
            plt.scatter(y, y)
        plt.close("all")

    @pytest.mark.cupy
    def test_is_cupy_array(self):
        try:
            import cupy as cp  # type: ignore
            from numba import cuda
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))

        # is_cupy_array() recognizes a numba.cuda DeviceNDArray purely by
        # duck-typing its `gpu_data` attribute, so a spec-restricted Mock
        # exercises that branch without creating a real CUDA context.
        # (A real `cuda.to_device()` call was used here previously, but
        # initializing a CUDA context has been observed to segfault the
        # whole test process on the shared MIG-sliced GPU CI runner.)
        numba_array_dummy = Mock(spec=[])
        numba_array_dummy.gpu_data = True
        with self.assertRaises(TypeError):
            is_cupy_array(numba_array_dummy)

        self.assertEqual(is_cupy_array(cp.ones(10)), True)

        self.assertEqual(is_cupy_array(np.ones(10)), False)
        self.assertEqual(is_cupy_array([1, 2, 3]), False)
        self.assertEqual(is_cupy_array("Not an array"), False)
        self.assertEqual(is_cupy_array(cuda), False)


class TestImportCupy(BLonDTestCase):
    """`import_cupy_with_error_hint` turns a missing CuPy into an actionable error."""

    def test_returns_module_when_cupy_is_installed(self):
        cupy = pytest.importorskip("cupy")

        from blond.generals.cupy_ import import_cupy_with_error_hint

        self.assertIs(import_cupy_with_error_hint(), cupy)

    def test_raises_module_not_found_when_cupy_is_missing(self):
        from blond.generals.cupy_ import import_cupy_with_error_hint

        # `None` in sys.modules makes `import cupy` raise
        # ModuleNotFoundError, which is exactly what a CPU-only (e.g.
        # aarch64) install does -- without uninstalling anything.
        with patch.dict(sys.modules, {"cupy": None}):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                import_cupy_with_error_hint()

        message = str(ctx.exception)
        # The message must name what is missing, how to install it, and
        # the CPU way out. A bare "No module named 'cupy'" does none of
        # these.
        self.assertIn("CuPy", message)
        self.assertIn("gpu_cuda12", message)
        self.assertIn("BLOND_BACKEND_MODE", message)
        # The original exception stays attached for debugging.
        self.assertIsInstance(ctx.exception.__cause__, ModuleNotFoundError)


class TestCudaBackendWithoutCupy(BLonDTestCase):
    """Requesting the GPU without CuPy fails with a readable error."""

    def test_cupy_backend_reports_missing_cupy(self):
        from blond.core.backends.backend import Cupy64Bit

        # patch.dict restores the whole mapping on exit, so evicting the
        # CUDA callables module here cannot leak into other tests.
        with patch.dict(sys.modules):
            # The module may already be imported from an earlier test;
            # evict it so its top-level cupy import runs again.
            sys.modules.pop("blond.core.backends.cuda.callables", None)
            sys.modules["cupy"] = None
            with self.assertRaises(ModuleNotFoundError) as ctx:
                Cupy64Bit()

        self.assertIn("gpu_cuda12", str(ctx.exception))

    def test_cpu_backend_explains_that_cuda_needs_a_gpu_backend(self):
        from blond.core.backends.backend import Numpy64Bit
        from blond.generals.exceptions_ import UnknownBackendMode

        with self.assertRaises(UnknownBackendMode) as ctx:
            Numpy64Bit().set_specials("cuda")

        message = str(ctx.exception)
        # Must not read as "'cuda' is not a valid mode" -- it is, just
        # not for a CPU backend.
        self.assertIn("GPU", message)
        self.assertIn("Cupy64Bit", message)
