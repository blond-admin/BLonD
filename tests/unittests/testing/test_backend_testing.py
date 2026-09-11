import os
import unittest
from unittest import mock

import numpy as np
import pytest

import blond.testing.backend_testing as bend_test
from blond.core.backends import backend

try:
    import cupy

    cupy_available = True
except (ModuleNotFoundError, ImportError):
    cupy_available = False


class InvalidBackendTestError(Exception): ...


class InvalidBackend(backend.Numpy64Bit):
    def __init__(self):
        raise InvalidBackendTestError


class TestBackendTesting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flag_init = bend_test.FORCE_ALL_BACKENDS
        cls.all_init = backend.ALL_BACKENDS.copy()
        backend.ALL_BACKENDS["Invalid"] = InvalidBackend

    @classmethod
    def tearDownClass(cls):
        bend_test.FORCE_ALL_BACKENDS = cls.flag_init
        backend.ALL_BACKENDS.clear()
        for k, v in cls.all_init.items():
            backend.ALL_BACKENDS[k] = v

    def setUp(self):
        self.init_backend = backend.backend.__class__

    def tearDown(self):
        backend.backend.change_backend(self.init_backend)

    def test_set_forcing(self):
        flag_str = "BLOND_FORCE_TEST_ALL_BACKENDS"
        init = os.environ.get(flag_str, "False")

        os.environ[flag_str] = "True"
        self.assertTrue(bend_test._set_forcing())
        os.environ[flag_str] = "False"
        self.assertFalse(bend_test._set_forcing())

        os.environ[flag_str] = "Test"
        with self.assertRaises(EnvironmentError):
            bend_test._set_forcing()

        os.environ["BLOND_FORCE_TEST_ALL_BACKENDS"] = init

    def test_backend_selection(self):
        bend_test.FORCE_ALL_BACKENDS = False

        available_list = bend_test._backend_selection(
            *backend.ALL_BACKENDS.keys()
        )

        bend_test.FORCE_ALL_BACKENDS = True
        all_list = bend_test._backend_selection(*backend.ALL_BACKENDS.keys())

        self.assertListEqual(
            available_list, list(backend.AVAILABLE_BACKENDS.values())
        )
        self.assertListEqual(all_list, list(backend.ALL_BACKENDS.values()))

        self.assertFalse(InvalidBackend in available_list)
        self.assertTrue(InvalidBackend in all_list)

    @pytest.mark.cupy
    @unittest.skipIf(not cupy_available, "Cupy not found")
    def test_backend_validity(self):
        bend_test.FORCE_ALL_BACKENDS = False
        available_list = bend_test._backend_selection(
            *backend.ALL_BACKENDS.keys()
        )

        for b_end in available_list:
            b_end()

        bend_test.FORCE_ALL_BACKENDS = True
        all_list = bend_test._backend_selection(*backend.ALL_BACKENDS.keys())

        for b_end in all_list:
            if b_end is not InvalidBackend:
                b_end()
            else:
                with self.assertRaises(InvalidBackendTestError):
                    b_end()

    def test_warning(self):
        bend_test.FORCE_ALL_BACKENDS = False
        with self.assertWarns(Warning):
            bend_test._backend_selection(*backend.ALL_BACKENDS.keys())

        bend_test.FORCE_ALL_BACKENDS = True
        # Should pass if no warning, but there's no test for that
        # Catch and suppress AssertionError, which is thrown if warning
        # is not raised.  If no AssertionError, warning WAS received,
        # therefore an AssertionError SHOULD be raised.
        try:
            with self.assertWarns(Warning):
                bend_test._backend_selection(*backend.ALL_BACKENDS.keys())
        except AssertionError:
            pass
        else:
            raise AssertionError("Warning should not have been raised")

    def test_multi_backend_testcase_no_forcing(self):
        used_backends = []

        bend_test.FORCE_ALL_BACKENDS = False

        if self.init_backend == "Numpy64Bit":
            backend.backend.change_backend(backend.Cupy64Bit)
        else:
            backend.backend.change_backend(backend.Numpy64Bit)

        test_init_backend = backend.backend.__class__

        @bend_test.multi_backend_testcase
        def a_test(self):
            used_backends.append(backend.backend.__class__.__name__)

        a_test(self)

        self.assertListEqual(
            used_backends, list(backend.AVAILABLE_BACKENDS.keys())
        )
        self.assertTrue(backend.backend.__class__ is test_init_backend)

    @pytest.mark.cupy
    @unittest.skipIf(not cupy_available, "Cupy not found")
    def test_multi_backend_testcase_with_forcing(self):
        used_backends = []
        bend_test.FORCE_ALL_BACKENDS = True

        class DummyCase(unittest.TestCase):
            @bend_test.multi_backend_testcase
            def test_method(self):
                used_backends.append(backend.backend.__class__.__name__)

        result = unittest.TestResult()
        DummyCase("test_method").run(result)

        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.failures, [])
        self.assertEqual(len(result.errors), 1)
        failing_subtest, _ = result.errors[0]
        self.assertEqual(failing_subtest.params["backend"], "InvalidBackend")

        self.assertListEqual(
            used_backends, list(backend.AVAILABLE_BACKENDS.keys())
        )

    def test_multi_backend_testcase_failsafe(self):
        bend_test.FORCE_ALL_BACKENDS = False

        if self.init_backend == "Numpy64Bit":
            backend.backend.change_backend(backend.Cupy64Bit)
        else:
            backend.backend.change_backend(backend.Numpy64Bit)

        test_init_backend = backend.backend.__class__

        # Each test spawns a subtest for each backend, testing requires
        # a new TestCase and TestResult to handle the behaviour.
        class ErrorCase(unittest.TestCase):
            @bend_test.multi_backend_testcase
            def test_method(self):
                raise RuntimeError

        class FailureCase(unittest.TestCase):
            @bend_test.multi_backend_testcase
            def test_method(self):
                self.fail("deliberate assertion failure")

        error_result = unittest.TestResult()
        ErrorCase("test_method").run(error_result)

        failure_result = unittest.TestResult()
        FailureCase("test_method").run(failure_result)

        self.assertFalse(error_result.wasSuccessful())
        self.assertFalse(failure_result.wasSuccessful())

        # Assertion errors route through TestResult.failure, other
        # exceptions route through TestResult.errors
        self.assertEqual(error_result.failures, [])
        self.assertEqual(
            len(error_result.errors), len(backend.AVAILABLE_BACKENDS)
        )
        self.assertEqual(failure_result.errors, [])
        self.assertEqual(
            len(failure_result.failures), len(backend.AVAILABLE_BACKENDS)
        )

        for result_list in (error_result.errors, failure_result.failures):
            tested_backends = [
                test.params["backend"] for test, _ in result_list
            ]
            self.assertListEqual(
                tested_backends, list(backend.AVAILABLE_BACKENDS.keys())
            )

        self.assertTrue(backend.backend.__class__ is test_init_backend)

    def test_array_like_scan(self):
        types = [list, tuple, np.array]
        if cupy_available:
            types.append(cupy.array)
        scanner = bend_test.ArrayLikeScan(types)

        inp_1 = [1, 2, 3]
        inp_2 = (1, 2, 3)
        inp_3 = np.array([1, 2, 3])

        inputs = [inp_1, inp_2, inp_3]

        if cupy_available:
            inp_4 = cupy.array([1, 2, 3])
            inputs.append(inp_4)

        for input_array_like in inputs:
            for i, inp_cast in enumerate(scanner):
                cast = inp_cast(input_array_like)

                if i < 2:
                    self.assertIsInstance(cast, types[i])
                elif i == 2:
                    self.assertIsInstance(cast, np.ndarray)
                elif i == 3:
                    self.assertIsInstance(cast, cupy.ndarray)


class TestPinFastTestBackends(unittest.TestCase):
    """Tests for the autouse-fixture helper that pins fast test backends."""

    def setUp(self):
        self.init_backend = backend.backend.__class__
        self.init_specials = backend.backend.specials_mode
        # The python/numba specials only exist on the CPU backend; the GPU
        # backend (Cupy64Bit) accepts only "cuda". Pin Numpy64Bit so these
        # tests behave the same regardless of the ambient backend.
        backend.backend.change_backend(backend.Numpy64Bit)

    def tearDown(self):
        backend.backend.change_backend(self.init_backend)
        if backend.backend.specials_mode != self.init_specials:
            backend.backend.set_specials(self.init_specials)

    @pytest.mark.backend_mutation
    def test_resets_blond3_python_specials(self):
        # The pure-python kernels are slow; the helper must move the ambient
        # default off "python" (tests that want python set it themselves).
        backend.backend.set_specials("python")
        self.assertEqual(backend.backend.specials_mode, "python")

        bend_test.pin_fast_test_backends()

        self.assertNotEqual(backend.backend.specials_mode, "python")

    @pytest.mark.backend_mutation
    def test_leaves_non_python_blond3_specials_untouched(self):
        backend.backend.set_specials("numba")

        bend_test.pin_fast_test_backends()

        self.assertEqual(backend.backend.specials_mode, "numba")

    @pytest.mark.backend_mutation
    def test_resets_legacy_blond2_python_backend(self):
        from blond.legacy.blond2.utils import bmath as bm

        bm.use_py()
        self.assertEqual(type(bm).__name__, "PyBackend")

        bend_test.pin_fast_test_backends()

        self.assertNotEqual(type(bm).__name__, "PyBackend")


class LeakedBackend(backend.Numpy64Bit):
    """Stand-in for an array backend left active by an earlier test."""


class TestPinFastTestBackends(unittest.TestCase):
    """`pin_fast_test_backends` must reset the *array* backend too.

    A test (or a module imported during collection) that switches the
    global array backend and does not restore it would otherwise leave
    every following test running on the wrong array namespace.
    """

    def setUp(self):
        self.init_backend = backend.backend.__class__

    def tearDown(self):
        backend.backend.change_backend(self.init_backend)

    def test_restores_numpy_backend_for_a_cpu_environment(self):
        with mock.patch.dict(os.environ, {"BLOND_BACKEND_MODE": "numba"}):
            backend.backend.change_backend(LeakedBackend)
            self.assertIs(backend.backend.__class__, LeakedBackend)

            bend_test.pin_fast_test_backends()

            self.assertIs(backend.backend.__class__, backend.Numpy64Bit)

    @pytest.mark.cupy
    @unittest.skipUnless(cupy_available, "Requires cupy")
    def test_restores_cupy_backend_for_a_cuda_environment(self):
        with mock.patch.dict(os.environ, {"BLOND_BACKEND_MODE": "cuda"}):
            backend.backend.change_backend(backend.Numpy64Bit)

            bend_test.pin_fast_test_backends()

            self.assertIs(backend.backend.__class__, backend.Cupy64Bit)


class DummyBackendAwareCase(bend_test.BLonDTestCase):
    """Test methods driven by `TestBackendAwareTestCase`, not run directly."""

    # Exclude from pytest's own collection - these are fixtures run
    # manually by `TestBackendAwareTestCase`, not real tests.
    __test__ = False

    def test_pass(self):
        self.assertTrue(True)

    def test_fail(self):
        self.assertTrue(False)

    def test_error(self):
        raise ValueError("boom")

    def test_skip(self):
        self.skipTest("skip reason")

    def test_mutates_then_fails(self):
        backend.backend.set_specials("numba")
        self.assertTrue(False)

    def test_changes_backend_and_specials_then_fails(self):
        backend.backend.change_backend(LeakedBackend)
        backend.backend.set_specials("numba")
        self.assertTrue(False)


class TestBackendAwareTestCase(unittest.TestCase):
    """Tests for `BackendAwareTestCase`."""

    def setUp(self):
        self.init_backend = backend.backend.__class__
        self.init_specials = backend.backend.specials_mode
        backend.backend.change_backend(backend.Numpy64Bit)
        backend.backend.set_specials("python")

    def tearDown(self):
        backend.backend.change_backend(self.init_backend)
        if backend.backend.specials_mode != self.init_specials:
            backend.backend.set_specials(self.init_specials)

    @staticmethod
    def _run(test_name):
        result = unittest.TestResult()
        DummyBackendAwareCase(test_name).run(result)
        return result

    def test_backend_state(self):
        self.assertEqual(bend_test._backend_state(), ("Numpy64Bit", "python"))

        backend.backend.set_specials("numba")
        self.assertEqual(bend_test._backend_state(), ("Numpy64Bit", "numba"))

    def test_passing_test_is_unaffected(self):
        result = self._run("test_pass")

        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.failures, [])
        self.assertEqual(result.errors, [])

    def test_failure_message_reports_backend_state(self):
        result = self._run("test_fail")

        self.assertEqual(len(result.failures), 1)
        _, traceback_text = result.failures[0]
        self.assertIn("False is not true", traceback_text)
        self.assertIn(
            "[backend at start: Numpy64Bit, at failure: Numpy64Bit]",
            traceback_text,
        )
        self.assertIn(
            "[specials at start: python, at failure: python]",
            traceback_text,
        )

    def test_error_message_reports_backend_state(self):
        result = self._run("test_error")

        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.failures, [])
        _, traceback_text = result.errors[0]
        self.assertIn("boom", traceback_text)
        self.assertIn(
            "[backend at start: Numpy64Bit, at failure: Numpy64Bit]",
            traceback_text,
        )
        self.assertIn(
            "[specials at start: python, at failure: python]",
            traceback_text,
        )

    def test_skip_is_not_annotated(self):
        result = self._run("test_skip")

        self.assertEqual(len(result.skipped), 1)
        _, reason = result.skipped[0]
        self.assertEqual(reason, "skip reason")

    @pytest.mark.backend_mutation
    def test_failure_reports_backend_changed_during_test(self):
        result = self._run("test_mutates_then_fails")

        self.assertEqual(len(result.failures), 1)
        _, traceback_text = result.failures[0]
        self.assertIn(
            "[backend at start: Numpy64Bit, at failure: Numpy64Bit]",
            traceback_text,
        )
        self.assertIn(
            "[specials at start: python, at failure: numba]",
            traceback_text,
        )

    @pytest.mark.backend_mutation
    def test_failure_reports_backend_class_changed_during_test(self):
        result = self._run("test_changes_backend_and_specials_then_fails")

        self.assertEqual(len(result.failures), 1)
        _, traceback_text = result.failures[0]
        self.assertIn(
            "[backend at start: Numpy64Bit, at failure: LeakedBackend]",
            traceback_text,
        )
        self.assertIn(
            "[specials at start: python, at failure: numba]",
            traceback_text,
        )
