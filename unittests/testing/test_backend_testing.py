import unittest

import blond.core.backends.backend as backend
import blond.testing.backend_testing as bend_test


class InvalidBackendTestError(Exception): ...


class InvalidBackend(backend.Numpy32Bit):
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

        if self.init_backend == "Numpy32Bit":
            backend.backend.change_backend(backend.Numpy64Bit)
        else:
            backend.backend.change_backend(backend.Numpy32Bit)

        test_init_backend = backend.backend.__class__

        @bend_test.multi_backend_testcase
        def a_test(self):
            used_backends.append(backend.backend.__class__.__name__)

        a_test(self)

        self.assertListEqual(
            used_backends, list(backend.AVAILABLE_BACKENDS.keys())
        )
        self.assertTrue(backend.backend.__class__ is test_init_backend)

    def test_multi_backend_testcase_with_forcing(self):
        used_backends = []
        bend_test.FORCE_ALL_BACKENDS = True

        @bend_test.multi_backend_testcase
        def a_test(self):
            used_backends.append(backend.backend.__class__.__name__)

        with self.assertRaises(InvalidBackendTestError):
            a_test(self)

        self.assertListEqual(
            used_backends, list(backend.AVAILABLE_BACKENDS.keys())
        )
