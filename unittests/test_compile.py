import unittest

from blond.compile import compile_cpp_library, compile_cuda_library, main, run_compile


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_compile_cpp_library(self):
        # TODO: implement test for `compile_cpp_library`
        compile_cpp_library(
            args=None, cflags=None, float_flags=None, libs=None, cpp_files=None
        )

    @unittest.skip
    def test_compile_cuda_library(self):
        # TODO: implement test for `compile_cuda_library`
        compile_cuda_library(
            args=None, nvccflags=None, float_flags=None, cuda_files=None, nvcc=None
        )

    @unittest.skip
    def test_main(self):
        # TODO: implement test for `main`
        main()

    @unittest.skip
    def test_run_compile(self):
        # TODO: implement test for `run_compile`
        run_compile(command=None, libname=None)
