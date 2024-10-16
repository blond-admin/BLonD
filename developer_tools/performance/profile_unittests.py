import os.path
import unittest


def discover_and_run_tests():
    # Define the directory containing the tests
    test_dir = os.path.abspath(os.path.dirname(__file__) + "../../../unittests/")

    # Discover all test cases in the specified directory
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir)

    # Create a test runner
    runner = unittest.TextTestRunner()

    # Run the tests
    runner.run(suite)


def run_pytests():
    import pytest
    test_dir = os.path.abspath(os.path.dirname(__file__) + "../../../unittests/")

    # Run pytest with the specified directory
    exit_code = pytest.main([test_dir])
    return exit_code


def main():
    # setup profiler
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

    # discover_and_run_tests()  # this gets profiled
    run_pytests()
    # collect profiling infos
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    result = s.getvalue()
    print(result[:10000])
    with open("unittests_performance.txt", "w") as file:
        file.write(result)


if __name__ == '__main__':
    main()
