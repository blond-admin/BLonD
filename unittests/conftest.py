"""
The fixtures to be shared among all tests in the test suite
"""
import pytest


def deactivate_savefig():
    def dummy_savefig(*args, **kwargs):
        pass

    from matplotlib import pyplot

    pyplot.savefig = dummy_savefig


@pytest.fixture(scope='session', autouse=True)
def setup_function():
    # This function will run once before all tests
    deactivate_savefig()  # To improve runtime of tests. Savefig is rather safe to be executed.
