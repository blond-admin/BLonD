"""
**__init__.py: Locate and load the compiled blond library.**

:Authors: Konstantinos Iliakis**
"""


# todo add most important imports here to make it more accessible
def test():
    """A simple test to verify that blond has been correctly installed."""
    try:
        # Test that all the blond modules exist
        # Verify all the dependencies are also installed
        import __future__
        from packaging import version
        import platform
        import h5py
        import matplotlib
        import mpmath
        import numpy
        import scipy

        from . import (
            beam,
            impedances,
            input_parameters,
            llrf,
            monitors,
            plots,
            synchrotron_radiation,
            toolbox,
            trackers,
            utils,
        )

        # This should report the backend that is used
        from .utils import bmath

        bmath.report_backend()
        print("\nBLonD installed successfully!")
        return 0
    except Exception as exception:
        print("\nBLonD installation verification failed: ", exception)
        return -1
