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
        import __future__  # NOQA
        from packaging import version  # NOQA
        import platform  # NOQA
        import h5py  # NOQA
        import matplotlib  # NOQA
        import mpmath  # NOQA
        import numpy  # NOQA
        import scipy  # NOQA

        assert version.parse(matplotlib.__version__) >= version.parse("3.7"), (
            f"matplotlib version {matplotlib.__version__} does not meet minimum required version 3.7"
        )
        assert version.parse(platform.python_version()) >= version.parse(
            "3.8"
        ), (
            f"python version {platform.python_version()} does not meet minimum required version 3.8"
        )
        assert version.parse(numpy.__version__) >= version.parse("1.20"), (
            "numpy version {numpy.__version__} does not meet minimum required version 1.20"
        )

        from . import (
            beam,  # NOQA
            impedances,  # NOQA
            input_parameters,  # NOQA
            llrf,  # NOQA
            monitors,  # NOQA
            plots,  # NOQA
            synchrotron_radiation,  # NOQA
            toolbox,  # NOQA
            trackers,  # NOQA
            utils,  # NOQA
        )

        # This should report the backend that is used
        from .utils import bmath

        bmath.report_backend()
        print("\nBLonD installed successfully!")
        return 0
    except Exception as exception:
        print("\nBLonD installation verification failed: ", exception)
        return -1
