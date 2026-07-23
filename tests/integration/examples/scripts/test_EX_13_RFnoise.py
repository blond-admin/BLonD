import os
import unittest

import pytest

from blond.core.backends.backend import (
    Cupy64Bit,
    Numpy64Bit,
    backend,
)
from blond.interfaces.rf_noise_cpp.wrap_rf_noise import (
    rf_noise_library_available,
)

# In CI the rf-noise-cpp library is provided, so this test must run (and fail
# loudly if the library is missing) rather than silently skip. Locally, skip
# gracefully when the library is unavailable. GitLab sets ``CI=true``.
_RUN_RF_NOISE = os.environ.get("CI") == "true" or rf_noise_library_available()


@unittest.skipUnless(
    _RUN_RF_NOISE,
    "rf-noise-cpp library not available",
)
class TestEX_13_RFnoise(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_13_RFnoise

        # full script. just checking if it crashes
        EX_13_RFnoise.main(n_turns=200)

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_13_RFnoise

        # full script. just checking if it crashes
        EX_13_RFnoise.main(n_turns=200)
        backend.zeros(100)
