import unittest
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from blond import Beam, backend, uranium_29
from blond.core.beam.beams import ProbeBeam
from blond.physics.profiles_sparse import EquidistantMultiProfile


class TestEquidistantMultiProfile(unittest.TestCase):
    def setUp(self):
        self.multiprofile_equidistant = EquidistantMultiProfile.headless(
            t_rev=5 * 10.0,
            filling_pattern=np.ones(5, bool),
            bins_per_profile=4,
            offset=0,
        )
        start = float(self.multiprofile_equidistant.profiles[0].hist_x[0])
        stop = float(self.multiprofile_equidistant.profiles[-1].hist_x[-1])

        base = np.linspace(
            start,
            stop,
            5 * 4,
            endpoint=True,
        )
        dt = np.concatenate([[b] * int(b) for b in base])
        self.beam = ProbeBeam(
            dt=dt,
            particle_type=uranium_29,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_track(self):
        DEV_DRAW = False  # TODO false
        backend.set_specials(
            "cpp"  # TODO remove
        )

        independent_profiles = deepcopy(self.multiprofile_equidistant.profiles)

        self.multiprofile_equidistant.track(self.beam)

        for profile_expected in independent_profiles:
            profile_expected.track(self.beam)
            if DEV_DRAW:
                profile_expected.plot()
        if DEV_DRAW:
            self.multiprofile_equidistant.plot(linestyle="--")
        for i, profile_expected in enumerate(independent_profiles):
            profile_actual = self.multiprofile_equidistant.profiles[i]
            np.testing.assert_allclose(
                profile_actual.hist_x, profile_expected.hist_x
            )

            np.testing.assert_allclose(
                profile_actual.hist_y, profile_expected.hist_y
            )
        if DEV_DRAW:
            plt.show()


if __name__ == "__main__":
    unittest.main()
