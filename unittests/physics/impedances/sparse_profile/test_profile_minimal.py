import unittest
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from blond import Beam, backend, uranium_29
from blond.core.beam.beams import ProbeBeam
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.profiles_sparse import EquidistantMultiProfile


class MyTestCase(unittest.TestCase):
    def setUp(self):
        filling_pattern = np.ones(5, bool)
        self.multiprofile_equidistant = EquidistantMultiProfile.headless(
            t_rev=5 * 10.0,
            filling_pattern=filling_pattern,
            bins_per_profile=4,
            offset=0,
        )
        start = self.multiprofile_equidistant.profiles[0].hist_x[0]
        stop = self.multiprofile_equidistant.profiles[-1].hist_x[-1]

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

        independent_profiles = self.expected_exec(DEV_DRAW)
        equidistant = self.actual_exec(DEV_DRAW)
        if DEV_DRAW:
            plt.show()
        for i, profile_expected in enumerate(independent_profiles):
            profile_actual = equidistant.profiles[i]
            np.testing.assert_allclose(
                copy_to_cpu(profile_actual.hist_x),
                copy_to_cpu(profile_expected.hist_x),
            )

            np.testing.assert_allclose(
                copy_to_cpu(profile_actual.hist_y),
                copy_to_cpu(profile_expected.hist_y),
            )

    def actual_exec(self, DEV_DRAW):
        equidistant = deepcopy(self.multiprofile_equidistant)
        equidistant._bind_profiles()
        equidistant.track(self.beam)
        if DEV_DRAW:
            equidistant.plot(linestyle="--")

        return equidistant

    def expected_exec(self, DEV_DRAW):
        independent_profiles = deepcopy(self.multiprofile_equidistant.profiles)

        for profile_expected in independent_profiles:
            profile_expected.track(self.beam)
            if DEV_DRAW:
                profile_expected.plot()
        return independent_profiles


if __name__ == "__main__":
    unittest.main()
