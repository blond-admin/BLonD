import copy
import unittest
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from blond import EmptyBeam, uranium_29
from blond.core.beam.beams import ProbeBeam
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.profiles_sparse import EquidistantMultiProfile
from unittests.physics.impedances.comparisons.mtw import harmonic


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

    def test_init_from_padded_filling_pattern(self):
        sparse_profile = (
            EquidistantMultiProfile.init_from_padded_filling_pattern(
                harmonic=10,
                filling_pattern=np.array([0, 1]),
                bins_per_profile=16,
                offset=-1,
                section_index=0,
                name="barney",
            )
        )
        self.assertEqual(len(sparse_profile._filling_pattern), 10)

    def test_track_empty(self):
        beam = EmptyBeam(
            particle_type=uranium_29,
            reference_time=0,
            reference_total_energy=1e3,
        )
        self.multiprofile_equidistant.track(beam)

    def test_track(self):
        DEV_DRAW = False

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
                copy_to_cpu(profile_actual.hist_x),
                copy_to_cpu(profile_expected.hist_x),
            )

            np.testing.assert_allclose(
                copy_to_cpu(profile_actual.hist_y),
                copy_to_cpu(profile_expected.hist_y),
            )
        if DEV_DRAW:
            plt.show()

    def test_track_after_deepcopy(self):
        DEV_DRAW = False
        for fun in (copy.copy, copy.deepcopy):
            _multiprofile_equidistant = EquidistantMultiProfile.headless(
                t_rev=5 * 10.0,
                filling_pattern=np.ones(5, bool),
                bins_per_profile=4,
                offset=0,
            )

            equidistant_profile = fun(_multiprofile_equidistant)
            independent_profiles = deepcopy(equidistant_profile.profiles)

            equidistant_profile.track(self.beam)

            for profile_expected in independent_profiles:
                profile_expected.track(self.beam)
                if DEV_DRAW:
                    profile_expected.plot()
            if DEV_DRAW:
                equidistant_profile.plot(linestyle="--")
            for i, profile_expected in enumerate(independent_profiles):
                profile_actual = equidistant_profile.profiles[i]
                np.testing.assert_allclose(
                    copy_to_cpu(profile_actual.hist_x),
                    copy_to_cpu(profile_expected.hist_x),
                )

                np.testing.assert_allclose(
                    copy_to_cpu(profile_actual.hist_y),
                    copy_to_cpu(profile_expected.hist_y),
                )


if __name__ == "__main__":
    unittest.main()
