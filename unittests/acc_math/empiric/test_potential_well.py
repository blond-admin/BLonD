import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from blond.acc_math.empiric.potential_well import PotentialWellHelper
from blond.handle_results.helpers import callers_relative_path


class TestPotentialWellHelper(unittest.TestCase):
    def test_analyze_buckets(self):
        DEV_PLOT = False
        xs = np.linspace(-10, 20, 1000)
        if DEV_PLOT:
            plt.subplot(3, 1, 1)
        ys = np.sin(xs)
        pwh = PotentialWellHelper(xs, ys)
        np.testing.assert_allclose(
            pwh.bucket_list,
            [
                [-4.714714714714715, 1.591591591591591],
                [1.561561561561561, 7.897897897897899],
                [-4.654654654654655, 1.561561561561561],
                [7.867867867867869, 14.144144144144143],
                [1.621621621621621, 7.867867867867869],
                [7.897897897897899, 14.144144144144143],
            ],
        )
        if DEV_PLOT:
            pwh.plot()
            plt.subplot(3, 1, 2)
        ys = np.sin(xs) + xs / 10
        pwh = PotentialWellHelper(xs, ys)
        np.testing.assert_allclose(
            pwh.bucket_list,
            [
                [-4.624624624624625, 0.5105105105105103],
                [1.681681681681681, 6.816816816816818],
                [7.957957957957959, 13.093093093093092],
                [14.234234234234233, 19.36936936936937],
            ],
        )

        if DEV_PLOT:
            pwh.plot()
            plt.subplot(3, 1, 3)
        ys = np.sin(xs) + 0.5 * np.sin(xs * 2 + 1.1) + xs / 10
        pwh = PotentialWellHelper(xs, ys)
        np.testing.assert_allclose(
            pwh.bucket_list,
            [
                [-9.6996996996997, -6.426426426426426],
                [-5.555555555555555, -0.03003003003003002],
                [-3.423423423423423, -0.1501501501501501],
                [-4.924924924924925, -3.423423423423423],
                [0.7207207207207205, 6.246246246246248],
                [2.8528528528528536, 6.1561561561561575],
                [1.3513513513513509, 2.8528528528528536],
                [7.027027027027028, 12.552552552552552],
                [9.12912912912913, 12.432432432432432],
                [7.6576576576576585, 9.12912912912913],
                [13.303303303303302, 18.82882882882883],
                [15.435435435435434, 18.70870870870871],
                [13.933933933933933, 15.435435435435434],
            ],
        )

        if DEV_PLOT:
            pwh.plot()
            plt.twinx()
            plt.plot(xs, pwh.get_in_bucket_mask())
            plt.show()

    def test_get_principal_bucket_slices(self):
        xs = np.linspace(-10, 20, 1000)
        ys = np.sin(xs) + 0.5 * np.sin(xs * 2 + 1.1) + xs / 10
        pwh = PotentialWellHelper(xs, ys)
        mask = pwh.get_in_bucket_mask()
        slices = pwh.get_principal_bucket_slices()
        for slice_ in slices:
            assert np.all(mask[slice_])
            # show that the next one is already outside the mask
            slice_wrong_left = slice(slice_.start - 1, slice_.stop)
            assert not np.all(mask[slice_wrong_left])

            # show that the next one is already outside the mask
            slice_wrong_left = slice(slice_.start, slice_.stop + 1)
            assert not np.all(mask[slice_wrong_left])

    def test_get_principal_bucket_slices_border(self):
        xs = np.linspace(-10, 20, 1000)
        ys = np.sin(xs) + 0.5 * np.sin(xs * 2 + 1.1) + xs / 10
        pwh = PotentialWellHelper(xs, ys)

        expected_mask = np.ones(len(xs), dtype=bool)

        with patch.object(
            PotentialWellHelper,
            "get_in_bucket_mask",
            return_value=expected_mask,
        ):
            mask = pwh.get_in_bucket_mask()
            slices = pwh.get_principal_bucket_slices()
        assert len(slices) == 1
        assert (mask == expected_mask).all()
        np.testing.assert_allclose(mask[slices[0]], mask)

    def test_analyze_bug(self):
        ys = np.loadtxt(
            callers_relative_path("resources/ys.csv", stacklevel=1)
        )
        xs = np.arange(len(ys)) * 1e-9

        pwh = PotentialWellHelper(xs, ys)
        pinned = [[4.00e-08, 1.12e-07]]
        np.testing.assert_allclose(pwh.bucket_list, pinned)
        DEV_DEBUG = False
        if DEV_DEBUG:
            pwh.plot()
            plt.show()

    def test_plot_executes(self):
        xs = np.linspace(-10, 20, 1000)
        ys = np.sin(xs)
        pwh = PotentialWellHelper(xs, ys)
        pwh.plot()
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
