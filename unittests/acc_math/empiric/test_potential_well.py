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
                [-4.714714714714715, 1.561561561561561],
                [1.561561561561561, 7.867867867867869],
                [-4.684684684684685, 1.561561561561561],
                [7.867867867867869, 14.114114114114113],
                [1.591591591591591, 7.867867867867869],
                [7.867867867867869, 14.144144144144143],
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
                [-4.624624624624625, 0.4804804804804803],
                [1.681681681681681, 6.786786786786788],
                [7.957957957957959, 13.063063063063062],
                [14.234234234234233, 19.33933933933934],
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
                [-9.6996996996997, -6.456456456456456],
                [-5.555555555555555, -0.06006006006006004],
                [-3.423423423423423, -0.18018018018018012],
                [-4.954954954954955, -3.423423423423423],
                [0.7207207207207205, 6.216216216216218],
                [2.8528528528528536, 6.1261261261261275],
                [1.3213213213213209, 2.8528528528528536],
                [7.027027027027028, 12.522522522522522],
                [9.12912912912913, 12.402402402402402],
                [7.6276276276276285, 9.12912912912913],
                [13.303303303303302, 18.7987987987988],
                [15.435435435435434, 18.67867867867868],
                [13.903903903903903, 15.435435435435434],
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
        DEV_DEBUG = False
        if DEV_DEBUG:
            pwh.plot()
            plt.show()
        np.testing.assert_allclose(pwh.bucket_list, pinned)

    def test_analyze_bug(self):
        DEV_DEBUG = True
        pinned_ = [[4.00e-08, 1.12e-07]] # TODO pinn values for testing.

        for i in range(3):
            data = np.load(
                callers_relative_path(
                    f"resources/test_potential_complex_case{i}.npz",
                    stacklevel=1)
            )
            xs = data["time_array"]
            ys = data["voltage_array"]

            pwh = PotentialWellHelper(xs, ys)
            if DEV_DEBUG:
                pwh.plot()
                plt.show()
            # np.testing.assert_allclose(pwh.bucket_list, pinned)

    def test_plot_executes(self):
        xs = np.linspace(-10, 20, 1000)
        ys = np.sin(xs)
        pwh = PotentialWellHelper(xs, ys)
        pwh.plot()
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
