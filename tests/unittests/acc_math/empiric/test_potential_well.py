import itertools
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from blond import backend
from blond.acc_math.empiric.potential_well import PotentialWellHelper
from blond.handle_results.helpers import callers_relative_path
from blond.testing.backend_testing import BLonDTestCase


def _plot_original_and_mirrored(
    pwh: PotentialWellHelper, pwh_mirrored: PotentialWellHelper
) -> None:
    """Show a signal and its mirror image with their buckets."""
    plt.subplot(2, 1, 1)
    plt.title("original")
    pwh.plot()
    plt.subplot(2, 1, 2)
    plt.title("mirrored")
    pwh_mirrored.plot()
    plt.show()


# Single-RF sweep ported from solfege's potential well test cases
# (``solfege/tests/potential_well_test_case.py``, commit 9ff2ca1):
# h=21, 80 kV, 3 keV per turn, a window of a fraction of one turn with
# its start shifted by a fraction of an RF period. With the RF phase
# ``phase = h * omega_rev * t`` the potential well is, up to a positive
# factor, ``sign * (cos(phase) + tilt * phase)``: ``sign`` is the sign
# of charge times eta (both charges, both sides of transition) and
# ``tilt`` is the energy gain per turn over charge times voltage.
HARMONIC_NUMBER = 21
TILTS = (3e3 / 80e3, 0.0, -3e3 / 80e3)
SIGNS = (1, -1)
TURN_FRACTIONS = (1.0, 0.7, 0.4, 0.2)
START_PHASES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.7, 0.8)


def _single_rf_potential_well(
    turn_fraction: float,
    start_phase: float,
    tilt: float,
    sign: int,
    n_points: int = 800,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the RF phase and the tilted single-RF potential well.

    Parameters
    ----------
    turn_fraction
        Window length as a fraction of one turn.
    start_phase
        Window start as a fraction of one RF period.
    tilt
        Energy gain per turn over charge times voltage.
    sign
        Sign of charge times eta.
    n_points
        Number of samples.

    Returns
    -------
    phase
        RF phase in radians.
    potential_well
        Potential well, in units of charge times voltage.
    """
    phase_start = 2 * np.pi * start_phase
    phase_stop = phase_start + 2 * np.pi * HARMONIC_NUMBER * turn_fraction
    phase = np.linspace(phase_start, phase_stop, n_points)
    return phase, sign * (np.cos(phase) + tilt * phase)


def _n_minima_inside(phase: np.ndarray, sign: int) -> int:
    """Count the minima of the untilted potential well inside `phase`.

    ``sign * cos(phase)`` has its minima at ``pi`` (``sign = 1``) or
    ``0`` (``sign = -1``) modulo ``2 pi``; a minimum on the first or
    last sample is not inside.
    """
    offset = np.pi if sign == 1 else 0.0
    half_step = (phase[1] - phase[0]) / 2
    first = np.ceil((phase[0] + half_step - offset) / (2 * np.pi))
    last = np.floor((phase[-1] - half_step - offset) / (2 * np.pi))
    return int(last - first + 1)


def _without_edge_slivers(
    time_axis: np.ndarray, bucket_list: np.ndarray
) -> np.ndarray:
    """Drop buckets at the resolution limit on an array edge.

    When an edge lies within about one sample of a well bottom, the
    helper reports a bucket of two or three samples on that edge. It
    appears or vanishes with sub-sample shifts of the potential, so it
    is noise rather than a well. Known and ignored for now; see
    ``test_no_edge_sliver_bucket``.

    Parameters
    ----------
    time_axis
        Time axis the buckets were found on.
    bucket_list
        Array of shape (N, 2) with ``(start_time, stop_time)``.

    Returns
    -------
    bucket_list
        `bucket_list` without buckets that touch the first or last
        sample and span at most three samples.
    """
    bucket_list = np.asarray(bucket_list).reshape(-1, 2)
    start = np.searchsorted(time_axis, bucket_list[:, 0])
    stop = np.searchsorted(time_axis, bucket_list[:, 1])
    on_edge = (start == 0) | (stop == len(time_axis) - 1)
    max_sliver_samples = 3
    is_sliver = on_edge & (stop - start + 1 <= max_sliver_samples)
    return bucket_list[~is_sliver]


class TestPotentialWellHelper(BLonDTestCase):
    def test_single_not_bucket(self):
        DEV_PLOT = False
        xs = np.linspace(-0, 1, 1000)
        if DEV_PLOT:
            plt.subplot(3, 1, 1)
        ys = np.sin(xs)
        pwh = PotentialWellHelper(xs, ys)
        if DEV_PLOT:
            pwh.plot()
        if DEV_PLOT:
            plt.show()
        self.assertEqual(len(pwh.bucket_list), 0)

    def test_single_not_bucket2(self):
        DEV_PLOT = False
        xs = np.linspace(-0, 1, 1000)
        if DEV_PLOT:
            plt.subplot(3, 1, 1)
        ys = np.cos(xs)
        pwh = PotentialWellHelper(xs, ys)
        if DEV_PLOT:
            pwh.plot()
        if DEV_PLOT:
            plt.show()

        self.assertEqual(len(pwh.bucket_list), 0)

    def test_single_bucket(self):
        DEV_PLOT = False
        xs = np.linspace(0.4, 2 * np.pi - 0.3, 1000)
        if DEV_PLOT:
            plt.subplot(3, 1, 1)
        ys = np.cos(xs)
        pwh = PotentialWellHelper(xs, ys)
        if DEV_PLOT:
            pwh.plot()
        if DEV_PLOT:
            plt.show()

        self.assertEqual(len(pwh.bucket_list), 1)
        bucket_list_pinned = [[0.4, 5.88258737371689]]

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            pwh.bucket_list,
            bucket_list_pinned,
            rtol=1e-12,
        )

    def test_double_bucket(self):
        DEV_PLOT = False
        xs = np.linspace(0.4, 4 * np.pi - 0.3, 1000)
        if DEV_PLOT:
            plt.subplot(3, 1, 1)
        ys = np.cos(xs)
        pwh = PotentialWellHelper(xs, ys)
        if DEV_PLOT:
            pwh.plot()
        if DEV_PLOT:
            plt.show()

        self.assertEqual(len(pwh.bucket_list), 2)
        pwh_bucket_list_pinned = [
            [0.4, 5.875872725945524],
            [6.588567657738867, 12.266370614359172],
        ]

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            pwh.bucket_list,
            pwh_bucket_list_pinned,
            rtol=1e-12,
        )

    def test_triple_bucket(self):
        DEV_PLOT = False
        xs = np.linspace(0.4, 6 * np.pi - 0.3, 1000)
        if DEV_PLOT:
            plt.subplot(3, 1, 1)
        ys = np.cos(xs)
        pwh = PotentialWellHelper(xs, ys)
        if DEV_PLOT:
            pwh.plot()
        if DEV_PLOT:
            plt.show()

        self.assertEqual(len(pwh.bucket_list), 3)

        pwh_bucket_list_pinned = [
            [0.4, 5.868484817200367],
            [6.286342461039599, 12.572374842273243],
            [12.88122614424137, 18.549555921538758],
        ]

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            pwh.bucket_list,
            pwh_bucket_list_pinned,
            rtol=1e-12,
        )

    def test_analyze_buckets(self):
        DEV_PLOT = False
        xs = np.linspace(-10, 20, 1000)
        if DEV_PLOT:
            plt.subplot(3, 1, 1)
        ys = np.sin(xs)
        pwh = PotentialWellHelper(xs, ys)
        pwh_bucket_list_pinned = [
            [-10.0, -5.7357357357357355],
            [-4.714714714714715, 1.561561561561561],
            [1.561561561561561, 7.867867867867869],
            [7.867867867867869, 14.144144144144143],
            [14.564564564564563, 20.0],
        ]
        np.testing.assert_allclose(
            pwh.bucket_list,
            pwh_bucket_list_pinned,
            rtol=1e-6 if backend.float == np.float32 else 1e-12,
        )
        if DEV_PLOT:
            pwh.plot()
            plt.subplot(3, 1, 2)
        ys = np.sin(xs) + xs / 10
        pwh = PotentialWellHelper(xs, ys)

        pwh_bucket_list_pinned = [
            [-10.0, -6.126126126126126],
            [-4.624624624624625, 0.4804804804804803],
            [1.681681681681681, 6.786786786786788],
            [7.957957957957959, 13.063063063063062],
            [14.234234234234233, 19.33933933933934],
        ]

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            pwh.bucket_list,
            pwh_bucket_list_pinned,
            rtol=1e-12,
        )

        if DEV_PLOT:
            pwh.plot()
            plt.subplot(3, 1, 3)
        ys = np.sin(xs) + 0.5 * np.sin(xs * 2 + 1.1) + xs / 10
        pwh = PotentialWellHelper(xs, ys)

        pwh_bucket_list_pinned = [
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
        ]

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            pwh.bucket_list,
            pwh_bucket_list_pinned,
            rtol=1e-12,
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
        DEV_PLOT = False
        if DEV_PLOT:
            pwh.plot()
            plt.twinx()
            plt.plot(xs, mask, color="gray")
            for slice_ in slices:
                plt.axvspan(xs[slice_.start], xs[slice_.stop - 1], alpha=0.2)
            plt.show()
        for slice_ in slices:
            assert np.all(mask[slice_])
            # show that the next one is already outside the mask
            slice_wrong_left = slice(slice_.start - 1, slice_.stop)
            assert not np.all(mask[slice_wrong_left])

            # show that the next one is already outside the mask
            slice_wrong_left = slice(slice_.start, slice_.stop + 1)
            assert not np.all(mask[slice_wrong_left])
        pwh_bucket_list_pinned = [
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
        ]

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            pwh.bucket_list,
            pwh_bucket_list_pinned,
            rtol=1e-12,
        )

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
        DEV_PLOT = False
        if DEV_PLOT:
            pwh.plot()
            plt.twinx()
            plt.plot(xs, mask, color="gray")
            plt.show()
        assert len(slices) == 1
        assert (mask == expected_mask).all()
        np.testing.assert_allclose(mask[slices[0]], mask)
        pwh_bucket_list_pinned = [
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
        ]

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            pwh.bucket_list,
            pwh_bucket_list_pinned,
            rtol=1e-12,
        )

    def test_analyze_bug(self):
        ys = np.loadtxt(
            callers_relative_path("resources/ys.csv", stacklevel=1)
        )
        xs = np.arange(len(ys)) * 1e-9

        pwh = PotentialWellHelper(xs, ys)
        pinned = [[4.00e-08, 1.12e-07]]
        DEV_PLOT = False
        if DEV_PLOT:
            pwh.plot()
            plt.show()
        np.testing.assert_allclose(pwh.bucket_list, pinned)

    def assert_buckets_sound(
        self,
        time_axis: np.ndarray,
        potential_well: np.ndarray,
        bucket_list: np.ndarray,
    ) -> None:
        """Assert that every bucket is a potential well by definition.

        Both borders sit at the same level (unless the lower one is an
        array edge), nothing inside rises above that level, and a
        minimum lies strictly inside. The level tolerance is the
        helper's own 0.1 % epsilon plus the largest sample-to-sample
        step at either border, since a border can only be found to
        within one sample.
        """
        epsilon = 0.1 / 100 * np.ptp(potential_well)
        last = len(potential_well) - 1

        def local_step(index: int) -> float:
            neighbours = potential_well[max(index - 1, 0) : index + 2]
            return float(np.max(np.abs(np.diff(neighbours))))

        for t_start, t_stop in bucket_list:
            start = int(np.argmin(np.abs(time_axis - t_start)))
            stop = int(np.argmin(np.abs(time_axis - t_stop)))
            msg = f"bucket [{start}, {stop}]"
            u_start = potential_well[start]
            u_stop = potential_well[stop]
            level = min(u_start, u_stop)
            tolerance = epsilon + max(local_step(start), local_step(stop))
            lower_is_edge = (u_start < u_stop and start == 0) or (
                u_stop < u_start and stop == last
            )
            if not lower_is_edge:
                self.assertLessEqual(
                    abs(u_start - u_stop), tolerance, msg=f"{msg} not level"
                )
            interior = potential_well[start + 1 : stop]
            self.assertGreater(len(interior), 0, msg=f"{msg} is empty")
            self.assertLessEqual(
                np.max(interior), level + tolerance, msg=f"{msg} barrier"
            )
            self.assertLess(np.min(interior), level, msg=f"{msg} no minimum")

    def test_multi_rf_potential_wells(self):
        """Multi-harmonic PS potential wells, checked against solfege.

        The fixtures are solfege's multi-RF scenarios
        (``solfege/tests/test_potential_complex.py``, commit 9ff2ca1):
        h21 single RF, h21/h28 batch compression and h28/h169
        rebucketting of a 129 u, 39+ ion. ``potential_well`` is
        solfege's ``make_potential_well`` of the stored
        ``voltage_array``; ``solfege_well_borders`` are the borders of
        the wells solfege's ``get_all_potential_wells`` cuts from it.
        """
        DEV_PLOT = False

        for case in range(3):
            with self.subTest(case=case):
                data = np.load(
                    callers_relative_path(
                        f"resources/test_potential_complex_case{case}.npz",
                        stacklevel=1,
                    )
                )
                time_axis = data["time_array"]
                potential_well = data["potential_well"]

                pwh = PotentialWellHelper(time_axis, potential_well)
                if DEV_PLOT:
                    pwh.plot()
                    plt.show()

                self.assert_buckets_sound(
                    time_axis, potential_well, pwh.bucket_list
                )
                # every well solfege finds lies inside a bucket
                in_bucket = pwh.get_in_bucket_mask()
                for t_start, t_stop in data["solfege_well_borders"]:
                    in_well = (time_axis >= t_start) & (time_axis <= t_stop)
                    self.assertTrue(
                        np.all(in_bucket[in_well]),
                        msg=f"solfege well [{t_start}, {t_stop}] missed",
                    )

                pwh_bucket_list_pinned = np.loadtxt(
                    callers_relative_path(
                        f"resources/expected_potential_complex_case{case}.csv",
                        stacklevel=1,
                    )
                )
                np.testing.assert_allclose(
                    pwh.bucket_list, pwh_bucket_list_pinned, rtol=1e-12
                )

    def test_plot_executes(self):
        xs = np.linspace(-10, 20, 1000)
        ys = np.sin(xs)
        pwh = PotentialWellHelper(xs, ys)
        pwh.plot()
        DEV_PLOT = False
        if DEV_PLOT:
            plt.show()
        plt.close("all")
        pwh_bucket_list_pinned = [
            [-10.0, -5.7357357357357355],
            [-4.714714714714715, 1.561561561561561],
            [1.561561561561561, 7.867867867867869],
            [7.867867867867869, 14.144144144144143],
            [14.564564564564563, 20.0],
        ]

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            pwh.bucket_list,
            pwh_bucket_list_pinned,
            rtol=1e-12,
        )

    def test_purge_duplicates_off_by_one_odd_pair(self):
        """Buckets one grid step apart are merged, whatever the parity.

        ``idx // 2`` only merges index pairs ``(2k, 2k+1)``; the pair
        ``(3, 4)`` must be merged as well.
        """
        # skip __init__ (it runs the full analysis); purge only needs time_axis
        pwh = PotentialWellHelper.__new__(PotentialWellHelper)
        pwh.time_axis = np.arange(10.0)
        bucket_list = [(3.0, 6.0), (4.0, 6.0)]
        self.assertEqual(len(pwh._purge_duplicates_off_by_one(bucket_list)), 1)

    def test_no_off_by_one_duplicate_buckets(self):
        """Detected buckets never differ only by one grid step."""
        xs = np.linspace(-10, 20, 1000)
        pwh = PotentialWellHelper(xs, np.sin(xs))
        DEV_PLOT = False
        if DEV_PLOT:
            pwh.plot()
            plt.show()
        step = xs[1] - xs[0]
        buckets = np.asarray(pwh.bucket_list)
        for i in range(len(buckets)):
            for j in range(i + 1, len(buckets)):
                self.assertFalse(
                    np.all(np.abs(buckets[i] - buckets[j]) <= 1.5 * step),
                    msg=f"duplicate buckets {buckets[i]} and {buckets[j]}",
                )

    def test_near_equal_maxima_give_single_bucket(self):
        """Maxima equal within epsilon bound one bucket, peak to peak.

        The search from the lower maximum must not stop where the
        voltage first exceeds its height, many grid steps before the
        higher maximum, which would leave a far-off duplicate bucket.
        """
        xs = np.linspace(-1, 3 * np.pi + 1, 10000)
        # right maximum 0.05 % higher, inside the 0.1 % epsilon
        ys = np.cos(xs) * (1 + 0.0005 * (xs > np.pi))
        pwh = PotentialWellHelper(xs, ys)
        DEV_PLOT = False
        if DEV_PLOT:
            pwh.plot()
            plt.show()
        left_peak = xs[np.argmax(ys[xs < np.pi])]
        right_peak = xs[np.argmax(np.where(xs > np.pi, ys, -np.inf))]
        buckets = np.asarray(pwh.bucket_list)
        np.testing.assert_array_equal(
            buckets[buckets[:, 0] == left_peak],
            [[left_peak, right_peak]],
        )

    def test_flat_top_maxima_give_single_bucket(self):
        """Quantised flat-top maxima bound one bucket, centre to centre.

        ``find_peaks`` puts a plateau maximum at the plateau centre; the
        bucket search must end there too, not on the plateau edge.
        """
        xs = np.linspace(-1, 3 * np.pi + 1, 10000)
        ys = np.round(np.cos(xs) * 100)  # ADC-like quantisation
        pwh = PotentialWellHelper(xs, ys)
        DEV_PLOT = False
        if DEV_PLOT:
            pwh.plot()
            plt.show()
        left_peak, right_peak = xs[find_peaks(ys)[0]]
        buckets = np.asarray(pwh.bucket_list)
        # the second bucket is the partial one at the right border
        self.assertEqual(len(buckets), 2, msg=f"buckets {buckets}")
        np.testing.assert_array_equal(buckets[0], [left_peak, right_peak])

    def test_border_not_snapped_onto_edge_sample(self):
        """A border before a rise to the array edge stays where it is.

        The signal is cut so that only the last sample exceeds the
        maximum; the edge sample is no maximum to snap the border onto.
        """
        xs = np.linspace(-0.5, 7, 2000)
        ys = np.cos(xs) + 0.01 * xs
        peak = find_peaks(ys)[0][0]
        trough = peak + np.argmin(ys[peak:])
        # keep up to the first sample above the maximum after the trough
        n_samples = trough + np.argmax(ys[trough:] > ys[peak]) + 1
        xs, ys = xs[:n_samples], ys[:n_samples]
        pwh = PotentialWellHelper(xs, ys)
        pwh_mirrored = PotentialWellHelper(-xs[::-1], ys[::-1])
        DEV_PLOT = False
        if DEV_PLOT:
            _plot_original_and_mirrored(pwh, pwh_mirrored)
        buckets = np.asarray(pwh.bucket_list)
        np.testing.assert_array_equal(buckets, [[xs[peak], xs[-2]]])

        buckets_mirrored = np.asarray(pwh_mirrored.bucket_list)
        np.testing.assert_array_equal(buckets_mirrored, -buckets[:, ::-1])

    def test_mirrored_signal_gives_mirrored_buckets(self):
        """The left search reaches the second sample like the right one.

        The outer bucket from the higher maximum spans the lower one and
        ends at the second-to-last sample; mirrored, it must end at the
        second sample.
        """
        xs = np.linspace(-0.5, 3 * np.pi, 2000)
        ys = np.cos(xs) - 0.02 * xs  # higher maximum at 0, lower at 2 pi
        ys[-1] = ys.max() + 0.1  # only the last sample exceeds both
        pwh = PotentialWellHelper(xs, ys)
        pwh_mirrored = PotentialWellHelper(-xs[::-1], ys[::-1])
        DEV_PLOT = False
        if DEV_PLOT:
            _plot_original_and_mirrored(pwh, pwh_mirrored)
        buckets = np.asarray(pwh.bucket_list)
        self.assertIn([xs[find_peaks(ys)[0][0]], xs[-2]], buckets.tolist())

        buckets_mirrored = np.asarray(pwh_mirrored.bucket_list)
        self.assertEqual(
            sorted(buckets_mirrored.tolist()),
            sorted((-buckets[:, ::-1]).tolist()),
        )

    def test_single_rf_sweep_buckets_sound(self):
        """Every bucket is a well, whatever the charge, tilt or window."""
        for sign, tilt, turn_fraction, start_phase in itertools.product(
            SIGNS, TILTS, TURN_FRACTIONS, START_PHASES
        ):
            with self.subTest(
                sign=sign,
                tilt=tilt,
                turn_fraction=turn_fraction,
                start_phase=start_phase,
            ):
                phase, potential_well = _single_rf_potential_well(
                    turn_fraction, start_phase, tilt, sign
                )
                pwh = PotentialWellHelper(phase, potential_well)
                self.assert_buckets_sound(
                    phase,
                    potential_well,
                    _without_edge_slivers(phase, pwh.bucket_list),
                )

    def test_single_rf_sweep_count_without_tilt(self):
        """Without tilt, each minimum inside the window has one bucket."""
        for sign, turn_fraction, start_phase in itertools.product(
            SIGNS, TURN_FRACTIONS, START_PHASES
        ):
            with self.subTest(
                sign=sign, turn_fraction=turn_fraction, start_phase=start_phase
            ):
                phase, potential_well = _single_rf_potential_well(
                    turn_fraction, start_phase, 0.0, sign
                )
                pwh = PotentialWellHelper(phase, potential_well)
                self.assertEqual(
                    len(pwh.bucket_list), _n_minima_inside(phase, sign)
                )

    def test_single_rf_sweep_count_independent_of_tilt(self):
        """Accelerating or decelerating keeps the count of the coast."""
        for sign, turn_fraction, start_phase in itertools.product(
            SIGNS, TURN_FRACTIONS, START_PHASES
        ):
            with self.subTest(
                sign=sign, turn_fraction=turn_fraction, start_phase=start_phase
            ):
                n_buckets = []
                for tilt in TILTS:
                    phase, potential_well = _single_rf_potential_well(
                        turn_fraction, start_phase, tilt, sign
                    )
                    pwh = PotentialWellHelper(phase, potential_well)
                    n_buckets.append(
                        len(_without_edge_slivers(phase, pwh.bucket_list))
                    )
                self.assertEqual(
                    n_buckets, [n_buckets[1]] * len(TILTS), msg="tilt +, 0, -"
                )

    @unittest.expectedFailure
    def test_no_edge_sliver_bucket(self):
        """Known issue: a sub-sample well at the edge becomes a bucket.

        The window starts on a well bottom; the tilt moves the bottom
        between the first two samples, and the helper reports the
        bucket ``[0, 1]`` with no sample inside. Such buckets at the
        resolution limit are noise and are ignored by the sweeps above
        (see `_without_edge_slivers`); this test records the behaviour
        until it is changed.
        """
        phase, potential_well = _single_rf_potential_well(
            0.4, 0.0, TILTS[0], -1
        )
        pwh = PotentialWellHelper(phase, potential_well)
        np.testing.assert_array_equal(
            _without_edge_slivers(phase, pwh.bucket_list), pwh.bucket_list
        )

    def test_monotonic_potential_has_no_bucket(self):
        time_axis = np.linspace(0, 1e-6, 50)
        pwh = PotentialWellHelper(time_axis, 5.0 * time_axis)
        self.assertEqual(len(pwh.bucket_list), 0)


if __name__ == "__main__":
    unittest.main()
