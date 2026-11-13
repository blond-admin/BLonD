from unittest import mock

import numpy as np

from blond.experimental.beam_preparation.bucket_windows import (
    BucketWindows,
    _stitch_potential_wells,
    check_bucket_continuity,
    cut_at_separatrix,
    select_bucket,
    split_macroparticles,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.generals.exceptions_ import SemiEmpiricMatcherError
from blond.testing.backend_testing import BLonDTestCase


def _synthetic_well(x, shift=0.0, slope=0.0, ratio=0.0, n=2, psi=0.0):
    """Potential well of single/double harmonic RF, ``x`` in RF periods.

    The bucket centre is at ``x = 0.5`` without ``shift``; ``slope`` tilts
    the well like acceleration (:math:`\\sin\\phi_s`), ``ratio`` and ``n``
    add a higher harmonic: ``psi = 0`` lengthens (flat bottom),
    ``psi = pi`` shortens the bunch.
    """
    phase = 2 * np.pi * (x - shift)
    well = np.cos(phase) + ratio / n * np.cos(n * phase + psi) + slope * phase
    return well - well.min()


class TestSelectBucket(BLonDTestCase):
    """`select_bucket` in a window of +- one RF period around the centre."""

    x = np.linspace(-0.5, 1.5, 1023)  # in RF periods, nominal centre 0.5

    def _select(self, **well_kwargs):
        selected = select_bucket(
            ts=self.x,
            potential_well=_synthetic_well(self.x, **well_kwargs),
            window_centre=0.5,
            bucket_length=1.0,
        )
        return self.x[selected][0], self.x[selected][-1]

    def assertBucket(self, bucket, start, stop):  # noqa: N802
        np.testing.assert_allclose(bucket, (start, stop), atol=0.01)

    def test_single_rf(self):
        self.assertBucket(self._select(), 0.0, 1.0)

    def test_follows_phase_shift(self):
        for shift in (-0.3, 0.17, 0.45):
            with self.subTest(shift=shift):
                self.assertBucket(
                    self._select(shift=shift), shift, 1.0 + shift
                )

    def test_accelerating_bucket_is_bounded_by_lower_barrier(self):
        # separatrix through the unstable fixed point at x = 0.083, closing
        # at the same height at x = 0.613
        self.assertBucket(self._select(slope=0.5), 0.083, 0.613)

    def test_two_complete_buckets_takes_nearest_to_centre(self):
        # small accelerating buckets: the window holds two complete ones,
        # and neither contains the nominal centre
        self.assertBucket(self._select(slope=0.9), 0.18, 0.39)

    def test_double_well_is_filled_as_one_bucket(self):
        # bunch lengthening mode, ratio 1: two minima inside the bucket
        self.assertBucket(self._select(ratio=1.0, psi=0.0), 0.0, 1.0)

    def test_split_barrier_keeps_central_bucket(self):
        # bunch shortening mode, ratio 1: each barrier splits in two, with
        # a small bucket in between that belongs to neither neighbour
        self.assertBucket(self._select(ratio=1.0, psi=np.pi), 0.17, 0.83)

    def test_no_bucket_raises(self):
        with self.assertRaisesRegex(
            SemiEmpiricMatcherError, "No complete bucket"
        ):
            self._select(slope=1.2)

    def test_bucket_further_than_half_a_bucket_raises(self):
        # the only complete bucket, [0.18, 0.39], is 0.6 away from the
        # nominal centre
        x = np.linspace(0.0, 1.0, 511)
        well = _synthetic_well(x, slope=0.9)
        with self.assertRaisesRegex(SemiEmpiricMatcherError, "half a bucket"):
            select_bucket(
                ts=x, potential_well=well, window_centre=0.9, bucket_length=1.0
            )


class TestStitchPotentialWells(BLonDTestCase):
    """Undo the arbitrary offset of each window, for plotting."""

    @staticmethod
    def _windowed(windows, n_points=201):
        windows_ts = [
            np.linspace(start, stop, n_points) for start, stop in windows
        ]
        wells = [_synthetic_well(ts, slope=0.3) for ts in windows_ts]
        return windows_ts, wells

    def test_overlapping_windows_give_one_continuous_well(self):
        # unordered on purpose
        windows_ts, wells = self._windowed(
            [(1.5, 3.5), (-0.5, 1.5), (0.5, 2.5)]
        )

        stitched = _stitch_potential_wells(windows_ts, wells)

        all_ts = np.concatenate(windows_ts)
        reference = _synthetic_well(all_ts, slope=0.3)  # one integral
        reference -= reference.min()
        np.testing.assert_allclose(
            np.concatenate(stitched), reference, atol=1e-12
        )

    def test_separate_windows_keep_their_own_minimum(self):
        windows_ts, wells = self._windowed([(-0.5, 1.5), (4.5, 6.5)])

        stitched = _stitch_potential_wells(windows_ts, wells)

        for well, stitched_well in zip(wells, stitched, strict=True):
            np.testing.assert_allclose(stitched_well, well, atol=1e-12)


class TestCheckBucketContinuity(BLonDTestCase):
    def test_drift_is_accepted(self):
        check_bucket_continuity(
            previous_centres=[0.5, 1.5],
            centres=[0.3, 1.8],
            bucket_lengths=[1.0, 1.0],
        )

    def test_jump_raises(self):
        # +0.55 is indistinguishable from -0.45 in a single snapshot, only
        # the previous iteration tells that the bucket left its window
        with self.assertRaisesRegex(
            SemiEmpiricMatcherError, "bucket 1 jumped"
        ):
            check_bucket_continuity(
                previous_centres=[0.5, 1.5],
                centres=[0.5, 0.95],
                bucket_lengths=[1.0, 1.0],
            )


class TestSplitMacroparticles(BLonDTestCase):
    def test_split_by_weights(self):
        np.testing.assert_array_equal(
            split_macroparticles([1.0, 3.0], 20_000), [5_000, 15_000]
        )

    def test_remainder_goes_to_heaviest_bucket(self):
        n_per_bucket = split_macroparticles([1.0, 2.0, 1.0], 10)
        np.testing.assert_array_equal(n_per_bucket, [2, 6, 2])

    def test_zero_weight_gives_empty_bucket(self):
        np.testing.assert_array_equal(
            split_macroparticles([0.0, 1.0], 10), [0, 10]
        )


class TestCutAtSeparatrix(BLonDTestCase):
    def test_density_above_lower_barrier_is_removed(self):
        # H(t, dE) = V(t) + dE**2, the bucket edges at V = 1 and V = 2:
        # the separatrix is the lower one
        potential_well = np.array([1.0, 0.5, 0.0, 0.5, 2.0])
        deltaE = np.linspace(-1, 1, 5)
        hamilton_2D = potential_well[:, None] + deltaE[None, :] ** 2
        density = np.ones_like(hamilton_2D)

        cut, fraction_outside = cut_at_separatrix(density, hamilton_2D)

        outside = hamilton_2D > 1.0
        np.testing.assert_array_equal(cut[outside], 0.0)
        np.testing.assert_array_equal(cut[~outside], 1.0)
        self.assertAlmostEqual(fraction_outside, outside.mean())
        # the input is not modified
        np.testing.assert_array_equal(density, 1.0)

    def test_empty_density(self):
        hamilton_2D = np.zeros((3, 3))
        _, fraction_outside = cut_at_separatrix(np.zeros((3, 3)), hamilton_2D)
        self.assertEqual(fraction_outside, 0.0)


class TestBucketWindowsFromTimeLimit(BLonDTestCase):
    def _from(self, time_limit, **kwargs):
        kwargs = {
            "hamilton_to_density_kwargs": {},
            "bucket_weights": None,
            "n_time": 11,
            **kwargs,
        }
        return BucketWindows.from_time_limit(time_limit=time_limit, **kwargs)

    def test_single_time_limit_is_one_window_not_per_bucket(self):
        bucket_windows = self._from((0.0, 1.0))
        self.assertFalse(bucket_windows.per_bucket)
        self.assertEqual(bucket_windows.windows, [(0.0, 1.0)])
        self.assertEqual(bucket_windows.span, (0.0, 1.0))

    def test_windows(self):
        bucket_windows = self._from([(0.0, 2.0), (1.0, 3.0)])
        self.assertEqual(bucket_windows.span, (0.0, 3.0))
        self.assertEqual(len(bucket_windows.windows_ts), 2)
        np.testing.assert_allclose(
            bucket_windows.windows_ts[1], np.linspace(1.0, 3.0, 11)
        )
        np.testing.assert_array_equal(bucket_windows.bucket_weights, [1, 1])

    def test_invalid_arguments_raise(self):
        windows = [(0.0, 2.0), (1.0, 3.0)]
        for kwargs in (
            dict(bucket_weights=[1.0]),
            dict(bucket_weights=[1.0, -1.0]),
            dict(bucket_weights=[0.0, 0.0]),
            dict(hamilton_to_density_kwargs=[{}]),
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self._from(windows, **kwargs)
        with self.assertRaises(ValueError):
            self._from([(1.0, 0.0)])
        with self.assertRaises(ValueError):
            self._from([(0.0, 1.0, 2.0)])
        with self.assertRaises(ValueError):
            # per-window arguments without windows
            self._from((0.0, 1.0), bucket_weights=[1.0])


class TestBucketWindowsPopulateBeamSingleTimeLimit(BLonDTestCase):
    def _populate(self, debug_helper=None):
        bucket_windows = BucketWindows.from_time_limit(
            time_limit=(0.0, 2.0),
            hamilton_to_density_kwargs={},
            bucket_weights=None,
            n_time=201,
        )
        ts = bucket_windows.windows_ts[0]
        beam = mock.Mock()
        beam.reference.total_energy = 1.0
        beam.reference.beta = 1.0
        bucket_windows.populate_beam(
            beam=beam,
            # two buckets, minima at 0.5 and 1.5
            potential_well=np.cos(2 * np.pi * ts) + 5.0,
            eta=1.0,
            n_energy=11,
            hamilton_to_density_function=lambda **grids: np.ones_like(
                grids["hamilton_2D"]
            ),
            n_macroparticles=1000,
            seed=1,
            backend_dependent_random_numbers=False,
            debug_helper=debug_helper,
        )
        return bucket_windows, beam

    def test_fills_whole_window_without_selection_or_cut(self):
        bucket_windows, beam = self._populate()
        dt = copy_to_cpu(beam.setup_beam.call_args.kwargs["dt"])
        self.assertEqual(len(dt), 1000)
        # both buckets, and outside the separatrix, around t = 1
        self.assertLess(dt.min(), 0.1)
        self.assertGreater(dt.max(), 1.9)
        self.assertGreater(np.count_nonzero(np.abs(dt - 1.0) < 0.05), 0)
        self.assertIsNone(bucket_windows.selected_buckets)

    def test_density_function_gets_grid_after_one_turn_map_correction(self):
        # so that e.g. a fitted bunch length is the one of the particles
        received = {}

        def record(**grids):
            received.update(grids)
            return np.ones_like(grids["hamilton_2D"])

        bucket_windows = BucketWindows.from_time_limit(
            time_limit=(0.0, 2.0),
            hamilton_to_density_kwargs={},
            bucket_weights=None,
            n_time=21,
        )
        beam = mock.Mock()
        beam.reference.total_energy = 1.0
        beam.reference.beta = 1.0
        ts = bucket_windows.windows_ts[0]
        bucket_windows.populate_beam(
            beam=beam,
            potential_well=np.cos(2 * np.pi * ts) + 5.0,
            eta=1.0,
            n_energy=11,
            hamilton_to_density_function=record,
            n_macroparticles=100,
            seed=1,
            backend_dependent_random_numbers=False,
            grid_correction=(0.01, 2.0),  # (shear_dt_per_dE, deltaE_scale)
        )
        deltaE_max = np.sqrt(2 / 0.5)  # well depth 2, drift term 1
        np.testing.assert_allclose(
            received["deltaE_grid"].max(), 2.0 * deltaE_max
        )
        np.testing.assert_allclose(
            received["time_grid"][0, -1], 0.0 + 0.01 * 2.0 * deltaE_max
        )

    def test_hamiltonian_keeps_offset_of_potential_well(self):
        debug_helper = mock.Mock()
        self._populate(debug_helper=debug_helper)
        # arrays of the single window, not a list per bucket
        hamilton_2D = debug_helper.last_hamilton_2D
        self.assertEqual(hamilton_2D.shape, (201, 11))
        self.assertAlmostEqual(float(hamilton_2D.min()), 4.0, places=3)
