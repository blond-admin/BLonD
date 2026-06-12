import pickle
import warnings

import numpy as np
import pytest

from blond.cycles.filling_patterns import (
    Batch,
    BunchTable,
    FillingPattern,
    Gap,
    PatternSegment,
    Train,
    n_buckets_from_time,
)


class TestNameCollisions:
    def test_constructor_rejects_tier_property_collision(self):
        with pytest.raises(ValueError, match="foo"):
            BunchTable(
                bucket_indices=np.array([0]),
                n_buckets=1,
                tiers={"foo": np.array([0])},
                properties={"foo": np.array([1.0])},
            )

    def test_constructor_rejects_structural_property_name(self):
        with pytest.raises(ValueError, match="bucket_indices"):
            BunchTable(
                bucket_indices=np.array([0]),
                n_buckets=1,
                properties={"bucket_indices": np.array([1.0])},
            )

    def test_add_rejects_tier_property_collision(self):
        left = Batch(n_bunches=2, bunch_gap=1)
        left.intensity = np.array([1.0, 2.0])
        right = Batch(n_bunches=2, bunch_gap=1).label("intensity")
        with pytest.raises(ValueError, match="intensity"):
            left + right


class TestFromSpacingErrors:
    def test_train_from_spacing_too_short_distance(self):
        unit = Batch(n_bunches=72, bunch_gap=9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="start_to_start_distance"):
                Train.from_spacing(
                    unit,
                    n_copies=2,
                    start_to_start_distance=1e-9,
                    f_rf=400.789e6,
                )

    def test_batch_from_spacing_too_short_distance(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="start_to_start_distance"):
                Batch.from_spacing(
                    n_bunches=4, start_to_start_distance=0.0, f_rf=1.0
                )


class TestStructuralImmutability:
    def test_bucket_indices_read_only(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        with pytest.raises(ValueError):
            pattern.bucket_indices[0] = 99

    def test_tier_columns_read_only(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        with pytest.raises(ValueError):
            pattern.tier("batch")[0] = 7

    def test_constructor_does_not_freeze_caller_array(self):
        caller_bucket_indices = np.array([0, 2], dtype=np.int64)
        table = BunchTable(bucket_indices=caller_bucket_indices, n_buckets=5)
        caller_bucket_indices[0] = 1
        assert table.bucket_indices[0] == 0

    def test_property_assignment_copies_source(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        source = np.ones(4)
        pattern.intensity = source
        source[0] = 999.0
        assert pattern.intensity[0] == 1.0

    def test_property_dict_is_snapshot(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        pattern.properties["bogus"] = np.zeros(2)
        assert "bogus" not in pattern.properties

    def test_tiers_dict_is_snapshot(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        pattern.tiers["bogus"] = np.zeros(4, dtype=np.int32)
        assert "bogus" not in pattern.tiers

    def test_property_masked_assignment_idiom_still_works(self):
        pattern = FillingPattern(
            Batch(4, 1).with_trailing_gap(3) + Batch(4, 1), harmonic_number=100
        )
        pattern.intensity = np.full(pattern.n_bunches, 2.0)
        pattern.intensity[pattern.tier("batch") == 1] = 0.5
        assert np.array_equal(
            pattern.intensity, [2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5]
        )


class TestIntegralInputs:
    def test_gap_rejects_fractional(self):
        with pytest.raises(ValueError, match="integer"):
            Gap(5.5)

    def test_gap_accepts_integral_float(self):
        assert Gap(5.0).n_buckets == 5

    def test_batch_rejects_fractional_n_bunches(self):
        with pytest.raises(ValueError, match="integer"):
            Batch(n_bunches=3.7, bunch_gap=2)

    def test_batch_accepts_integral_floats(self):
        batch = Batch(n_bunches=3.0, bunch_gap=2.0)
        assert batch.n_bunches == 3
        assert batch.n_buckets == 7

    def test_filling_pattern_rejects_fractional_harmonic_number(self):
        with pytest.raises(ValueError, match="integer"):
            FillingPattern(Gap(1), harmonic_number=10.5)


class TestValidation:
    def test_train_rejects_negative_copy_gap_even_for_single_copy(self):
        with pytest.raises(ValueError, match="copy_gap"):
            Train(Batch(3, 1), n_copies=1, copy_gap=-50)

    def test_harmonic_number_must_be_positive(self):
        with pytest.raises(ValueError, match="harmonic_number"):
            FillingPattern(Gap(0), harmonic_number=0)

    def test_n_groups_counts_distinct_indices(self):
        segment = PatternSegment(
            bucket_indices=np.array([0, 5]),
            n_buckets=10,
            tiers={"batch": np.array([0, 5])},
        )
        assert segment.n_groups("batch") == 2

    def test_n_groups_composed(self):
        two = Batch(2, 1).with_trailing_gap(3) + Batch(2, 1)
        assert two.n_groups("batch") == 2

    def test_n_groups_absent_tier(self):
        assert Gap(5).n_groups("batch") == 0


class TestFromPlacements:
    def test_negative_start_bucket_raises_clear_error(self):
        with pytest.raises(ValueError, match="start_bucket"):
            FillingPattern.from_placements(100, [(Batch(2, 1), -3)])

    def test_overlap_raises(self):
        segment = Batch(2, 1)  # occupies 3 buckets
        with pytest.raises(ValueError, match="overlap"):
            FillingPattern.from_placements(100, [(segment, 0), (segment, 2)])

    def test_disjoint_placements(self):
        pattern = FillingPattern.from_placements(
            100, [(Batch(2, 1), 10), (Batch(2, 1), 50)]
        )
        assert np.array_equal(pattern.bucket_indices, [10, 12, 50, 52])
        assert np.array_equal(pattern.tier("batch"), [0, 0, 1, 1])


class TestWarningLocation:
    def test_n_buckets_from_time_warns_at_caller(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            n_buckets_from_time(2.3, 1.0)
        assert len(caught) == 1
        assert caught[0].filename == __file__

    def test_from_spacing_warns_at_caller(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Batch.from_spacing(
                n_bunches=2, start_to_start_distance=2.3, f_rf=1.0
            )
        assert len(caught) == 1
        assert caught[0].filename == __file__


class TestRelativeTolerance:
    def test_standard_lhc_spacings_pass_silently(self):
        # Nominal ns spacings deviate from the exact bucket multiple by a
        # fixed *fraction* (~0.2 % at 400.789 MHz), so the absolute
        # deviation grows with distance; none of these may warn.
        f_rf = 400.789e6
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for spacing_ns in (25, 50, 75, 100, 225):
                n_buckets_from_time(spacing_ns * 1e-9, f_rf)

    def test_misaligned_distance_warns(self):
        with pytest.warns(UserWarning, match="not an integer"):
            n_buckets_from_time(2.3, 1.0)

    def test_zero_distance_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert n_buckets_from_time(0.0, 400.789e6) == 0


class TestpropertyDtype:
    def test_property_stored_as_float(self):
        batch = Batch(2, 1)
        batch.n_injected = np.array([1, 2])
        assert batch.n_injected.dtype == np.float64

    def test_int_property_merge_keeps_nan_contract(self):
        left = Batch(2, 1)
        left.n_injected = np.array([1, 2])
        merged = left + Batch(2, 1)
        assert np.all(np.isnan(merged.n_injected[2:]))

    def test_assignment_rejects_string_property(self):
        batch = Batch(2, 1)
        with pytest.raises(ValueError, match="tag"):
            batch.tag = np.array(["x", "y"])

    def test_constructor_rejects_string_property(self):
        with pytest.raises(ValueError, match="tag"):
            BunchTable(
                bucket_indices=np.array([0]),
                n_buckets=1,
                properties={"tag": np.array(["x"])},
            )


class TestReservedpropertyNames:
    def test_segment_rejects_harmonic_number_property(self):
        batch = Batch(2, 1)
        with pytest.raises(AttributeError, match="harmonic_number"):
            batch.harmonic_number = np.ones(2)

    def test_segment_rejects_has_bunch_property(self):
        batch = Batch(2, 1)
        with pytest.raises(AttributeError, match="has_bunch"):
            batch.has_bunch = np.ones(2)

    def test_constructor_rejects_harmonic_number_property(self):
        with pytest.raises(ValueError, match="harmonic_number"):
            BunchTable(
                bucket_indices=np.array([0]),
                n_buckets=1,
                properties={"harmonic_number": np.array([1.0])},
            )


class TestMultiplierValidation:
    def test_mul_rejects_fractional(self):
        with pytest.raises(ValueError, match="integer"):
            Batch(2, 1) * 2.5

    def test_mul_accepts_integral_float(self):
        assert (Batch(2, 1) * 2.0).n_bunches == 4

    def test_rmul_rejects_fractional(self):
        with pytest.raises(ValueError, match="integer"):
            2.5 * Batch(2, 1)


class TestRegressionGuards:
    def test_composition_renumbers_tiers(self):
        batch = Batch(n_bunches=2, bunch_gap=1)
        train = Train(unit=batch, n_copies=2, copy_gap=5)
        injection = train.label("injection")
        full = injection.with_trailing_gap(10) * 2
        assert np.array_equal(full.tier("batch"), [0, 0, 1, 1, 2, 2, 3, 3])
        assert np.array_equal(full.tier("train"), [0, 0, 0, 0, 1, 1, 1, 1])
        assert np.array_equal(full.tier("injection"), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_property_nan_merge(self):
        left = Batch(2, 1)
        left.intensity = np.array([1.0, 2.0])
        right = Batch(2, 1)
        merged = left + right
        assert np.array_equal(merged.intensity[:2], [1.0, 2.0])
        assert np.all(np.isnan(merged.intensity[2:]))

    def test_pickle_roundtrip(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        pattern.intensity = np.ones(4)
        restored = pickle.loads(pickle.dumps(pattern))
        assert np.array_equal(restored.bucket_indices, pattern.bucket_indices)
        assert np.array_equal(restored.intensity, pattern.intensity)
        assert restored.harmonic_number == 100

    def test_has_bunch(self):
        pattern = FillingPattern(Batch(2, 1), harmonic_number=5)
        assert np.array_equal(
            pattern.has_bunch, [True, False, True, False, False]
        )
