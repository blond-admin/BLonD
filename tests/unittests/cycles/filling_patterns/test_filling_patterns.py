import pickle
import warnings

import numpy as np
import pytest

from blond import (
    Batch,
    BunchTable,
    FillingPattern,
    Gap,
    PatternSegment,
    Train,
    n_buckets_from_time,
)


class TestNameCollisions:
    def test_constructor_rejects_label_quantity_collision(self):
        with pytest.raises(ValueError, match="foo"):
            BunchTable(
                bucket_indices=np.array([0]),
                n_buckets=1,
                labels={"foo": np.array([0])},
                quantities={"foo": np.array([1.0])},
            )

    def test_constructor_rejects_structural_quantity_name(self):
        with pytest.raises(ValueError, match="bucket_indices"):
            BunchTable(
                bucket_indices=np.array([0]),
                n_buckets=1,
                quantities={"bucket_indices": np.array([1.0])},
            )

    def test_add_rejects_label_quantity_collision(self):
        left = Batch(n_bunches=2, bunch_gap=1)
        left.intensity = np.array([1.0, 2.0])
        right = Batch(n_bunches=2, bunch_gap=1).with_label("intensity")
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

    def test_label_columns_read_only(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        with pytest.raises(ValueError):
            pattern.label("batch")[0] = 7

    def test_constructor_does_not_freeze_caller_array(self):
        caller_bucket_indices = np.array([0, 2], dtype=np.int64)
        table = BunchTable(bucket_indices=caller_bucket_indices, n_buckets=5)
        caller_bucket_indices[0] = 1
        assert table.bucket_indices[0] == 0

    def test_quantity_assignment_copies_source(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        source = np.ones(4)
        pattern.intensity = source
        source[0] = 999.0
        assert pattern.intensity[0] == 1.0

    def test_quantity_dict_is_snapshot(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        pattern.quantities["bogus"] = np.zeros(2)
        assert "bogus" not in pattern.quantities

    def test_labels_dict_is_snapshot(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        pattern.labels["bogus"] = np.zeros(4, dtype=np.int32)
        assert "bogus" not in pattern.labels

    def test_quantity_masked_assignment_idiom_still_works(self):
        pattern = FillingPattern(
            Batch(4, 1).with_trailing_gap(3) + Batch(4, 1), harmonic_number=100
        )
        pattern.intensity = np.full(pattern.n_bunches, 2.0)
        pattern.intensity[pattern.label("batch") == 1] = 0.5
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
            labels={"batch": np.array([0, 5])},
        )
        assert segment.n_groups("batch") == 2

    def test_n_groups_composed(self):
        two = Batch(2, 1).with_trailing_gap(3) + Batch(2, 1)
        assert two.n_groups("batch") == 2

    def test_n_groups_unknown_label_raises(self):
        # Same strictness as label(): a typo must fail loudly, not
        # silently report zero groups.
        with pytest.raises(KeyError, match="batch"):
            Gap(5).n_groups("batch")


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
        assert np.array_equal(pattern.label("batch"), [0, 0, 1, 1])


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


class TestQuantityDtype:
    def test_quantity_stored_as_float(self):
        batch = Batch(2, 1)
        batch.n_injected = np.array([1, 2])
        assert batch.n_injected.dtype == np.float64

    def test_int_quantity_merge_keeps_nan_contract(self):
        left = Batch(2, 1)
        left.n_injected = np.array([1, 2])
        merged = left + Batch(2, 1)
        assert np.all(np.isnan(merged.n_injected[2:]))

    def test_assignment_rejects_string_quantity(self):
        batch = Batch(2, 1)
        with pytest.raises(ValueError, match="tag"):
            batch.tag = np.array(["x", "y"])

    def test_constructor_rejects_string_quantity(self):
        with pytest.raises(ValueError, match="tag"):
            BunchTable(
                bucket_indices=np.array([0]),
                n_buckets=1,
                quantities={"tag": np.array(["x"])},
            )


class TestReservedQuantityNames:
    def test_segment_rejects_harmonic_number_quantity(self):
        batch = Batch(2, 1)
        with pytest.raises(AttributeError, match="harmonic_number"):
            batch.harmonic_number = np.ones(2)

    def test_segment_rejects_has_bunch_quantity(self):
        batch = Batch(2, 1)
        with pytest.raises(AttributeError, match="has_bunch"):
            batch.has_bunch = np.ones(2)

    def test_constructor_rejects_harmonic_number_quantity(self):
        with pytest.raises(ValueError, match="harmonic_number"):
            BunchTable(
                bucket_indices=np.array([0]),
                n_buckets=1,
                quantities={"harmonic_number": np.array([1.0])},
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
    def test_composition_renumbers_labels(self):
        batch = Batch(n_bunches=2, bunch_gap=1)
        train = Train(unit=batch, n_copies=2, copy_gap=5)
        injection = train.with_label("injection")
        full = injection.with_trailing_gap(10) * 2
        assert np.array_equal(full.label("batch"), [0, 0, 1, 1, 2, 2, 3, 3])
        assert np.array_equal(full.label("train"), [0, 0, 0, 0, 1, 1, 1, 1])
        assert np.array_equal(
            full.label("injection"), [0, 0, 0, 0, 1, 1, 1, 1]
        )

    def test_quantity_nan_merge(self):
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


class TestFromSpacingPhysics:
    # Pins the central convention: nominal ns spacings -> integer gaps.
    # A regression here silently shifts every bunch in the machine.

    def test_lhc_25ns_gives_bunch_gap_9_not_10(self):
        # 25 ns on the 400.789 MHz RF is 10.02 buckets -> stride 10,
        # i.e. bunch_gap=9. Writing bunch_gap=10 is THE classic mistake
        # this constructor exists to prevent; it must also not warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            batch = Batch.from_spacing(
                n_bunches=72, start_to_start_distance=25e-9, f_rf=400.789e6
            )
        assert np.array_equal(
            batch.bucket_indices, np.arange(72, dtype=np.int64) * 10
        )
        assert batch.n_buckets == 711  # 72 bunches + 71 gaps of 9

    def test_train_from_spacing_start_to_start(self):
        # Unit of 3 buckets, copies starting every 8 buckets (f_rf=1 Hz
        # makes seconds == buckets): copy_gap must be 8 - 3 = 5, NOT 8
        # (start-to-start vs gap confusion is the bug to prevent).
        unit = Batch(n_bunches=2, bunch_gap=1)  # buckets 0, 2 -> 3 long
        train = Train.from_spacing(
            unit, n_copies=2, start_to_start_distance=8.0, f_rf=1.0
        )
        assert np.array_equal(train.bucket_indices, [0, 2, 8, 10])
        assert train.n_buckets == 11  # no trailing gap after last copy

    def test_train_unit_trailing_gap_counts_toward_start_to_start(self):
        # A unit that ends in a trailing gap is longer; the same
        # start-to-start distance must yield a smaller copy_gap.
        unit = Batch(n_bunches=2, bunch_gap=1).with_trailing_gap(2)  # 5 long
        train = Train.from_spacing(
            unit, n_copies=2, start_to_start_distance=8.0, f_rf=1.0
        )
        assert np.array_equal(train.bucket_indices, [0, 2, 8, 10])


class TestWithLabelErrors:
    def test_duplicate_label_raises(self):
        labeled = Batch(2, 1).with_label("injection")
        with pytest.raises(ValueError, match="injection"):
            labeled.with_label("injection")

    def test_train_of_train_raises(self):
        # Nesting Train in Train would silently overwrite the inner
        # 'train' grouping; it must fail and point at the conflict.
        inner = Train(Batch(2, 1), n_copies=2, copy_gap=3)
        with pytest.raises(ValueError, match="train"):
            Train(inner, n_copies=2, copy_gap=10)

    def test_label_name_colliding_with_quantity_raises(self):
        segment = Batch(2, 1)
        segment.intensity = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="intensity"):
            segment.with_label("intensity")

    def test_with_label_returns_independent_copy(self):
        # with_label must deep-copy quantities: mutating the copy must
        # not write through to the original (and vice versa).
        segment = Batch(2, 1)
        segment.intensity = np.array([1.0, 2.0])
        labeled = segment.with_label("injection")
        labeled.intensity[0] = 99.0
        assert segment.intensity[0] == 1.0
        segment.intensity[1] = -5.0
        assert labeled.intensity[1] == 2.0

    def test_label_unknown_lists_available(self):
        segment = Batch(2, 1)
        with pytest.raises(KeyError, match="batch"):
            segment.label("injction")


class TestCompositionAlgebra:
    # The module promises an algebra; these pin its laws so a refactor
    # of _merge_columns/__add__ cannot quietly bend them.

    def _abc(self):
        a = Batch(2, 0)
        a.intensity = np.array([1.0, 2.0])
        b = Gap(3)
        c = Batch(3, 1).with_label("injection")
        return a, b, c

    def test_concatenation_is_associative(self):
        a, b, c = self._abc()
        left = (a + b) + c
        right = a + (b + c)
        assert np.array_equal(left.bucket_indices, right.bucket_indices)
        assert left.n_buckets == right.n_buckets
        assert sorted(left.labels) == sorted(right.labels)
        for name in left.labels:
            assert np.array_equal(left.label(name), right.label(name))
        np.testing.assert_array_equal(left.intensity, right.intensity)

    def test_repeat_equals_explicit_addition(self):
        _, _, c = self._abc()
        assert np.array_equal(
            (c * 3).bucket_indices, (c + c + c).bucket_indices
        )
        assert np.array_equal(
            (c * 3).label("injection"), [0, 0, 0, 1, 1, 1, 2, 2, 2]
        )

    def test_rmul_equals_mul(self):
        _, _, c = self._abc()
        assert np.array_equal((2 * c).bucket_indices, (c * 2).bucket_indices)

    def test_empty_gap_is_concatenation_identity(self):
        a, _, _ = self._abc()
        for combined in (a + Gap(0), Gap(0) + a):
            assert np.array_equal(combined.bucket_indices, a.bucket_indices)
            assert combined.n_buckets == a.n_buckets
            assert np.array_equal(combined.label("batch"), a.label("batch"))

    def test_filling_pattern_preserves_segment_exactly(self):
        a, b, c = self._abc()
        segment = a + b + c
        pattern = FillingPattern(segment, harmonic_number=50)
        assert np.array_equal(pattern.bucket_indices, segment.bucket_indices)
        for name in segment.labels:
            assert np.array_equal(pattern.label(name), segment.label(name))
        np.testing.assert_array_equal(pattern.intensity, segment.intensity)
        assert pattern.n_buckets == 50  # n_buckets becomes the ring size


class TestRenumbering:
    def test_one_sided_labels_stay_unassigned_on_the_other_side(self):
        # 'batch' exists only left, 'x' only right: the missing side
        # must be -1, and the right side's 'x' must start at 0 because
        # an all-unassigned left column contributes no used indices.
        left = Batch(2, 0)
        right = PatternSegment(
            bucket_indices=np.array([0, 1]),
            n_buckets=2,
            labels={"x": np.array([0, 0])},
        )
        combined = left + right
        assert np.array_equal(combined.label("batch"), [0, 0, -1, -1])
        assert np.array_equal(combined.label("x"), [-1, -1, 0, 0])

    def test_renumbering_uses_max_plus_one_not_group_count(self):
        # Hand-built non-contiguous indices: the right side must shift
        # past the maximum (5 -> offset 6), not past the count (2).
        left = PatternSegment(
            bucket_indices=np.array([0, 5]),
            n_buckets=10,
            labels={"g": np.array([0, 5])},
        )
        right = PatternSegment(
            bucket_indices=np.array([0]),
            n_buckets=1,
            labels={"g": np.array([0])},
        )
        combined = left + right
        assert np.array_equal(combined.label("g"), [0, 5, 6])
        assert combined.n_groups("g") == 3


class TestQuantityMergeSemantics:
    def test_both_sides_concatenate_in_order(self):
        a = Batch(2, 0)
        a.intensity = np.array([1.0, 2.0])
        b = Batch(2, 0)
        b.intensity = np.array([3.0, 4.0])
        assert np.array_equal((a + b).intensity, [1.0, 2.0, 3.0, 4.0])

    def test_repeat_duplicates_quantities_per_copy(self):
        a = Batch(2, 0)
        a.intensity = np.array([1.0, 2.0])
        assert np.array_equal((a * 2).intensity, [1.0, 2.0, 1.0, 2.0])

    def test_nan_input_is_legal_unspecified_marker(self):
        a = Batch(2, 0)
        a.bunch_length = np.array([np.nan, 1e-9])
        assert np.isnan(a.bunch_length[0])
        assert a.bunch_length[1] == 1e-9


class TestCompletePatternIsSealed:
    def _pattern_and_segment(self):
        segment = Batch(2, 1)
        return FillingPattern(segment, harmonic_number=10), segment

    def test_segment_plus_pattern_raises(self):
        pattern, segment = self._pattern_and_segment()
        with pytest.raises(TypeError, match="complete"):
            segment + pattern

    def test_pattern_plus_segment_raises(self):
        pattern, segment = self._pattern_and_segment()
        with pytest.raises(TypeError):
            pattern + segment

    def test_pattern_cannot_be_repeated(self):
        pattern, _ = self._pattern_and_segment()
        with pytest.raises(TypeError):
            pattern * 2

    def test_segment_longer_than_ring_raises(self):
        with pytest.raises(ValueError, match="harmonic_number"):
            FillingPattern(Gap(101), harmonic_number=100)


class TestFromPlacementsBoundaries:
    def test_adjacent_placement_at_exact_end_is_allowed(self):
        segment = Batch(2, 0)  # 2 buckets
        pattern = FillingPattern.from_placements(
            10, [(segment, 0), (Batch(1, 0), 2)]
        )
        assert np.array_equal(pattern.bucket_indices, [0, 1, 2])

    def test_trailing_gap_blocks_placement_inside_it(self):
        # Documented: a segment's range includes its trailing gap.
        with_gap = Batch(2, 0).with_trailing_gap(5)  # occupies [0, 7)
        with pytest.raises(ValueError, match="overlap"):
            FillingPattern.from_placements(
                100, [(with_gap, 0), (Batch(1, 0), 5)]
            )

    def test_placement_beyond_ring_raises(self):
        with pytest.raises(ValueError, match="harmonic_number"):
            FillingPattern.from_placements(100, [(Batch(2, 0), 99)])

    def test_empty_placements_give_empty_ring(self):
        pattern = FillingPattern.from_placements(100, [])
        assert pattern.n_bunches == 0
        assert not pattern.has_bunch.any()

    def test_placements_preserve_quantities(self):
        segment = Batch(2, 0)
        segment.intensity = np.array([1.0, 2.0])
        pattern = FillingPattern.from_placements(
            10, [(segment, 3), (Batch(1, 0), 7)]
        )
        assert np.array_equal(pattern.intensity[:2], [1.0, 2.0])
        assert np.isnan(pattern.intensity[2])


class TestEdgeCases:
    def test_single_bunch_batch_has_one_bucket(self):
        # Off-by-one trap in n + (n-1)*gap: the gap must not count for
        # a single bunch.
        batch = Batch(n_bunches=1, bunch_gap=9)
        assert batch.n_buckets == 1
        assert np.array_equal(batch.bucket_indices, [0])

    def test_with_trailing_gap_zero_is_identity(self):
        segment = Batch(3, 1)
        same = segment.with_trailing_gap(0)
        assert np.array_equal(same.bucket_indices, segment.bucket_indices)
        assert same.n_buckets == segment.n_buckets

    def test_has_bunch_returns_fresh_array(self):
        # The occupancy is derived data: corrupting one returned array
        # must not corrupt subsequent reads.
        pattern = FillingPattern(Batch(2, 1), harmonic_number=5)
        first = pattern.has_bunch
        first[:] = False
        assert pattern.has_bunch.sum() == 2

    def test_half_bucket_distance_warns(self):
        with pytest.warns(UserWarning, match="not an integer"):
            n_buckets_from_time(10.5, 1.0)


class TestLhcPatternEndToEnd:
    def test_full_lhc_like_pattern_numbers(self):
        # The EX_29 pattern with every count computed by hand:
        # batch: 72 bunches, stride 10 -> 711 buckets
        # train: 4 batches, copy_gap 8 -> 4*711 + 3*8 = 2868 buckets
        # injection + 38 gap -> 2906; 11 copies + bare injection
        #   -> 11*2906 + 2868 = 34834 buckets, last bunch at 34833
        batch = Batch(n_bunches=72, bunch_gap=9)
        train = Train(unit=batch, n_copies=4, copy_gap=8)
        injection = train.with_label("injection")
        pattern = FillingPattern(
            injection.with_trailing_gap(38) * 11 + injection,
            harmonic_number=35640,
        )
        assert pattern.n_bunches == 12 * 4 * 72  # 3456
        assert pattern.n_groups("batch") == 48
        assert pattern.n_groups("train") == 12
        assert pattern.n_groups("injection") == 12
        assert pattern.bucket_indices[-1] == 34833
        abort_gap = pattern.harmonic_number - 1 - pattern.bucket_indices[-1]
        assert abort_gap == 806
        assert pattern.has_bunch.sum() == 3456
