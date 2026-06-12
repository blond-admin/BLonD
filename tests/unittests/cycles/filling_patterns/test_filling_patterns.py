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
    as_n_buckets,
)


class TestNameCollisions:
    def test_constructor_rejects_tier_payload_collision(self):
        with pytest.raises(ValueError, match="foo"):
            BunchTable(
                positions=np.array([0]),
                length=1,
                tiers={"foo": np.array([0])},
                payload={"foo": np.array([1.0])},
            )

    def test_constructor_rejects_structural_payload_name(self):
        with pytest.raises(ValueError, match="positions"):
            BunchTable(
                positions=np.array([0]),
                length=1,
                payload={"positions": np.array([1.0])},
            )

    def test_add_rejects_tier_payload_collision(self):
        left = Batch(n_bunches=2, bunch_spacing=1)
        left.intensity = np.array([1.0, 2.0])
        right = Batch(n_bunches=2, bunch_spacing=1).label("intensity")
        with pytest.raises(ValueError, match="intensity"):
            left + right


class TestFromSpacingErrors:
    def test_train_from_spacing_too_short_distance(self):
        unit = Batch(n_bunches=72, bunch_spacing=9)
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
    def test_positions_read_only(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        with pytest.raises(ValueError):
            pattern.positions[0] = 99

    def test_tier_columns_read_only(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        with pytest.raises(ValueError):
            pattern.tier("batch")[0] = 7

    def test_constructor_does_not_freeze_caller_array(self):
        caller_positions = np.array([0, 2], dtype=np.int64)
        table = BunchTable(positions=caller_positions, length=5)
        caller_positions[0] = 1
        assert table.positions[0] == 0

    def test_payload_assignment_copies_source(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        source = np.ones(4)
        pattern.intensity = source
        source[0] = 999.0
        assert pattern.intensity[0] == 1.0

    def test_payload_dict_is_snapshot(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        pattern.payload["bogus"] = np.zeros(2)
        assert "bogus" not in pattern.payload

    def test_tiers_dict_is_snapshot(self):
        pattern = FillingPattern(Batch(4, 1), harmonic_number=100)
        pattern.tiers["bogus"] = np.zeros(4, dtype=np.int32)
        assert "bogus" not in pattern.tiers

    def test_payload_masked_assignment_idiom_still_works(self):
        pattern = FillingPattern(
            Batch(4, 1).gap(3) + Batch(4, 1), harmonic_number=100
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
        assert Gap(5.0).length == 5

    def test_batch_rejects_fractional_n_bunches(self):
        with pytest.raises(ValueError, match="integer"):
            Batch(n_bunches=3.7, bunch_spacing=2)

    def test_batch_accepts_integral_floats(self):
        batch = Batch(n_bunches=3.0, bunch_spacing=2.0)
        assert batch.n_bunches == 3
        assert batch.length == 7

    def test_filling_pattern_rejects_fractional_harmonic_number(self):
        with pytest.raises(ValueError, match="integer"):
            FillingPattern(Gap(1), harmonic_number=10.5)


class TestValidation:
    def test_train_rejects_negative_copy_spacing_even_for_single_copy(self):
        with pytest.raises(ValueError, match="copy_spacing"):
            Train(Batch(3, 1), n_copies=1, copy_spacing=-50)

    def test_harmonic_number_must_be_positive(self):
        with pytest.raises(ValueError, match="harmonic_number"):
            FillingPattern(Gap(0), harmonic_number=0)

    def test_n_in_tier_counts_distinct_indices(self):
        segment = PatternSegment(
            positions=np.array([0, 5]),
            length=10,
            tiers={"batch": np.array([0, 5])},
        )
        assert segment.n_in_tier("batch") == 2

    def test_n_in_tier_composed(self):
        two = Batch(2, 1).gap(3) + Batch(2, 1)
        assert two.n_in_tier("batch") == 2

    def test_n_in_tier_absent_tier(self):
        assert Gap(5).n_in_tier("batch") == 0


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
        assert np.array_equal(pattern.positions, [10, 12, 50, 52])
        assert np.array_equal(pattern.tier("batch"), [0, 0, 1, 1])


class TestWarningLocation:
    def test_as_n_buckets_warns_at_caller(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            as_n_buckets(2.3, 1.0)
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


class TestRegressionGuards:
    def test_composition_renumbers_tiers(self):
        batch = Batch(n_bunches=2, bunch_spacing=1)
        train = Train(unit=batch, n_copies=2, copy_spacing=5)
        injection = train.label("injection")
        full = injection.gap(10) * 2
        assert np.array_equal(full.tier("batch"), [0, 0, 1, 1, 2, 2, 3, 3])
        assert np.array_equal(full.tier("train"), [0, 0, 0, 0, 1, 1, 1, 1])
        assert np.array_equal(full.tier("injection"), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_payload_nan_merge(self):
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
        assert np.array_equal(restored.positions, pattern.positions)
        assert np.array_equal(restored.intensity, pattern.intensity)
        assert restored.harmonic_number == 100

    def test_has_bunch(self):
        pattern = FillingPattern(Batch(2, 1), harmonic_number=5)
        assert np.array_equal(
            pattern.has_bunch, [True, False, True, False, False]
        )
