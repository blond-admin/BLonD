import json
from pathlib import Path

import numpy as np
import pytest

import blond.examples.scripts
from blond.specifics.cern.lhc import filling_pattern_from_scheme_file

SCHEME_PATH = (
    Path(blond.examples.scripts.__file__).parent
    / "resources"
    / "EX_29"
    / "25ns_1500b_1488_684_729_240bpi_8inj_HItests_Fill4.json"
)


class TestSchemeFileLoader:
    def test_beam1_counts(self):
        pattern = filling_pattern_from_scheme_file(path=SCHEME_PATH, beam=1)
        assert pattern.harmonic_number == 35640
        assert pattern.n_bunches == 1500
        assert pattern.n_groups("injection") == 8
        # 1 + 1 + 6 injections of 5 PS batches = 32 batches
        assert pattern.n_groups("batch") == 32

    def test_bunches_sit_on_the_25ns_slot_grid(self):
        # Every LHC bunch occupies the first bucket of its 10-bucket
        # slot; any other remainder means the slot->bucket mapping broke.
        pattern = filling_pattern_from_scheme_file(path=SCHEME_PATH, beam=1)
        assert np.all(pattern.bucket_indices % 10 == 0)

    def test_pattern_matches_schemebeam_mask_independently(self):
        # Independent cross-check against the file's redundant per-slot
        # mask (the loader validates internally; this guards the test
        # itself against a loader that "validates" its own mistake).
        data = json.loads(SCHEME_PATH.read_text())
        for beam, key in ((1, "schemebeam1"), (2, "schemebeam2")):
            pattern = filling_pattern_from_scheme_file(
                path=SCHEME_PATH, beam=beam
            )
            slots_from_pattern = np.zeros(3564, dtype=int)
            slots_from_pattern[pattern.bucket_indices // 10] = 1
            assert np.array_equal(slots_from_pattern, data[key])

    def test_first_injection_position(self):
        # beam 1, first injection starts at LHC slot 20 -> bucket 200.
        pattern = filling_pattern_from_scheme_file(path=SCHEME_PATH, beam=1)
        assert pattern.bucket_indices[0] == 200

    def test_unequal_batches_within_one_injection(self):
        # Injection 0 has a single 12-bunch batch, injection 2 has five
        # 48-bunch batches: exactly the case the legacy loader's stride
        # formula got wrong. Pin the batch sizes per injection.
        pattern = filling_pattern_from_scheme_file(path=SCHEME_PATH, beam=1)
        injection = pattern.label("injection")
        assert (injection == 0).sum() == 12
        assert (injection == 1).sum() == 48
        assert (injection == 2).sum() == 5 * 48

    def test_invalid_beam_raises(self):
        with pytest.raises(ValueError, match="beam"):
            filling_pattern_from_scheme_file(path=SCHEME_PATH, beam=3)

    def test_bunch_beyond_abort_gap_keeper_raises(self, tmp_path):
        # AGK marks where the abort gap must begin (in RF buckets); a
        # scheme with bunches at or beyond it violates machine
        # protection and must be refused.
        data = json.loads(SCHEME_PATH.read_text())
        data["AGK"] = 200  # first bunch of beam 1 sits at bucket 200
        corrupt = tmp_path / "agk_violated.json"
        corrupt.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="AGK"):
            filling_pattern_from_scheme_file(path=corrupt, beam=1)

    def test_missing_agk_is_tolerated(self, tmp_path):
        # AGK is a sanity check, not construction data; files without
        # it still load.
        data = json.loads(SCHEME_PATH.read_text())
        del data["AGK"]
        no_agk = tmp_path / "no_agk.json"
        no_agk.write_text(json.dumps(data))
        pattern = filling_pattern_from_scheme_file(path=no_agk, beam=1)
        assert pattern.n_bunches == 1500

    def test_inconsistent_file_raises(self, tmp_path):
        # Corrupt the redundant slot mask: the loader's internal
        # cross-validation must refuse the file instead of silently
        # returning a pattern that contradicts it.
        data = json.loads(SCHEME_PATH.read_text())
        data["schemebeam1"][0] = 1  # slot 0 is empty in this scheme
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="schemebeam1"):
            filling_pattern_from_scheme_file(path=corrupt, beam=1)
