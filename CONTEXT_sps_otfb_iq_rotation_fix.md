# Session context — SPS OTFB beam-current IQ-rotation fix

Date: 2026-07-15
Branch: `feedback_tests` (BLonD submodule)

## Task

Two SPS tests in the muon-collider-blonder feedback suite were failing; find the
root cause and fix without breaking other production code (LHC comparisons, the
newly-refactored mucol feedback code).

Failing tests:
- `tests/unittests/physics/feedbacks/accelerators/sps/test_impulse_response.py::TestTravelingWaveCavity::test_vind`
- `...::TestTravelingWaveCavity::test_beam_fine_coarse`

## Root cause

The beam-induced voltage from the SPS OTFB was rotated by exactly **90 deg**
(`reference = (+1j) * measured`, uniform on both fine and coarse grids, dT = 0).
The impulse response (`h_beam`) was correct — only `V_IND_FINE_BEAM` /
`V_IND_COARSE_BEAM` were rotated.

Origin of the 90 deg:
- The shared `rf_beam_current` — `blond/physics/feedbacks/beam_current.py:237`
  (the mucol/reworked convention) demodulates with `exp(1j*(dphi + pi/2))`.
- blond2 legacy — `blond/legacy/blond2/llrf/signal_processing.py:264` — uses
  `exp(1j*dphi)` (no `+pi/2`). The SPS OTFB reference values encode the blond2
  convention.

The extra `pi/2` is the mucol convention. The LHC loop already compensates for it
in its own `circuit_track`
(`blond/experimental/physics/feedbacks/accelerators/lhc/cavity_feedback.py:425`,
`I_BEAM *= exp(1j*phi_s)`). The SPS OTFB — which inherits the same
`IQCavityFeedbackOld` base and calls the same shared `rf_beam_current` — never got
the analogous compensation, so its beam-induced voltage stayed rotated 90 deg.

## Fix

Added an SPS-local override of `rf_beam_current` in
`blond/experimental/physics/feedbacks/accelerators/sps/cavity_feedback.py`
(class `SPSOneTurnFeedback`, just before `circuit_track`). It calls `super()`
then applies the constant `+pi/2` (factor `+1j`) rotation to
`I_BEAM_FINE` and `I_BEAM_COARSE[-n_coarse:]`, bringing the beam current into the
SPS reference frame before `beam_model` uses it. Mirrors the LHC compensation.

Deliberately NOT changed (protects other production code):
- The shared `rf_beam_current` — the mucol feedback code needs the `+pi/2`.
- The `IQCavityFeedbackOld` base — the LHC comparison path relies on its convention.

## Verification (local venv `.venv\Scripts\python.exe`, run from `BLonD/`)

- SPS feedback suite: **23 passed, 7 skipped** (was 21 passed + 2 failed).
  The `TestSPSCavityFeedback` beam-loading / V_sum tests are
  `@unittest.skip("...beam feedback is anyway not working for now")` — they do
  not run, so they place no constraint on this path. If that SPS feedback work is
  ever re-enabled, re-check those baselines against this convention.
- LHC comparison-with-blond2: **21 passed** (unchanged baseline).
- mucol feedback suite (`unittests/physics/feedbacks/accelerators/mucol/` +
  `tests/unittests/physics/feedbacks/`): **360 passed, 17 skipped, 1 xfailed**.

## Open item (not addressed — separate from this fix)

The 1 mucol xfail is
`unittests/physics/feedbacks/accelerators/mucol/test_mtw_vs_nondriven_feedback.py::TestMultiTurnFeedbackVsConvolution::test_multiturn_nondivisible_harmonic`,
marked `@unittest.expectedFailure` (line 1246).

Verified runtime behavior: it raises `ValueError` at
`blond/physics/feedbacks/beam_current.py:469` ("Beam charge was downsampled into
the first coarse-grid cell"). This is a real, deterministic limitation: a harmonic
not divisible by `2 * n_sections` de-aligns the coarse-grid tiling from the
profile's zeroed leading edge, so beam charge lands in the first coarse cell and
the guard rejects it before any voltage is produced.

Design note: because the failure mode is a systematic, deterministic raise, an
`assertRaises(ValueError)` test would pin the contract more precisely (green for
the right reason, and sensitive if the raise ever changes) than
`@unittest.expectedFailure`, which xfails for any failure mode and turns into an
XPASS reminder once the coarse grid is fixed. The current author intent is the
latter — a "to-do marker" until incommensurate harmonics are supported. Left
unchanged pending a decision.
