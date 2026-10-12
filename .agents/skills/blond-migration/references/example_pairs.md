# Legacy example ↔ BLonD 3 example

Reading a matching pair side by side is the fastest way to absorb the BLonD 3
idiom for a given piece of physics. Legacy scripts live in
`legacy/__EXAMPLES/main_files/`, current ones in `blond/examples/scripts/`.

The pairings below are by *physics topic*, not by number — the numbering was
reshuffled. Where a row says "closest", the new script demonstrates the
technique but not the same machine or study.

| Legacy `legacy/__EXAMPLES/main_files/` | BLonD 3 `blond/examples/scripts/` | Note |
|---|---|---|
| `EX_01_Acceleration.py` (LHC ramp, no intensity effects) | `EX_01_Minimum_working_example.py`, then `EX_06_Acceleration_match_density.py` | `EX_01` is the smallest end-to-end run; `EX_02_Magnetic_ramp_by_turn.py` is the direct analogue of the `np.linspace` momentum programme |
| `EX_02_Main_long_ps_booster.py` (PSB, impedance tables) | `EX_23_Main_long_ps_booster.py` | direct port of the same study — **the reference for any PSB impedance script** |
| `EX_03_RFnoise.py` | `EX_13_RFnoise.py` | check noise-generator coverage against the legacy `rf_noise.py` use |
| `EX_04_Stationary_multistation.py` | `EX_14_Stationary_multistation.py` | direct; note `section_index` is 0-based now |
| `EX_05_Wake_impedance.py` (SPS, resonator table, time + freq domain) | `EX_05_Wake_impedance.py` | direct; also `EX_12_Wake_impedance_pooled.py` |
| `EX_06_Preprocess.py` (`preprocess_ramp`, PSB) | `EX_03_Magnetic_ramp_by_time.py` | `RingOptions` preprocessing → `MagneticCycleByTime(interpolator=...)` |
| `EX_07_Ions.py` | `EX_17_Ions.py` | direct |
| `EX_08_Phase_Loop.py` (PSB) | — | **no stable equivalent**; `blond/experimental/physics/feedbacks/beam_feedback.py` |
| `EX_09_Radial_Loop.py` (PSB) | — | **no stable equivalent** (as above) |
| `EX_10_Fixed_frequency.py` (ω_rf ≠ h·ω_rev, PSB) | closest: `EX_21_Acceleration_revolution_time.py` | legacy `RFStation(omega_rf=...)` has no direct constructor equivalent — verify carefully |
| `EX_11_comparison_music_fourier_analytical.py` | closest: `EX_05_Wake_impedance.py` | MuSiC tracking has no BLonD 3 element |
| `EX_12_synchrotron_frequency_distribution.py` | — | `blond/acc_math/` and `blond/utilities/separatrix/` hold the maths; no example |
| `EX_13_synchrotron_radiation.py` (FCC-ee) | `EX_11_Synchrotron_Radiation.py`, `EX_27_Synchrotron_Radiation_Matched.py` | see also `blond/specifics/fccee/` |
| `EX_14_sparse_slicing.py` | `EX_20_Acceleration_sparse_profiles.py` | `EquidistantMultiProfile` |
| `EX_15_sparse_multi_bunch.py` | `EX_28_Multiturn_sparse_sps.py`, `EX_04_Multibunch_beam.py` | `make_multibunch_beam` |
| `EX_16_impedance_test.py` (PSB, narrow-band resonator) | `EX_05_Wake_impedance.py` | single-turn wake |
| `EX_17_multi_turn_wake.py` (PSB) | `EX_09_Multi_turn_wake.py` | direct |
| `EX_18_robinson_instability.py` (PSB) | closest: `EX_09_Multi_turn_wake.py` | same machinery, different study |
| `EX_19_bunch_generation.py` (LHC, matched distributions) | `EX_08_Semi_empiric_matcher.py`, `EX_18_Filamentation_matcher.py` | **not a drop-in** — different matching algorithms |
| `EX_20_bunch_generation_multibunch.py` | `EX_04_Multibunch_beam.py` | |
| `EX_21_bunch_distribution.py` (`toolbox.action`) | — | `toolbox/` largely absent |
| `EX_22_Coherent_Radiation.py` (CSR impedance) | — | `CoherentSynchrotronRadiation` source has no equivalent |
| `EX_23_Barrier_Bucket.py` | `EX_15_Barrier_Bucket.py` | `BarrierRF` |
| `EX_23…EX_27_single_particle_*.py` (LHC/PSB single particle, one-turn matrix, ramps) | `tests/integration/blond2_regression/tracking/test_kickdrift.py` | that test *is* the worked single-particle port, with the BLonD 2 reference alongside — the best template for verification |
| `EX_28_phase_loop_lhc.py` | — | **no stable equivalent** |
| `EX_29_haissinski_solution.py` | — | no equivalent |

BLonD 3 examples with no legacy ancestor, worth knowing about when a port needs
something the old API could not express:

- `EX_07_Acceleration_interrupted.py` — stopping and resuming a run
- `EX_10_MultiRFManipulation_TripleSplitting.py` — scheduled multi-harmonic manipulations
- `EX_16_MuonCollider_asynchronous_ramp.py`
- `EX_19_Observable_as_element.py` — observations placed inside the ring
- `EX_22_Acceleration_no_beam.py` — cycle without tracking
- `EX_24_Xsuite_Matching.py` — XSuite interop
- `EX_25_main_user.py` — a realistic user-style main file
- `EX_26_custom_trackable.py` — `UserDefinedElement`, the escape hatch for
  custom per-turn physics (useful when porting something with no element)

Machine-specific helpers live in `blond/specifics/cern/{lhc,ps,psb,sps}/`,
`blond/specifics/fccee/` and `blond/specifics/muon_collider/` — check there
before hand-rolling machine parameters.
