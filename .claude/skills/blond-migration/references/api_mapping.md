# BLonD 2 → BLonD 3 API mapping

Ground truth is always the source: legacy under `blond/legacy/blond2/`, current
under `blond/`. This table records the translation and the traps; when it
disagrees with the code, the code wins — and please flag the discrepancy.

## Contents

1. [Imports and entry point](#1-imports-and-entry-point)
2. [Ring, optics and the energy programme](#2-ring-optics-and-the-energy-programme)
3. [RF stations](#3-rf-stations)
4. [Trackers and the turn loop](#4-trackers-and-the-turn-loop)
5. [Beam and beam generation](#5-beam-and-beam-generation)
6. [Profiles](#6-profiles)
7. [Impedances and induced voltage](#7-impedances-and-induced-voltage)
8. [Losses](#8-losses)
9. [Monitors, plots and output](#9-monitors-plots-and-output)
10. [Turn-by-turn programmes: `schedule()`](#10-turn-by-turn-programmes-schedule)
11. [Backends](#11-backends)
12. [Synchrotron radiation](#12-synchrotron-radiation)
13. [No stable equivalent — flag these](#13-no-stable-equivalent--flag-these)
14. [Trap checklist](#14-trap-checklist)

---

## 1. Imports and entry point

Everything public comes from the top-level package:

```python
from blond import Ring, Simulation, SingleHarmonicRFStation, DriftSimple, Beam, proton
```

`blond/__init__.py` is the definitive list of the supported public API. If a
name is not there, it is either an internal module path
(`blond.physics.impedances.readers`, `blond.handle_results.helpers`) or
experimental.

Legacy scripts are flat module-level code. Port into a `def main():` with an
`if __name__ == "__main__": main()` guard, matching `blond/examples/scripts/`.
This is what makes `Simulation.from_locals(locals())` usable.

Standard preamble in BLonD 3 examples:

```python
from blond import setup_backend
from blond.testing import pytest_active

if not pytest_active():  # pragma: no cover
    setup_backend("auto")
```

---

## 2. Ring, optics and the energy programme

Legacy signature:

```python
Ring(ring_length, alpha_0, synchronous_data, particle, n_turns=1,
     synchronous_data_type="momentum", bending_radius=None, n_sections=1,
     alpha_1=None, alpha_2=None, ring_options=None)
```

This single call becomes **three** BLonD 3 objects.

| Legacy argument | BLonD 3 home |
|---|---|
| `ring_length` (circumference) | `Ring(circumference=...)` **and** `DriftSimple(orbit_length=...)` |
| `alpha_0` | `DriftSimple(momentum_compaction_factor=...)` |
| `synchronous_data` (+ `synchronous_data_type`, `bending_radius`) | a `MagneticCycle*` object |
| `particle` | `reference_particle=` on the magnetic cycle, and `particle_type=` on `Beam` |
| `n_turns` | argument of `sim.run_simulation(n_turns=...)` |
| `n_sections` | `section_index=` on each element (**0-based**, see traps) |
| `ring_options=RingOptions(...)` (preprocessing/interpolation of a time programme) | `MagneticCycleByTime(..., interpolator=...)` |

```python
# BLonD 2
ring = Ring(26658.883, 1/55.759505**2, np.linspace(450e9, 460e9, N_t + 1),
            Proton(), N_t)

# BLonD 3
ring = Ring(circumference=26658.883)
drift = DriftSimple(
    orbit_length=26658.883,
    momentum_compaction_factor=momentum_compaction_factor(
        transition_gamma=55.759505),
)
programme = np.linspace(450e9, 460e9, N_t + 1)
magnetic_cycle = MagneticCyclePerTurn(
    reference_particle=proton,
    value_init=programme[0],
    values_after_turn=programme[1:],
    in_unit="momentum",
)
ring.add_elements((drift, rf_station), reorder=True)
sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
```

Particle types: legacy `Proton()` is an instantiated class; BLonD 3 `proton` is
a module-level singleton — **no parentheses**. Available: `proton`, `electron`,
`positron`, `mu_plus`, `mu_minus`, `uranium_29`.

Momentum compaction: legacy examples usually spell `1/gamma_transition**2`
inline. `momentum_compaction_factor(transition_gamma=...)` is the BLonD 3
helper for the same quantity (it returns exactly `1/γ_tr²`); a plain float is
equally acceptable.

**Higher-order compaction** (legacy `alpha_1`, `alpha_2`) is *not* a
`DriftSimple` argument — it needs `DriftExact`:

```python
from blond.physics.drifts import DriftExact
drift = DriftExact(
    orbit_length=...,
    momentum_compaction_factor=alpha_0,
    higher_order_alpha=np.array([alpha_1, alpha_2]),   # ascending order
)
```

Silently dropping `alpha_1` by porting onto `DriftSimple` changes the
off-momentum dynamics. If the legacy `Ring(...)` passed `alpha_1`/`alpha_2`,
use `DriftExact` and say so.

### Choosing the magnetic cycle

| Legacy | BLonD 3 |
|---|---|
| scalar `synchronous_data` (no acceleration) | `ConstantMagneticCycle(value=..., reference_particle=..., in_unit=...)` |
| array of length `n_turns + 1` | `MagneticCyclePerTurn(value_init=arr[0], values_after_turn=arr[1:], ...)` |
| `(time_array, value_array)` tuple + `RingOptions` preprocessing | `MagneticCycleByTime(reference_time=..., reference_values=..., interpolator=...)` |
| per-section programme (`n_sections > 1`) | `MagneticCyclePerTurnAllRFStations` |

`in_unit` replaces `synchronous_data_type` and takes the same vocabulary:
`"momentum"` [eV/c], `"total energy"` [eV], `"kinetic energy"` [eV],
`"bending field"` [T] (with `bending_radius=` in [m]).

**Array length.** A legacy programme has `n_turns + 1` entries. `value_init`
takes the first, `values_after_turn` takes the remaining `n_turns`. Passing the
whole array to `values_after_turn` silently shifts the entire ramp by one turn.

---

## 3. RF stations

```python
# BLonD 2 — lists, even for a single system; ring threaded in
RFStation(ring, harmonic, voltage, phi_rf_d, n_rf=1, section_index=1,
          omega_rf=None, phi_noise=None, phi_modulation=None,
          rf_station_options=None)

# BLonD 3 — the station is itself the ring element
SingleHarmonicRFStation(voltage=6e6, phi_rf=0.0, harmonic=35640,
                        section_index=0, local_wakefield=None,
                        cavity_feedback=None, beam_feedback=None, name=None)

MultiHarmonicRFStation(n_harmonics=2, main_harmonic_idx=0,
                       voltage=np.array([...]), phi_rf=np.array([...]),
                       harmonic=np.array([...]), section_index=0)
```

- `n_rf == 1` → `SingleHarmonicRFStation` with **scalars**, not one-element
  lists. `n_rf > 1` → `MultiHarmonicRFStation` with arrays plus
  `n_harmonics=` and `main_harmonic_idx=` (legacy had no explicit main-harmonic
  index; it is usually the lowest-frequency system, index 0).
- `phi_rf_d` ("design phase", often written `phi_offset` in legacy scripts) is
  the constructor argument `phi_rf=` but the **attribute is
  `phi_rf_design`**. `rf.phi_rf` on a BLonD 3 station is the actual /
  instantaneous phase and is a different quantity — do not conflate them.
  Setting after construction therefore reads:
  ```python
  rf = SingleHarmonicRFStation()
  rf.harmonic = 35640
  rf.voltage = 6e6
  rf.phi_rf_design = np.pi
  ```
- No `ring` argument. The station learns `t_rev`, `omega_rf`, β, γ from
  `Simulation`'s late init.
- `omega_rf=` (fixed-frequency operation) and `phi_modulation=` /
  `phi_noise=` have no direct constructor equivalent; look at
  `blond/cycles/noise_generators/` and `EX_13_RFnoise.py` for the BLonD 3 way
  of injecting RF noise, and flag anything you cannot reproduce.

---

## 4. Trackers and the turn loop

`RingAndRFTracker` and `FullRingAndRF` **do not exist in BLonD 3** and need no
replacement — the RF station *is* the tracking element and `Simulation` owns
the loop.

```python
# BLonD 2
long_tracker = RingAndRFTracker(rf, beam, solver="simple",
                                profile=profile, total_induced_voltage=tiv)
map_ = [long_tracker, profile, bunchmonitor, plots]
for i in range(1, N_t + 1):
    for m in map_:
        m.track()

# BLonD 3
ring.add_elements((rf_station, drift, wakefield), reorder=False)
sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
sim.print_one_turn_execution_order()      # read this, always
sim.run_simulation(beams=(beam,), n_turns=N_t, observe=(...,),
                   callbacks=my_callback)
```

Argument translation:

| Legacy `RingAndRFTracker` argument | BLonD 3 |
|---|---|
| `rf_station`, `beam` | not needed — wired by `Simulation` |
| `solver="simple"` | `DriftSimple` |
| `solver="exact"` | `DriftExact` (`blond.physics.drifts`) — also the one that accepts `higher_order_alpha` |
| `solver="legacy"` | no counterpart — flag |
| `profile=` | a `StaticProfile` / `DynamicProfileConstNBins` element added to the ring |
| `total_induced_voltage=` | a `WakeField` element added to the ring |
| `interpolation=True` | how the RF kick is applied when intensity effects are present — verify numerically rather than assuming a default matches |
| `beam_feedback=`, `cavity_feedback=` | see §13 |
| `periodicity=` | no direct equivalent — flag |

Anything the legacy script did *inside* the loop (progress prints, custom
diagnostics, per-turn manipulation of `beam.dt`) becomes a `callbacks=`
function. Note it is `callbacks=` — plural. The callback receives the
`Simulation` (and, for some signatures, the beam); `simulation.turn_counter.value`
is the current turn.

`ring.add_elements(elements, reorder=False, deepcopy=False, section_index=None)`:
`reorder=False` preserves the order you give (use it to mirror a legacy `map_`
exactly), `reorder=True` lets BLonD order elements by physics.

---

## 5. Beam and beam generation

```python
# BLonD 2
beam = Beam(ring, n_macroparticles, intensity)
bigaussian(ring, rf_station, beam, sigma_dt, sigma_dE=None, seed=1234,
           reinsertion=False)

# BLonD 3
beam = Beam(intensity=1e9, particle_type=proton)
sim.prepare_beam(
    beam=beam,
    preparation_routine=BiGaussian(n_macroparticles=1e6, sigma_dt=0.1e-9,
                                   sigma_dE=None, reinsertion=False, seed=0),
)
```

- **`n_macroparticles` moved off `Beam` and onto the preparation routine.** It
  may be passed as a float (`1e6`).
- `prepare_beam` must be called **after** `Simulation(...)` — matching needs the
  bucket, which only exists once everything is linked.
- Default seeds differ (legacy `bigaussian` uses `1234`, `BiGaussian` uses `0`).
  The RNG streams are not identical anyway, so do not expect particle-for-
  particle agreement between frameworks; compare distribution moments instead.
- To port a script that set coordinates by hand
  (`Beam(ring, n, I, dt=..., dE=...)`): `beam.setup_beam(dt=..., dE=...)`.

Matching routines:

| Legacy | BLonD 3 |
|---|---|
| `bigaussian` | `BiGaussian` (stable, `blond.beam_preparation.bigaussian`) |
| `matched_from_distribution_function` | no drop-in. `FilamentationMatcher` (`blond/beam_preparation/`) or `SemiEmpiricMatcher` / `EmpiricMatcher` (`blond/experimental/`). Different algorithm and different arguments — flag it to the user. |
| `matched_from_line_density` | as above — flag |
| coasting-beam helpers | `Coasting` (`blond.beam_preparation.coasting`) |
| multibunch distributions (`distributions_multibunch`) | `make_multibunch_beam`; see `EX_04_Multibunch_beam.py` |

Reading coordinates back:

| Legacy | BLonD 3 |
|---|---|
| `beam.dt`, `beam.dE` (plain NumPy) | `beam.read_partial_dt()` / `read_partial_dE()` for reading, `write_partial_dt()` / `write_partial_dE()` for in-place modification |
| `np.mean(beam.dt)` | `backend.mean(beam.read_partial_dt())`, or `np.mean(copy_to_cpu(beam.read_partial_dt()))` |

The returned array may live on the **GPU**. Never call `.get()`,
`np.array(...)` or `np.asarray(...)` on it directly — use
`copy_to_cpu(arr)` from `blond.generals.cupy_.no_cupy_import` (exported as
`blond.copy_to_cpu`). Legacy scripts are full of bare NumPy calls on
`beam.dt`; every one of them needs this treatment. Keep the conversion out of
the per-turn loop — it forces a host↔device round trip.

`beam.sigma_dt`, `beam.mean_dt` and friends: check
`blond/core/beam/beams.py` for what exists; `beam.rms_emittance` and the
statistics observations cover most legacy uses.

---

## 6. Profiles

```python
# BLonD 2
profile = Profile(beam,
                  CutOptions(cut_left=0, cut_right=2*np.pi, n_slices=2**8,
                             rf_station=rf, cuts_unit="rad"),
                  FitOptions(fit_option="gaussian"))

# BLonD 3 — seconds
profile = StaticProfile(cut_left=-5.7e-7, cut_right=5.7e-7, n_bins=10_000)

# BLonD 3 — radians (needs the period explicitly, since there is no `ring`)
profile = StaticProfile.from_rad(
    0, 2 * np.pi, 2**8,
    magnetic_cycle.get_t_rev_init(ring.circumference, particle_type=proton) / harmonic,
)
```

- `n_slices` → **`n_bins`**.
- `cuts_unit="rad"` → `StaticProfile.from_rad(cut_left_rad, cut_right_rad,
  n_bins, t_period)`. The `t_period` argument is what `rf_station=` used to
  supply implicitly.
- `n_sigma=` (auto-sizing cuts) → `DynamicProfileConstNBins(n_bins=...)`, which
  tracks the beam instead of fixed cuts.
- `FitOptions(fit_option="gaussian")` has no profile-level equivalent. Bunch
  length now comes from observations / `beam.rms_emittance`, or you fit the
  recorded profile yourself. Say so if the legacy script consumed
  `profile.bunchLength`.
- The profile is a **ring element** — add it to the ring (or pass it as
  `WakeField(profile=...)`), do not call `profile.track()`.
- Sparse / multi-bunch slicing (`sparse_profiles.py`) →
  `EquidistantMultiProfile`; see `EX_20` and `EX_28`.

---

## 7. Impedances and induced voltage

BLonD 2 mixed "what the impedance is" with "how the voltage is computed". BLonD
3 splits them: a `WakeField` element holds **sources** (physics) and one
**solver** (numerics).

```python
# BLonD 2
tiv = TotalInducedVoltage(beam, profile, [
    InducedVoltageFreq(beam, profile, [Resonators(R_S, f_r, Q)],
                       frequency_resolution=1e5),
])

# BLonD 3
wakefield = WakeField(
    sources=(Resonators(R_S, f_r, Q),),
    solver=PeriodicFreqSolver(t_periodicity=1/1e5),
    profile=profile,
)
ring.add_elements((..., wakefield))
```

Sources:

| BLonD 2 (`impedance_sources.py`) | BLonD 3 (`blond.physics.impedances.sources`) |
|---|---|
| `Resonators(R_S, frequency_R, Q)` | `Resonators(shunt_impedances, center_frequencies, quality_factors)` — same order |
| `InputTable(f, ReZ, ImZ)` | `ImpedanceTableFreq(freq_x, freq_y)`, or `ImpedanceTableFreq.from_file(path, reader)` with a reader from `blond.physics.impedances.readers` (`CsvReader`, `ExampleImpedanceReader1/2`) |
| `TravelingWaveCavity(...)` | `TravelingWaveCavity` |
| `ResistiveWall(...)` | no direct equivalent — flag |
| `CoherentSynchrotronRadiation(...)` | no direct equivalent — flag |
| `InductiveImpedance(beam, profile, Z_over_n, rf_station)` (a *voltage* class in legacy) | source `InductiveImpedance(Z_over_n)` + `InductiveImpedanceSolver()` |

Induced-voltage classes → solvers:

| BLonD 2 | BLonD 3 solver |
|---|---|
| `InducedVoltageTime(...)` | `TimeDomainFftSolver()` — for profiles short compared with `t_rev` |
| `InducedVoltageFreq(..., frequency_resolution=Δf)` | `PeriodicFreqSolver(t_periodicity=1/Δf)` |
| `InducedVoltageResonator(...)` | `blond.physics.impedances.solvers.SingleTurnResonatorConvolutionSolver()` / `blond.physics.impedances.solvers.MultiPassResonatorSolver(retune_to_rf=...)` |
| `multi_turn_wake=True`, `mtw_mode=...` | `blond.physics.impedances.solvers.ContinuousMultiTurnTimeDomainSolver(n_turns=...)` / `blond.physics.impedances.solvers.MultiPassResonatorSolver(retune_to_rf=...)`; see `EX_09_Multi_turn_wake.py` |
| `InductiveImpedance` | `InductiveImpedanceSolver()` |
| `TotalInducedVoltage([...])` combining several | either several sources in one `WakeField`, or several `WakeField` elements added to the ring (`EX_23` does both) |

!!! note "`MultiPassResonatorSolver` needs `retune_to_rf`"
    Pass `retune_to_rf` explicitly. It selects the *physics*, not a
    formatting detail, and it is the **only** thing that does:

    * `retune_to_rf=True` re-centres the resonator on the RF station's
      design frequency every pass, and enables the carried-wake phase
      clock. This is the accelerating **fundamental** mode.
    * `retune_to_rf=False` (the default) keeps the resonator at its
      constructed centre frequency. This is the fixed-frequency
      **higher-order-mode** case.

    `delta_f` is orthogonal to it: a pure frequency offset in [Hz],
    added to the design frequency of every pass when retuning, and to
    the constructed centre frequency once, at late init, when not. No
    spelling of `delta_f` switches retuning on, and a `delta_f` given
    *without* `retune_to_rf` is a perfectly ordinary configuration --
    a fixed-frequency resonator deliberately offset from nominal, e.g.
    a detuned higher-order mode.

    **Migration hazard: nothing raises.** `delta_f=0.0` used to mean
    "retune", and now means "no offset, and (at the default
    `retune_to_rf=False`) no retuning" -- the same call, the opposite
    physics, with no error and no warning. A ported BLonD 2 call that
    relied on the old sentinel encoding therefore runs a
    fixed-frequency resonator silently. State the mode explicitly.
    Neither argument takes `None`: `delta_f=None` dies in `float(None)`
    with a `TypeError`, while `retune_to_rf=None` is quietly coerced to
    `False` by `bool(None)` -- so it does *not* fail loudly either.


Note `frequency_resolution` [Hz] → `t_periodicity = 1 / frequency_resolution`
[s]; the quantity is inverted, not renamed. This inversion is the single most
consequential impedance conversion and the two codebases do not share a symbol
for it, so **treat it as a hypothesis and confirm it against the induced
voltage numerically** (see `verification.md`) rather than trusting the table.
Getting it wrong rescales the wake length and gives a plausible-looking but
wrong induced voltage.

`beam` and `profile` are no longer arguments of the sources or the solver. A
`WakeField` either gets its own `profile=`, or uses a profile element already
in the ring.

---

## 8. Losses

| BLonD 2 (called manually in the loop) | BLonD 3 (a ring element) |
|---|---|
| `beam.losses_longitudinal_cut(dt_min, dt_max)` | `BoxLosses(t_min=..., t_max=..., purge_flagged_macroparticles=...)` |
| `beam.losses_energy_cut(dE_min, dE_max)` | `BoxLosses(e_min=..., e_max=...)` |
| `beam.losses_below_energy(dE_min)` | `BoxLosses(e_min=...)` |
| `beam.losses_separatrix(ring, rf_station)` | **no element equivalent.** `blond/utilities/separatrix/` has the separatrix maths; a separatrix cut has to be done in a `callbacks=` function or a `UserDefinedElement`. Flag this — it changes the surviving intensity. |

`purge_flagged_macroparticles=True` actually removes lost particles from the
arrays; `False` only flags them. Legacy losses flagged (set `id=0`) and kept
the array size, so `False` is the closer match unless the user wants the speed
of a shrinking array.

---

## 9. Monitors, plots and output

The `monitors/` and `plots/` modules do not exist in BLonD 3. Output is now
**observations** passed to `run_simulation(observe=...)`, plus your own
matplotlib.

| BLonD 2 | BLonD 3 |
|---|---|
| `BunchMonitor(ring, rf, beam, filename, profile=...)` | `BeamStatisticsOncePerTurn(each_turn_i=1, folder=...)` (moments) and/or `BeamObservationOncePerTurn(each_turn_i=..., folder=...)` (full coordinates) |
| `SlicesMonitor(...)` | `StaticProfileObservation(...)` |
| `MultiBunchMonitor(...)` | no direct equivalent — compose observations, or flag |
| `Plot(...)` (live phase-space / separatrix plots) | no equivalent. Use `beam.plot_hist2d()` / `plot_scatter()` for snapshots, or a `callbacks=` function that draws each turn (`EX_06` shows the pattern). |
| `.h5` output files | observations write to `folder=`; `sim.load_results(...)` reads them back instead of rerunning |
| RF-phase diagnostics | `RFStationPhaseObservation(each_turn_i=1, rf_station=rf)` |
| 2-D histogram per turn | `BeamHist2dOncePerTurn` |
| impedance plots (`plot_impedance.py`) | `WakeFieldObservation`, then plot yourself |

Observations record on a stride: `each_turn_i=10` means every tenth turn. A
legacy monitor with `buffer_time=` was about I/O buffering, not stride — do not
translate it into `each_turn_i`.

Recorded buffers may be on the GPU — `copy_to_cpu(obs.dts)` before plotting.

---

## 10. Turn-by-turn programmes: `schedule()`

BLonD 2 accepted arrays wherever a scalar was allowed (voltage, phase, α), and
indexed them by the internal turn counter. BLonD 3 keeps constructor arguments
scalar and adds an explicit scheduling call:

```python
element.schedule("attribute_name", values_per_turn)          # ScheduledArray
element.schedule("attribute_name", (times, values))          # ScheduledInterpolation
```

`ScheduledArray` is indexed as `values[turn_i]`, with `turn_i` starting at 0.

```python
rf_station.schedule("voltage", voltage_programme)
rf_station.schedule("phi_rf_design", phi_programme[:-1, np.newaxis])   # MultiHarmonic: (n_turns, n_harmonics)
drift.schedule("momentum_compaction_factor", alpha_programme[1:])
```

**Array-length and offset care is required here.** A legacy programme of length
`n_turns + 1` cannot be passed through unchanged, and the correct slice is not
the same for every quantity: `tests/integration/blond2_regression/tracking/test_kickdrift.py`
schedules `phi_rf[:-1]` on the RF station but
`momentum_compaction_factor[1:]` on the drift — the kick uses the
start-of-turn value while the drift uses the end-of-turn value. Do not assume;
this is precisely what the side-by-side numerical check in
`verification.md` is for. If a one-turn offset appears in the comparison, this
is the first place to look.

For `MultiHarmonicRFStation`, scheduled per-harmonic quantities are 2-D:
shape `(n_turns, n_harmonics)`.

---

## 11. Backends

| BLonD 2 | BLonD 3 |
|---|---|
| `from blond.utils import bmath; bmath.use_cpu()` | `setup_backend("auto")`, or `backend.set_specials("numba"/"cpp"/"cuda"/"python")` |
| `bmath.use_gpu()` | `setup_backend("auto")` picks CUDA if available; `backend.change_backend(Cupy64Bit)` forces it |
| `np.` throughout user code | `backend.` where the array must land on the active device |

The legacy `bm` singleton is a **mutable global**. If you import BLonD 2 and
BLonD 3 in the same process (as the verification recipe does), pin the legacy
backend explicitly with `bmath.use_cpu()` — otherwise it may inherit whatever
an earlier import left active, including the very slow pure-Python path.

---

## 12. Synchrotron radiation

Legacy `SynchrotronRadiation` (one class, constructed with ring/rf/beam) →
`blond/physics/synchrotron_radiation/`, driven by `SynchrotronRadiationMaster`
and configured through the ring's `radiation_integrals`. See
`EX_11_Synchrotron_Radiation.py` and `EX_27_Synchrotron_Radiation_Matched.py`
(and `SynchrotronRadiationMatcher` in `blond/experimental/` for matching with
radiation).

---

## 13. No stable equivalent — flag these

Port everything else, then report these explicitly with the nearest
alternative. Do not silently drop them, and do not quietly substitute an
experimental class without saying so.

| BLonD 2 feature | Status in BLonD 3 |
|---|---|
| `llrf/beam_feedback.py` (`BeamFeedback`: phase loop, radial loop, SPS/PSB/LHC variants) | `blond/physics/feedbacks/` has only the base classes; the implementations live in `blond/experimental/physics/feedbacks/beam_feedback.py`. Unstable — offer it, with the warning. RF stations do accept `beam_feedback=` / `cavity_feedback=`. |
| `llrf/cavity_feedback.py` (`SPSCavityFeedback`, `LHCCavityLoop`, …) | as above; `tests/integration/test_sps_cavity_feedback/` shows how far it has been carried |
| `llrf/rf_noise.py`, `rf_modulation.py` | partially: `blond/cycles/noise_generators/`, `EX_13_RFnoise.py`, `blond/interfaces/rf_noise_cpp/`. Check coverage against the legacy script's use. |
| `monitors/`, `plots/` | replaced by observations + your own plotting (§9) |
| `utils/mpi_config.py`, `mpi_main_files/` | BLonD 3 distribution is different (`blond/generals/distributed/`, `blond/core/backends/mpi_distributed/`). Not a port — a redesign. Flag. |
| `toolbox/` (`tomoscope`, `parameter_scaling`, `filters_and_fitting`, `action`, `diffusion`) | mostly absent; some maths lives in `blond/acc_math/`. Check case by case. |
| `matched_from_distribution_function` / `matched_from_line_density` | see §5 |
| `losses_separatrix` | see §8 |
| `InducedVoltageAnalytical`, `music.py` (MuSiC tracking) | `analytical_gaussian_resonator` is still imported *from legacy* even by BLonD 3 examples (`EX_05`). MuSiC has no BLonD 3 element — flag. |
| `ResistiveWall`, `CoherentSynchrotronRadiation` impedance sources | no equivalent (§7) |
| `periodicity=True` tracking | no equivalent — flag |

Importing the legacy module for a genuinely missing piece
(`from blond.legacy.blond2... import ...`) is acceptable and is done by the
shipped examples — but say clearly that this part of the script is still
running on BLonD 2 code.

---

## 14. Trap checklist

Run through this before declaring a port done. Each of these produces code that
*runs*.

- [ ] **`section_index` is 0-based in BLonD 3, 1-based in BLonD 2.** Legacy
      `RFStation(..., section_index=1)` is the first section, and internally
      stores `section_index - 1`. Passing `1` to a BLonD 3 element puts it in
      the *second* section.
- [ ] **`n_slices` → `n_bins`.**
- [ ] **`Proton()` → `proton`** (no parentheses).
- [ ] **`phi_rf_d` / `phi_offset` → constructor `phi_rf=`, attribute
      `phi_rf_design`.** `rf.phi_rf` is a different quantity.
- [ ] **`n_macroparticles` belongs to `BiGaussian`, not `Beam`.**
- [ ] **Programme arrays are `n_turns + 1` long in BLonD 2.** Split into
      `value_init` + `values_after_turn`, and check the slice for scheduled
      quantities (§10).
- [ ] **`frequency_resolution` → `t_periodicity = 1 / frequency_resolution`.**
- [ ] **`n_rf=1` means scalars, not one-element lists.**
- [ ] **`prepare_beam` comes after `Simulation(...)`**, and anything reading
      `t_rev` / `omega_rf` / β / γ must also come after it.
- [ ] **Element order**: `reorder=False` to mirror the legacy `map_`; always
      read `sim.print_one_turn_execution_order()`.
- [ ] **`beam.dt` → `beam.read_partial_dt()`, wrapped in `copy_to_cpu` before
      any NumPy call.**
- [ ] **`callbacks=` is plural.**
- [ ] Units are unchanged between versions and are SI/accelerator standard:
      volts, eV, eV/c, seconds, metres, radians.
