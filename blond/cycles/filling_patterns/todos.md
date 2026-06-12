# Filling patterns — development todos

Concrete, individually testable work items derived from the GitLab threads.
Each checkbox is one deliverable; tests will be written per item.

## 1. Naming convention & glossary (#316)

https://gitlab.cern.ch/blond/BLonD/-/work_items/316

- [ ] Write a glossary in the module docstring defining **bucket, slot, bunch,
      batch, train, fill/ring, gap, abort gap** as used by this module.
- [ ] Add a terminology mapping table (BLonD term ↔ common LHC/SPS/PSB usage,
      e.g. "PS batch" / "SPS train" / "injection") to the docs.
- [ ] Audit public API names (`Batch`, `Train`, `unit`, `copy_spacing`,
      `bunch_spacing`) against the agreed glossary and rename where they
      conflict; keep deprecated aliases if needed.

## 2. Spacing and bunch/slot definitions (#318)

https://gitlab.cern.ch/blond/BLonD/-/work_items/318

- [x] Fix `helpers.py`: `as_n_buckets` uses `math.floor` without
      `import math` (NameError on first call).
      *Done — rewritten as `round(time_distance * f_rf)`.*
- [ ] Decide and document the canonical spacing convention
      (start-to-start vs. last-bunch-to-first-bunch gap) for every parameter
      taking a spacing; state it in each docstring.
      *Partial — `bunch_spacing`/`copy_spacing` docstrings say "empty buckets
      between", and `from_spacing` constructors say start-to-start; no single
      stated module-wide convention yet.*
- [ ] Add conversion helpers: `gap_to_start_to_start(gap, unit_length)` and
      inverse, so users can express spacings either way.
      *Partial — private `_spacing_from_distance` covers physical
      start-to-start → gap; no public helpers.*
- [ ] Add slot↔bucket conversion: `slots_to_buckets(n_slots, buckets_per_slot)`
      and `buckets_to_slots(...)`; LHC default `buckets_per_slot=10`.
- [ ] Make `as_n_buckets` raise/warn when `time_distance` is not an integer
      multiple of `1/f_rf` within tolerance.
      *Rounding to nearest is done; the tolerance check is not.*
- [ ] Support slot-based construction, e.g.
      `Batch.from_slots(...)` / `Ring.from_slots(...)` for LHC-style input.

## 3. Nesting of trains (#319)

https://gitlab.cern.ch/blond/BLonD/-/work_items/319

- [ ] Preserve inner train indices when nesting: `Train(unit=some_train, ...)`
      currently overwrites all train indices to 0, destroying the
      sub-structure of the inner pattern.
- [ ] Generalize the fixed `batch`/`train` tiers to arbitrary named tier
      levels (e.g. `injection`, `sps_train`) so an LHC fill can contain
      labeled SPS patterns: `ring.injection == 3` masking, analogous to
      `ring.batch == 2`.
- [ ] Keep `+` and `*` re-numbering semantics working for every tier level,
      including user-defined ones.
      *Partial — works for the fixed `batch`/`train` tiers today.*
- [ ] Add an explicit `label(pattern, tier_name)` operation (or `Tier`
      wrapper) that stamps a new tier index onto an existing pattern without
      modifying lower tiers.
- [ ] Demo/example: build a realistic LHC fill out of an SPS train built out
      of PS batches, with all three tiers queryable.

## 4. Random filling pattern (#315)

https://gitlab.cern.ch/blond/BLonD/-/work_items/315

- [ ] Implement `random_pattern(harmonic_number, n_bunches, seed)` returning a
      `Ring` with `n_bunches` uniformly random distinct buckets.
- [ ] Support constraints: minimum bunch spacing, forbidden region (e.g.
      reserved abort gap), and fill fraction instead of bunch count.
- [ ] Reproducibility: identical output for identical `seed`
      (use `np.random.default_rng`).
- [ ] Raise a clear error when constraints are unsatisfiable
      (too many bunches for the spacing/region given).

## 5. High-level pattern presets (#294)

https://gitlab.cern.ch/blond/BLonD/-/work_items/294

- [ ] Collect the most-used filling patterns (LHC 25 ns standard, 8b4e, SPS,
      PSB single/double batch) and record them as target cases in the docs.
- [ ] Implement preset factory functions for at least two machines under
      `blond.specifics.cern` (e.g. `lhc_standard_25ns(n_bunches_per_injection, ...)`)
      built on `Batch`/`Train`/`Ring`.
- [ ] Parser for LPC-style filling scheme names
      (e.g. `25ns_2748b_2736_2258_2374_288bpi_13inj`) → `Ring`, or — if the
      thread prefers — an importer for LPC scheme files.
- [ ] Round-trip check: preset → `Ring.has_bunch` matches published bunch
      counts for the chosen reference schemes.

## 6. Sparse profile integration (#317)

https://gitlab.cern.ch/blond/BLonD/-/work_items/317

- [ ] Add an export from `Ring` to a sparse-profile specification: bin edges /
      cut windows covering only filled buckets
      (e.g. `ring.filled_bucket_windows(f_rf, margin_buckets=0)`).
      *Building block exists: `Ring.has_bunch` gives the boolean occupancy
      array over all harmonic_number buckets.*
- [ ] Wire that export into the profile machinery so a sparse profile can be
      constructed **in advance** from a `Ring` (the BLonD3-preferred path per
      the thread).
- [ ] Benchmark: in-advance vs. on-the-fly sparse profile creation in BLonD3
      ("should be checked" per the thread); record the result in the work
      item and pick the default accordingly.

## Module hygiene (found while reviewing, not from threads)

- [ ] Propagate payload through `Ring.from_batch_list` (currently dropped —
      no payload argument is forwarded).
- [ ] Move the top-level `matplotlib` import (palettes at import time) into
      the plot functions so the module works headless without matplotlib.
- [ ] Decide whether `__add__` on a `Ring` should be allowed (currently
      returns a plain `FillingPattern`, silently discarding
      `harmonic_number`).
