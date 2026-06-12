# Filling patterns — development status

Design decisions made 2026-06-12 (discussion with slauber); core rework
implemented. Tests for each implemented item are still to be written.

## Implemented (core rework)

- [x] Rename: top-level complete object is now `FillingPattern(segment,
      harmonic_number)` (was `Ring`, which collided with `blond.Ring`);
      composable base class is `PatternSegment` with `Gap`, `Bunch`,
      `Batch`, `Train` subclasses.
- [x] Named tiers (#319, #316): `_tiers: dict[str, np.ndarray]` replaces
      the fixed batch/train columns. `segment.label(name)` adds a tier;
      `+`/`*` re-number every tier independently; `tier(name)` /
      `n_in_tier(name)` accessors; `.batch`/`.train` remain as sugar.
      `Train(Train(...))` and duplicate `label()` raise instead of
      silently destroying inner structure.
- [x] Glossary + conventions (#316, #318): module docstring defines
      bucket/slot/bunch/batch/train/gap/abort gap; single spacing
      convention (integer spacings = empty buckets between; physical
      times = start-to-start seconds). No converter zoo.
- [x] Slots (#318): no core concept, by design — derive per bunch as
      `positions // buckets_per_slot` or store as a tier.
- [x] `as_n_buckets`: rounds to nearest bucket, warns above tolerance
      (default 0.05 buckets; LHC 25 ns ≈ 10.02 buckets passes silently).
- [x] Payload contract (consumer interface): guarded attributes —
      structural/tier name collisions raise; conventional payload names
      documented (`intensity`, `bunch_length`, `emittance`); NaN =
      unspecified.
- [x] Hygiene: matplotlib imports lazy (plotting split into `plot.py`,
      tier-generic coloring via `face_tier`/`edge_tier`); payload
      propagates through all constructors incl. `from_batch_list`;
      composition on a complete `FillingPattern` raises TypeError;
      constructor validates sorted/unique/in-range positions.

## Implemented (2026-06-12 refactor, architecture phase)

- [x] **Composition over LSP-violating inheritance**: extracted public
      read-only base `BunchTable` (positions/length/tiers/payload +
      payload attribute routing) = the consumer interface.
      `PatternSegment(BunchTable)` adds composition (`+`, `*`, `.gap()`,
      `.label()`); `FillingPattern(BunchTable)` adds `harmonic_number` /
      `has_bunch`. The five raising composition blockers on
      `FillingPattern` are gone — composing with a complete pattern now
      fails naturally (one isinstance guard in `PatternSegment.__add__`
      keeps a helpful message).
- [x] **API surface trimmed** (no consumers existed yet): removed `Bunch`
      (= `Batch(1, 0)` w/o tier), `__len__` (buckets-vs-bunches ambiguity
      trap), `bunch` ordinal property, the `batch`/`train`/`n_batches`/
      `n_trains` sugar (uniform access via `tier(name)`/`n_in_tier(name)`),
      and `FillingPattern.from_trains`/`from_spacing` (redundant with
      `FillingPattern(Train(...), h)` composition). `from_batch_list`
      renamed `from_placements`. Kept `Batch.from_spacing` /
      `Train.from_spacing` (they encode the seconds->buckets convention).
- [x] **Docstring diet within tooling rules**: private helpers and dunders
      carry comments instead of numpydoc blocks (ruff D1 skips private
      names; numpydoc GL08 is excluded; `D105` added to the ruff ignore
      list in pyproject.toml). Full numpydoc docs remain on the public
      API only. `_merge_tiers`/`_merge_payload` merged into one
      `_merge_columns`.
- [x] **File layout**: `helpers.py` (`as_n_buckets`) folded into
      `filling_patterns.py`; `__main__` demo moved to
      `blond/examples/scripts/EX_29_Filling_pattern.py`. Package is now
      `filling_patterns.py` + `plot.py`.

## Implemented (2026-06-12 hardening, after critical review)

- [x] Tier/payload name collisions now validated in `BunchTable.__init__`
      (the single chokepoint) — previously `+` could smuggle a name that
      was both tier and payload, leaving it permanently un-assignable.
      Payload names colliding with structural attributes are also caught
      at construction (not only via `__setattr__`).
- [x] Structure frozen after construction: `positions` and tier columns
      are read-only arrays (constructor copies its inputs, so caller
      arrays are not frozen); `tiers`/`payload` properties return
      snapshot dicts. Payload arrays stay live/mutable in place (the
      masked-assignment interface) and are copied on assignment.
- [x] Integer inputs validated via `_as_int` (integral floats like 5.0
      accepted, fractional values raise instead of silently truncating):
      `Gap`, `Batch`, `Train`, `FillingPattern.harmonic_number` (also
      must be >= 1), `from_placements` start buckets (must be >= 0).
- [x] `from_spacing` with a distance shorter than the unit now raises an
      error naming `start_to_start_distance` and the unit length, instead
      of leaking an internal `n_empty_buckets`/`bunch_spacing` message.
- [x] `Train` rejects negative `copy_spacing` even for `n_copies=1`.
- [x] `n_in_tier` counts distinct assigned indices (was max+1, which
      disagreed with its docstring for hand-built non-contiguous tiers).
- [x] `as_n_buckets` gained a `stacklevel` parameter; `from_spacing`
      warnings now point at the user's call site.
- [x] `from_placements` overlap error explains that ranges include
      trailing gaps.

## Implemented (2026-06-12 vocabulary/naming pass)

Names now carry the module's conventions ("gap" = integer empty buckets,
"spacing" = physical start-to-start seconds). Applied while there are no
external consumers; sections above use the old names where they describe
historical states.

- [x] `bunch_spacing` -> `bunch_gap`, `copy_spacing` -> `copy_gap`
      ("spacing" was actively misleading: in accelerator jargon "25 ns
      bunch spacing" is start-to-start, but the integer parameter counts
      empty buckets — LHC 25 ns is `bunch_gap=9`, not 10). `from_spacing`
      constructors keep their name; "spacing" now exclusively means
      physical start-to-start.
- [x] `length` -> `n_buckets` (parallel to `n_bunches`; says what it
      counts, including the trailing gap).
- [x] `positions` -> `bucket_indices` (the unit is now in the name;
      slot recipe reads `bucket_indices // buckets_per_slot`).
- [x] `n_in_tier` -> `n_groups` (it counts groups, e.g. batches — the old
      name suggested counting bunches).
- [x] `payload` -> per-bunch "properties" (`.properties` accessor;
      "payload" was transport jargon revealing nothing of the physics).
- [x] `as_n_buckets` -> `n_buckets_from_time` (reads as the unit
      conversion it is).

## Next: tests

- [ ] Write tests per implemented item above (composition/renumbering,
      label/nesting errors, payload guards + NaN merge, from_placements,
      as_n_buckets tolerance, completeness errors, has_bunch).
      Partially done: `tests/unittests/cycles/filling_patterns/` covers
      the 2026-06-12 hardening items plus regression guards (tier
      renumbering, NaN merge, pickle, has_bunch, from_placements).

## Deferred follow-up packages (decisions logged)

- [ ] **Consumers** — pattern stays a passive data contract for now.
      Future consumers, in cost order: clone-and-place beam population,
      per-bunch generation, self-consistent matching, multi-turn
      injection between rings (needs an 'injection' tier + turn mapping;
      the tier mechanism already supports it).
      (https://gitlab.cern.ch/blond/BLonD/-/work_items/294)
- [ ] **LPC importer + machine presets** in `blond.specifics.cern`:
      import actual LPC scheme files (not name parsing); slot-based input
      helpers live there too.
      (https://gitlab.cern.ch/blond/BLonD/-/work_items/294, /318)
- [ ] **Sparse profile init** (#317): add
      `EquidistantMultiProfile.from_filling_pattern(...)` classmethod in
      `blond/physics/profiles_sparse.py`; `has_bunch` / positions are the
      input. Benchmark in-advance vs on-the-fly creation first.
      (https://gitlab.cern.ch/blond/BLonD/-/work_items/317)
- [ ] **Random patterns** (#315): deferred entirely — ask the thread
      author for the driving use case before building anything.
      Unconstrained random fills are unphysical; likely real needs are
      random per-bunch intensity (already works via payload) and random
      bunch dropout.
      (https://gitlab.cern.ch/blond/BLonD/-/work_items/315)
