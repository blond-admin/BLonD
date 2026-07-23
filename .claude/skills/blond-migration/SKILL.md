---
name: blond-migration
description: >
  Port BLonD 2 (legacy) simulation scripts to the BLonD 3 API. Use this skill
  whenever the user wants to migrate, port, convert, update or "get running
  again" an old BLonD script, or asks "what is the BLonD 3 equivalent of X".
  Trigger on any BLonD 2 symbol appearing in the user's code or question:
  RingAndRFTracker, FullRingAndRF, RFStation(ring, ...), Ring(C, alpha,
  momentum, Proton(), n_turns), Beam(ring, n_macroparticles, intensity),
  bigaussian(), matched_from_distribution_function, CutOptions, FitOptions,
  Profile(beam, ...), TotalInducedVoltage, InducedVoltageFreq,
  InducedVoltageTime, InputTable, BunchMonitor, SlicesMonitor, Plot,
  BeamFeedback, CavityFeedback, blond.utils.bmath, or an explicit
  `from blond.legacy.blond2...` / old-style `from blond.input_parameters...`
  import. Also trigger when a user's BLonD script fails with ImportError or
  TypeError after upgrading, when they mention "my old main file", "BLonD2",
  "blond 2", "the old API", or "legacy BLonD" — even if they never say the
  word "migrate".
---

# Porting BLonD 2 scripts to BLonD 3

BLonD 3 is a **redesign, not a rename**. Almost nothing ports by search and
replace: objects were split apart, ownership of parameters moved, and the
per-turn loop is no longer written by the user. Porting line by line produces
code that either does not run or — much worse — runs and gives quietly wrong
physics.

So the job is: **recover the physics the old script describes, then re-express
that physics in BLonD 3, then prove numerically that the two agree.** The last
step is not optional; see "Verify the port" below.

## The one structural change to understand first

In BLonD 2, `Ring` is a god-object. It holds the circumference, the momentum
compaction, the *entire momentum programme*, the particle type and `n_turns`,
and every other object takes `ring` (and often `beam` and `profile`) as a
constructor argument. The user then hand-builds a `map_` list and writes the
turn loop.

In BLonD 3 those responsibilities are separated, and `Simulation` wires them
together *late*, after construction:

| Concern | BLonD 2 | BLonD 3 |
|---|---|---|
| Geometry + which elements a turn traverses | `Ring(...)` + hand-written `map_` | `Ring(circumference)` + `ring.add_elements([...])` |
| Energy programme | `Ring(..., synchronous_data, n_turns)` | `ConstantMagneticCycle` / `MagneticCyclePerTurn` / `MagneticCycleByTime` |
| Optics / drift | folded into `Ring`'s `alpha_0` | `DriftSimple` — an explicit ring **element** |
| RF kick | `RFStation` + `RingAndRFTracker` | `SingleHarmonicRFStation` / `MultiHarmonicRFStation` (one object, it *is* the element) |
| Macroparticles | `Beam(ring, n_macroparticles, intensity)` | `Beam(intensity, particle_type)`; the count belongs to the **preparation routine** |
| Beam generation | `bigaussian(ring, rf, beam, ...)` | `sim.prepare_beam(beam=..., preparation_routine=BiGaussian(...))` |
| Turn loop | `for i in range(N_t): for m in map_: m.track()` | `sim.run_simulation(beams=..., n_turns=...)` |
| Output | `BunchMonitor`, `SlicesMonitor`, `Plot` | Observations passed to `run_simulation(observe=...)` |

Two consequences that catch people out:

- **Derived quantities do not exist until `Simulation` is built.** `t_rev`,
  `omega_rf`, `beta`, `gamma`, the separatrix — in BLonD 2 these were available
  on `ring`/`rf_station` immediately. In BLonD 3 they are resolved when
  `Simulation(...)` runs its late-init. Any legacy line that read
  `ring.t_rev[0]` or `rf.omega_rf[0,0]` to compute a *constructor argument*
  has to be reordered, or replaced with the corresponding query on the
  magnetic cycle (e.g. `magnetic_cycle.get_t_rev_init(circumference,
  particle_type=proton)`, as `EX_05_Wake_impedance.py` does).
- **Element order is explicit.** The old `map_` list *was* the turn order. Use
  `ring.add_elements((...), reorder=False)` to preserve the legacy order
  literally, or `reorder=True` to let BLonD sort by physics. Either way, run
  `sim.print_one_turn_execution_order()` and read it — a reordered kick and
  drift changes the result at the 1/turn level and is exactly the kind of bug
  the numerical check below is there to catch.

## Workflow

### 1. Inventory the legacy script — physics, not syntax

Read the whole old script and write down what it actually simulates:
machine circumference, γ_tr / α, energy programme (constant? ramp? by turn or
by time?), RF systems (how many harmonics, V, φ), number of turns, beam
intensity and macroparticle count, how the beam is generated, profile cuts and
bin count, impedance sources and which induced-voltage class, losses, what is
monitored/plotted, feedbacks, and the exact `map_` order.

Do this *before* writing any BLonD 3 code. The inventory is the specification;
the old source is only evidence for it.

### 2. Find the closest BLonD 3 example and start from it

Many legacy examples have a direct BLonD 3 counterpart. Reading the pair side
by side is the fastest way to see the idiom. The mapping is in
`references/example_pairs.md` — consult it early.

### 3. Translate object by object

`references/api_mapping.md` is the full BLonD 2 → BLonD 3 table, with the
argument-level differences and the traps (0- vs 1-based `section_index`,
`n_slices` → `n_bins`, `phi_rf_d` → `phi_rf` / `phi_rf_design`, programme array
lengths, `Proton()` → `proton`, …). Read it whenever you touch a class you have
not already ported in this session.

Write the port in the BLonD 3 example style: a `def main():` containing the
build-up, and `if __name__ == "__main__":`. That is not cosmetic —
`Simulation.from_locals(locals())` is a genuinely convenient way to port a flat
legacy script (it picks up every element defined so far), and it only reads the
enclosing function's locals.

### 4. Verify the port numerically — do not skip this

BLonD 2 is still importable in the same interpreter as
`blond.legacy.blond2.*`, so the old and new scripts can be run back to back and
compared. This is the only way to distinguish "it runs" from "it is right".
`references/verification.md` has the recipe, the pitfalls (turn alignment,
legacy's mutable global backend, RNG differences) and a template. The in-repo
precedent is `tests/integration/blond2_regression/tracking/test_kickdrift.py`.

If the comparison cannot be run (missing input files, a feature with no BLonD 3
equivalent, prohibitive runtime), say so explicitly rather than declaring the
port correct.

### 5. Report what did not come across

Some BLonD 2 features have no stable BLonD 3 equivalent yet — beam/cavity
feedback, the monitor and plot modules, MPI, several matched-distribution
generators. **Port everything portable, then list what was dropped, why, and
the nearest alternative.** Never silently drop physics: an omitted phase loop
turns a controlled ramp into a different simulation, and the user must be the
one to decide whether that is acceptable. `references/api_mapping.md` marks
these cases; the experimental replacements live in `blond/experimental/` and
should only be suggested with an explicit warning that they are unstable.

## Stay skeptical

BLonD 3 is under active development and the legacy code is old. If a legacy
script does something that looks physically wrong, or if a BLonD 3 API behaves
differently from what its docstring claims, **say so** — flag it to the user
rather than silently reproducing the oddity or "fixing" it to match your
expectation. A migration is a good moment to surface a latent bug, and a bad
moment to paper over one.

Two specific things worth naming when you meet them:

- **`phi_s` is not "the stable fixed point".** It is the single-harmonic
  synchronous phase. Legacy code that used `phi_s` to locate a bucket centre
  under multiple harmonics was already wrong; do not carry the assumption over.
- **Do not police `phi_rf`.** A phase offset between the old and new script
  shifts the stable fixed point but leaves the reference particle untouched —
  it is a legitimate modelling choice, not an error to warn about.

## Related skills

- `blond-assistant` — the BLonD 3 API itself
  (`.agents/skills/blond-assistant/references/api_reference.md` is a full class
  reference). Consult it for anything about the *target* API; this skill covers
  only the translation.
- `blond-dev` — working on the BLonD framework itself (tests, backends,
  pre-commit, docs).

## Reference files

- `references/api_mapping.md` — BLonD 2 → BLonD 3, class by class and argument
  by argument, including what has no equivalent.
- `references/example_pairs.md` — which `legacy/__EXAMPLES/main_files/` script
  corresponds to which `blond/examples/scripts/EX_*.py`.
- `references/verification.md` — how to run both versions side by side and
  compare, with a template and the known pitfalls.
