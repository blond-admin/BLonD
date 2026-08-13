# Verifying a port numerically

A port that imports and runs proves almost nothing. Every trap in
`api_mapping.md` — a one-turn offset in the momentum programme, a 1-based
`section_index`, `frequency_resolution` not inverted into `t_periodicity`,
kick and drift swapped by `reorder=True` — produces a script that runs happily
and gives different physics. The only way to tell the difference is to run both
frameworks and compare numbers.

BLonD 2 is still importable alongside BLonD 3 in the same interpreter, as
`blond.legacy.blond2.*`, so this is cheap to do. The worked in-repo precedent is
`tests/integration/blond2_regression/tracking/test_kickdrift.py`.

## Strategy: start with one particle

Multi-particle runs cannot agree exactly — the two frameworks generate
distributions with different RNG streams, so seeding does not make them
identical. That noise floor hides the very offsets you are hunting for.

So compare in two stages:

1. **Single macroparticle, coordinates set by hand.** No RNG, so the
   trajectories should agree to near machine precision. This isolates the
   kick/drift/programme wiring — where nearly all porting bugs live. A
   discrepancy usually shows up within a handful of turns, so a short run
   (~100 turns) is enough and keeps the Python-level legacy loop cheap.
2. **Full beam, statistical comparison.** Only once stage 1 is clean. Compare
   moments (mean and std of `dt` and `dE`, bunch length, emittance) with
   `rtol`, not element-wise.

If the port includes intensity effects, do a third check on the induced voltage
itself for one profile — `tests/unittests/physics/impedances/compare_with_legacy/`
shows this pattern.

## Template

```python
import numpy as np

N_TURNS = 100
CIRCUMFERENCE = 2 * np.pi * 25
HARMONIC = 1
VOLTAGE = 8e3
PHI_RF = np.pi
GAMMA_T = 4.4
MOMENTUM = ...            # legacy programme, length N_TURNS + 1
INITIAL_T = np.array([0.4e-6])
INITIAL_E = np.array([25e6])

# ── BLonD 3 ────────────────────────────────────────────────────────────
from blond import (Beam, BeamObservationOncePerTurn, DriftSimple,
                   MagneticCyclePerTurn, Ring, Simulation,
                   SingleHarmonicRFStation, momentum_compaction_factor, proton)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu

ring = Ring(circumference=CIRCUMFERENCE)
magnetic_cycle = MagneticCyclePerTurn(
    reference_particle=proton,
    value_init=float(MOMENTUM[0]),
    values_after_turn=MOMENTUM[1:].copy(),
)
rf_station = SingleHarmonicRFStation(
    harmonic=HARMONIC, voltage=VOLTAGE, phi_rf=PHI_RF)
drift = DriftSimple(
    orbit_length=CIRCUMFERENCE,
    momentum_compaction_factor=momentum_compaction_factor(
        transition_gamma=GAMMA_T),
)
ring.add_elements((rf_station, drift), reorder=False, section_index=0)

beam3 = Beam(intensity=1, particle_type=proton)
beam3.setup_beam(dt=INITIAL_T.copy(), dE=INITIAL_E.copy())

sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
sim.print_one_turn_execution_order()     # confirm the order matches the legacy map_

observation = BeamObservationOncePerTurn(each_turn_i=1)
sim.run_simulation(beams=(beam3,), n_turns=N_TURNS, observe=(observation,))

dt_blond3 = copy_to_cpu(observation.dts)[:, 0]
dE_blond3 = copy_to_cpu(observation.dEs)[:, 0]

# ── BLonD 2 ────────────────────────────────────────────────────────────
from blond.legacy.blond2.beam.beam import Beam as Beam2, Proton
from blond.legacy.blond2.input_parameters.ring import Ring as Ring2
from blond.legacy.blond2.input_parameters.rf_parameters import RFStation
from blond.legacy.blond2.trackers.tracker import FullRingAndRF, RingAndRFTracker
from blond.legacy.blond2.utils import bmath

bmath.use_cpu()          # see "Pitfalls" — the legacy backend is a mutable global

ring2 = Ring2(CIRCUMFERENCE, 1 / GAMMA_T**2, MOMENTUM.copy(), Proton(),
              n_turns=N_TURNS)
beam2 = Beam2(ring2, len(INITIAL_T), 1,
              dt=INITIAL_T.copy(), dE=INITIAL_E.copy())
rf2 = RFStation(ring2, HARMONIC, VOLTAGE, PHI_RF)
tracker = FullRingAndRF([RingAndRFTracker(rf2, beam2, solver="simple")])

dt_blond2 = np.empty(N_TURNS + 1)
dE_blond2 = np.empty(N_TURNS + 1)
dt_blond2[0], dE_blond2[0] = beam2.dt[0], beam2.dE[0]
for turn in range(N_TURNS):
    tracker.track()
    dt_blond2[turn + 1] = beam2.dt[0]
    dE_blond2[turn + 1] = beam2.dE[0]

# ── Compare ────────────────────────────────────────────────────────────
np.testing.assert_allclose(dt_blond3, dt_blond2[1:], rtol=1e-8)
np.testing.assert_allclose(dE_blond3, dE_blond2[1:], rtol=1e-8)
```

## Pitfalls

**Turn alignment.** The two frameworks do not index history the same way. In
the template the BLonD 3 observation is compared against `dt_blond2[1:]`,
because the legacy history array here starts with the *pre-tracking* state.
Confirm the alignment on the first two or three turns before trusting a
`rtol` failure — an apparent 1e-3 disagreement that vanishes under a shift of
one index is an indexing artefact, not physics. Conversely, a real one-turn
offset in the *momentum programme* or a `schedule()` slice looks exactly like
this, so do not "fix" it by shifting the comparison until you know which it is.

**The legacy backend is a mutable global.** `blond.legacy.blond2.utils.bmath`
holds a module-level singleton. Without an explicit `bmath.use_cpu()` it can
inherit whatever an earlier import left active — in the worst case the pure
Python path, which makes the reference loop ~50× slower and can make you think
the comparison is infeasible.

**Cost scales with turns on the BLonD 2 side.** The legacy reference is a
Python-level per-turn loop. Slice all programmes consistently and run a short
comparison; a wiring bug diverges fast.

**GPU arrays.** BLonD 3 observation buffers may be CuPy. Always
`copy_to_cpu(...)` before comparing against the CPU-only legacy reference —
`np.asarray` on a device array does not do what you want.

**RNG.** Do not seed both and expect matching particles. Compare moments.

**Element order.** If stage 1 disagrees, print
`sim.print_one_turn_execution_order()` and compare it against the legacy `map_`
list before touching anything else. Add elements with `reorder=False` while
debugging so the order is exactly yours.

## Reporting the result

State plainly what was compared, over how many turns, and to what tolerance —
and if the comparison could not be run (missing input files, an unportable
feature, prohibitive runtime), say that instead of implying the port was
validated. "It runs without error" is not verification and should never be
presented as such.
