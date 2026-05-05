# BLonD API Reference

## Table of Contents
1. [Ring](#ring)
2. [RF Stations](#rf-stations)
3. [Drift](#drift)
4. [Magnetic Cycles](#magnetic-cycles)
5. [Beam & Particles](#beam--particles)
6. [Simulation](#simulation)
7. [Beam Preparation](#beam-preparation)
8. [Observations](#observations)
9. [Intensity Effects](#intensity-effects)
10. [Backends](#backends)
11. [Utility helpers](#utility-helpers)

---

## Ring

```python
from blond import Ring
ring = Ring(circumference: float)
```

- `circumference` — machine circumference in metres
- `ring.add_elements(elements, reorder=True)` — add a list/tuple of elements
  - `reorder=True`: BLonD sorts elements into the canonical order (drifts → RF → wakefields → profiles)
  - `reorder=False`: use your explicit ordering (required for multi-section machines)
- `ring.circumference` — read back circumference

---

## RF Stations

### SingleHarmonicRFStation

```python
from blond import SingleHarmonicRFStation
rf = SingleHarmonicRFStation(
    harmonic=None,        # harmonic number h (integer)
    voltage=None,         # peak RF voltage [V]
    phi_rf=None,          # RF phase offset [rad]
    section_index=0,      # section index for multi-section rings
)
```

Attributes (can also be set after construction):
- `rf.harmonic` — harmonic number
- `rf.voltage` — RF voltage [V]
- `rf.phi_rf_design` — design/target RF phase [rad]
- `rf.phi_rf` — actual instantaneous phase [rad] (read/write)
- `rf.schedule(attribute, value)` — schedule a per-turn array for any attribute
  - e.g. `rf.schedule("phi_rf_design", noise_array)` for RF noise

### MultiHarmonicRFStation

```python
from blond import MultiHarmonicRFStation
import numpy as np
rf = MultiHarmonicRFStation(
    voltage=np.array([V1, V2, ...]),       # voltages per harmonic [V]
    phi_rf=np.array([phi1, phi2, ...]),    # phases per harmonic [rad]
    harmonic=np.array([h1, h2, ...]),      # harmonic numbers
    n_harmonics=N,                          # number of harmonics
    main_harmonic_idx=0,                    # index of the fundamental harmonic
    section_index=0,
)
```

---

## Drift

```python
from blond import DriftSimple, momentum_compaction_factor
drift = DriftSimple(
    orbit_length: float,                    # length of drift section [m]
    momentum_compaction_factor=None,        # α (float or result of helper)
    section_index=0,
)
# or set after construction:
drift.momentum_compaction_factor = momentum_compaction_factor(transition_gamma=55.76)
```

### momentum_compaction_factor helper

```python
from blond import momentum_compaction_factor
alpha = momentum_compaction_factor(transition_gamma=γ_tr)
# Returns α = 1/γ_tr²  (non-relativistic approximation for large γ_tr)
```

### ReferenceEnergyChange

For multi-section ramps where the reference energy changes mid-turn:
```python
from blond import ReferenceEnergyChange
rec = ReferenceEnergyChange(section_index=i)
```

---

## Magnetic Cycles

All cycles accept `in_unit` with options:
- `"momentum"` (default) — synchronous momentum [eV/c]
- `"total energy"` — total energy [eV]
- `"kinetic energy"` — kinetic energy [eV]
- `"bending field"` — magnetic rigidity [T·m]

### ConstantMagneticCycle

No acceleration — flat energy throughout simulation.
```python
from blond import ConstantMagneticCycle, proton
cycle = ConstantMagneticCycle(
    value: float,               # energy/momentum value
    reference_particle=proton,
    in_unit="momentum",         # unit of value
)
```

### MagneticCyclePerTurn

Turn-by-turn energy ramp (most common for acceleration).
```python
from blond import MagneticCyclePerTurn
import numpy as np
cycle = MagneticCyclePerTurn(
    value_init: float,                    # energy at turn 0
    values_after_turn: np.ndarray,        # energy after each turn, shape (N_TURNS,)
    reference_particle=proton,
    in_unit="momentum",
)
```
Note: `len(values_after_turn)` must equal `N_TURNS` passed to `run_simulation`.

### MagneticCycleByTime

Energy ramp defined by interpolation over continuous time.
```python
from blond import MagneticCycleByTime
import numpy as np
cycle = MagneticCycleByTime(
    reference_particle=proton,
    base_time: np.ndarray,       # time points [s]
    base_values: np.ndarray,     # energy values at each time point
    in_unit="momentum",
    interpolator=np.interp,      # interpolation function (default: np.interp)
)
```

### MagneticCyclePerTurnAllRFStations

Full control over each RF station's energy contribution per turn.
```python
from blond.cycles.magnetic_cycle import MagneticCyclePerTurnAllRFStations
cycle = MagneticCyclePerTurnAllRFStations(
    value_init: float,
    values_after_rf_station_per_turn: np.ndarray,  # shape (n_rf_stations, N_TURNS)
    reference_particle=proton,
    in_unit="momentum",
)
```

---

## Beam & Particles

### Beam

```python
from blond import Beam, proton
beam = Beam(
    intensity: float,          # number of real particles (not macroparticles)
    particle_type=proton,
)
```

Key methods/attributes:
- `beam.read_partial_dt()` — view of time coordinates [s] (read-only reference)
- `beam.write_partial_dt()` — writable view of time coordinates [s]
- `beam.read_partial_dE()` — view of energy deviation coordinates [eV]
- `beam.write_partial_dE()` — writable view of energy deviation coordinates [eV]
- `beam.intensity` — total real particle count
- `beam.plot_hist2d()` — quick 2D phase space histogram (requires matplotlib)

For multi-bunch:
```python
from blond import make_multibunch_beam
beam = make_multibunch_beam(n_bunches=4, intensity_per_bunch=1e9, particle_type=proton)
```

### Particle types

```python
from blond import proton, electron, positron, mu_plus, mu_minus, uranium_29
```

### EmptyBeam

Beam with no particles (useful for testing machine setup):
```python
from blond import EmptyBeam
beam = EmptyBeam(particle_type=proton)
```

---

## Simulation

### Construction

```python
from blond import Simulation

# Explicit (recommended):
sim = Simulation(ring=ring, magnetic_cycle=cycle)

# Auto-discover from local variables (convenient for scripts):
sim = Simulation.from_locals(locals())
# WARNING: all components must be defined BEFORE this call
```

### Key methods

```python
sim.print_one_turn_execution_order()    # print element execution order (debugging)

sim.prepare_beam(
    beam=beam,
    preparation_routine=BiGaussian(...),
)

sim.run_simulation(
    beams=(beam,),                # tuple of beams
    n_turns=int(1e4),
    observe=(obs1, obs2, ...),   # optional observation objects
    callback=my_function,        # optional: called each turn with (sim, beam)
)

# Load previously saved results instead of rerunning:
sim.load_results(
    beams=(beam,),
    n_turns=N_TURNS,
    observe=(obs1, obs2, ...),
)
```

---

## Beam Preparation

### BiGaussian (standard, most common)

```python
from blond import BiGaussian
routine = BiGaussian(
    sigma_dt: float,            # time spread σ_t [s]
    sigma_dE=None,              # energy spread σ_E [eV] (optional, matched if None)
    n_macroparticles=1e6,       # number of macroparticles (can be float)
    seed=1,                     # random seed for reproducibility
    reinsertion=False,          # re-insert lost particles back into bucket
)
```

### SemiEmpiricMatcher (experimental, physics-based)

```python
from blond.experimental import SemiEmpiricMatcher
routine = SemiEmpiricMatcher(
    n_macroparticles=1e6,
    seed=0,
    hamilton_max=1e-6,          # Hamiltonian cutoff
    density_modifier=1.0,       # density profile sharpness
    time_limit=(t_min, t_max),  # phase space time window [s]
    maxiter_intensity_effects=0,
)
```

### XsuiteRFBucketMatcher (requires xpart)

```python
from blond.interfaces.xsuite import XsuiteRFBucketMatcher
from xpart.longitudinal.rfbucket_matching import QGaussianDistribution
routine = XsuiteRFBucketMatcher(
    distribution_type=QGaussianDistribution,
    sigma_z=2.5e-9 / 4,        # bunch length [m or s depending on version]
    n_macroparticles=int(1e3),
    seed=42,
)
```

---

## Observations

All observations share:
- `each_turn_i: int` — record every Nth turn (1 = every turn)
- `folder: str` (optional) — path prefix for saving/loading data

### BeamObservationOncePerTurn

Records full phase space (dt, dE) of all macroparticles once per turn.
```python
from blond import BeamObservationOncePerTurn
obs = BeamObservationOncePerTurn(each_turn_i=1)
# After simulation:
obs.dts     # shape (n_recorded_turns, n_macroparticles) [s]
obs.dEs     # shape (n_recorded_turns, n_macroparticles) [eV]
obs.flags   # particle status flags
```

Alias: `BeamObservationEndOfTurn` (same class, different import name).

### RFStationPhaseObservation

Tracks RF station parameters turn by turn.
```python
from blond import RFStationPhaseObservation
obs = RFStationPhaseObservation(each_turn_i=1, rf_station=rf)
# After simulation:
obs.phases    # RF phase [rad], shape (n_recorded_turns, n_harmonics)
obs.omegas    # angular frequency [rad/s]
obs.voltages  # RF voltage [V]
```

### StaticProfileObservation

```python
from blond import StaticProfileObservation
obs = StaticProfileObservation(each_turn_i=1, profile=profile, obs_per_turn=1)
obs.hist_y   # shape (n_observations, n_bins)
```

### BeamObservationInRingElement

```python
from blond import BeamObservationInRingElement
obs = BeamObservationInRingElement(each_turn_i=1)
```

### SimulationObservation / DriftObservation

```python
from blond import SimulationObservation, DriftObservation
```

### Custom observations

Subclass `ObservablesEndOfTurnBase` from `blond.handle_results.observables`
and implement `on_run_simulation()` and `update()`.

---

## Intensity Effects

Intensity effects (collective effects / wake fields) require:
1. A **Profile** (beam profile / line density)
2. A **WakeField** with sources and a solver

### Profiles

```python
from blond import StaticProfile
profile = StaticProfile(
    cut_left: float,    # left edge of profile window [s]
    cut_right: float,   # right edge [s]
    n_bins: int,        # number of bins
    section_index=0,
)
# Convenience: from_rad() creates profile in rad units
profile = StaticProfile.from_rad(phi_left, phi_right, n_bins, t_rev)
```

```python
from blond import DynamicProfileConstNBins
profile = DynamicProfileConstNBins(
    n_bins: int,        # number of bins (edges adapt to beam extent each turn)
    section_index=0,
)
```

Profile attributes:
- `profile.hist_x` — bin centres [s]
- `profile.hist_y` — bin amplitudes (line charge density)

### WakeField

```python
from blond import WakeField
wakefield = WakeField(
    sources: tuple,     # tuple of impedance/wake sources (see below)
    solver,             # wake field solver (see below)
    profile=None,       # if None, uses the Profile already in the ring
)
```

**Sources:**

```python
from blond import Resonators
wake = Resonators(
    R_shunt,    # shunt impedance [Ω], array or scalar
    f_res,      # resonant frequency [Hz], array or scalar
    Q_factor,   # quality factor, array or scalar
)
```

```python
from blond import InductiveImpedance
wake = InductiveImpedance(Z_over_n: float)  # broadband inductive impedance [Ω]
```

```python
from blond import ImpedanceTableFreq
wake = ImpedanceTableFreq.from_file(filepath, reader)
# reader is a class that parses the file format (see blond.physics.impedances.readers)
```

**Solvers:**

```python
from blond import PeriodicFreqSolver
solver = PeriodicFreqSolver(t_periodicity=1/f_rev)   # frequency-domain, periodic
# t_periodicity=None for non-periodic

from blond import TimeDomainFftSolver
solver = TimeDomainFftSolver()   # time-domain FFT solver

from blond import InductiveImpedanceSolver
solver = InductiveImpedanceSolver()   # fast solver for pure inductive impedance
```

**Accessing results:**
```python
wakefield.induced_voltage   # induced voltage array [V]
```

### WakeFieldObservation

```python
from blond.handle_results.observables import WakeFieldObservation
obs = WakeFieldObservation(each_turn_i=1, wakefield=wakefield, obs_per_turn=1)
obs.induced_voltage   # shape (n_observations, n_bins)
```

---

## Backends

```python
from blond import backend

backend.set_specials("cpp")    # compiled C++ (compile first: blond-compile-cpp)
backend.set_specials("cuda")   # GPU via CUDA (compile first: blond-compile-cuda)
backend.set_specials("numpy")  # pure NumPy (default, no compilation needed)

backend.is_gpu   # True if GPU backend active
```

Backend classes (for type hints / advanced use):
```python
from blond import Numpy32Bit, Numpy64Bit, Cupy32Bit, Cupy64Bit
```

---

## Utility helpers

### callers_relative_path

Resolves file paths relative to the calling script (useful for example scripts):
```python
from blond.handle_results.helpers import callers_relative_path
path = callers_relative_path("resources/my_file.txt", stacklevel=1)
```

### AllowPlotting

Context manager to enable plotting (BLonD suppresses matplotlib by default in
non-interactive mode):
```python
from blond import AllowPlotting
with AllowPlotting():
    plt.plot(...)
    plt.show()
```

### VariNoise (experimental)

Generate phase noise schedules for RF stations:
```python
from blond.experimental import VariNoise
noise_array = VariNoise().get_noise(n_turns=200)
rf.schedule("phi_rf_design", noise_array)
```

---

## Examples quick index

| File | What it shows |
|------|---------------|
| `minimum_working_example.py` | Simplest possible simulation (LHC, proton, stationary) |
| `EX_01_Acceleration.py` | Basic acceleration with observations and caching |
| `EX_01_Acceleration_no_beam.py` | Machine setup without beam |
| `EX_02_Main_long_ps_booster.py` | PS Booster with impedance tables and WakeField |
| `EX_03_RFnoise.py` | RF phase noise, `VariNoise`, `DynamicProfileConstNBins` |
| `EX_04_Stationary_multistation.py` | Multi-RF-station ring, two DriftSimple + two RF stations |
| `EX_05_Wake_impedance.py` | SPS with resonators, time-domain and frequency-domain solvers |
| `EX_07_Xsuite_Matching.py` | XSuite RF bucket matching for beam preparation |
| `EX_08_MuCol_asynchronous_ramp.py` | Muon collider, `MagneticCycleByTime`, `mu_plus`, `ReferenceEnergyChange` |
| `main_user.py` | Object-oriented helper class pattern, multi-harmonic, `WakeField` |
| `custom_trackable.py` | How to define a `UserDefinedElement` custom tracking element |
