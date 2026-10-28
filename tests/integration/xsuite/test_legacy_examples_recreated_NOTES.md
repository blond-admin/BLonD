There is no bug in the new interface — but Birk's difference is real, and we can now say exactly
what it is. Three things were tangled together in the legacy interface, and keeping them apart is
the whole story:

- **Coordinate systems** (pure bookkeeping): BLonD pins `dt = 0` to the RF clock zero; xsuite pins
  `zeta = 0` to the reference particle. The transform between the frames is `zeta = −dt·β₀c` —
  no physics in it, and frequency-agnostic: with RF detuned off design the *bucket* drifts, but
  identically in both frames, so the bridge stays the identity. What differs there is the RF
  *model*: a BLonD station with `delta_omega_rf` accumulates the per-turn phase slip
  (`_dphi_rf_next` in cavities.py, as BLonD2 did), while a default `xt.Cavity`
  (`absolute_time=False`) computes its phase from zeta alone — static offset, no slip
  (`absolute_time=True` restores the drift). The slip belongs to exactly one element — whichever
  owns the RF phase — never computed in both codes, which would double-count it. The legacy
  `ZetaShift` radial steering was their bunch-placement convention extended to detuned RF (keep
  the bunch at zeta ≈ 0 on the steered orbit). Detuned-RF operation is untested with the new
  wrappers so far, like EX_06.
- **Stable fixed points** (dynamics): the phases where the total RF kick equals the per-turn
  energy gain with the right stability sign (set by η). A single harmonic h has h of them per
  turn — one per bucket — and multi-harmonic systems can have several within one fundamental
  period. They sit at the same physical times in both codes; neither frame owns them.
- **φ_s, the synchronous phase** (a scalar): `qV sin φ_s` equals the *per-turn synchronous energy
  gain* (not the particle coordinate dE), so φ_s characterizes the strength of acceleration. Only
  for a single harmonic, and only after convention choices (which root, transition handling,
  which bucket), does it double as the location of one particular stable point. For
  multi-harmonic RF it locates nothing.

1. The reference-system difference exists, but it is only the origin labeling above. The legacy
   interface's `zeta = −(dt − (φ_s−φ_rf)/ω)·β₀c` shift was *not* part of the coordinate
   transform: it was a bunch-placement choice that used φ_s to nominate one stable point (bucket
   0, with a transition-dependent root) and relocate it to `zeta ≈ 0`, where xsuite users
   conventionally seed bunches. It inherits every weak-definition problem of φ_s and cannot
   generalize beyond a single harmonic. For the stationary above-transition LHC the nominated
   stable point sits at exactly `t_rf/2` in BLonD's frame — confirmed against the analytic
   prediction in all three machines (LHC stationary, LHC ramp, PSB below transition with
   φ_rf = π). That's the nuance Birk saw — it's in where the bunch was placed, not in the
   coordinates and not in the physics.

2. The new interface needs no shift — by construction, verified turn-by-turn. Our transform
   preserves the RF phase, so a native BLonD3 run and a WrapBlond4Xsuite run map onto each other
   with the identity: dt and dE agree turn-by-turn over 481 stationary turns and over the EX_02
   ramp (450→450.1 GeV/c in 330 turns — 30× steeper per turn than anything we'd tested before,
   the regime where a double-counted acceleration kick or one-turn lag would explode). Wrapped vs
   pure xt.Cavity agrees at 10⁻⁹ of amplitude in all three machines. Bunch placement is left
   where it belongs: with the user seeding the particles, in whichever bucket they mean.

3. The likely origin of "we can see it in those examples": applying the legacy shift on top of
   the new interface mirrors the orbit about the bucket centre (the stable point) — the very
   first energy kick flips sign (±1.1858 MeV, measured). Anyone comparing legacy-convention
   coordinates against the new interface without knowing the seeding differs would see
   trajectories that look genuinely wrong.

4. The one real discrepancy found is a drift-model difference, not an interface one — and the
   xtrack source (`linesegmentmap.h`) pins down exactly which models are in play. The segment
   map's "nonlinear" mode is *not* full slippage: the "nonlinear" refers to the sinusoidal RF
   kick, while the drift is `zeta −= η₀·L·δ` — linear in the exact δ, first-order momentum
   compaction only. So three models: DriftSimple (linear in dE), the map (linear in δ), and
   DriftExact (exact kinematics and velocity slippage; both BLonD C++ kernels verified correct
   term by term). The PSB case (γ≈2.5, δ up to 2.5×10⁻³) dephases between DriftSimple and the
   map at 3.6×10⁻⁴ of amplitude over 6000 turns — that is purely the dE→δ conversion
   nonlinearity — and is invisible at LHC γ≈480. DriftExact disagrees with the map *more*
   (9.6×10⁻³), which is now explained, not curious: the map drops the O(δ²) slippage terms that
   DriftExact keeps, and at this γ they dominate — the most exact BLonD model is necessarily the
   furthest from the map. No xsuite-side model matches either BLonD drift exactly (the other
   LineSegmentMap modes are full linear rotations; xtrack's true drifts carry no momentum
   compaction), so the clincher remains: running both interface directions with the *identical*
   drift object (BLonD-master with wrapped segment map vs xsuite-master with wrapped cavity)
   agrees to 8×10⁻¹² over 6000 turns, growing ~√turn — pure float accumulation.

5. A follow-up finding on *where in the turn* the reference-energy step belongs. The first
   version of these tests ported the legacy relabel position verbatim (line start, indexing
   `at_turn`, matching xtrack's turn-start `EnergyProgram` convention) and aligned BLonD to it
   by shifting the momentum program (`values_after_turn = p_program[:-1]`). That agreed to
   1.5×10⁻⁷ over the PSB 6000 turns — but measuring the growth law exposed it as ~N⁰·⁹, a
   *systematic*, not float noise: BLonD computes the kick's omega_rf from the *incoming*
   reference beta (physically correct — the beam approaches the gap at beta_a; the reference
   steps at the cavity), while under the turn-start convention the wrapped station kicked with
   the *post-step* beta. One turn's Δβ/β (~8×10⁻¹¹ per turn at PSB), every turn, coherently.
   The fix: move the xsuite-side relabel element to directly after the cavity (idiomatic
   xsuite — `ReferenceEnergyIncrease` is a placeable element), so both codes share the model
   "kick at incoming reference, relabel where the energy changes". This also lets the BLonD
   runs consume the program unshifted (`p_program[1:]`, ramp starts on turn 0, every node
   used) and dropped the cross-direction residual by ~4 orders of magnitude to the rounding
   floor; the test now asserts at 10⁻⁹ of amplitude. Note this is a sub-turn *modeling choice*
   with negligible physical impact (bounded by one turn's Δβ) — the win is diagnostic
   sharpness, not physics. xtrack's built-in `line.energy_program` remains turn-start by
   construction, so the harness ramp tests keep that convention (both their sides already
   share it exactly).

So: Birk is right that the difference exists and shows in those examples; you're right that it's
not a bug — and now there are 12 permanent integration tests proving both statements, with the
convention documented in the test module docstring. Not recreated: EX_03 (two RF stations) and
EX_06 (phase loop) need multi-station/feedback support the wrappers don't have yet — worth noting
as future scope.
