# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Recreation of the legacy ``blond-and-xsuite-interface`` integration examples.

The legacy interface (Karlsen-Baeck / van Rijswijk / Timko) bridged BLonD2
and xsuite with the transform
``zeta = -(dt - (phi_s - phi_rf)/omega_rf) * beta0 * c``. Three distinct
things are tangled in that shift, and these tests keep them apart:

* **Coordinate systems** (pure bookkeeping): BLonD pins ``dt = 0`` to the
  RF clock zero; xsuite pins ``zeta = 0`` to the reference particle. The
  transform between the frames is ``zeta = -dt * beta0 * c`` — no physics
  in it, and frequency-agnostic: with RF detuned off design the *bucket*
  drifts, but identically in both frames, so the bridge stays the
  identity. What differs there is the RF *model*: a BLonD station with
  ``delta_omega_rf`` accumulates the per-turn phase slip
  (``_dphi_rf_next``, as BLonD2 did), while a default ``xt.Cavity``
  (``absolute_time=False``) computes its phase from zeta alone — a static
  offset, no slip. The slip belongs to exactly one element (whichever
  owns the RF phase), never both. The legacy ``ZetaShift`` radial
  steering was the bunch-placement convention extended to detuned RF;
  detuned operation is untested with the new wrappers, like EX_06.
* **Stable fixed points** (dynamics): the phases where the total RF kick
  equals the per-turn energy gain with the right stability sign (set by
  eta). A single harmonic ``h`` has ``h`` of them per turn — one per
  bucket — and multi-harmonic systems can have several within one
  fundamental period. They sit at the same physical times in both codes;
  neither frame owns them.
* **phi_s, the synchronous phase** (a scalar): ``q V sin(phi_s)`` equals
  the *per-turn synchronous energy gain* (not the particle coordinate
  ``dE``), so phi_s characterizes the strength of acceleration. Only for
  a single harmonic, and only after convention choices (which root,
  transition handling, which bucket), does it double as the location of
  one particular stable point.

The legacy ``(phi_s - phi_rf)/omega`` term was therefore not part of the
coordinate transform: it was a *bunch-placement* choice that nominated one
stable point and relocated it to ``zeta ~ 0``, where xsuite users
conventionally seed bunches.

These tests recreate the legacy single-particle examples (LHC stationary,
LHC ramp, PSB ramp below transition) with the new ``WrapBlond4Xsuite`` and
separate two questions the legacy examples mixed together:

1. **Interface correctness**: a wrapped BLonD station must reproduce the
   equivalent pure ``xt.Cavity`` *exactly* (same line, same seeding) — any
   mismatch here is a genuine bug in the new interface.
2. **Bunch placement**: the new interface's transform is *RF-phase
   preserving* (no shift), so BLonD-frame ``dt`` maps onto the wrapped run
   **identically** — the cross-code bridge is the identity, asserted here
   turn by turn. The bucket centre is a property of the dynamics and sits
   at the stable-point time in either frame
   (``zeta_stable = -dt_stable * beta0 * c``), quantified and pinned
   below. Placement is left to the user's seeding; seeding with the
   legacy shift anyway *mirrors* the orbit about the bucket centre
   (verified as a sign flip of the first energy kick).
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")
pytest.importorskip("xpart")

from blond import (  # noqa: E402
    Beam,
    BeamObservationOncePerTurn,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    copy_to_cpu,
    proton,
)
from blond.interfaces.xsuite.elements import WrapBlond4Xsuite  # noqa: E402
from tests.integration.xsuite._lhc_harness import (  # noqa: E402
    matched_context,
)

# ---------------------------------------------------------------------------
# Machine parameter sets, copied verbatim from the legacy examples.
# ---------------------------------------------------------------------------

LHC = {
    "circumference": 26658.8832,
    "alpha": 0.00034849575112251314,
    "harmonic": 35640,
    "voltage": 5e6,
    "phi_rf": 0.0,  # legacy ``dphi = 0``
}
# EX_01: stationary, 481 turns, particle at 2*blen - 0.1 ns.
LHC_STATIONARY = {
    "p_program_ev": np.full(481 + 1, 450e9),
    "n_turns": 481,
    "input_dt": 2 * 1.25e-9 - 0.1e-9,
}
# EX_02: 450 -> 450.1 GeV/c in 330 turns, particle at 2*blen - 0.4 ns.
LHC_RAMP = {
    "p_program_ev": np.linspace(450e9, 450.1e9, 330 + 1),
    "n_turns": 330,
    "input_dt": 2 * 1.25e-9 - 0.4e-9,
}

# EX_05: PSB below transition, h=1, phi_rf=pi, tiny ramp over 6000 turns.
_PSB_E0 = proton.mass  # [eV]
_PSB_EKIN = 1.4e9
_PSB_P = float(np.sqrt((_PSB_E0 + _PSB_EKIN) ** 2 - _PSB_E0**2))
PSB = {
    "circumference": 2 * np.pi * 25.000057945260654,
    "alpha": 1.0 / 4.11635447373496**2,
    "harmonic": 1,
    "voltage": 8e3,
    "phi_rf": np.pi,  # legacy ``dphi = pi``
}
PSB_RAMP = {
    "p_program_ev": np.linspace(_PSB_P, _PSB_P + 0.001e9, 6000 + 1),
    "n_turns": 6000,
    "input_dt": 5.720357923415153e-07 - 180e-9 / 4,
}


def _total_energy(p0c: float) -> float:
    return float(np.sqrt(p0c**2 + proton.mass**2))


def _beta(p0c: float) -> float:
    return p0c / _total_energy(p0c)


def _omega_rf(machine: dict, p0c: float) -> float:
    return (
        machine["harmonic"]
        * 2.0
        * np.pi
        * _beta(p0c)
        * c
        / machine["circumference"]
    )


def _is_above_transition(machine: dict, p0c: float) -> bool:
    gamma = _total_energy(p0c) / proton.mass
    return machine["alpha"] - 1.0 / gamma**2 > 0


def _stable_point_dt(machine: dict, p0c: float, delta_e: float) -> float:
    """
    BLonD-frame time of a stable fixed point, in [s].

    Solves ``qV sin(omega dt + phi_rf) = delta_e`` for a *stable* root
    (``cos < 0`` above transition, ``cos > 0`` below) and picks the one in
    ``[0, t_rf)``. Both picks are conventions, not physics: with harmonic
    ``h`` there are ``h`` such points per turn. This nominated point is
    exactly what the legacy interface relocated to ``zeta ~ 0`` via its
    ``(phi_s - phi_rf) / omega_rf`` bunch-placement shift.
    """
    omega = _omega_rf(machine, p0c)
    k = delta_e / machine["voltage"]  # proton: q = 1 e, V in volts
    assert abs(k) <= 1.0
    phase = float(np.arcsin(k))  # stable below transition (cos > 0)
    if _is_above_transition(machine, p0c):
        phase = np.pi - phase  # stable above transition (cos < 0)
    dt_s = (phase - machine["phi_rf"]) / omega
    t_rf = 2.0 * np.pi / omega
    return dt_s % t_rf


# ---------------------------------------------------------------------------
# Native BLonD3 run (the legacy examples' BLonD side, on the new codebase).
# ---------------------------------------------------------------------------


def _run_native_blond(machine: dict, case: dict):
    """Track one particle with a plain BLonD3 Simulation.

    Returns per-turn (dt, dE, reference total energy), one entry per
    tracked turn.
    """
    n_turns = case["n_turns"]
    p_program = case["p_program_ev"]

    ring = Ring(circumference=machine["circumference"])
    rf_station = SingleHarmonicRFStation(
        voltage=machine["voltage"],
        phi_rf=machine["phi_rf"],
        harmonic=machine["harmonic"],
    )
    drift = DriftSimple(
        orbit_length=machine["circumference"],
        momentum_compaction_factor=machine["alpha"],
    )
    ring.add_elements((rf_station, drift), reorder=True)

    # ``values_after_turn[i]`` is the reference the station relabels to
    # during turn ``i`` (kick at the incoming reference, relabel at the
    # cavity): the natural reading of the program, using every node. The
    # xsuite runs share this model by placing their relabel element
    # directly after the cavity (see _PerTurnEnergyUpdate).
    energy_cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        value_init=float(p_program[0]),
        values_after_turn=np.asarray(p_program[1:], dtype=float),
        in_unit="momentum",
    )

    beam = Beam(intensity=1.0, particle_type=proton)
    beam.setup_beam(
        dt=np.array([case["input_dt"]]),
        dE=np.array([0.0]),
        reference_total_energy=_total_energy(float(p_program[0])),
    )

    bunch_obs = BeamObservationOncePerTurn(each_turn_i=1, warn=False)
    simulation = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    simulation.run_simulation(
        beams=(beam,),
        n_turns=n_turns,
        observe=(bunch_obs,),
        show_progressbar=False,
        verbose=False,
    )

    dts = copy_to_cpu(bunch_obs.dts)[:, 0]
    dEs = copy_to_cpu(bunch_obs.dEs)[:, 0]
    ref_E = copy_to_cpu(bunch_obs.reference_total_energy)
    return dts, dEs, ref_E


def _run_blond_master_wrapped_drift(machine: dict, case: dict):
    """
    BLonD-master twin of the wrapped-cavity run.

    Native BLonD RF station plus the *same* xsuite segment map (via
    ``WrapXsuite4Blond``) as the drift. Comparing this against the
    xsuite-master run isolates the interface itself — both runs use the
    identical drift model and the identical BLonD kick, only the loop
    ownership differs.
    """
    from blond.interfaces.xsuite.elements import WrapXsuite4Blond

    n_turns = case["n_turns"]
    p_program = case["p_program_ev"]

    segment_map = xt.LineSegmentMap(
        qx=1.1,
        qy=1.2,
        betx=1.0,
        bety=1.0,
        # RF zeroed -> pure drift, linear in delta (zeta -= eta0*L*delta).
        # xtrack infers its "nonlinear" mode from these (the name refers to
        # the sinusoidal RF kick, skipped at frequency 0); omitting them
        # entirely would freeze the longitudinal plane instead. lag_rf must
        # stay explicit: older xtrack (as on CI) asserts it is not None.
        voltage_rf=0.0,
        frequency_rf=0.0,
        lag_rf=0.0,
        momentum_compaction_factor=machine["alpha"],
        length=machine["circumference"],
    )
    ring = Ring(circumference=machine["circumference"])
    rf_station = SingleHarmonicRFStation(
        voltage=machine["voltage"],
        phi_rf=machine["phi_rf"],
        harmonic=machine["harmonic"],
    )
    wrapper = WrapXsuite4Blond(
        xt.Line(elements=[segment_map], element_names=["m"])
    )
    ring.add_elements((rf_station, wrapper), section_index=0)

    # Same model as _run_native_blond: unshifted program, relabel at the
    # cavity (the wrapped-drift twin of the xsuite element order).
    energy_cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        value_init=float(p_program[0]),
        values_after_turn=np.asarray(p_program[1:], dtype=float),
        in_unit="momentum",
    )
    beam = Beam(intensity=1.0, particle_type=proton)
    beam.setup_beam(
        dt=np.array([case["input_dt"]]),
        dE=np.array([0.0]),
        reference_total_energy=_total_energy(float(p_program[0])),
    )
    bunch_obs = BeamObservationOncePerTurn(each_turn_i=1, warn=False)
    simulation = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    simulation.run_simulation(
        beams=(beam,),
        n_turns=n_turns,
        observe=(bunch_obs,),
        show_progressbar=False,
        verbose=False,
    )
    dts = copy_to_cpu(bunch_obs.dts)[:, 0]
    dEs = copy_to_cpu(bunch_obs.dEs)[:, 0]
    return dts, dEs


# ---------------------------------------------------------------------------
# xsuite runs (the legacy examples' xsuite side): same line for the pure
# cavity and the wrapped BLonD station. The per-turn relabel and cavity-
# frequency elements port the legacy EnergyUpdate / EnergyFrequencyUpdate,
# with one deliberate improvement: the relabel sits directly after the
# cavity (BLonD's model) instead of at the line start — see
# _PerTurnEnergyUpdate for why.
# ---------------------------------------------------------------------------


class _PerTurnEnergyUpdate:
    """Per-turn reference relabel, placed directly *after* the cavity.

    Relabels p0c to the program value the current turn's cavity kick
    accelerates towards (``p_program[at_turn + 1]``). The legacy
    ``EnergyUpdate`` instead ran at the line *start* with
    ``p_program[at_turn]`` (a no-op on turn 0, mirroring xtrack's
    turn-start ``EnergyProgram`` convention). Relabelling at the cavity
    instead shares BLonD's model — kick with the incoming reference,
    relabel where the energy changes — so the BLonD runs consume the
    momentum program unshifted and the RF frequency sampling agrees
    between the codes on every turn (the turn-start convention leaves a
    one-turn lag in the kick's omega_rf, a systematic that grows
    linearly with turn number; see
    ``test_both_wrapper_directions_agree_with_identical_drift``).
    """

    def __init__(self, p_program: np.ndarray):
        self._p_program = np.asarray(p_program, dtype=float)
        # Standalone element: ``track`` runs on the element's own context,
        # so it must live on the same context as the particles (GPU CI).
        self._increase = xt.ReferenceEnergyIncrease(
            Delta_p0c=0.0, _context=matched_context()
        )

    def track(self, particles) -> None:
        turn = int(copy_to_cpu(particles.at_turn)[0])
        p0c_now = float(copy_to_cpu(particles.p0c)[0])
        # The extra tracked turn (discarded from the comparison window)
        # has no successor node; relabelling to the last node is a no-op.
        turn_next = min(turn + 1, len(self._p_program) - 1)
        self._increase.Delta_p0c = self._p_program[turn_next] - p0c_now
        self._increase.track(particles)


class _PerTurnCavityFrequency:
    """Port of the legacy ``EnergyFrequencyUpdate``'s frequency half.

    Pins the cavity frequency to ``h * f_rev`` of the current turn's
    *incoming* momentum (the RF frequency the beam approaches the gap
    with — the relabel to the next node happens after the cavity).
    """

    def __init__(self, cavity, machine: dict, p_program: np.ndarray):
        self._cavity = cavity
        self._f_program = np.array(
            [
                machine["harmonic"]
                * _beta(float(p))
                * c
                / machine["circumference"]
                for p in p_program
            ]
        )

    def track(self, particles) -> None:
        turn = int(copy_to_cpu(particles.at_turn)[0])
        self._cavity.frequency = self._f_program[turn]


def _run_xsuite(machine: dict, case: dict, zeta0: float, wrapped: bool):
    """Track one particle in xsuite.

    The RF is either a pure ``xt.Cavity`` or the wrapped BLonD station.
    Returns per-turn (zeta, ptau, p0c) with
    entry ``i`` = the state *after* turn ``i`` (monitor snapshots shifted by
    one, like the existing ramp tests).
    """
    n_turns = case["n_turns"]
    p_program = case["p_program_ev"]
    p0 = float(p_program[0])

    drift = xt.LineSegmentMap(
        qx=1.1,
        qy=1.2,
        betx=1.0,
        bety=1.0,
        # RF zeroed -> pure drift, linear in delta (zeta -= eta0*L*delta).
        # xtrack infers its "nonlinear" mode from these (the name refers to
        # the sinusoidal RF kick, skipped at frequency 0); omitting them
        # entirely would freeze the longitudinal plane instead. lag_rf must
        # stay explicit: older xtrack (as on CI) asserts it is not None.
        voltage_rf=0.0,
        frequency_rf=0.0,
        lag_rf=0.0,
        momentum_compaction_factor=machine["alpha"],
        length=machine["circumference"],
    )

    if wrapped:
        rf_element = WrapBlond4Xsuite(
            element=SingleHarmonicRFStation.headless(
                section_index=0,
                voltage=machine["voltage"],
                phi_rf=machine["phi_rf"],
                harmonic=machine["harmonic"],
                circumference=machine["circumference"],
                beam_reference_beta=_beta(p0),
                magnetic_cycle=None,
            )
        )
        frequency_update = None
    else:
        rf_element = xt.Cavity(
            voltage=machine["voltage"],
            frequency=machine["harmonic"]
            * _beta(p0)
            * c
            / machine["circumference"],
            lag=np.degrees(machine["phi_rf"]),
        )
        frequency_update = _PerTurnCavityFrequency(
            rf_element, machine, p_program
        )

    # Element order implements the shared model: RF kick at the *incoming*
    # reference, then the relabel (as BLonD does at the station), then the
    # drift at the new reference. (The legacy line ran the relabel first,
    # i.e. the turn-start convention — see _PerTurnEnergyUpdate.)
    elements = {}
    if frequency_update is not None:
        elements["frequency_update"] = frequency_update
    elements["rf"] = rf_element
    elements["energy_update"] = _PerTurnEnergyUpdate(p_program)
    elements["drift"] = drift
    line = xt.Line(
        elements=list(elements.values()), element_names=list(elements.keys())
    )
    line.particle_ref = xt.Particles(
        p0c=p0, mass0=proton.mass, q0=proton.charge
    )
    line.build_tracker(_context=matched_context())

    particles = line.build_particles(
        x=0, px=0, y=0, py=0, zeta=zeta0, ptau=0.0
    )
    line.track(particles, num_turns=n_turns + 1, turn_by_turn_monitor=True)
    rec = line.record_last_track
    # rec[:, t] is the state before turn t -> after-turn-i is rec[:, i+1].
    zeta = copy_to_cpu(rec.zeta)[0, 1 : n_turns + 1]
    ptau = copy_to_cpu(rec.ptau)[0, 1 : n_turns + 1]
    p0c = copy_to_cpu(rec.p0c)[0, 1 : n_turns + 1]
    return zeta, ptau, p0c


# ---------------------------------------------------------------------------
# The recreated examples.
# ---------------------------------------------------------------------------


class _RecreatedLegacyExample:
    """Mixin with the three checks; subclasses bind machine + case."""

    machine: dict
    case: dict
    # Interface check: wrapped vs pure cavity, identical seeding/line.
    rtol_interface = 1e-9
    # Cross-code check: identity-bridged wrapped run vs native BLonD3.
    rtol_cross_code = 1e-5

    def _amplitudes(self, dts, dEs):
        return (
            max(float(np.max(np.abs(dts))), 1e-12),
            max(float(np.max(np.abs(dEs))), 1.0),
        )

    def test_wrapped_equals_pure_xsuite(self):
        """The new interface must be indistinguishable from xt.Cavity."""
        zeta0 = (
            -self.case["input_dt"]
            * _beta(float(self.case["p_program_ev"][0]))
            * c
        )

        zx, px_, _ = _run_xsuite(self.machine, self.case, zeta0, wrapped=False)
        zw, pw, _ = _run_xsuite(self.machine, self.case, zeta0, wrapped=True)

        z_amp = max(float(np.max(np.abs(zx))), 1e-12)
        p_amp = max(float(np.max(np.abs(px_))), 1e-15)
        np.testing.assert_allclose(
            zw, zx, atol=z_amp * self.rtol_interface, rtol=0.0
        )
        np.testing.assert_allclose(
            pw, px_, atol=p_amp * self.rtol_interface, rtol=0.0
        )

    def test_wrapped_run_matches_native_blond_identically(self):
        """Cross-code check: the bridge between frames is the *identity*.

        The new interface preserves the RF phase across the boundary, so a
        particle seeded at BLonD-frame ``dt`` as ``zeta = -dt * beta0 * c``
        must reproduce the native BLonD3 run turn by turn with the plain
        inverse transform — no bunch-placement shift anywhere. (The
        remaining tolerance covers the different drift models: BLonD's
        ``DriftSimple`` is linear in dE while the segment map's slippage
        is linear in the exact delta — see the EX_05 class docstring.)
        """
        machine, case = self.machine, self.case
        n_turns = case["n_turns"]

        dts_b, dEs_b, ref_E_b = _run_native_blond(machine, case)

        p0 = float(case["p_program_ev"][0])
        zeta0 = -case["input_dt"] * _beta(p0) * c
        zeta_w, ptau_w, p0c_w = _run_xsuite(machine, case, zeta0, wrapped=True)

        # Reference energies must agree turn by turn (alignment check).
        energy_w = np.array([_total_energy(p) for p in p0c_w])
        np.testing.assert_allclose(
            energy_w,
            ref_E_b[:n_turns],
            rtol=1e-12,
            err_msg="reference-energy programs misaligned between runs",
        )

        beta_w = p0c_w / energy_w
        dts_w = -zeta_w / (beta_w * c)
        dEs_w = ptau_w * beta_w * energy_w

        dt_amp, dE_amp = self._amplitudes(dts_b - np.mean(dts_b), dEs_b)
        np.testing.assert_allclose(
            dEs_w,
            dEs_b,
            atol=dE_amp * self.rtol_cross_code,
            rtol=0.0,
            err_msg="energy evolution differs between codes",
        )
        np.testing.assert_allclose(
            dts_w,
            dts_b,
            atol=dt_amp * self.rtol_cross_code,
            rtol=0.0,
            err_msg="time evolution differs between codes",
        )

    def test_both_wrapper_directions_agree_with_identical_drift(self):
        """Strongest interface check: same drift model, opposite directions.

        BLonD-master (native station + wrapped segment map) vs xsuite-master
        (wrapped station + native segment map) share every physics model;
        only the interface direction differs. Any disagreement beyond float
        accumulation would be an interface bug. Measured residual for the
        PSB case: ~8e-12 of amplitude over 6000 turns, growing ~sqrt(turn)
        (rounding diffusion). With the relabel at the legacy line-start
        position instead, the kick's omega_rf lags the reference by one
        turn's beta step in exactly one of the two runs — a systematic that
        grew ~linearly to ~2e-7 over the same 6000 turns; the 1e-9
        tolerance is set to reject any such systematic while keeping >100x
        margin over the rounding floor.
        """
        machine, case = self.machine, self.case

        dts_bm, dEs_bm = _run_blond_master_wrapped_drift(machine, case)

        p0 = float(case["p_program_ev"][0])
        zeta0 = -case["input_dt"] * _beta(p0) * c
        zeta_w, ptau_w, p0c_w = _run_xsuite(machine, case, zeta0, wrapped=True)
        energy_w = np.array([_total_energy(p) for p in p0c_w])
        beta_w = p0c_w / energy_w
        dts_w = -zeta_w / (beta_w * c)
        dEs_w = ptau_w * beta_w * energy_w

        dt_amp, dE_amp = self._amplitudes(dts_bm - np.mean(dts_bm), dEs_bm)
        np.testing.assert_allclose(
            dEs_w,
            dEs_bm,
            atol=dE_amp * 1e-9,
            rtol=0.0,
            err_msg="interface directions disagree (dE)",
        )
        np.testing.assert_allclose(
            dts_w,
            dts_bm,
            atol=dt_amp * 1e-9,
            rtol=0.0,
            err_msg="interface directions disagree (dt)",
        )

    def test_bucket_centre_sits_at_predicted_stable_point(self):
        """The bucket centre is the stable point the legacy shift relocated.

        The bucket centre is a property of the dynamics, not of either
        frame: in BLonD coordinates it sits at the stable-point time
        ``dt_stable`` (mod ``t_rf``). The legacy interface nominated this
        point via phi_s and parked it at ``zeta ~ 0`` — a bunch-placement
        choice, not a frame difference (the frames differ only in origin
        labeling, bridged by the identity transform asserted above).
        Pinning the analytic prediction here keeps that choice documented
        and quantified.
        """
        machine, case = self.machine, self.case
        p0 = float(case["p_program_ev"][0])
        delta_E0 = _total_energy(
            float(case["p_program_ev"][1])
        ) - _total_energy(p0)
        dt_stable0 = _stable_point_dt(machine, p0, delta_E0)
        t_rf = 2.0 * np.pi / _omega_rf(machine, p0)

        dts_b, dEs_b, _ = _run_native_blond(machine, case)
        # A pendulum orbit is symmetric in time about its centre, so the
        # midpoint of the turning points estimates the bucket centre.
        centre_estimate = (
            (float(np.min(dts_b)) + float(np.max(dts_b))) / 2.0 % t_rf
        )
        self.assertLess(
            abs(centre_estimate - dt_stable0),
            0.05 * t_rf,
            msg=(
                f"BLonD bucket centre {centre_estimate:.3e} s is not at the "
                f"predicted stable-point time {dt_stable0:.3e} s"
            ),
        )


class TestLegacyEX01LhcStationary(_RecreatedLegacyExample, unittest.TestCase):
    """EX_01: LHC, 450 GeV/c, no ramp.

    The stable point sits exactly half an RF period from the RF clock
    zero (stationary, above transition, phi_rf = 0).
    """

    machine = LHC
    case = LHC_STATIONARY


class TestLegacyEX02LhcRamp(_RecreatedLegacyExample, unittest.TestCase):
    """EX_02: LHC ramp 450 -> 450.1 GeV/c in 330 turns.

    Thirty times steeper per turn than the harness ramp tests —
    sensitive to one-turn-lag bugs.
    """

    machine = LHC
    case = LHC_RAMP


class TestLegacyEX05PsbRamp(_RecreatedLegacyExample, unittest.TestCase):
    """EX_05: PSB below transition, h=1, phi_rf=pi, 6000-turn mini-ramp.

    At gamma ~ 2.5 the drift *models* genuinely differ. Three models are
    in play (xtrack ``linesegmentmap.h``: the "nonlinear" in the mode name
    refers to the sinusoidal RF kick — the slippage itself is linear):

    * BLonD ``DriftSimple``: ``dt += T*eta0*dE/(beta^2 E)`` — linear in dE;
    * segment map: ``zeta -= eta0*L*delta`` — linear in the *exact* delta,
      first-order momentum compaction only;
    * BLonD ``DriftExact``: exact kinematics and velocity slippage.

    Simple vs map differ only by the dE -> delta conversion nonlinearity:
    with delta up to ~2.5e-3 the synchrotron frequencies differ at the
    1e-4 level and the runs dephase slowly (measured: 3.6e-4 of amplitude
    over 6000 turns; the identical-drift cross-direction test pins the
    interface itself at ~8e-12). ``DriftExact`` disagrees with the map
    *more* (~9.6e-3) because the map drops the O(delta^2) slippage terms
    that ``DriftExact`` keeps — large at this gamma, so the most exact
    BLonD model is necessarily the furthest from the map. At LHC energies
    (gamma ~ 480) the same model differences fall below the 1e-5 test
    tolerance over the few hundred tracked turns.
    """

    machine = PSB
    case = PSB_RAMP
    rtol_cross_code = 5e-4


if __name__ == "__main__":
    unittest.main()
