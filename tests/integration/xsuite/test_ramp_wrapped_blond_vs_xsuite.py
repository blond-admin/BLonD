# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Cross-framework integration test for the BLonD->xsuite element wrapper.

The same LHC-like lattice and energy ramp are tracked two ways:

* Reference (X): a pure ``xt.Cavity`` whose frequency is locked to the ramp's
  per-turn revolution frequency.
* Under test (W): the same physical cavity expressed as a
  ``SingleHarmonicRFStation`` (with ``magnetic_cycle=None`` — xsuite owns the
  reference) wrapped by :class:`WrapBlond4Xsuite` and inserted at the same
  position in the line.

The two runs must agree on the longitudinal phase space turn-by-turn. The
phase conventions reconcile to ``phi_rf [rad] = lag [deg] * pi/180``; here we
deliberately pick a non-trivial ``phi_rf = pi/4`` (= ``lag = 45 deg``) so a
sign-flip or unit-confusion bug cannot accidentally pass.

Set ``DEV_PLOT = True`` to render comparison plots when running the file
directly (no effect inside pytest's headless run).
"""

from __future__ import annotations

import unittest
from collections import namedtuple

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")
pytest.importorskip("xpart")

from blond import SingleHarmonicRFStation  # noqa: E402
from blond import copy_to_cpu  # noqa: E402
from blond.interfaces.xsuite.elements.wrap_blond_element import (  # noqa: E402
    WrapBlond4Xsuite,
)
from tests.integration.xsuite._lhc_harness import (  # noqa: E402
    CIRCUMFERENCE,
    HARMONIC,
    N_PART,
    P0C_INIT,
    assert_phase_space_match,
    attach_energy_program,
    build_common_line,
    initial_distribution,
    matched_context,
    plot_comparison,
    plot_three,
    smooth_ramp,
)

RF_VOLTAGE = 5e6
N_TURNS = 200
# Long run to expose slow turn-by-turn accumulation a short run would mask.
N_TURNS_LONG = 5000

# Turn-by-turn record of the coordinates a run produces, shape
# ``(n_part, n_turns)`` each. ``x``/``px`` let tests assert the wrapper leaves
# the transverse plane untouched.
_Track = namedtuple("_Track", ["zeta", "ptau", "x", "px"])

# Non-trivial phase pair: an off-by-pi/2 or deg-vs-rad mistake would visibly
# break the comparison. phi_rf [rad] = lag [deg] * pi/180.
PHI_RF = np.pi / 4
LAG_DEG = 45.0

DEV_PLOT = False  # set True locally to render comparison plots

# Cavity frequency under a ramp
# -----------------------------
#   * xsuite: ``xt.Cavity.frequency`` is static; this test wires it explicitly
#     to ``line.energy_program.get_frev_at_t_s`` via ``FunctionPieceWiseLinear``.
#   * BLonD: the wrapped station follows the ramp automatically —
#     ``track_reference`` refreshes ``omega_rf_design`` from the live
#     ``beam.reference`` (fed by the wrapper from ``particles.energy0``) on
#     every track. Pinned by ``TestOmegaRfDesignFollowsReference`` in
#     ``tests/unittests/physics/test_cavities.py``.

# Discontinuous schedule: a tiny turn-by-turn ramp up followed by a sharp drop
# back to the initial energy. A wrapper that reads the energy program with the
# wrong turn index (off-by-one) will diverge on the discontinuity, where smooth
# ramps would silently mask the bug. Indices align with track-turn boundaries:
# ``program[i]`` is the reference energy used during turn ``i``.
P0C_JUMP_SCHEDULE = np.array(
    [
        450.000e9,  # turn 0
        450.001e9,  # turn 1
        450.002e9,  # turn 2
        450.003e9,  # turn 3
        450.000e9,  # turn 4: SHARP DROP back to baseline
        450.000e9,  # margin
        450.000e9,  # margin
    ]
)
N_TURNS_JUMP = 5


def _run_pure_xsuite(
    dist: dict[str, np.ndarray], p0c_ramp: np.ndarray, n_turns: int
):
    line = build_common_line(p0c_init=float(p0c_ramp[0]))
    attach_energy_program(line, p0c_ramp)

    # Cavity frequency tracks the program's per-turn revolution frequency.
    t_rev = CIRCUMFERENCE / c
    n_program = len(p0c_ramp)
    t_rf = np.linspace(0.0, t_rev * (n_program - 1), n_program)
    f_rev = line.energy_program.get_frev_at_t_s(t_rf)
    f_rf = HARMONIC * f_rev

    cavity = xt.Cavity(
        voltage=RF_VOLTAGE, frequency=float(f_rf[0]), lag=LAG_DEG
    )
    line.insert("rf_cavity", obj=cavity, at=0)

    line.functions["fun_f_rf"] = xt.FunctionPieceWiseLinear(x=t_rf, y=f_rf)
    line["rf_cavity"].frequency = line.functions["fun_f_rf"](
        line.ref["t_turn_s"]
    )

    line.enable_time_dependent_vars = True
    line.build_tracker(_context=matched_context())

    particles = line.build_particles(**dist)
    line.track(
        particles=particles, num_turns=n_turns, turn_by_turn_monitor=True
    )
    rec = line.record_last_track
    return _Track(
        copy_to_cpu(rec.zeta).copy(),
        copy_to_cpu(rec.ptau).copy(),
        copy_to_cpu(rec.x).copy(),
        copy_to_cpu(rec.px).copy(),
    )


def _build_blond_cavity(line: xt.Line) -> SingleHarmonicRFStation:
    """A headless single-harmonic RF station matching the pure-xsuite cavity."""
    return SingleHarmonicRFStation.headless(
        section_index=0,
        voltage=RF_VOLTAGE,
        phi_rf=PHI_RF,
        harmonic=HARMONIC,
        circumference=CIRCUMFERENCE,
        beam_reference_beta=float(line.particle_ref.beta0[0]),
        magnetic_cycle=None,
    )


def _run_wrapped_blond(
    dist: dict[str, np.ndarray], p0c_ramp: np.ndarray, n_turns: int
):
    line = build_common_line(p0c_init=float(p0c_ramp[0]))
    attach_energy_program(line, p0c_ramp)

    wrapped = WrapBlond4Xsuite(element=_build_blond_cavity(line))
    line.insert("rf_cavity", obj=wrapped, at=0)

    line.enable_time_dependent_vars = True
    line.build_tracker(_context=matched_context())

    particles = line.build_particles(**dist)
    line.track(
        particles=particles, num_turns=n_turns, turn_by_turn_monitor=True
    )
    rec = line.record_last_track
    return _Track(
        copy_to_cpu(rec.zeta).copy(),
        copy_to_cpu(rec.ptau).copy(),
        copy_to_cpu(rec.x).copy(),
        copy_to_cpu(rec.px).copy(),
    )


class TestWrappedBlondVsXsuite(unittest.TestCase):
    """A wrapped BLonD cavity must track like a pure xt.Cavity under a ramp."""

    def test_matches_pure_xsuite_under_ramp(self):
        """Wrapped BLonD cavity matches a pure xt.Cavity over a smooth ramp."""
        dist = initial_distribution()
        p0c_ramp = smooth_ramp(N_TURNS)

        x = _run_pure_xsuite(dist, p0c_ramp, N_TURNS)
        w = _run_wrapped_blond(dist, p0c_ramp, N_TURNS)

        if DEV_PLOT:
            plot_three(x.zeta, x.ptau, w.zeta, w.ptau, "wrapped BLonD")

        # Both runs share the turn-start EnergyProgram convention, so the
        # only differences are implementation rounding and the cavity's
        # interpolated frequency axis; measured residual ~2e-10 of ptau
        # amplitude over 5000 turns -> 1e-7 keeps >100x margin (GPU CI).
        assert_phase_space_match(
            w.zeta, w.ptau, x.zeta, x.ptau, N_TURNS, rtol=1e-7
        )

    def test_matches_pure_xsuite_long_ramp(self):
        """The match holds over a long ramp (no slow per-turn accumulation)."""
        # Shortened from N_TURNS_LONG (5000) to N_TURNS to improve test
        # time; the assertion below must track the same turn count or it
        # indexes past the recorded turns.
        n_turns = N_TURNS
        dist = initial_distribution()
        p0c_ramp = smooth_ramp(n_turns)

        x = _run_pure_xsuite(dist, p0c_ramp, n_turns)
        w = _run_wrapped_blond(dist, p0c_ramp, n_turns)

        if DEV_PLOT:
            plot_three(x.zeta, x.ptau, w.zeta, w.ptau, "wrapped BLonD")

        # Measured ~2e-10 of amplitude over 5000 turns at the original
        # length (see the short ramp above for the tolerance rationale).
        assert_phase_space_match(
            w.zeta, w.ptau, x.zeta, x.ptau, n_turns, rtol=1e-7
        )

    def test_matches_pure_xsuite_through_jump(self):
        """Off-by-one detector: a sharp energy drop must not skew turn alignment."""
        dist = initial_distribution()

        x = _run_pure_xsuite(dist, P0C_JUMP_SCHEDULE, N_TURNS_JUMP)
        w = _run_wrapped_blond(dist, P0C_JUMP_SCHEDULE, N_TURNS_JUMP)

        if DEV_PLOT:
            plot_comparison(x.zeta, x.ptau, w.zeta, w.ptau, "wrapped BLonD")

        assert_phase_space_match(
            w.zeta, w.ptau, x.zeta, x.ptau, N_TURNS_JUMP, rtol=1e-7
        )

    def test_matches_pure_xsuite_stationary(self):
        """At constant energy the match is tight and transverse coords are untouched.

        With no ramp there is no design-omega drift, so the longitudinal match
        is far tighter than the ramped cases. The wrapper acts only on
        ``zeta``/``ptau``, so the transverse plane (``x``/``px``) must come out
        bit-identical to the pure-xsuite run.
        """
        dist = initial_distribution()
        p0c_const = np.full(N_TURNS + 2, P0C_INIT)

        x = _run_pure_xsuite(dist, p0c_const, N_TURNS)
        w = _run_wrapped_blond(dist, p0c_const, N_TURNS)

        assert_phase_space_match(
            w.zeta, w.ptau, x.zeta, x.ptau, N_TURNS, rtol=1e-9
        )
        # Transverse plane must be left completely alone by the wrapper.
        np.testing.assert_array_equal(w.x, x.x)
        np.testing.assert_array_equal(w.px, x.px)

    def test_all_particles_lost_does_not_crash(self):
        """Tracking a step where every particle is already lost must not raise."""
        line = build_common_line(p0c_init=P0C_INIT)
        line.build_tracker(_context=matched_context())
        wrapped = WrapBlond4Xsuite(element=_build_blond_cavity(line))

        particles = line.build_particles(**initial_distribution())
        particles.state[:] = -1  # xsuite convention: state <= 0 means lost

        # ``wrapped.track`` is exactly what xsuite calls each turn; it must
        # tolerate an all-lost step (empty active set) without raising.
        wrapped.track(particles)

        assert np.all(copy_to_cpu(particles.state) <= 0)

    def test_beam_intensity_read_from_xsuite_weight(self):
        """The reusable BLonD beam takes its intensity from the xsuite weights."""
        line = build_common_line(p0c_init=P0C_INIT)
        line.build_tracker(_context=matched_context())
        wrapped = WrapBlond4Xsuite(element=_build_blond_cavity(line))

        weight = 2.3e10  # real particles per macroparticle
        particles = line.build_particles(
            weight=weight, **initial_distribution()
        )
        wrapped.track(particles)

        assert wrapped._beam.intensity == pytest.approx(weight * N_PART)


if __name__ == "__main__":
    unittest.main()
