# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Million-turn pure-xsuite vs pure-BLonD longitudinal cross-check (PSB).

This is the strongest cross-code check in the suite: no interface/wrapper is
used. The *same* single-harmonic RF bucket is tracked for one million turns by
two independent codes and the longitudinal phase space is compared
turn-by-turn.

* Pure xsuite: a line of ``xt.Cavity`` (the RF kick) followed by an
  ``xt.LineSegmentMap`` with its RF disabled (the drift).
* Pure BLonD: a ring of :class:`SingleHarmonicRFStation` (the kick) followed by
  :class:`_DriftLikeLineSegment` (the drift).

``_DriftLikeLineSegment`` reproduces the ``LineSegmentMap`` drift model (linear
slip factor + exact relativistic delta), and the BLonD kick matches the cavity
kick, so the two trajectories must stay together. The per-turn agreement is at
machine precision; over a million turns the only drift is floating-point
round-off (numpy vs xsuite's compiled C), which accumulates roughly linearly to
~1e-10 of the bucket amplitude -- there is *no* systematic model difference
(the synchrotron tunes are identical).

Conventions (validated by the other xsuite tests):
* xsuite cavity ``frequency = harmonic * beta0 * c / circumference``;
* ``phi_rf [rad] = lag [deg] * pi/180`` (here both zero: a stationary bucket);
* BLonD ``BeamObservationOncePerTurn`` with ``each_turn_i = STRIDE`` records at
  ``turns_array[i]`` but *after* that turn's elements, i.e. after
  ``turns_array[i] + 1`` physical turns -- so the xsuite reference is sampled at
  exactly those turn counts.

Set ``DEV_PLOT = True`` (or env ``DEV_PLOT=1``) to render three figures: the
input distribution, the output distribution (both codes overlaid), and the
per-particle distance vs turn.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")
xp = pytest.importorskip("xpart")

from scipy.constants import c  # noqa: E402

from blond import (  # noqa: E402
    Beam,
    BeamObservationOncePerTurn,
    ConstantMagneticCycle,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    copy_to_cpu,
    momentum_compaction_factor,
    proton,
)
from blond.interfaces.xsuite.elements.helpers import (  # noqa: E402
    ReferenceFrame,
    dE_to_ptau,
    dt_to_zeta,
    ptau_to_dE,
    zeta_to_dt,
)
from blond.physics.drifts import _DriftLikeLineSegment  # noqa: E402

# PSB flat-bottom parameters (van Rijswijk 2024, Table 1).
CIRCUMFERENCE = 157.0800
TRANSITION_GAMMA = 4.1164
KINETIC_ENERGY = 1.4e9  # eV
VOLTAGE = 8e3  # V
HARMONIC = 1
LAG_DEG = 0.0  # stationary bucket below transition

ALPHA_0 = momentum_compaction_factor(TRANSITION_GAMMA)
MASS = float(proton.mass)
TOTAL_ENERGY = KINETIC_ENERGY + MASS
MOMENTUM = float(np.sqrt(TOTAL_ENERGY**2 - MASS**2))
BETA0 = MOMENTUM / TOTAL_ENERGY
F_RF = HARMONIC * BETA0 * c / CIRCUMFERENCE
T_RF = 1.0 / F_RF

# Reference frame for the BLonD (dt, dE) <-> xsuite (zeta, ptau) conversions,
# using the same helpers the wrapper relies on.
FRAME = ReferenceFrame(beta0=BETA0, energy0=TOTAL_ENERGY)

# Default is CI-friendly (1e5 turns, ~15 s) and already shows the linear
# round-off growth; set ``BLOND_XSUITE_NTURNS=1000000`` for the full
# million-turn run (the verified result is ~7e-11 normalised distance, dominated
# by floating-point round-off). BLonD's per-turn Python loop makes 1e6 turns
# take a few minutes, hence it is not the default.
N_TURNS = int(os.getenv("BLOND_XSUITE_NTURNS", "100000"))
STRIDE = max(1, N_TURNS // 1000)  # ~1000 comparison points
N_PART = 7

DEV_PLOT = os.getenv("DEV_PLOT", "0") == "1"


def _initial_distribution() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic spread inside the stationary bucket (zeta [m], ptau)."""
    zeta = np.linspace(-0.3, 0.3, N_PART) * (BETA0 * c * T_RF / 2.0)
    ptau = np.linspace(-1.0, 1.0, N_PART) * (3e6 / (BETA0 * TOTAL_ENERGY))
    return zeta, ptau


def _run_blond(zeta0, ptau0):
    """Pure BLonD: RF station + _DriftLikeLineSegment.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(turns, zeta, ptau)`` where ``turns[i]`` is the physical turn count
        of snapshot ``i`` and the coordinate arrays have shape
        ``(n_snapshots, n_part)``.
    """
    ring = Ring(circumference=CIRCUMFERENCE)
    cycle = ConstantMagneticCycle(
        value=MOMENTUM, reference_particle=proton, in_unit="momentum"
    )
    rf_station = SingleHarmonicRFStation(
        voltage=VOLTAGE, phi_rf=np.deg2rad(LAG_DEG), harmonic=HARMONIC
    )
    drift = _DriftLikeLineSegment(
        orbit_length=CIRCUMFERENCE, momentum_compaction_factor=ALPHA_0
    )
    ring.add_elements((rf_station, drift), reorder=False, section_index=0)

    beam = Beam(intensity=1.0, particle_type=proton)
    beam.setup_beam(
        dt=zeta_to_dt(zeta0, FRAME),
        dE=ptau_to_dE(ptau0, FRAME),
        reference_total_energy=TOTAL_ENERGY,
    )

    obs = BeamObservationOncePerTurn(each_turn_i=STRIDE, warn=False)
    sim = Simulation(ring=ring, magnetic_cycle=cycle)
    sim.run_simulation(beams=(beam,), n_turns=N_TURNS, observe=(obs,))

    # dts[i] is recorded after the turn at counter turns_array[i] has fired,
    # i.e. after turns_array[i] + 1 physical turns.
    turns = copy_to_cpu(obs.turns_array) + 1
    dts = copy_to_cpu(obs.dts)
    dEs = copy_to_cpu(obs.dEs)
    zeta = dt_to_zeta(dts, FRAME)
    ptau = dE_to_ptau(dEs, FRAME)
    return turns, zeta, ptau


def _run_xsuite(zeta0, ptau0, turns):
    """Pure xsuite: Cavity + LineSegmentMap, sampled after ``turns`` turns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(zeta, ptau)`` of shape ``(len(turns), n_part)``.
    """
    cav = xt.Cavity(voltage=VOLTAGE, frequency=F_RF, lag=LAG_DEG)
    lsm = xt.LineSegmentMap(
        longitudinal_mode="nonlinear",
        qx=1.1,
        qy=1.2,
        betx=1.0,
        bety=1.0,
        voltage_rf=0.0,
        frequency_rf=0.0,
        lag_rf=0.0,
        momentum_compaction_factor=ALPHA_0,
        length=CIRCUMFERENCE,
    )
    line = xt.Line(elements=[cav, lsm], element_names=["cav", "map"])
    line.particle_ref = xp.Particles(p0c=MOMENTUM, mass0=MASS, q0=1.0)
    line.build_tracker()
    p = line.build_particles(
        x=0, px=0, y=0, py=0, zeta=zeta0.copy(), ptau=ptau0.copy()
    )

    zeta = np.empty((len(turns), N_PART))
    ptau = np.empty((len(turns), N_PART))
    done = 0
    for i, t in enumerate(turns):
        if t > done:
            line.track(p, num_turns=int(t - done))
            done = int(t)
        zeta[i] = copy_to_cpu(p.zeta)
        ptau[i] = copy_to_cpu(p.ptau)
    return zeta, ptau


class TestPSBPureCavityMillionTurns(unittest.TestCase):
    """Pure BLonD must track pure xsuite for 1e6 turns (round-off only)."""

    def test_million_turn_agreement(self):
        """Per-turn machine precision; ~1e-10 round-off drift over 1e6 turns."""
        zeta0, ptau0 = _initial_distribution()

        turns, zb, ptb = _run_blond(zeta0, ptau0)
        zx, ptx = _run_xsuite(zeta0, ptau0, turns)

        zamp = float(np.max(np.abs(zx)))
        pamp = float(np.max(np.abs(ptx)))
        # Per-particle normalised phase-space distance at each snapshot.
        dist = np.sqrt(((zb - zx) / zamp) ** 2 + ((ptb - ptx) / pamp) ** 2)

        print(
            f"\n[{N_TURNS} turns] turn-1 dist={dist[0].max():.2e}  "
            f"final dist={dist[-1].max():.2e}  max dist={dist.max():.2e}"
        )

        if DEV_PLOT:  # pragma: no cover - manual dev aid
            self._plot(zeta0, ptau0, zx, ptx, zb, ptb, turns, dist, zamp, pamp)

        # First snapshot (after one turn) is at machine precision.
        self.assertLess(float(dist[0].max()), 1e-12)
        # Over 1e6 turns only floating-point round-off accumulates (~1e-10);
        # 1e-7 leaves generous margin across backends / FP environments.
        self.assertLess(float(dist.max()), 1e-7)

    @staticmethod
    def _plot(zeta0, ptau0, zx, ptx, zb, ptb, turns, dist, zamp, pamp):
        # pragma: no cover
        import matplotlib.pyplot as plt

        # 1) input coordinates
        plt.figure("input coordinates")
        plt.scatter(zeta0, ptau0, c=range(N_PART), cmap="viridis")
        plt.xlabel(r"$\zeta$ [m]")
        plt.ylabel(r"$p_\tau$")
        plt.title("Input coordinates")
        plt.grid(True, alpha=0.3)

        # 2) output coordinates (both codes overlaid)
        plt.figure("output coordinates")
        plt.scatter(zx[-1], ptx[-1], marker="x", label="xsuite", c="C0")
        plt.scatter(
            zb[-1],
            ptb[-1],
            marker="o",
            facecolors="none",
            edgecolors="C1",
            label="BLonD",
        )
        plt.xlabel(r"$\zeta$ [m]")
        plt.ylabel(r"$p_\tau$")
        plt.title(f"Output coordinates after {turns[-1]} turns")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3) per-particle distance vs turn
        plt.figure("distance per particle")
        for k in range(N_PART):
            plt.semilogy(turns, dist[:, k] + 1e-300, label=f"particle {k}")
        plt.xlabel("turn")
        plt.ylabel("normalised phase-space distance")
        plt.title("BLonD vs xsuite distance per particle")
        # The synchronous particle sits at the origin (distance exactly 0);
        # clamp the range to the round-off band of the moving particles.
        plt.ylim(1e-17, 1e-8)
        plt.legend(ncol=2, fontsize="small")
        plt.grid(True, which="both", alpha=0.3)
        plt.show()


if __name__ == "__main__":
    unittest.main()
