# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
PSB flat-bottom cross-check: BLonD ``_DriftLikeLineSegment`` vs xsuite.

The benchmark report (van Rijswijk, Karlsen-Baeck and Timko, "Creating and
benchmarking an interface between Xsuite and BLonD", CERN, 2024) found that at
the Proton Synchrotron Booster flat bottom BLonD's ``DriftSimple`` drifts
differently from an xsuite ``LineSegmentMap``. The cause is the drift model:
``LineSegmentMap`` uses the *linear* slip factor :math:`\\eta_0` with the
*exact* relativistic :math:`\\delta(dE)`, whereas ``DriftSimple`` also
linearises :math:`\\delta`. ``_DriftLikeLineSegment`` is the BLonD-native drift
with exactly xsuite's combination (linear slip + exact :math:`\\delta`), so the
two must agree to machine precision.

To isolate the drift, both runs share the *same* BLonD RF kick and
constant-energy program; they differ only in the drift element:

* Reference (X): the longitudinal drift is an ``xt.LineSegmentMap`` (RF off)
  wrapped by :class:`WrapXsuite4Blond`.
* Under test (B): BLonD's :class:`_DriftLikeLineSegment`.

A single particle is launched with an energy offset inside the stationary
bucket and tracked for a large fraction of a synchrotron period. The two
phase-space trajectories must coincide to ~1e-9 of the amplitude. (For
contrast, ``DriftSimple`` would diverge by ~1e-4 here, and ``DriftExact`` --
which uses the *full* slip factor -- by ~1e-2; neither equals xsuite.)

Set ``DEV_PLOT = True`` to render comparison plots when run directly.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")
pytest.importorskip("xpart")

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
from blond.interfaces.xsuite.elements import WrapXsuite4Blond  # noqa: E402
from blond.physics.drifts import _DriftLikeLineSegment  # noqa: E402

# PSB flat-bottom parameters (van Rijswijk 2024, Table 1).
CIRCUMFERENCE = 157.0800
TRANSITION_GAMMA = 4.1164
KINETIC_ENERGY = 1.4e9  # eV
VOLTAGE = 8e3  # V
HARMONIC = 1

ALPHA_0 = momentum_compaction_factor(TRANSITION_GAMMA)
MASS = float(proton.mass)
TOTAL_ENERGY = KINETIC_ENERGY + MASS
MOMENTUM = float(np.sqrt(TOTAL_ENERGY**2 - MASS**2))

# ~half a synchrotron period: long enough that any per-turn drift-model
# difference would accumulate into a resolvable phase-space difference.
N_TURNS = 2000
DE_OFFSET = 2.0e6  # eV, well inside the ~10 MeV stationary bucket

DEV_PLOT = False


def _segment_map() -> xt.LineSegmentMap:
    """PSB longitudinal map with RF disabled: a pure drift (linear slip)."""
    return xt.LineSegmentMap(
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


def _run(use_xsuite_drift: bool):
    """Track one off-energy particle for ``N_TURNS`` at the PSB flat bottom.

    Both branches use the identical BLonD RF station and constant-energy
    program; only the drift element differs.

    Parameters
    ----------
    use_xsuite_drift
        If ``True`` the drift is an xsuite ``LineSegmentMap`` (wrapped);
        otherwise BLonD's ``_DriftLikeLineSegment``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Per-turn ``(dt, dE)`` of the single particle.
    """
    ring = Ring(circumference=CIRCUMFERENCE)
    cycle = ConstantMagneticCycle(
        value=MOMENTUM, reference_particle=proton, in_unit="momentum"
    )
    rf_station = SingleHarmonicRFStation(
        voltage=VOLTAGE, phi_rf=0.0, harmonic=HARMONIC
    )
    if use_xsuite_drift:
        drift = WrapXsuite4Blond(
            xt.Line(elements=[_segment_map()], element_names=["matrix"])
        )
    else:
        drift = _DriftLikeLineSegment(
            orbit_length=CIRCUMFERENCE, momentum_compaction_factor=ALPHA_0
        )
    ring.add_elements((rf_station, drift), reorder=False, section_index=0)

    beam = Beam(intensity=1.0, particle_type=proton)
    beam.setup_beam(
        dt=np.array([0.0]),
        dE=np.array([DE_OFFSET]),
        reference_total_energy=TOTAL_ENERGY,
    )

    obs = BeamObservationOncePerTurn(each_turn_i=1, warn=False)
    sim = Simulation(ring=ring, magnetic_cycle=cycle)
    sim.run_simulation(beams=(beam,), n_turns=N_TURNS, observe=(obs,))
    return copy_to_cpu(obs.dts)[:, 0], copy_to_cpu(obs.dEs)[:, 0]


class TestPSBFlatBottomDrift(unittest.TestCase):
    """``_DriftLikeLineSegment`` must equal an xsuite drift on the PSB."""

    def test_matches_xsuite_line_segment_map(self):
        """Same RF kick; drift via _DriftLikeLineSegment vs xsuite.

        The two drifts are the same physical model (linear slip + exact
        delta), so the trajectories coincide to ~machine precision over a
        synchrotron period at the PSB flat bottom.
        """
        dt_b, dE_b = _run(use_xsuite_drift=False)  # _DriftLikeLineSegment
        dt_x, dE_x = _run(use_xsuite_drift=True)  # xsuite LineSegmentMap

        if DEV_PLOT:  # pragma: no cover - manual dev aid
            import matplotlib.pyplot as plt

            fig, (a, b) = plt.subplots(2, 1, sharex=True)
            a.plot(dE_x, label="xsuite drift")
            a.plot(dE_b, "--", label="_DriftLikeLineSegment")
            a.set_ylabel(r"$dE$ [eV]")
            a.legend()
            b.semilogy(np.abs(dE_b - dE_x) / np.max(np.abs(dE_x)) + 1e-300)
            b.set_ylabel(r"norm. $|\Delta dE|$")
            b.set_xlabel("turn")
            plt.show()

        # Amplitude-normalised tolerance: the two are the same drift model, so
        # they agree far better than DriftSimple (~1e-4) or DriftExact (~1e-2).
        amp_dt = float(np.max(np.abs(dt_x)))
        amp_dE = float(np.max(np.abs(dE_x)))
        np.testing.assert_allclose(
            dt_b,
            dt_x,
            atol=amp_dt * 1e-9,
            rtol=0.0,
            err_msg="_DriftLikeLineSegment dt must match xsuite "
            "LineSegmentMap.",
        )
        np.testing.assert_allclose(
            dE_b,
            dE_x,
            atol=amp_dE * 1e-9,
            rtol=0.0,
            err_msg="_DriftLikeLineSegment dE must match xsuite "
            "LineSegmentMap.",
        )


if __name__ == "__main__":
    unittest.main()
