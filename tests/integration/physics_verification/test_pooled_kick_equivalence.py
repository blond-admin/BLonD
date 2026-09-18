# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Pooling an RF kick must not change the beam it produces.

`PooledInterpolationKick` exists to save a pass over the particle arrays
when several elements kick on the same time axis: instead of each one
touching `dE`, they register their voltage and a single element applies
the sum. It is a performance device, so a simulation must end up with the
same beam whether or not the kick is pooled.

The two paths are not bit-identical -- the unpooled `RFStation` evaluates
its sine exactly, while the pooled one interpolates on a sampled axis --
so this compares against an interpolation-error budget rather than for
equality. That budget is thousands of times smaller than the errors it is
written to catch: with the reference energy change scaled by the charge,
the lead case below disagreed by 1.7e9 eV, against 0.3 eV once fixed.

Accelerating is essential here. With a constant magnetic cycle the
reference energy change is zero every turn and both paths agree whatever
they do with it, which is why the unit tests in
`tests/unittests/physics/test_kick_pooling.py` pin the kernel arithmetic
directly as well.

Authors: Simon Lauber
"""

import unittest

import numpy as np
import pytest

from blond import (
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    PooledInterpolationKick,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    momentum_compaction_factor,
)
from blond.core.beam.particle_types import lead_82, proton
from blond.testing.backend_testing import BLonDTestCase

CIRCUMFERENCE = 6911.56
HARMONIC = 4620
N_TURNS = 3
N_MACROPARTICLES = 2000
TRANSITION_GAMMA = 22.82177322938192
SPEED_OF_LIGHT = 299792458.0

# Momentum per unit of proton mass, so every species is launched at the
# same beta and therefore sees the same bucket and revolution time.
MOMENTUM_PER_PROTON_MASS = 25.92e9
MOMENTUM_RAMP_PER_PROTON_MASS = 25.9201e9

# Voltage per unit charge: the kernel applies `charge * voltage`, so
# holding `charge * voltage` fixed keeps the synchrotron motion -- and
# hence the bucket the beam has to stay inside -- the same for every
# species.
VOLTAGE_PER_CHARGE = 0.9e6

# Interpolation-error budget, in [eV]. Measured worst case across the
# cases below is 58 eV; the defects this guards against are seven orders
# of magnitude larger.
MAX_ENERGY_DEVIATION_EV = 5.0e3


def _run_simulation(pooling: bool, particle_type) -> tuple:
    """Track a beam through one RF station, pooled or not.

    Parameters
    ----------
    pooling
        Whether the RF station hands its kick to a
        `PooledInterpolationKick` instead of applying it directly.
    particle_type
        Particle species. Its charge and mass scale the voltage and
        momentum so that every species follows the same trajectory.

    Returns
    -------
    tuple of numpy.ndarray
        Final `dt` in [s] and `dE` in [eV], on the host.
    """
    mass_ratio = particle_type.mass / proton.mass
    ring = Ring(circumference=CIRCUMFERENCE)

    # A ramp, not a constant cycle: a zero reference energy change would
    # make both paths agree no matter how they treat it.
    magnetic_cycle = MagneticCyclePerTurn(
        reference_particle=particle_type,
        value_init=MOMENTUM_PER_PROTON_MASS * mass_ratio,
        values_after_turn=np.linspace(
            MOMENTUM_PER_PROTON_MASS * mass_ratio,
            MOMENTUM_RAMP_PER_PROTON_MASS * mass_ratio,
            N_TURNS,
        ),
        in_unit="momentum",
    )
    drift = DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=TRANSITION_GAMMA
        ),
        orbit_length=ring.circumference,
    )
    rf_period = (
        magnetic_cycle.get_t_rev_init(
            ring.circumference, particle_type=particle_type
        )
        / HARMONIC
    )
    profile = StaticProfile.from_rad(0, 2 * np.pi, 2**12, rf_period)

    pooled_kick = PooledInterpolationKick()
    rf_station = SingleHarmonicRFStation(
        harmonic=HARMONIC,
        voltage=VOLTAGE_PER_CHARGE / particle_type.charge,
        phi_rf=0.0,
        delayed_kick=pooled_kick if pooling else None,
        delayed_kick_time_axis=profile.hist_x if pooling else None,
    )

    # Built directly rather than matched: a matched distribution needs
    # `phi_s`, and the point here is that both runs start identically,
    # not that the bunch is in equilibrium. The spread is kept narrow so
    # the beam stays well inside the interpolation axis for the few
    # turns tracked.
    beam = Beam.simple_gaussian(
        n_macroparticles=N_MACROPARTICLES,
        intensity=1e10,
        particle_type=particle_type,
        dt_scale=2e-11,
        dE_scale=1e4,
        dt_offset=CIRCUMFERENCE / SPEED_OF_LIGHT / HARMONIC / 2,
        seed=1,
    )

    ring.add_elements((drift, rf_station), reorder=True)
    if pooling:
        # The pool applies what was registered, so it cannot be ordered
        # automatically -- it has to come after the elements feeding it.
        ring.add_element(pooled_kick)

    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    simulation.run_simulation(beams=beam, n_turns=N_TURNS)
    return beam.dt.copy_as_numpy(), beam.dE.copy_as_numpy()


@pytest.mark.integration
class TestPooledKickMatchesUnpooledRFStation(BLonDTestCase):
    """An accelerating RF station gives the same beam pooled or not."""

    def _assert_same_beam(self, particle_type):
        """Compare a pooled and an unpooled run of the same simulation.

        Parameters
        ----------
        particle_type
            Particle species to run both ways.
        """
        dt_unpooled, dE_unpooled = _run_simulation(False, particle_type)
        dt_pooled, dE_pooled = _run_simulation(True, particle_type)

        energy_deviation = np.abs(dE_pooled - dE_unpooled).max()
        self.assertLess(
            energy_deviation,
            MAX_ENERGY_DEVIATION_EV,
            f"Pooling changed the beam energy by {energy_deviation:.3e} eV "
            f"for {particle_type.charge:.0f}-charged particles, beyond the "
            f"{MAX_ENERGY_DEVIATION_EV:.0e} eV interpolation budget.",
        )
        # `dt` only moves through `dE` over one drift, so it is the
        # weaker of the two checks, but it catches a kick applied at the
        # wrong point of the turn.
        np.testing.assert_allclose(dt_pooled, dt_unpooled, rtol=0, atol=1e-11)

    def test_singly_charged_beam(self):
        self._assert_same_beam(proton)

    def test_highly_charged_ion(self):
        """Charge 82 separates `charge * dE_ref` from `dE_ref`.

        For a proton the two are equal, so this case is what makes the
        comparison able to see the reference energy change being scaled
        by the charge.
        """
        self._assert_same_beam(lead_82)


if __name__ == "__main__":
    unittest.main()
