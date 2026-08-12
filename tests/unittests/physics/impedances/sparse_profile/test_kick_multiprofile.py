"""Regression tests: wakefield kick on a gapped EquidistantMultiProfile.

``WakeField._track`` interpolates the induced voltage onto the particles
with ``kick_interpolated``, which assumes ONE equidistant grid of bin
centers. ``EquidistantMultiProfile.hist_x`` is the concatenation of the
filled buckets only -- a gapped grid -- so the kick must be applied via
the packed (gap-free) coordinate mapping instead. Applying the raw
concatenated grid mis-kicks every particle (wrong assumed bin width).
"""

import unittest

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    WakeField,
    backend,
    make_multibunch_beam,
    momentum_compaction_factor,
    proton,
)
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.profiles_sparse import EquidistantMultiProfile

CIRCUMFERENCE = 6911.56
TRANSITION_GAMMA = 22.82177322938192
HARMONIC = 4620
SYNC_MOMENTUM = 25.92e9  # [eV/c]
BUNCH_INTENSITY = 1e11
BINS_PER_PROFILE = 64
N_MACROPARTICLES = 3000
BUNCH_SPACING = 5  # buckets between the two bunches
SEED = 12


class TestPackedKickMapping(unittest.TestCase):
    """The packed mapping must reproduce per-segment interpolation exactly."""

    def test_mapping_matches_per_segment_kick(self):
        n_slots = 8
        t_rev = n_slots * 5e-9
        bins_per_profile = 16
        filling_pattern = np.zeros(n_slots, dtype=bool)
        filling_pattern[0] = True
        filling_pattern[5] = True

        profile = EquidistantMultiProfile.headless(
            t_rev=t_rev,
            filling_pattern=filling_pattern,
            bins_per_profile=bins_per_profile,
        )

        rng = np.random.default_rng(3)
        n_bins_total = 2 * bins_per_profile
        voltage = backend.array(
            1e6 * rng.standard_normal(n_bins_total), dtype=backend.float
        )
        # particles everywhere: inside both buckets (incl. the edge
        # half-bins), in the gaps, and outside the grid
        dt = backend.array(
            np.sort(rng.uniform(-2e-9, t_rev + 2e-9, 5000)),
            dtype=backend.float,
        )

        # reference: each segment kicked in isolation on its own
        # (contiguous) grid -- the defining semantics of the multi profile
        dE_expected = backend.zeros(len(dt), dtype=backend.float)
        for i, segment in enumerate(profile.profiles):
            backend.specials.kick_interpolated(
                dt=dt,
                dE=dE_expected,
                voltage=voltage[
                    i * bins_per_profile : (i + 1) * bins_per_profile
                ],
                bin_centers=segment.hist_x,
                charge=1.0,
                acceleration_kick=0.0,
            )

        dE_actual = backend.zeros(len(dt), dtype=backend.float)
        backend.specials.kick_interpolated(
            dt=profile.map_dt_to_packed(dt),
            dE=dE_actual,
            voltage=voltage,
            bin_centers=profile.packed_hist_x,
            charge=1.0,
            acceleration_kick=0.0,
        )

        np.testing.assert_allclose(
            dE_actual,
            dE_expected,
            rtol=0.0,
            atol=1e-9 * float(np.max(np.abs(dE_expected))),
        )


class TestWakeFieldKickMultiProfile(unittest.TestCase):
    """A bunch must get the same wake kick alone and inside a pattern.

    With a fast-decaying resonator (Q=1 at 1 GHz, wake dead after ~2 ns)
    two bunches 5 buckets (25 ns) apart cannot influence each other within
    one turn, so each bunch of the two-bunch beam must receive exactly the
    kick the single bunch receives.
    """

    def _wake_kick_one_turn(self, filling_pattern, n_bunches):
        ring = Ring(circumference=CIRCUMFERENCE)
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=SYNC_MOMENTUM,
            in_unit="momentum",
        )
        drift = DriftSimple(
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=TRANSITION_GAMMA
            ),
            orbit_length=CIRCUMFERENCE,
        )
        rf_station = SingleHarmonicRFStation(
            harmonic=HARMONIC, voltage=0.9e6, phi_rf=0.0
        )
        t_rf = (
            magnetic_cycle.get_t_rev_init(
                ring.circumference, particle_type=proton
            )
            / HARMONIC
        )
        profile = EquidistantMultiProfile(
            filling_pattern=filling_pattern,
            bins_per_profile=BINS_PER_PROFILE,
        )
        wakefield = WakeField(
            sources=(Resonators(1e6, 1e9, 1.0),),
            solver=MultiPoleSparseSolve(),
            profile=profile,
        )
        ring.add_elements((drift, rf_station, wakefield))
        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)

        bunch = Beam(intensity=BUNCH_INTENSITY, particle_type=proton)
        sim.prepare_beam(
            beam=bunch,
            preparation_routine=BiGaussian(
                sigma_dt=0.5e-9,
                seed=SEED,
                n_macroparticles=N_MACROPARTICLES,
            ),
        )
        beam = make_multibunch_beam(
            beam=bunch,
            n_times=n_bunches,
            t_distance=BUNCH_SPACING * t_rf,
        )
        # freeze everything but the wake kick
        rf_station.voltage = 0.0
        drift.orbit_length = 0.0
        sim.check_circumference = "ignore"

        dE_before = np.array(beam.dE.array_local, dtype=float)
        sim.run_simulation(beams=(beam,), n_turns=1)
        dE_after = np.array(beam.dE.array_local, dtype=float)
        return dE_after - dE_before

    def test_bunch_kick_independent_of_pattern(self):
        single = np.zeros(HARMONIC, dtype=bool)
        single[0] = True
        kick_alone = self._wake_kick_one_turn(single, n_bunches=1)
        self.assertGreater(np.max(np.abs(kick_alone)), 0.0)

        double = np.zeros(HARMONIC, dtype=bool)
        double[0] = True
        double[BUNCH_SPACING] = True
        kick_pattern = self._wake_kick_one_turn(double, n_bunches=2)

        atol = 1e-9 * np.max(np.abs(kick_alone))
        # make_multibunch_beam interleaves the copies
        np.testing.assert_allclose(
            kick_pattern[0::2],
            kick_alone,
            rtol=0.0,
            atol=atol,
            err_msg="leading bunch kick differs from the isolated bunch",
        )
        np.testing.assert_allclose(
            kick_pattern[1::2],
            kick_alone,
            rtol=0.0,
            atol=atol,
            err_msg="trailing bunch kick differs from the isolated bunch",
        )


if __name__ == "__main__":
    unittest.main()
