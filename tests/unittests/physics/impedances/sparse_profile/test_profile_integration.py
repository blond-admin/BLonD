import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest

import blond
from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Numpy64Bit,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
    make_multibunch_beam,
    momentum_compaction_factor,
    proton,
)
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.profiles_sparse import EquidistantMultiProfile

resonator_data = np.loadtxt(
    os.path.join(
        os.path.dirname(blond.__file__),
        "examples/scripts/resources/EX_05_new_HQ_table.txt",
    ),
    comments="!",
)
sync_momentum = 25.92e9  # [eV / c]

R_shunt = resonator_data[:, 2] * 10**6
f_res = resonator_data[:, 0] * 10**9
Q_factor = resonator_data[:, 1]


class TestSparseProfileIntegration(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_compare_both_profiles(self):
        backend.change_backend(Numpy64Bit)
        DEV_DRAW = False

        profile, profile_wanted = self._exec_full_sim_with_profiles()
        self._test_both_parameters_equal(profile, profile_wanted)

        if DEV_DRAW:
            plt.figure("compare")
            ax1 = plt.subplot(3, 1, 1)
            plt.xlim(4e-8, 6e-8)
            plt.plot(
                profile._continuous_memory_hist_x,
                profile._continuous_memory_hist_y,
                "o",
            )

        if DEV_DRAW:
            plt.figure("compare")
            ax1 = plt.subplot(3, 1, 1)
            plt.plot(profile_wanted._hist_x, profile_wanted._hist_y, "x")
            plt.xlim(4e-8, 6e-8)
            plt.axvline(4.9940e-8)
            plt.show()
        self._test_both_results_equal(profile, profile_wanted)

    def _test_both_results_equal(self, profile, profile_wanted):
        # from plot, see `axvline`
        start_idx = np.argmax(profile_wanted._hist_x > 4.9940e-8)
        second_peak_wanted = profile_wanted._hist_y[
            start_idx : start_idx + 2**8
        ]
        second_peak_actual = profile.profiles[1].hist_y
        np.testing.assert_array_equal(second_peak_actual, second_peak_wanted)

    def _test_both_parameters_equal(
        self, profile: EquidistantMultiProfile, profile_wanted: StaticProfile
    ):
        self.assertAlmostEqual(
            profile.profiles[0].cut_left, profile_wanted.cut_left
        )

        self.assertAlmostEqual(
            profile.profiles[-1].cut_right, profile_wanted.cut_right
        )

    def _exec_full_sim_with_profiles(
        self,
    ) -> (EquidistantMultiProfile, StaticProfile):
        ring = Ring(
            circumference=6911.56,
        )
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=sync_momentum,
            in_unit="momentum",
        )
        _bunch = Beam(
            intensity=1e10,
            particle_type=proton,
        )
        drift = DriftSimple(
            momentum_compaction_factor=momentum_compaction_factor(
                22.82177322938192
            ),
            orbit_length=1.0 * ring.circumference,
        )
        rf_station = SingleHarmonicRFStation(
            harmonic=4620,
            voltage=0.9e6,
            phi_rf=0.0,
        )
        t_rf = (
            magnetic_cycle.get_t_rev_init(
                ring.circumference,
                particle_type=proton,
            )
            / rf_station.harmonic
        )
        filling_pattern = np.zeros(rf_station.harmonic, bool)
        filling_pattern[::10] = 1

        profile = EquidistantMultiProfile(
            filling_pattern=filling_pattern,
            bins_per_profile=2**8,
            offset=0,
        )
        profile_wanted = StaticProfile.from_rad(
            0,
            2 * np.pi,
            2**8 * 4620,
            magnetic_cycle.get_t_rev_init(
                ring.circumference,
                particle_type=proton,
            ),
        )
        ring.add_elements((profile, profile_wanted, rf_station, drift))
        sim = Simulation(
            ring=ring,
            magnetic_cycle=magnetic_cycle,
        )

        sim.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=2e-9 / 4,
                seed=1,
                n_macroparticles=1e4,
            ),
            beam=_bunch,
        )

        beam = make_multibunch_beam(
            beam=_bunch,
            n_times=int(rf_station.harmonic // 10),
            t_distance=t_rf * 10,
        )
        drift.orbit_length = 0
        rf_station.voltage = 0.0
        sim.check_circumference = "ignore"

        sim.run_simulation(beams=beam, n_turns=1)
        return profile, profile_wanted

    @pytest.mark.backend_mutation
    def test_induced_voltage_track_with_gapped_filling_pattern_does_not_raise(
        self,
    ):
        # Build the same simulation as _exec_full_sim_with_profiles, but
        # swap the filling pattern used for the EquidistantMultiProfile to
        # one with a genuine internal gap (fill/fill/empty/empty, repeated,
        # not merely a shorter contiguous run followed by trailing zeros),
        # attach a real resonator impedance, then run
        # `WakeField._track()` and confirm it completes without the
        # `ValueError` raised by `kick_interpolated`'s uniform-spacing
        # guard, and that a non-zero kick was actually applied.
        backend.change_backend(Numpy64Bit)

        ring = Ring(
            circumference=6911.56,
        )
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=sync_momentum,
            in_unit="momentum",
        )
        _bunch = Beam(
            intensity=1e10,
            particle_type=proton,
        )
        drift = DriftSimple(
            momentum_compaction_factor=momentum_compaction_factor(
                22.82177322938192
            ),
            orbit_length=1.0 * ring.circumference,
        )
        rf_station = SingleHarmonicRFStation(
            harmonic=4620,
            voltage=0.9e6,
            phi_rf=0.0,
        )
        t_rf = (
            magnetic_cycle.get_t_rev_init(
                ring.circumference,
                particle_type=proton,
            )
            / rf_station.harmonic
        )

        # Genuine internal gap: two filled buckets, two empty buckets,
        # repeated across the whole ring -- as opposed to one contiguous
        # run of filled buckets followed by zero-padding at the end.
        bucket_index = np.arange(rf_station.harmonic)
        filling_pattern = (bucket_index % 4) < 2

        profile = EquidistantMultiProfile(
            filling_pattern=filling_pattern,
            bins_per_profile=2**8,
            offset=0,
        )
        wakefield = WakeField(
            sources=(Resonators(R_shunt, f_res, Q_factor),),
            solver=MultiPoleSparseSolve(),
            profile=profile,
        )
        ring.add_elements((wakefield, rf_station, drift), reorder=True)
        sim = Simulation(
            ring=ring,
            magnetic_cycle=magnetic_cycle,
        )

        sim.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=2e-9 / 4,
                seed=1,
                n_macroparticles=1e4,
            ),
            beam=_bunch,
        )

        beam = make_multibunch_beam(
            beam=_bunch,
            n_times=int(rf_station.harmonic // 10),
            t_distance=t_rf * 10,
        )
        dE_before = beam.dE.copy_as_numpy()

        # Isolate the wakefield's kick: no drift, no RF kick, so any
        # change in `dE` after tracking can only come from
        # `WakeField._track()` -> `kick_interpolated`.
        drift.orbit_length = 0
        rf_station.voltage = 0.0
        sim.check_circumference = "ignore"

        sim.run_simulation(beams=beam, n_turns=1)

        dE_after = beam.dE.copy_as_numpy()
        self.assertFalse(
            np.allclose(dE_before, dE_after),
            "Expected the sparse-aware induced-voltage kick to "
            "change dE, but dE was unchanged.",
        )

    @pytest.mark.backend_mutation
    def test_multiturn_gapped_matches_dense_single_bucket(self):
        # End-to-end regression test for the sparse `kick_interpolated`
        # fix: a multi-turn simulation of one physical bunch, sitting
        # in a `EquidistantMultiProfile` whose filling pattern has
        # several OTHER buckets structurally allocated (creating real
        # gaps in the concatenated `hist_x`/`hist_y` memory) but no
        # macroparticles in them, must produce bit-for-bit-equivalent
        # (to tight rtol) induced-voltage kick physics as the same
        # bunch tracked in complete isolation with a plain, fully
        # dense `StaticProfile` covering only its own bucket. Because
        # the "extra" buckets carry zero charge, there is no genuine
        # inter-bunch wake cross-talk to confound the comparison --
        # any mismatch can only come from how the sparse profile's
        # gapped memory is indexed.
        backend.change_backend(Numpy64Bit)
        n_turns = 10

        dt_sparse, dE_sparse = self._exec_multiturn_sparse_sim(n_turns)
        dt_dense, dE_dense = self._exec_multiturn_dense_sim(n_turns)

        np.testing.assert_allclose(dt_sparse, dt_dense, rtol=1e-8)
        np.testing.assert_allclose(dE_sparse, dE_dense, rtol=1e-8)

    def _build_ring_rf_drift(self):
        ring = Ring(
            circumference=6911.56,
        )
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=sync_momentum,
            in_unit="momentum",
        )
        drift = DriftSimple(
            momentum_compaction_factor=momentum_compaction_factor(
                22.82177322938192
            ),
            orbit_length=1.0 * ring.circumference,
        )
        rf_station = SingleHarmonicRFStation(
            harmonic=4620,
            voltage=0.9e6,
            phi_rf=0.0,
        )
        return ring, magnetic_cycle, drift, rf_station

    def _exec_multiturn_sparse_sim(
        self, n_turns: int
    ) -> tuple[np.ndarray, np.ndarray]:
        ring, magnetic_cycle, drift, rf_station = self._build_ring_rf_drift()

        # Genuine structural gaps: several buckets are allocated as
        # profile "islands" in the sparse representation, but only
        # the first (bucket 0) is physically populated -- this
        # isolates the array-indexing correctness of the sparse kick
        # (the historical bug) from any real inter-bunch wake
        # cross-talk, which a solo single-bucket simulation could
        # never reproduce.
        harmonic = rf_station.harmonic
        filling_pattern = np.zeros(harmonic, bool)
        filling_pattern[[0, 700, 1500, 2600, 3800]] = True

        profile = EquidistantMultiProfile(
            filling_pattern=filling_pattern,
            bins_per_profile=2**8,
            offset=0,
        )
        wakefield = WakeField(
            sources=(Resonators(R_shunt, f_res, Q_factor),),
            solver=MultiPoleSparseSolve(),
            profile=profile,
        )
        ring.add_elements((wakefield, rf_station, drift), reorder=True)
        sim = Simulation(
            ring=ring,
            magnetic_cycle=magnetic_cycle,
        )

        beam = Beam(
            intensity=1e10,
            particle_type=proton,
        )
        sim.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=2e-9 / 4,
                seed=1,
                n_macroparticles=1e4,
            ),
            beam=beam,
        )

        sim.run_simulation(beams=beam, n_turns=n_turns)

        return beam.dt.copy_as_numpy(), beam.dE.copy_as_numpy()

    def _exec_multiturn_dense_sim(
        self, n_turns: int
    ) -> tuple[np.ndarray, np.ndarray]:
        ring, magnetic_cycle, drift, rf_station = self._build_ring_rf_drift()

        # Exactly one RF period wide, matching bucket 0's
        # `cut_left`/`cut_right` in `EquidistantMultiProfile`
        # (`profile_width = t_rev / n_slots`, `starts[0] = offset`).
        t_rev = magnetic_cycle.get_t_rev_init(
            ring.circumference,
            particle_type=proton,
        )
        cut_left = 0.0
        cut_right = t_rev / rf_station.harmonic

        profile = StaticProfile(
            cut_left=cut_left,
            cut_right=cut_right,
            n_bins=2**8,
        )
        wakefield = WakeField(
            sources=(Resonators(R_shunt, f_res, Q_factor),),
            solver=MultiPoleSparseSolve(),
            profile=profile,
        )
        ring.add_elements((wakefield, rf_station, drift), reorder=True)
        sim = Simulation(
            ring=ring,
            magnetic_cycle=magnetic_cycle,
        )

        beam = Beam(
            intensity=1e10,
            particle_type=proton,
        )
        sim.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=2e-9 / 4,
                seed=1,
                n_macroparticles=1e4,
            ),
            beam=beam,
        )

        sim.run_simulation(beams=beam, n_turns=n_turns)

        return beam.dt.copy_as_numpy(), beam.dE.copy_as_numpy()


if __name__ == "__main__":
    unittest.main()
