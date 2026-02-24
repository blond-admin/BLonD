import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

import blond
from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MagneticCyclePerTurn,
    Numpy32Bit,
    Numpy64Bit,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    TimeDomainFftSolver,
    WakeField,
    backend,
    make_multibunch_beam,
    momentum_compaction_factor,
    proton,
)
from blond.physics.profiles_sparse import EquidistantMultiProfile

resonator_data = np.loadtxt(
    os.path.join(
        os.path.dirname(blond.__file__),
        "examples/resources/EX_05_new_HQ_table.txt",
    ),
    comments="!",
)
sync_momentum = 25.92e9  # [eV / c]

R_shunt = resonator_data[:, 2] * 10**6
f_res = resonator_data[:, 0] * 10**9
Q_factor = resonator_data[:, 1]


class MyTestCase(unittest.TestCase):
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
            n_times=int((rf_station.harmonic // 10)),
            t_distance=t_rf * 10,
        )
        drift.orbit_length = 0
        rf_station.voltage = 0.0
        sim.check_circumference = "ignore"

        sim.run_simulation(beams=beam, n_turns=1)
        return profile, profile_wanted


if __name__ == "__main__":
    unittest.main()
