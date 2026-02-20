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
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    TimeDomainFftSolver,
    WakeField,
    backend,
    make_multibunch_beam,
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
    def compare(
        self, profile: EquidistantMultiProfile, profile_wanted: StaticProfile
    ):
        self.assertAlmostEqual(
            profile.profiles[0].cut_left, profile_wanted.cut_left
        )

        self.assertAlmostEqual(
            profile.profiles[-1].cut_right, profile_wanted.cut_right
        )

    def test_something(self):
        self.fail("TODO")
        for induces_voltage in (None,):  # TODO
            profile, profile_wanted = self.multiturn(
                induced_voltage=induces_voltage
            )
            self.compare(profile, profile_wanted)

            DEV_DRAW = True
            if DEV_DRAW:
                plt.figure("compare")
                ax1 = plt.subplot(3, 1, 1)
                plt.xlim(4e-8, 6e-8)
                plt.plot(
                    profile._continuous_memory_hist_x,
                    profile._continuous_memory_hist_y,
                    "o",
                )

            DEV_DRAW = True
            if DEV_DRAW:
                plt.figure("compare")
                ax1 = plt.subplot(3, 1, 1)
                plt.plot(profile_wanted._hist_x, profile_wanted._hist_y, "x")
                plt.xlim(4e-8, 6e-8)

                plt.show()

    def non_sparse_fake_multiturn(self, induced_voltage) -> WakeField:
        FAKE_TUNRS = 1
        wake_solver = TimeDomainFftSolver(allow_next_fast_len=False)
        ring = Ring(
            circumference=6911.56,
        )
        magnetic_cycle = MagneticCyclePerTurn(
            reference_particle=proton,
            values_after_turn=np.linspace(sync_momentum, sync_momentum, 2),
            value_init=sync_momentum,
            in_unit="momentum",
        )
        _bunch = Beam(
            intensity=1e10,
            particle_type=proton,
        )
        drift = DriftSimple(
            transition_gamma=22.82177322938192,
            orbit_length=1.0 * ring.circumference,
        )
        rf_station = SingleHarmonicRFStation(
            harmonic=4620,
            voltage=0.9e6,
            phi_rf=0.0,
        )

        ring.add_elements((profile, rf_station, drift))
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
        omega = rf_station.calc_omega(
            beam_beta=_bunch.reference.beta,
            ring_circumference=ring.circumference,
        )
        t_rf = 1 / (omega / (2 * np.pi))
        beam = make_multibunch_beam(
            beam=_bunch,
            n_times=int((rf_station.harmonic // 10) * FAKE_TUNRS),
            t_distance=t_rf * 10,
        )

        sim.run_simulation(beams=beam, n_turns=1)

        return profile

    def multiturn(self, induced_voltage) -> WakeField:
        backend.set_specials("cpp")  # TODO remove
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
            transition_gamma=22.82177322938192,
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
