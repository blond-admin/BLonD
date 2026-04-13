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
    Resonators,
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
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.profiles import ProfileBaseClass
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

n_macroparticles = int(1e4)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        for induces_voltage in (None,):  # TODO
            wakefield = self.multiturn(induced_voltage=induces_voltage)
            profile: EquidistantMultiProfile = wakefield.profile
            DEV_DRAW = True
            if DEV_DRAW:
                plt.figure("compare")
                ax1 = plt.subplot(3, 1, 1)
                plt.xlim(4e-8, 6e-8)
                profile.plot(marker="x")
                plt.subplot(3, 1, 2, sharex=ax1)
                plt.plot(
                    profile.hist_x,
                    wakefield.induced_voltage,
                    label="multiturn",
                )

            wakefield = self.non_sparse_fake_multiturn(
                induced_voltage=induces_voltage
            )
            profile: ProfileBaseClass = wakefield.profile
            DEV_DRAW = True
            if DEV_DRAW:
                plt.figure("compare")
                ax1 = plt.subplot(3, 1, 1)
                center = profile.hist_x[len(profile.hist_x) // 2 - 1]
                center = 0
                plt.xlim(4e-8, 6e-8)
                plt.plot(profile.hist_x - center, profile.hist_y)
                plt.subplot(3, 1, 2, sharex=ax1)
                plt.plot(
                    profile.hist_x - center,
                    wakefield.induced_voltage,
                    "--",
                    label="single turn",
                )
                plt.legend()
                plt.subplot(3, 1, 1)
                plt.xlim(4e-8, 6e-8)
                plt.subplot(3, 1, 3)
                plt.cla()

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

        profile = StaticProfile.from_rad(
            0,
            2 * np.pi * FAKE_TUNRS,
            2**8 * 4620 * FAKE_TUNRS,
            magnetic_cycle.get_t_rev_init(
                ring.circumference,
                particle_type=proton,
            ),
        )
        wakefield = WakeField(
            sources=(Resonators(R_shunt, f_res, Q_factor),),
            solver=wake_solver,
            profile=profile,
        )
        ring.add_elements(
            (
                wakefield,
                drift,
                rf_station,
            )
        )
        sim = Simulation(
            ring=ring,
            magnetic_cycle=magnetic_cycle,
        )

        sim.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=2e-9 / 4,
                seed=1,
                n_macroparticles=n_macroparticles,
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
        save_profile = True
        if save_profile:
            pass
            """print()
            print(
                np.savetxt(
                    callers_relative_path(
                        "resources/hist_y.npy", stacklevel=1
                    ),
                    profile._hist_y,
                )
            )
            print("saved hist_y")"""
        else:
            sim.intensity_effect_manager.set_profiles(
                active=False
            )  # freeze profiles
            profile._hist_y = hist_y_single_peak
            profile.hist_y_to_density_factor = 1.0 / beam.common_array_size

        return wakefield

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
        bins_per_profile = 2**8
        filling_pattern = np.zeros(rf_station.harmonic, bool)
        filling_pattern[::10] = 1
        profile = EquidistantMultiProfile(
            bins_per_profile=bins_per_profile,
            filling_pattern=filling_pattern,
            offset=0,
        )

        wakefield = WakeField(
            sources=(Resonators(R_shunt, f_res, Q_factor),),
            solver=MultiPoleSparseSolve(),
            profile=profile,
        )
        ring.add_elements(
            (
                wakefield,
                drift,
                rf_station,
            )
        )
        sim = Simulation(
            ring=ring,
            magnetic_cycle=magnetic_cycle,
        )

        sim.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=2e-9 / 4,
                seed=1,
                n_macroparticles=n_macroparticles,
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
        return wakefield


if __name__ == "__main__":
    unittest.main()
