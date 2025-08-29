import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_p

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    StaticProfile,
    WakeField,
    proton,
)
from blond._core.backends.backend import Numpy64Bit, backend
from blond.physics.impedances.solvers import PeriodicFreqSolver
from blond.physics.impedances.sources import InductiveImpedance

DEV_PLOT = False


class Blond2:
    def __init__(self):
        import numpy as np

        from blond.legacy.blond2.beam.beam import Beam, Proton
        from blond.legacy.blond2.beam.distributions import bigaussian
        from blond.legacy.blond2.beam.profile import CutOptions, Profile
        from blond.legacy.blond2.impedances.impedance import (
            InductiveImpedance,
        )
        from blond.legacy.blond2.input_parameters.rf_parameters import (
            RFStation,
        )
        from blond.legacy.blond2.input_parameters.ring import Ring

        E_0 = m_p * c**2 / e  # [eV]
        ring = Ring(
            2 * np.pi * 25.0,
            1 / 4.4**2,
            np.sqrt((E_0 + 1.4e9) ** 2 - E_0**2),
            Proton(),
            1,
        )

        rf_station = RFStation(
            ring,
            [1],
            [8e3],
            [np.pi],
            1,
        )

        beam = Beam(ring, 10000001, 1e11)
        bucket_length = 2.0 * np.pi / rf_station.omega_rf[0, 0]

        bigaussian(ring, rf_station, beam, 180e-9 / 4, seed=1)
        self.beam = beam

        number_slices = int(100 * 2.5)

        profile = Profile(
            beam,
            CutOptions(
                cut_left=0, cut_right=bucket_length, n_slices=number_slices
            ),
        )
        self.profile = profile

        inductive_impedance = InductiveImpedance(
            beam, profile, [100] * 1, rf_station
        )
        inductive_impedance.induced_voltage_generation()
        self.induced_voltage = inductive_impedance.induced_voltage
        if DEV_PLOT:
            plt.figure(0)
            plt.plot(self.profile.bin_centers, self.profile.n_macroparticles)
            plt.figure(1)

            plt.plot(self.induced_voltage, label="blond2")
            plt.legend()


class Blond3:
    def __init__(self):
        blond2 = Blond2()
        self.blond2 = blond2
        circumference = 2 * np.pi * 25.0
        ring = Ring(circumference=circumference)
        drift = DriftSimple(orbit_length=circumference)
        drift.transition_gamma = 4.4
        cavity = SingleHarmonicCavity()
        cavity.harmonic = 1
        cavity.voltage = 8e3
        cavity.phi_rf = np.pi
        profile = StaticProfile(
            blond2.profile.cut_left,
            blond2.profile.cut_right,
            blond2.profile.n_slices,
        )

        wake = WakeField(
            sources=(InductiveImpedance(100 * 1),),
            # solver=InductiveImpedanceSolver(),
            solver=PeriodicFreqSolver(
                t_periodicity=5.720344547649417e-07, allow_next_fast_len=True
            ),
            profile=profile,
        )
        ring.add_elements((drift, cavity, profile, wake), reorder=True)
        beam = Beam(n_particles=1e11, particle_type=proton)

        E_0 = m_p * c**2 / e  # [eV]

        sim = Simulation(
            ring=ring,
            magnetic_cycle=ConstantMagneticCycle(
                value=np.sqrt((E_0 + 1.4e9) ** 2 - E_0**2),
                reference_particle=proton,
            ),
        )
        beam.setup_beam(
            dt=blond2.beam.dt,
            dE=blond2.beam.dE,
            reference_total_energy=sim.magnetic_cycle.get_total_energy_init(
                turn_i_init=0, t_init=0, particle_type=beam.particle_type
            ),
        )
        profile.track(beam)
        profile.invalidate_cache()

        self.induced_voltage = wake.calc_induced_voltage(beam)
        if DEV_PLOT:
            plt.figure(0)
            plt.plot(profile.hist_x, profile.hist_y, ".-")
            plt.figure(1)
            plt.plot(self.induced_voltage, "--", color="C1", label="blond3")
            plt.legend()


class TestBothBlonds(unittest.TestCase):
    def setUp(self):
        backend.change_backend(Numpy64Bit)
        self.blond3 = Blond3()
        if DEV_PLOT:
            plt.show()

    def test_induced_voltage(self):
        np.testing.assert_allclose(
            self.blond3.blond2.induced_voltage + 1,
            self.blond3.induced_voltage + 1,
            rtol=1e-12,
        )
