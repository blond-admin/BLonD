import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_p

from blond3 import (
    Beam,
    Ring,
    Simulation,
    proton,
    ConstantMagneticCycle,
    StaticProfile,
    SingleHarmonicCavity,
    DriftSimple,
    WakeField,
)
from blond3.physics.impedances.sources import InductiveImpedance
from blond3.physics.impedances.solvers import (
    PeriodicFreqSolver,
    TimeDomainSolver,
    InductiveImpedanceSolver,
)

DEV_PLOT = False


class TestInductiveImpedances(unittest.TestCase):
    def setUp(self):
        from blond3._core.backends.backend import backend, Numpy64Bit

        backend.change_backend(Numpy64Bit)

    def tearDown(self):
        from blond3._core.backends.backend import backend, Numpy32Bit

        backend.change_backend(Numpy32Bit)

    def test_equal(self):
        voltages = {}
        solver1 = PeriodicFreqSolver(
            t_periodicity=100,
            allow_next_fast_len=False,
        )
        solver2 = TimeDomainSolver()

        solver3 = InductiveImpedanceSolver()

        for i, solver in enumerate([solver1, solver2, solver3]):
            ring = Ring(
                circumference=2 * np.pi * 25.0,
            )
            drift = DriftSimple(
                orbit_length=ring.circumference,
            )
            drift.transition_gamma = 4.4
            cavity = SingleHarmonicCavity()
            cavity.harmonic = 1
            cavity.voltage = 8e3
            cavity.phi_rf = np.pi
            profile = StaticProfile(
                0,
                10,
                256,
            )

            wake = WakeField(
                sources=(InductiveImpedance(100 * 1),),
                solver=solver,
                profile=profile,
            )
            ring.add_elements(
                (drift, cavity, profile, wake),
                reorder=True,
            )
            beam = Beam(
                n_particles=1e11,
                particle_type=proton,
            )
            np.random.seed(1)
            distr = np.random.randn(10000, 2)
            E_0 = m_p * c**2 / e  # [eV]

            cycle = ConstantMagneticCycle(
                value=np.sqrt((E_0 + 1.4e9) ** 2 - E_0**2),
                reference_particle=proton,
            )
            beam.setup_beam(
                dt=distr[:, 0] + 5,
                dE=distr[:, 1],
                reference_total_energy=cycle.get_total_energy_init(
                    turn_i_init=0,
                    t_init=0,
                    particle_type=cycle.reference_particle,
                ),
            )
            profile.track(beam)
            profile.invalidate_cache()

            sim = Simulation(
                ring=ring,
                magnetic_cycle=cycle,
            )
            induced_voltage = wake.calc_induced_voltage(beam)
            voltages[str(solver)] = induced_voltage
            if DEV_PLOT:
                plt.figure(0)
                plt.plot(
                    profile.hist_x,
                    profile.hist_y,
                    ".-",
                )
                plt.figure(1)
                plt.plot(
                    induced_voltage,
                    ["-", "--", ":"][i],
                    label=str(type(solver)),
                )
                plt.legend()
        if DEV_PLOT:
            plt.show()
        for i, solver in enumerate(voltages.keys()):
            if i == 0:
                reference = voltages[solver]  # arbitrary choice
            else:
                np.testing.assert_allclose(
                    reference * 1e13 ,
                    voltages[solver] * 1e13 ,
                    atol= 1e-10,
                )
