import matplotlib.pyplot as plt
import numpy as np

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
from blond3.physics.impedances.sources import Resonators
from blond3.physics.impedances.solvers import PeriodicFreqSolver, TimeDomainSolver

for i, solver in enumerate(
    (
        PeriodicFreqSolver(t_periodicity=10, allow_next_fast_len=True),
        TimeDomainSolver(),
    )
):
    ring = Ring(circumference=6911.56)
    profile = StaticProfile(
        0,
        10 * 96,
        256 * 96,
    )
    cavity1 = SingleHarmonicCavity()
    cavity1.voltage = 0
    cavity1.phi_rf = 0
    cavity1.harmonic = 1
    drift = DriftSimple()
    drift.transition_gamma = 1
    resonators = Resonators(
        shunt_impedances=1 * np.ones(1),
        center_frequencies=1 * np.ones(1),
        quality_factors=0.6 * np.ones(1),
    )
    np.random.seed(1)
    distr = np.random.randn(10000, 2)

    beam = Beam(n_particles=1e10, particle_type=proton)
    beam.setup_beam(dt=distr[:, 0] + 5, dE=distr[:, 1])
    profile.track(beam)
    plt.figure(0)
    plt.subplot(2, 1, 1)
    plt.plot(profile.hist_x, profile.hist_y)

    wake = WakeField(
        sources=(resonators,),
        solver=solver,
        profile=profile,
    )
    ring.add_elements((profile, cavity1, drift, wake))
    magnetic_cycle = ConstantMagneticCycle(25.92e9, 10)
    sim = Simulation(ring=ring, beams=(beam,), magnetic_cycle=magnetic_cycle)
    wake_ = np.fft.irfft(resonators.get_wake_impedance(profile.hist_x, simulation=sim))
    plt.figure(0)
    plt.subplot(2, 1, 2)
    plt.plot(profile.hist_x, wake_)
    induced_voltage = wake.calc_induced_voltage(beam=beam)
    plt.figure(1)
    plt.plot(induced_voltage, "--", label=str(type(solver)))
    # plt.plot(np.convolve(profile.hist_y, wake_))
    plt.legend()
plt.show()
