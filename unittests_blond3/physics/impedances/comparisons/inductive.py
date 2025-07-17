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
from blond3.physics.impedances.solvers import PeriodicFreqSolver, TimeDomainSolver

solver1 = PeriodicFreqSolver(
    t_periodicity=100,
    allow_next_fast_len=True,
)
solver2 = TimeDomainSolver()

for solver in (solver1, solver2):
    ring = Ring(circumference=2 * np.pi * 25.0)
    drift = DriftSimple()
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
    beam = Beam(n_particles=1e11, particle_type=proton)
    np.random.seed(1)
    distr = np.random.randn(10000, 2)
    beam.setup_beam(dt=distr[:, 0] + 5, dE=distr[:, 1])
    profile.track(beam)
    profile.invalidate_cache()
    E_0 = m_p * c**2 / e  # [eV]

    sim = Simulation(
        ring=ring,
        beams=(beam,),
        magnetic_cycle=ConstantMagneticCycle(
            value=np.sqrt((E_0 + 1.4e9) ** 2 - E_0**2),
            reference_particle=proton,
        ),
    )
    induced_voltage = wake.calc_induced_voltage(beam)
    plt.figure(0)
    plt.plot(profile.hist_x, profile.hist_y, ".-")
    plt.figure(1)
    plt.plot(induced_voltage, "--", label=str(type(solver)))
    plt.legend()
plt.show()
