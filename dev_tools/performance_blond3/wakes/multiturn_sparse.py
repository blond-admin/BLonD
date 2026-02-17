"""Example of how to configure a simulation with sparse multiturn wakefields."""

import os
from pstats import SortKey

import numpy as np

import blond
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
    proton,
)
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.profiles_sparse import EquidistantMultiProfile

# backend.change_backend(Cupy64Bit)
# backend.set_specials("cuda")
backend.set_specials("cpp")

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
    offset=t_rf / 2,
)
wakefield = WakeField(
    sources=(Resonators(R_shunt, f_res, Q_factor),),
    solver=MultiPoleSparseSolve(),
    profile=profile,
)
ring.add_elements(
    (
        wakefield,
        # profile,
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
        n_macroparticles=1e4,
    ),
    beam=_bunch,
)

beam = make_multibunch_beam(
    beam=_bunch,
    n_times=int(rf_station.harmonic // 10),
    t_distance=t_rf * 10,
)


sim.profiling(
    beams=beam, n_turns=100, sortby=SortKey.CUMULATIVE, start_turn_i=2
)

sim.run_simulation(beams=beam, n_turns=3000)
