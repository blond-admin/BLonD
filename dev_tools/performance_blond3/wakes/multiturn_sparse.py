import os

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
    proton,
)
from blond.physics.impedances.sparse_profile.solvers import (
    MultiTurnSparseProfileSolver,
)
from blond.physics.profiles_sparse import EquidistantMultiProfile


def make_multibunch_beam(
    beam: Beam, n_times: int, t_distance: float, common_offset: float = 0.0
) -> Beam:
    full_beam = Beam(
        intensity=n_times * beam.intensity,
        particle_type=beam.particle_type,
        is_counter_rotating=beam.is_counter_rotating,
    )

    size = beam._dt.local_size
    full_dE = backend.repeat(beam._dE.array_local, n_times)

    full_dt = backend.empty(
        full_dE.shape,
        dtype=backend.float,
    )
    for i in range(n_times):
        t_offset = t_distance * i + common_offset
        sel = slice(i * size, (i + 1) * size)
        full_dt[sel] = beam._dt.array_local + t_offset

    full_beam.setup_beam(
        dt=full_dt,
        dE=full_dE,
        mpi_mode="all-ranks",
    )
    return full_beam


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

profile = EquidistantMultiProfile(
    n_profiles=int(rf_station.harmonic // 10),
    width_per_profile=magnetic_cycle.get_t_rev_init(
        ring.circumference,
        particle_type=proton,
    )
    / rf_station.harmonic,
    bins_per_profile=2**8,
    offset=t_rf / 2,
)
wakefield = WakeField(
    sources=(Resonators(R_shunt, f_res, Q_factor),),
    solver=MultiTurnSparseProfileSolver(n_turns=10),
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
        n_macroparticles=1e4,
    ),
    beam=_bunch,
)

beam = make_multibunch_beam(
    beam=_bunch,
    n_times=int(rf_station.harmonic // 10),
    t_distance=t_rf * 10,
)
for i, _prof in enumerate(profile.profiles):
    if i % 10 == 0:
        pass
    else:
        _prof.active = False

drift.orbit_length = 0
rf_station.voltage = 0.0
sim.check_circumference = "ignore"

backend.set_specials("cpp")
# sim.profiling(
#    beams=beam, n_turns=10, sortby=SortKey.TIME, start_turn_i=2
# )
sim.run_simulation(beams=beam, n_turns=3000)
