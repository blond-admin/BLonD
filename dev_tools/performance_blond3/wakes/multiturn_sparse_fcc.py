"""Example of how to configure a simulation with sparse multiturn wakefields."""

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
    electron,
)
from blond.beam_preparation.helpers import make_multibunch_beam
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.profiles_sparse import StaticMultiProfile
from pstats import SortKey

backend.set_specials("cpp")


sync_momentum = 20e9  # [eV]

ring = Ring(
    circumference=90.65874532 * 1e3,
)
magnetic_cycle = ConstantMagneticCycle(
    reference_particle=electron,
    value=sync_momentum,
    in_unit="total energy",
)
_bunch = Beam(
    intensity=1e10,
    particle_type=electron,
)
drift = DriftSimple(
    momentum_compaction_factor=0.646747216157 / ring.circumference,
    orbit_length=1.0 * ring.circumference,
)
rf_station = SingleHarmonicRFStation(
    harmonic=242400,
    voltage=50.1e6,
    phi_rf=0.0,
)
t_rf = (
    magnetic_cycle.get_t_rev_init(
        ring.circumference,
        particle_type=electron,
    )
    / rf_station.harmonic
)

n_profiles = 1120
t_rev = magnetic_cycle.get_t_rev_init(
    ring.circumference,
    particle_type=electron,
)
width_per_profile = t_rf
bins_per_profile = 2**10
profile_offset = -t_rf / 2
profile_center_distance = t_rf * int(rf_station.harmonic / n_profiles)
t_starts = profile_center_distance * np.arange(n_profiles) + profile_offset
profiles = (
    StaticProfile(
        cut_left=float(t_starts[i]),
        cut_right=float(t_starts[i] + width_per_profile),
        n_bins=bins_per_profile,
    )
    for i in range(n_profiles)
)
profile = StaticMultiProfile(profiles=profiles)

R_over_Q = 315.2
Q_factor = Q_loaded = 1e7
R_shunt = R_over_Q / Q_loaded
f_res = 1 / t_rf

wakefield = WakeField(
    sources=(
        Resonators(
            shunt_impedances=np.array([R_shunt]),
            center_frequencies=np.array([f_res]),
            quality_factors=np.array([Q_factor]),
        ),
    ),
    solver=MultiPoleSparseSolve(),
    profile=profile,  # type: ignore
)
ring.add_elements(
    (
        # wakefield,
        profile,
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
        sigma_dt=50e-12,
        seed=1,
        n_macroparticles=1e5,
    ),
    beam=_bunch,
)

beam = make_multibunch_beam(
    beam=_bunch,
    n_times=n_profiles,
    t_distance=profile_center_distance,
    common_offset=0,  # t_rf / 2,
)

sim.profiling(
    beams=beam, n_turns=10, sortby=SortKey.CUMULATIVE, start_turn_i=2
)


def my_callback(simulation: Simulation, beam: Beam) -> None:
    if False:
        plt.figure(1)
        beam.plot_hist2d(
            range=[
                [
                    -t_rf + 5 * profile_center_distance,
                    t_rf + 5 * profile_center_distance,
                ],
                [
                    -7e8,
                    7e8,
                ],
            ]
        )
    plt.figure(2)
    plt.cla()
    profile.profiles[-1].plot()
    plt.draw()
    plt.pause(0.1)


my_callback.each_turn_i = 10

sim.run_simulation(beams=beam, n_turns=3000, callbacks=my_callback)
