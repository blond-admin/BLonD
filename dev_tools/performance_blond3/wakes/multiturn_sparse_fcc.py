"""Example of how to configure a simulation with sparse multiturn wakefields."""

import math
from pstats import SortKey

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
from blond.physics.profiles_sparse import EquidistantMultiProfile

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
t_rev_init = magnetic_cycle.get_t_rev_init(
    ring.circumference,
    particle_type=electron,
)
t_rf = t_rev_init / rf_station.harmonic

n_profiles = 1118
bins_per_profile = 2**10
filling_pattern = np.zeros(rf_station.harmonic, bool)
step = int(math.ceil(rf_station.harmonic / n_profiles))
filling_pattern[::step] = 1
assert np.sum(filling_pattern) == n_profiles, (
    f"{np.sum(filling_pattern)} == {n_profiles}"
)

profile = EquidistantMultiProfile(
    bins_per_profile=bins_per_profile,
    filling_pattern=filling_pattern,
    offset=-t_rf / 2,
)

profile2 = StaticProfile(
    cut_left=0,
    cut_right=t_rev_init,
    n_bins=np.sum(filling_pattern)
    * bins_per_profile,  # wrong, but interesting
    # for performance comparison
)


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
        profile2,
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
        sigma_dt=50e-12,
        seed=1,
        n_macroparticles=1e4,
    ),
    beam=_bunch,
)

beam = make_multibunch_beam(
    beam=_bunch,
    n_times=n_profiles,
    t_distance=profile.profiles[1].cut_left - profile.profiles[0].cut_left,
    common_offset=0,  # t_rf / 2,
)

sim.profiling(
    beams=beam, n_turns=10, sortby=SortKey.CUMULATIVE, start_turn_i=2
)


def my_callback(simulation: Simulation, beam: Beam) -> None:
    """Utility function for plotting.

    Parameters
    ----------
    simulation
        `Simulation` context manager
    beam
        Simulation beam object
    """
    plt.figure(2)
    plt.cla()
    profile.profiles[-1].plot()
    plt.draw()
    plt.pause(0.1)


my_callback.each_turn_i = 10

sim.run_simulation(
    beams=beam,
    n_turns=3000,
    callbacks=my_callback,
)
