"""Example of how to configure a simulation with sparse multiturn wakefields."""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
    proton,
)
from blond.beam_preparation.helpers import make_multibunch_beam
from blond.physics.impedances.base import WakeFieldSource
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.profiles_sparse import StaticMultiProfile

# backend.change_backend(Cupy64Bit)
# backend.set_specials("cuda")
backend.set_specials("cpp")


sync_momentum = 25.92e9  # [eV / c]


class SupportsPoles(ABC):
    @abstractmethod
    def get_vectorfit(self):
        pass


class Poles(WakeFieldSource, SupportsPoles):
    def __init__(self, poles, residues):
        assert len(poles) == len(residues), f"{len(poles)=}  {len(residues)=}"
        self.poles = poles
        self.residues = residues

    def sort(self, by: str = "residues"):
        if by == "residues":
            order = np.argsort(np.abs(self.residues))
        elif by == "poles":
            order = np.argsort(np.abs(self.poles))
        else:
            raise NameError(str(by))
        self.poles = self.poles[order]
        self.residues = self.residues[order]

    @staticmethod
    def from_file(loc: str):
        data = np.load(loc)
        flatten = data["residues"].flatten()
        print(f"Loaded {len(flatten)} wake sources.")
        return Poles(poles=data["poles"].flatten(), residues=flatten)

    def get_vectorfit(self):
        return self.poles, self.residues


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
n_profiles = int(rf_station.harmonic // 10)
t_rev = magnetic_cycle.get_t_rev_init(
    ring.circumference,
    particle_type=proton,
)
width_per_profile = t_rev / rf_station.harmonic
bins_per_profile = 2**8
offset = 0  # t_rf / 2
step = t_rev / n_profiles
t_starts = step * np.arange(n_profiles) + offset
profiles = (
    StaticProfile(
        cut_left=float(t_starts[i]),
        cut_right=float(t_starts[i] + width_per_profile),
        n_bins=bins_per_profile,
    )
    for i in range(n_profiles)
)
profile = StaticMultiProfile(profiles=profiles)

poles = Poles.from_file("resources/1_sps_gen_new.npz")
poles.sort(by="residues")
wakefield = WakeField(
    sources=(poles,),
    solver=MultiPoleSparseSolve(),
    profile=profile,  # type: ignore
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

"""sim.profiling(
    beams=beam, n_turns=100, sortby=SortKey.CUMULATIVE, start_turn_i=2
)"""

from blond import Beam, Simulation

ax = plt.subplot(2, 1, 1)

ax2 = plt.subplot(2, 1, 2, sharex=ax)
cmap = plt.cm.get_cmap("plasma", 337)

lims = [
    [_bunch._dt.min(), _bunch._dt.max()],
    [_bunch._dE.min(), _bunch._dE.max()],
]


def my_callback(simulation: Simulation, beam: Beam) -> None:
    if simulation.turn_i.value == 0:
        return
    solver_: MultiPoleSparseSolve = wakefield.solver  # type: ignore

    plt.figure(80920)
    plt.cla()
    beam.plot_hist2d(bins=(2096, 128), range=lims)
    plt.draw()
    plt.pause(1)
    """
    residues = solver_._residues
    states = solver_._states[:-1]
    print(len(residues),len(states))
    plt.figure(80920)
    #plt.cla()
    #beam.plot_hist2d(bins=(2096, 128))
    plt.figure(8092)
    plt.sca(ax)
    artists = solver_._profile.plot()
    plt.sca(ax2)

    artists2 = plt.plot(solver_._profile._continuous_memory_hist_x,
             solver_._voltage)
    artists.extend(artists2)"""
    if False:
        plt.figure(8091)
        print(f"{len(states[:])=}")
        plt.title(simulation.turn_i.value)
        artist = plt.scatter(
            simulation.turn_i.value * np.ones(len(states[:])),
            np.real(residues[:] * states[:]),
            c=cmap(np.arange(len(states[:]))),
        )
    plt.draw()
    plt.pause(0.1)
    # artist.remove()
    # for artist in artists:
    #    artist.remove()
    # input("continue?")


sim.run_simulation(beams=beam, n_turns=3000, callbacks=(my_callback,))
