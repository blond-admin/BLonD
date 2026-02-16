"""Example of how to configure a simulation with sparse multiturn wakefields."""

from abc import ABC, abstractmethod
from pstats import SortKey

import matplotlib
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
    WakeField,
    backend,
    proton,
)
from blond.beam_preparation.helpers import make_multibunch_beam
from blond.physics.impedances.base import WakeFieldSource
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.profiles_sparse import EquidistantMultiProfile

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
        residues = data["residues"].flatten()
        poles = data["poles"].flatten()
        print(f"Loaded {len(residues)} wake sources.")
        return Poles(poles=poles, residues=residues)

    def plot(self, freq):
        """Omega = 2 * np.pi * freq
        s = 1j * omega
        h = np.zeros_like(s)
        for i in range(len(self.poles)):
            pk = self.poles[i]
            ck = self.residues[i]
            h += ck / (s - pk)
        plt.subplot(3,1,1)
        plt.plot(freq, np.abs(h))
        plt.subplot(3,1,2)
        plt.plot(freq, np.real(h))
        plt.subplot(3,1,3)
        plt.plot(freq, np.imag(h))
        """
        import skrf as rf

        freq = rf.Frequency.from_f(freq, unit="Hz")
        ntwk = rf.Network(
            frequency=freq, s=np.zeros(len(freq), float).reshape(-1, 1, 1)
        )
        vf = rf.VectorFitting(ntwk)
        # vf.proportional_coeff = np.array([np.sum(self.proportional_coeff)])
        vf.proportional_coeff = np.array([0.0])
        vf.constant_coeff = np.array([0.0])
        # vf.constant_coeff = np.array([np.sum(self.constant_coeff)])
        vf.poles = np.array(self.poles)[:]

        vf.residues = np.array(self.residues)[np.newaxis, :]
        plt.subplot(3, 1, 1)
        vf.plot_s_db()  # overlay fit vs original

        plt.subplot(3, 1, 2)
        vf.plot_s_re()  # overlay fit vs original

        plt.subplot(3, 1, 3)
        vf.plot_s_im()  # overlay fit vs original

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
    voltage=4e6,
    phi_rf=0.0,
)
t_rf = (
    magnetic_cycle.get_t_rev_init(
        ring.circumference,
        particle_type=proton,
    )
    / rf_station.harmonic
)
t_rev = magnetic_cycle.get_t_rev_init(
    ring.circumference,
    particle_type=proton,
)


filling_pattern = np.zeros(rf_station.harmonic, bool)
filling_pattern[::10] = 1
# filling_pattern[0] = 1
bins_per_profile = 256

profile = EquidistantMultiProfile(
    filling_pattern=filling_pattern,
    bins_per_profile=bins_per_profile,
)

poles = Poles.from_file("resources/1_sps_gen_new.npz")
# poles.plot(np.linspace(0, 10e9, 10000))
plt.show()
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
        rf_station,
        drift,
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

from blond import Beam, Simulation

ax = plt.subplot(2, 1, 1)

ax2 = plt.subplot(2, 1, 2, sharex=ax)
cmap = matplotlib.colormaps["plasma"]
t_rev = sim.get_t_rev_init()
lims = [
    [profile.profiles[-1].cut_left, profile.profiles[-1].cut_right],
    [2 * _bunch._dE.min(), 2 * _bunch._dE.max()],
]

plt.figure(8092)
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2, sharex=ax1)


def my_callback(simulation: Simulation, beam: Beam) -> None:
    solver_: MultiPoleSparseSolve = wakefield.solver  # type: ignore
    if True:
        plt.figure(80920)
        plt.cla()
        beam.plot_hist2d(bins=(2096, 128), range=lims)
        plt.draw()
        plt.pause(1)

    """
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
        plt.title(simulation.turn_i.value)
        states = solver_._states[:-1]
        residues = solver_._residues
        artist = plt.scatter(
            simulation.turn_i.value * np.ones(len(states[:])),
            np.real(residues[:] * states[:]),
            c=cmap(np.arange(len(states[:]))),
        )
    if False:
        plt.figure(8092)
        plt.sca(ax1)
        plt.cla()
        plt.plot(
            profile._continuous_memory_hist_x[:] + beam.reference.time,
            profile._continuous_memory_hist_y[:],
        )

        plt.sca(ax2)
        plt.cla()
        plt.plot(
            profile._continuous_memory_hist_x[:] + beam.reference.time,
            wakefield.solver._voltage[:],
        )

    plt.draw()
    plt.pause(0.1)
    if simulation.turn_i.value == 0:
        print("saved histogram")
        p = profile.profiles[0]
        np.savez(
            "/home/slauber/PycharmProjects/deleteme/manyideas"
            "/linear_runtime_wakes/resources/hist.npz",
            hist_x=p.hist_x,
            hist_y=p.hist_y,
        )
    # if simulation.turn_i.value == 1:
    #    plt.show()
    # artist.remove()
    # for artist in artists:
    #    artist.remove()
    # input("continue?")


my_callback.each_turn_i = 10
sim.run_simulation(
    beams=beam,
    n_turns=3000,
    # callbacks=(my_callback,),
)
