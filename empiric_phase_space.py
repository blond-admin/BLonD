from copy import deepcopy
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy import dtype, ndarray

from blond import (
    Beam,
    BoxLosses,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)

# logging.basicConfig(level=logging.INFO)


def main():
    beam1, sim = build_simulation()

    XLIM = (0, 3e-9)
    YLIM = (-4e8, 4e8)
    GRID_RESOLUTION = (1024, 1024)
    n_warmup = 1000

    ALPHA = 0.1  # EMA weight for new observations
    STALE_STEPS = 100  # steps without an active visit before pixel → nan

    beam_tmp = deepcopy(beam1)
    grid_e = np.full(GRID_RESOLUTION, np.nan)
    grid_last_visit = np.full(GRID_RESOLUTION, -(STALE_STEPS + 1), dtype=int)
    dt, dE = np.meshgrid(
        np.linspace(*XLIM, grid_e.shape[0]),
        np.linspace(*YLIM, grid_e.shape[1]),
        indexing="ij",
    )
    energy_init = sim.magnetic_cycle.get_total_energy_init()
    time_init = beam1.reference.time
    beam_tmp.setup_beam(
        dE=dE.flatten(),
        dt=dt.flatten(),
        reference_total_energy=energy_init,
        reference_time=time_init,
    )

    def as_index(x, lims, axis):
        range_ = lims[1] - lims[0]
        return ((x - lims[0]) * grid_e.shape[axis] / range_).astype(int)

    max_energy = np.zeros(beam_tmp.dE.array_local.shape[0])
    step = 0

    for i in range(n_warmup):
        beam1.reference.total_energy = energy_init
        beam1.reference.time = time_init
        deepcopy(sim).run_simulation(
            beam_tmp, n_turns=1, show_progressbar=False, verbose=False
        )

    while True:
        step += 1
        beam1.reference.total_energy = energy_init
        beam1.reference.time = time_init
        deepcopy(sim).run_simulation(
            beam_tmp, n_turns=1, show_progressbar=False, verbose=False
        )
        x_idxs = as_index(beam_tmp.dt.array_local, XLIM, axis=0)
        y_idxs = as_index(beam_tmp.dE.array_local, YLIM, axis=1)
        active_ids = beam_tmp.ids.array_local

        # Update max_energy for active particles
        sel = np.abs(beam_tmp.dE.array_local) > max_energy[active_ids]
        max_energy[active_ids[sel]] = np.abs(beam_tmp.dE.array_local[sel])

        # Mark lost particles and far-escaped active particles as nan
        lost_mask = np.ones(max_energy.shape[0], dtype=bool)
        lost_mask[active_ids] = False
        max_energy[lost_mask] = np.nan
        far_out = (
            (x_idxs < -grid_e.shape[0] / 2)
            | (x_idxs > grid_e.shape[0] * 3 / 2)
            | (y_idxs < -grid_e.shape[1] / 2)
            | (y_idxs > grid_e.shape[1] * 3 / 2)
        )
        max_energy[active_ids[far_out]] = np.nan

        # EMA update: only active particles with a valid max_energy contribute
        for i in range(len(active_ids)):
            xi, yi = x_idxs[i], y_idxs[i]
            if 0 <= xi < grid_e.shape[0] and 0 <= yi < grid_e.shape[1]:
                e_val = max_energy[active_ids[i]]
                if not np.isnan(e_val):
                    if np.isnan(grid_e[xi, yi]):
                        grid_e[xi, yi] = e_val
                    else:
                        grid_e[xi, yi] = (
                            ALPHA * e_val + (1 - ALPHA) * grid_e[xi, yi]
                        )
                    grid_last_visit[xi, yi] = step

        # Pixels not visited by any active particle for STALE_STEPS → nan
        # grid_e[step - grid_last_visit > STALE_STEPS] = np.nan

        plot_coordinates(beam_tmp, XLIM, YLIM)
        plot_grid(grid_e)
    plt.show()


def plot_coordinates(beam_tmp: Beam, xlim, ylim):
    plt.subplot(1, 2, 1)
    plt.cla()
    plt.scatter(beam_tmp.dt.array_local, beam_tmp.dE.array_local)
    plt.xlim(xlim)
    plt.ylim(ylim)


def plot_grid(grid_e: ndarray[tuple[int, int], dtype[Any]]):
    plt.subplot(1, 2, 2)
    plt.cla()
    plt.imshow(grid_e.T[::-1, :], cmap="viridis")
    plt.draw()
    plt.pause(0.01)


def build_simulation() -> tuple[Beam, Simulation]:
    ring = Ring(26658.883)
    CHAOS_FACTOR = 40
    rf_station = SingleHarmonicRFStation()
    rf_station.harmonic = 35640
    rf_station.voltage = 6e6 * CHAOS_FACTOR
    rf_station.phi_rf_design = 0

    N_TURNS = int(1e3)

    energy_cycle = MagneticCyclePerTurn.init_from_linspace(
        values=np.linspace(450e9, 600e9, N_TURNS + 1),
        in_unit="kinetic energy",
        reference_particle=proton,
    )

    drift1 = DriftSimple(
        orbit_length=26658.883,
    )
    drift1.momentum_compaction_factor = (
        momentum_compaction_factor(transition_gamma=55.759505) * CHAOS_FACTOR
    )
    losses = BoxLosses(
        True,
        0,
        3e-9,
        -4e8,
        4e8,
    )
    beam1 = Beam(
        intensity=1e9,
        particle_type=proton,
    )

    sim = Simulation.from_locals(locals())
    return beam1, sim


if __name__ == "__main__":
    main()
