"""SPS IONS."""


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sps_tilted_plotting import plot_fitted_ellipse

from blond import (
    AllowPlotting,
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
)
from blond.core.backends.backend import backend
from blond.core.beam.beams import ProbeBeam
from blond.core.beam.particle_types import lead_82
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.handle_results.observables_as_elements import (
    BunchObservationMetaParams,
)

# backend.change_backend(Cupy64Bit)
backend.set_specials("cpp")


def run_simulation_SPS_ions_flat_bot(
    voltage_multiplier: float,
    phase_jump: float,
    bunch_length,
    no_obs_mode: bool = False,
):
    """Execute the SPS simulation main script."""
    ring = Ring(circumference=6912)

    magnetic_cycle = ConstantMagneticCycle(
        value=0.0768,
        in_unit="bending field",
        reference_particle=lead_82,
        bending_radius=(4657.4400 / (2 * np.pi)),
    )

    beam = Beam(
        intensity=1e8,
        particle_type=lead_82,
        is_counter_rotating=False,
    )

    bunch_obs = BunchObservationMetaParams(each_turn_i=1, beam=beam)

    one_turn_model = []
    # after_turn_prof = StaticProfile.from_rad(cut_left_rad=-0.1*np.pi, cut_right_rad=0.1*np.pi, t_period=5e-9, n_bins=256)
    # prof_obs = StaticProfileObservation(profile=after_turn_prof, each_turn_i=each_turn_i)

    one_turn_model.extend(
        [
            DriftSimple(
                transition_gamma=-22.774,
                orbit_length=ring.circumference / 2,
                section_index=0,
            ),
            SingleHarmonicRfStation(
                voltage=(10e6 * voltage_multiplier / 2),
                phi_rf=0,
                harmonic=4620,
                section_index=0,
            ),
        ]
    )

    one_turn_model.extend(
        [
            DriftSimple(
                transition_gamma=-22.774,
                orbit_length=ring.circumference / 2,
                section_index=1,
            ),
            SingleHarmonicRfStation(
                voltage=(10e6 * voltage_multiplier / 2),
                phi_rf=0,
                harmonic=4620,
                section_index=1,
            ),
        ]
    )
    # if not no_obs_mode:
    #    one_turn_model.append(bunch_obs)

    ring.add_elements(one_turn_model, reorder=False)
    sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    # sim.print_one_turn_execution_order()
    # if no_obs_mode:

    scalar = voltage_multiplier / 3.0
    sim.prepare_beam(
        beam=beam,
        # preparation_routine=BiGaussian(n_macroparticles=1e6, sigma_dt=bunch_length))
        preparation_routine=SemiEmpiricMatcher(
            time_limit=(-0.5e-9, 0.5e-9),
            n_macroparticles=1e6,
            hamilton_to_density_kwargs={
                "density_modifier": 2,
                "hamilton_max": scalar * 5000,
            },
            animate=True,
            internal_grid_shape=(1024 * 8, 1024 * 8),
        ),
    )
    plt.figure("debug")
    """SHEAR_X = 7e18
    SHEAR_X = 2.5e19
    beam._dE *= (1 + (0.7 / 3.0) *  .15)
    beam._dt += (1 / SHEAR_X) * beam._dE"""
    show_stability = False
    if show_stability:
        plt.figure("debug")
        beam.plot_hist2d()
        x, y = copy_to_cpu(beam._dt), copy_to_cpu(beam._dE)
        ax = plt.gca()
        hist, xedges, yedges, img = ax.hist2d(x, y, bins=128, cmap="viridis")

        # Compute the centers of bins for plotting contours
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])

        # Plot contour lines (equipotentials of the histogram density)
        CS = ax.contour(
            xcenters,
            ycenters,
            gaussian_filter(hist, sigma=10).T,
            colors="white",
            linewidths=1,
        )
        memory = np.zeros((11, 2))
        memory2 = np.zeros((11, 2, 10000))

        def plot_beam(simulation, beam):
            if simulation.turn_i.value % 1 == 0:  # Every 100 turns
                memory2[:, 0, simulation.turn_i.value] = beam._dt
                memory2[:, 1, simulation.turn_i.value] = beam._dE
                colors = [
                    "red",
                    "green",
                    "blue",
                    "orange",
                    "purple",
                    "cyan",
                    "magenta",
                    "yellow",
                    "black",
                    "brown",
                    "pink",
                ]
                for i in range(len(memory)):
                    if memory[i, 0] < beam._dE[i]:
                        memory[i, 0] = beam._dE[i]
                        memory[i, 1] = beam._dE[i] / beam._dt[i]
                plt.figure("debug_2")
                plt.cla()
                plt.scatter(
                    np.arange(len(memory[:, 0])), memory[:, 1], c=colors
                )
                plt.figure("debug")
                # plt.clf()
                beam.plot_scatter(c=colors)
                if simulation.turn_i.value == 10:
                    for inmdex in [2, 8]:
                        plot_fitted_ellipse(
                            memory2[inmdex, 0, : simulation.turn_i.value],
                            memory2[inmdex, 1, : simulation.turn_i.value],
                        )
                plt.draw()
                plt.pause(0.01)

        beam2 = ProbeBeam(
            particle_type=beam.particle_type,
            dt=np.linspace(0.01 * beam.dt_max, 0.8 * beam.dt_max, 11),
            intensity=beam.intensity,
        )
        plt.figure("debug")
        plt.axvline(np.mean(copy_to_cpu(beam._dt)))

        sim.run_simulation(beams=(beam2,), n_turns=10, callback=plot_beam)
    else:
        plt.figure("debug2")
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

        def plot_beam(simulation, beam):
            if simulation.turn_i.value % 1000 == 0:  # Every 100 turns
                with AllowPlotting():
                    plt.sca(ax2)
                    plt.cla()
                    beam.plot_hist2d(
                        range=(
                            (-5e-10, 5e-10),
                            (-np.sqrt(scalar) * 12e8, np.sqrt(scalar) * 12e8),
                        ),
                        bins=256 * 2,
                    )
                    plt.draw()
                    plt.pause(1)

        plt.sca(ax1)
        beam.plot_hist2d(
            range=(
                (-5e-10, 5e-10),
                (-np.sqrt(scalar) * 12e8, np.sqrt(scalar) * 12e8),
            ),
            bins=256 * 2,
        )
        plt.draw()
        plt.pause(1)
        sim.run_simulation(beams=(beam,), n_turns=1e12, callback=plot_beam)


if __name__ == "__main__":
    for bunch_length in [1.5e-10, 3e-10, 1e-9, 2e-9]:
        for phase_jump in [10, 30, 50, 70]:  # , 30, 50, 70, 90]:
            for volt_mult in [3.0, 0.1]:
                run_simulation_SPS_ions_flat_bot(
                    voltage_multiplier=volt_mult,
                    phase_jump=phase_jump,
                    bunch_length=bunch_length,
                    no_obs_mode=False,
                )
            plt.show()
