# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
PS triple splitting with BLonD owning the loop and xsuite the drift.

The same physics as ``EX_10_MultiRFManipulation_TripleSplitting``, but the
longitudinal drift is an xsuite ``LineSegmentMap`` wrapped with
``WrapXsuite4Blond`` and inserted into the BLonD ring as a black-box
length-bearing element. The h=7/14/21 splitting kicks, the time-keyed
voltage programs (``ScheduledInterpolation``), the beam matching and the
profile observation all stay native BLonD.

See ``EX_29_BLonD_in_Xsuite`` for the mirrored setup where xsuite owns the
loop and BLonD provides the cavity.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from blond import (
    Beam,
    ConstantMagneticCycle,
    MultiHarmonicRFStation,
    Ring,
    ScheduledInterpolation,
    Simulation,
    StaticProfile,
    StaticProfileObservation,
    copy_to_cpu,
    momentum_compaction_factor,
    proton,
    setup_backend,
)
from blond.interfaces.xsuite.elements import WrapXsuite4Blond
from blond.testing import pytest_active

if not pytest_active():  # pragma: no cover
    setup_backend("auto")

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"


def main(
    run_n_turns: int | None = None, n_macroparticles: int = int(1e4)
) -> None:
    momentum = 3.9051e9
    circumference = 2 * np.pi * 100
    transition_gamma = 6.1
    bunch_population = 2.6e11 * 12
    each_turn_i_profile = 100

    ring = Ring(
        circumference=circumference,
    )

    energy_cycle = ConstantMagneticCycle(
        value=momentum,
        reference_particle=proton,
    )

    main_rf_station = MultiHarmonicRFStation(
        n_harmonics=3, main_harmonic_idx=0
    )
    main_rf_station.harmonic = np.array([7, 14, 21])
    main_rf_station.phi_rf_design = np.array([0, np.pi, 0])

    voltage_h7_prog = np.loadtxt(
        f"{this_directory}/resources/EX_10_programs/voltage_h7.txt"
    )
    voltage_h14_prog = np.loadtxt(
        f"{this_directory}/resources/EX_10_programs/voltage_h14.txt"
    )
    voltage_h21_prog = np.loadtxt(
        f"{this_directory}/resources/EX_10_programs/voltage_h21.txt"
    )

    scheduler = ScheduledInterpolation(
        times=voltage_h7_prog[:, 0] * 1e-3,
        values=np.vstack(
            (
                voltage_h7_prog[:, 1] * 1e3,
                voltage_h14_prog[:, 1] * 1e3,
                voltage_h21_prog[:, 1] * 1e3,
            )
        ),
    )

    main_rf_station.schedule(
        attribute="voltage",
        value=scheduler,
    )

    # ------------------------------------------------------------------
    # The drift lives on the xsuite side: a LineSegmentMap carrying the
    # momentum compaction (its RF is disabled), wrapped as a black-box
    # BLonD ring element. It occupies the full circumference and advances
    # the reference clock; BLonD's analytics (eta_0, phi_s) never see it.
    # ------------------------------------------------------------------
    segment_map = xt.LineSegmentMap(
        longitudinal_mode="nonlinear",
        qx=1.1,
        qy=1.2,
        betx=1.0,
        bety=1.0,
        voltage_rf=0.0,
        frequency_rf=0.0,
        lag_rf=0.0,
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma
        ),
        length=circumference,
    )
    drift = WrapXsuite4Blond(
        xt.Line(elements=[segment_map], element_names=["matrix"])
    )

    beam = Beam(
        intensity=bunch_population,
        particle_type=proton,
    )

    t_rev = energy_cycle.get_t_rev_init(ring.circumference, proton)
    t_rf = t_rev / main_rf_station.get_main_harmonic()

    profile = StaticProfile(
        cut_left=-t_rf / 2,
        cut_right=t_rf / 2,
        n_bins=2**11,
    )

    ring.add_elements((main_rf_station, profile, drift), reorder=True)

    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    sim.print_one_turn_execution_order()

    n_turns = int((voltage_h7_prog[-1, 0] * 1e-3) / t_rev)

    # ------------------------------------------------------------------
    # Seed an approximately matched Gaussian in the initial h=7 bucket by
    # hand: the black-box drift hides the optics from BLonD's analytics
    # (eta_0 averaging), so hamiltonian-based matchers (SemiEmpiricMatcher,
    # BiGaussian) cannot run with a wrapped drift.
    # ------------------------------------------------------------------
    total_energy = float(np.sqrt(momentum**2 + proton.mass**2))
    beta0 = momentum / total_energy
    gamma = total_energy / proton.mass
    eta = abs(momentum_compaction_factor(transition_gamma) - 1 / gamma**2)
    voltage_init = voltage_h7_prog[0, 1] * 1e3
    omega_s = (2 * np.pi / t_rev) * np.sqrt(
        7 * voltage_init * eta / (2 * np.pi * beta0**2 * total_energy)
    )
    sigma_dt = t_rf / 8
    sigma_dE = omega_s * beta0**2 * total_energy / eta * sigma_dt

    rng = np.random.default_rng(seed=42)
    beam.setup_beam(
        dt=rng.normal(0.0, sigma_dt, n_macroparticles),
        dE=rng.normal(0.0, sigma_dE, n_macroparticles),
        reference_total_energy=total_energy,
    )

    profile_observation = StaticProfileObservation(
        each_turn_i=each_turn_i_profile,
        profile=profile,
    )

    sim.run_simulation(
        beams=(beam,),
        n_turns=n_turns if run_n_turns is None else run_n_turns,
        observe=(profile_observation,),
    )

    bunch_time = copy_to_cpu(profile.hist_x) * 1e9
    cycle_time = profile_observation.turns_array * t_rev * 1e3
    plt.figure("bunch evolution")
    plt.clf()
    plt.imshow(
        profile_observation.hist_y,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        extent=[bunch_time[0], bunch_time[-1], cycle_time[0], cycle_time[-1]],
    )
    plt.xlabel("Time [ns]")
    plt.ylabel("Cycle time [ms]")
    plt.tight_layout()
    os.makedirs("results/EX_30_Xsuite_in_BLonD/", exist_ok=True)
    plt.savefig("results/EX_30_Xsuite_in_BLonD/triple_splitting.png")


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
