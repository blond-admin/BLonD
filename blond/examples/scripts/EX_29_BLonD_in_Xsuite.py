# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
PS triple splitting with xsuite owning the loop and BLonD the cavity.

The same physics as ``EX_10_MultiRFManipulation_TripleSplitting``, but the
tracking loop belongs to xsuite: an ``xt.LineSegmentMap`` provides the
longitudinal drift (momentum compaction) while the BLonD
``MultiHarmonicRFStation`` — wrapped with ``WrapBlond4Xsuite`` — applies the
h=7/14/21 splitting kicks. The h7 -> h14 -> h21 voltage programs are fed to
the BLonD station as a *turn-keyed* schedule (``ScheduledArray``): the
wrapper drives the station's turn counter from xsuite's
``particles.at_turn``, so no BLonD ``Simulation`` is involved at all.

See ``EX_30_Xsuite_in_BLonD`` for the mirrored setup where BLonD owns the
loop and xsuite provides the drift.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt
from scipy.constants import c

from blond import (
    MultiHarmonicRFStation,
    ScheduledArray,
    backend,
    copy_to_cpu,
    momentum_compaction_factor,
    proton,
)
from blond.interfaces.xsuite.elements import WrapBlond4Xsuite

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# No ``setup_backend`` here: xsuite owns the loop, and ``WrapBlond4Xsuite``
# follows the particles' xobjects context (switching BLonD to cuda itself
# when the particles live on a ``ContextCupy``).


def main(
    run_n_turns: int | None = None,
    n_macroparticles: int = int(1e4),
    each_turn_i_profile: int = 100,
) -> list[int]:
    momentum = 3.9051e9
    circumference = 2 * np.pi * 100
    transition_gamma = 6.1
    bunch_population = 2.6e11 * 12
    harmonics = np.array([7.0, 14.0, 21.0])
    phi_rf = np.array([0.0, np.pi, 0.0])

    total_energy = float(np.sqrt(momentum**2 + proton.mass**2))
    beta0 = momentum / total_energy
    gamma = total_energy / proton.mass
    t_rev = circumference / (beta0 * c)
    t_rf = t_rev / harmonics[0]
    alpha = momentum_compaction_factor(transition_gamma)

    # ------------------------------------------------------------------
    # Voltage programs (time-based files) -> one voltage triplet per turn.
    # xsuite drives the loop, so the schedule must be turn-keyed: at
    # constant energy turn n simply happens at t = n * t_rev.
    # ------------------------------------------------------------------
    programs = [
        np.loadtxt(
            f"{this_directory}/resources/EX_10_programs/voltage_h{h:.0f}.txt"
        )
        for h in harmonics
    ]
    program_duration = programs[0][-1, 0] * 1e-3
    n_turns_program = int(program_duration / t_rev)
    n_turns = n_turns_program if run_n_turns is None else run_n_turns

    turn_times = np.arange(n_turns + 1) * t_rev
    voltage_per_turn = np.column_stack(
        [
            np.interp(turn_times, program[:, 0] * 1e-3, program[:, 1] * 1e3)
            for program in programs
        ]
    )

    # ------------------------------------------------------------------
    # BLonD cavity: headless (no Simulation, no magnetic cycle — xsuite
    # owns the reference) with the per-turn voltage schedule attached
    # before wrapping.
    # ------------------------------------------------------------------
    rf_station = MultiHarmonicRFStation.headless(
        section_index=0,
        voltage=voltage_per_turn[0],
        phi_rf=phi_rf,
        harmonic=harmonics,
        circumference=circumference,
        main_harmonic_idx=0,
        beam_reference_beta=beta0,
        magnetic_cycle=None,
    )
    rf_station.schedule(
        attribute="voltage", value=ScheduledArray(voltage_per_turn)
    )

    # ------------------------------------------------------------------
    # xsuite line: the segment map is the drift (momentum compaction),
    # the wrapped BLonD station is the only RF. The context follows
    # BLonD's backend so the example also runs on the GPU.
    # ------------------------------------------------------------------
    context = xo.ContextCupy() if backend.is_gpu else xo.ContextCpu()
    drift = xt.LineSegmentMap(
        longitudinal_mode="nonlinear",
        qx=1.1,
        qy=1.2,
        betx=1.0,
        bety=1.0,
        voltage_rf=0.0,
        frequency_rf=0.0,
        lag_rf=0.0,
        momentum_compaction_factor=alpha,
        length=circumference,
    )
    line = xt.Line(
        elements=[drift, WrapBlond4Xsuite(element=rf_station)],
        element_names=["drift", "blond_cavity"],
    )
    line.particle_ref = xt.Particles(
        p0c=momentum, mass0=proton.mass, q0=proton.charge
    )
    line.build_tracker(_context=context)

    # ------------------------------------------------------------------
    # Approximately matched Gaussian bunch in the initial h=7 bucket.
    # ------------------------------------------------------------------
    eta = abs(alpha - 1 / gamma**2)
    omega_s = (2 * np.pi / t_rev) * np.sqrt(
        harmonics[0]
        * voltage_per_turn[0, 0]
        * eta
        / (2 * np.pi * beta0**2 * total_energy)
    )
    sigma_dt = t_rf / 8
    sigma_dE = omega_s * beta0**2 * total_energy / eta * sigma_dt

    rng = np.random.default_rng(seed=42)
    dt = rng.normal(0.0, sigma_dt, n_macroparticles)
    dE = rng.normal(0.0, sigma_dE, n_macroparticles)
    particles = xt.Particles(
        _context=context,
        p0c=momentum,
        mass0=proton.mass,
        q0=proton.charge,
        zeta=-dt * beta0 * c,
        ptau=dE / (beta0 * total_energy),
        weight=np.full(n_macroparticles, bunch_population / n_macroparticles),
    )

    # ------------------------------------------------------------------
    # Track in chunks, recording a bunch profile every
    # ``each_turn_i_profile`` turns (the BLonD observation machinery
    # belongs to its Simulation, so the waterfall is built by hand here).
    # ------------------------------------------------------------------
    bin_edges = np.linspace(-t_rf / 2, t_rf / 2, 2**11 + 1)

    def bunch_profile() -> np.ndarray:
        state = copy_to_cpu(particles.state)
        zeta = copy_to_cpu(particles.zeta)
        dt_now = -zeta[state > 0] / (beta0 * c)
        histogram, _ = np.histogram(dt_now, bins=bin_edges)
        return histogram

    profiles = [bunch_profile()]
    turns_recorded = [0]
    for turn in range(each_turn_i_profile, n_turns + 1, each_turn_i_profile):
        line.track(particles, num_turns=each_turn_i_profile)
        profiles.append(bunch_profile())
        turns_recorded.append(turn)
    remainder = n_turns - turns_recorded[-1]
    if remainder > 0:
        line.track(particles, num_turns=remainder)
        profiles.append(bunch_profile())
        turns_recorded.append(n_turns)

    bunch_time = 0.5 * (bin_edges[1:] + bin_edges[:-1]) * 1e9
    cycle_time = np.asarray(turns_recorded) * t_rev * 1e3
    plt.figure("bunch evolution")
    plt.clf()
    plt.imshow(
        np.asarray(profiles),
        aspect="auto",
        origin="lower",
        cmap="turbo",
        extent=[bunch_time[0], bunch_time[-1], cycle_time[0], cycle_time[-1]],
    )
    plt.xlabel("Time [ns]")
    plt.ylabel("Cycle time [ms]")
    plt.tight_layout()
    os.makedirs("results/EX_29_BLonD_in_Xsuite/", exist_ok=True)
    plt.savefig("results/EX_29_BLonD_in_Xsuite/triple_splitting.png")

    return turns_recorded


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
