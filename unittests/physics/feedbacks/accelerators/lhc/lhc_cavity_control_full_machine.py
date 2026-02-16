import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from blond.legacy.blond2.beam.beam import Beam, Proton
from blond.legacy.blond2.beam.distributions import (
    bigaussian,
)
from blond.legacy.blond2.beam.profile import CutOptions, FitOptions, Profile
from blond.legacy.blond2.impedances.impedance_sources import Resonators
from blond.legacy.blond2.input_parameters.rf_parameters import RFStation
from blond.legacy.blond2.input_parameters.ring import Ring
from blond.legacy.blond2.llrf.cavity_feedback import (
    LHCCavityLoop,
    LHCCavityLoopCommissioning,
)

batch_spacings = np.array(
    [
        330,
        80,
        80,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        80,
        80,
        0,
    ]
)


def setup_blond2():
    V_tot = 7.9e6
    N_p = 2.3e11
    Q_L = 20000
    tau = 1.25e-9
    batch_lengths = np.ones(38, dtype=int) * 72
    batch_lengths = np.concatenate(([12], batch_lengths), dtype=int)

    # Constants
    N_m = 50000  # Macro-particles
    C = 26658.8832  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    h = 35640  # Harmonic number
    dphi = 0  # Phase modulation/offset
    gamma_t = 53.8  # Transition gamma
    alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor

    R_over_Q = 45  # Cavity R/Q [Ohms]
    G_a = 6.79e-6  # Analog FB gain [A/V]
    G_d = 10  # Digital FB gain [-]
    tau_loop = 650e-9  # Overall loop delay [s]
    tau_a = 170e-6  # Analog FB delay [s]
    tau_d = 400e-6  # Digital FB delay [s]
    a_comb = 15 / 16  # Comb filter alpha [-]
    G_otfb = 10
    tau_comp = 1200e-9  # Complimentary delay in OTFB [s]
    G_gen = 1
    tau_o = 110e-6
    df_hd = -10.373079819809341e3

    injection_scheme = np.zeros(np.sum(batch_lengths), dtype=int)
    NB = len(injection_scheme)

    disable_plots = False

    ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=1)
    rf = RFStation(
        ring, [h], [V_tot], [dphi]
    )  # Assume filamented with SPS emittance
    bunch = Beam(ring, N_m, N_p)
    bigaussian(ring, rf, bunch, sigma_dt=tau / 4, seed=1234)

    beam = Beam(ring, N_m * NB, N_p * NB)
    buckets = rf.t_rf[0, 0] * 10

    n_batch = 0
    n_bunch = 0
    db = 0
    for i in range(len(injection_scheme)):
        injection_scheme[i] = db
        n_bunch += 1
        if n_bunch == batch_lengths[n_batch]:
            n_bunch = 0
            db += batch_spacings[n_batch]
            n_batch += 1
        else:
            db += 10

    for i in range(len(injection_scheme)):
        beam.dt[i * N_m : (i + 1) * N_m] = (
            bunch.dt[0:N_m]
            + 100 * buckets
            + injection_scheme[i] * rf.t_rf[0, 0]
        )
        beam.dE[i * N_m : (i + 1) * N_m] = bunch.dE[0:N_m]

    profile = Profile(
        beam,
        CutOptions(
            n_slices=int(2**6 * (35640)),
            cut_left=0,  # 80 * buckets,
            cut_right=rf.t_rev[
                0
            ],  # 80 * buckets + tot_buckets * rf.t_rf[0, 0]
        ),
    )
    profile.track()

    if not disable_plots:
        plt.figure()
        plt.plot(profile.bin_centers, profile.n_macroparticles)

    # DUMMY TO CALCULATE PEAK BEAM CURRENT
    RFFB = LHCCavityLoopCommissioning(
        G_a=G_a,
        G_d=G_d,
        tau_d=tau_d,
        tau_a=tau_a,
        alpha=a_comb,
        tau_o=tau_o,
        open_otfb=False,
        G_o=G_otfb,
        mu=-20,
        open_tuner=False,
        d_phi_ad=0,
        enable_klystron=False,
    )

    CL = LHCCavityLoop(
        rf_station=rf,
        profile=profile,
        f_c=rf.omega_rf[0, 0] / (2 * np.pi) + df_hd,
        I_gen_offset=0,
        n_cavities=8,
        n_pretrack=200,
        Q_L=Q_L,
        R_over_Q=R_over_Q,
        tau_loop=tau_loop,
        tau_otfb=tau_comp,
        G_gen=G_gen,
        RFFB=RFFB,
    )
    CL.disable_fine_grid = True

    n_detuning = 50
    detunings = np.zeros(n_detuning)

    logging.info(f"Tracking the cavity loop")
    time_now = time.time()
    for i in range(n_detuning):
        CL.track()
        detunings[i] = CL.detuning

    logging.info(
        f"Tracking takes {(time.time() - time_now) / n_detuning:.3f} s/turn"
    )

    transient = CL.generator_power()
    transient = transient * np.exp(1j * np.angle(CL.I_GEN_COARSE))

    if not disable_plots:
        plt.figure()
        plt.plot(np.abs(transient))
        plt.grid()

        # plt.xlim(0, tot_buckets // 10 + 200)

        plt.figure()
        plt.plot(detunings)
        plt.show()

    R_S = Q_L * 45
    resonator = Resonators(
        R_S=R_S, frequency_R=rf.omega_rf[0, 0] / (2 * np.pi), Q=Q_L
    )
    freq = np.linspace(
        rf.omega_rf[0, 0] / (2 * np.pi) - 25e3,
        rf.omega_rf[0, 0] / (2 * np.pi) + 25e3,
        int(1e6),
    )
    resonator.imped_calc(freq)

    logging.info("\n")

    np.savez(
        "lhc_rf_power_full_machine_new.npz",
        rf_power=transient[-CL.n_coarse :],
        rf_voltage=CL.V_ANT_COARSE[-CL.n_coarse :],
        rf_beam_current=CL.I_BEAM_COARSE[-CL.n_coarse :],
        profile_bin_centers=profile.bin_centers,
        profile_n_macroparticles=profile.n_macroparticles,
        detunings=detunings,
        rf_beam_current_fine=CL.I_BEAM_FINE[-profile.n_slices :],
        set_point=CL.V_SET[-CL.n_coarse :],
    )


if __name__ == "__main__":
    setup_blond2()
