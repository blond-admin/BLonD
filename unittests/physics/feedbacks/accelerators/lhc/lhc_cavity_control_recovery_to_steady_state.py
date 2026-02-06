import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import blond objects


def setup_b2():
    from blond.legacy.blond2.beam.beam import Beam, Proton
    from blond.legacy.blond2.beam.distributions import bigaussian, parabolic
    from blond.legacy.blond2.beam.profile import CutOptions, Profile
    from blond.legacy.blond2.input_parameters.rf_parameters import RFStation
    from blond.legacy.blond2.input_parameters.ring import Ring
    from blond.legacy.blond2.llrf.beam_feedback import BeamFeedback
    from blond.legacy.blond2.llrf.cavity_feedback import (
        LHCCavityLoop,
        LHCCavityLoopCommissioning,
    )
    from blond.legacy.blond2.trackers.tracker import RingAndRFTracker
    from blond.legacy.blond2.utils import bmath as bm

    bm.use_cpp()

    # Options
    PLT_SIMS = False
    SAVE_SIM = True
    DISABLE_PL = False

    data_folder = "data/convergence_to_steadystate/"

    # Initialize the accelerator

    # The synchrotron ring
    C = 26658.8832  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    gamma_t = 53.606713  # Transition gamma [-]
    alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor [-]
    n_turns = 500  # Number of turns to track [-]

    ring = Ring(C, alpha, p_s, Proton(), n_turns=n_turns + 1)

    # The RF station
    h = 35640  # Harmonic number [-]
    V = 5e6  # RF voltage [V]
    dphi = 0  # Phase modulation/offset [rad]

    rfstation = RFStation(ring, [h], [V], [dphi], n_rf=1)

    # The beam
    number_of_bunches = 36  # Length of the batch [number of bunches]
    bunch_intensity = 1.6e11  # Bunch intensity [p/b]
    n_macroparticles = 100_000  # Number of macroparticles per bunch [-]
    tau_bunch = 1.2e-9  # Bunch length [s]
    bunch_spacing = 10  # Bunch spacing [number of rf buckets]
    injection_energy_error = 0  # Injection energy error [eV]
    injection_phase_error = 40
    bucket_shift = 1000

    # Beam object for the batch
    N_m = n_macroparticles * number_of_bunches
    N_p = bunch_intensity * number_of_bunches
    beam = Beam(ring, N_m, N_p)

    # First generate a single gaussian bunch
    single_bunch = Beam(ring, n_macroparticles, bunch_intensity)
    bigaussian(
        ring, rfstation, single_bunch, sigma_dt=tau_bunch / 4, seed=1234
    )

    # Copy the bunch throughout the batch
    for i in range(number_of_bunches):
        beam.dE[i * n_macroparticles : (i + 1) * n_macroparticles] = (
            single_bunch.dE
        )
        beam.dt[i * n_macroparticles : (i + 1) * n_macroparticles] = (
            single_bunch.dt + i * bunch_spacing * rfstation.t_rf[0, 0]
        )

    # Add final corrections to the bunch positions
    bucket_shift = 10000
    beam.dt += (
        bucket_shift * rfstation.t_rf[0, 0]
        + injection_phase_error * rfstation.t_rf[0, 0] / 360
    )
    beam.dE += injection_energy_error

    # The beam profile
    cut_options = CutOptions(
        cut_left=(-5.5 + bucket_shift) * rfstation.t_rf[0, 0],
        cut_right=(6.5 + number_of_bunches * bunch_spacing + bucket_shift)
        * rfstation.t_rf[
            0,
            0,
        ],
        n_slices=(10 * number_of_bunches + 12) * 2**5,
    )
    profile = Profile(beam, cut_options)

    # Plot profile
    if PLT_SIMS:
        profile.track()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(profile.bin_centers * 1e6, profile.n_macroparticles)
        ax.set_xlabel(r"$\Delta t$ [$\mu$s]")
        ax.set_ylabel(r"$\lambda (\Delta t)$ [arb. units]")
        ax.set_yticks([])

        plt.show()

    # Cavity Controller
    G_a = 6.79e-6  # Analog FB gain [A/V]
    G_d = 10  # Digital FB gain [-]
    tau_loop = 650e-9  # Overall loop delay [s]
    tau_a = 170e-6  # Analog FB delay [s]
    tau_d = 400e-6  # Digital FB delay [s]
    a_comb = 15 / 16  # Comb filter alpha [-]
    Q_L = 20000  # Loaded Quality factor [-]
    G_otfb = 10  # OTFB gain [-]
    tau_comp = 1200e-9  # Complimentary delay in OTFB [s]
    delta_f = -3480  # Initial detuning due to 12 bunches [Hz]

    commissioning = LHCCavityLoopCommissioning(
        G_a=G_a,
        G_d=G_d,
        tau_d=tau_d,
        tau_a=tau_a,
        alpha=a_comb,
        G_o=G_otfb,
        open_tuner=True,
        open_rffb=False,
        enable_klystron=False,
    )

    cavity_loop = LHCCavityLoop(
        rfstation,
        profile,
        RFFB=commissioning,
        f_c=rfstation.omega_rf[0, 0] / (2 * np.pi) + delta_f,
        Q_L=Q_L,
        tau_loop=tau_loop,
        tau_otfb=tau_comp,
        n_pretrack=200,
        n_cavities=8,
        n_h=0,
    )

    # Beam-phase loop
    # Beam Loops
    PL_gain = 1 / (5 * ring.t_rev[0]) * int(not DISABLE_PL)
    SL_gain = PL_gain / 10
    bl_config = {"machine": "LHC", "PL_gain": PL_gain, "SL_gain": SL_gain}

    beam_loop = BeamFeedback(
        ring,
        rfstation,
        profile,
        bl_config,
        CavityFeedback=cavity_loop,
        current_thres=0.5,
    )

    # The RF tracker
    rftracker = RingAndRFTracker(
        rfstation,
        beam,
        Profile=profile,
        interpolation=True,
        BeamFeedback=beam_loop,
        CavityFeedback=cavity_loop,
    )

    # Initialize data arrays
    rf_power = np.zeros((n_turns, cavity_loop.n_coarse), dtype=complex)
    rf_voltage = np.zeros((n_turns, cavity_loop.n_coarse), dtype=complex)
    rf_beam_current = np.zeros((n_turns, cavity_loop.n_coarse), dtype=complex)
    rf_beam_current_phase = np.zeros((n_turns, number_of_bunches))
    beam_loop_phase = np.zeros(n_turns)

    print(profile.bin_size * 1e12)

    # if DISABLE_PL:
    #    profile.track()
    #    beam_loop.track()
    # Tracking
    profile.track()
    line_density = np.copy(profile.n_macroparticles)
    bin_centers = np.copy(profile.bin_centers)

    for i in tqdm(range(n_turns)):
        profile.track()
        rftracker.track()
        cavity_loop.generator_power()

        if i == 0:
            rf_beam_current_fine = cavity_loop.I_BEAM_FINE[-profile.n_slices :]

        rf_power[i, :] = cavity_loop.generator_power()[-cavity_loop.n_coarse :]
        rf_voltage[i, :] = cavity_loop.V_ANT_COARSE[-cavity_loop.n_coarse :]
        rf_beam_current[i, :] = cavity_loop.I_BEAM_COARSE[
            -cavity_loop.n_coarse :
        ]
        beam_loop_phase[i] = beam_loop.phi_beam * 180 / np.pi
        rf_beam_current_phase[i, :] = -np.angle(
            cavity_loop.I_BEAM_COARSE[
                cavity_loop.n_coarse
                + bucket_shift // 10 : cavity_loop.n_coarse
                + bucket_shift // 10
                + number_of_bunches
            ]
        )

    rf_beam_current_phase = np.mean(
        np.unwrap(rf_beam_current_phase) * 180 / np.pi, axis=1
    )
    rf_beam_current_phase = (
        rf_beam_current_phase
        - rf_beam_current_phase[0]
        + injection_phase_error
    )
    beam_loop_phase = (
        beam_loop_phase - beam_loop_phase[0] + injection_phase_error
    )

    if PLT_SIMS:
        plt.figure("Phase evolution")
        plt.plot(rf_beam_current_phase, color="black", label="RF beam current")
        plt.plot(beam_loop_phase, color="r", label="Beam-phase loop")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.xlim(0, n_turns - 1)

        plt.figure("Phase difference")
        plt.plot(
            100 * (rf_beam_current_phase - beam_loop_phase) / beam_loop_phase
        )
        plt.tight_layout()
        plt.grid()
        plt.xlim(0, n_turns - 1)

        plt.show()

    if SAVE_SIM:
        if DISABLE_PL:
            np.savez(
                f"generate_blond2_data/feedbacks/lhc/data/lhc_convergence_to_steadystate_{injection_phase_error:.1f}deg_nopl",
                rf_power=rf_power,
                rf_voltage=rf_voltage,
                rf_beam_current=rf_beam_current,
                beam_loop_phase=beam_loop_phase,
                rf_beam_current_phase=rf_beam_current_phase,
                line_density=line_density,
                bin_centers=bin_centers,
                rf_beam_current_fine=rf_beam_current_fine,
            )
        else:
            np.savez(
                f"lhc_convergence_to_steadystate_{injection_phase_error:.1f}deg",
                rf_power=rf_power,
                rf_voltage=rf_voltage,
                rf_beam_current=rf_beam_current,
                beam_loop_phase=beam_loop_phase,
                rf_beam_current_phase=rf_beam_current_phase,
                line_density=line_density,
                bin_centers=bin_centers,
                rf_beam_current_fine=rf_beam_current_fine,
            )


if __name__ == "__main__":
    setup_b2()
