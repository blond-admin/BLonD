import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

# Import blond objects
from blond.legacy.blond2.beam.beam import Beam, Proton
from blond.legacy.blond2.beam.distributions import bigaussian
from blond.legacy.blond2.beam.profile import CutOptions, Profile
from blond.legacy.blond2.input_parameters.rf_parameters import RFStation
from blond.legacy.blond2.input_parameters.ring import Ring
from blond.legacy.blond2.llrf.beam_feedback import BeamFeedback
from blond.legacy.blond2.trackers.tracker import RingAndRFTracker

# from beam_dynamics_tools.beam_profiles.bunch_profile_tools import (
#     get_beam_pattern,
# )


DEBUG_PLOTTING = False

#
# def get_beam_pattern(
#     timeScale,
#     frames,
#     height_factor=0.015,
#     distance=500,
#     N_bunch_max=3564,
#     baseline_length=1,
#     BASE=False,
#     wind_len=10,
# ):
#     fit_option = "fwhm"
#     appy_tf = False


def get_beam_pattern(
    profiles,
    t,
    height_factor=0.015,
    distance=500,
    n_bunch_max=3564,
    wind_len=2.5e-9,
    single_turn=False,
):
    def interp_f(time, bunch, level):
        bunch_th = level * bunch.max()
        time_bet_points = time[1] - time[0]
        taux = np.where(bunch >= bunch_th)
        taux1, taux2 = taux[0][0], taux[0][-1]
        t1 = (
            time[taux1]
            - (bunch[taux1] - bunch_th)
            / (bunch[taux1] - bunch[taux1 - 1])
            * time_bet_points
        )
        t2 = (
            time[taux2]
            + (bunch[taux2] - bunch_th)
            / (bunch[taux2] - bunch[taux2 + 1])
            * time_bet_points
        )

        return t1, t2

    def intensity(y):
        offset_level = np.mean(y[0:5])
        return np.sum(y - offset_level)

    def fwhm(x, y, level=0.5):
        offset_level = np.mean(y[0:5])
        amp = np.max(y) - offset_level
        t1, t2 = interp_f(x, y, level)
        mu = (t1 + t2) / 2.0
        sigma = (t2 - t1) / 2.35482
        popt = (mu, sigma, amp)

        return popt

    if single_turn:
        profiles = np.array([profiles])

    dt = t[1] - t[0]

    fit_window = int(round(wind_len / dt / 2))
    n_frames = profiles.shape[0]

    n_bunches = np.zeros(n_frames, dtype=int)
    bunch_positions = np.zeros((n_frames, n_bunch_max))
    bunch_lengths = np.zeros((n_frames, n_bunch_max))
    bunch_peaks = np.zeros((n_frames, n_bunch_max))
    bunch_peak_position = np.zeros((n_frames, n_bunch_max))
    bunch_intensities = np.zeros((n_frames, n_bunch_max))

    for i in np.arange(n_frames):
        frame = profiles[i, :]

        pos, _ = find_peaks(frame, height=height_factor, distance=distance)
        n_bunches[i] = len(pos)

        for j, v in enumerate(pos):
            x = t[v - fit_window : v + fit_window]
            y = frame[v - fit_window : v + fit_window]

            try:
                (mu, sigma, amp) = fwhm(x, y, level=0.5)
            except:
                print(f"Something went wrong with bunch {j} at turn {i}...")
                mu, sigma, amp = 0, 0, 0

            bunch_lengths[i, j] = 4 * sigma
            bunch_positions[i, j] = mu
            bunch_peaks[i, j] = amp
            # bunch_peak_position[i, j] = peak_position(x, y, level=0.5)
            bunch_intensities[i, j] = intensity(y)

    n_bunch_max = np.max(n_bunches)
    bunch_peaks = bunch_peaks[:, :n_bunch_max]
    bunch_lengths = bunch_lengths[:, :n_bunch_max]
    bunch_positions = bunch_positions[:, :n_bunch_max]
    bunch_peak_position = bunch_peak_position[:, :n_bunch_max]
    bunch_intensities = bunch_intensities[:, :n_bunch_max]

    if single_turn:
        return (
            bunch_positions[0, :],
            bunch_lengths[0, :],
            bunch_peaks[0, :],
            bunch_peak_position[0, :],
            bunch_intensities[0, :],
        )

    return (
        bunch_positions,
        bunch_lengths,
        bunch_peaks,
        bunch_peak_position,
        bunch_intensities,
    )


def setup_blond2():
    # Options
    SAVE_SIM = True
    DISABLE_PL = False

    # Initialize the accelerator

    # The synchrotron ring
    C = 26658.8832  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    gamma_t = 53.8  # Transition gamma [-]
    alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor [-]
    n_turns = 2000  # Number of turns to track [-]

    ring = Ring(C, alpha, p_s, Proton(), n_turns=n_turns + 1)

    # The RF station
    h = 35640  # Harmonic number [-]
    V = 5e6  # RF voltage [V]
    dphi = 0  # Phase modulation/offset [rad]

    rfstation = RFStation(ring, [h], [V], [dphi], n_rf=1)

    # The beam
    number_of_bunches = 1  # Length of the batch [number of bunches]
    bunch_intensity = 1.6e11  # Bunch intensity [p/b]
    n_macroparticles = 1_000_000  # Number of macroparticles per bunch [-]
    tau_bunch = 1.2e-9  # Bunch length [s]
    bunch_spacing = 10  # Bunch spacing [number of rf buckets]
    injection_energy_error = 0  # Injection energy error [eV]
    injection_phase_error = 40

    # First generate a single gaussian bunch
    beam = Beam(ring, n_macroparticles, bunch_intensity)
    bigaussian(ring, rfstation, beam, sigma_dt=tau_bunch / 4, seed=1234)

    # Add final corrections to the bunch positions
    bucket_shift = 0
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

    if DEBUG_PLOTTING:
        # Plot profile
        profile.track()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(profile.bin_centers * 1e6, profile.n_macroparticles)
        ax.set_xlabel(r"$\Delta t$ [$\mu$s]")
        ax.set_ylabel(r"$\lambda (\Delta t)$ [arb. units]")
        ax.set_yticks([])

        plt.show()

    # Beam-phase loop
    # Beam Loops
    PL_gain = 1 / (5 * ring.t_rev[0]) * int(not DISABLE_PL)
    SL_gain = PL_gain / 10
    bl_config = {"machine": "LHC", "PL_gain": PL_gain, "SL_gain": SL_gain}

    beam_loop = BeamFeedback(ring, rfstation, profile, bl_config)

    # The RF tracker
    rftracker = RingAndRFTracker(
        rfstation,
        beam,
        Profile=profile,
        interpolation=True,
        BeamFeedback=beam_loop,
    )

    # Initialize data arrays
    bunch_length = np.zeros(n_turns)
    bunch_length_spread = np.zeros(n_turns)
    beam_loop_error = np.zeros(n_turns)
    synchro_loop_error = np.zeros(n_turns)

    omega_rf = np.zeros(n_turns)
    phi_rf = np.zeros(n_turns)

    for i in tqdm(range(n_turns)):
        profile.track()
        rftracker.track()

        bpos, blen, bpk, bpkpos, bint = get_beam_pattern(
            profile.n_macroparticles,
            profile.bin_centers,
            height_factor=100,
            distance=500,
            single_turn=True,
        )

        bunch_length[i] = np.mean(blen)
        bunch_length_spread[i] = np.std(blen)
        beam_loop_error[i] = beam_loop.dphi * 180 / np.pi
        synchro_loop_error[i] = rfstation.dphi_rf * 180 / np.pi
        omega_rf[i] = rfstation.omega_rf[0, i]
        phi_rf[i] = rfstation.phi_rf[0, i]

    if DEBUG_PLOTTING:
        beam_loop_error = (
            beam_loop_error - beam_loop_error[0] + injection_phase_error
        )

        plt.figure("Phase evolution")
        plt.plot(beam_loop_error, color="r", label="Beam-phase loop")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.xlim(0, n_turns - 1)

        plt.figure("Phase evolution")
        plt.plot(synchro_loop_error, color="r", label="Beam-phase loop")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.xlim(0, n_turns - 1)

        plt.show()

    if SAVE_SIM:
        if DISABLE_PL:
            np.savez(
                f"lhc_beam_control_{injection_phase_error:.1f}deg_nopl_new",
                unch_length=bunch_length,
                bunch_length_spread=bunch_length_spread,
                beam_loop_error=beam_loop_error,
                synchro_loop_error=synchro_loop_error,
                omega_rf=omega_rf,
                phi_rf=phi_rf,
            )
        else:
            np.savez(
                f"lhc_beam_control_{injection_phase_error:.1f}deg_new",
                bunch_length=bunch_length,
                bunch_length_spread=bunch_length_spread,
                beam_loop_error=beam_loop_error,
                synchro_loop_error=synchro_loop_error,
                omega_rf=omega_rf,
                phi_rf=phi_rf,
            )


if __name__ == "__main__":
    setup_blond2()
