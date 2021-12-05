import numpy as np
import matplotlib.pyplot as plt
import data_utilities as dut
from scipy.signal import find_peaks


plt.rcParams.update({
    'text.usetex':True,
    'text.latex.preamble': r'\usepackage{fourier}',
    'font.family': 'serif'
})


def plot_comparison(title, gen, meas):
    plt.figure()
    plt.title(title)
    plt.plot(gen, '.', label=r'Generated')
    plt.plot(meas, '.', label=r'Measured')
    plt.legend()


def import_measured_profile():
    # The directory name of the data
    data_dir = '/Users/bkarlsen/cernbox/SPSOTFB_benchmark_data/data/'

    # Three different dates '2021-11-03/', '2021-11-04/', '2021-11-05/'
    data_date = '2021-11-05/'
    data_dir = data_dir + data_date

    # Profile measurements (no measurements for 2021-11-03
    profile_folder = 'profiles_SPS_OTFB_flattop/'

    if not data_date == '2021-11-03/':
        profile_dir = data_dir + profile_folder

    # Profile measurements
    profile_datas = np.zeros((25, 9999900))
    n = 0
    for i in range(104, 130):

        profile_file = profile_dir + f'MD_{i}.npy'
        if i != 112:
            profile_datas[n, :] = np.load(profile_file)
            n += 1

    samplerate = 10e9
    seconds_per_sample = 1 / samplerate

    profile_data1 = profile_datas[0, :]
    profile_data1 = np.reshape(profile_data1, (100, 99999))
    time_array = np.linspace(0, (99999 - 1) * seconds_per_sample, 99999)

    return profile_data1, time_array




gen_prof = np.load('generated_profile_fwhm.npy')
gen_bin = np.load('generated_profile_bins_fwhm.npy')


bunch_intensities = np.load('avg_bunch_intensities_red.npy')
bunch_lengths_fl = np.load('avg_bunch_length_full_length_red.npy')
bunch_lengths_fwhm = np.load('avg_bunch_length_FWHM.npy')
exponents = np.load('avg_exponent_red.npy')
positions = np.load('avg_positions_red.npy')


gen_prof = np.array([gen_prof])

N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, Bunch_peaksFit, Bunch_exponent, Goodness_of_fit = \
            dut.getBeamPattern_4(gen_bin, gen_prof.T, distance=150, fit_option='fwhm', plot_fit=False, baseline_length=25,) #d_interval=3)


exp_pos = np.linspace(0, 71 * 5 * 5e-9, 72)
for i in range(3):
    exp_pos = np.concatenate((exp_pos, np.linspace(exp_pos[-1] + 250e-9, exp_pos[-1] + 250e-9 + 71 * 5 * 5e-9, 72)))
norm_pos = Bunch_positions[0,:] - Bunch_positions[0,0] - exp_pos
norm_pos_meas = positions - positions[0] - exp_pos


bunch_pos, fittet_pos, bbb_dev = dut.bunch_by_bunch_deviation(gen_prof[0,:], gen_bin, 30, 100, 4, True)


plot_comparison('length', Bunch_lengths[0,:], bunch_lengths_fwhm)

bunch_intensities = 3385.8196 * 10**10 * bunch_intensities / np.sum(bunch_intensities)
Bunch_intensities[0,:] = 3385.8196 * 10**10 * Bunch_intensities[0,:] / np.sum(Bunch_intensities[0,:])

plot_comparison('intensity', Bunch_intensities[0,:], bunch_intensities)

plt.show()