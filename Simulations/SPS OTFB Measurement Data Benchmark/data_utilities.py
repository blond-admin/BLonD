import numpy as np
import matplotlib.pyplot as plt
from blond_common.fitting.profile import binomial_amplitudeN_fit, FitOptions
from blond_common.interfaces.beam.analytic_distribution import binomialAmplitudeN
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.interpolate import interp1d

def save_data(OTFB, dir):

    # 3-section signals:
    np.save(dir + '3sec_Vant', OTFB.OTFB_1.V_ANT[-OTFB.OTFB_1.n_coarse:])
    np.save(dir + '3sec_power', OTFB.OTFB_1.P_GEN[-OTFB.OTFB_1.n_coarse:])
    np.save(dir + '3sec_Igen', OTFB.OTFB_1.I_GEN[-OTFB.OTFB_1.n_coarse:])
    np.save(dir + '3sec_Vindgen', OTFB.OTFB_1.V_IND_COARSE_GEN[-OTFB.OTFB_1.n_coarse:])

    # 4-section signals
    np.save(dir + '4sec_Vant', OTFB.OTFB_2.V_ANT[-OTFB.OTFB_2.n_coarse:])
    np.save(dir + '4sec_power', OTFB.OTFB_2.P_GEN[-OTFB.OTFB_2.n_coarse:])
    np.save(dir + '4sec_Igen', OTFB.OTFB_2.I_GEN[-OTFB.OTFB_2.n_coarse:])
    np.save(dir + '4sec_Vindgen', OTFB.OTFB_2.V_IND_COARSE_GEN[-OTFB.OTFB_2.n_coarse:])



def pos_from_fwhm(profile, t, max_pos, window, N_interp):
    max_val = profile[max_pos]
    hm = max_val / 2
    sliced_prof = profile[max_pos - window: max_pos + window]
    sliced_t = t[max_pos - window: max_pos + window]

    # Find the measurements points closes to the half-max
    left_w = find_nearest_index(sliced_prof[:window], hm)              # max_pos is absolute which is wrong
    right_w = find_nearest_index(sliced_prof[window:], hm) + max_pos
    left_prof_points = profile[left_w - 1:left_w + 2]
    left_t_points = t[left_w - 1:left_w + 2]
    right_prof_points = profile[right_w - 1:right_w + 2]
    right_t_points = t[right_w - 1:right_w + 2]

    left_t_array = np.linspace(t[left_w - 1], t[left_w + 1], N_interp)
    right_t_array = np.linspace(t[right_w - 1], t[right_w + 1], N_interp)

    left_prof_interp = np.interp(left_t_array, left_prof_points, left_t_points)
    right_prof_interp = np.interp(right_t_array, right_prof_points, right_t_points)

    left_ind = find_nearest_index(left_prof_interp, hm)
    right_ind = find_nearest_index(right_prof_interp, hm)

    return (left_t_array[left_ind] + right_t_array[right_ind]) / 2


def bunch_by_bunch_deviation(profile, t, distance=20, height=0.15, N_batch = 1, from_fwhm=False):     # Continue writing lol
    dt = t[1] - t[0]
    pos, _ = find_peaks(profile, height=height, distance=distance)
    bunch_pos = t[pos]
    bunch_nr = np.linspace(1, len(pos), len(pos))
    bunch_per_batch = int(len(pos) / N_batch)

    if from_fwhm:
        for i in range(len(bunch_pos)):
            bunch_pos[i] = pos_from_fwhm(profile, t, pos[i], int(10e-9 / 2 / dt), 1000)

    bunch_pos = bunch_pos.reshape((N_batch, bunch_per_batch))
    bunch_nr = bunch_nr.reshape((N_batch, bunch_per_batch))
    fittet_line = np.zeros((N_batch, bunch_per_batch))
    bbb_dev = np.zeros((N_batch, bunch_per_batch))

    for i in range(N_batch):
        slope, intercept, r_val, p_val, std_err = linregress(bunch_nr[i,:], bunch_pos[i,:])
        fittet_line[i,:] = slope * bunch_nr[i,:] + intercept

        bbb_dev[i,:] = bunch_pos[i,:] - fittet_line[i,:]

    return bunch_pos.flatten(), fittet_line.flatten(), bbb_dev.flatten()


def import_profiles(dir, files, N_samples_per_file = 9999900):
    profile_datas = np.zeros((len(files), N_samples_per_file))
    profile_datas_corr = np.zeros((len(files), N_samples_per_file))
    n = 0
    for f in files:
        profile_datas[n,:] = np.load(dir + f + '.npy')

        conf_f = open(dir + f + '.asc', 'r')
        acq_params = conf_f.readlines()
        conf_f.close()

        delta_t = float(acq_params[6][39:-1])
        frame_length = [int(s) for s in acq_params[7].split() if s.isdigit()][0]
        N_frames = [int(s) for s in acq_params[8].split() if s.isdigit()][0]
        trigger_offsets = np.zeros(N_frames, )
        for line in np.arange(19, len(acq_params) - 1):
            print(float(acq_params[line][35:-2]), line)
            print((acq_params[line]))
            trigger_offsets[line - 20] = float(acq_params[line][35:-2])

        timeScale = np.arange(frame_length) * delta_t

        # data = np.load(fullpath)
        data = np.reshape(np.load(dir + f + '.npy'), (N_frames, frame_length))
        data_corr = np.zeros((N_frames, frame_length))

        for i in range(N_frames):
            x = timeScale + trigger_offsets[i]
            A = interp1d(x, data[i,:])
            data_corr[i,:] = A(timeScale)

        profile_datas_corr[n,:] = data_corr.flatten()
        n += 1

    return profile_datas, profile_datas_corr



def restack_turns(prof_array, t_array, T_rev):
    N_turns = int(t_array[-1] // T_rev)
    n_samples = int(len(t_array[np.where(t_array < T_rev)]))
    new_prof_array = np.zeros((N_turns, n_samples))
    new_t_array = np.zeros(((N_turns, n_samples)))

    for i in range(N_turns):
        new_prof_array[i,:] = prof_array[i * n_samples: (i + 1)*n_samples]
        new_t_array[i, :] = t_array[i * n_samples: (i + 1) * n_samples] - i * T_rev

    return new_prof_array, new_t_array



def find_nearest_index(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()



def find_first_bunch(prof, t, interval_t):
    interval = find_nearest_index(t, interval_t)
    max_ind = np.argmax(prof[:interval])

    return t[max_ind]



def find_first_bunch_interp(prof, t, max_t_interval, avg_t_interval, N_interp):
    # Find indices of interest
    max_interval = find_nearest_index(t, max_t_interval)
    avg_interval = find_nearest_index(t, avg_t_interval)
    prof_sliced = prof[:max_interval]
    max_ind = np.argmax(prof[:max_interval])

    # Find the half-maximum
    max_val = prof[max_ind]
    minimum_val = np.mean(prof[:avg_interval])
    half_max = (max_val - minimum_val)/2

    # Find the measurements points closes to the half-max
    left_w = find_nearest_index(prof_sliced[:max_ind], half_max)
    right_w = find_nearest_index(prof_sliced[max_ind:], half_max) + max_ind
    left_prof_points = prof[left_w - 1:left_w + 2]
    left_t_points = t[left_w - 1:left_w + 2]
    right_prof_points = prof[right_w - 1:right_w + 2]
    right_t_points = t[right_w - 1:right_w + 2]

    left_t_array = np.linspace(t[left_w - 1], t[left_w + 1], N_interp)
    right_t_array = np.linspace(t[right_w - 1], t[right_w + 1], N_interp)

    left_prof_interp = np.interp(left_t_array, left_prof_points, left_t_points)
    right_prof_interp = np.interp(right_t_array, right_prof_points, right_t_points)

    left_ind = find_nearest_index(left_prof_interp, half_max)
    right_ind = find_nearest_index(right_prof_interp, half_max)

    return (left_t_array[left_ind] + right_t_array[right_ind]) / 2




def fit_beam(prof, t, first_bunch_pos, bunch_spacing, bucket_length, batch_spacing, batch_length, N_bunches, N_interp):
    bucket_length_indices = int(bucket_length // ((t[1] - t[0]) * 2))
    bunch_spacing_indices = int(bunch_spacing // (t[1] - t[0]))
    batch_spacing_indices = int(batch_spacing // (t[1] - t[0]))

    next_peak = first_bunch_pos

    amplitudes = np.zeros(N_bunches)
    positions = np.zeros(N_bunches)
    full_lengths = np.zeros(N_bunches)
    exponents = np.zeros(N_bunches)

    for i in range(N_bunches):
        next_peak_index = find_nearest_index(t, next_peak)
        prof_i = prof[next_peak_index - bucket_length_indices: next_peak_index + bucket_length_indices]
        t_i = t[next_peak_index - bucket_length_indices: next_peak_index + bucket_length_indices]
        amplitudes[i], positions[i], full_lengths[i], exponents[i] = binomial_amplitudeN_fit(t_i, prof_i)

        if (i + 1) % batch_length == 0:
            prof_i = prof[next_peak_index + bucket_length_indices: next_peak_index + 2 * bucket_length_indices + batch_spacing_indices]
            t_i = t[next_peak_index + bucket_length_indices: next_peak_index + 2 * bucket_length_indices + batch_spacing_indices]
            if i != N_bunches - 1:
                next_peak = find_first_bunch_interp(prof_i, t_i, t_i[-1], t_i[-3 * bucket_length_indices], N_interp)

        else:
            prof_i = prof[next_peak_index + bucket_length_indices: next_peak_index + 2 * bucket_length_indices + bunch_spacing_indices]
            t_i = t[next_peak_index + bucket_length_indices: next_peak_index + 2 * bucket_length_indices + bunch_spacing_indices]
            if i != N_bunches - 1:
                next_peak = find_first_bunch_interp(prof_i, t_i, t_i[-1], t_i[-3 * bucket_length_indices], N_interp)

    return amplitudes, positions, full_lengths, exponents


def reject_outliers(data, m=2.):
    d = np.abs(data - np.mean(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def reject_outliers_2(data, m=2.):
    return data[np.abs(data - np.mean(data)) < m * np.std(data)]



def filtered_data_mean_and_std(data, m=2.):
    means = np.zeros((data.shape[1]))
    stds = np.zeros((data.shape[1]))

    for i in range(data.shape[1]):
        filtered_data_i = data[:,i]
        filtered_data_i = filtered_data_i[filtered_data_i < m]
        means[i] = np.mean(filtered_data_i)
        stds[i] = np.std(filtered_data_i)

    return means, stds



def fwhm(x, y, level=0.5):
    offset_level = np.mean(y[0:5])
    amp = np.max(y) - offset_level
    t1, t2 = interp_f(x, y, level)
    mu = (t1 + t2) / 2.0
    sigma = (t2 - t1) / 2.35482
    popt = (mu, sigma, amp)

    return popt


def interp_f(time, bunch, level):
    bunch_th = level * bunch.max()
    time_bet_points = time[1] - time[0]
    taux = np.where(bunch >= bunch_th)
    taux1, taux2 = taux[0][0], taux[0][-1]
    t1 = time[taux1] - (bunch[taux1] - bunch_th) / (bunch[taux1] - bunch[taux1 - 1]) * time_bet_points
    t2 = time[taux2] + (bunch[taux2] - bunch_th) / (bunch[taux2] - bunch[taux2 + 1]) * time_bet_points

    return t1, t2


def getBeamPattern_3(timeScale, frames, heightFactor=0.3, distance=500, N_bunch_max=3564,
                     fit_option='fwhm', plot_fit=False, baseline_length = 0):
    dt = timeScale[1] - timeScale[0]
    fit_window = int(round(10 * 1e-9 / dt / 2))
    N_frames = frames.shape[1]
    N_bunches = np.zeros((N_frames,), dtype=int)
    Bunch_positions = np.zeros((N_frames, N_bunch_max))
    Bunch_lengths = np.zeros((N_frames, N_bunch_max))
    Bunch_peaks = np.zeros((N_frames, N_bunch_max))
    Bunch_intensities = np.zeros((N_frames, N_bunch_max))
    Bunch_positionsFit = np.zeros((N_frames, N_bunch_max))
    Bunch_peaksFit = np.zeros((N_frames, N_bunch_max))
    Bunch_Exponent = np.zeros((N_frames, N_bunch_max))
    Goodness_of_fit = np.zeros((N_frames, N_bunch_max))

    for i in np.arange(N_frames):
        frame = frames[:, i]
        # pos, _ = find_peaks(frame,height=np.max(frames[:,i])*heightFactor,distance=distance)
        pos, _ = find_peaks(frame, height=0.015, distance=distance)
        N_bunches[i] = len(pos)
        Bunch_positions[i, 0:N_bunches[i]] = timeScale[pos]
        Bunch_peaks[i, 0:N_bunches[i]] = frame[pos]

        for j, v in enumerate(pos):
            x = 1e9 * timeScale[v - fit_window:v + fit_window]
            y = frame[v - fit_window:v + fit_window]
            baseline = np.mean(y[:baseline_length])

            y = y - baseline
            try:
                if fit_option == 'fwhm':
                    (mu, sigma, amp) = fwhm(x, y, level=0.5)
            #                    (mu2, sigma2, amp2) = fwhm(x,y,level=0.95)
                else:
                    (amp, mu, sigma, exponent) = binomial_amplitudeN_fit(x, y)
                    y_fit = binomialAmplitudeN(x, *[amp, mu, sigma, exponent])


                    if plot_fit: #or exponent > 5:
                        print(amp, mu, sigma, exponent)

                        plt.plot(x, y, label='measurement')
                        plt.plot(x, y_fit, label='fit')
                        plt.vlines(x[baseline_length], np.min(y), np.max(y), linestyle='--')
                        plt.legend()
                        plt.show()

                    sigma /= 4
            except:
                #plt.figure()
                #plt.plot(x, y, 'r', linewidth=3)
                pass

            Bunch_lengths[i, j] = 4 * sigma
            Bunch_intensities[i, j] = np.sum(y)
            Bunch_positionsFit[i, j] = mu
            Bunch_peaksFit[i, j] = amp
            if fit_option != 'fwhm':
                Bunch_Exponent[i, j] = exponent
                Goodness_of_fit[i, j] = np.mean(np.abs(y - y_fit)/np.max(y)) * 100

    N_bunches_max = np.max(N_bunches)
    Bunch_positions = Bunch_positions[:, 0:N_bunches_max]
    Bunch_peaks = Bunch_peaks[:, 0:N_bunches_max]
    Bunch_lengths = Bunch_lengths[:, 0:N_bunches_max]
    Bunch_intensities = Bunch_intensities[:, 0:N_bunches_max]
    Bunch_positionsFit = Bunch_positionsFit[:, 0:N_bunches_max]
    Bunch_peaksFit = Bunch_peaksFit[:, 0:N_bunches_max]
    Bunch_Exponent = Bunch_Exponent[:, 0:N_bunches_max]
    Goodness_of_fit = Goodness_of_fit[:, 0:N_bunches_max]

    return N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit


def getBeamPattern_4(timeScale, frames, heightFactor=0.3, distance=500, N_bunch_max=3564,
                     fit_option='fwhm', plot_fit=False, baseline_length = 0, d_interval = 1):
    dt = timeScale[1] - timeScale[0]
    fit_window = int(round(10 * 1e-9 / dt / 2))
    N_frames = frames.shape[1]
    N_bunches = np.zeros((N_frames,), dtype=int)
    Bunch_positions = np.zeros((N_frames, N_bunch_max))
    Bunch_lengths = np.zeros((N_frames, N_bunch_max))
    Bunch_peaks = np.zeros((N_frames, N_bunch_max))
    Bunch_intensities = np.zeros((N_frames, N_bunch_max))
    Bunch_positionsFit = np.zeros((N_frames, N_bunch_max))
    Bunch_peaksFit = np.zeros((N_frames, N_bunch_max))
    Bunch_Exponent = np.zeros((N_frames, N_bunch_max))
    Goodness_of_fit = np.zeros((N_frames, N_bunch_max))

    for i in np.arange(N_frames):
        frame = frames[:, i]
        # pos, _ = find_peaks(frame,height=np.max(frames[:,i])*heightFactor,distance=distance)
        pos, _ = find_peaks(frame, height=0.015, distance=distance)
        N_bunches[i] = len(pos)
        Bunch_positions[i, 0:N_bunches[i]] = timeScale[pos]
        Bunch_peaks[i, 0:N_bunches[i]] = frame[pos]

        for j, v in enumerate(pos):
            x = 1e9 * timeScale[v - fit_window:v + fit_window]
            y = frame[v - fit_window:v + fit_window]
            baseline = np.mean(y[:baseline_length])
            y = y - baseline

            (mu, sigma, amp) = fwhm(x, y, level=0.5)
            N = int(round(4 * sigma * y.shape[0] / (2 * (x[-1] - x[0])))) + d_interval
            peak_ind = np.argmax(y)
            y = y[peak_ind - N: peak_ind + N + 1]
            x = x[peak_ind - N: peak_ind + N + 1]


            if fit_option == 'fwhm':
                (mu, sigma, amp) = fwhm(x, y, level=0.5)
        #                    (mu2, sigma2, amp2) = fwhm(x,y,level=0.95)
            else:
                (amp, mu, sigma, exponent) = binomial_amplitudeN_fit(x, y)
                y_fit = binomialAmplitudeN(x, *[amp, mu, sigma, exponent])

                if y.shape[0] != y_fit.shape[0]:
                    print(y.shape, y_fit.shape, x.shape)

                if plot_fit: #or exponent > 5:
                    print(amp, mu, sigma, exponent)

                    plt.plot(x, y, label='measurement')
                    plt.plot(x, y_fit, label='fit')
                    #plt.vlines(x[baseline_length], np.min(y), np.max(y), linestyle='--')
                    plt.legend()
                    plt.show()

                sigma /= 4

            Bunch_lengths[i, j] = 4 * sigma
            Bunch_intensities[i, j] = np.sum(y)
            Bunch_positionsFit[i, j] = mu
            Bunch_peaksFit[i, j] = amp
            if fit_option != 'fwhm':
                Bunch_Exponent[i, j] = exponent
                Goodness_of_fit[i, j] = np.mean(np.abs(y - y_fit)/np.max(y)) * 100

    N_bunches_max = np.max(N_bunches)
    Bunch_positions = Bunch_positions[:, 0:N_bunches_max]
    Bunch_peaks = Bunch_peaks[:, 0:N_bunches_max]
    Bunch_lengths = Bunch_lengths[:, 0:N_bunches_max]
    Bunch_intensities = Bunch_intensities[:, 0:N_bunches_max]
    Bunch_positionsFit = Bunch_positionsFit[:, 0:N_bunches_max]
    Bunch_peaksFit = Bunch_peaksFit[:, 0:N_bunches_max]
    Bunch_Exponent = Bunch_Exponent[:, 0:N_bunches_max]
    Goodness_of_fit = Goodness_of_fit[:, 0:N_bunches_max]

    return N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit


def save_plots_OTFB(O, dir, i):

    # 3 section
    plt.figure()
    plt.suptitle(f'3sec, V antenna, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_1.V_ANT[-O.OTFB_1.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_1.V_ANT[-O.OTFB_1.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_1.V_ANT[-O.OTFB_1.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'3sec_V_ANT_turn{i}')

    plt.figure()
    plt.suptitle(f'3sec, I gen, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_1.I_GEN[-O.OTFB_1.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_1.I_GEN[-O.OTFB_1.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_1.I_GEN[-O.OTFB_1.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'3sec_I_GEN_turn{i}')

    plt.figure()
    plt.title(f'3sec, power, turn{i}')
    plt.plot(np.abs(O.OTFB_1.P_GEN[-O.OTFB_1.n_coarse:]), 'g', label='abs')
    plt.savefig(dir + f'3sec_P_GEN_turn{i}')

    plt.figure()
    plt.suptitle(f'3sec, V ind gen, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_1.V_IND_COARSE_GEN[-O.OTFB_1.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_1.V_IND_COARSE_GEN[-O.OTFB_1.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_1.V_IND_COARSE_GEN[-O.OTFB_1.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'3sec_V_IND_GEN_turn{i}')

    plt.figure()
    plt.suptitle(f'3sec, V ind beam, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_1.V_IND_COARSE_BEAM[-O.OTFB_1.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_1.V_IND_COARSE_BEAM[-O.OTFB_1.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_1.V_IND_COARSE_BEAM[-O.OTFB_1.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'3sec_V_IND_BEAM_turn{i}')


    # 4 section
    plt.figure()
    plt.suptitle(f'4sec, V antenna, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_2.V_ANT[-O.OTFB_2.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_2.V_ANT[-O.OTFB_2.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_2.V_ANT[-O.OTFB_2.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'4sec_V_ANT_turn{i}')

    plt.figure()
    plt.suptitle(f'4sec, I gen, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_2.I_GEN[-O.OTFB_2.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_2.I_GEN[-O.OTFB_2.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_2.I_GEN[-O.OTFB_2.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'4sec_I_GEN_turn{i}')

    plt.figure()
    plt.title(f'4sec, power, turn {i}')
    plt.plot(np.abs(O.OTFB_2.P_GEN[-O.OTFB_2.n_coarse:]), 'g', label='abs')

    plt.savefig(dir + f'4sec_P_GEN_turn{i}')

    plt.figure()
    plt.suptitle(f'4sec, V ind gen, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_2.V_IND_COARSE_GEN[-O.OTFB_2.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_2.V_IND_COARSE_GEN[-O.OTFB_2.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_2.V_IND_COARSE_GEN[-O.OTFB_2.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'4sec_V_IND_GEN_turn{i}')

    plt.figure()
    plt.suptitle(f'4sec, V ind beam, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_2.V_IND_COARSE_BEAM[-O.OTFB_2.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_2.V_IND_COARSE_BEAM[-O.OTFB_2.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_2.V_IND_COARSE_BEAM[-O.OTFB_2.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'4sec_V_IND_BEAM_turn{i}')


