'''
This file imports the data from the acquisition that were done in the SPS.

Author: Birk E. Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import data_utilities as dut

plt.rcParams.update({
    'text.usetex':True,
    'text.latex.preamble':r'\usepackage{fourier}',
    'font.family':'serif'
})


# Time Stamps ---------------------------------------------------------------------------------------------------------

# 2021-11-05
timestamps_2021_11_05_original = np.array([['095231', '095733', '103331', '104127', '104643'],
                                  ['100107', '103517', '104220'],
                                  ['100344', '103610', '104312'],
                                  ['102732', '103703', '104405'],
                                  ['095417', '134027', '103058', '103941', '104458'],
                                  ['133655', '103247', '104034', '104551']])

timestamps_2021_11_05 = np.array([['103331', '104127', '104643'],
                                  ['100107', '103517', '104220'],
                                  ['100344', '103610', '104312'],
                                  ['102732', '103703', '104405'],
                                  ['103058', '103941', '104458'],
                                  ['103247', '104034', '104551']])

# Find Data -----------------------------------------------------------------------------------------------------------

# The directory name of the data
data_dir = '/Users/bkarlsen/cernbox/SPSOTFB_benchmark_data/data/'

# Three different dates '2021-11-03/', '2021-11-04/', '2021-11-05/'
data_date = '2021-11-05/'
data_dir = data_dir + data_date

# Profile measurements (no measurements for 2021-11-03
profile_folder = 'profiles_SPS_OTFB_flattop/'

if not data_date == '2021-11-03/':
    profile_dir = data_dir + profile_folder

# Signal Aquisitions
cavitynumber = 1
date = '20211106'

# Open Data -----------------------------------------------------------------------------------------------------------

# Cavity signals
data_run = np.zeros((len(timestamps_2021_11_05[cavitynumber-1,:]),2**16))

for i in range(len(timestamps_2021_11_05[cavitynumber-1,:])):
    time = timestamps_2021_11_05[cavitynumber - 1,i]
    acq_file = f'sps_otfb_data__all_buffers__cavity{cavitynumber}__flattop__{date}_{time}.json'
    acq_file = data_dir + acq_file

    f = open(acq_file)
    data = json.load(f)
    print(data.keys())
    data_run[i, :] = data[f'SA.TWC200_expertVcavAmp.C{cavitynumber}-ACQ']['data']
    #data_run[i, :] = data[f'SA.TWC200_expertIcFwdPower.C{cavitynumber}-ACQ']['data']

# Profile measurements
profile_datas = np.zeros((25, 9999900))
n = 0
for i in range(104, 130):

    profile_file = profile_dir + f'MD_{i}.npy'
    if i != 112:
        profile_datas[n,:] = np.load(profile_file)
        n += 1

files = []
for i in range(104, 130):
    if i != 112:
        files.append(f'MD_{i}')

profile_datas, profile_datas_corr = dut.import_profiles(profile_dir, files)

# Data Manipulation ---------------------------------------------------------------------------------------------------
samplerate = 10e9
seconds_per_sample = 1 / samplerate
T_rev = 10e-6

DO_CALC = False
if DO_CALC:
    Bunch_positions_total = np.zeros((profile_datas.shape[0], 100, 288))
    Bunch_length_total = np.zeros((profile_datas.shape[0], 100, 288))
    Bunch_exponents_total = np.zeros((profile_datas.shape[0], 100, 288))
    Bunch_intensities_total = np.zeros((profile_datas.shape[0], 100, 288))
    Goodness_of_fit_total = np.zeros((profile_datas.shape[0], 100, 288))

    for i in range(profile_datas.shape[0]):
        print(i ,'out of', profile_datas.shape[0])
        profile_data1 = profile_datas[i,:]
        profile_data1 = np.reshape(profile_data1, (100, 99999))
        time_array = np.linspace(0, (99999 - 1) * seconds_per_sample, 99999)


        N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, Bunch_peaksFit, Bunch_exponent, Goodness_of_fit = \
            dut.getBeamPattern_4(time_array, profile_data1.T, distance=200, fit_option='binomial', plot_fit=False, baseline_length=35)

        Bunch_positions_total[i,:,:] = Bunch_positions
        Bunch_length_total[i,:,:] = Bunch_lengths
        Bunch_exponents_total[i,:,:] = Bunch_exponent
        Bunch_intensities_total[i,:,:] = Bunch_intensities
        Goodness_of_fit_total[i,:,:] = Goodness_of_fit


    np.save('goodness_of_fit_total_red', Goodness_of_fit_total)
    np.save('bunch_positions_total_red', Bunch_positions_total)
    np.save('bunch_length_total_red', Bunch_length_total)
    np.save('bunch_intensities_total_red', Bunch_intensities_total)
    np.save('bunch_exponents_total_red', Bunch_exponents_total)

PLOT_CALC = False
if PLOT_CALC:
    Bunch_exponents_total = np.load('bunch_exponents_total.npy')
    Bunch_length_total_full_length = np.load('bunch_length_total.npy')
    Bunch_length_total_FWHM = np.load('bunch_length_total_FWHM.npy')
    Bunch_intensities_total = np.load('bunch_intensities_total_red.npy')
    Bunch_positions_total = np.load('bunch_positions_total.npy')

    Bunch_exponents_total_flatten = Bunch_exponents_total.reshape((25 * 100, 288))
    Bunch_length_total_full_length_flatten = Bunch_length_total_full_length.reshape((25 * 100, 288))
    Bunch_length_total_FWHM_flatten = Bunch_length_total_FWHM.reshape((25 * 100, 288))
    Bunch_intensities_total_flatten = Bunch_intensities_total.reshape((25 * 100, 288))

    plt.figure()
    plt.title('Binomial Exponent')
    plt.imshow(Bunch_exponents_total_flatten, aspect='auto', origin='lower', vmin=1.5, vmax=6)
    plt.vlines(72, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 2, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 3, 0, 2500, color='r', linestyles='--')
    plt.ylim(0, 2500)
    plt.ylabel(r'Turn number')
    plt.xlabel(r'Bunch number')

    plt.figure()
    plt.title('Bunch Length, full length')
    plt.imshow(Bunch_length_total_full_length_flatten, aspect='auto', origin='lower', vmin=2, vmax=3)
    plt.vlines(72, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 2, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 3, 0, 2500, color='r', linestyles='--')
    plt.ylim(0, 2500)
    plt.ylabel(r'Turn number')
    plt.xlabel(r'Bunch number')

    plt.figure()
    plt.title('Bunch Length, FWHM')
    plt.imshow(Bunch_length_total_FWHM_flatten, aspect='auto', origin='lower', vmin=1.5, vmax=2.2)
    plt.vlines(72, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 2, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 3, 0, 2500, color='r', linestyles='--')
    plt.ylim(0, 2500)
    plt.ylabel(r'Turn number')
    plt.xlabel(r'Bunch number')

    plt.figure()
    plt.title('Bunch Intensities')
    plt.imshow(Bunch_intensities_total_flatten, aspect='auto', origin='lower', vmin=1, vmax=4)
    plt.vlines(72, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 2, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 3, 0, 2500, color='r', linestyles='--')
    plt.ylim(0, 2500)
    plt.ylabel(r'Turn number')
    plt.xlabel(r'Bunch number')

    plt.show()

    avg_bunch_exponents = np.mean(Bunch_exponents_total_flatten, axis=0)
    std_bunch_exponents = np.std(Bunch_exponents_total_flatten, axis=0)

    avg_bunch_length_full_length = np.mean(Bunch_length_total_full_length_flatten, axis=0)
    std_bunch_length_full_length = np.std(Bunch_length_total_full_length_flatten, axis=0)

    avg_bunch_length_FWHM = np.mean(Bunch_length_total_FWHM_flatten, axis=0)
    std_bunch_length_FWHM = np.std(Bunch_length_total_FWHM_flatten, axis=0)

    avg_bunch_intensities = np.mean(Bunch_intensities_total_flatten, axis=0)
    std_bunch_intensities = np.std(Bunch_intensities_total_flatten, axis=0)

    plt.figure()
    plt.title(r'Exponent')
    plt.plot(avg_bunch_exponents, '.', color='b')
    plt.fill_between(range(288), avg_bunch_exponents - std_bunch_exponents, avg_bunch_exponents + std_bunch_exponents, alpha=0.5)
    plt.vlines(72, np.min(avg_bunch_exponents - std_bunch_exponents), np.max(avg_bunch_exponents + std_bunch_exponents), color='r', linestyles='--')
    plt.vlines(72 * 2, np.min(avg_bunch_exponents - std_bunch_exponents), np.max(avg_bunch_exponents + std_bunch_exponents), color='r', linestyles='--')
    plt.vlines(72 * 3, np.min(avg_bunch_exponents - std_bunch_exponents), np.max(avg_bunch_exponents + std_bunch_exponents), color='r', linestyles='--')
    plt.xlabel(r'Bunch number')
    plt.ylabel(r'Exponent')

    plt.figure()
    plt.title(r'Bunch Length, full length')
    plt.plot(avg_bunch_length_full_length, '.', color='b')
    plt.fill_between(range(288), avg_bunch_length_full_length - std_bunch_length_full_length,
                     avg_bunch_length_full_length + std_bunch_length_full_length, alpha=0.5)
    plt.vlines(72, np.min(avg_bunch_length_full_length - std_bunch_length_full_length),
               np.max(avg_bunch_length_full_length + std_bunch_length_full_length), color='r', linestyles='--')
    plt.vlines(72 * 2, np.min(avg_bunch_length_full_length - std_bunch_length_full_length),
               np.max(avg_bunch_length_full_length + std_bunch_length_full_length), color='r', linestyles='--')
    plt.vlines(72 * 3, np.min(avg_bunch_length_full_length - std_bunch_length_full_length),
               np.max(avg_bunch_length_full_length + std_bunch_length_full_length), color='r', linestyles='--')
    plt.xlabel(r'Bunch number')
    plt.ylabel(r'Bunch length')

    plt.figure()
    plt.title(r'Bunch Length, FWHM')
    plt.plot(avg_bunch_length_FWHM, '.', color='b')
    plt.fill_between(range(288), avg_bunch_length_FWHM - std_bunch_length_FWHM, avg_bunch_length_FWHM + std_bunch_length_FWHM, alpha=0.5)
    plt.vlines(72, np.min(avg_bunch_length_FWHM - std_bunch_length_FWHM), np.max(avg_bunch_length_FWHM + std_bunch_length_FWHM), color='r', linestyles='--')
    plt.vlines(72 * 2, np.min(avg_bunch_length_FWHM - std_bunch_length_FWHM), np.max(avg_bunch_length_FWHM + std_bunch_length_FWHM), color='r', linestyles='--')
    plt.vlines(72 * 3, np.min(avg_bunch_length_FWHM - std_bunch_length_FWHM), np.max(avg_bunch_length_FWHM + std_bunch_length_FWHM), color='r', linestyles='--')
    plt.xlabel(r'Bunch number')
    plt.ylabel(r'Bunch length')

    plt.figure()
    plt.title(r'Bunch Intensities')
    plt.plot(avg_bunch_intensities, '.', color='b')
    plt.fill_between(range(288), avg_bunch_intensities - std_bunch_intensities, avg_bunch_intensities + std_bunch_intensities, alpha=0.5)
    plt.vlines(72, np.min(avg_bunch_intensities - std_bunch_intensities), np.max(avg_bunch_intensities + std_bunch_intensities), color='r', linestyles='--')
    plt.vlines(72 * 2, np.min(avg_bunch_intensities - std_bunch_intensities), np.max(avg_bunch_intensities + std_bunch_intensities), color='r', linestyles='--')
    plt.vlines(72 * 3, np.min(avg_bunch_intensities - std_bunch_intensities), np.max(avg_bunch_intensities + std_bunch_intensities), color='r', linestyles='--')
    plt.xlabel(r'Bunch number')
    plt.ylabel(r'Intensity')

    plt.show()

PLOT_CALC_REDJ = False
if PLOT_CALC_REDJ:
    Bunch_exponents_total = np.load('bunch_exponents_total_red.npy')
    Bunch_length_total_full_length = np.load('bunch_length_total_red.npy')
    Bunch_length_total_FWHM = np.load('bunch_length_total_FWHM.npy')
    Bunch_intensities_total = np.load('bunch_intensities_total_red.npy')
    Bunch_positions_total = np.load('bunch_positions_total_red.npy')

    Bunch_exponents_total_flatten = Bunch_exponents_total.reshape((25 * 100, 288))
    Bunch_length_total_full_length_flatten = Bunch_length_total_full_length.reshape((25 * 100, 288))
    Bunch_length_total_FWHM_flatten = Bunch_length_total_FWHM.reshape((25 * 100, 288))
    Bunch_intensities_total_flatten = Bunch_intensities_total.reshape((25 * 100, 288))


    avg_bunch_exponents, std_bunch_exponents = dut.filtered_data_mean_and_std(Bunch_exponents_total_flatten, 100)
    avg_bunch_length_full_length, std_bunch_length_full_length = dut.filtered_data_mean_and_std(Bunch_length_total_full_length_flatten, 50)

    avg_bunch_length_FWHM = np.mean(Bunch_length_total_FWHM_flatten, axis=0)
    std_bunch_length_FWHM = np.std(Bunch_length_total_FWHM_flatten, axis=0)

    avg_bunch_intensities = np.mean(Bunch_intensities_total_flatten, axis=0)
    std_bunch_intensities = np.std(Bunch_intensities_total_flatten, axis=0)

    plt.figure()
    plt.title(r'Exponent')
    plt.plot(avg_bunch_exponents, '.', color='b')
    plt.fill_between(range(288), avg_bunch_exponents - std_bunch_exponents, avg_bunch_exponents + std_bunch_exponents,
                     alpha=0.5)
    plt.vlines(72, np.min(avg_bunch_exponents - std_bunch_exponents), np.max(avg_bunch_exponents + std_bunch_exponents),
               color='r', linestyles='--')
    plt.vlines(72 * 2, np.min(avg_bunch_exponents - std_bunch_exponents),
               np.max(avg_bunch_exponents + std_bunch_exponents), color='r', linestyles='--')
    plt.vlines(72 * 3, np.min(avg_bunch_exponents - std_bunch_exponents),
               np.max(avg_bunch_exponents + std_bunch_exponents), color='r', linestyles='--')
    plt.xlabel(r'Bunch number')
    plt.ylabel(r'Exponent')

    plt.figure()
    plt.title(r'Bunch Length, full length')
    plt.plot(avg_bunch_length_full_length, '.', color='b')
    plt.fill_between(range(288), avg_bunch_length_full_length - std_bunch_length_full_length,
                     avg_bunch_length_full_length + std_bunch_length_full_length, alpha=0.5)
    plt.vlines(72, np.min(avg_bunch_length_full_length - std_bunch_length_full_length),
               np.max(avg_bunch_length_full_length + std_bunch_length_full_length), color='r', linestyles='--')
    plt.vlines(72 * 2, np.min(avg_bunch_length_full_length - std_bunch_length_full_length),
               np.max(avg_bunch_length_full_length + std_bunch_length_full_length), color='r', linestyles='--')
    plt.vlines(72 * 3, np.min(avg_bunch_length_full_length - std_bunch_length_full_length),
               np.max(avg_bunch_length_full_length + std_bunch_length_full_length), color='r', linestyles='--')
    plt.xlabel(r'Bunch number')
    plt.ylabel(r'Bunch length')

    plt.figure()
    plt.title(r'Bunch Length, FWHM')
    plt.plot(avg_bunch_length_FWHM, '.', color='b')
    plt.fill_between(range(288), avg_bunch_length_FWHM - std_bunch_length_FWHM,
                     avg_bunch_length_FWHM + std_bunch_length_FWHM, alpha=0.5)
    plt.vlines(72, np.min(avg_bunch_length_FWHM - std_bunch_length_FWHM),
               np.max(avg_bunch_length_FWHM + std_bunch_length_FWHM), color='r', linestyles='--')
    plt.vlines(72 * 2, np.min(avg_bunch_length_FWHM - std_bunch_length_FWHM),
               np.max(avg_bunch_length_FWHM + std_bunch_length_FWHM), color='r', linestyles='--')
    plt.vlines(72 * 3, np.min(avg_bunch_length_FWHM - std_bunch_length_FWHM),
               np.max(avg_bunch_length_FWHM + std_bunch_length_FWHM), color='r', linestyles='--')
    plt.xlabel(r'Bunch number')
    plt.ylabel(r'Bunch length')

    plt.figure()
    plt.title(r'Bunch Intensities')
    plt.plot(avg_bunch_intensities, '.', color='b')
    plt.fill_between(range(288), avg_bunch_intensities - std_bunch_intensities,
                     avg_bunch_intensities + std_bunch_intensities, alpha=0.5)
    plt.vlines(72, np.min(avg_bunch_intensities - std_bunch_intensities),
               np.max(avg_bunch_intensities + std_bunch_intensities), color='r', linestyles='--')
    plt.vlines(72 * 2, np.min(avg_bunch_intensities - std_bunch_intensities),
               np.max(avg_bunch_intensities + std_bunch_intensities), color='r', linestyles='--')
    plt.vlines(72 * 3, np.min(avg_bunch_intensities - std_bunch_intensities),
               np.max(avg_bunch_intensities + std_bunch_intensities), color='r', linestyles='--')
    plt.xlabel(r'Bunch number')
    plt.ylabel(r'Intensity')

    plt.show()

    np.save('avg_exponent_red', avg_bunch_exponents)
    np.save('avg_bunch_length_full_length_red', avg_bunch_length_full_length)
    np.save('avg_bunch_length_FWHM', avg_bunch_length_FWHM)
    np.save('avg_bunch_intensities_red', avg_bunch_intensities)


ANALYZE_GOODNESS = False
if ANALYZE_GOODNESS:
    goodness_of_fit = np.load('goodness_of_fit_total.npy')

    goodness_of_fit_flatten = goodness_of_fit.reshape((25 * 100, 288))

    avg_goodness = np.mean(goodness_of_fit_flatten, axis=0)
    std_goodness = np.std(goodness_of_fit_flatten, axis=0)

    plt.figure()
    plt.title(r'Goodness of binomial fit')
    plt.imshow(goodness_of_fit_flatten, aspect='auto', origin='lower', vmin=0, vmax=0.01)
    plt.vlines(72, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 2, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 3, 0, 2500, color='r', linestyles='--')
    plt.ylim(0, 2500)
    plt.ylabel(r'Turn number')
    plt.xlabel(r'Bunch number')

    plt.figure()
    plt.title(r'Goodness of binomial fit')
    plt.plot(avg_goodness, '.', color='b')
    plt.fill_between(range(288), avg_goodness - std_goodness,
                     avg_goodness + std_goodness, alpha=0.5)
    plt.vlines(72, np.min(avg_goodness - std_goodness),
               np.max(avg_goodness + std_goodness), color='r', linestyles='--')
    plt.vlines(72 * 2, np.min(avg_goodness - std_goodness),
               np.max(avg_goodness + std_goodness), color='r', linestyles='--')
    plt.vlines(72 * 3, np.min(avg_goodness - std_goodness),
               np.max(avg_goodness + std_goodness), color='r', linestyles='--')
    plt.xlabel(r'Bunch number')
    plt.ylabel(r'Goodness')

    plt.show()


CALC_POSITIONS = False
if CALC_POSITIONS:
    bunch_positions_total = np.load('bunch_positions_total_red.npy')
    bunch_positions_flatten = bunch_positions_total.reshape((25 * 100, 288))

    avg_position = np.mean(bunch_positions_flatten, axis=0)
    std_position = np.std(bunch_positions_flatten, axis=0)

    plt.figure()
    plt.title(r'Positions')
    plt.imshow(bunch_positions_flatten, aspect='auto', origin='lower',) #vmin=0, vmax=0.01)
    plt.vlines(72, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 2, 0, 2500, color='r', linestyles='--')
    plt.vlines(72 * 3, 0, 2500, color='r', linestyles='--')
    plt.ylim(0, 2500)
    plt.ylabel(r'Turn number')
    plt.xlabel(r'Bunch number')

    plt.figure()
    plt.title(r'Positions')
    plt.plot(avg_position, '.', color='b')
    plt.fill_between(range(288), avg_position - std_position,
                     avg_position + std_position, alpha=0.5)
    plt.vlines(72, np.min(avg_position - std_position),
               np.max(avg_position + std_position), color='r', linestyles='--')
    plt.vlines(72 * 2, np.min(avg_position - std_position),
               np.max(avg_position + std_position), color='r', linestyles='--')
    plt.vlines(72 * 3, np.min(avg_position - std_position),
               np.max(avg_position + std_position), color='r', linestyles='--')
    plt.xlabel(r'Bunch number')
    plt.ylabel(r'Position')

    plt.show()

    np.save('avg_positions_red', avg_position)

# Plot ----------------------------------------------------------------------------------------------------------------
time_array_profile = np.linspace(0, seconds_per_sample * len(profile_datas[0,:]), len(profile_datas[0,:]))

prof_stack, t_stack = dut.restack_turns(profile_datas[0,:], time_array_profile, T_rev)


PLOT_CAV_SIG = False
if PLOT_CAV_SIG:
    ti = np.linspace(0, 8e-9 * (data_run[0,:].shape[0] - 1), data_run[0,:].shape[0])
    for i in range(len(timestamps_2021_11_05[cavitynumber-1,:])):
        plt.plot(ti, data_run[i,:])

    plt.show()


BBB_DEV = True
if BBB_DEV:
    i = 0
    print(i ,'out of', profile_datas_corr.shape[0])
    profile_data1 = profile_datas_corr[i,:]
    profile_data1 = np.reshape(profile_data1, (100, 99999))
    time_array = np.linspace(0, (99999 - 1) * seconds_per_sample, 99999)


    bunch_pos, fitted_pos, bbb_dev = dut.bunch_by_bunch_deviation(profile_data1[0,:], time_array, N_batch=4, from_fwhm=True)

    plt.figure()
    plt.plot(bbb_dev)

    plt.figure()
    plt.plot(bunch_pos)
    plt.plot(fitted_pos)
    plt.show()





PLOT_PROFILE_MEASUREMENT = False
if PLOT_PROFILE_MEASUREMENT:


    first_bunch_pos = np.zeros(prof_stack.shape[0])
    amplitudes = np.zeros((72 * 4, prof_stack.shape[0]))
    positions = np.zeros((72 * 4, prof_stack.shape[0]))
    full_lengths = np.zeros((72 * 4, prof_stack.shape[0]))
    exponents = np.zeros((72 * 4, prof_stack.shape[0]))

    for i in range(prof_stack.shape[0]):
        first_bunch_pos[i] = dut.find_first_bunch_interp(prof_stack[i,:], t_stack[i,:], 2.77e-7, 2.3e-7, 5000)

    bunch_position_correction = first_bunch_pos[0] - first_bunch_pos
    t_stack_corrected = np.zeros(t_stack.shape)

    for i in range(prof_stack.shape[0]):
        t_stack_corrected[i,:] = t_stack[i,:] + bunch_position_correction[i]

    #for i in range(prof_stack.shape[0]):
    #    plt.plot(t_stack_corrected[i,:], prof_stack[i,:])
    #    plt.xlim(0, 3e-6)

    #plt.show()

    for i in range(prof_stack.shape[0]):
        amplitudes[:,i], positions[:,i], full_lengths[:,i], exponents[:,i] = dut.fit_beam(
            prof_stack[i,:], t_stack_corrected[i,:], first_bunch_pos[0], 25e-9, 10e-9, 250e-9, 72, 72 * 4, 1000
        )
        if i % 10 == 0:
            print(i)

    np.save('amplitudes_f10ns', amplitudes)
    np.save('positions_f10ns', positions)
    np.save('full_lengths_f10ns', full_lengths)
    np.save('exponents_f10ns', exponents)

PLOT_ANALYSIS = False
if PLOT_ANALYSIS:
    amplitudes = np.load('amplitudes.npy')
    positions = np.load('positions.npy')
    full_lengths = np.load('full_lengths.npy')
    exponents = np.load('exponents.npy')

    plt.plot(amplitudes[:,1])
    plt.title('Amplitude')
    plt.show()

    plt.plot(positions[:,1])
    plt.title('Positions')
    plt.show()

    plt.plot(full_lengths[:,1])
    plt.title('Full Lengths')
    plt.show()

    plt.plot(exponents[:,1])
    plt.title('Exponents')
    plt.show()


FURTHER_ANALYSIS = False
if FURTHER_ANALYSIS:
    amplitudes = np.load('amplitudes_f10ns.npy')
    positions = np.load('positions_f10ns.npy')
    full_lengths = np.load('full_lengths_f10ns.npy')
    exponents = np.load('exponents.npy')

    # Find beam average over the 99 turns
    avg_amplitudes = np.mean(amplitudes, axis=1)
    avg_positions = np.mean(positions, axis=1)
    avg_full_lengths = np.mean(full_lengths, axis=1)
    avg_exponents = np.mean(exponents, axis=1)

    plt.plot(avg_exponents)
    plt.show()

