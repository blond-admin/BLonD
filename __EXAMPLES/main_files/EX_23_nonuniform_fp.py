#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:33:30 2020

@author: MarkusArbeit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e

# BLonD imports
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.sparse_slices import SparseSlices
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage, \
    InducedVoltageSparse
from blond.impedances.induced_voltage_analytical import \
    analytical_gaussian_resonator
from blond.impedances.impedance_sources import Resonators
from blond.beam.distributions import bigaussian
from blond.impedances.impedance import sample_function, FourierTransform


def _compute_impedance(frequency, impedance_source_list):
    """ helper function to compute real part of impedance """
    impedance = np.zeros_like(frequency, dtype=complex)

    for impedance_source in impedance_source_list:
        impedance_source.imped_calc(frequency)
        impedance += impedance_source.impedance

    return impedance


np.random.seed(1984)

intensity_pb = 1.0e11
sigma = 0.2e-9  # bunch RMS

n_macroparticles_pb = int(1e4)
n_bunches =3
bunch_spacing = 5  # RF buckets

### --- Ring and RF ----------------------------------------------
intensity = n_bunches * intensity_pb  # total intensity SPS
n_turns = 1
# Ring parameters SPS
circumference = 6911.5038  # Machine circumference [m]
sync_momentum = 25.92e9  # SPS momentum at injection [eV/c]

gamma_transition = 17.95142852  # Q20 Transition gamma
momentum_compaction = 1. / gamma_transition ** 2  # Momentum compaction array

ring = Ring(circumference, momentum_compaction, sync_momentum, Proton(),
            n_turns=n_turns)
tRev = ring.t_rev[0]

# RF parameters SPS
rf_station = RFStation(ring, 4620, 3.5e6, 0, n_rf=1)
t_rf = rf_station.t_rf[0, 0]

### two resonators with a short and long range wake field, respectively

R_s = np.asarray([1e6, 1e6])
f_r = np.asarray([200e6, 400e6])
Q = np.array([0.5, 4]) * np.pi * f_r * t_rf

# use `python` since the C code does not allow to compute Z(0)=0 yet
impedance_model = [Resonators(R_s, f_r, Q, method='python')]

### create `n_bunches` Gaussian bunches spaced `bunch_spacing` RF-buckets apart
n_macroparticles = n_bunches * n_macroparticles_pb
beam = Beam(ring, n_macroparticles, intensity)

for bunch in range(n_bunches):
    bunchBeam = Beam(ring, n_macroparticles_pb, intensity_pb)
    bigaussian(ring, rf_station, bunchBeam, sigma, reinsertion=True,
               seed=1984 + bunch)

    beam.dt[bunch * n_macroparticles_pb: (bunch + 1) * n_macroparticles_pb] \
        = bunchBeam.dt + bunch * bunch_spacing * t_rf
    beam.dE[bunch * n_macroparticles_pb: (
                                                     bunch + 1) * n_macroparticles_pb] = bunchBeam.dE

### uniform profile

profile_margin = 0 * t_rf

t_batch_begin = 0 * t_rf
t_batch_end = (bunch_spacing * (n_bunches - 1) + 1) * t_rf

n_slices_rf = 64  # number of slices per RF-bucket

cut_left = t_batch_begin - profile_margin
cut_right = t_batch_end + profile_margin

# number of rf-buckets of the beam
# + rf-buckets before the beam + rf-buckets after the beam
n_slices = n_slices_rf * (bunch_spacing * (n_bunches - 1) + 1 \
                          + int(np.round((t_batch_begin - cut_left) / t_rf)) \
                          + int(np.round((cut_right - t_batch_end) / t_rf)))

uniform_profile = Profile(beam, CutOptions=CutOptions(cut_left=cut_left,
                                                      cut_right=cut_right,
                                                      n_slices=n_slices))
uniform_profile.track()

### Induced voltage calculated by the 'frequency' method of InducedVoltageFreq for a uniform profile
frequency_step = 10e6  # using default frequency resolution is insufficient
uniform_frequency_object = InducedVoltageFreq(beam, uniform_profile,
                                              impedance_model, frequency_step)

induced_voltage = TotalInducedVoltage(beam, uniform_profile,
                                      [uniform_frequency_object])
induced_voltage.induced_voltage_sum()

### Non-uniform profile with `SparseSlices`; only sample the buckets with beam

filling_pattern = np.zeros(bunch_spacing * (n_bunches - 1) + 1)
filling_pattern[::bunch_spacing] = 1

nonuniform_profile = SparseSlices(rf_station, beam, n_slices_rf,
                                  filling_pattern,
                                  tracker='onebyone', direct_slicing=True)

time = nonuniform_profile.bin_centers_array.flatten()
profile = nonuniform_profile.n_macroparticles_array.flatten()

### compute analytical induced voltage
Vind_anal = np.zeros_like(time)
for bunch in range(n_bunches):
    for it in range(len(R_s)):
        Vind_anal += analytical_gaussian_resonator(sigma, Q[it], R_s[it],
                                                   2 * np.pi * f_r[it],
                                                   time - (
                                                               bunch * bunch_spacing + 0.5) * t_rf,
                                                   intensity_pb)

### induced voltage for non-uniform sampling

# direct implementation; this was a first trail
# adapt frequency based on real part of the impedance;
freq, tmp = sample_function(
    lambda f: _compute_impedance(f, impedance_model).real,
    np.linspace(0, 3e9, 1000), tol=0.01)
Z = _compute_impedance(freq, impedance_model)
Lambda = FourierTransform(2 * np.pi * freq, time,
                          profile / np.trapz(profile, time))
Y = Z * Lambda
Vind_nonuni = -2 * intensity_pb * e * FourierTransform(-time, 2 * np.pi * freq,
                                                       Y).real / np.pi

# use the InducedVoltageSparse object; it samples the frequencies to adapt for 'Z * beam_spectrum'
Vind_sparse = InducedVoltageSparse(beam, nonuniform_profile,
                                   np.linspace(0, 6e9, 120),
                                   impedance_model,
                                   adaptive_frequency_sampling=True)
Vind_sparse.induced_voltage_1turn()

# use the InducedVoltageSparse object for the uniform profile (including empty buckets) ...
# and no adaptrive frequency sampling
Vind_uniform = InducedVoltageSparse(beam, uniform_profile,
                                    uniform_frequency_object.freq,
                                    impedance_model)

Vind_uniform.induced_voltage_1turn()

print(
    f"Data points for InducedVoltageFreq object:\t\t {len(uniform_frequency_object.total_impedance)}")
print(f"Data points for nonuniform profile:\t {len(Z)}")
print(
    f"Data points for InducedVoltageSparse object:\t {len(Vind_sparse.impedance)}")

plt.figure('profile', clear=True)
plt.grid()
plt.xlabel('time / ns')
plt.ylabel('macro-particles')
tmp, = plt.plot(uniform_profile.bin_centers * 1e9,
                uniform_profile.n_macroparticles, '.',
                label='Profile, uniform')
for bunch in range(n_bunches):
    indexes = (time > nonuniform_profile.cut_left_array[bunch]) * (
                time < nonuniform_profile.cut_right_array[bunch])
    tmp2, = plt.plot(time[indexes] * 1e9, profile[indexes], 'C1-',
                     label='SparseSlices')
plt.legend(handles=[tmp, tmp2])
plt.tight_layout()

plt.figure('impedance', clear=True)
plt.grid()
plt.plot(uniform_frequency_object.freq / 1e6,
         uniform_frequency_object.total_impedance.real * uniform_profile.bin_size / 1e6,
         '.')
plt.plot(freq / 1e6, Z.real / 1e6)

# test how the 'integrand' is sampled
plt.figure('integrand', clear=True)
plt.grid()
plt.plot(uniform_frequency_object.freq / 1e6,
         (
                     uniform_frequency_object.total_impedance * uniform_profile.beam_spectrum).real
         * uniform_profile.bin_size / n_macroparticles)
plt.plot(freq / 1e6, Y.real)
plt.plot(Vind_sparse.frequency_array / 1e6,
         (Vind_sparse.impedance * FourierTransform(
             2 * np.pi * Vind_sparse.frequency_array,
             time, profile / np.trapz(profile, time))).real,
         '--')

plt.figure('voltage', clear=True)
plt.grid()
plt.xlabel('time / ns')
plt.ylabel('induced voltage / MV')
tmp, = plt.plot(induced_voltage.time_array * 1e9,
                induced_voltage.induced_voltage / 1e6, '.',
                label='InducedVoltageFreq')
for bunch in range(n_bunches):
    indexes = (time > nonuniform_profile.cut_left_array[bunch]) * (
                time < nonuniform_profile.cut_right_array[bunch])

    tmp2, = plt.plot(time[indexes] * 1e9, Vind_nonuni[indexes] / 1e6, 'C1-',
                     label='non-uniform')
    tmp3, = plt.plot(time[indexes] * 1e9, Vind_anal[indexes] / 1e6, 'C2--',
                     label='analytic')

    tmp4, = plt.plot(time[indexes] * 1e9,
                     Vind_sparse.induced_voltage[indexes] / 1e6, 'C3--',
                     label='InducedVoltageSparse')
tmp5, = plt.plot(uniform_profile.bin_centers * 1e9,
                 Vind_uniform.induced_voltage / 1e6, 'C4--',
                 label='InducedVoltageSparse for uniform profile')
plt.legend(handles=[tmp, tmp2, tmp3, tmp4, tmp5])
plt.tight_layout()

plt.show()
# some iPython magic commands to evaluate computation speed
# %timeit -r 100 -n 7 uniform_frequency_object.induced_voltage_1turn()
# %timeit -r 100 -n 7 Vind_uniform.induced_voltage_1turn()
# %timeit -r 100 -n 7 Vind_sparse.induced_voltage_1turn()