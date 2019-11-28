'''
BLonD physics wrapper functions

@author Konstantinos Iliakis
@date 20.10.2017
'''

import ctypes as ct
import numpy as np
# from setup_cpp import libblondphysics as __lib
from .. import libblond as __lib


def __getPointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def __getLen(x):
    return ct.c_int(len(x))


def beam_phase(bin_centers, profile, alpha, omegarf, phirf, bin_size):
    __lib.beam_phase.restype = ct.c_double
    coeff = __lib.beam_phase(__getPointer(bin_centers),
                             __getPointer(profile),
                             ct.c_double(alpha),
                             ct.c_double(omegarf),
                             ct.c_double(phirf),
                             ct.c_double(bin_size),
                             __getLen(profile))
    return coeff


def rf_volt_comp(voltages, omega_rf, phi_rf, ring):
    # voltages = np.ascontiguousarray(ring.voltage[:, ring.counter[0]])
    # omega_rf = np.ascontiguousarray(ring.omega_rf[:, ring.counter[0]])
    # phi_rf = np.ascontiguousarray(ring.phi_rf[:, ring.counter[0]])

    rf_voltage = np.zeros(len(ring.profile.bin_centers))

    __lib.rf_volt_comp(__getPointer(voltages),
                       __getPointer(omega_rf),
                       __getPointer(phi_rf),
                       __getPointer(ring.profile.bin_centers),
                       __getLen(voltages),
                       __getLen(rf_voltage),
                       __getPointer(rf_voltage))
    return rf_voltage


def kick(ring, dt, dE, turn):
    voltage_kick = np.ascontiguousarray(ring.charge*ring.voltage[:, turn])
    omegarf_kick = np.ascontiguousarray(ring.omega_rf[:, turn])
    phirf_kick = np.ascontiguousarray(ring.phi_rf[:, turn])

    __lib.kick(__getPointer(dt),
               __getPointer(dE),
               ct.c_int(ring.n_rf),
               __getPointer(voltage_kick),
               __getPointer(omegarf_kick),
               __getPointer(phirf_kick),
               __getLen(dt),
               ct.c_double(ring.acceleration_kick[turn]))


def drift(ring, dt, dE, turn):

    __lib.drift(__getPointer(dt),
                __getPointer(dE),
                ct.c_char_p(ring.solver),
                ct.c_double(ring.t_rev[turn]),
                ct.c_double(ring.length_ratio),
                ct.c_double(ring.alpha_order),
                ct.c_double(ring.eta_0[turn]),
                ct.c_double(ring.eta_1[turn]),
                ct.c_double(ring.eta_2[turn]),
                ct.c_double(ring.alpha_0[turn]),
                ct.c_double(ring.alpha_1[turn]),
                ct.c_double(ring.alpha_2[turn]),
                ct.c_double(ring.rf_params.beta[turn]),
                ct.c_double(ring.rf_params.energy[turn]),
                __getLen(dt))


def linear_interp_kick(dt, dE, voltage,
                       bin_centers, charge,
                       acceleration_kick):
    __lib.linear_interp_kick(__getPointer(dt),
                             __getPointer(dE),
                             __getPointer(voltage),
                             __getPointer(bin_centers),
                             ct.c_double(charge),
                             __getLen(bin_centers),
                             __getLen(dt),
                             ct.c_double(acceleration_kick))


def linear_interp_kick_n_drift(dt, dE, total_voltage, bin_centers, charge,
                               acc_kick, solver, t_rev, length_ratio,
                               alpha_order, eta_0, eta_1, eta_2, beta, energy):
    __lib.linear_interp_kick_n_drift(__getPointer(dt),
                                     __getPointer(dE),
                                     __getPointer(total_voltage),
                                     __getPointer(bin_centers),
                                     __getLen(bin_centers),
                                     __getLen(dt),
                                     ct.c_double(acc_kick),
                                     ct.c_char_p(solver),
                                     ct.c_double(t_rev),
                                     ct.c_double(length_ratio),
                                     ct.c_double(alpha_order),
                                     ct.c_double(eta_0),
                                     ct.c_double(eta_1),
                                     ct.c_double(eta_2),
                                     ct.c_double(beta),
                                     ct.c_double(energy),
                                     ct.c_double(charge))


def linear_interp_time_translation(ring, dt, dE, turn):
    pass


def slice(beam_dt, profile, cut_left, cut_right):
    __lib.histogram(__getPointer(beam_dt),
                    __getPointer(profile),
                    ct.c_double(cut_left),
                    ct.c_double(cut_right),
                    __getLen(profile),
                    __getLen(beam_dt))


def slice_smooth(beam_dt, profile, cut_left, cut_right):
    __lib.smooth_histogram(__getPointer(beam_dt),
                           __getPointer(profile),
                           ct.c_double(cut_left),
                           ct.c_double(cut_right),
                           __getLen(profile),
                           __getLen(beam_dt))


def music_track(music):
    __lib.music_track(__getPointer(music.beam.dt),
                      __getPointer(music.beam.dE),
                      __getPointer(music.induced_voltage),
                      __getPointer(music.array_parameters),
                      __getLen(music.beam.dt),
                      ct.c_double(music.alpha),
                      ct.c_double(music.omega_bar),
                      ct.c_double(music.const),
                      ct.c_double(music.coeff1),
                      ct.c_double(music.coeff2),
                      ct.c_double(music.coeff3),
                      ct.c_double(music.coeff4))


def music_track_multiturn(music):
    __lib.music_track_multiturn(__getPointer(music.beam.dt),
                                __getPointer(music.beam.dE),
                                __getPointer(music.induced_voltage),
                                __getPointer(music.array_parameters),
                                __getLen(music.beam.dt),
                                ct.c_double(music.alpha),
                                ct.c_double(music.omega_bar),
                                ct.c_double(music.const),
                                ct.c_double(music.coeff1),
                                ct.c_double(music.coeff2),
                                ct.c_double(music.coeff3),
                                ct.c_double(music.coeff4))


def synchrotron_radiation(SyncRad, turn):
    __lib.synchrotron_radiation(
        __getPointer(SyncRad.beam.dE),
        ct.c_double(SyncRad.U0 / SyncRad.n_kicks),
        ct.c_int(SyncRad.beam.n_macroparticles),
        ct.c_double(SyncRad.tau_z * SyncRad.n_kicks),
        ct.c_int(SyncRad.n_kicks))


def synchrotron_radiation_full(SyncRad, turn):
    __lib.synchrotron_radiation_full(
        __getPointer(SyncRad.beam.dE),
        ct.c_double(SyncRad.U0 / SyncRad.n_kicks),
        ct.c_int(SyncRad.beam.n_macroparticles),
        ct.c_double(SyncRad.sigma_dE),
        ct.c_double(SyncRad.tau_z * SyncRad.n_kicks),
        ct.c_double(SyncRad.general_params.energy[0, turn]),
        __getPointer(SyncRad.random_array),
        ct.c_int(SyncRad.n_kicks))


def fast_resonator(R_S, Q, frequency_array, frequency_R, impedance=None):
    realImp = np.zeros(len(frequency_array), dtype=np.float64)
    imagImp = np.zeros(len(frequency_array), dtype=np.float64)

    __lib.fast_resonator_real_imag(
        __getPointer(realImp),
        __getPointer(imagImp),
        __getPointer(frequency_array),
        __getPointer(R_S),
        __getPointer(Q),
        __getPointer(frequency_R),
        __getLen(R_S),
        __getLen(frequency_array))

    impedance = realImp + 1j * imagImp
    return impedance
