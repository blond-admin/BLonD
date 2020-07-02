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


def rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers):

    rf_voltage = np.zeros(len(bin_centers))

    __lib.rf_volt_comp(__getPointer(voltages),
                       __getPointer(omega_rf),
                       __getPointer(phi_rf),
                       __getPointer(bin_centers),
                       __getLen(voltages),
                       __getLen(rf_voltage),
                       __getPointer(rf_voltage))
    return rf_voltage


def kick(dt, dE, voltage, omega_rf, phi_rf, charge, n_rf, acceleration_kick):
    voltage_kick = np.ascontiguousarray(charge*voltage)
    omegarf_kick = np.ascontiguousarray(omega_rf)
    phirf_kick = np.ascontiguousarray(phi_rf)

    __lib.kick(__getPointer(dt),
               __getPointer(dE),
               ct.c_int(n_rf),
               __getPointer(voltage_kick),
               __getPointer(omegarf_kick),
               __getPointer(phirf_kick),
               __getLen(dt),
               ct.c_double(acceleration_kick))


def drift(dt, dE, solver, t_rev, length_ratio, alpha_order, eta_0,
          eta_1, eta_2, alpha_0, alpha_1, alpha_2, beta, energy):

    __lib.drift(__getPointer(dt),
                __getPointer(dE),
                ct.c_char_p(solver),
                ct.c_double(t_rev),
                ct.c_double(length_ratio),
                ct.c_double(alpha_order),
                ct.c_double(eta_0),
                ct.c_double(eta_1),
                ct.c_double(eta_2),
                ct.c_double(alpha_0),
                ct.c_double(alpha_1),
                ct.c_double(alpha_2),
                ct.c_double(beta),
                ct.c_double(energy),
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


def music_track(dt, dE, induced_voltage, array_parameters, alpha, omega_bar,
                const, coeff1, coeff2, coeff3, coeff4):
    __lib.music_track(__getPointer(dt),
                      __getPointer(dE),
                      __getPointer(induced_voltage),
                      __getPointer(array_parameters),
                      __getLen(dt),
                      ct.c_double(alpha),
                      ct.c_double(omega_bar),
                      ct.c_double(const),
                      ct.c_double(coeff1),
                      ct.c_double(coeff2),
                      ct.c_double(coeff3),
                      ct.c_double(coeff4))


def music_track_multiturn(dt, dE, induced_voltage, array_parameters, alpha,
                          omega_bar, const, coeff1, coeff2, coeff3, coeff4):
    __lib.music_track_multiturn(__getPointer(dt),
                                __getPointer(dE),
                                __getPointer(induced_voltage),
                                __getPointer(array_parameters),
                                __getLen(dt),
                                ct.c_double(alpha),
                                ct.c_double(omega_bar),
                                ct.c_double(const),
                                ct.c_double(coeff1),
                                ct.c_double(coeff2),
                                ct.c_double(coeff3),
                                ct.c_double(coeff4))


def synchrotron_radiation(dE, U0, n_kicks, tau_z):
    __lib.synchrotron_radiation(
        __getPointer(dE),
        ct.c_double(U0 / n_kicks),
        __getLen(dE),
        ct.c_double(tau_z * n_kicks),
        ct.c_int(n_kicks))


def synchrotron_radiation_full(dE, U0, n_kicks, tau_z, sigma_dE, energy,
                               random_array):
    __lib.synchrotron_radiation_full(
        __getPointer(dE),
        ct.c_double(U0 / n_kicks),
        __getLen(dE),
        ct.c_double(sigma_dE),
        ct.c_double(tau_z * n_kicks),
        ct.c_double(energy),
        __getPointer(random_array),
        ct.c_int(n_kicks))


def set_random_seed(seed):
    __lib.set_random_seed(ct.c_int(seed))


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
