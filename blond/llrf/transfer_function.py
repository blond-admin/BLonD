# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Transfer function analysis for open and closed loop**

:Authors: **Jelena Banjac, Helga Timko**
'''

from __future__ import division
import numpy as np
import numpy.fft as npfft
from matplotlib.mlab import psd, csd
import matplotlib.pyplot as plt
import scipy.signal as scs

# Set up logging
import logging
logger = logging.getLogger(__name__)

class TransferFunction(object):
    r'''Reconstructing the transfer function of a DUT based on input and output
    signals.

    Parameters
    ----------
    signal_in : float array
        Complex signal going into the DUT
    signal_out : float array
        Complex signal coming out from the DUT
    T_s : float
        sampling time
    plot : bool
        Enable/disable plotting; default is False
    '''

    def __init__(self, signal_in, signal_out, T_s, plot=False):

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("TransferFunction class initialized")

        # Read in variables
        self.signal_in = signal_in
        self.signal_out = signal_out
        self.T_s = T_s
        self.plot =plot


    def analyse(self, data_cut):

        # input signal
        input_signal = self.signal_in[data_cut:]
        self.logger.debug(f"OUTPUT signal: {input_signal.shape}")

        # input signal
        output_signal = self.signal_out[data_cut:]
        self.logger.debug(f"INPUT signal: {output_signal.shape}")

        # Estimate transfer function
        self.f_est, self.H_est = self.estimate_transfer_function(input_signal,
            output_signal, self.T_s, self.plot, self.logger)


    @staticmethod
    def estimate_transfer_function(input, output, T_s, plot, logger):
        # Calculate transfer function
        f_s = 1/T_s
        n_fft = int(np.floor(len(input)/4))
        f_est, H_est = tf_estimate(input, output, window=np.hamming(n_fft),
                                   noverlap=0, Fs=f_s, NFFT=n_fft)
        if plot:
            TransferFunction.plot_magnitude_and_phase(f_est, H_est)

        # reorder results to be form -freq to +freq
        f_est = [x + f_s/2 if x < 0 else x - f_s/2 for x in f_est]

        # Spectrum of the input signal
        f_max = TransferFunction.input_signal_spectrum(input, n_fft, f_s, plot)
        f_baseband = npfft.fftshift(f_est)

        low = np.where((0 > f_baseband) & (f_baseband > -f_max))[0]
        high = np.where((0 < f_baseband) & (f_baseband < f_max))[0]

        # Reconstruct the transfer function
        f = np.concatenate([f_baseband[low], f_baseband[high]])
        H = np.concatenate([H_est[low], H_est[high]])

        logger.debug("Transfer function reconstructed")

        if plot:
            TransferFunction.plot_magnitude_and_phase(f, H)

        return f, H


    @staticmethod
    def input_signal_spectrum(input, n, f_s, plot):
        """
        Prepare for drawing the input signal spectrum.

        Parameters
        ----------
        input : complex array
            Input signal of the system
        n : integer
            Number of points in the spectrum
        f_s : float
            Sampling frequency [1/s]

        Returns
        -------
        f_max : double
            Maximum frequency

        Attributes
        ----------
        N_sp: float
            Number of sections that the N-data spectrum is divided to
            calculate periodogram [1..N]
        f_m : ndarray
            Array of sample frequencies.
        P_ss: ndarray
            Power spectral density or power spectrum of input signal `In`.
            Afterwards, it shifted the
            zero-frequency component to the center of the spectrum.
        P_min : float
            -Pmin dB defines the minimum signal [35 dB].
        P_max : float
            Defines the maximum signal, depending on a `P_ss`
        P_bw : float
            Signal bandwidth.
        interval : int array
            Interval in which the P_ss signal is greater that min P_ss
        f_m_bw :
            Array of sample frequencies for a bandwidth of power spectrum.
        P_ss_bw :
            Bandwidth of power spectral density or power spectrum of input
            signal `In`.
        """
        N_sp = 16

        # power spectral density or power spectrum of input signal `input`.
        f_m, P_ss = scs.welch(input, window='hamming', noverlap=0,
                              nperseg=int(np.floor(n / N_sp)), fs=f_s)

        # reorder results to be form -freq to +freq
        f_m = [f - f_s if f >= f_s / 2 else f for f in f_m]

        # shift zero-frequency component to the center of the spectrum
        f_m, P_ss = npfft.fftshift(f_m).T, npfft.fftshift(P_ss)

        # minimum signal and maximum signal
        P_min, P_max = 35, 20 * np.log10(np.max(np.abs(P_ss)))

        # signal bandwidth
        P_bw = P_max - P_min

        # interval in which the P_ss signal is greater that min P_ss
        interval = np.where(abs(P_ss) >= 10 ** (P_bw / 20))[0]

        f_m_bw, P_ss_bw = f_m[interval], P_ss[interval]

        #if plot:
        #    TransferFunction.plot_input_signal_spectrum(f_m, P_ss, f_m_bw,
        #                                                P_ss_bw, P_bw, interval)

        f_max = min(-min(f_m_bw), max(f_m_bw))

        return f_max

    @staticmethod
    def plot_magnitude_and_phase(freq, H):

        magnitude = 20 * np.log10(np.abs(H))
        phases = np.unwrap(np.angle(H))

        plt.rc('axes', labelsize=12, labelweight='normal', axisbelow=True)
        plt.rc('legend', fontsize=11, fancybox=True, framealpha=0.5)
        plt.rc('figure', figsize=(8, 6), titlesize=14)
        plt.rc('grid', c='lightgray')

        fig = plt.figure()
        gs = plt.GridSpec(2, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Transfer function')
        ax1.plot(freq / 10**6, magnitude, 'r', linewidth=0.3)
        ax1.set_xlabel('Frequency [MHz]')
        ax1.set_ylabel('Gain [dB]')

        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.plot(freq / 10**6, (180/np.pi)*phases, 'r', linewidth=0.3)
        ax2.set_xlabel('Frequency [MHz]')
        ax2.set_ylabel('Phase [degrees]')
        plt.show()


def tf_estimate(x, y, *args, **kwargs):
    """
    Estimate transfer function from x to y, see csd (from
    matplotlib.mlab package) for calling convention.
    Link: https://stackoverflow.com/questions/28462144/python-version-of
    -matlab-signal-toolboxs-tfestimate

    The vectors *x* and *y* are divided into *NFFT* length segments.
    Each segment is detrended by function *detrend* and windowed by
    function *window*.
    *noverlap* gives the length of the overlap between segments.

    Parameters
    ----------
    x : 1-D arrays or sequences
        Arrays or sequences containing the data
    y : 1-D arrays or sequences
        Arrays or sequences containing the data
    args : Default keyword values: None
    kwargs : NFFT, Fs, detrend, window, noverlap, pad_to, sides, scale_by_freq

    Returns
    -------
    frequency : float array
        Frequency array
    H : complex array
        Transfer function estimate

    Attributes
    ----------
    p_xy: float array
        Cross spectral density
    p_xx: float array
        Power spectral density
    frequencies_csd: array
        Frequency array for cross spectral density
    frequencies_psd: array
        Frequency array for power spectral density
    """
    p_xy, frequencies_csd = csd(y, x, *args, **kwargs)
    p_xx, frequencies_psd = psd(x, *args, **kwargs)

    return frequencies_csd, (p_xy/p_xx).conjugate()


