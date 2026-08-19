# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Transfer function analysis for open and closed loop.

Notes
-----
Authors:
Jelena Banjac
Helga Timko
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.fft as npfft
import scipy.signal as scs
from matplotlib.mlab import csd, psd

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def estimate_transfer_function(
    input_signal: NumpyArray,
    output_signal: NumpyArray,
    t_s: float,
    data_cut: int = 0,
):
    """
    Analyse the input and output signals to get the transfer function.

    Parameters
    ----------
    input_signal
        Input signal to the DUT.
    output_signal
        Output signal from the DUT.
    t_s
        Sampling time of the input and output signal.
    data_cut
        Optional argument to cut data at the beginning of the signals.

    Returns
    -------
    frequency
        Frequency array of the transfer function.
    h_est
        Estimated transfer function.
    """
    # Calculate transfer function
    f_s = 1 / t_s
    n_fft = int(np.floor(len(input_signal) / 4))

    input_signal = input_signal[data_cut:]
    output_signal = output_signal[data_cut:]

    f_est, h_est = tf_estimate(
        input_signal,
        output_signal,
        window=np.hamming(n_fft),
        noverlap=0,
        Fs=f_s,
        NFFT=n_fft,
    )

    # reorder results to be form -freq to +freq
    f_est = [x + f_s / 2 if x < 0 else x - f_s / 2 for x in f_est]

    # Spectrum of the input signal
    f_max = input_signal_spectrum(input_signal, n_fft, f_s)
    f_baseband = npfft.fftshift(f_est)

    low = np.where((f_baseband < 0) & (f_baseband > -f_max))[0]
    high = np.where((f_baseband > 0) & (f_baseband < f_max))[0]

    # Reconstruct the transfer function
    frequency = np.concatenate([f_baseband[low], f_baseband[high]])
    h_est = np.concatenate([h_est[low], h_est[high]])

    return frequency, h_est


def input_signal_spectrum(input_signal: NumpyArray, n: int, f_s: float):
    """
    Prepare for drawing the input signal spectrum.

    Parameters
    ----------
    input_signal
        Input signal of the system.
    n
        Number of points in the spectrum.
    f_s
        Sampling frequency [1/s].

    Returns
    -------
    f_max
        Maximum frequency.
    """
    N_sp = 16

    # power spectral density or power spectrum of input signal `input`.
    f_m, P_ss = scs.welch(
        input_signal,
        window="hamming",
        noverlap=0,
        nperseg=int(np.floor(n / N_sp)),
        fs=f_s,
    )

    # reorder results to be form -freq to +freq
    f_m = [f - f_s if f >= f_s / 2 else f for f in f_m]

    # shift zero-frequency component to the center of the spectrum
    f_m, P_ss = npfft.fftshift(f_m).T, npfft.fftshift(P_ss)

    # minimum signal and maximum signal
    p_min, p_max = 35, 20 * np.log10(np.max(np.abs(P_ss)))

    # signal bandwidth
    p_bw = p_max - p_min

    # interval in which the P_ss signal is greater that min P_ss
    interval = np.where(abs(P_ss) >= 10 ** (p_bw / 20))[0]

    f_m_bw = f_m[interval]

    f_max = min(-min(f_m_bw), max(f_m_bw))

    return f_max


def tf_estimate(x: NumpyArray, y: NumpyArray, *args, **kwargs):
    """
    Estimate the transfer function using csd and psd functions from mlab.

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
    x
        Arrays or sequences containing the data.
    y
        Arrays or sequences containing the data.
    *args
        Default keyword values.
    **kwargs
        NFFT, Fs, detrend, window, noverlap, pad_to, sides, scale_by_freq.

    Returns
    -------
    frequency
        Frequency array.
    transfer_function
        Transfer function estimate.
    """
    p_xy, frequencies_csd = csd(y, x, *args, **kwargs)
    p_xx, frequencies_psd = psd(x, *args, **kwargs)

    return frequencies_csd, (p_xy / p_xx).conjugate()
