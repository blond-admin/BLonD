# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Filters and methods for control loops**

:Authors: **Helga Timko**
'''

from __future__ import division
import numpy as np
from scipy.constants import e
from scipy import signal as sgn
import matplotlib.pyplot as plt

# Set up logging
import logging
logger = logging.getLogger(__name__)

from blond.llrf.impulse_response import TravellingWaveCavity


def polar_to_cartesian(amplitude, phase):
    """Convert data from polar to cartesian (I,Q) coordinates.

    Parameters
    ----------
    amplitude : float array
        Amplitude of signal
    phase : float array
        Phase of signal

    Returns
    -------
    complex array
        Signal with in-phase and quadrature (I,Q) components
    """

    logger.debug("Converting from polar to Cartesian")

    return amplitude*(np.cos(phase) + 1j*np.sin(phase))


def cartesian_to_polar(IQ_vector):
    """Convert data from Cartesian (I,Q) to polar coordinates.

    Parameters
    ----------
    IQ_vector : complex array
        Signal with in-phase and quadrature (I,Q) components

    Returns
    -------
    float array
        Amplitude of signal
    float array
        Phase of signal

    """

    logger.debug("Converting from Cartesian to polar")

    return np.absolute(IQ_vector), np.angle(IQ_vector)


def modulator(signal, omega_i, omega_f, T_sampling):
    """Demodulate a signal from initial frequency to final frequency. The two
    frequencies should be close.

    Parameters
    ----------
    signal : float array
        Signal to be demodulated
    omega_i : float
        Initial revolution frequency [1/s] of signal (before demodulation)
    omega_f : float
        Final revolution frequency [1/s] of signal (after demodulation)
    T_sampling : float
        Sampling period (temporal bin size) [s] of the signal

    Returns
    -------
    float array
        Demodulated signal at f_final

    """

    if len(signal) < 2:
        #TypeError
        raise RuntimeError("ERROR in filters.py/demodulator: signal should" +
                           " be an array!")
    delta_phi = (omega_i - omega_f)*T_sampling * np.arange(len(signal))
    # Pre compute sine and cosine for speed up
    cs = np.cos(delta_phi)
    sn = np.sin(delta_phi)
    I_new = cs*signal.real - sn*signal.imag
    Q_new = sn*signal.real + cs*signal.imag

    return I_new + 1j*Q_new


def rf_beam_current(Profile, omega_c, T_rev, lpf=True, downsample=None):
    r"""Function calculating the beam charge at the (RF) frequency, slice by
    slice. The charge distribution [C] of the beam is determined from the beam
    profile :math:`\lambda_i`, the particle charge :math:`q_p` and the real vs.
    macro-particle ratio :math:`N_{\mathsf{real}}/N_{\mathsf{macro}}`

    .. math::
        Q_i = \frac{N_{\mathsf{real}}}{N_{\mathsf{macro}}} q_p \lambda_i

    The total charge [C] in the beam is then

    .. math::
        Q_{\mathsf{tot}} = \sum_i{Q_i}

    The DC beam current [A] is the total number of charges per turn :math:`T_0`

    .. math:: I_{\mathsf{DC}} = \frac{Q_{\mathsf{tot}}}{T_0}

    The RF beam charge distribution [C] at a revolution frequency
    :math:`\omega_c` is the complex quantity

    .. math::
        \left( \begin{matrix} I_{rf,i} \\
        Q_{rf,i} \end{matrix} \right)
        = 2 Q_i \left( \begin{matrix} \cos(\omega_c t_i) \\
        \sin(\omega_c t_i)\end{matrix} \right) \, ,

    where :math:`t_i` are the time coordinates of the beam profile. After de-
    modulation, a low-pass filter at 20 MHz is applied.

    Parameters
    ----------
    Profile : class
        A Profile type class
    omega_c : float
        Revolution frequency [1/s] at which the current should be calculated
    T_rev : float
        Revolution period [s] of the machine
    lpf : bool
        Apply low-pass filter; default is True
    downsample : dict
        Dictionary containing float value for 'Ts' sampling time and int value
        for 'points'. Will downsample the RF beam charge onto a coarse time
        grid with 'Ts' sampling time and 'points' points.

    Returns
    -------
    complex array
        RF beam charge array [C] at 'frequency' omega_c, with the sampling time
        of the Profile object. To obtain current, divide by the sampling time
    (complex array)
        If time_coarse is specified, returns also the RF beam charge array [C]
        on the coarse time grid

    """

    # Convert from dimensionless to Coulomb/Ampères
    # Take into account macro-particle charge with real-to-macro-particle ratio
    charges = Profile.Beam.ratio*Profile.Beam.Particle.charge*e\
        * np.copy(Profile.n_macroparticles)
    logger.debug("Sum of particles: %d, total charge: %.4e C",
                 np.sum(Profile.n_macroparticles), np.sum(charges))
    logger.debug("DC current is %.4e A", np.sum(charges)/T_rev)

    # Mix with frequency of interest; remember factor 2 demodulation
    I_f = 2.*charges*np.cos(omega_c*Profile.bin_centers)
    Q_f = 2.*charges*np.sin(omega_c*Profile.bin_centers)

    # Pass through a low-pass filter
    if lpf is True:
        # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
        cutoff = 20.e6*2.*Profile.bin_size
        I_f = low_pass_filter(I_f, cutoff_frequency=cutoff)
        Q_f = low_pass_filter(Q_f, cutoff_frequency=cutoff)
    logger.debug("RF total current is %.4e A", np.fabs(np.sum(I_f))/T_rev)

    charges_fine = I_f + 1j*Q_f
    if downsample:
        try:
            T_s = float(downsample['Ts'])
            n_points = int(downsample['points'])
        except:
            raise RuntimeError('Downsampling input erroneous in rf_beam_current')

        # Find which index in fine grid matches index in coarse grid
        ind_fine = np.floor((Profile.bin_centers - 0.5*Profile.bin_size)/T_s)
        ind_fine = np.array(ind_fine, dtype=int)
        indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]

        # Pick total current within one coarse grid
        charges_coarse = np.zeros(n_points, dtype=np.complex) #+ 1j*np.zeros(n_points)
        charges_coarse[0] = np.sum(charges_fine[np.arange(indices[0])])
        for i in range(1, len(indices)):
            charges_coarse[i] = np.sum(charges_fine[np.arange(indices[i-1],
                                                              indices[i])])
        return charges_fine, charges_coarse

    else:
        return charges_fine


def comb_filter(y, x, a):
    """Feedback comb filter.
    """

    return a*y + (1 - a)*x


def low_pass_filter(signal, cutoff_frequency=0.5):
    """Low-pass filter based on Butterworth 5th order digital filter from
    scipy,
    http://docs.scipy.org

    Parameters
    ----------
    signal : float array
        Signal to be filtered
    cutoff_frequency : float
        Cutoff frequency [1] corresponding to a 3 dB gain drop, relative to the
        Nyquist frequency of 1; default is 0.5

    Returns
    -------
    float array
        Low-pass filtered signal

    """

    b, a = sgn.butter(5, cutoff_frequency, 'low', analog=False)

    return sgn.filtfilt(b, a, signal)


def moving_average(x, N, x_prev=None):
    """Function to calculate the moving average (or running mean) of the input
    data.

    Parameters
    ----------
    x : float array
        Data to be smoothed
    N : int
        Window size in points
    x_prev : float array
        Data to pad with in front

    Returns
    -------
    float array
        Smoothed data array of size
            * len(x) - N + 1, if x_prev = None
            * len(x) + len(x_prev) - N + 1, if x_prev given

    """

    if x_prev is not None:
        # Pad in front with x_prev signal
        x = np.concatenate((x_prev, x))

    # based on https://stackoverflow.com/a/14314054
    mov_avg = np.cumsum(x)
    mov_avg[N:] = mov_avg[N:] - mov_avg[:-N]
    return mov_avg[N-1:] / N


def feedforward_filter(TWC: TravellingWaveCavity, T_s, debug=False):

    # TEMP!!
    TWC.tau=420e-9
    print(TWC.tau)
    print(T_s)

    # Filling time in samples
    n_filling = int(TWC.tau/T_s)
    logger.debug("Filling time in samples: %d", n_filling)
    # Number of FIR filter taps
    n_taps = 31 #2*int(0.5*n_filling) + 13 #31
    n_taps_2 = int(0.5*(n_taps+1))
    if n_taps % 2 == 0:
        raise RuntimeError("Number of taps in feedforward filter must be odd!")
    logger.debug("Number of taps: %d", n_taps)
    # Fitting samples
    n_fit = int(n_taps + n_filling)
    logger.debug("Fitting samples: %d", n_fit)

    # Even-symmetric feed-forward filter matrix
    even = np.matrix(np.zeros(shape=(n_taps,n_taps_2)), dtype=np.float64)
    for i in range(n_taps):
        even[i,abs(n_taps_2-i-1)] = 1

    # Odd-symmetric feed-forward filter matrix
    odd = np.matrix(np.zeros(shape=(n_taps, n_taps_2-1)), dtype=np.float64)
    for i in range(n_taps_2-1):
        odd[i,abs(n_taps_2-i-2)] = -1
        odd[n_taps-i-1, abs(n_taps_2 - i - 2)] = 1

    # Generator-cavity response matrix: non-zero during filling time
    resp = np.matrix(np.zeros(shape=(n_fit, n_fit+n_filling-1)), dtype=np.float64)
    for i in range(n_fit):
        resp[i,i:i+n_filling] = 1

    # Convolution with beam step current
    conv = np.matrix(np.zeros(shape=(n_fit+n_filling-1, n_taps)), dtype=np.float64)
    for i in range(n_taps):
        conv[i+n_filling, 0:i] = 1
    conv[n_taps+n_filling:, :] = 1

    if debug:
        np.set_printoptions(threshold=10000, linewidth=100)
        print("Even matrix shape", even.shape)
        print(even)
        print("Odd matrix shape", odd.shape)
        print(odd)
        print("Response matrix shape", resp.shape)
        print(resp)
        print("Convolution matrix shape", conv.shape)
        print(conv)
        print("\n\n")


    # Impulse response from cavity towards beam
#    TWC.tau = (n_filling + 0.5)*T_s
    time_array = np.linspace(0, n_fit*T_s, num=n_fit) - TWC.tau/2
#    time_array = np.linspace(-n_filling/2*T_s, n_fit*T_s-n_filling/2*T_s, num=n_fit)
    TWC.impulse_response_beam(TWC.omega_r, time_array)
    h_beam_real = TWC.h_beam.real/TWC.R_beam*TWC.tau
#    h_beam_real = np.zeros(n_fit)
#    h_beam_real[0:14] = np.linspace(2,0,num=14)
#    h_beam_real[0] = 1
    print(h_beam_real)
    h_beam_even = np.zeros(n_fit)
    h_beam_odd = np.zeros(n_fit)
    if n_filling % 2 == 0:
        n_c = int((n_fit-1)*0.5)
        h_beam_even[n_c] = h_beam_real[0]
        h_beam_even[n_c + 1:] = 0.5*h_beam_real[1:n_c + 1]
        h_beam_even[:n_c] = 0.5*(h_beam_real[1:n_c + 1])[::-1]
        h_beam_odd[n_c] = 0
        h_beam_odd[n_c + 1:] = 0.5*h_beam_real[1:n_c + 1]
        h_beam_odd[:n_c] = 0.5*(-h_beam_real[1:n_c + 1])[::-1]
    else:
        n_c = int(n_fit*0.5)
        h_beam_even[n_c:] = 0.5*h_beam_real[1:n_c+1]
        h_beam_even[:n_c] = 0.5*(h_beam_real[1:n_c+1])[::-1]
        h_beam_odd[n_c:] = 0.5*h_beam_real[1:n_c+1]
        h_beam_odd[:n_c] = 0.5*(-h_beam_real[1:n_c+1])[::-1]
    print(n_c)

    I_beam_step = np.ones(n_fit)
#    I_beam_step = np.zeros(n_fit)
#    I_beam_step[int(n_filling/2):int(n_filling/2)+n_taps] = 1
#    I_beam_step[0:n_filling] = 1
#    I_beam_step[int(n_filling/2):] = 1
#    I_beam_step[1:] = 1
    I_beam_step[0] = 0 #0.5
    I_beam_step[1] = 0.5 #0.5

    V_beam_even = sgn.fftconvolve(I_beam_step, h_beam_even, mode='full')[:I_beam_step.shape[0]]
    V_beam_odd = sgn.fftconvolve(I_beam_step, h_beam_odd, mode='full')[:I_beam_step.shape[0]]
    # Normalised respons
    norm = np.max(V_beam_even)
    V_beam_even /= norm
    V_beam_odd /= norm
#    print(h_beam_even)

    if debug:
        plt.rc('lines', linewidth=0.5, markersize=3)
        plt.rc('axes', labelsize=12, labelweight='normal')

        plt.figure("Impulse response")
        plt.plot(time_array*1e6, h_beam_even, 'bo-', label='even')
        plt.plot(time_array*1e6, h_beam_odd, 'ro-', label='odd')
        plt.plot(time_array*1e6, h_beam_even+h_beam_odd, 'go-', label='total')
        plt.axhline(0, color='grey', alpha=0.5)
        plt.xlabel("Time [us]")
        plt.legend()

        plt.figure("Beam-induced voltage")
        plt.plot(V_beam_even, 'bo-', label='even')
        plt.plot(V_beam_odd, 'ro-', label='odd')
        plt.plot(V_beam_even+V_beam_odd, 'go-', label='total')
        plt.axhline(0, color='grey', alpha=0.5)
        plt.xlabel("Samples [1]")
        plt.legend()

#    temp_1 = even.transpose() @ conv.transpose() @ resp.transpose()
#    temp_2 = resp @ conv @ even
#    temp_4 = temp_1 @ temp_2
#    temp_3 = np.linalg.inv(temp_4/np.max(temp_4))/np.max(temp_4)

#    print(temp_1)
#    print(temp_2)
#    print(temp_4)
#    print(temp_3)
#    print("\n \n \n ")
#    print(temp_3 @ temp_4)
#    print(temp_3.shape)
#    print(temp_4.shape)
#    print((temp_3 @ temp_4).shape)

#    print("")
#    print(even.shape)
#    print(conv.shape)
#    print(resp.shape)
#    print("")
#    print(temp_1.shape)
#    print(temp_2.shape)
#    print(temp_3.shape)

#    h_ff_even = np.matmul(even, np.matmul(temp_3, np.matmul(temp_1, V_beam_even)))
#    h_ff_even = even @ temp_3 @ temp_1 @ V_beam_even

#    weight = np.identity(n_fit)
#    h_ff_even = even @ np.linalg.inv(even.transpose() @ conv.transpose() @
#                                     resp.transpose() @ weight @ resp @ conv @ even) @ \
#        even.transpose() @ conv.transpose() @ resp.transpose() @ weight @ V_beam_even


#    weight = np.matrix(np.zeros(shape=(n_fit, n_fit)))
#    for i in range(20):
#        weight[i,i] = 1

#    h_ff_even_old = even * (even.T * conv.T * resp.T * resp * conv * even).I * \
#        even.T * conv.T * resp.T * V_beam_even

    V_beam_even = np.matrix(V_beam_even).transpose()
#    V_beam_even[:7] = 0
    print(V_beam_even)
    print(V_beam_even.shape)
    print((resp*conv*even).shape)
    h_ff_even = even * (resp * conv * even).I * V_beam_even
#    h_ff_even = (resp * conv).I * V_beam_even

#    h_ff_even_2, residuals, rank, singular_vals = np.linalg.lstsq(resp*conv*even, V_beam_even, rcond=None)
#    h_ff_even_2 = even*h_ff_even_2
#    print(h_ff_even_2)
#    print(h_ff_even_2.shape)


#    print(h_ff_even.shape)
#    print((even.T * conv.T * resp.T * resp * conv * even))
#    print((even.T * conv.T * resp.T * resp * conv * even).I * (even.T * conv.T * resp.T * resp * conv * even))

    # TEMPORARY
#    h_ff_even_id = np.copy(h_ff_even)
#    h_ff_even_id[0:8] = 0
#    h_ff_even_id[8] = 0.0008
#    h_ff_even_id[9] = 0.0052
#    h_ff_even_id[10:-10] = 0.0058
#    h_ff_even_id[-10] = 0.0052
#    h_ff_even_id[-9] = 0.0008
#    h_ff_even_id[-8:] = 0

#    temp_1 = np.matmul(odd.transpose(),
#                       np.matmul(conv.transpose(), resp.transpose()))
#    temp_2 = np.matmul(resp, np.matmul(conv, odd))
#    temp_3 = np.linalg.inv(np.matmul(temp_1, temp_2))


    V_beam_odd = np.matrix(V_beam_odd).transpose()
#    h_ff_odd = np.matmul(odd, np.matmul(temp_3, np.matmul(temp_1, V_beam_odd)))
#    h_ff_odd_old = odd * (odd.T * conv.T * resp.T * resp * conv * odd).I * \
#        odd.T * conv.T * resp.T * V_beam_odd

    h_ff_odd = odd * (resp * conv * odd).I * V_beam_odd
#    h_ff_odd = (resp*conv).I * V_beam_odd


    if debug:
        plt.figure("FF filter")
        plt.plot(h_ff_even, 'bo-', label='even')
#        plt.plot(h_ff_even_2, 'yo-', label='even')
#        plt.plot(h_ff_even_id, color='grey', label='ideal')
        plt.plot(h_ff_odd, 'ro-', label='odd')
        plt.plot(h_ff_even+h_ff_odd, 'go-', label='total')
        plt.axhline(0, color='grey', alpha=0.5)
        plt.xlabel("Samples [1]")
        plt.legend()

        # Reconstructed signal
        V_even = np.matmul(resp, np.matmul(conv, h_ff_even))
        V_odd = np.matmul(resp, np.matmul(conv, h_ff_odd))

        plt.figure("Reconstructed signal")
        plt.plot(V_even, 'bo-', label='even')
        plt.plot(V_odd, 'ro-', label='odd')
        plt.plot(V_even+V_odd, 'go-', label='total')
        plt.axhline(0, color='grey', alpha=0.5)
        plt.xlabel("Samples [1]")
        plt.legend()
        plt.show()

    return h_ff_even + h_ff_odd


