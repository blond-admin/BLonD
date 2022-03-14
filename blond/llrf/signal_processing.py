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


def modulator(signal, omega_i, omega_f, T_sampling, phi_0=0):
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
    delta_phi = (omega_i - omega_f) * T_sampling * np.arange(len(signal))
    # Pre compute sine and cosine for speed up
    cs = np.cos(delta_phi + phi_0)
    sn = np.sin(delta_phi + phi_0)
    I_new = cs*signal.real + sn*signal.imag
    Q_new = - sn*signal.real + cs*signal.imag

    return I_new + 1j*Q_new


def rf_beam_current(Profile, omega_c, T_rev, lpf=True, downsample=None, external_reference=True):
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
    Q_f = -2.*charges*np.sin(omega_c*Profile.bin_centers)

    # Pass through a low-pass filter
    if lpf is True:
        # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
        cutoff = 20.e6*2.*Profile.bin_size
        I_f = low_pass_filter(I_f, cutoff_frequency=cutoff)
        Q_f = low_pass_filter(Q_f, cutoff_frequency=cutoff)
    logger.debug("RF total current is %.4e A", np.fabs(np.sum(I_f))/T_rev)

    charges_fine = I_f + 1j*Q_f
    if external_reference:
        # Phase correction
        bucket = 2 * np.pi/(omega_c)
        # This term takes into account where the sampling of the profile starts
        add_corr = Profile.bin_centers[0] / (bucket/2) - int(Profile.bin_centers[0] / (bucket/2)) \
                   - Profile.bin_size / bucket
        phase = (Profile.bin_centers[0] - Profile.bin_size/2 - 0.5*bucket)/bucket*2*np.pi \
                + np.angle(charges_fine)[0] - np.pi * add_corr
        charges_fine = charges_fine * np.exp(-1j * phase)  # TODO: plus or minus

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
        charges_coarse = np.zeros(n_points, dtype=complex) #+ 1j*np.zeros(n_points)
        charges_coarse[ind_fine[0]] = np.sum(charges_fine[np.arange(indices[0])])
        for i in range(1, len(indices)):
            charges_coarse[i + ind_fine[0]] = np.sum(charges_fine[np.arange(indices[i-1],
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


def moving_average_improved(x, N, x_prev=None):

    if x_prev is not None:
        x = np.concatenate((x_prev, x))


    mov_avg = sgn.fftconvolve(x, (1/N)*np.ones(N), mode='full')[-x.shape[0]:]

    return mov_avg[:x.shape[0] - N + 1]

def H_cav(x, n_sections, x_prev=None):

    if x_prev is not None:
        x = np.concatenate((x_prev, x))

    if n_sections == 3:
        h = np.array([-0.04120219, -0.00765499, -0.00724786, -0.00600952, -0.00380694, -0.00067663,
                      0.00343537, 0.0084533, 0.01421418, 0.02071802, 0.02764441, 0.03476114,
                      0.04193753, 0.04882965, 0.05522681, 0.06083675, 0.0654471, 0.06887487,
                      0.07100091, 0.09043617, 0.07100091, 0.06887487, 0.0654471, 0.06083675,
                      0.05522681, 0.04882965, 0.04193753, 0.03476114, 0.02764441, 0.02071802,
                      0.01421418, 0.0084533, 0.00343537, -0.00067663, -0.00380694, -0.00600952,
                      -0.00724786, -0.00765499, -0.04120219])
    else:
        h = np.array([-0.0671217,   0.01355402,  0.01365686,  0.01444814,  0.01571424,  0.01766679,
                      0.01996413,  0.02251791,  0.02529718,  0.02817416,  0.03113348,  0.03398052,
                      0.03674144,  0.03924433,  0.04153931,  0.04344182,  0.04502165,  0.04612467,
                      0.04685122,  0.06409968,  0.04685122,  0.04612467,  0.04502165,  0.04344182,
                      0.04153931,  0.03924433,  0.03674144,  0.03398052,  0.03113348,  0.02817416,
                      0.02529718,  0.02251791,  0.01996413,  0.01766679,  0.01571424,  0.01444814,
                      0.01365686,  0.01355402, -0.0671217 ])

    resp = sgn.fftconvolve(x, h, mode='full')[-x.shape[0]:]

    return resp[:x.shape[0] - h.shape[0] + 1]


def feedforward_filter(TWC: TravellingWaveCavity, T_s, debug=False, taps=None,
                       opt_output=False):
    """Function to design n-tap FIR filter for SPS TravellingWaveCavity.

    Parameters
    ----------
    TWC : TravellingWaveCavity
        TravellingWaveCavity type class
    T_s : float
        Sampling time [s]
    debug : bool
        When True, activates printouts and plots; default is False
    taps : int
        User-defined number of taps; default is None and number of taps is
        calculated from the filling time
    opt_output : bool
        When True, activates optional output; default is False

    Returns
    -------
    float array
        FIR filter coefficients
    int
        Optional output: Number of FIR filter taps
    int
        Optional output: Filling time in samples
    int
        Optional output: Fitting time in samples, n_filling, n_fit
    """

    # Filling time in samples
    n_filling = int(TWC.tau/T_s)
    logger.debug("Filling time in samples: %d", n_filling)

    # Number of FIR filter taps
    if taps is not None:
        n_taps = int(taps)
    else:
        n_taps = 2*int(0.5*n_filling) + 13 #31
    n_taps_2 = int(0.5*(n_taps+1))
    if n_taps % 2 == 0:
        raise RuntimeError("Number of taps in feedforward filter must be odd!")
    logger.debug("Number of taps: %d", n_taps)

    # Fitting samples
    n_fit = int(n_taps + n_filling)
    logger.debug("Fitting samples: %d", n_fit)

    # Even-symmetric feed-forward filter matrix
    even = np.zeros(shape=(n_taps,n_taps_2), dtype=np.float64)
    for i in range(n_taps):
        even[i,abs(n_taps_2-i-1)] = 1

    # Odd-symmetric feed-forward filter matrix
    odd = np.zeros(shape=(n_taps, n_taps_2-1), dtype=np.float64)
    for i in range(n_taps_2-1):
        odd[i,abs(n_taps_2-i-2)] = -1
        odd[n_taps-i-1, abs(n_taps_2 - i - 2)] = 1

    # Generator-cavity response matrix: non-zero during filling time
    resp = np.zeros(shape=(n_fit, n_fit+n_filling-1), dtype=np.float64)
    for i in range(n_fit):
        resp[i,i:i+n_filling] = 1

    # Convolution with beam step current
    conv = np.zeros(shape=(n_fit+n_filling-1, n_taps), dtype=np.float64)
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
    time_array = np.linspace(0, n_fit*T_s, num=n_fit) - TWC.tau/2
    TWC.impulse_response_beam(TWC.omega_r, time_array)
    h_beam_real = TWC.h_beam.real/TWC.R_beam*TWC.tau

    # Even and odd parts of impulse response
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

    # Beam current step for step response
    I_beam_step = np.ones(n_fit)
    I_beam_step[0] = 0
    I_beam_step[1] = 0.5

    # Even and odd parts of induced voltage
    V_beam_even = sgn.fftconvolve(I_beam_step, h_beam_even, mode='full')[:I_beam_step.shape[0]]
    V_beam_odd = sgn.fftconvolve(I_beam_step, h_beam_odd, mode='full')[:I_beam_step.shape[0]]
    # Normalised response
    norm = np.max(V_beam_even)
    V_beam_even /= norm
    V_beam_odd /= norm

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

    # FIR filter even and odd parts
    h_ff_even = even @ np.linalg.pinv(resp @ conv @ even) @ V_beam_even
    h_ff_odd = odd @ np.linalg.pinv(resp @ conv @ odd) @ V_beam_odd

    if debug:
        plt.figure("FF filter")
        plt.plot(h_ff_even, 'bo-', label='even')
        plt.plot(h_ff_odd, 'ro-', label='odd')
        plt.plot(h_ff_even+h_ff_odd, 'go-', label='total')
        plt.axhline(0, color='grey', alpha=0.5)
        plt.xlabel("Samples [1]")
        plt.legend()

        # Reconstructed signal
        V_even = resp @ conv @ h_ff_even
        V_odd = resp @ conv @ h_ff_odd

        plt.figure("Reconstructed signal")
        plt.plot(V_even, 'bo-', label='even')
        plt.plot(V_odd, 'ro-', label='odd')
        plt.plot(V_even+V_odd, 'go-', label='total')
        plt.axhline(0, color='grey', alpha=0.5)
        plt.xlabel("Samples [1]")
        plt.legend()
        plt.show()

    # Return with or without optional output
    if opt_output:
        return h_ff_even + h_ff_odd, n_taps, n_filling, n_fit
    else:
        return h_ff_even + h_ff_odd


#feedforward_filter_TWC3 = np.array(
#    [-0.0070484734, 0.0161859736, 0.0020289928, 0.0020289928,
#      0.0020289928, -0.0071641302, -0.0162319424, -0.0070388194,
#      0.0020289928, 0.0020289928, 0.0020289928, - 0.0050718734,
#      0.0065971343, 0.0030434892, 0.0030434892, 0.0030434892,
#      0.0030434892, 0.0030434892, -0.0004807475, 0.011136476,
#      0.0040579856, 0.0040579856, 0.0040579856, 0.0132511086,
#      0.019651364, 0.0074147518, -0.0020289928, -0.0020289928,
#     -0.0020289928, -0.0162307252, 0.0071072903])

feedforward_filter_TWC3 = np.array(
    [-0.00760838, 0.01686764, 0.00205761, 0.00205761,
     0.00205761, 0.00205761, -0.03497942, 0.00205761,
     0.00205761, 0.00205761, 0.00205761, -0.0053474,
     0.00689061, 0.00308642, 0.00308642, 0.00308642,
     0.00308642, 0.00308642, -0.00071777, 0.01152024,
     0.00411523, 0.00411523, 0.00411523, 0.00411523,
     0.03806584, -0.00205761, -0.00205761, -0.00205761,
     -0.00205761, -0.01686764, 0.00760838])


#feedforward_filter_TWC4 = np.array(
#    [0.0048142895, 0.0035544775, 0.0011144336, 0.0011144336,
#     0.0011144336, -0.0056984584, -0.0122587698, -0.0054458778,
#     0.0011144336, 0.0011144336, 0.0011144336, -0.0001684528,
#     -0.000662115, 0.0016716504, 0.0016716504, 0.0016716504,
#     0.0016716504, 0.0016716504, 0.0016716504, 0.0016716504,
#     0.0016716504, 0.0016716504, 0.0016716504, 0.0016716504,
#     0.0040787952, 0.0034488892, 0.0022288672, 0.0022288672,
#     0.0022288672, 0.0090417593, 0.0146881621, 0.0062036196,
#     -0.0011144336, -0.0011144336, -0.0011144336, -0.0036802064,
#     -0.0046675309])

feedforward_filter_TWC4 = np.array(
    [0.01050256, -0.0014359, 0.00106667, 0.00106667,
     0.00106667, -0.01226667, -0.01226667, 0.00106667,
     0.00106667, 0.00106667, 0.00231795, -0.00365128,
     0.0016, 0.0016, 0.0016, 0.0016,
     0.0016, 0.0016, 0.0016, 0.0016,
     0.0016, 0.0016, 0.0016, 0.0016,
     0.0016, 0.00685128, 0.00088205, 0.00213333,
     0.00213333, 0.00213333, 0.01506667, 0.01266667,
     -0.00106667, -0.00106667, -0.00106667, 0.0014359,
     -0.01050256])

feedforward_filter_TWC5 = np.array(
    [0.0189205535, -0.0105637125, 0.0007262783, 0.0007262783,
     0.0006531768, -0.0105310359, -0.0104579343, 0.0007262783,
     0.0007262783, 0.0007262783, 0.0063272331, -0.0083221785,
     0.0010894175, 0.0010894175, 0.0010894175, 0.0010894175,
     0.0010894175, 0.0010894175, 0.0010894175, 0.0010894175,
     0.0010894175, 0.0010894175, 0.0010894175, 0.0010894175,
     0.0010894175, 0.0010894175, 0.0010894175, 0.0010894175,
     0.0010894175, 0.0010894175, 0.0010894175, 0.0105496942,
     -0.0041924387, 0.0014525567, 0.0014525567, 0.0013063535,
     0.0114011487, 0.0104579343, -0.0007262783, -0.0007262783,
     -0.0007262783, 0.0104756312, -0.018823192])
