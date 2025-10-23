# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Methods to generate RF phase noise from noise spectrum and feedback noise
amplitude as a function of bunch length**

:Authors: **Helga Timko**
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from blond.utils import bmath as bm

import numpy as np
import numpy.random as rnd

from ..beam.profile import Profile
from ..plots.plot import fig_folder
from ..plots.plot_llrf import plot_phase_noise, plot_noise_spectrum
from ..toolbox.next_regular import next_regular
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Optional, Literal, Tuple, Sequence, Callable

    from numpy.typing import ArrayLike, NDArray as NumpyArray

    from ..input_parameters.rf_parameters import RFStation
    from ..input_parameters.ring import Ring

CFWHM = np.sqrt(2.0 / np.log(2.0))


class FlatSpectrum:
    @handle_legacy_kwargs
    def __init__(
        self,
        ring: Ring,
        rf_station: RFStation,
        delta_f: float = 1,
        corr_time: int = 10000,
        fmin_s0: float = 0.8571,
        fmax_s0: float = 1.1,
        initial_amplitude: float = 1.0e-6,
        seed1: int = 1234,
        seed2: int = 7564,
        predistortion: Optional[Literal["weightFunction"]] = None,
        continuous_phase: bool = False,
        folder_plots: str = "fig_noise",
        print_option: bool = True,
        initial_final_turns: Tuple[int] = (0, -1),
    ):
        """
        Generate phase noise from a band-limited spectrum.
        Input frequency band using 'fmin' and 'fmax' w.r.t. the synchrotron
        frequency. Input double-sided spectrum amplitude [rad^2/Hz] using
        'initial_amplitude'. Fix seeds to obtain reproducible phase noise.
        Select 'time_points' suitably to resolve the spectrum in frequency
        domain. After 'corr_time' turns, the seed is changed to cut numerical
        correlated sequences of the random number generator.
        """
        self.total_n_turns = ring.n_turns
        self.initial_final_turns = list(initial_final_turns)
        if self.initial_final_turns[1] == -1:
            self.initial_final_turns[1] = self.total_n_turns + 1

        # revolution frequency in Hz
        self.f0 = ring.f_rev[
            self.initial_final_turns[0] : self.initial_final_turns[1]
        ]
        self.delta_f = delta_f  # frequency resolution [Hz]
        self.corr = corr_time  # adjust noise every 'corr' time steps
        self.fmin_s0 = fmin_s0  # spectrum lower bound in synchr. freq.
        self.fmax_s0 = fmax_s0  # spectrum upper bound in synchr. freq.
        self.A_i = initial_amplitude  # initial spectrum amplitude [rad^2/Hz]
        self.seed1 = seed1
        self.seed2 = seed2
        self.predistortion = predistortion
        if self.predistortion == "weightfunction":
            # Overwrite frequencies
            self.fmin_s0 = 0.8571
            self.fmax_s0 = 1.001
        self.fs = rf_station.omega_s0[
            self.initial_final_turns[0] : self.initial_final_turns[1]
        ] / (2 * np.pi)  # synchrotron frequency in Hz
        self.n_turns = len(self.fs) - 1
        self.dphi = bm.zeros(self.n_turns + 1)
        self.continuous_phase = continuous_phase
        if self.continuous_phase:
            self.dphi2 = bm.zeros(self.n_turns + 1 + self.corr / 4)
        self.folder_plots = folder_plots
        self.print_option = print_option

    def spectrum_to_phase_noise(
        self,
        freq: NumpyArray,
        spectrum: NumpyArray,
        transform: Optional[str] = None,
    ):
        nf = len(spectrum)
        fmax = freq[nf - 1]

        # Resolution in time domain
        if transform is None or transform == "r":
            nt = 2 * (nf - 1)
            dt = 1 / (2 * fmax)  # in [s]
        elif transform == "c":
            nt = nf
            dt = 1.0 / fmax  # in [s]
        else:
            # NoiseError
            raise RuntimeError(
                'ERROR: The choice of Fourier transform for the\
             RF noise generation could not be recognized. Use "r" or "c".'
            )

        # Generate white noise in time domain
        rnd.seed(self.seed1)
        r1 = rnd.random_sample(nt)
        rnd.seed(self.seed2)
        r2 = rnd.random_sample(nt)
        if transform is None or transform == "r":
            Gt = np.cos(2 * np.pi * r1) * np.sqrt(-2 * np.log(r2))
        elif transform == "c":
            Gt = np.exp(2 * np.pi * 1j * r1) * np.sqrt(-2 * np.log(r2))

        # FFT to frequency domain
        if transform is None or transform == "r":
            Gf = bm.fft.rfft(Gt)
        elif transform == "c":
            Gf = bm.fft.fft(Gt)

        # Multiply by desired noise probability density
        if transform is None or transform == "r":
            s = bm.sqrt(2 * fmax * spectrum)  # in [rad]
        elif transform == "c":
            s = bm.sqrt(fmax * spectrum)  # in [rad]
        dPf = s * Gf.real + 1j * s * Gf.imag  # in [rad]

        # FFT back to time domain to get final phase shift
        if transform is None or transform == "r":
            dPt = bm.fft.irfft(dPf)  # in [rad]
        elif transform == "c":
            dPt = bm.fft.ifft(dPf)  # in [rad]

        # Use only real part for the phase shift and normalize
        self.t = bm.linspace(0, float(nt * dt), nt)
        self.dphi_output = dPt.real

    def generate(self):
        for i in range(0, int(np.ceil(self.n_turns / self.corr))):
            # Scale amplitude to keep area (phase noise amplitude) constant
            k = i * self.corr  # current time step
            ampl = self.A_i * self.fs[0] / self.fs[k]

            # Calculate the frequency step
            f_max = self.f0[k] / 2
            n_points_pos_f_incl_zero = int(np.ceil(f_max / self.delta_f) + 1)
            nt = 2 * (n_points_pos_f_incl_zero - 1)
            nt_regular = next_regular(int(nt))
            if nt_regular % 2 != 0 or nt_regular < self.corr:
                # NoiseError
                raise RuntimeError("Error in noise generation!")
            n_points_pos_f_incl_zero = int(nt_regular / 2 + 1)
            freq = bm.linspace(0, float(f_max), n_points_pos_f_incl_zero)
            delta_f = f_max / (n_points_pos_f_incl_zero - 1)

            # Construct spectrum
            nmin = int(np.floor(self.fmin_s0 * self.fs[k] / delta_f))
            nmax = int(np.ceil(self.fmax_s0 * self.fs[k] / delta_f))

            # To compensate the notch due to PL at central frequency
            if self.predistortion == "exponential":
                spectrum = bm.concatenate(
                    (
                        bm.zeros(nmin),
                        ampl
                        * bm.exp(
                            np.log(100.0)
                            * bm.arange(0, nmax - nmin + 1)
                            / (nmax - nmin)
                        ),
                        bm.zeros(n_points_pos_f_incl_zero - nmax - 1),
                    )
                )

            elif self.predistortion == "linear":
                spectrum = bm.concatenate(
                    (
                        bm.zeros(nmin),
                        bm.linspace(0, float(ampl), nmax - nmin + 1),
                        bm.zeros(n_points_pos_f_incl_zero - nmax - 1),
                    )
                )

            elif self.predistortion == "hyperbolic":
                spectrum = bm.concatenate(
                    (
                        bm.zeros(nmin),
                        ampl
                        * bm.ones(nmax - nmin + 1)
                        * 1
                        / (
                            1
                            + 0.99
                            * (nmin - bm.arange(nmin, nmax + 1))
                            / (nmax - nmin)
                        ),
                        bm.zeros(n_points_pos_f_incl_zero - nmax - 1),
                    )
                )

            elif self.predistortion == "weightfunction":
                frel = (
                    freq[nmin : nmax + 1] / self.fs[k]
                )  # frequency relative to fs0
                frel[bm.where(frel > 0.999)[0]] = (
                    0.999  # truncate center freqs
                )
                sigma = (
                    0.754  # rms bunch length in rad corresponding to 1.2 ns
                )
                gamma = 0.577216
                weight = (4.0 * np.pi * frel / sigma**2) ** 2 * bm.exp(
                    -16.0 * (1.0 - frel) / sigma**2
                ) + 0.25 * (
                    1
                    + 8.0
                    * frel
                    / sigma**2
                    * bm.exp(-8.0 * (1.0 - frel) / sigma**2)
                    * (
                        gamma
                        + bm.log(8.0 * (1.0 - frel) / sigma**2)
                        + 8.0 * (1.0 - frel) / sigma**2
                    )
                ) ** 2
                weight /= weight[0]  # normalise to have 1 at fmin
                spectrum = bm.concatenate(
                    (
                        bm.zeros(nmin),
                        ampl * weight,
                        bm.zeros(n_points_pos_f_incl_zero - nmax - 1),
                    )
                )

            else:
                spectrum = bm.concatenate(
                    (
                        bm.zeros(nmin),
                        ampl * bm.ones(nmax - nmin + 1),
                        bm.zeros(n_points_pos_f_incl_zero - nmax - 1),
                    )
                )

            # Fill phase noise array
            if i < int(self.n_turns / self.corr) - 1:
                kmax = (i + 1) * self.corr
            else:
                kmax = self.n_turns + 1

            self.spectrum_to_phase_noise(freq, spectrum)
            self.seed1 += 239
            self.seed2 += 158
            self.dphi[k:kmax] = self.dphi_output[0 : (kmax - k)]

            if self.continuous_phase:
                if i == 0:
                    self.spectrum_to_phase_noise(freq, spectrum)
                    self.seed1 += 239
                    self.seed2 += 158
                    self.dphi2[: self.corr / 4] = self.dphi_output[
                        : self.corr / 4
                    ]

                self.spectrum_to_phase_noise(freq, spectrum)
                self.seed1 += 239
                self.seed2 += 158
                self.dphi2[(k + self.corr / 4) : (kmax + self.corr / 4)] = (
                    self.dphi_output[0 : (kmax - k)]
                )

            if self.folder_plots is not None:
                fig_folder(self.folder_plots)
                plot_noise_spectrum(
                    freq,
                    spectrum,
                    sampling=1,
                    figno=i,
                    dirname=self.folder_plots,
                )
                plot_phase_noise(
                    self.t[0 : (kmax - k)],
                    self.dphi_output[0 : (kmax - k)],
                    sampling=1,
                    figno=i,
                    dirname=self.folder_plots,
                )

            rms_noise = bm.std(self.dphi_output)
            if self.print_option:
                print(
                    "RF noise for time step %.4e s (iter %d) has r.m.s. phase %.4e rad (%.3e deg)"
                    % (self.t[1], i, rms_noise, rms_noise * 180 / np.pi)
                )

        if self.continuous_phase:
            psi = bm.arange(0, self.n_turns + 1) * 2 * np.pi / self.corr
            self.dphi = self.dphi * bm.sin(
                psi[: self.n_turns + 1]
            ) + self.dphi2[: (self.n_turns + 1)] * bm.cos(
                psi[: self.n_turns + 1]
            )

        if (self.initial_final_turns[0] > 0) or (
            self.initial_final_turns[1] < self.total_n_turns + 1
        ):
            self.dphi = bm.concatenate(
                (
                    bm.zeros(self.initial_final_turns[0]),
                    self.dphi,
                    bm.zeros(
                        1 + self.total_n_turns - self.initial_final_turns[1]
                    ),
                )
            )


class LHCNoiseFB:
    """
    *Feedback on phase noise amplitude for LHC controlled longitudinal emittance
    blow-up using noise injection through cavity controller or phase loop.
    The feedback compares the FWHM bunch length of the bunch to a target value
    and scales the phase noise to keep the targeted value.
    Activate the feedback either by passing it in RfStation or in
    the PhaseLoop object.
    Update the noise amplitude scaling using track().
    Pass the bunch pattern (occupied bucket numbers from 0...h-1) in buckets
    for multi-bunch simulations; the feedback uses the average bunch length.*

    Input parameters:
    - rf_station: RFStation object
    - profile: Profile object
    - bl_target: target bunch length [s]
    - gain: feedback gain [1/s]
    - factor: feedback recursion scaling factor
    - update_frequency: update feedback every n_update turns
    - variable_gain: switch to use constant or variable gain
    - bunch_pattern: bunch pattern for multi-bunch simulations
    - old_FESA_class: buffer size for noise injection 2s and delayed application of 2 buffers
    - no_delay: switch to not use delay on the BQM and noise injection
    - seed: seed for the random number generator
    """

    def __init__(
        self,
        rf_station: RFStation,
        profile: Profile,
        f_rev: float,
        bl_target: float,
        gain: float = 0.1e9,
        factor: float = 0.93,
        update_frequency: int = 11245,
        variable_gain: bool = True,
        bunch_pattern: Optional[ArrayLike] = None,
        old_FESA_class: bool = False,
        no_delay: bool = False,
        seed: int | None = 1313,
    ):
        self.LHC_frev = round(f_rev)  # LHC revolution frequency in Hz

        #: | *Import RfStation*
        self.rf_params = rf_station

        #: | *Import Profile*
        self.profile: Profile = profile

        #: | *Phase noise scaling factor. Initially 0.*
        self.x = 0.0

        #: | *Target bunch length [s], 4-sigma value.*
        self.bl_targ = bl_target

        #: | *Measured bunch length [s], FWHM.*
        self.bl_meas = bl_target

        #: | *Feedback recursion scaling factor.*
        self.a = factor

        #: | *Update feedback every n_update turns.*
        self.n_update = update_frequency

        #: | *Switch to use constant or variable gain*
        self.variable_gain = variable_gain

        #: | *Feedback gain [1/s].*
        if self.variable_gain:
            self.g = (
                gain
                * (self.rf_params.omega_s0[0] / self.rf_params.omega_s0) ** 2
            )
        else:
            self.g = gain * bm.ones(self.rf_params.n_turns + 1)

        #: | *Bunch pattern for multi-bunch simulations*
        self.bunch_pattern = bunch_pattern

        #: | *Flag to not use delay on the BQM and noise injection - it measures self.bl_meas and updates self.x instantly*
        self.no_delay = no_delay

        #: | *Switch to use old FESA class: buffer size for noise injection 2s and delayed application of 2 buffers*
        self.old_FESA_class = old_FESA_class

        #: | *Function dictionary to calculate FWHM bunch length*
        fwhm_functions = {
            "single": self.fwhm_single_bunch,
            "multi": self.fwhm_multi_bunch,
        }
        if self.bunch_pattern is None:
            self.fwhm = fwhm_functions["single"]
            self.bl_meas_bbb = None
        else:
            self.bunch_pattern = bm.ascontiguousarray(self.bunch_pattern)
            self.bl_meas_bbb = bm.zeros(len(self.bunch_pattern))
            self.fwhm = fwhm_functions["multi"]

        # Initialize the BQM delay in respect to noise injection
        rnd.seed(seed)
        self.delay = int(rnd.uniform(0, 1.1) * self.LHC_frev)  # in turns

        # Initialize buffers for the last 5 bqm measurements and their timestamps
        self.last_bqm_measurements = bm.empty(5)
        self.time_array = bm.empty(5)
        self.update_x = False

        if self.old_FESA_class:
            # In the old FESA class the x_amplitudes buffer was updated every 2s
            self.timers = [
                CallEveryNTurns(
                    self.LHC_frev * 2, self.update_noise_amplitude
                ),
                CallEveryNTurns(
                    int(self.LHC_frev * 1.1),
                    self.update_bqm_measurement,
                    delay=self.delay,
                ),
            ]

            self.delay_noise_inj = (
                2 * self.LHC_frev * 2
            )  # in turns - delay noise injection for 2 chunks
        else:
            # In the new FESA class the x_amplitudes buffer is updated every 1s
            self.timers = [
                CallEveryNTurns(self.LHC_frev, self.update_noise_amplitude),
                CallEveryNTurns(
                    int(self.LHC_frev * 1.1),
                    self.update_bqm_measurement,
                    delay=self.delay,
                ),
            ]

            self.delay_noise_inj = (
                self.LHC_frev
            )  # in turns - delay noise injection for 1 chunk

    def track(self):
        """
        *Calculate PhaseNoise Feedback scaling factor as a function of
        measured FWHM bunch length.* Take into account the delay and asynchronisation between the BQM and the x update.
        """

        if self.no_delay:
            # Track only in certain turns
            if (self.rf_params.counter[0] % self.n_update) == 0:
                # Update bunch length, every x turns determined in main file
                self.fwhm()

                # Update noise amplitude-scaling factor
                self.x = self.a * self.x + self.g[
                    self.rf_params.counter[0]
                ] * (self.bl_targ - self.bl_meas)

                # Limit to range [0,1]
                self.x = bm.maximum(0, bm.minimum(self.x, 1))

        else:
            for timer in self.timers:
                timer.tick()

    def update_bqm_measurement(self):
        # Takes the bunch length measurement and updates self.bl_meas
        self.fwhm()

        if self.timers[1].counter == self.delay:
            # Write buffers using the first measurement
            self.last_bqm_measurements = bm.full(5, self.bl_meas)
            self.time_array = bm.full(5, self.timers[1].counter)
            # Checks that the first bqm measurement was taken
            self.update_x = True
            return

        # Update buffers by rotating them to the left and adding the new measurement at the end
        self.last_bqm_measurements = bm.roll(self.last_bqm_measurements, -1)
        self.last_bqm_measurements[-1] = self.bl_meas
        self.time_array = bm.roll(self.time_array, -1)
        self.time_array[-1] = self.timers[1].counter

    def update_noise_amplitude(self):
        # timestamp in turns, before which the last bqm measurement was taken
        timestamp = self.timers[0].counter - self.delay_noise_inj

        if not self.update_x or timestamp < self.delay:
            # If the first bqm measurement has not been taken yet, or cannot be used yet because of the delay,
            # set x to 0
            self.x = 0
            return

        # Find the index of the last bqm measurement taken before timestamp
        idx = bm.amax(bm.where(self.time_array < timestamp)[0])
        bqm_measurement = self.last_bqm_measurements[idx]

        # Update noise amplitude-scaling factor
        x = self.a * self.x + self.g[self.rf_params.counter[0]] * (
            self.bl_targ - bqm_measurement
        )
        self.x = bm.maximum(0, bm.minimum(x, 1))

    def fwhm_interpolation(
        self, index: Sequence[int], half_height: int
    ) -> float:
        time_resolution = (
            self.profile.bin_centers[1] - self.profile.bin_centers[0]
        )

        left = (
            self.profile.bin_centers[index[0]]
            - (self.profile.n_macroparticles[index[0]] - half_height)
            / (
                self.profile.n_macroparticles[index[0]]
                - self.profile.n_macroparticles[index[0] - 1]
            )
            * time_resolution
        )

        right = (
            self.profile.bin_centers[index[-1]]
            + (self.profile.n_macroparticles[index[-1]] - half_height)
            / (
                self.profile.n_macroparticles[index[-1]]
                - self.profile.n_macroparticles[index[-1] + 1]
            )
            * time_resolution
        )

        return CFWHM * (right - left)

    def fwhm_single_bunch(self):
        """
        *Single-bunch FWHM bunch length calculation with interpolation.*
        """

        half_height = bm.max(self.profile.n_macroparticles) / 2.0
        index = bm.where(self.profile.n_macroparticles > half_height)[0]

        self.bl_meas = self.fwhm_interpolation(index, half_height)

    def fwhm_multi_bunch(self):
        """
        *Multi-bunch FWHM bunch length calculation with interpolation.*
        """

        # Find correct RF buckets
        phi_RF = self.rf_params.phi_rf[0, self.rf_params.counter[0]]
        omega_RF = self.rf_params.omega_rf[0, self.rf_params.counter[0]]
        bucket_min = (phi_RF + 2.0 * np.pi * self.bunch_pattern) / omega_RF
        bucket_max = bucket_min + 2.0 * np.pi / omega_RF

        # Bunch-by-bunch FWHM bunch length
        for i in range(len(self.bunch_pattern)):
            bind = bm.where(
                (self.profile.bin_centers - bucket_min[i])
                * (self.profile.bin_centers - bucket_max[i])
                < 0
            )[0]
            hheight = bm.max(self.profile.n_macroparticles[bind]) / 2.0
            index = bm.where(self.profile.n_macroparticles[bind] > hheight)[0]
            self.bl_meas_bbb[i] = self.fwhm_interpolation(bind[index], hheight)

        # Average FWHM bunch length
        self.bl_meas = bm.mean(self.bl_meas_bbb)


class CallEveryNTurns:
    """
    *Call a function every n turns

    n_turns: number of turns between calls
    function: function to call
    delay: delay in turns before the first call*
    """

    def __init__(self, n_turns: int, function: Callable, delay: int = 0):
        self.n_turns = n_turns
        self.counter = 0
        self.function = function
        self.delay = delay

    def __call__(self):
        self.tick()

    def tick(self):
        if (self.counter - self.delay) % self.n_turns == 0:
            self.function()
        self.counter += 1
