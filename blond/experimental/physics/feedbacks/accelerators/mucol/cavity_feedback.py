# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Cavity feedback stubs for the muon collider."""
from typing import Any
from warnings import warn

import numpy as np
from numpy.typing import NDArray as NumpyArray
from scipy.constants import e
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from blond import Simulation, StaticProfile
from blond.core.beam.base import BeamBaseClass
from blond.experimental.physics.feedbacks.accelerators.lhc.helpers import (
    cavity_response_sparse_matrix,
)
from blond.experimental.physics.feedbacks.cavity_feedback import (
    IQCavityFeedback,
)
from blond.experimental.physics.feedbacks.helpers import (
    low_pass_filter,
)

MINIMUM_QL_FEEDBACK_MODEL = 0.5

class PassiveCavity(IQCavityFeedback):
    r"""
    Passive Cavity, implementing the beam-cavity interaction formulas without a feedback involved.

    Parameters
    ----------
        profile
            profile on which the feedback should act
        R_over_Q
            shunt impedance over quality factor of one cavity [$$\Omega$$]
        Q_L
            Loaded quality factor of one cavity [1]
        f_center
            center frequency of the cavity [Hz]
        f_detuning
            detuning of the cavity [Hz]
        n_cavities
            number of cavities
        generator_current
            given in [A]
        generator_phase
            given in [rad]
        injection_phase
            In :func:`xxxx` the cavity will optimise the phase at injection towards
            this value by adjusting the initial parameters at the beginning of the internal tracking given in [rad]
        injection_voltage
            In :func:`xxxx` the cavity will optimise the voltage at injection towards
            this value by adjusting the initial parameters at the beginning of the internal tracking given in [V]
        harmonic_index
            only the default of 0 is allowed
        n_periods_coarse
            number of RF periods, one coarse grid corresponds to
        section_index
            section which the feedback belongs to
        use_lowpass_filter
            Used in :func:xxx
        name
            If not given, is automatically chosen
        fine_RK
            Use Runge Kutta direct calculation for fine grid instead of matrix formalism
        fine_RK
            Use Runge Kutta direct calculation for coarse grid instead of matrix formalism
    """

    def __init__(self,
                 profile: StaticProfile,  # is this stricly necessary?
                 R_over_Q: float,
                 Q_L: float,
                 f_center: float,
                 f_detuning: float,
                 n_cavities: int,
                 generator_current: float,
                 generator_phase: float = 0,
                 injection_phase: float = -1,
                 injection_voltage: float = -1,
                 harmonic_index: int = 0,
                 n_periods_coarse: int = 1,
                 section_index: int = 0,
                 use_lowpass_filter: bool = False,
                 name: str | None = None,
                 fine_RK: bool = False,
                 coarse_RK: bool = False) -> None:
        if harmonic_index != 0:
            raise NotImplementedError("harmonic indices other than 0 are not supported with this module")

        assert R_over_Q >= 0, "R_over_Q must be >= 0"
        self.R_over_Q = R_over_Q

        assert Q_L >= MINIMUM_QL_FEEDBACK_MODEL, "Q_L must be >= 0.5"
        self.Q_L = Q_L

        assert f_center >= 0, "f_center must be >= 0"  # TODO: does this make sense here?
        self.f_center = f_center

        assert f_detuning >= 0, "fset must be >= 0"
        self.f_detuning = f_detuning
        self.omega_detuning = 2 * np.pi * self.f_detuning
        self.omega_center = 2 * np.pi * self.f_center - self.omega_detuning

        assert n_cavities > 0, "n_cavities must be > 0"
        self.n_cavities = n_cavities

        self.generator_current = generator_current
        self.generator_phase = generator_phase
        self.injection_phase = injection_phase
        self.injection_voltage = injection_voltage

        if use_lowpass_filter:
            warn("lowpass filter is not used in this class", stacklevel=2)

        super().__init__(profile=profile,
                         n_cavities=n_cavities,
                         section_index=section_index,
                         # TODO: this should not be necessary or? The parent cavity already has this information
                         name=name,
                         n_periods_coarse=n_periods_coarse,
                         harmonic_index=harmonic_index,
                         use_lowpass_filter=False)

        # lateinit arrays
        self.sampling_time: float | None = None
        self.n_coarse: int | None = None
        self.omega_carrier: float | None = None

        self.i_generator_fine: NumpyArray | None = None
        self.i_generator_coarse: NumpyArray | None = None

        self.v_antenna_fine: NumpyArray | None = None
        self.v_antenna_coarse: NumpyArray | None = None

        self.i_beam_fine: NumpyArray | None = None
        self.i_beam_coarse: NumpyArray | None = None

        self.i_beam_gradient_fine: NumpyArray | None = None
        self.i_beam_gradient_coarse: NumpyArray | None = None

        self.relative_voltage_correction: float | None = None
        self.phase_correction: float | None = None

        self.fine_RK = fine_RK
        self.coarse_RK = coarse_RK

        self.samples_coarse: int | None = None
        self.samples_fine: int | None = None
        self.relative_detuning: float | None = None

        self.delta_t: float | None = None

    def update_fb_variables(self) -> None:
        """Method to update the variables specific to the turn."""
        self.omega_center = self.omega_rf - self.omega_detuning
        omega_deviation = self.omega_center - self.omega_rf
        # Dimensionless
        self.samples_coarse = self.omega_rf * self.T_s
        self.samples_fine = self.omega_rf * self.profile.hist_step
        self.relative_detuning = omega_deviation / self.omega_center

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """Hook called when ``run_simulation`` is invoked.

        This is a lifecycle hook that can be overridden by subclasses or used by
        simulation elements to perform setup tasks before the main simulation loop
        begins. It is called automatically by ``finalize()``.

        All objects in the simulation hierarchy that have an ``on_run_simulation``
        method will have it called in a specific dependency order.

        Parameters
        ----------
        simulation
            The simulation instance (usually ``self``).
        beam
            The first beam that will be tracked (primary beam in multi-beam scenarios).
        n_turns
            Number of turns that will be simulated.
        turn_i_init
            Starting turn number.
        **kwargs
            Additional keyword arguments for extendability.

        Notes
        -----
        - This is called before each ``run_simulation()`` or ``load_results()`` call.
        - Useful for pre-allocating arrays, resetting state, or computing derived parameters.
        - The base implementation does nothing.

        See Also
        --------
        on_init_simulation
        finalize
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
            **kwargs,
        )

        self.update_fb_variables()
        self.update_rf_variables()

        t_rf = 2 * np.pi / self._parent_rf_station.omega_rf
        self.sampling_time = self.n_periods_coarse * t_rf

    def circuit_track(self, no_beam: bool = False) -> None:
        r"""Tracking of the LLRF circuit."""
        # Compute antenna voltage
        self.V_ANT_COARSE[:self.n_coarse] = self.V_ANT_COARSE[-self.n_coarse:]
        time = np.arange(0, self.n_coarse) * self.T_s
        V_init = self.V_ANT_COARSE[-(1 + self.n_coarse)]
        dV_init = (self.V_ANT_COARSE[-(2 + self.n_coarse)] - self.V_ANT_COARSE[-(1 + self.n_coarse)]) / self.T_s
        v_ant = None
        if self.coarse_RK:
            _, v_ant = self.runge_kutta_tryout_2nd_order(V_init=V_init, dV_ant_init=dV_init,
                                                         delta_omega=self.omega_detuning,
                                                         omega=self.omega_carrier, bin_centers=time)
        else:
            v_ant = cavity_response_sparse_matrix(I_beam=self.I_BEAM_COARSE[-self.n_coarse:],
                                                  I_gen=self.I_GEN_COARSE[-self.n_coarse:],
                                                  n_samples=self.n_coarse,
                                                  V_ant_init=self.V_ANT_COARSE[-(1 + self.n_coarse)],
                                                  I_gen_init=self.I_GEN_COARSE[-(1 + self.n_coarse)],
                                                  samples_per_rf=self.samples_coarse,
                                                  R_over_Q=self.R_over_Q, Q_L=self.Q_L, detuning=self.relative_detuning)
        self.V_ANT_COARSE[-self.n_coarse:] = v_ant[-self.n_coarse:]
        # np.savez("coarse_array_elements_2.npz", I_GEN_COARSE=self.I_GEN_COARSE, I_BEAM_COARSE=self.I_BEAM_COARSE,
        #          samples_per_rf=self.samples, n_samples=self.n_coarse,
        #          I_gen_init=self.I_GEN_COARSE[-(1 + self.n_coarse)], V_ant_init=self.V_ANT_COARSE[-(1 + self.n_coarse)],
        #          V_ANT_COARSE=self.V_ANT_COARSE)
        if not no_beam:
            # Compute generator current on fine-grid
            self.I_GEN_FINE = np.interp(self.profile.hist_x, self.rf_centers,
                                        self.I_GEN_COARSE[-self.n_coarse:])
            # Compute antenna voltage on the fine-grid
            self.cavity_response_fine()

    def runge_kutta_tryout_2nd_order(self, V_init, bin_centers, omega, delta_omega,
                                     method="RK23", min_val=True, dV_ant_init=0+0j):
        """DOCM."""
        max_tstep = bin_centers[1] - bin_centers[0]

        dcurrent = interp1d(bin_centers, -.5 * self.i_beam_gradient_coarse[-self.n_coarse:] + 2j * omega * self.I_GEN_COARSE[-self.n_coarse:])

        coeff_A = -2 * (0.5 * delta_omega ** 2 + delta_omega * omega) / (omega * self.R_over_Q ) + 1j * omega / (
                    self.R_over_Q * self.Q_L)
        coeff_dA = (2j * (1 + delta_omega / omega) + 1 / self.Q_L) / self.R_over_Q

        def fun(t, Y, curr):
            a, dA = Y
            res = omega * self.R_over_Q * (curr(t) - coeff_A * a - coeff_dA * dA)  #TODO: find out, why /2 is required
            return [dA, res]

        A0 = V_init
        dA0 = dV_ant_init
        init_vals = [A0, dA0]
        if min_val:
            sol = solve_ivp(fun, (bin_centers[0], bin_centers[-1]), init_vals, t_eval=bin_centers, method=method,
                             max_step=max_tstep,
                             args=[dcurrent])
        else:
            sol = solve_ivp(fun, (bin_centers[0], bin_centers[-1]), init_vals, t_eval=bin_centers, method=method,
                             args=[dcurrent])

        return sol["t"], sol["y"][0]

    def cavity_response_fine(self):
        r"""ACS cavity response model in matrix form on the fine-grid."""
        # Number of samples on fine grid
        self.samples_fine = self.omega_rf * self.profile.hist_step

        # Find initial value of antenna voltage and generator current
        t_at_init = self.profile.hist_x[0] - self.profile.hist_step

        V_A_init = interp1d(np.concatenate((self.rf_centers - self.T_s * self.n_coarse, self.rf_centers)),
                            self.V_ANT_COARSE)(t_at_init)
        dV_A_init = interp1d(np.concatenate((self.rf_centers - self.T_s * self.n_coarse, self.rf_centers)),
                            np.append(np.diff(self.V_ANT_COARSE), 0) / self.T_s)(t_at_init)
        # print(dV_A_init)
        # dV_A_init = 0 + 0j
        if self.fine_RK:
            _, self.V_ANT_FINE = self.runge_kutta_tryout_2nd_order(dV_ant_init=dV_A_init,
                                                                   delta_omega=self.omega_detuning,
                                                                   V_init=V_A_init, bin_centers=self.profile.hist_x,
                                                                   min_val=True, omega=self.omega_center)
        else:
            I_gen_init = interp1d(np.concatenate((self.rf_centers - self.T_s * self.n_coarse, self.rf_centers)),
                                  self.I_BEAM_COARSE)(t_at_init)

            relative_detuning = self.omega_detuning / self.omega_center
            self.V_ANT_FINE = cavity_response_sparse_matrix(I_beam=self.I_BEAM_FINE,
                                                            I_gen=self.I_GEN_FINE,
                                                            n_samples=self.profile.n_bins,
                                                            V_ant_init=V_A_init,
                                                            I_gen_init=I_gen_init,
                                                            samples_per_rf=self.samples_fine,
                                                            R_over_Q=self.R_over_Q, Q_L=self.Q_L, detuning=relative_detuning)

        self.V_ANT_FINE[-self.profile.n_bins:] = self.V_ANT_FINE[-self.profile.n_bins:] * self.n_cavities
        # np.savez("params_fine_RCS4.npz", V_ant_fine=self.V_ANT_FINE, I_GEN_fine=self.I_GEN_FINE,
        #          I_BEAM_FINE=self.I_BEAM_FINE, samples=self.samples_fine,
        #          I_gen_init=I_gen_init, V_ant_init=V_A_init,
        #          n_samples=self.profile.n_slices, R_over_Q=self.R_over_Q,
        #          Q_L=self.Q_L, detuning=self.detuning,
        #          bin_centers=self.profile.bin_centers,
        #          n_macroparticles=self.profile.n_macroparticles,
        #          omega_det=self.omega_det,
        #          omega_rf=self.omega_c,
        #          ratio=self.profile.beam.ratio,
        #          charge=self.profile.beam.particle.charge)

    def rf_beam_current(
        self,
        beam: BeamBaseClass,
        use_lowpass_filter: bool = False,
    ) -> None:
        r"""Update the RF beam current."""
        if not self.fine_RK and not self.coarse_RK:
            super().rf_beam_current(beam=beam, use_lowpass_filter=use_lowpass_filter)
        else:
            self.i_beam_gradient_fine, self.i_beam_gradient_coarse[-self.n_coarse:] = self.rf_beam_current_gradient(
                beam=beam,
                lpf=use_lowpass_filter,
                downsample={"Ts": self.T_s, "points": self.n_coarse},
                external_reference=True,
                delta_t=self.delta_t,
            )

            # Convert RF beam current gradients to be in units of Amperes
            self.i_beam_gradient_fine = -self.i_beam_gradient_fine / self.profile.hist_step
            self.i_beam_gradient_coarse[-self.n_coarse:] = (
                    -self.i_beam_gradient_coarse[-self.n_coarse:] / self.T_s
            )

    def rf_beam_current_gradient(self,
                                 beam: BeamBaseClass,
                                 lpf: bool = True,
                                 downsample: dict | None = None,
                                 external_reference: bool = True,
                                 delta_t: float = 0,
                                 ) -> tuple[NumpyArray, NumpyArray]:
        r"""Function calculating the beam charge gradient at the (RF) frequency.

        The charge distribution [C] of the beam is determined from the beam
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

        For multi-bunch cases, make sure that the real beam intensity is the total
        number of charges in the ring.

        Parameters
        ----------
        lpf : bool
            Apply low-pass filter; default is True
        downsample : dict
            Dictionary containing float value for 'Ts' sampling time and int value
            for 'points'. Will downsample the RF beam charge onto a coarse time
            grid with 'Ts' sampling time and 'points' points.
        external_reference : bool
            Option to include the changing external reference of the time-grid

        Returns
        -------
        complex array
            RF beam charge gradient array [C] at 'frequency' omega_c, with the sampling time
            of the Profile object. To obtain current, divide by the sampling time
        (complex array)
            If time_coarse is specified, returns also the RF beam charge gradient array [C]
            on the coarse time grid
        """
        # Convert from dimensionless to Coulomb/Ampères
        # Take into account macro-particle charge with real-to-macro-particle ratio
        charges = (self.profile.hist_y_to_density_factor * beam.particle_type.charge * e
                   * np.copy(self.profile.hist_y))
        # logger.debug("Sum of particles: %d, total charge: %.4e C",
        #              np.sum(profile.hist_y), np.sum(charges))
        # logger.debug("DC current is %.4e A", np.sum(charges) / T_rev)

        # Mix with frequency of interest; remember factor 2 demodulation
        charge_gradient = np.gradient(charges, self.profile.hist_x)
        I_f_gradient = 2. * (1j * self.omega_carrier * charges + charge_gradient) * np.cos(
            self.omega_carrier * self.profile.hist_x)
        Q_f_gradient = -2. * (1j * self.omega_carrier * charges + charge_gradient) * np.sin(
            self.omega_carrier * self.profile.hist_x)

        # Pass through a low-pass filter
        if lpf:
            # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
            cutoff = 20.e6 * 2. * self.profile.hist_x
            I_f_gradient = low_pass_filter(I_f_gradient, cutoff_frequency=cutoff)
            Q_f_gradient = low_pass_filter(Q_f_gradient, cutoff_frequency=cutoff)

        gradient_fine = I_f_gradient + 1j * Q_f_gradient
        if external_reference:
            # slippage in phase due to a non-integer harmonic number
            dphi = delta_t * self.omega_carrier
            # Total phase correction
            phase = dphi
            gradient_fine = gradient_fine * np.exp(1j * phase)

        if downsample:
            try:
                T_s = float(downsample['Ts'])
                n_points = int(downsample['points'])
            except Exception as exception:
                raise RuntimeError('Downsampling input erroneous in rf_beam_current') from exception

            # Find which index in fine grid matches index in coarse grid
            ind_fine = np.round((self.profile.hist_x + delta_t - np.pi / self.omega_carrier) / T_s)
            ind_fine = np.array(ind_fine, dtype=int)
            indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]
            if len(indices) == 0:
                indices = [ind_fine[0]]

            # Pick total current within one coarse grid
            gradient_coarse = np.zeros(n_points, dtype=complex)
            gradient_coarse[ind_fine[0]] = np.sum(gradient_fine[np.arange(indices[0])])
            for i in range(1, len(indices)):
                gradient_coarse[i + ind_fine[0]] = np.sum(gradient_fine[np.arange(indices[i - 1],
                                                                                  indices[i])])
            return gradient_fine, gradient_coarse

        else:
            return gradient_fine

    # TODO: remove following section
    @property
    def V_ANT_COARSE(self) -> NumpyArray:
        """Translates to v_antenna_coarse."""
        return self.v_antenna_coarse

    @V_ANT_COARSE.setter
    def V_ANT_COARSE(self, value: NumpyArray) -> None:
        """Translates to v_antenna_coarse."""
        self.v_antenna_coarse = value

    @property
    def V_ANT_FINE(self) -> NumpyArray:
        """Translates to v_antenna_fine."""
        return self.v_antenna_fine

    @V_ANT_FINE.setter
    def V_ANT_FINE(self, value: NumpyArray) -> None:
        """Translates to v_antenna_fine."""
        self.v_antenna_fine = value

    @property
    def I_GEN_COARSE(self) -> NumpyArray:
        """Translates to i_generator_coarse."""
        return self.i_generator_coarse

    @I_GEN_COARSE.setter
    def I_GEN_COARSE(self, value: NumpyArray) -> None:
        """Translates to i_generator_coarse."""
        self.i_generator_coarse = value

    @property
    def I_GEN_FINE(self) -> NumpyArray:
        """Translates to i_generator_fine."""
        return self.i_generator_fine

    @I_GEN_FINE.setter
    def I_GEN_FINE(self, value: NumpyArray) -> None:
        """Translates to i_generator_fine."""
        self.i_generator_fine = value

    @property
    def I_BEAM_COARSE(self) -> NumpyArray:
        """Translates to i_beam_coarse."""
        return self.i_beam_coarse

    @I_BEAM_COARSE.setter
    def I_BEAM_COARSE(self, value: NumpyArray) -> None:
        """Translates to i_beam_coarse."""
        self.i_beam_coarse = value

    @property
    def I_BEAM_FINE(self) -> NumpyArray:
        """Translates to i_beam_fine."""
        return self.i_beam_fine

    @I_BEAM_FINE.setter
    def I_BEAM_FINE(self, value: NumpyArray) -> None:
        """Translates to i_beam_fine."""
        self.i_beam_fine = value

    @property
    def V_corr(self) -> float:
        """Translates to relative_voltage_correction."""
        return self.relative_voltage_correction

    @V_corr.setter
    def V_corr(self, value: float) -> None:
        """Translates to relative_voltage_correction."""
        self.relative_voltage_correction = value

    @property
    def phi_corr(self) -> float:
        """Translates to phase_correction."""
        return self.phase_correction

    @phi_corr.setter
    def phi_corr(self, value: NumpyArray) -> None:
        """Translates to phase_correction."""
        self.phase_correction = value

    @property
    def T_s(self) -> float:
        """Translates to sampling_time."""
        return self.sampling_time

    @T_s.setter
    def T_s(self, value: float) -> None:
        """Translates to sampling_time."""
        self.sampling_time = value
