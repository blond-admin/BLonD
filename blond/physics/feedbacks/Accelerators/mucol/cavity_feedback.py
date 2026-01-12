# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Cavity feedback stubs for the muon collider."""

from functools import cached_property
from typing import Any
from warnings import warn

import numpy as np
from numpy.typing import NDArray as NumpyArray
from scipy.constants import e
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from blond import Simulation, StaticProfile
from blond.core.beam.base import BeamBaseClass
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedback,
)
from blond.physics.feedbacks.helpers import (
    cavity_response_sparse_matrix,
    low_pass_filter,
)

MINIMUM_QL_FEEDBACK_MODEL = 0.5


class PassiveCavity(IQCavityFeedback):
    r"""Passive Cavity, implementing the beam-cavity interaction formulas without a feedback involved.

    Parameters
    ----------
        profile
            profile on which the feedback should act.
        R_over_Q
            shunt impedance over quality factor of one cavity [$$\Omega$$].
        Q_L
            Loaded quality factor of one cavity [1].
        f_center
            center frequency of the cavity [Hz].
        f_detuning
            detuning of the cavity [Hz].
        n_cavities
            number of cavities.
        generator_current
            given in [A].
        generator_phase
            given in [rad].
        injection_phase
            In :func:`xxxx` the cavity will optimise the phase at injection towards
            this value by adjusting the initial parameters at the beginning of the internal tracking given in [rad].
        injection_voltage
            In :func:`xxxx` the cavity will optimise the voltage at injection towards
            this value by adjusting the initial parameters at the beginning of the internal tracking given in [V].
        harmonic_index
            only the default of 0 is allowed.
        n_rf_periods_per_coarse_grid
            number of RF periods, one coarse grid corresponds to.
        use_lowpass_filter
            Used in :func:xxx
        name
            If not given, is automatically chosen.
        fine_RK
            Use Runge Kutta direct calculation for fine grid instead of matrix formalism.
        fine_RK
            Use Runge Kutta direct calculation for coarse grid instead of matrix formalism.
    """

    def __init__(
        self,
        profile: StaticProfile,  # is this stricly necessary?
        R_over_Q: float,
        Q_L: float,
        f_center: float,
        f_detuning: float,
        n_cavities: int,
        generator_current: float,
        n_pretrack: int | None = None,
        initial_v_coarse: NumpyArray | None = None,
        generator_phase: float = 0,
        injection_phase: float = -1,
        injection_voltage: float = -1,
        harmonic_index: int = 0,
        n_rf_periods_per_coarse_grid: int | float = 1,
        use_lowpass_filter: bool = False,
        name: str | None = None,
        fine_RK: bool = False,
        coarse_RK: bool = False,
    ) -> None:
        if harmonic_index != 0:
            raise NotImplementedError(
                "harmonic indices other than 0 are not supported with this module"
            )

        assert R_over_Q >= 0, "R_over_Q must be >= 0"
        self.R_over_Q = R_over_Q

        assert Q_L >= MINIMUM_QL_FEEDBACK_MODEL, "Q_L must be >= 0.5"
        self.Q_L = Q_L

        assert f_center >= 0, (
            "f_center must be >= 0"
        )  # TODO: does this make sense here?
        self.f_center = f_center

        self.f_detuning = f_detuning
        self.omega_detuning = 2 * np.pi * self.f_detuning
        self.omega_center = 2 * np.pi * self.f_center - self.omega_detuning

        assert n_cavities > 0, "n_cavities must be > 0"
        self.n_cavities = n_cavities

        if n_pretrack is not None:
            assert n_pretrack > 0, "n_pretrack must be > 0"
        self.n_pretrack: float | None = n_pretrack

        self._initial_v_coarse = initial_v_coarse

        self.generator_current = generator_current
        self.generator_phase = generator_phase
        self.injection_phase = injection_phase
        self.injection_voltage = injection_voltage

        if use_lowpass_filter:
            warn("lowpass filter is not used in this class", stacklevel=2)

        super().__init__(
            profile=profile,
            n_cavities=n_cavities,
            name=name,
            n_rf_periods_per_coarse_grid=n_rf_periods_per_coarse_grid,
            harmonic_index=harmonic_index,
            use_lowpass_filter=False,
        )

        self.fine_RK = fine_RK
        self.coarse_RK = coarse_RK

        self.relative_detuning: float | None = None

        self.delta_t: float | None = None

        self.beam_current_gradient_coarse_grid: NumpyArray | None = None
        self.beam_current_gradient_fine_grid: NumpyArray | None = None

    def update_feedback_variables(self) -> None:
        """Method to update the variables specific to the turn."""
        self.omega_center = self.omega_rf_actual - self.omega_detuning
        omega_deviation = self.omega_center - self.omega_rf_actual

        self.relative_detuning = omega_deviation / self.omega_center
        # Dimensionless

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
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
            **kwargs,
        )

        self.update_feedback_variables()

        self.generator_current_fine_grid = (
            np.ones_like(self.beam_current_fine_grid, dtype=complex)
            * self.generator_current
            * np.exp(1j * self.generator_phase)
        )
        self.generator_current_coarse_grid = (
            np.ones_like(self.beam_current_coarse_grid, dtype=complex)
            * self.generator_current
            * np.exp(1j * self.generator_phase)
        )

        self.beam_current_gradient_coarse_grid = np.zeros(
            self.n_samples_coarse, dtype=complex
        )
        self.beam_current_gradient_fine_grid = np.zeros(
            self.profile.n_bins, dtype=complex
        )

        if self._initial_v_coarse is None:
            self.track_no_beam(self.n_pretrack)
        else:
            assert len(self._initial_v_coarse) == len(
                self.antenna_voltage_coarse_grid
            ), (
                f"initial v_coarse should have length of antenna_voltage_coarse_grid "
                f"({len(self.antenna_voltage_coarse_grid)}), but was {len(self._initial_v_coarse)}."
            )
            self.antenna_voltage_coarse_grid = self._initial_v_coarse

    def track_no_beam(self, n_pretrack: int | None = None) -> None:
        r"""Tracking method of the cavity feedback without beam in the accelerator.

        Parameters
        ----------
        n_pretrack
            number of turns to pretrack.
        """
        self.update_feedback_variables()
        if n_pretrack is None:
            self.circuit_track(no_beam=True)
        else:
            pretrack_helper = np.zeros(
                self.n_samples_coarse * 3, dtype=np.complex128
            )
            for _i in range(n_pretrack):
                self.circuit_track(no_beam=True)
                if self.injection_voltage != -1:
                    pretrack_helper[0 : self.n_samples_coarse * 2] = (
                        pretrack_helper[
                            self.n_samples_coarse : self.n_samples_coarse * 3
                        ]
                    )
                    pretrack_helper[
                        self.n_samples_coarse : self.n_samples_coarse * 3
                    ] = self.antenna_voltage_coarse_grid
                    print(np.abs(self.antenna_voltage_coarse_grid[-1]))
                    if (
                        np.abs(self.antenna_voltage_coarse_grid[-1])
                        > self.injection_voltage
                    ):
                        inj_ind = np.argmin(
                            np.abs(
                                np.abs(pretrack_helper)
                                - self.injection_voltage
                            )
                        )
                        self.antenna_voltage_coarse_grid = pretrack_helper[
                            inj_ind - self.n_samples_coarse * 2 : inj_ind
                        ]
                        if (
                            len(self.antenna_voltage_coarse_grid)
                            != 2 * self.n_samples_coarse
                        ):
                            raise RuntimeError("too much was cut off")
                        break

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

        simulation
            `Simulation` context manager
        """
        pass

    def circuit_track(self, no_beam: bool = False) -> None:
        """Function to simulate the internal circuit during one turn.

        This function will compute the cavity behaviour on the coarse grid.

        Internally, the function uses the sparse matrix formalism to compute the coarse
        grid, which is only possible since the generator current is a constant value.

        If no beam is present, this function can be called no_beam=False,
        at which point, only the coarse grid will be updated.

        Parameters
        ----------
        no_beam
            if false, both coarse and fine grids will be updated, if true, only the coarse grid.

        """
        # Compute antenna voltage
        time = np.arange(0, self.n_samples_coarse) * self.sampling_time_coarse
        V_init = self.antenna_voltage_coarse_grid[-1]
        dV_init = (
            self.antenna_voltage_coarse_grid[-2]
            - self.antenna_voltage_coarse_grid[-1]
        ) / self.sampling_time_coarse
        v_ant = None
        if self.coarse_RK:
            _, v_ant = self.runge_kutta_tryout_2nd_order(
                V_init=V_init,
                dV_ant_init=dV_init,
                delta_omega=self.omega_detuning,
                omega=self.omega_carrier,
                bin_centers=time,
            )
        else:
            samples_per_rf_coarse = (
                self.omega_rf_actual * self.sampling_time_coarse
            )

            self.antenna_voltage_coarse_grid = cavity_response_sparse_matrix(
                I_beam=self.beam_current_coarse_grid,
                I_gen=self.generator_current_coarse_grid,
                V_ant_init=self.antenna_voltage_coarse_grid[-1],
                samples_per_rf=samples_per_rf_coarse,
                R_over_Q=self.R_over_Q,
                Q_L=self.Q_L,
                relative_detuning=self.relative_detuning,
            )[-self.n_samples_coarse :]  # TODO: is this correct?
        # np.savez("coarse_array_elements_2.npz", I_GEN_COARSE=self.generator_current_coarse_grid, I_BEAM_COARSE=self.I_BEAM_COARSE,
        #          samples_per_rf=self.samples, n_samples=self.n_samples_coarse,
        #          I_gen_init=self.generator_current_coarse_grid[-(1 + self.n_samples_coarse)], V_ant_init=self.antenna_voltage_coarse_grid[-(1 + self.n_samples_coarse)],
        #          V_ANT_COARSE=self.antenna_voltage_coarse_grid)
        if not no_beam:
            # Compute generator current on fine-grid
            self.generator_current_fine_grid = np.interp(
                self.profile.hist_x,
                self.time_coarse_grid,
                self.generator_current_coarse_grid,
            )
            # Compute antenna voltage on the fine-grid
            self.cavity_response_fine()

    def runge_kutta_tryout_2nd_order(
        self,
        V_init,
        bin_centers,
        omega,
        delta_omega,
        method="RK23",
        min_val=True,
        dV_ant_init=0 + 0j,
    ):
        """DOCM."""
        max_tstep = bin_centers[1] - bin_centers[0]

        dcurrent = interp1d(
            bin_centers,
            -0.5 * self.beam_current_gradient_coarse_grid
            + 2j * omega * self.generator_current_coarse_grid,
        )

        r_over_q = self.R_over_Q
        Q_l = self.Q_L
        coeff_A = -2 * (0.5 * delta_omega**2 + delta_omega * omega) / (
            omega * r_over_q
        ) + 1j * omega / (r_over_q * Q_l)
        coeff_dA = (2j * (1 + delta_omega / omega) + 1 / Q_l) / r_over_q

        def fun(t, Y, curr):
            a, dA = Y
            res = (
                omega * r_over_q * (curr(t) - coeff_A * a - coeff_dA * dA)
            )  # TODO: find out, why /2 is required
            return [dA, res]

        A0 = V_init
        dA0 = dV_ant_init
        init_vals = [A0, dA0]
        if min_val:
            sol = solve_ivp(
                fun,
                (bin_centers[0], bin_centers[-1]),
                init_vals,
                t_eval=bin_centers,
                method=method,
                max_step=max_tstep,
                args=[dcurrent],
            )
        else:
            sol = solve_ivp(
                fun,
                (bin_centers[0], bin_centers[-1]),
                init_vals,
                t_eval=bin_centers,
                method=method,
                args=[dcurrent],
            )

        return sol["t"], sol["y"][0]

    def cavity_response_fine(self):
        r"""ACS cavity response model in matrix form on the fine-grid."""
        # TODO: reenable interpolation
        # Find initial value of antenna voltage and generator current
        # t_at_init = self.profile.hist_x[0] - self.profile.hist_step
        # V_A_init = interp1d(
        #     np.concatenate(
        #         (self.time_coarse_grid - self.sampling_time_coarse * self.n_samples_coarse, self.time_coarse_grid)
        #     ),
        #     self.antenna_voltage_coarse_grid,
        # )(t_at_init)
        # dV_A_init = interp1d(
        #     np.concatenate(
        #         (self.time_coarse_grid - self.sampling_time_coarse * self.n_samples_coarse, self.time_coarse_grid)
        #     ),
        #     np.append(np.diff(self.antenna_voltage_coarse_grid), 0) / self.sampling_time_coarse,
        # )(t_at_init)
        V_A_init = self.antenna_voltage_coarse_grid[-1]
        dV_A_init = (
            self.antenna_voltage_coarse_grid[-2]
            - self.antenna_voltage_coarse_grid[-1]
        ) / self.sampling_time_coarse
        # print(dV_A_init)
        # dV_A_init = 0 + 0j
        if self.fine_RK:
            _, self.antenna_voltage_fine_grid = (
                self.runge_kutta_tryout_2nd_order(
                    dV_ant_init=dV_A_init,
                    delta_omega=self.omega_detuning,
                    V_init=V_A_init,
                    bin_centers=self.profile.hist_x,
                    min_val=True,
                    omega=self.omega_center,
                )
            )
        else:
            samples_per_rf_fine_grid = (
                self.omega_rf_actual * self.profile.hist_step
            )

            self.antenna_voltage_fine_grid = cavity_response_sparse_matrix(
                I_beam=self.generator_current_fine_grid,
                I_gen=self.generator_current_fine_grid,
                V_ant_init=V_A_init,
                samples_per_rf=samples_per_rf_fine_grid,
                R_over_Q=self.R_over_Q,
                Q_L=self.Q_L,
                relative_detuning=self.relative_detuning,
            )

        self.antenna_voltage_fine_grid *= self.n_cavities

    def rf_beam_current(
        self,
        beam: BeamBaseClass,
        use_lowpass_filter: bool = False,
    ) -> None:
        r"""Update the RF beam current."""
        if not self.fine_RK and not self.coarse_RK:
            super().calculate_rf_beam_current(
                beam=beam, use_lowpass_filter=use_lowpass_filter
            )
        else:
            (
                self.beam_current_gradient_fine_grid,
                self.beam_current_gradient_coarse_grid,
            ) = self.rf_beam_current_gradient(
                beam=beam,
                lpf=use_lowpass_filter,
                downsample={
                    "Ts": self.sampling_time_coarse,
                    "points": self.n_samples_coarse,
                },
                external_reference=True,
                delta_t=self.delta_t,
            )

            # Convert RF beam current gradients to be in units of Amperes
            self.beam_current_gradient_fine_grid /= self.profile.hist_step

            self.beam_current_gradient_coarse_grid = (
                self.beam_current_gradient_coarse_grid
                / self.sampling_time_coarse
            )

    def rf_beam_current_gradient(
        self,
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
        # TODO: carrier frequency might be missing in heres

        # Convert from dimensionless to Coulomb/Ampères
        # Take into account macro-particle charge with real-to-macro-particle ratio
        charges = (
            self.profile.hist_y_to_density_factor
            * beam.particle_type.charge
            * e
            * np.copy(self.profile.hist_y)
        )
        # logger.debug("Sum of particles: %d, total charge: %.4e C",
        #              np.sum(profile.hist_y), np.sum(charges))
        # logger.debug("DC current is %.4e A", np.sum(charges) / T_rev)

        # Mix with frequency of interest; remember factor 2 demodulation
        charge_gradient = np.gradient(charges, self.profile.hist_x)
        I_f_gradient = (
            2.0
            * (1j * self.omega_carrier * charges + charge_gradient)
            * np.cos(self.omega_carrier * self.profile.hist_x)
        )
        Q_f_gradient = (
            -2.0
            * (1j * self.omega_carrier * charges + charge_gradient)
            * np.sin(self.omega_carrier * self.profile.hist_x)
        )

        # Pass through a low-pass filter
        if lpf:
            # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
            cutoff = 20.0e6 * 2.0 * self.profile.hist_x
            I_f_gradient = low_pass_filter(
                I_f_gradient, cutoff_frequency=cutoff
            )
            Q_f_gradient = low_pass_filter(
                Q_f_gradient, cutoff_frequency=cutoff
            )

        gradient_fine = I_f_gradient + 1j * Q_f_gradient
        if external_reference:
            # slippage in phase due to a non-integer harmonic number
            dphi = delta_t * self.omega_carrier
            # Total phase correction
            phase = dphi
            gradient_fine = gradient_fine * np.exp(1j * phase)

        if downsample:
            try:
                T_s = float(downsample["Ts"])
                n_points = int(downsample["points"])
            except Exception as exception:
                raise RuntimeError(
                    "Downsampling input erroneous in rf_beam_current"
                ) from exception

            # Find which index in fine grid matches index in coarse grid
            ind_fine = np.round(
                (self.profile.hist_x + delta_t - np.pi / self.omega_carrier)
                / T_s
            )
            ind_fine = np.array(ind_fine, dtype=int)
            indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]
            if len(indices) == 0:
                indices = [ind_fine[0]]

            # Pick total current within one coarse grid
            gradient_coarse = np.zeros(n_points, dtype=complex)
            gradient_coarse[ind_fine[0]] = np.sum(
                gradient_fine[np.arange(indices[0])]
            )
            for i in range(1, len(indices)):
                gradient_coarse[i + ind_fine[0]] = np.sum(
                    gradient_fine[np.arange(indices[i - 1], indices[i])]
                )
            return gradient_fine, gradient_coarse

        else:
            return gradient_fine

    cached_props = ("harmonic",)

    @cached_property
    def voltage_setpoint(self) -> NumpyArray:
        """Voltage setpoint on the fine grid [V]."""
        return (
            np.ones_like(self.voltage_setpoint)
            * self.get_voltage_from_parent_rf_station()
        )

    def invalidate_cache(self) -> None:
        """Delete the stored values of functions with @cached_property."""
        self._invalidate_cache(PassiveCavity.cached_props)
