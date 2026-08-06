"""Unit tests for the PI-controlled generator current cavity feedback."""

import os
import unittest

import numpy as np

from blond import StaticProfile
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedbackTimingClass,
)
from blond.physics.feedbacks.cavity_solvers import (
    cavity_response_sparse_matrix,
)
from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentPIController,
)

# RCS1-like cavity parameters (see mucol_cav_fdbk.py)
R_OVER_Q = 518.0
Q_L = 1287601.7251526634
F_RF = 1.3e9
OMEGA_RF = 2.0 * np.pi * F_RF
T_RF = 1.0 / F_RF
V0 = 30.0e6

# No-beam steady-state generator current of the cavity model on resonance:
# V0 = 2 * (R/Q) * Q_L * I_ff, so starting from V_ant = V0 with I_gen = I_ff
# the antenna voltage stays constant until the beam arrives.
I_gen_bias = V0 / (2.0 * R_OVER_Q * Q_L)

N_BINS = 256

# Per-step proportional loop gain g = K_p * (R/Q) * omega * dt = 0.1
# (omega * dt = 2 pi for one rf period per coarse-grid sample), well below
# the delay-limited stability bound ~ 1 / n_delay.
GAIN_P = 0.1 / (R_OVER_Q * 2.0 * np.pi)
# Integral loop ~30x slower than the proportional one.
GAIN_I = GAIN_P / (30.0 * T_RF)
N_DELAY = 5

# RF beam current of the high-intensity bunch (real -> in phase with the
# setpoint, i.e. pure beam loading pulling the voltage down).
I_BEAM = 0.02

# File-wide switch for the diagnostic plots in this module. None disables
# them (the default, for CI); "save" writes a PNG next to this file; "show"
# also opens an interactive window. The plot-producing tests skip when this
# is None, so flipping it here is all that is needed to inspect the figures.
PLOT_DIAGNOSTICS: str | None = None


def build_controller(**kwargs):
    """
    Construct a GeneratorCurrentPIController with RCS1-like defaults.

    Parameters
    ----------
    **kwargs
        Overrides for the controller defaults.

    Returns
    -------
    GeneratorCurrentPIController
        Controller tuned like the unit-test feedback.
    """
    params = {
        "gain_proportional": GAIN_P,
        "gain_integral": GAIN_I,
        "generator_current_bias": I_gen_bias + 0.0j,
        "n_delay": N_DELAY,
    }
    params.update(kwargs)
    return GeneratorCurrentPIController(**params)


# Sentinel so ``controller=None`` explicitly selects the constant-current
# mode, distinct from "use the default PI controller".
_DEFAULT_CONTROLLER = object()


def build_feedback(controller=_DEFAULT_CONTROLLER, **kwargs):
    """
    Construct an IQCavityFeedbackTimingClass with RCS1-like defaults.

    Parameters
    ----------
    controller
        Controller to attach; defaults to :func:`build_controller`. Pass
        None to build a constant-current feedback.
    **kwargs
        Overrides for the constructor defaults.

    Returns
    -------
    IQCavityFeedbackTimingClass
        Feedback instance initialised at the no-beam steady state.
    """
    if controller is _DEFAULT_CONTROLLER:
        controller = build_controller()
    params = {
        "profile": StaticProfile.from_rad(
            1.5 * np.pi, 4.5 * np.pi, N_BINS, T_RF
        ),
        "R_over_Q": R_OVER_Q,
        "Q_L": Q_L,
        "generator_current_bias": I_gen_bias + 0.0j,
        "n_cavities": 1,
        "controller": controller,
        "voltage_setpoint": V0 + 0.0j,
        "initial_voltage": V0,
    }
    params.update(kwargs)
    return IQCavityFeedbackTimingClass(**params)


def run_coarse_transient(
    cav, n_steps, beam_on_index, beam_current, beam_off_index=None
):
    """
    Drive circuit_track() on a hand-built constant-step rf-centers grid.

    The feedback starts in the exact no-beam steady state; at sample
    ``beam_on_index`` a constant RF beam current ``beam_current`` switches
    on (a high-intensity bunch passing the cavity every rf period). With
    ``beam_off_index`` set, the current switches off again at that sample,
    modelling a single bunch passage rather than sustained loading.

    Parameters
    ----------
    cav
        Feedback instance to drive.
    n_steps
        Number of coarse-grid samples to track.
    beam_on_index
        Sample index at which the beam current switches on.
    beam_current
        Complex RF beam current [A] while the beam is on.
    beam_off_index
        Sample index at which the beam current switches off again; if None
        the beam stays on until the end of the run (sustained loading).
    """
    cav._rf_centers = (np.arange(n_steps) + 0.5) * T_RF
    cav._rf_centers_lengths = np.array([n_steps])
    cav._residual_time_last_rf_centers_calculation = 0.0
    cav._last_rf_centers_entry = None
    cav.reset_arrays()

    currents = np.zeros(n_steps, dtype=complex)
    stop = n_steps if beam_off_index is None else beam_off_index
    currents[beam_on_index:stop] = beam_current
    cav.beam_current_forward_coarse_grid = currents
    cav.beam_current_fine_grid = np.zeros(N_BINS, dtype=complex)

    cav.circuit_track(
        omega_input=OMEGA_RF,
        no_beam=False,
        start_index=0,
        end_index=n_steps,
    )


def run_multi_turn_fine_grid(
    cav, n_turns, beam_on_turn, peak_beam_current, sigma_t_frac=0.08
):
    """
    Drive the feedback turn-by-turn, returning per-turn fine-grid voltages.

    Each turn the cavity is tracked over a short coarse grid spanning the
    profile window (one coarse sample per RF period). From ``beam_on_turn``
    onwards a high-intensity Gaussian bunch sits in the central RF period:
    its full shape feeds the fine-grid beam current, and its average over
    that period feeds the matching coarse sample. The PI-controller state
    (delay line, integrator) and the cavity voltage persist across turns,
    so the per-profile (fine-grid) antenna voltage shows the beam-loading
    transient build up and then recover as the loop ramps the generator
    current.

    Parameters
    ----------
    cav
        Feedback instance to drive.
    n_turns
        Number of turns to track.
    beam_on_turn
        Turn index at which the bunch first appears.
    peak_beam_current
        Peak RF beam current [A] of the Gaussian bunch.
    sigma_t_frac
        Bunch RMS length as a fraction of the RF period.

    Returns
    -------
    hist_x
        Profile time base [s] (the fine grid).
    fine_voltage_per_turn
        Complex per-profile antenna voltage, shape ``(n_turns, n_bins)``.
    """
    profile = cav.profile
    n_coarse = 3
    rf_centers = (np.arange(n_coarse) + 0.5) * T_RF
    middle = n_coarse // 2
    t_center = rf_centers[middle]
    sigma_t = sigma_t_frac * T_RF
    bunch_shape = np.exp(-0.5 * ((profile.hist_x - t_center) / sigma_t) ** 2)
    # Coarse RF current of the bunch == its average over the RF period.
    coarse_bunch = peak_beam_current * np.sqrt(2 * np.pi) * sigma_t / T_RF

    fine_voltage_per_turn = []
    for turn in range(n_turns):
        beam_on = turn >= beam_on_turn
        cav._rf_centers = rf_centers.copy()
        cav._rf_centers_lengths = np.array([n_coarse])
        cav._residual_time_last_rf_centers_calculation = 0.0
        cav._last_rf_centers_entry = None
        cav.reset_arrays()

        coarse_beam = np.zeros(n_coarse, dtype=complex)
        fine_beam = np.zeros(N_BINS, dtype=complex)
        if beam_on:
            coarse_beam[middle] = coarse_bunch
            fine_beam = peak_beam_current * bunch_shape
        cav.beam_current_forward_coarse_grid = coarse_beam
        cav._last_val_beam_current = coarse_beam[-1]
        cav.beam_current_fine_grid = fine_beam

        cav.circuit_track(
            omega_input=OMEGA_RF,
            no_beam=False,
            start_index=0,
            end_index=n_coarse,
        )
        fine_voltage_per_turn.append(cav.antenna_voltage_fine_grid.copy())

    return copy_to_cpu(profile.hist_x), np.array(fine_voltage_per_turn)


def _open_diagnostic_plot():
    """
    Return ``matplotlib.pyplot`` configured for ``PLOT_DIAGNOSTICS``.

    Mirrors :func:`support.open_debug_plot` but is driven by this module's
    own :data:`PLOT_DIAGNOSTICS` flag, so the plots are toggled file-wide.

    Returns
    -------
    pyplot
        The ``matplotlib.pyplot`` module, or None when plotting is disabled.
    """
    if not PLOT_DIAGNOSTICS:
        return None
    import matplotlib

    if PLOT_DIAGNOSTICS != "show":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish_plot(fig, filename):
    """
    Save ``fig`` next to this test module; show it when ``PLOT_DIAGNOSTICS``.

    Parameters
    ----------
    fig
        Matplotlib figure to write out.
    filename
        Output PNG file name (saved next to this test module).
    """
    import matplotlib.pyplot as plt

    out = os.path.join(os.path.dirname(__file__), filename)
    fig.savefig(out, dpi=120)
    print(f"\n[diagnostic plot] saved to {out}")
    if PLOT_DIAGNOSTICS == "show":
        plt.show()
    plt.close(fig)


def plot_coarse_power_and_voltage(
    cav, v_ant, i_gen, beam_on, n_delay, title, filename
):
    """
    Plot the coarse-grid generator power and antenna voltage evolution.

    Top panel: antenna-voltage deviation from the setpoint (I/Q
    components, in kV) so the beam-loading dip and its recovery to zero are
    visible. Bottom panel: generator power [kW] on the left axis and, on
    the right axis, the generator and beam current magnitudes [mA] -- so
    the chain "beam current switches on -> generator current rises ->
    generator power rises" can be read off directly. The two vertical
    guides mark the bunch arrival and the loop-delayed reaction.

    Parameters
    ----------
    cav
        Feedback instance (used for the power conversion and the beam
        current ``beam_current_forward_coarse_grid``).
    v_ant
        Complex coarse-grid antenna voltage [V].
    i_gen
        Complex coarse-grid generator current [A].
    beam_on
        Coarse-grid sample at which the beam current switches on.
    n_delay
        Loop delay in coarse-grid samples.
    title
        Figure title.
    filename
        Output PNG file name (saved next to this test module).
    """
    plt = _open_diagnostic_plot()
    if plt is None:
        return
    samples = np.arange(len(v_ant))
    power = cav.generator_power(i_gen)
    i_beam = np.asarray(cav.beam_current_forward_coarse_grid)

    fig, (ax_v, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    fig.suptitle(title)

    deviation = v_ant - V0
    ax_v.plot(
        samples, np.real(deviation) / 1e3, color="C0", label="Re(V_ant - V0)"
    )
    ax_v.plot(
        samples, np.imag(deviation) / 1e3, color="C1", label="Im(V_ant - V0)"
    )
    ax_v.axhline(0.0, color="k", ls=":", label="setpoint V0")
    ax_v.set_ylabel("antenna voltage - V0 [kV]")
    ax_v.legend(loc="best")

    power_line = ax_p.plot(
        samples, power / 1e3, color="C3", label="generator power"
    )[0]
    ax_p.set_ylabel("generator power [kW]", color="C3")
    ax_p.tick_params(axis="y", labelcolor="C3")
    ax_p_i = ax_p.twinx()
    igen_line = ax_p_i.plot(
        samples,
        np.abs(i_gen) * 1e3,
        color="C4",
        lw=0.9,
        ls="--",
        label="|I_gen|",
    )[0]
    ibeam_line = ax_p_i.plot(
        samples,
        np.abs(i_beam) * 1e3,
        color="C2",
        lw=1.1,
        ls=":",
        label="|I_beam|",
    )[0]
    ax_p_i.set_ylabel("current [mA]")
    ax_p.set_xlabel("coarse-grid sample")
    ax_p.legend(
        handles=[power_line, igen_line, ibeam_line], loc="center right"
    )

    for ax in (ax_v, ax_p):
        ax.axvline(beam_on, color="0.5", lw=0.8)
        ax.axvline(beam_on + n_delay, color="0.5", ls="--", lw=0.8)

    fig.tight_layout()
    _finish_plot(fig, filename)


def plot_per_profile_voltage_over_turns(
    hist_x, fine_voltage_per_turn, title, filename
):
    """
    Plot the per-profile (fine-grid) antenna voltage over multiple turns.

    One curve per turn, coloured by turn index. Both panels show the
    deviation of the antenna voltage from the setpoint along the profile
    (top: in-phase component, bottom: magnitude), which isolates the
    beam-loading dip and its turn-by-turn recovery toward zero.

    Parameters
    ----------
    hist_x
        Profile time base [s].
    fine_voltage_per_turn
        Complex per-profile antenna voltage, shape ``(n_turns, n_bins)``.
    title
        Figure title.
    filename
        Output PNG file name (saved next to this test module).
    """
    plt = _open_diagnostic_plot()
    if plt is None:
        return
    n_turns = len(fine_voltage_per_turn)
    t_ns = hist_x * 1e9
    cmap = plt.get_cmap("viridis")

    fig, (ax_re, ax_abs) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    fig.suptitle(title)
    for turn in range(n_turns):
        color = cmap(turn / max(n_turns - 1, 1))
        deviation = fine_voltage_per_turn[turn] - V0
        ax_re.plot(t_ns, np.real(deviation) / 1e3, color=color, lw=0.8)
        ax_abs.plot(t_ns, np.abs(deviation) / 1e3, color=color, lw=0.8)

    ax_re.axhline(0.0, color="k", ls=":", label="setpoint V0")
    ax_re.set_ylabel("Re(V_ant) - V0 over profile [kV]")
    ax_re.legend(loc="best")
    ax_abs.set_ylabel("|V_ant - V0| over profile [kV]")
    ax_abs.set_xlabel("time along profile [ns]")

    scalar_mappable = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(n_turns - 1, 1))
    )
    cbar = fig.colorbar(scalar_mappable, ax=(ax_re, ax_abs))
    cbar.set_label("turn")

    _finish_plot(fig, filename)


def plot_single_pulse_vs_sustained(
    cav_pulse, cav_sustained, beam_on, beam_off, title, filename
):
    """
    Compare a single bunch passage against sustained loading (clamped).

    Both runs use the same power-limited klystron. The single pulse sags
    while the bunch is present and then recovers to the setpoint once it
    has passed (the generator current relaxes back to the constant current); the
    sustained load keeps drifting because the clamped generator can never
    supply enough power. The beam current of each run is overlaid on the
    power panel (right axis), so the power difference can be traced back to
    the beam current switching off (single bunch) or staying on
    (sustained).

    Parameters
    ----------
    cav_pulse
        Feedback instance of the single-pulse run.
    cav_sustained
        Feedback instance of the sustained-loading run.
    beam_on, beam_off
        Coarse-grid samples bounding the bunch passage (guides).
    title
        Figure title.
    filename
        Output PNG file name (saved next to this test module).
    """
    plt = _open_diagnostic_plot()
    if plt is None:
        return
    v_pulse = cav_pulse.antenna_voltage_coarse_grid
    i_pulse = cav_pulse.generator_current_coarse_grid
    ib_pulse = np.asarray(cav_pulse.beam_current_forward_coarse_grid)
    v_sust = cav_sustained.antenna_voltage_coarse_grid
    i_sust = cav_sustained.generator_current_coarse_grid
    ib_sust = np.asarray(cav_sustained.beam_current_forward_coarse_grid)
    samples = np.arange(len(v_pulse))

    fig, (ax_v, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    fig.suptitle(title)

    ax_v.plot(
        samples,
        np.real(v_pulse - V0) / 1e3,
        color="C0",
        label="single bunch (passes)",
    )
    ax_v.plot(
        samples,
        np.real(v_sust - V0) / 1e3,
        color="C3",
        ls="--",
        label="sustained loading",
    )
    ax_v.axhline(0.0, color="k", ls=":", label="setpoint V0")
    ax_v.set_ylabel("Re(V_ant - V0) [kV]")
    ax_v.legend(loc="best")

    p_pulse = ax_p.plot(
        samples,
        cav_pulse.generator_power(i_pulse) / 1e3,
        color="C0",
        label="power (single bunch)",
    )[0]
    p_sust = ax_p.plot(
        samples,
        cav_sustained.generator_power(i_sust) / 1e3,
        color="C3",
        ls="--",
        label="power (sustained)",
    )[0]
    ax_p.set_ylabel("generator power [kW]")
    ax_p.set_xlabel("coarse-grid sample")

    ax_pb = ax_p.twinx()
    b_pulse = ax_pb.plot(
        samples,
        np.abs(ib_pulse) * 1e3,
        color="C0",
        lw=1.0,
        ls=":",
        label="I_beam (single bunch)",
    )[0]
    b_sust = ax_pb.plot(
        samples,
        np.abs(ib_sust) * 1e3,
        color="C3",
        lw=1.0,
        ls=":",
        label="I_beam (sustained)",
    )[0]
    ax_pb.set_ylabel("beam current [mA]")
    ax_p.legend(handles=[p_pulse, p_sust, b_pulse, b_sust], loc="center right")

    for ax in (ax_v, ax_p):
        ax.axvline(beam_on, color="0.5", lw=0.8)
        ax.axvline(beam_off, color="0.5", lw=0.8)

    fig.tight_layout()
    _finish_plot(fig, filename)


class TestGeneratorPower(unittest.TestCase):
    """Tests for the klystron power diagnostic on the feedback."""

    def test_generator_power_formula(self):
        """Check the power formula P = 0.5 * (R/Q) * Q_L * |I|^2."""
        cav = build_feedback()
        current = 0.02 * np.exp(1j * 0.3)
        self.assertAlmostEqual(
            cav.generator_power(current),
            0.5 * R_OVER_Q * Q_L * 0.02**2,
            places=6,
        )


class _RecordingController:
    """
    Stub controller recording its calls and returning a sentinel.

    Parameters
    ----------
    sentinel
        Value returned from every :meth:`update` call.
    """

    def __init__(self, sentinel):
        self.sentinel = sentinel
        self.calls = []

    def update_generator_current(self, error, delta_t):
        """
        Record the call arguments and return the sentinel.

        Parameters
        ----------
        error
            Voltage error passed by the feedback.
        delta_t
            Per-step time passed by the feedback.

        Returns
        -------
        sentinel
            The fixed sentinel value.
        """
        self.calls.append((error, delta_t))
        return self.sentinel

    def limit(self, generator_current):
        """
        Return the current unchanged (no actuator limit).

        Parameters
        ----------
        generator_current
            Generator current to (not) limit.

        Returns
        -------
        generator_current
            The input, unchanged.
        """
        return generator_current


class TestFeedbackControllerDelegation(unittest.TestCase):
    """The feedback forms the error/step and delegates to its controller."""

    def test_update_delegates_error_and_step_to_controller(self):
        """The controller update gets the right error and step."""
        sentinel = 0.077 + 0.011j
        stub = _RecordingController(sentinel)
        cav = build_feedback(controller=stub)

        idx = 3
        v_ant = 29.0e6 + 1.0e6j
        cav.antenna_voltage_coarse_grid = np.zeros(idx + 1, dtype=complex)
        cav.antenna_voltage_coarse_grid[idx] = v_ant
        cav.generator_current_coarse_grid = np.zeros(idx + 1, dtype=complex)
        cav._omega_input_for_pi = OMEGA_RF
        omega_times_T_s = 2 * np.pi  # one rf period

        cav._update_generator_current(
            omega_times_T_s=omega_times_T_s,
            coarse_grid_index_to_update=idx,
        )

        self.assertEqual(len(stub.calls), 1)
        error, delta_t = stub.calls[0]
        # error = setpoint - antenna voltage; setpoint is V0 here.
        self.assertAlmostEqual(error, (V0 + 0.0j) - v_ant, places=6)
        self.assertAlmostEqual(delta_t, omega_times_T_s / OMEGA_RF, places=18)
        # ...and the controller's output is written to the coarse grid.
        self.assertEqual(cav.generator_current_coarse_grid[idx], sentinel)

    def test_no_controller_keeps_constant_current(self):
        """Without a controller the generator current stays at the constant value."""
        cav = build_feedback(controller=None)
        self.assertFalse(cav._controller_active)

        run_coarse_transient(
            cav, n_steps=50, beam_on_index=10, beam_current=0j
        )

        np.testing.assert_allclose(
            cav.generator_current_coarse_grid, I_gen_bias, rtol=1e-12
        )


class TestHighIntensityBunchTransient(unittest.TestCase):
    """
    Transient of a high-intensity bunch passage with the PI loop closed.

    A constant RF beam current switches on mid-turn (a high-intensity
    bunch passing the cavity every rf period). The beam loading pulls
    the antenna voltage down; after the loop delay the PI controller
    raises the generator current (and therefore the klystron power) until
    the voltage vector has returned to its setpoint, while the generator
    current settles at the beam-loading compensation value
    I_ff + I_beam / 2.
    """

    n_steps = 3000
    beam_on = 500

    @classmethod
    def setUpClass(cls):
        """Run the transient once and share the result between tests."""
        cls.cav = build_feedback()
        run_coarse_transient(cls.cav, cls.n_steps, cls.beam_on, I_BEAM + 0.0j)
        cls.v_ant = cls.cav.antenna_voltage_coarse_grid
        cls.i_gen = cls.cav.generator_current_coarse_grid

    def test_steady_state_before_the_bunch(self):
        """Before the bunch arrives, voltage and current are stationary."""
        np.testing.assert_allclose(self.v_ant[: self.beam_on], V0, rtol=1e-9)
        np.testing.assert_allclose(
            self.i_gen[: self.beam_on], I_gen_bias, rtol=1e-9
        )

    def test_generator_current_reacts_only_after_the_loop_delay(self):
        """I_gen stays at the constant current for n_delay samples after beam-on."""
        np.testing.assert_allclose(
            self.i_gen[self.beam_on : self.beam_on + N_DELAY],
            I_gen_bias,
            rtol=1e-9,
        )
        self.assertGreater(
            np.abs(self.i_gen[self.beam_on + N_DELAY] - I_gen_bias),
            1e-4,
        )

    def test_beam_loading_pulls_the_voltage_down_first(self):
        """Within the delay window the antenna voltage sags."""
        dip = V0 - np.real(self.v_ant[self.beam_on + N_DELAY])
        self.assertGreater(dip, 100.0)

    def test_voltage_vector_returns_to_setpoint(self):
        """The integrator restores the voltage vector to its setpoint."""
        np.testing.assert_allclose(self.v_ant[-200:], V0 + 0.0j, rtol=1e-4)
        # The complex deviation (I and Q) is small, not just the magnitude
        self.assertLess(np.max(np.abs(self.v_ant[-200:] - V0)), 1e-4 * V0)

    def test_generator_current_settles_at_compensation_value(self):
        """I_gen ends at the beam-loading compensation I_ff + I_b/2."""
        expected = I_gen_bias + I_BEAM / 2.0
        self.assertLess(np.abs(self.i_gen[-1] - expected), 0.02 * expected)

    def test_generator_power_goes_up(self):
        """The klystron power rises while the voltage is restored."""
        p_before = self.cav.generator_power(self.i_gen[self.beam_on - 1])
        p_after = self.cav.generator_power(self.i_gen[-1])
        self.assertGreater(p_after, 1.5 * p_before)

    def test_plot_coarse_power_and_voltage(self):
        """Diagnostic plot of the coarse power/voltage transient."""
        if not PLOT_DIAGNOSTICS:
            self.skipTest("PLOT_DIAGNOSTICS disabled")
        plot_coarse_power_and_voltage(
            self.cav,
            self.v_ant,
            self.i_gen,
            self.beam_on,
            N_DELAY,
            title="PI feedback: high-intensity bunch transient",
            filename="pi_feedback_coarse_power_voltage.png",
        )


class TestKlystronPowerLimit(unittest.TestCase):
    """With the current clamped below the compensation value, V_ant sags."""

    n_steps = 3000
    beam_on = 500
    # Limit below the required compensation current I_ff + I_beam/2
    i_max = I_gen_bias + I_BEAM / 4.0

    @classmethod
    def setUpClass(cls):
        """Run the limited transient once and share the result."""
        cls.cav = build_feedback(
            controller=build_controller(max_output=cls.i_max)
        )
        run_coarse_transient(cls.cav, cls.n_steps, cls.beam_on, I_BEAM + 0.0j)
        cls.v_ant = cls.cav.antenna_voltage_coarse_grid
        cls.i_gen = cls.cav.generator_current_coarse_grid

    def test_generator_current_never_exceeds_the_limit(self):
        """The clamp keeps |I_gen| at or below the configured maximum."""
        self.assertLessEqual(
            np.max(np.abs(self.i_gen)), self.i_max * (1.0 + 1e-12)
        )

    def test_generator_current_saturates_at_the_limit(self):
        """The controller drives the klystron into saturation."""
        np.testing.assert_allclose(
            np.abs(self.i_gen[-1]), self.i_max, rtol=1e-12
        )

    def test_voltage_does_not_return_to_setpoint(self):
        """With saturated power, the voltage cannot be restored."""
        sag = V0 - np.real(self.v_ant[-1])
        self.assertGreater(sag, 1e4)

    def test_plot_coarse_power_and_voltage(self):
        """Diagnostic plot of the power-limited (saturated) transient."""
        if not PLOT_DIAGNOSTICS:
            self.skipTest("PLOT_DIAGNOSTICS disabled")
        plot_coarse_power_and_voltage(
            self.cav,
            self.v_ant,
            self.i_gen,
            self.beam_on,
            N_DELAY,
            title="PI feedback: klystron power limit (saturation)",
            filename="pi_feedback_coarse_power_voltage_limited.png",
        )


class TestSinglePulseBunchTransient(unittest.TestCase):
    """
    A single bunch passage (beam on then off) with a power-limited klystron.

    Unlike the sustained-loading case in
    :class:`TestKlystronPowerLimit` -- where the clamped generator can
    never supply enough power and the voltage keeps drifting toward a
    sagged steady state -- a bunch that *passes* loads the cavity only
    while it is present. The voltage sags during the passage (the clamp is
    active) but, once the beam is gone, the constant generator current alone
    sustains the setpoint, the generator current relaxes out of the clamp,
    and the voltage vector returns to its previous value.
    """

    n_steps = 8000
    beam_on = 500
    beam_off = 2000
    i_max = I_gen_bias + I_BEAM / 4.0  # below the sustained-loading demand

    @classmethod
    def setUpClass(cls):
        """Run the single-pulse and sustained transients (both clamped)."""
        cls.cav = build_feedback(
            controller=build_controller(max_output=cls.i_max)
        )
        run_coarse_transient(
            cls.cav,
            cls.n_steps,
            cls.beam_on,
            I_BEAM + 0.0j,
            beam_off_index=cls.beam_off,
        )
        cls.v_pulse = cls.cav.antenna_voltage_coarse_grid
        cls.i_pulse = cls.cav.generator_current_coarse_grid

        cls.cav_sustained = build_feedback(
            controller=build_controller(max_output=cls.i_max)
        )
        run_coarse_transient(
            cls.cav_sustained, cls.n_steps, cls.beam_on, I_BEAM + 0.0j
        )
        cls.v_sustained = cls.cav_sustained.antenna_voltage_coarse_grid
        cls.i_sustained = cls.cav_sustained.generator_current_coarse_grid

    def test_clamp_is_active_during_the_passage(self):
        """While the bunch is present the generator current saturates."""
        in_bunch = np.abs(self.i_pulse[self.beam_on : self.beam_off])
        self.assertAlmostEqual(in_bunch.max(), self.i_max, places=9)
        # ...and the voltage cannot be held, so it sags
        sag = V0 - np.real(self.v_pulse[self.beam_off - 1])
        self.assertGreater(sag, 1e4)

    def test_voltage_returns_to_setpoint_after_the_bunch(self):
        """Once the bunch has passed, the voltage vector recovers to V0."""
        self.assertLess(np.abs(self.v_pulse[-1] - V0), 1e3)
        # the generator current has relaxed back out of the clamp to I_ff
        self.assertAlmostEqual(np.abs(self.i_pulse[-1]), I_gen_bias, places=4)

    def test_generator_power_returns_to_baseline(self):
        """The klystron power rises during the bunch and then drops back."""
        p_before = self.cav.generator_power(self.i_pulse[self.beam_on - 1])
        p_peak = self.cav.generator_power(self.i_pulse).max()
        p_after = self.cav.generator_power(self.i_pulse[-1])
        self.assertGreater(p_peak, 1.2 * p_before)
        self.assertAlmostEqual(p_after, p_before, delta=1e-3 * p_before)

    def test_sustained_loading_does_not_recover(self):
        """The contrast case keeps sagging instead of returning to V0."""
        pulse_residual = np.abs(self.v_pulse[-1] - V0)
        sustained_residual = np.abs(self.v_sustained[-1] - V0)
        self.assertGreater(sustained_residual, 5e4)
        self.assertGreater(sustained_residual, 10.0 * pulse_residual)

    def test_plot_single_pulse_vs_sustained(self):
        """Diagnostic plot contrasting a bunch passage with sustained load."""
        if not PLOT_DIAGNOSTICS:
            self.skipTest("PLOT_DIAGNOSTICS disabled")
        plot_single_pulse_vs_sustained(
            self.cav,
            self.cav_sustained,
            self.beam_on,
            self.beam_off,
            title=(
                "PI feedback (klystron-limited): "
                "single bunch passage vs sustained loading"
            ),
            filename="pi_feedback_single_pulse_vs_sustained.png",
        )


GENERATOR_CURRENT_THRESHOLD = 1e-6


class TestLoopDelaySampleSemantics(unittest.TestCase):
    """
    ``n_delay`` counts coarse-grid SAMPLES, not time.

    With a sub-stepped feedback (``n_rf_periods_per_coarse_grid < 1``) the
    coarse sample is shorter than an RF period, so the *physical* loop
    delay is ``n_delay * n_rf_periods_per_coarse_grid * t_rf`` -- it
    shrinks with the sub-step. This test pins the sample-based semantics:
    the generator current first reacts at the same sample offset after
    beam-on for the standard and the sub-stepped grid.
    """

    def _first_reaction_offset(self, n_rf_periods: float) -> int:
        """
        Sample offset (after beam-on) of the first generator-current move.

        Parameters
        ----------
        n_rf_periods
            ``n_rf_periods_per_coarse_grid`` of the feedback.

        Returns
        -------
        int
            Samples between beam-on and the first reaction of the current.
        """
        cav = build_feedback(
            controller=build_controller(n_delay=N_DELAY),
            n_rf_periods_per_coarse_grid=n_rf_periods,
        )
        step = n_rf_periods * T_RF
        n_steps = 40
        beam_on = 10
        cav._rf_centers = (np.arange(n_steps) + 0.5) * step
        cav._rf_centers_lengths = np.array([n_steps])
        cav._residual_time_last_rf_centers_calculation = 0.0
        cav._last_rf_centers_entry = None
        cav.reset_arrays()
        currents = np.zeros(n_steps, dtype=complex)
        currents[beam_on:] = I_BEAM
        cav.beam_current_forward_coarse_grid = currents
        cav.beam_current_fine_grid = np.zeros(N_BINS, dtype=complex)
        cav.circuit_track(
            omega_input=OMEGA_RF,
            no_beam=False,
            start_index=0,
            end_index=n_steps,
        )
        # Threshold well above the float-level integrator creep of the
        # steady state (~1e-12 A) and well below a real reaction (~5e-4 A).
        moved = (
            np.abs(cav.generator_current_coarse_grid - I_gen_bias)
            > GENERATOR_CURRENT_THRESHOLD
        )
        return int(np.argmax(moved)) - beam_on

    def test_delay_is_counted_in_samples_not_time(self):
        """The reaction offset in samples is identical for n = 1 and 0.5."""
        offset_standard = self._first_reaction_offset(1)
        offset_substepped = self._first_reaction_offset(0.5)
        # The core semantics: the offset is a *sample* count, unchanged by
        # the sub-step (the physical delay time halves at n = 0.5). The
        # exact offset anatomy relative to N_DELAY is pinned by
        # TestHighIntensityBunchTransient
        # .test_generator_current_reacts_only_after_the_loop_delay.
        self.assertEqual(offset_substepped, offset_standard)
        self.assertGreaterEqual(offset_standard, N_DELAY - 1)


class TestResponseMatrixClamping(unittest.TestCase):
    """The fine-grid response matrix only sees the limited current."""

    def test_fine_grid_solve_uses_clamped_generator_current(self):
        """Clamp I_gen before the fine-grid response-matrix solve."""
        i_max = 0.03
        cav = build_feedback(controller=build_controller(max_output=i_max))
        profile = cav.profile
        samples_per_rf = OMEGA_RF * profile.hist_step

        # Fine-grid current partly above the limit, with varying phase
        phases = np.linspace(0.0, 1.0, N_BINS)
        i_gen_fine = np.linspace(0.5 * i_max, 2.0 * i_max, N_BINS) * np.exp(
            1j * phases
        )
        cav.beam_current_fine_grid = np.zeros(N_BINS, dtype=complex)
        cav.generator_current_fine_grid = i_gen_fine.copy()

        i_gen_init = 2.0 * i_max + 0.0j
        cav.cavity_response_fine(
            initial_voltage_fine_grid=0.0,
            initial_generator_current_fine_grid=i_gen_init,
            samples_per_rf_fine_grid=samples_per_rf,
            relative_detuning=0.0,
        )

        # The stored fine-grid current was clamped in place
        self.assertLessEqual(
            np.max(np.abs(cav.generator_current_fine_grid)),
            i_max * (1.0 + 1e-12),
        )

        # And the matrix solve used exactly the clamped current
        magnitude = np.abs(i_gen_fine)
        clamped = np.where(
            magnitude > i_max, i_gen_fine * (i_max / magnitude), i_gen_fine
        )
        expected = cavity_response_sparse_matrix(
            I_beam=np.zeros(N_BINS, dtype=complex),
            I_gen=clamped,
            V_ant_init=0.0,
            I_gen_init=i_max + 0.0j,
            samples_per_rf=samples_per_rf,
            R_over_Q=R_OVER_Q,
            Q_L=Q_L,
            relative_detuning=0.0,
        )
        np.testing.assert_allclose(
            cav.antenna_voltage_fine_grid, expected, rtol=1e-12
        )


class TestPerProfileVoltageOverTurns(unittest.TestCase):
    """
    Per-profile (fine-grid) antenna voltage recovering over many turns.

    A high-intensity bunch starts passing the cavity on a fixed turn. The
    beam loading distorts the antenna voltage seen along the profile; over
    the following turns the PI loop ramps the generator current until the
    per-profile voltage has recovered toward the setpoint. Only a few
    coarse samples are updated per turn here (the loop runs roughly at the
    revolution rate), so a short loop delay and a stronger integrator are
    used than in the within-turn coarse transient above.
    """

    n_turns = 70
    beam_on_turn = 6
    peak_beam_current = 0.4  # high-intensity bunch

    @classmethod
    def setUpClass(cls):
        """Run the multi-turn transient once and share the result."""
        cls.cav = build_feedback(
            controller=build_controller(n_delay=1, gain_integral=GAIN_I * 15.0)
        )
        cls.hist_x, cls.fine_v = run_multi_turn_fine_grid(
            cls.cav,
            cls.n_turns,
            cls.beam_on_turn,
            cls.peak_beam_current,
        )
        cls.deviation_per_turn = np.max(np.abs(cls.fine_v - V0), axis=1)

    def test_no_distortion_before_the_bunch(self):
        """Before the bunch, the per-profile voltage sits at the setpoint."""
        np.testing.assert_allclose(
            self.fine_v[self.beam_on_turn - 1], V0 + 0.0j, rtol=1e-6
        )

    def test_beam_loading_distorts_the_profile_voltage(self):
        """The bunch produces a clear transient distortion of the voltage."""
        self.assertGreater(self.deviation_per_turn.max(), 100.0)

    def test_distortion_peaks_right_after_the_bunch_arrives(self):
        """The largest distortion is the beam-arrival transient, not drift."""
        self.assertLessEqual(
            int(self.deviation_per_turn.argmax()), self.beam_on_turn + 3
        )

    def test_per_profile_voltage_recovers_toward_setpoint(self):
        """The final-turn distortion is well below the transient peak."""
        self.assertLess(
            self.deviation_per_turn[-1],
            0.5 * self.deviation_per_turn.max(),
        )

    def test_plot_per_profile_voltage_over_turns(self):
        """Diagnostic plot of the per-profile voltage over turns."""
        if not PLOT_DIAGNOSTICS:
            self.skipTest("PLOT_DIAGNOSTICS disabled")
        plot_per_profile_voltage_over_turns(
            self.hist_x,
            self.fine_v,
            title="PI feedback: per-profile voltage over turns",
            filename="pi_feedback_per_profile_voltage_over_turns.png",
        )


if __name__ == "__main__":
    unittest.main()
