"""
Watts-level energy consistency of the klystron-power normalization.

``generator_power`` (``P = 0.5 (R/Q) Q_L |I_gen|**2``) and the beam-loading
compensation ``I_gen = I_ff + I_beam/2`` are each covered in isolation
elsewhere, but nothing checks that they agree in *true watts*. They must,
because ``current_limit_from_power`` inverts ``generator_power`` to enforce a
user's klystron limit in real watts: an off-by-2 in the power normalization or
in the single-sideband demodulation factor would pass the rest of the suite yet
make every klystron limit twice wrong.

The absolute normalization is pinned here by an independent energy balance:
the incremental klystron forward power the generator must deliver to service
the beam is derived analytically from the phasors (not mirrored from the
implementation) and compared to the power the beam extracts from the cavity.

Notes
-----
**Where the factor-2 enters.**
The coarse cavity envelope is driven by ``(R/Q) omega (I_gen - I_beam/2)``
(see ``_advance_coarse_voltage`` / ``cavity_response_sparse_matrix``, whose
first-order form is ``A (2 I_gen - I_beam)``). ``I_beam`` is the *raw*
``rf_beam_current`` phasor, which carries the single-sideband factor 2
(``charges_fine = 2 * charges * exp(-i omega_c t)``). So the beam's actual
cavity-driving fundamental current, in the *same* units as the generator
current ``I_gen``, is ``I_beam/2``. The generator therefore compensates a
steady beam load by adding exactly ``I_beam/2`` and the power the beam
extracts must be computed from ``I_beam/2``. Computing it from the raw
``I_beam`` instead double-counts and yields half the delivered power -- the
factor-2 shortfall this module guards against.
"""

import unittest

import numpy as np

from blond import StaticProfile
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedbackTimingClass,
)
from blond.physics.feedbacks.cavity_solvers import (
    cavity_response_sparse_matrix,
)
from blond.physics.feedbacks.generator_current_controller import (
    current_limit_from_power,
)

# RCS1-like cavity (matches the sibling feedback tests).
R_OVER_Q = 518.0
Q_L = 1287601.7251526634
F_RF = 1.3e9
T_RF = 1.0 / F_RF
V_SET = 30.0e6 + 0.0j

N_BINS = 256


def build_feedback():
    """
    Build a minimal timing-class feedback to exercise ``generator_power``.

    Only ``R_over_Q`` and ``Q_L`` are load-bearing for ``generator_power`` when
    an explicit current is passed, but the real production object is
    constructed so the actual method (not a re-implementation) is tested.

    Returns
    -------
    IQCavityFeedbackTimingClass
        A feedback instance with the RCS1-like cavity parameters.
    """
    return IQCavityFeedbackTimingClass(
        profile=StaticProfile.from_rad(1.5 * np.pi, 4.5 * np.pi, N_BINS, T_RF),
        R_over_Q=R_OVER_Q,
        Q_L=Q_L,
        generator_current_bias=0.0 + 0.0j,
        n_cavities=1,
        voltage_setpoint=V_SET,
        initial_voltage=float(np.real(V_SET)),
    )


def feedforward_current(voltage_setpoint, relative_detuning):
    r"""
    Matched no-beam generator current for a ``voltage_setpoint`` fixed point.

    The first-order recursion is
    ``V_n = B V_{n-1} + A (2 I_gen - I_beam)`` with ``A = 0.5 (R/Q) s`` and
    ``1 - B = s (0.5/Q_L - i delta)`` (``s = omega dt`` cancels). Its no-beam
    fixed point ``V = (R/Q) I_ff / (0.5/Q_L - i delta)`` inverts to

    .. math::
        I_\mathsf{ff} = V \,(0.5/Q_L - i\,\delta) / (R/Q).

    Parameters
    ----------
    voltage_setpoint
        Target antenna-voltage phasor [V].
    relative_detuning
        Cavity detuning divided by the RF frequency ``delta = Delta f / f``.

    Returns
    -------
    complex
        Feed-forward generator current [A] holding ``voltage_setpoint``.
    """
    return voltage_setpoint * (0.5 / Q_L - 1j * relative_detuning) / R_OVER_Q


def settled_voltage(
    generator_current, beam_current, relative_detuning, n_samples=64
):
    """
    Run the production cavity solver from the setpoint and return the voltage.

    The cavity is initialised exactly at ``V_SET`` with the no-beam
    feed-forward current as the pre-domain sample, then driven by constant
    ``generator_current`` and ``beam_current``. When ``generator_current`` is
    the correct compensation the setpoint is already a fixed point, so the
    returned voltage stays at ``V_SET`` without any transient.

    Parameters
    ----------
    generator_current
        Constant generator current [A] applied on the solution domain.
    beam_current
        Constant raw RF beam current [A] (the ``rf_beam_current`` phasor).
    relative_detuning
        Cavity detuning divided by the RF frequency.
    n_samples
        Number of coarse-grid samples to solve.

    Returns
    -------
    numpy.ndarray
        The antenna-voltage samples returned by the solver.
    """
    i_ff = feedforward_current(V_SET, relative_detuning)
    return cavity_response_sparse_matrix(
        I_beam=np.full(n_samples, beam_current, dtype=complex),
        I_gen=np.full(n_samples, generator_current, dtype=complex),
        V_ant_init=V_SET,
        I_gen_init=i_ff,
        samples_per_rf=2.0 * np.pi,
        R_over_Q=R_OVER_Q,
        Q_L=Q_L,
        relative_detuning=relative_detuning,
    )


# A spread of beam-current phasors: pure beam loading (real), a rotated
# phasor with a reactive part, and a large one to exercise the quadratic term.
BEAM_PHASORS = (
    0.02 + 0.0j,
    0.02 * np.exp(1j * 0.9),
    0.05 - 0.01j,
)
DETUNINGS = (0.0, 2.0e-6, -5.0e-6)


class TestBeamLoadingCompensationSustainsSetpoint(unittest.TestCase):
    """
    ``I_gen = I_ff + I_beam/2`` holds the setpoint in the real cavity solver.

    This pins the production 2:1 generator/beam weighting of the cavity drive
    (``2 I_gen - I_beam``): it is the physical meaning of the single-sideband
    factor-2 (the beam's cavity-driving current is ``I_beam/2``), and every
    watts-level balance below rests on it.
    """

    def test_compensation_is_a_fixed_point_for_all_detunings(self):
        """With the compensation applied, the voltage never leaves V_SET."""
        for delta in DETUNINGS:
            i_ff = feedforward_current(V_SET, delta)
            for i_beam in BEAM_PHASORS:
                with self.subTest(delta=delta, i_beam=i_beam):
                    voltage = settled_voltage(
                        i_ff + i_beam / 2.0, i_beam, delta
                    )
                    np.testing.assert_allclose(
                        voltage, V_SET, rtol=0.0, atol=1e-6
                    )

    def test_forgetting_the_factor_two_breaks_the_fixed_point(self):
        """
        Show raw-``I_beam`` compensation (no ``/2``) drifts off the setpoint.

        This is the in-test mutation proving the ``/2`` is load-bearing: the
        correct compensation leaves the voltage exactly at ``V_SET`` (residual
        at floating noise), whereas the raw over-compensation makes it drift
        measurably -- by many orders of magnitude more.
        """
        delta = 0.0
        i_ff = feedforward_current(V_SET, delta)
        i_beam = 0.02 + 0.0j
        n_samples = 4000

        correct = settled_voltage(
            i_ff + i_beam / 2.0, i_beam, delta, n_samples
        )
        wrong = settled_voltage(i_ff + i_beam, i_beam, delta, n_samples)

        correct_drift = np.max(np.abs(correct - V_SET))
        wrong_drift = np.max(np.abs(wrong - V_SET))

        # The correct compensation holds the setpoint exactly.
        self.assertLess(correct_drift, 1e-6)
        # The raw over-compensation drifts by many volts, dwarfing that.
        self.assertGreater(wrong_drift, 100.0)
        self.assertGreater(wrong_drift, 1e6 * max(correct_drift, 1e-30))


class TestGeneratorPowerBeamPowerBalance(unittest.TestCase):
    """
    The klystron forward-power increment equals the power the beam extracts.

    ``generator_power`` is the real production method; the expected beam power
    is derived analytically from the phasors (``P = 0.5 Re[V conj(I)]`` for
    amplitude phasors) so the test cannot merely mirror the implementation.
    """

    @classmethod
    def setUpClass(cls):
        """Build the feedback whose ``generator_power`` method is tested."""
        cls.cav = build_feedback()

    def _delivered_to_beam(self, i_ff, i_beam):
        r"""
        Incremental klystron forward power that services the beam load.

        The forward power ``P(I) = 0.5 (R/Q) Q_L |I|**2`` of the compensated
        current splits into three parts,

        .. math::
            P(I_\mathsf{ff} + I_\mathsf{beam}/2) = P(I_\mathsf{ff})
                + \underbrace{(R/Q) Q_L\,\mathrm{Re}[I_\mathsf{ff}\,
                  \overline{I_\mathsf{beam}/2}]}_{\text{delivered to beam}}
                + P(I_\mathsf{beam}/2),

        the mutual (cross) term being the power the generator delivers to the
        beam, and ``P(I_beam/2)`` the extra self-dissipation of the
        compensation current. This returns the cross term, formed from the
        production ``generator_power`` by subtracting the two self-powers.

        Parameters
        ----------
        i_ff
            No-beam feed-forward current [A].
        i_beam
            Raw RF beam-current phasor [A].

        Returns
        -------
        float
            Forward power [W] delivered to the beam.
        """
        gp = self.cav.generator_power
        return gp(i_ff + i_beam / 2.0) - gp(i_ff) - gp(i_beam / 2.0)

    def test_balance_closes_to_one_on_resonance(self):
        """
        On resonance the delivered power equals the beam power exactly.

        Beam power uses the *physical* fundamental current ``I_beam/2``:
        ``P_beam = 0.5 Re[V conj(I_beam/2)]``. The ratio is 1 to machine
        precision for any beam phasor.
        """
        i_ff = feedforward_current(V_SET, 0.0)
        for i_beam in BEAM_PHASORS:
            with self.subTest(i_beam=i_beam):
                delivered = self._delivered_to_beam(i_ff, i_beam)
                p_beam = 0.5 * np.real(V_SET * np.conj(i_beam / 2.0))
                self.assertAlmostEqual(delivered / p_beam, 1.0, places=12)

    def test_raw_beam_current_gives_the_factor_two_shortfall(self):
        """
        Computing the beam power from the *raw* ``I_beam`` halves the balance.

        This is the review's prediction and the discriminator: if the
        single-sideband factor-2 were mis-accounted (in the demodulation or in
        this bookkeeping) the balance would read 0.5, not 1. Pinning both the
        correct value (1) and the naive value (0.5) makes the test fail for any
        stray factor-2.
        """
        i_ff = feedforward_current(V_SET, 0.0)
        for i_beam in BEAM_PHASORS:
            with self.subTest(i_beam=i_beam):
                delivered = self._delivered_to_beam(i_ff, i_beam)
                p_beam_naive = 0.5 * np.real(V_SET * np.conj(i_beam))
                self.assertAlmostEqual(
                    delivered / p_beam_naive, 0.5, places=12
                )

    def test_balance_with_detuning_carries_only_the_reactive_term(self):
        r"""
        With detuning the delivered power gains a pure reactive correction.

        Carrying the detuning through the steady state gives

        .. math::
            P_\mathsf{delivered} = P_\mathsf{beam}
                + \delta\,Q_L\,\mathrm{Im}[V\,\overline{I_\mathsf{beam}/2}],

        so the balance still closes exactly on resonance and for any in-phase
        (pure beam-loading) current, and the only departure is the reactive
        power set by the detuning and the beam's quadrature component. The
        real-power normalization is unchanged -- this confirms detuning does
        not smuggle in a spurious factor.
        """
        i_beam = 0.05 - 0.01j
        for delta in (2.0e-6, -5.0e-6):
            with self.subTest(delta=delta):
                i_ff = feedforward_current(V_SET, delta)
                delivered = self._delivered_to_beam(i_ff, i_beam)
                p_beam = 0.5 * np.real(V_SET * np.conj(i_beam / 2.0))
                reactive = delta * Q_L * np.imag(V_SET * np.conj(i_beam / 2.0))
                np.testing.assert_allclose(
                    delivered, p_beam + reactive, rtol=1e-10
                )


class TestPowerCurrentRoundTrip(unittest.TestCase):
    """``current_limit_from_power`` inverts ``generator_power`` in watts."""

    @classmethod
    def setUpClass(cls):
        """Build the feedback whose ``generator_power`` method is tested."""
        cls.cav = build_feedback()

    def test_power_to_current_to_power_is_identity(self):
        """Power to current back to power round-trips to the same watts."""
        for power_limit in (5.0e5, 1.0e6, 4.2e6, 1.25e7):
            with self.subTest(power_limit=power_limit):
                i_max = current_limit_from_power(power_limit, R_OVER_Q, Q_L)
                np.testing.assert_allclose(
                    self.cav.generator_power(i_max),
                    power_limit,
                    rtol=1e-12,
                )

    def test_current_to_power_to_current_is_identity(self):
        """Current to power back to current round-trips to the same amps."""
        for magnitude in (0.01, 0.055, 1.0, 3.1):
            with self.subTest(magnitude=magnitude):
                power = self.cav.generator_power(magnitude + 0.0j)
                np.testing.assert_allclose(
                    current_limit_from_power(power, R_OVER_Q, Q_L),
                    magnitude,
                    rtol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
