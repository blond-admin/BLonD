import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    momentum_compaction_factor,
    proton,
)
from blond.physics.impedances.solvers import (
    MultiPassResonatorSolver,
    MultiPoleSparseSolve,
    PeriodicFreqSolver,
    SingleTurnResonatorConvolutionSolver,
    TimeDomainFftSolver,
)
from blond.physics.impedances.sources import Resonators

DEV_PLOT = False


def _under_resolved_low_q_resonator(t_rf, t_rev):
    """Single broadband Q=1 resonator at 20x the RF harmonic (Z/n = 0.7 Ohm).

    Its wake oscillates ~20 times per RF bucket, so on the coarse grids used
    here it is heavily under-resolved -- the regime the bin-integration fix
    targets.

    Parameters
    ----------
    t_rf
        RF period, in [s].
    t_rev
        Revolution period, in [s].

    Returns
    -------
    resonators
        The under-resolved low-Q resonator.
    """
    f_res = 20.0 / t_rf
    r_shunt = 0.7 * 1.0 * f_res * t_rev  # Z/n = 0.7 Ohm, Q = 1
    return Resonators(r_shunt, f_res, 1.0)


def _mixed_resolution_resonators(t_rf, t_rev):
    """One under-resolved low-Q mode plus one well-resolved higher-Q mode.

    Superposition guard: bin-integration must fix the aliased low-Q mode
    without corrupting the mode the grid already resolves.

    Parameters
    ----------
    t_rf
        RF period, in [s].
    t_rev
        Revolution period, in [s].

    Returns
    -------
    resonators
        Two-mode resonator (under-resolved Q=1 + resolved Q=30).
    """
    f_low = 20.0 / t_rf  # under-resolved
    f_high = 2.0 / t_rf  # comfortably resolved
    return Resonators(
        shunt_impedances=np.array([0.7 * f_low * t_rev, 0.3 * f_high * t_rev]),
        center_frequencies=np.array([f_low, f_high]),
        quality_factors=np.array([1.0, 30.0]),
    )


def _low_q_run(
    solver, n_bins, make_resonators=_under_resolved_low_q_resonator
):
    """Run one solver on the fixed under-resolved harness.

    The default resonator sits at 20x the RF harmonic, so its wake oscillates
    several times within a handful of profile bins -- the regime where point-
    sampling the wake aliases badly. The beam is a deterministic Gaussian
    histogram (no Monte-Carlo noise), so the only difference between solvers is
    how each one represents the wake, not the profile they see.

    Parameters
    ----------
    solver
        A time- or frequency-domain wakefield solver instance.
    n_bins
        Number of profile bins; smaller is more under-resolved.
    make_resonators
        Factory ``(t_rf, t_rev) -> Resonators`` building the source(s). The
        default is a single under-resolved low-Q resonator.

    Returns
    -------
    hist_x
        Bin-centre time axis, in [s].
    hist_y
        Deterministic Gaussian histogram (same for every solver).
    induced_voltage
        Induced voltage over the profile, in [V].
    """
    from blond.core.backends.backend import Numpy64Bit, backend

    backend.change_backend(Numpy64Bit)

    circumference = 2 * np.pi * 4242.89
    ring = Ring(circumference=circumference)
    cycle = ConstantMagneticCycle(
        reference_particle=proton, value=6800e9, in_unit="momentum"
    )
    rf = SingleHarmonicRFStation()
    rf.voltage = 6e6
    rf.phi_rf_design = 0
    rf.harmonic = 35640
    drift = DriftSimple(orbit_length=circumference)
    drift.momentum_compaction_factor = momentum_compaction_factor(55.759505)
    t_rev = cycle.get_t_rev_init(circumference=circumference)
    t_rf = float(t_rev / 35640)
    resonators = make_resonators(t_rf, t_rev)

    profile = StaticProfile(cut_left=0, cut_right=2 * t_rf, n_bins=n_bins)
    sigma = 0.83162555241781e-9 / 2
    centre = 2.5e-9
    x = np.asarray(profile.hist_x)
    profile._hist_y = backend.array(np.exp(-0.5 * ((x - centre) / sigma) ** 2))
    # normally set by profile.track; irrelevant here as it is identical for
    # every solver and cancels in the relative comparison
    profile.hist_y_to_density_factor = 1.0
    beam = Beam(intensity=3e11, particle_type=proton)

    wake = WakeField(sources=(resonators,), solver=solver, profile=profile)
    ring.add_elements((profile, rf, drift, wake))
    Simulation(ring=ring, magnetic_cycle=cycle)
    return (
        np.asarray(profile.hist_x),
        np.asarray(profile.hist_y),
        np.asarray(wake.calc_induced_voltage(beam=beam)),
    )


def _low_q_induced_voltage(
    solver, n_bins, make_resonators=_under_resolved_low_q_resonator
):
    """Induced voltage of an under-resolved resonator on a fixed grid.

    Thin wrapper over :func:`_low_q_run` that returns only the induced
    voltage. See :func:`_low_q_run` for the harness details.

    Parameters
    ----------
    solver
        A time- or frequency-domain wakefield solver instance.
    n_bins
        Number of profile bins; smaller is more under-resolved.
    make_resonators
        Factory ``(t_rf, t_rev) -> Resonators`` building the source(s).

    Returns
    -------
    induced_voltage
        Induced voltage over the profile, in [V].
    """
    return _low_q_run(solver, n_bins, make_resonators)[2]


def _plot_low_q(
    make_time_solvers,
    n_bins=256,
    make_resonators=_under_resolved_low_q_resonator,
    title="low-Q resonator",
):
    """Dev-only inspection plot: profile, induced voltages, and residuals.

    Overlays the frequency-domain reference against each time-domain solver on
    the coarse (under-resolved) grid. Gated behind the module-level
    ``DEV_PLOT`` flag in the tests; never runs in CI.

    Parameters
    ----------
    make_time_solvers
        Iterable of zero-argument time-domain solver factories to overlay.
    n_bins
        Profile bins to plot (default 256, the under-resolved grid).
    make_resonators
        Factory ``(t_rf, t_rev) -> Resonators`` for the source(s).
    title
        Figure title / window name.
    """
    hist_x, hist_y, v_freq = _low_q_run(
        PeriodicFreqSolver(warn_above_n_time=None), n_bins, make_resonators
    )
    t_ns = hist_x * 1e9
    _, (ax_profile, ax_voltage, ax_residual) = plt.subplots(
        3, 1, sharex=True, num=title
    )
    ax_profile.plot(t_ns, hist_y, "0.5")
    ax_profile.set_ylabel("profile")
    ax_voltage.plot(t_ns, v_freq, "k", lw=2, label="PeriodicFreqSolver (ref)")
    for make_time_solver in make_time_solvers:
        _, _, v_time = _low_q_run(make_time_solver(), n_bins, make_resonators)
        label = make_time_solver.__name__
        ax_voltage.plot(t_ns, v_time, "--", label=label)
        ax_residual.plot(t_ns, v_time - v_freq, label=label)
    ax_voltage.set_ylabel("induced voltage [V]")
    ax_voltage.legend(fontsize="small")
    ax_residual.set_ylabel("time - freq [V]")
    ax_residual.set_xlabel("time [ns]")
    ax_residual.legend(fontsize="small")
    plt.suptitle(f"{title} (n_bins={n_bins})")
    plt.show()


def _low_q_max_rel_dev(
    make_time_solver, n_bins, make_resonators=_under_resolved_low_q_resonator
):
    """Peak-normalized deviation of a time solver from the freq solver.

    Both solvers see the identical deterministic histogram, so any deviation
    is purely the wake representation. The frequency-domain solver is the
    reference; ``warn_above_n_time=None`` silences its short-profile FFT
    performance warning (irrelevant to correctness here).

    Parameters
    ----------
    make_time_solver
        Zero-argument factory returning a fresh time-domain solver (solvers
        are stateful and consumed per run, so one is built per call).
    n_bins
        Number of profile bins.
    make_resonators
        Factory ``(t_rf, t_rev) -> Resonators`` passed through to
        :func:`_low_q_induced_voltage`.

    Returns
    -------
    max_rel_dev
        ``max|v_time - v_freq| / max|v_freq|``.
    """
    v_freq = _low_q_induced_voltage(
        PeriodicFreqSolver(warn_above_n_time=None), n_bins, make_resonators
    )
    v_time = _low_q_induced_voltage(
        make_time_solver(), n_bins, make_resonators
    )
    return np.max(np.abs(v_time - v_freq)) / np.max(np.abs(v_freq))


class TestResonatorImpedances(unittest.TestCase):
    def setUp(self):
        from blond.core.backends.backend import Numpy64Bit, backend

        backend.change_backend(Numpy64Bit)

    def tearDown(self):
        from blond.core.backends.backend import Numpy64Bit, backend

        backend.change_backend(Numpy64Bit)

    @pytest.mark.backend_mutation
    def test_equal(self):
        voltages = {}
        for i, solver in enumerate(
            (
                PeriodicFreqSolver(
                    t_periodicity=960.0,
                    allow_next_fast_len=False,
                ),
                TimeDomainFftSolver(),
            )
        ):
            ring = Ring(
                circumference=6911.56,
            )
            profile = StaticProfile(
                cut_left=0,
                cut_right=1 * 96,
                n_bins=256 * 96,
            )
            cavity1 = SingleHarmonicRFStation()
            cavity1.voltage = 0
            cavity1.phi_rf_design = 0
            cavity1.harmonic = 1
            drift = DriftSimple(
                orbit_length=ring.circumference,
            )
            drift.momentum_compaction_factor = momentum_compaction_factor(1)
            resonators = Resonators(
                shunt_impedances=100 * np.ones(1),
                center_frequencies=10 * np.ones(1),
                quality_factors=100 * np.ones(1),
            )
            np.random.seed(1)
            distr = np.random.randn(10000, 2)

            beam = Beam(
                intensity=1e10,
                particle_type=proton,
            )
            beam.setup_beam(dt=distr[:, 0] + 5, dE=distr[:, 1])
            profile.track(beam)
            profile._hist_y[3000:] = 0
            if DEV_PLOT:
                plt.figure(0)
                plt.subplot(2, 1, 1)
                plt.plot(
                    profile.hist_x,
                    profile.hist_y,
                    ["-", "--", ":"][i],
                )

            wake = WakeField(
                sources=(resonators,),
                solver=solver,
                profile=profile,
            )
            ring.add_elements((profile, cavity1, drift, wake))
            magnetic_cycle = ConstantMagneticCycle(
                reference_particle=proton,
                value=25.92e9,
                in_unit="momentum",
            )
            sim = Simulation(
                ring=ring,
                magnetic_cycle=magnetic_cycle,
            )
            wake_ = np.fft.irfft(
                resonators.get_impedance_from_wake(
                    profile.hist_x,
                    simulation=sim,
                    beam=beam,
                    n_fft=profile.n_bins,
                )
            )
            induced_voltage = wake.calc_induced_voltage(
                beam=beam,
            )
            if DEV_PLOT:
                plt.figure(0)
                plt.subplot(2, 1, 2)
                plt.plot(
                    profile.hist_x,
                    wake_,
                    ["-", "--", ":"][i],
                )
                plt.figure(1)
                plt.plot(
                    induced_voltage * 1e9,
                    ["-", "--", ":"][i],
                    label=str(type(solver)),
                )
                # plt.plot(np.convolve(profile.hist_y, wake_))
                plt.legend()
            voltages[str(solver)] = induced_voltage
        if DEV_PLOT:
            plt.figure(0)
            plt.subplot(2, 1, 1)
            plt.xlim(0, 96)
            plt.subplot(2, 1, 2)
            plt.xlim(0, 96)
            plt.figure(1)
            plt.xlim(0, 96)
            plt.show()
        for i, solver in enumerate(voltages.keys()):
            if i == 0:
                reference = voltages[solver]  # arbitrary choice
            else:
                np.testing.assert_allclose(
                    reference * 1e9,
                    voltages[solver] * 1e9,
                    atol=0.03,  # because get wake and get impedance use two
                    # different formulas, the results differ more than only
                    # numerical noise.
                    # This is because the frequency domain is cut off
                    # instead of using all frequencies/impedances,
                    # that would clip to the lower frequency region.
                )

    @pytest.mark.backend_mutation
    def test_low_q_resonator_time_matches_freq(self):
        """Under-resolved low-Q resonator: TimeDomainFftSolver matches freq.

        Regression for the InducedVoltageTime vs InducedVoltageFreq
        discrepancy. The resonator sits at 20x the RF harmonic, so its wake
        oscillates several times within a handful of profile bins. Point-
        sampling the wake aliased badly (~230% error) and silently gave a
        wrong induced voltage; bin-integrating the wake makes the two solvers
        agree, and the residual shrinks as the profile grid is refined.
        """
        dev_coarse = _low_q_max_rel_dev(TimeDomainFftSolver, 256)
        dev_fine = _low_q_max_rel_dev(TimeDomainFftSolver, 1024)

        if DEV_PLOT:
            _plot_low_q([TimeDomainFftSolver], title="low-Q: TimeDomainFft")

        # point-sampling gave ~2.3 here; bin-integration keeps it small
        assert dev_coarse < 0.15, dev_coarse
        # and it converges as the grid is refined
        assert dev_fine < dev_coarse
        assert dev_fine < 0.05, dev_fine

    @pytest.mark.backend_mutation
    def test_low_q_resonator_convolution_matches_freq(self):
        """Under-resolved low-Q resonator: convolution solver matches freq.

        Same regression as ``test_low_q_resonator_time_matches_freq`` but for
        the single-turn convolution solver, which consumes the wake through
        ``TimeDomain.get_wake_per_bin`` directly (rather than via the FFT
        ``get_impedance_from_wake`` path). Without bin-integration it point-
        samples the same aliased wake (~2.8 relative deviation here).
        """
        dev_coarse = _low_q_max_rel_dev(
            SingleTurnResonatorConvolutionSolver, 256
        )
        dev_fine = _low_q_max_rel_dev(
            SingleTurnResonatorConvolutionSolver, 1024
        )

        if DEV_PLOT:
            _plot_low_q(
                [SingleTurnResonatorConvolutionSolver],
                title="low-Q: SingleTurnConvolution",
            )

        assert dev_coarse < 0.15, dev_coarse
        assert dev_fine < dev_coarse
        assert dev_fine < 0.05, dev_fine

    @pytest.mark.backend_mutation
    def test_low_q_resonator_pole_residue_matches_freq(self):
        """Under-resolved low-Q resonator: pole-residue solver matches freq.

        Same regression, exercising ``MultiPoleSparseSolve``. This solver does
        not build the wake via ``get_wake_per_bin``; it bin-averages each pole
        analytically (residue scaled by ``sinh(p*dt/2)/(p*dt/2)``) plus a
        causal self-bin correction. Without those corrections it is
        O((p*dt)^2) off the other solvers (~2.8 relative deviation here), so
        this pins the correction against the independent frequency-domain
        reference in the regime where it actually matters.
        """
        dev_coarse = _low_q_max_rel_dev(MultiPoleSparseSolve, 256)
        dev_fine = _low_q_max_rel_dev(MultiPoleSparseSolve, 1024)

        if DEV_PLOT:
            _plot_low_q([MultiPoleSparseSolve], title="low-Q: MultiPoleSparse")

        assert dev_coarse < 0.15, dev_coarse
        assert dev_fine < dev_coarse
        assert dev_fine < 0.05, dev_fine

    @pytest.mark.backend_mutation
    def test_low_q_resonator_multi_pass_matches_freq(self):
        """Under-resolved low-Q resonator: multi-pass solver matches freq.

        Same regression as the other per-solver tests, for
        ``MultiPassResonatorSolver`` (the multi-turn convolution solver). Run
        for a single turn its induced voltage is the self-wake, which must
        match the frequency-domain reference; it consumes the wake through
        ``TimeDomain.get_wake_per_bin``, so without bin-integration it point-
        samples the same aliased wake (~2.8 relative deviation here). This
        pins it against the *independent* freq reference, not just against the
        pole-residue solver (which shares the bin-average and so cannot catch a
        common error).
        """
        dev_coarse = _low_q_max_rel_dev(MultiPassResonatorSolver, 256)
        dev_fine = _low_q_max_rel_dev(MultiPassResonatorSolver, 1024)

        if DEV_PLOT:
            _plot_low_q([MultiPassResonatorSolver], title="low-Q: MultiPass")

        assert dev_coarse < 0.15, dev_coarse
        assert dev_fine < dev_coarse
        assert dev_fine < 0.05, dev_fine

    @pytest.mark.backend_mutation
    def test_mixed_resolution_resonators_match_freq(self):
        """Superposition: a resolved mode plus an under-resolved mode.

        The two-mode source mixes an under-resolved Q=1 resonator (which the
        bin-integration fix must correct) with a comfortably resolved Q=30
        resonator (which it must leave untouched). Every time-domain solver
        must still match the frequency-domain reference and converge as the
        grid is refined, confirming the fix is correct in superposition and
        does not corrupt an already-resolved mode.
        """
        solvers = (
            TimeDomainFftSolver,
            SingleTurnResonatorConvolutionSolver,
            MultiPoleSparseSolve,
            MultiPassResonatorSolver,
        )

        if DEV_PLOT:
            _plot_low_q(
                solvers,
                make_resonators=_mixed_resolution_resonators,
                title="mixed-resolution resonators",
            )

        for make_time_solver in solvers:
            with self.subTest(solver=make_time_solver.__name__):
                dev_coarse = _low_q_max_rel_dev(
                    make_time_solver,
                    256,
                    make_resonators=_mixed_resolution_resonators,
                )
                dev_fine = _low_q_max_rel_dev(
                    make_time_solver,
                    1024,
                    make_resonators=_mixed_resolution_resonators,
                )

                assert dev_coarse < 0.15, dev_coarse
                assert dev_fine < dev_coarse
                assert dev_fine < 0.05, dev_fine
