"""Tests for the AnalyticDistributionMatcher."""

import numpy as np
import pytest

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    momentum_compaction_factor,
    proton,
)
from blond.experimental.beam_preparation.analytic_matcher import (
    AnalyticDistributionMatcher,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.impedances.solvers import TimeDomainFftSolver

RF_PERIOD = 2.0 * np.pi / 2518229887.224505


def _build_simulation(resonator_r_shunt=None, intensity=1e11):
    ring = Ring(26658.883)
    rf_station = SingleHarmonicRFStation(harmonic=35640, voltage=6e6, phi_rf=0)
    drift = DriftSimple(
        orbit_length=26658.883,
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=55.759505
        ),
    )
    elements = [rf_station, drift]
    if resonator_r_shunt is not None:
        profile = StaticProfile(cut_left=0.0, cut_right=RF_PERIOD, n_bins=512)
        wakefield = WakeField(
            sources=(Resonators(resonator_r_shunt, 8e8, 1.0),),
            solver=TimeDomainFftSolver(),
            profile=profile,
        )
        elements += [wakefield, profile]
    ring.add_elements(elements, reorder=True)
    magnetic_cycle = ConstantMagneticCycle(
        value=450e9, reference_particle=proton
    )
    beam = Beam(intensity=intensity, particle_type=proton)
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    return simulation, beam


def test_matched_bunch_length_and_position():
    simulation, beam = _build_simulation()
    target = 1.2e-9  # 4-sigma rms
    matcher = AnalyticDistributionMatcher(
        n_macroparticles=20_000,
        distribution_type="parabolic_amplitude",
        bunch_length=target,
        seed=0,
        n_points_grid=500,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)

    dt = copy_to_cpu(beam.read_partial_dt())
    dE = copy_to_cpu(beam.read_partial_dE())
    assert len(dt) == 20_000
    # Matched density bunch length equals the target within the grid.
    assert np.isclose(matcher.matched_bunch_length, target, rtol=1e-2)
    # Sampled bunch length within statistics (20k particles ~ 1%).
    assert np.isclose(4.0 * np.std(dt), target, rtol=3e-2)
    # Centred on the stable phase (half an RF period for phi_rf=0
    # above transition).
    assert np.isclose(np.mean(dt), RF_PERIOD / 2.0, atol=0.02e-9)
    # All particles inside the bucket frame, energies inside the
    # separatrix half height (~390 MeV).
    assert dt.min() > 0.0 and dt.max() < RF_PERIOD
    assert np.max(np.abs(dE)) < 4.0e8


def test_emittance_target():
    simulation, beam = _build_simulation()
    matcher = AnalyticDistributionMatcher(
        n_macroparticles=5_000,
        distribution_type="gaussian",
        emittance=0.7,  # eV.s, inside the 1.24 eV.s bucket
        seed=0,
        n_points_grid=400,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)
    assert matcher.fitted_x_0 is not None
    assert 0.0 < matcher.fitted_x_0 < 53.6
    assert matcher.matched_bunch_length > 0.0


def test_seed_reproducibility():
    dts = []
    for _ in range(2):
        simulation, beam = _build_simulation()
        matcher = AnalyticDistributionMatcher(
            n_macroparticles=2_000,
            distribution_type="gaussian",
            bunch_length=1.0e-9,
            seed=42,
            n_points_grid=300,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        dts.append(copy_to_cpu(beam.read_partial_dt()).copy())
    np.testing.assert_array_equal(dts[0], dts[1])


def test_plot_smoke():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    simulation, beam = _build_simulation()
    matcher = AnalyticDistributionMatcher(
        n_macroparticles=1_000,
        distribution_type="gaussian",
        bunch_length=1.0e-9,
        seed=0,
        n_points_grid=200,
        plot=True,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)
    plt.close("all")


def test_target_validation():
    with pytest.raises(ValueError, match="exactly one"):
        AnalyticDistributionMatcher(
            n_macroparticles=1000,
            distribution_type="gaussian",
        )
    with pytest.raises(ValueError, match="exactly one"):
        AnalyticDistributionMatcher(
            n_macroparticles=1000,
            distribution_type="gaussian",
            bunch_length=1e-9,
            emittance=0.5,
        )


def test_matched_emittance_round_trip():
    # The bunch-length target reports the emittance of the matched
    # contour; targeting that emittance must recover the bunch length.
    simulation, beam = _build_simulation()
    matcher_length = AnalyticDistributionMatcher(
        n_macroparticles=2_000,
        distribution_type="parabolic_amplitude",
        bunch_length=1.2e-9,
        seed=0,
        n_points_grid=400,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher_length)
    assert matcher_length.matched_emittance is not None
    assert 0.0 < matcher_length.matched_emittance < 1.24  # bucket area

    simulation, beam = _build_simulation()
    matcher_emittance = AnalyticDistributionMatcher(
        n_macroparticles=2_000,
        distribution_type="parabolic_amplitude",
        emittance=matcher_length.matched_emittance,
        seed=0,
        n_points_grid=400,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher_emittance)
    assert np.isclose(
        matcher_emittance.matched_bunch_length, 1.2e-9, rtol=1e-2
    )
    assert np.isclose(
        matcher_emittance.matched_emittance,
        matcher_length.matched_emittance,
        rtol=1e-3,
    )


def _intensity_matcher(relaxation_factor=1.0, maxiter=100, target=1.2e-9):
    return AnalyticDistributionMatcher(
        n_macroparticles=2_000,
        distribution_type="parabolic_amplitude",
        bunch_length=target,
        seed=0,
        n_points_grid=300,
        maxiter_intensity_effects=maxiter,
        relaxation_factor=relaxation_factor,
        allow_inner_buckets=True,
    )


def test_intensity_effects_converge():
    simulation, beam = _build_simulation(resonator_r_shunt=1e4, intensity=2e11)
    matcher = _intensity_matcher()
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)
    assert 1 <= matcher.n_intensity_iterations <= 20
    assert matcher.final_potential_well_error < 1e-6
    assert np.isclose(matcher.matched_bunch_length, 1.2e-9, rtol=1e-2)
    assert len(matcher.intensity_residuals) == (matcher.n_intensity_iterations)
    # The contour emittance is evaluated in the distorted well.
    assert matcher.matched_emittance is not None
    assert 0.0 < matcher.matched_emittance < 1.24


def test_weak_intensity_matches_zero_intensity_limit():
    simulation, beam = _build_simulation(resonator_r_shunt=1.0, intensity=2e11)
    matcher_weak = _intensity_matcher()
    simulation.prepare_beam(beam=beam, preparation_routine=matcher_weak)
    simulation_0, beam_0 = _build_simulation()
    matcher_0 = _intensity_matcher()
    simulation_0.prepare_beam(beam=beam_0, preparation_routine=matcher_0)
    # A vanishing impedance must reproduce the no-wakefield result.
    # (Frames differ: the wakefield branch adds the legacy 40 % margin.)
    assert np.isclose(matcher_weak.fitted_x_0, matcher_0.fitted_x_0, rtol=1e-3)
    assert np.isclose(
        matcher_weak.matched_bunch_length,
        matcher_0.matched_bunch_length,
        rtol=1e-3,
    )


def test_relaxation_reaches_same_fixed_point():
    results = {}
    for relaxation_factor in (1.0, 0.5):
        simulation, beam = _build_simulation(
            resonator_r_shunt=1e5, intensity=2e11
        )
        matcher = _intensity_matcher(
            relaxation_factor=relaxation_factor, maxiter=200
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        assert matcher.final_potential_well_error < 1e-6
        results[relaxation_factor] = matcher
    # Different relaxations converge to the same self-consistent match.
    assert np.isclose(
        results[1.0].fitted_x_0, results[0.5].fitted_x_0, rtol=1e-3
    )
    assert np.isclose(
        results[1.0].matched_bunch_length,
        results[0.5].matched_bunch_length,
        rtol=1e-3,
    )
    # Under-relaxation takes more iterations.
    assert (
        results[0.5].n_intensity_iterations
        > results[1.0].n_intensity_iterations
    )


def test_relaxation_stabilizes_strong_intensity():
    # At this impedance the full-correction (BLonD 2) iteration
    # oscillates without converging (residual plateaus ~1e-3);
    # under-relaxation converges below 1e-6 in ~50 iterations.
    r_shunt, intensity = 2.5e5, 2e11
    simulation, beam = _build_simulation(
        resonator_r_shunt=r_shunt, intensity=intensity
    )
    matcher_full = _intensity_matcher(relaxation_factor=1.0, maxiter=60)
    with pytest.warns(UserWarning, match="did not converge"):
        simulation.prepare_beam(beam=beam, preparation_routine=matcher_full)
    assert matcher_full.final_potential_well_error > 1e-4

    simulation, beam = _build_simulation(
        resonator_r_shunt=r_shunt, intensity=intensity
    )
    matcher_relaxed = _intensity_matcher(relaxation_factor=0.5, maxiter=60)
    simulation.prepare_beam(beam=beam, preparation_routine=matcher_relaxed)
    assert matcher_relaxed.final_potential_well_error < 1e-6


def test_intensity_matched_bunch_is_stationary():
    simulation, beam = _build_simulation(resonator_r_shunt=1e4, intensity=2e11)
    matcher = _intensity_matcher()
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)
    dt = copy_to_cpu(beam.read_partial_dt())
    initial_length = 4.0 * float(np.std(dt))
    initial_position = float(np.mean(dt))
    simulation.run_simulation(
        beams=(beam,), n_turns=30, show_progressbar=False
    )
    final_dt = copy_to_cpu(beam.read_partial_dt())
    assert (
        abs(4.0 * float(np.std(final_dt)) - initial_length) / initial_length
        < 0.05
    )
    assert abs(float(np.mean(final_dt)) - initial_position) < 0.05e-9


def test_relaxation_factor_validation():
    for bad_value in (0.0, 1.5, -0.3):
        with pytest.raises(ValueError, match="relaxation_factor"):
            AnalyticDistributionMatcher(
                n_macroparticles=1000,
                distribution_type="gaussian",
                bunch_length=1e-9,
                relaxation_factor=bad_value,
            )


def test_matched_bunch_is_stationary_over_turns():
    simulation, beam = _build_simulation()
    target = 1.2e-9
    matcher = AnalyticDistributionMatcher(
        n_macroparticles=2_000,
        distribution_type="parabolic_amplitude",
        bunch_length=target,
        seed=1,
        n_points_grid=400,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)
    initial_length = 4.0 * float(np.std(copy_to_cpu(beam.read_partial_dt())))
    initial_position = float(np.mean(copy_to_cpu(beam.read_partial_dt())))
    simulation.run_simulation(
        beams=(beam,), n_turns=30, show_progressbar=False
    )
    final_dt = copy_to_cpu(beam.read_partial_dt())
    final_length = 4.0 * float(np.std(final_dt))
    final_position = float(np.mean(final_dt))
    # A matched bunch neither blows up nor drifts over 30 turns.
    assert abs(final_length - initial_length) / initial_length < 0.05
    assert abs(final_position - initial_position) < 0.05e-9
