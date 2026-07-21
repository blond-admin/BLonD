"""Tests for the AnalyticDistributionMatcher."""

import numpy as np
import pytest

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)
from blond.experimental.beam_preparation.analytic_matcher import (
    AnalyticDistributionMatcher,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu

RF_PERIOD = 2.0 * np.pi / 2518229887.224505


def _build_simulation():
    ring = Ring(26658.883)
    rf_station = SingleHarmonicRFStation(harmonic=35640, voltage=6e6, phi_rf=0)
    drift = DriftSimple(
        orbit_length=26658.883,
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=55.759505
        ),
    )
    ring.add_elements([rf_station, drift])
    magnetic_cycle = ConstantMagneticCycle(
        value=450e9, reference_particle=proton
    )
    beam = Beam(intensity=1e11, particle_type=proton)
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
