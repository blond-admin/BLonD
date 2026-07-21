"""Tests for the separatrix cut of analytic potential wells."""

import numpy as np
import pytest

from blond.experimental.beam_preparation.analytic_action import (
    action_from_potential_well,
)
from blond.experimental.beam_preparation.analytic_hamiltonian import (
    calc_eom_factor_dE,
)
from blond.experimental.beam_preparation.analytic_potential_well import (
    bucket_time_array,
    check_single_bucket_well,
    rf_potential_well,
)
from blond.experimental.beam_preparation.analytic_well_cut import (
    cut_potential_well,
)

# LHC-like reference (450 GeV protons).
OMEGA_RF = 2518229887.224505
VOLTAGE = 6.0e6
T_REV = 8.892465516509709e-05
ETA_0 = 3.172867586042721e-04
BETA = 0.9999978262922387
TOTAL_ENERGY = 450.00104432e9
RF_PERIOD = 2.0 * np.pi / OMEGA_RF

EOM_FACTOR_DE = calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)


def _well(time_array, phi_rf=0.0, eta_0=ETA_0, energy_gain_per_turn=0.0):
    total_voltage = VOLTAGE * np.sin(OMEGA_RF * time_array + phi_rf)
    return rf_potential_well(
        time_array,
        total_voltage,
        charge=1.0,
        t_rev=T_REV,
        eta_0=eta_0,
        energy_gain_per_turn=energy_gain_per_turn,
    )


def _separatrix_action(well_cut):
    deltaE_max = np.sqrt((well_cut.max() - well_cut.min()) / EOM_FACTOR_DE)
    return 4.0 * deltaE_max / (np.pi * OMEGA_RF)


def test_margined_frame_is_cut_to_one_bucket():
    time_array = bucket_time_array(
        OMEGA_RF, n_points=4000, dt_margin_fraction=0.4
    )
    well = _well(time_array)
    # The raw margined well violates the single-bucket contract...
    assert check_single_bucket_well(well, raise_error=False) is False
    time_cut, well_cut = cut_potential_well(time_array, well)
    # ...the cut restores it and spans one RF period.
    assert np.isclose(well_cut.min(), 0.0)
    span = time_cut[-1] - time_cut[0]
    assert np.isclose(span, RF_PERIOD, rtol=2e-3)
    # J at the separatrix matches the closed form on the cut well.
    sorted_h, sorted_j = action_from_potential_well(
        time_cut, well_cut, eom_factor_dE=EOM_FACTOR_DE
    )
    assert np.isclose(sorted_j[-1], _separatrix_action(well_cut), rtol=1e-4)


def test_cut_is_idempotent_on_clean_bucket():
    time_array = bucket_time_array(OMEGA_RF, n_points=2000)
    well = _well(time_array)
    time_cut, well_cut = cut_potential_well(time_array, well)
    # An already-cut single bucket passes through (near-)unchanged.
    assert len(time_cut) >= len(time_array) - 2
    assert np.isclose(well_cut.max(), well.max() - well.min(), rtol=1e-3)


def test_accelerating_well_cut_at_separatrix():
    time_array = bucket_time_array(OMEGA_RF, n_points=20000)
    well = _well(time_array, energy_gain_per_turn=1.0e5)
    # The raw tilted well violates the contract...
    assert check_single_bucket_well(well, raise_error=False) is False
    time_cut, well_cut = cut_potential_well(time_array, well)
    # ...the cut runs between the unstable point and the equal-potential
    # crossing: both edges sit at the separatrix level.
    amplitude = well_cut.max() - well_cut.min()
    assert well_cut[0] >= well_cut.max() - 1e-2 * amplitude
    assert well_cut[-1] >= well_cut.max() - 1e-2 * amplitude
    # The accelerating bucket is smaller than the stationary one.
    stationary_amplitude = VOLTAGE / (np.pi * OMEGA_RF * T_REV / (2.0 * np.pi))
    assert amplitude < stationary_amplitude
    # The previously guarded downstream chain now works end-to-end.
    sorted_h, sorted_j = action_from_potential_well(
        time_cut, well_cut, eom_factor_dE=EOM_FACTOR_DE
    )
    assert np.all(np.diff(sorted_j) >= -1e-12)
    assert np.all(np.isfinite(sorted_j))


def test_below_transition_with_convention_and_margin():
    time_array = bucket_time_array(
        OMEGA_RF, n_points=4000, dt_margin_fraction=0.2
    )
    well = _well(time_array, phi_rf=np.pi, eta_0=-ETA_0)
    time_cut, well_cut = cut_potential_well(time_array, well)
    # Minimum sits inside the cut, not on an edge.
    n = len(well_cut)
    assert 0.25 * n < well_cut.argmin() < 0.75 * n
    sorted_h, sorted_j = action_from_potential_well(
        time_cut, well_cut, eom_factor_dE=EOM_FACTOR_DE
    )
    assert np.isclose(sorted_j[-1], _separatrix_action(well_cut), rtol=1e-4)


def test_below_transition_without_convention_raises():
    time_array = bucket_time_array(OMEGA_RF, n_points=2000)
    well = _well(time_array, phi_rf=0.0, eta_0=-ETA_0)
    with pytest.raises(ValueError, match="phi_rf"):
        cut_potential_well(time_array, well)


def test_multibucket_span_selection():
    time_array = np.linspace(0.0, 3.0 * RF_PERIOD, 6000)
    well = _well(time_array)
    # "deepest" returns a single one-period bucket.
    time_cut, well_cut = cut_potential_well(time_array, well)
    span = time_cut[-1] - time_cut[0]
    assert np.isclose(span, RF_PERIOD, rtol=2e-2)
    # Explicit selection: the first bucket's minimum sits at half an RF
    # period, and the cut does not leak into the second bucket.
    time_cut_0, well_cut_0 = cut_potential_well(
        time_array, well, bucket_index=0
    )
    time_of_minimum = time_cut_0[int(well_cut_0.argmin())]
    assert np.isclose(time_of_minimum, 0.5 * RF_PERIOD, rtol=5e-2)
    assert time_cut_0[-1] <= 1.5 * RF_PERIOD
    # The third physical bucket is addressable and centred at 2.5 T_rf.
    time_cut_2, well_cut_2 = cut_potential_well(
        time_array, well, bucket_index=2
    )
    time_of_minimum_2 = time_cut_2[int(well_cut_2.argmin())]
    assert np.isclose(time_of_minimum_2, 2.5 * RF_PERIOD, rtol=5e-2)
    # Out-of-range selection fails loudly.
    with pytest.raises(ValueError, match="bucket_index"):
        cut_potential_well(time_array, well, bucket_index=99)


def test_duplicate_bucket_detection_is_merged():
    # PotentialWellHelper reports the margined single bucket twice (once
    # per bounding maximum); after deduplication exactly one physical
    # bucket must remain, so bucket_index=1 is out of range.
    time_array = bucket_time_array(
        OMEGA_RF, n_points=4000, dt_margin_fraction=0.4
    )
    well = _well(time_array)
    with pytest.raises(ValueError, match="bucket_index"):
        cut_potential_well(time_array, well, bucket_index=1)


def test_double_harmonic_sub_wells_characterization():
    # Characterizes the CURRENT multi-sub-well behavior (double harmonic
    # v2/v1=0.8 in phase: one outer bucket enclosing two sub-wells).
    # See plan.md: full multi-well support (solfege find_potential_wells
    # semantics) is a later, dedicated step.
    time_array = bucket_time_array(
        OMEGA_RF, n_points=4000, dt_margin_fraction=0.2
    )
    total_voltage = VOLTAGE * (
        np.sin(OMEGA_RF * time_array)
        + 0.8 * np.sin(2.0 * OMEGA_RF * time_array)
    )
    well = rf_potential_well(
        time_array,
        total_voltage,
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    # "deepest" resolves to the outer bucket, which contains the inner
    # separatrix -> loud rejection, no silent wrong physics.
    with pytest.raises(ValueError, match="local maximum"):
        cut_potential_well(time_array, well)
    # Opt-in acceptance (intensity-iteration route): outer bucket
    # returned with a warning, spanning both sub-wells.
    with pytest.warns(UserWarning, match="inner"):
        _, outer_well = cut_potential_well(
            time_array, well, allow_inner_buckets=True
        )
    assert outer_well.max() > 55.0  # eV, outer amplitude ~56.6
    # The two inner sub-wells are individually addressable and equal.
    _, sub_well_1 = cut_potential_well(time_array, well, bucket_index=1)
    _, sub_well_2 = cut_potential_well(time_array, well, bucket_index=2)
    amplitude_1 = sub_well_1.max() - sub_well_1.min()
    amplitude_2 = sub_well_2.max() - sub_well_2.min()
    assert np.isclose(amplitude_1, amplitude_2, rtol=1e-3)
    assert amplitude_1 < 0.1 * VOLTAGE / (
        np.pi * OMEGA_RF * T_REV / (2.0 * np.pi)
    )
    # The single-minimum double-harmonic cases keep working normally.
    steepened = rf_potential_well(
        time_array,
        VOLTAGE
        * (
            np.sin(OMEGA_RF * time_array)
            + 0.5 * np.sin(2.0 * OMEGA_RF * time_array + np.pi)
        ),
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    cut_potential_well(time_array, steepened)
    flattened = rf_potential_well(
        time_array,
        VOLTAGE
        * (
            np.sin(OMEGA_RF * time_array)
            + 0.5 * np.sin(2.0 * OMEGA_RF * time_array)
        ),
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    cut_potential_well(time_array, flattened)


def test_subtract_min_false_keeps_offset():
    time_array = bucket_time_array(
        OMEGA_RF, n_points=2000, dt_margin_fraction=0.4
    )
    well = _well(time_array) + 5.0
    _, well_cut = cut_potential_well(time_array, well, subtract_min=False)
    assert well_cut.min() > 4.0


def test_verbose_and_plot_smoke():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    time_array = bucket_time_array(
        OMEGA_RF, n_points=512, dt_margin_fraction=0.4
    )
    well = _well(time_array)
    cut_potential_well(time_array, well, verbose=True, plot=True)
    plt.close("all")
