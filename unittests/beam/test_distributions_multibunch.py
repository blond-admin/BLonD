import unittest

from blond.beam.distributions_multibunch import (
    compute_H0,
    compute_x_grid,
    match_a_bunch,
    match_beam_from_distribution,
    match_beam_from_distribution_multibatch,
    matched_from_distribution_density_multibunch,
    matched_from_line_density_multibunch,
)


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_compute_H0(self):
        # TODO: implement test for `compute_H0`
        compute_H0(emittance=None, H=None, J=None)

    @unittest.skip
    def test_compute_x_grid(self):
        # TODO: implement test for `compute_x_grid`
        compute_x_grid(
            normalization_DeltaE=None,
            time_array=None,
            potential_well=None,
            distribution_variable=None,
        )

    @unittest.skip
    def test_match_a_bunch(self):
        # TODO: implement test for `match_a_bunch`
        match_a_bunch(
            normalization_DeltaE=None,
            beam=None,
            potential_well_coordinates=None,
            potential_well=None,
            seed=None,
            distribution_options=None,
            full_ring_and_RF=None,
        )

    @unittest.skip
    def test_match_beam_from_distribution(self):
        # TODO: implement test for `match_beam_from_distribution`
        match_beam_from_distribution(
            beam=None,
            FullRingAndRF=None,
            GeneralParameters=None,
            distribution_options=None,
            n_bunches=None,
            bunch_spacing_buckets=None,
            main_harmonic_option=None,
            TotalInducedVoltage=None,
            n_iterations=None,
            n_points_potential=None,
            dt_margin_percent=None,
            seed=None,
        )

    @unittest.skip
    def test_match_beam_from_distribution_multibatch(self):
        # TODO: implement test for `match_beam_from_distribution_multibatch`
        match_beam_from_distribution_multibatch(
            beam=None,
            FullRingAndRF=None,
            GeneralParameters=None,
            distribution_options=None,
            n_bunches=None,
            bunch_spacing_buckets=None,
            n_batch=None,
            batch_spacing_buckets=None,
            main_harmonic_option=None,
            TotalInducedVoltage=None,
            n_iterations=None,
            n_points_potential=None,
            dt_margin_percent=None,
            seed=None,
        )

    @unittest.skip
    def test_matched_from_distribution_density_multibunch(self):
        # TODO: implement test for `matched_from_distribution_density_multibunch`
        matched_from_distribution_density_multibunch(
            beam=None,
            Ring=None,
            FullRingAndRF=None,
            distribution_options_list=None,
            n_bunches=None,
            bunch_spacing_buckets=None,
            intensity_list=None,
            minimum_n_macroparticles=None,
            main_harmonic_option=None,
            TotalInducedVoltage=None,
            n_iterations_input=None,
            plot_option=None,
            seed=None,
        )

    @unittest.skip
    def test_matched_from_line_density_multibunch(self):
        # TODO: implement test for `matched_from_line_density_multibunch`
        matched_from_line_density_multibunch(
            beam=None,
            Ring=None,
            FullRingAndRF=None,
            line_density_options_list=None,
            n_bunches=None,
            bunch_spacing_buckets=None,
            intensity_list=None,
            minimum_n_macroparticles=None,
            main_harmonic_option=None,
            TotalInducedVoltage=None,
            half_option=None,
            plot_option=None,
            seed=None,
        )
