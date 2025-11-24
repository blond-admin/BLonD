import unittest

from blond.beam.distributions import (
    x0_from_bunch_length,
    distribution_function,
    line_density,
    matched_from_distribution_function,
    matched_from_line_density,
    parabolic,
)


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_x0_from_bunch_length(self):
        # TODO: implement test for `x0_from_bunch_length`
        x0_from_bunch_length(
            bunch_length=None,
            bunch_length_fit=None,
            X_grid=None,
            sorted_X_dE0=None,
            n_points_grid=None,
            time_potential_low_res=None,
            distribution_function_=None,
            distribution_type=None,
            distribution_exponent=None,
            beam=None,
            full_ring_and_RF=None,
        )

    @unittest.skip
    def test_distribution_function(self):
        # TODO: implement test for `distribution_function`
        distribution_function(
            action_array=None, dist_type=None, length=None, exponent=None
        )

    @unittest.skip
    def test_line_density(self):
        # TODO: implement test for `line_density`
        line_density(
            coord_array=None,
            dist_type=None,
            bunch_length=None,
            bunch_position=None,
            exponent=None,
        )

    @unittest.skip
    def test_matched_from_distribution_function(self):
        # TODO: implement test for `matched_from_distribution_function`
        matched_from_distribution_function(
            beam=None,
            full_ring_and_RF=None,
            distribution_function_input=None,
            distribution_user_table=None,
            main_harmonic_option=None,
            TotalInducedVoltage=None,
            n_iterations=None,
            n_points_potential=None,
            n_points_grid=None,
            dt_margin_percent=None,
            extraVoltageDict=None,
            seed=None,
            distribution_exponent=None,
            distribution_type=None,
            emittance=None,
            bunch_length=None,
            bunch_length_fit=None,
            distribution_variable=None,
            process_pot_well=None,
            turn_number=None,
        )

    @unittest.skip
    def test_matched_from_line_density(self):
        # TODO: implement test for `matched_from_line_density`
        matched_from_line_density(
            beam=None,
            full_ring_and_RF=None,
            line_density_input=None,
            main_harmonic_option=None,
            TotalInducedVoltage=None,
            plot=None,
            figdir=None,
            half_option=None,
            extraVoltageDict=None,
            n_iterations=None,
            n_points_potential=None,
            n_points_grid=None,
            dt_margin_percent=None,
            n_points_abel=None,
            bunch_length=None,
            line_density_type=None,
            line_density_exponent=None,
            seed=None,
            process_pot_well=None,
        )

    @unittest.skip
    def test_parabolic(self):
        # TODO: implement test for `parabolic`
        parabolic(
            Ring=None,
            RFStation=None,
            Beam=None,
            bunch_length=None,
            bunch_position=None,
            bunch_energy=None,
            energy_spread=None,
            seed=None,
        )
