import unittest

import numpy as np

from blond.beam.distributions import (
    matched_from_line_density,
    matched_from_distribution_function,
    bigaussian,
    parabolic,
)


class TestExecutable(unittest.TestCase):
    """Test if functions crash when executing, without asserting behaviour"""

    def setUp(self):
        import numpy as np

        from blond.beam.beam import Beam, Proton
        from blond.beam.profile import CutOptions, Profile
        from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
        from blond.impedances.impedance_sources import Resonators
        from blond.input_parameters.rf_parameters import RFStation
        from blond.input_parameters.ring import Ring
        from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker

        self.ring = Ring(2 * np.pi * 4242.89, 1 / 55.76 ** 2,
                         7e12, Proton(), int(1e4))

        self.rf_station = RFStation(self.ring, [35640.0], [16e6],
                                    [0], 1)

        self.beam = Beam(ring=self.ring, n_macroparticles=int(1000), intensity=int(3e11))

        self.full_ring_and_rf = FullRingAndRF([RingAndRFTracker(self.rf_station, self.beam)])

        bucket_length = 2.0 * np.pi / self.rf_station.omega_rf[0, 0]

        profile = Profile(self.beam, CutOptions(cut_left=0,
                                                cut_right=21 * bucket_length,
                                                n_slices=300))

        ind_volt_freq = InducedVoltageFreq(self.beam, profile,
                                           [Resonators(
                                               R_S=5e8,
                                               frequency_R=2 * self.rf_station.omega_rf[0, 0] / 2.0 / np.pi,
                                               Q=10000)],
                                           frequency_resolution=1.90e+06)

        self.total_ind_volt = TotalInducedVoltage(self.beam, profile, [ind_volt_freq])

    def test_matched_from_line_density(self):
        for kwargs in [{"line_density_input": dict(time_line_den=np.linspace(-1, 1, 100) * 1e-9,
                                                   line_density=1 - np.linspace(-1, 1, 100) ** 2,
                                                   )
                        },
                       {"bunch_length": 0.5e-9,
                        }]:
            if "bunch_length" in kwargs.keys():
                line_density_types = ['waterbag', 'parabolic_amplitude',
                                      'parabolic_line', 'binomial',
                                      'gaussian', 'cosine_squared'
                                      ]
            else:
                line_density_types = ['user_input']
            for line_density_type in line_density_types:
                for main_harmonic_option in ['lowest_freq', 'highest_voltage']:
                    if line_density_type == 'binomial':
                        line_density_exponent = 1
                    else:
                        line_density_exponent = None
                    for half_option in ['first', 'second', 'both']:
                        self.setUp()
                        matched_from_line_density(
                            beam=self.beam,
                            half_option=half_option,
                            full_ring_and_rf=self.full_ring_and_rf,
                            line_density_type=line_density_type,
                            line_density_exponent=line_density_exponent,
                            total_induced_voltage=self.total_ind_volt,
                            main_harmonic_option=main_harmonic_option,
                            seed=8,
                            n_iterations=2,
                            n_points_potential=int(50),
                            n_points_grid=int(50),
                            n_points_abel=50,
                            **kwargs,

                        )

    def test_matched_from_distribution_function(self):
        for kwargs in [
            {"emittance": 6 * 11},
            {
                "distribution_user_table": dict(
                    user_table_action=np.linspace(0, 1),
                    user_table_distribution=np.linspace(0, 1)
                )
            },
            {"bunch_length": 0.5e-9,
             },
        ]:
            if "distribution_user_table" in kwargs.keys():
                distribution_types = ['user_input']
            else:
                distribution_types = ['waterbag', 'parabolic_amplitude',
                                      'parabolic_line', 'binomial',
                                      'gaussian']
            for distribution_type in distribution_types:
                for main_harmonic_option in ['lowest_freq', 'highest_voltage']:
                    if distribution_type == 'binomial':
                        distribution_exponent = 1
                    else:
                        distribution_exponent = None
                    for bunch_length_fit in ['full', 'fwhm', None]:
                        for distribution_variable in ['Action', 'Hamiltonian']:
                            self.setUp()
                            matched_from_distribution_function(
                                beam=self.beam,
                                main_harmonic_option=main_harmonic_option,
                                full_ring_and_rf=self.full_ring_and_rf,
                                distribution_type=distribution_type,
                                distribution_exponent=distribution_exponent,
                                bunch_length_fit=bunch_length_fit,
                                total_induced_voltage=self.total_ind_volt,
                                distribution_variable=distribution_variable,
                                seed=9,
                                n_iterations=2,
                                n_points_potential=100,
                                n_points_grid=100,
                                **kwargs,
                            )

    def test_bigaussian(self):
        bigaussian(
            ring=self.ring,
            rf_station=self.rf_station,
            beam=self.beam,
            sigma_dt=1e-9,
            sigma_dE=1e-9,
            seed=1,
            reinsertion=None,
        )

    def test_parabolic(self):
        parabolic(
            ring=self.ring,
            rf_station=self.rf_station,
            beam=self.beam,
            bunch_length=1e-9,
            bunch_position=0,
            bunch_energy=1e12,
            energy_spread=1e-3,
            seed=1,
        )


if __name__ == '__main__':
    unittest.main()
