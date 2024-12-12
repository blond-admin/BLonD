import unittest

import unittest

import numpy as np

from blond.beam.distributions_multibunch import (
    matched_from_distribution_density_multibunch,
    matched_from_line_density_multibunch,
    match_beam_from_distribution,
    match_beam_from_distribution_multibatch,
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

        self.ring = Ring(
            ring_length=2 * np.pi * 4242.89,
            alpha_0=1 / 55.76 ** 2,
            synchronous_data=7e12,
            particle=Proton(),
            n_turns=int(1e4)
        )

        self.rf_station = RFStation(
            ring=self.ring,
            harmonic=[35640.0],
            voltage=[16e6],
            phi_rf_d=[0],
            n_rf=1,
        )

        self.beam = Beam(
            ring=self.ring,
            n_macroparticles=int(999),
            intensity=int(3e11),
        )

        self.full_ring_and_rf = FullRingAndRF(
            ring_and_rf_section=[RingAndRFTracker(self.rf_station, self.beam)]
        )

        profile = Profile(
            beam=self.beam,
            cut_options=CutOptions(
                cut_left=0,
                cut_right=21 * 2.0 * np.pi / self.rf_station.omega_rf[0, 0],
                n_slices=300
            )
        )

        ind_volt_freq = InducedVoltageFreq(
            self.beam, profile,
            [Resonators(
                R_S=5e8,
                frequency_R=2 * self.rf_station.omega_rf[0, 0] / 2.0 / np.pi,
                Q=10000)],
            frequency_resolution=1.90e+06)

        self.total_ind_volt = TotalInducedVoltage(self.beam, profile, [ind_volt_freq])

    def test_matched_from_distribution_density_multibunch(self):
        self.setUp()
        for emittance, bunch_length in [(66, None), (None, 1e-9)]:
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
                        for distribution_variable in ['Hamiltonian', 'Action']:
                            distribution_options = dict(
                                type=distribution_type,
                                exponent=distribution_exponent,
                                emittance=emittance,
                                bunch_length=bunch_length,
                                bunch_length_fit=bunch_length_fit,
                                density_variable=distribution_variable,
                            )
                            n_bunches = 3
                            distribution_options_list = [distribution_options for _ in range(n_bunches)]
                            print(locals())
                            matched_from_distribution_density_multibunch(
                                beam=self.beam,
                                ring=self.ring,
                                full_ring_and_rf=self.full_ring_and_rf,
                                distribution_options_list=distribution_options_list,
                                n_bunches=n_bunches,
                                bunch_spacing_buckets=10,
                                intensity_list=[1e11, 1e11, 1e11],
                                main_harmonic_option=main_harmonic_option,
                                # total_induced_voltage=self.total_ind_volt,
                                seed=1,
                            )

    def test_matched_from_line_density_multibunch(self):
        self.setUp()
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
                        line_density_options = {
                            'type': line_density_type,
                            'exponent': line_density_exponent,

                        }
                    if line_density_type == 'user_input':
                        line_density_options["time_line_den"] = kwargs['line_density_input']['time_line_den']
                        line_density_options["line_density"] = kwargs['line_density_input']['line_density']
                    else:
                        line_density_options['bunch_length'] = kwargs["bunch_length"]

                    n_bunches = 3
                    line_density_options_list = [line_density_options for _ in range(n_bunches)]
                    matched_from_line_density_multibunch(
                        beam=self.beam,
                        ring=self.ring,
                        full_ring_and_rf=self.full_ring_and_rf,
                        line_density_options_list=line_density_options_list,
                        bunch_spacing_buckets=10,
                        n_bunches=n_bunches,
                        intensity_list=[1e11, 1e11, 1e11],
                        main_harmonic_option=main_harmonic_option,
                        # total_induced_voltage=self.total_ind_volt,
                        half_option=half_option,
                        seed=1,
                    )

    def test_match_beam_from_distribution_multibatch(self):
        self.setUp()
        self.skipTest("Implement performant version to test this module")
        for emittance, bunch_length in [(66, None), (None, 1e-9)]:
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
                        for distribution_variable in ['Hamiltonian', 'Action']:
                            distribution_options = dict(
                                type=distribution_type,
                                exponent=distribution_exponent,
                                emittance=emittance,
                                bunch_length=bunch_length,
                                bunch_length_fit=bunch_length_fit,
                                density_variable=distribution_variable,
                            )

                            match_beam_from_distribution_multibatch(
                                beam=self.beam,
                                ring=self.ring,
                                full_ring_and_rf=self.full_ring_and_rf,
                                distribution_options=distribution_options,
                                n_bunches=3,
                                bunch_spacing_buckets=10,
                                n_batch=3,
                                batch_spacing_buckets=10,
                                main_harmonic_option=main_harmonic_option,
                                #total_induced_voltage=self.total_ind_volt,
                                n_iterations=2,
                                #n_points_potential=None,
                                #dt_margin_percent=None,
                                seed=1,

                            )


if __name__ == '__main__':
    unittest.main()
