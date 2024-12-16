import unittest
from copy import deepcopy

import numpy as np
from parameterized import parameterized

from blond.beam.distributions import (
    matched_from_line_density,
    matched_from_distribution_function,
    bigaussian,
    parabolic,
)


def kwargs_matched_from_line_density():
    ret = []
    # expected results of std(dt), std(dE), Blond Version '2.1.12.dev82+gdaeac80' 12.2024
    expected_results = [
        (4.256876849425501e-10, 981205542.0063506),
        (4.704628571877641e-10, 1071157389.6466283),
        (4.936988589783262e-10, 1156913638.6638432),
        (4.256876849425501e-10, 981205542.0063506),
        (4.704628571877641e-10, 1071157389.6466283),
        (4.936988589783262e-10, 1156913638.6638432),
        (1.2470832262653816e-10, 357072300.2793948),
        (1.3324411189179774e-10, 376164332.05472755),
        (1.3221831290318827e-10, 370074759.8443698),
        (1.2470832262653816e-10, 357072300.2793948),
        (1.3324411189179774e-10, 376164332.05472755),
        (1.3221831290318827e-10, 370074759.8443698),
        (9.419801566634442e-11, 287666767.92084986),
        (9.728046617988963e-11, 298150293.22742134),
        (9.578992240391081e-11, 296243308.1748277),
        (9.419801566634442e-11, 287666767.92084986),
        (9.728046617988963e-11, 298150293.22742134),
        (9.578992240391081e-11, 296243308.1748277),
        (1.0891496265794326e-10, 322867323.2436636),
        (1.1214260883679538e-10, 335829018.3641582),
        (1.1045032464274181e-10, 332192047.69626313),
        (1.0891496265794326e-10, 322867323.2436636),
        (1.1214260883679538e-10, 335829018.3641582),
        (1.1045032464274181e-10, 332192047.69626313),
        (9.419801566634442e-11, 287666767.92084986),
        (9.728046617988963e-11, 298150293.22742134),
        (9.578992240391081e-11, 296243308.1748277),
        (9.419801566634442e-11, 287666767.92084986),
        (9.728046617988963e-11, 298150293.22742134),
        (9.578992240391081e-11, 296243308.1748277),
        (1.1742869205335188e-10, 346550654.40782875),
        (1.1594994095543772e-10, 347252307.9297878),
        (1.1828636763771954e-10, 347892504.2661228),
        (1.1742869205335188e-10, 346550654.40782875),
        (1.1594994095543772e-10, 347252307.9297878),
        (1.1828636763771954e-10, 347892504.2661228),
        (8.44894914379677e-11, 277281895.70319086),
        (8.737881916548479e-11, 280837914.4309487),
        (8.806500337778718e-11, 279841694.8624329),
        (8.44894914379677e-11, 277281895.70319086),
        (8.737881916548479e-11, 280837914.4309487),
        (8.806500337778718e-11, 279841694.8624329),
    ]
    i = 0
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
                    other_kwargs = dict(
                        line_density_type=line_density_type,
                        main_harmonic_option=main_harmonic_option,
                        line_density_exponent=line_density_exponent,
                        half_option=half_option,
                    )
                    kwargs.update(other_kwargs)
                    ret.append((deepcopy(kwargs), expected_results[i]))
                    i += 1
    return [ret[0], ret[-1]] # minimum selection of tests to save time

class BaseTest(unittest.TestCase):
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


class TestMatchedFromLineDensity(BaseTest, unittest.TestCase):
    """Test if functions crash when executing, without asserting behaviour"""


    @parameterized.expand(kwargs_matched_from_line_density)
    def test_matched_from_line_density(self, kwargs, expected_result):
        matched_from_line_density(
            beam=self.beam,
            half_option=kwargs.pop('half_option'),
            full_ring_and_rf=self.full_ring_and_rf,
            line_density_type=kwargs.pop('line_density_type'),
            line_density_exponent=kwargs.pop('line_density_exponent'),
            total_induced_voltage=self.total_ind_volt,
            main_harmonic_option=kwargs.pop('main_harmonic_option'),
            seed=8,
            n_iterations=2,
            n_points_potential=int(50),
            n_points_grid=int(50),
            n_points_abel=50,
            **kwargs,
        )
        # print(np.std(self.beam.dt), np.std(self.beam.dE)) # producing expected_results
        self.assertAlmostEqual(np.std(self.beam.dt), expected_result[0])
        self.assertAlmostEqual(np.std(self.beam.dE), expected_result[1])


def kwargs_matched_from_distribution_function():
    ret = []
    # expected results of std(dt), std(dE), Blond Version '2.1.12.dev82+gdaeac80' 12.2024
    expected_results = [
        (5.457266716481516e-10, 1120649647.953963),
        (5.457266716481516e-10, 1120649647.953963),
        (5.457266716481516e-10, 1120649647.953963),
        (5.457266716481516e-10, 1120649647.953963),
        (5.457266716481516e-10, 1120649647.953963),
        (5.457266716481516e-10, 1120649647.953963),
        (5.457266716481516e-10, 1120649647.953963),
        (5.457266716481516e-10, 1120649647.953963),
        (5.457266716481516e-10, 1120649647.953963),
        (5.457266716481516e-10, 1120649647.953963),
        (5.457266716481516e-10, 1120649647.953963),
        (5.457266716481516e-10, 1120649647.953963),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.458937404121279e-10, 1115558754.2854607),
        (4.611072133103596e-10, 1017319856.818047),
        (5.458937404121279e-10, 1115558754.2854607),
        (4.611072133103596e-10, 1017319856.818047),
        (5.458937404121279e-10, 1115558754.2854607),
        (4.611072133103596e-10, 1017319856.818047),
        (5.458937404121279e-10, 1115558754.2854607),
        (4.611072133103596e-10, 1017319856.818047),
        (5.458937404121279e-10, 1115558754.2854607),
        (4.611072133103596e-10, 1017319856.818047),
        (5.458937404121279e-10, 1115558754.2854607),
        (4.611072133103596e-10, 1017319856.818047),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.444810830951599e-10, 1110198590.8303099),
        (3.908399381787592e-10, 930540859.9407694),
        (5.372852980692379e-10, 1100758021.5546958),
        (4.273788412305807e-10, 939116497.5510793),
        (5.372852980692379e-10, 1100758021.5546958),
        (4.273788412305807e-10, 939116497.5510793),
        (5.372852980692379e-10, 1100758021.5546958),
        (4.273788412305807e-10, 939116497.5510793),
        (5.372852980692379e-10, 1100758021.5546958),
        (4.273788412305807e-10, 939116497.5510793),
        (5.372852980692379e-10, 1100758021.5546958),
        (4.273788412305807e-10, 939116497.5510793),
        (5.372852980692379e-10, 1100758021.5546958),
        (4.273788412305807e-10, 939116497.5510793),
        (6.397594517840831e-10, 1276356299.062496),
        (5.380133259286315e-10, 1122852461.13806),
        (6.397594517840831e-10, 1276356299.062496),
        (5.380133259286315e-10, 1122852461.13806),
        (6.397594517840831e-10, 1276356299.062496),
        (5.380133259286315e-10, 1122852461.13806),
        (6.397594517840831e-10, 1276356299.062496),
        (5.380133259286315e-10, 1122852461.13806),
        (6.397594517840831e-10, 1276356299.062496),
        (5.380133259286315e-10, 1122852461.13806),
        (6.397594517840831e-10, 1276356299.062496),
        (5.380133259286315e-10, 1122852461.13806),
        (1.323459076860634e-10, 520928057.490685),
        (1.328511929669344e-10, 539891045.7812585),
        (8.522944440992497e-11, 401920535.8810043),
        (9.026274626102464e-11, 412223842.1917212),
        (1.2880897891865767e-10, 533346275.75610894),
        (1.3034591002157393e-10, 542345277.0720755),
        (1.323459076860634e-10, 520928057.490685),
        (1.328511929669344e-10, 539891045.7812585),
        (8.522944440992497e-11, 401920535.8810043),
        (9.026274626102464e-11, 412223842.1917212),
        (1.2880897891865767e-10, 533346275.75610894),
        (1.3034591002157393e-10, 542345277.0720755),
        (1.0279518785954672e-10, 441539934.3729881),
        (1.0256548352183845e-10, 443091646.98571867),
        (1.0179814707419211e-10, 445490432.16623485),
        (1.0526679949654999e-10, 473890785.5546073),
        (1.3014018504393394e-10, 524802217.7413834),
        (1.2968634179861144e-10, 527247759.54811174),
        (1.0279518785954672e-10, 441539934.3729881),
        (1.0256548352183845e-10, 443091646.98571867),
        (1.0179814707419211e-10, 445490432.16623485),
        (1.0526679949654999e-10, 473890785.5546073),
        (1.3014018504393394e-10, 524802217.7413834),
        (1.2968634179861144e-10, 527247759.54811174),
        (1.1083399447940965e-10, 476740923.4360597),
        (1.1357275185013358e-10, 483992707.3200019),
        (9.585785079550984e-11, 428842816.7618905),
        (9.870415168715773e-11, 450907873.18413055),
        (1.2918605276912977e-10, 528574671.0099394),
        (1.2631618425803003e-10, 517447163.2561287),
        (1.1083399447940965e-10, 476740923.4360597),
        (1.1357275185013358e-10, 483992707.3200019),
        (9.585785079550984e-11, 428842816.7618905),
        (9.870415168715773e-11, 450907873.18413055),
        (1.2918605276912977e-10, 528574671.0099394),
        (1.2631618425803003e-10, 517447163.2561287),
        (1.0279518785954672e-10, 441539934.3729881),
        (1.0256548352183845e-10, 443091646.98571867),
        (1.0179814707419211e-10, 445490432.16623485),
        (1.0526679949654999e-10, 473890785.5546073),
        (1.3014018504393394e-10, 524802217.7413834),
        (1.2968634179861144e-10, 527247759.54811174),
        (1.0279518785954672e-10, 441539934.3729881),
        (1.0256548352183845e-10, 443091646.98571867),
        (1.0179814707419211e-10, 445490432.16623485),
        (1.0526679949654999e-10, 473890785.5546073),
        (1.3014018504393394e-10, 524802217.7413834),
        (1.2968634179861144e-10, 527247759.54811174),
        (1.3560113695129008e-10, 509642438.75252324),
        (1.4237086775456107e-10, 538334764.9426869),
        (1.2939974088313226e-10, 514628825.0901839),
        (1.5222202420794904e-10, 575520504.9357829),
        (1.2939974088313226e-10, 514628825.0901839),
        (1.2926413408073008e-10, 499838597.1548181),
        (1.3560113695129008e-10, 509642438.75252324),
        (1.4237086775456107e-10, 538334764.9426869),
        (1.2939974088313226e-10, 514628825.0901839),
        (1.5222202420794904e-10, 575520504.9357829),
        (1.2939974088313226e-10, 514628825.0901839),
        (1.2926413408073008e-10, 499838597.1548181),
    ]
    i = 0
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
                        other_kwargs = dict(
                            main_harmonic_option=main_harmonic_option,
                            distribution_type=distribution_type,
                            distribution_exponent=distribution_exponent,
                            bunch_length_fit=bunch_length_fit,
                            distribution_variable=distribution_variable,
                        )
                        kwargs.update(other_kwargs)
                        ret.append((deepcopy(kwargs), expected_results[i]
                                    ))
                        i += 1
    return [ret[0], ret[-1]] # minimum selection of tests to save time


class TestFromDistributionFunction(BaseTest, unittest.TestCase):
    """Test if functions crash when executing, without asserting behaviour"""

    @parameterized.expand(kwargs_matched_from_distribution_function)
    def test_matched_from_distribution_function(self, kwargs, expected_result):
        matched_from_distribution_function(
            beam=self.beam,
            main_harmonic_option=kwargs.pop("main_harmonic_option"),
            full_ring_and_rf=self.full_ring_and_rf,
            distribution_type=kwargs.pop("distribution_type"),
            distribution_exponent=kwargs.pop("distribution_exponent"),
            bunch_length_fit=kwargs.pop("bunch_length_fit"),
            total_induced_voltage=self.total_ind_volt,
            distribution_variable=kwargs.pop("distribution_variable"),
            seed=9,
            n_iterations=2,
            n_points_potential=100,
            n_points_grid=100,
            **kwargs,
        )
        print("result", np.std(self.beam.dt), np.std(self.beam.dE))  # producing expected_results
        # self.assertAlmostEqual(np.std(self.beam.dt), expected_result[0])
        # self.assertAlmostEqual(np.std(self.beam.dE), expected_result[1])


class TestOthers(BaseTest, unittest.TestCase):


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
