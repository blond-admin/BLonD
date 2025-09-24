import unittest

from blond.input_parameters.ring_options import load_data, convert_data, RingOptions


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_convert_data(self):
        # TODO: implement test for `convert_data`
        convert_data(
            synchronous_data=None,
            mass=None,
            charge=None,
            synchronous_data_type=None,
            bending_radius=None,
        )

    @unittest.skip
    def test_load_data(self):
        # TODO: implement test for `load_data`
        load_data(filename=None, ignore=None, delimiter=None)


class TestRingOptions(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.ring_options = RingOptions(
            interpolation=None,
            smoothing=None,
            flat_bottom=None,
            flat_top=None,
            t_start=None,
            t_end=None,
            plot=None,
            figdir=None,
            figname=None,
            sampling=None,
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test__akima_interpolation(self):
        # TODO: implement test for `_akima_interpolation`
        self.ring_options._akima_interpolation(
            time_interp=None,
            momentum_interp=None,
            beta_interp=None,
            circumference=None,
            time=None,
            momentum=None,
            mass=None,
        )

    @unittest.skip
    def test__cubic_interpolation(self):
        # TODO: implement test for `_cubic_interpolation`
        self.ring_options._cubic_interpolation(
            time_interp=None,
            momentum_interp=None,
            beta_interp=None,
            circumference=None,
            time=None,
            momentum=None,
            mass=None,
            time_start_ramp=None,
            time_end_ramp=None,
        )

    @unittest.skip
    def test__derivative_interpolation(self):
        # TODO: implement test for `_derivative_interpolation`
        self.ring_options._derivative_interpolation(
            time_interp=None,
            momentum_interp=None,
            beta_interp=None,
            circumference=None,
            time=None,
            momentum=None,
            mass=None,
        )

    @unittest.skip
    def test_preprocess(self):
        # TODO: implement test for `preprocess`
        self.ring_options.preprocess(
            mass=None, circumference=None, time=None, momentum=None
        )

    @unittest.skip
    def test_reshape_data(self):
        # TODO: implement test for `reshape_data`
        self.ring_options.reshape_data(
            input_data=None,
            n_turns=None,
            n_sections=None,
            interp_time=None,
            input_to_momentum=None,
            synchronous_data_type=None,
            mass=None,
            charge=None,
            circumference=None,
            bending_radius=None,
        )
