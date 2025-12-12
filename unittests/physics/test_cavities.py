import unittest
from unittest.mock import Mock

import numpy as np
from numpy._typing import NDArray as NumpyArray
from scipy.constants import speed_of_light as c0

from blond import Ring, Simulation, StaticProfile, proton
from blond.core.backends.backend import backend
from blond.core.base import DynamicParameter
from blond.core.beam.base import BeamBaseClass
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.experimental.physics.feedbacks.accelerators.sps.beam_feedback import (
    SpsRlBeamFeedback,
)
from blond.experimental.physics.feedbacks.accelerators.sps.cavity_feedback import (
    SPSOneTurnFeedback,
)
from blond.physics.cavities import (
    MultiHarmonicRfStation,
    RfStationBaseClass,
    SingleHarmonicRfStation,
)
from blond.physics.drifts import _assert_purely_real_or_imaginary
from blond.physics.impedances.base import WakeField


class TestRFStationBaseClass(unittest.TestCase):
    def setUp(self) -> None:
        self.beam = Mock(BeamBaseClass)
        self.beam.reference = Mock(ReferenceCoordinates)

        self.beam.particle_type = proton
        self.beam.reference.time = 0
        self.beam.reference.beta = 0.5
        self.beam.reference.velocity = self.beam.reference.beta * c0
        self.beam.reference.gamma = np.sqrt(1 - 0.25)  # beta**2
        self.beam.reference.total_energy = 938
        self.beam.dE = np.linspace(
            -1e6, 1e6, 10, dtype=backend.float
        )  # delta E
        # in eV
        self.beam.dt = np.linspace(
            -1e-6, 1e-6, 10, dtype=backend.float
        )  # delta t
        # in s
        self.beam.read_partial_dt.return_value = self.beam.dt
        self.beam.write_partial_dE.return_value = self.beam.dE

    def test_init_of_feedbacks(self):
        # default init
        SingleHarmonicRfStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        prof = StaticProfile.from_cutoff(0, 1e-9, 3e9)
        beam_feedback_good = SpsRlBeamFeedback(
            section_index=0, profile=prof, PL_gain=1
        )

        SingleHarmonicRfStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=beam_feedback_good,
            cavity_feedback=None,
        )
        with self.assertRaises(ValueError):
            SingleHarmonicRfStation(
                section_index=1,
                local_wakefield=None,
                beam_feedback=prof,
                cavity_feedback=None,
            )

        mhc = MultiHarmonicRfStation.headless(
            section_index=1,
            voltage=np.array([1]),
            harmonic=np.array([1]),
            phi_rf=np.array([1]),
            main_harmonic_idx=0,
            circumference=1,
            total_energy=1,
            reference_beta=1,
        )
        cavity_feedback_good = SPSOneTurnFeedback(
            profile=prof, _parent_rf_station=mhc, n_sections=3
        )

        # TODO: remove this, once cavity feedback setup is fixed
        MultiHarmonicRfStation(
            section_index=1,
            local_wakefield=None,
            main_harmonic_idx=0,
            n_harmonics=1,
            cavity_feedback=(cavity_feedback_good,),
        )
        with self.assertRaises(ValueError):
            SingleHarmonicRfStation(
                section_index=1, local_wakefield=None, cavity_feedback=(prof,)
            )

    def test_track_with_feedbacks(self):
        SingleHarmonicRfStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        # prof = StaticProfile.from_cutoff(0, 1e-9, 3e9)
        beam_feedback_good = Mock(
            SpsRlBeamFeedback
        )  # (section_index=0, profile=prof, PL_gain=1)
        beam_feedback_good.delay = 1e-9
        beam_feedback_good.domega_rf = 0
        # mhc = MultiHarmonicRfStation.headless(section_index=1, voltage=np.array([1]), harmonic=np.array([1]),
        #                                       phi_rf=np.array([1]), main_harmonic_idx=0, circumference=1,
        #                                       total_energy=1, reference.beta=1)
        cavity_feedback_good = Mock(
            SPSOneTurnFeedback
        )  # profile=prof, _parent_rf_station=mhc, n_sections=3)
        cavity_feedback_good.info_string.return_value = (
            "Unnamed-LocalFeedback-000"
        )

        # TODO: remove this, once cavity feedback setup is fixed
        mhc_feedbacks = MultiHarmonicRfStation(
            section_index=1,
            local_wakefield=None,
            main_harmonic_idx=0,
            n_harmonics=1,
            voltage=np.array([1]),
            phi_rf=np.array([1]),
            harmonic=np.array([1]),
            cavity_feedback=(cavity_feedback_good,),
            beam_feedback=beam_feedback_good,
        )

        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(1)
        simulation.ring.circumference = 456
        simulation.ring.section_lengths = np.array(
            [simulation.ring.circumference]
        )

        mhc_feedbacks.on_init_simulation(simulation=simulation)
        mhc_feedbacks.on_run_simulation(
            simulation=simulation, beam=self.beam, n_turns=100, turn_i_init=0
        )

        with self.assertRaises(TypeError):
            mhc_feedbacks.track(beam=self.beam)

            cavity_feedback_good.track.assert_called_once()

        info_str = mhc_feedbacks.info_string()
        assert "Feedback" in info_str

    def test_with_wakefields(self):
        wf = Mock(WakeField)
        shc = SingleHarmonicRfStation(
            section_index=0,
            harmonic=1,
            voltage=1,
            phi_rf=1,
            local_wakefield=wf,
        )
        shc._turn_i = DynamicParameter(0)
        shc._ring = Mock(Ring)
        shc._ring.circumference = 456
        with self.assertRaises(AttributeError):
            shc.track(beam=self.beam)
            assert wf.track.assert_called_once()


class TestCallables(unittest.TestCase):
    def test_valid_purely_real_or_imaginary(self):
        """Test that purely real, purely imaginary, and zero pass."""
        for val in [5 + 0j, 0 + 3j, 0j]:
            _assert_purely_real_or_imaginary(val)  # Should not raise

    def test_invalid_purely_real_or_imaginary(self):
        with self.assertRaises(ValueError):
            _assert_purely_real_or_imaginary(5 + 1j)  # Should  raise


class TestMultiHarmonicCavity(unittest.TestCase):
    def setUp(self) -> None:
        from blond.core.beam.base import BeamBaseClass

        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        beam.common_array_size = 1
        beam.particle_type = proton
        beam.reference.time = 0
        beam.reference.beta = 0.5
        beam.reference.velocity = beam.reference.beta * c0
        beam.reference.gamma = np.sqrt(1 - 0.25)  # beta**2
        beam.reference.total_energy = 938
        beam.dE = np.linspace(-1e6, 1e6, 10, dtype=backend.float)  # delta E
        # in eV
        beam.dt = np.linspace(-1e-6, 1e-6, 10, dtype=backend.float)  # delta t
        # in s
        beam.read_partial_dt.return_value = beam.dt
        beam.write_partial_dE.return_value = beam.dE

        self.beam = beam

        self.multi_harmonic_cavity = MultiHarmonicRfStation.headless(
            section_index=0,
            voltage=np.array([1e6, 2e6], dtype=backend.float),
            phi_rf=np.array([0.1 * np.pi, np.pi], dtype=backend.float),
            harmonic=np.array([1, 5], dtype=backend.float),
            circumference=456,
            local_wakefield=None,
            cavity_feedback=None,
            total_energy=939,
            main_harmonic_idx=0,
            reference_beta=1,
        )
        self.multi_harmonic_cavity._ring.section_lengths = [1, 2, 3]

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_track_increments(self) -> None:
        self.multi_harmonic_cavity
        self.multi_harmonic_cavity.delta_omega_rf = (
            0.1 * self.multi_harmonic_cavity._omega_rf
        )
        phi_a = self.multi_harmonic_cavity.delta_phi_rf.copy()
        self.multi_harmonic_cavity.track(beam=self.beam)
        phi_b = self.multi_harmonic_cavity.delta_phi_rf.copy()
        self.multi_harmonic_cavity.track(beam=self.beam)
        phi_c = self.multi_harmonic_cavity.delta_phi_rf.copy()
        print(phi_a, phi_b, phi_c)
        self.assertTrue(phi_a[0] < phi_b[0] < phi_c[0])

    def test_track(self) -> None:
        self.multi_harmonic_cavity.track(beam=self.beam)

        self.assertEqual(self.beam.reference.total_energy, 939)  # incremented
        self.assertEqual(self.beam.reference.time, 0)  # unchanged

        # print(self.beam.dE.tolist())
        np.testing.assert_allclose(  # changer/ test pinned to some value
            self.beam.dE,
            [
                -3553222.1295187217,
                229103.39306234661,
                -2334151.389566862,
                -1291443.680401674,
                1796893.796132672,
                -1195065.0503718334,
                1768699.6153487992,
                2588047.0010012407,
                -251122.31467230315,
                3259845.9525205432,
            ],
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )

        np.testing.assert_allclose(  # unchanged
            self.beam.dt,
            np.linspace(-1e-6, 1e-6, 10),
        )

    def test_wrong_array(self) -> None:
        local_cav = MultiHarmonicRfStation(
            n_harmonics=2,
            main_harmonic_idx=0,
            voltage=np.array([1, 2]),
            phi_rf=np.array([3, 4]),
            harmonic=np.array([5, 6]),
        )
        np.testing.assert_allclose(local_cav.voltage, np.array([1, 2]))
        np.testing.assert_allclose(local_cav.phi_rf, np.array([3, 4]))
        np.testing.assert_allclose(local_cav.harmonic, np.array([5, 6]))

        with self.assertRaises(ValueError):
            _ = MultiHarmonicRfStation(
                n_harmonics=2,
                main_harmonic_idx=0,
                voltage=np.array([1]),
                phi_rf=np.array([3, 4]),
                harmonic=np.array([5, 6]),
            )
        with self.assertRaises(ValueError):
            _ = MultiHarmonicRfStation(
                n_harmonics=2,
                main_harmonic_idx=0,
                voltage=np.array([1, 2]),
                phi_rf=np.array([3]),
                harmonic=np.array([5, 6]),
            )
        with self.assertRaises(ValueError):
            _ = MultiHarmonicRfStation(
                n_harmonics=2,
                main_harmonic_idx=0,
                voltage=np.array([1, 2]),
                phi_rf=np.array([3, 4]),
                harmonic=np.array([5]),
            )

    def test_on_init_simulation_fails(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.multi_harmonic_cavity.voltage = None
            self.multi_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_general_getters(self) -> None:
        self.multi_harmonic_cavity._update_beam_based_attributes(
            beam=self.beam
        )
        assert (
            self.multi_harmonic_cavity.get_main_harmonic()
            == self.multi_harmonic_cavity.harmonic[
                self.multi_harmonic_cavity.main_harmonic_idx
            ]
        )
        assert (
            self.multi_harmonic_cavity.get_main_harmonic_t_rf_current()
            == 2
            * np.pi
            / self.multi_harmonic_cavity._omega_rf[
                self.multi_harmonic_cavity.main_harmonic_idx
            ]
        )
        assert (
            self.multi_harmonic_cavity.calc_main_harmonic_t_rf(
                beam_beta=self.beam.reference.beta, ring_circumference=456
            )
            == self.multi_harmonic_cavity.get_main_harmonic_t_rf_current()
        )

    def test_on_init_simulation_fails2(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.multi_harmonic_cavity.phi_rf = None
            self.multi_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_on_init_simulation_fails3(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.multi_harmonic_cavity.harmonic = None
            self.multi_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_info_string(self):
        self.multi_harmonic_cavity.info_string()  # just hope it executes.


class TestSingleHarmonicCavity(unittest.TestCase):
    def setUp(self) -> None:
        from blond.core.beam.base import BeamBaseClass

        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        beam.common_array_size = 1
        beam.particle_type = proton
        beam.reference.time = backend.float(0)
        beam.reference.beta = 0.5
        beam.reference.velocity = backend.float(beam.reference.beta * c0)
        beam.reference.gamma = backend.float(np.sqrt(1 - 0.25))  # beta**2
        beam.reference.total_energy = backend.float(938)
        beam.dE = np.linspace(
            -1e6, 1e6, 10, dtype=backend.float
        )  # delta E in eV
        beam.dt = np.linspace(
            -1e-6, 1e-6, 10, dtype=backend.float
        )  # delta t in s
        beam.read_partial_dt.return_value = beam.dt
        beam.write_partial_dE.return_value = beam.dE

        self.beam = beam

        self.single_harmonic_cavity = SingleHarmonicRfStation.headless(
            section_index=0,
            voltage=1e6,
            phi_rf=np.pi * 0.3,
            harmonic=3.5,
            circumference=456,
            local_wakefield=None,
            cavity_feedback=None,
            total_energy=939.0,
        )
        self.single_harmonic_cavity._ring.section_lengths = [1, 2, 3]

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_track(self) -> None:
        self.single_harmonic_cavity.track(beam=self.beam)

        self.assertEqual(939, self.beam.reference.total_energy)  # incremented
        self.assertEqual(self.beam.reference.time, 0)  # unchanged
        np.testing.assert_allclose(  # test pinned to some value
            self.beam.dE,
            [
                -1003263.8619856804,
                221697.39838640607,
                -623504.6270207566,
                -1327969.3279760184,
                27701.968069498133,
                1095854.844844814,
                124356.9102684273,
                -414301.0439499994,
                1055852.7198949838,
                1950042.1738763654,
            ],
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )

        np.testing.assert_allclose(  # unchanged
            self.beam.dt,
            np.linspace(-1e-6, 1e-6, 10),
        )

    def test_general_getters(self) -> None:
        self.single_harmonic_cavity._update_beam_based_attributes(
            beam=self.beam
        )
        assert (
            self.single_harmonic_cavity.get_main_harmonic()
            == self.single_harmonic_cavity.harmonic
        )
        assert (
            self.single_harmonic_cavity.get_main_harmonic_t_rf_current()
            == 2 * np.pi / self.single_harmonic_cavity._omega_rf
        )
        assert (
            self.single_harmonic_cavity.calc_main_harmonic_t_rf(
                beam_beta=self.beam.reference.beta, ring_circumference=456
            )
            == self.single_harmonic_cavity.get_main_harmonic_t_rf_current()
        )

    def test_on_init_simulation_fails(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.single_harmonic_cavity.voltage = None
            self.single_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_on_init_simulation_fails2(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.single_harmonic_cavity.phi_rf = None
            self.single_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_on_init_simulation_fails3(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.single_harmonic_cavity.harmonic = None
            self.single_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_voltage_waveform_tmp(self):
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)

        time_array = np.array([1, 2, 3])
        self.single_harmonic_cavity._omega_rf = np.array([3.0e9])
        volt_calc = self.single_harmonic_cavity.voltage_waveform_tmp(
            time_array
        )
        assert len(volt_calc) == len(time_array)


if __name__ == "__main__":
    unittest.main()
