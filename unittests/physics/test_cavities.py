import unittest
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0

from blond import (
    ConstantMagneticCycle,
    Ring,
    Simulation,
    StaticProfile,
    proton,
)
from blond.acc_math.analytic.hamilton import (
    calc_synchrotron_tune_single_harmonic,
)
from blond.core.backends.backend import backend
from blond.core.base import DynamicParameter
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.particle_types import ParticleType
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.experimental.physics.feedbacks.base import (
    LocalFeedback,
)
from blond.experimental.physics.feedbacks.beam_feedback import BeamFeedbackBase
from blond.experimental.physics.feedbacks.cavity_feedback import (
    IQCavityFeedback,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.cavities import (
    MultiHarmonicRFStation,
    SingleHarmonicRFStation,
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
        self.beam.dE = backend.linspace(
            -1e6, 1e6, 10, dtype=backend.float
        )  # delta E
        # in eV
        self.beam.dt = backend.linspace(
            -1e-6, 1e-6, 10, dtype=backend.float
        )  # delta t
        # in s
        self.beam.read_partial_dt.return_value = self.beam.dt
        self.beam.write_partial_dE.return_value = self.beam.dE

        self.beam.common_array_size = len(self.beam.dE)

    def test_init_of_feedbacks(self):
        # default init
        SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        prof = StaticProfile.from_cutoff(0, 1e-9, 3e9)
        beam_feedback_good = Mock(spec=BeamFeedbackBase)

        SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=beam_feedback_good,
            cavity_feedback=None,
        )
        with self.assertRaises(TypeError):
            SingleHarmonicRFStation(
                section_index=1,
                local_wakefield=None,
                beam_feedback=prof,
                cavity_feedback=None,
            )

        cavity_feedback_good = Mock(spec=LocalFeedback)

        mhc = MultiHarmonicRFStation.headless(
            section_index=1,
            voltage=np.array([1]),
            harmonic=np.array([1]),
            phi_rf=np.array([1]),
            main_harmonic_idx=0,
            circumference=1,
            total_energy=1,
            beam_reference_beta=1,
            cavity_feedback=cavity_feedback_good,
        )
        cavity_feedback_good._parent_rf_station = None  # reset
        # TODO: remove this, once cavity feedback setup is fixed
        MultiHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            main_harmonic_idx=0,
            n_harmonics=1,
            voltage=np.array([1]),
            harmonic=np.array([1]),
            phi_rf=np.array([1]),
            cavity_feedback=[
                cavity_feedback_good,
            ],
        )
        with self.assertRaises(TypeError):
            SingleHarmonicRFStation(
                section_index=1,
                local_wakefield=None,
                cavity_feedback=[
                    prof,
                ],
            )

    def test_raising_error_setters_omega_rf_phi_rf(self):
        shc = SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        with self.assertRaisesRegex(AttributeError, "`omega_rf` can not be"):
            shc.omega_rf = 0
        with self.assertRaisesRegex(AttributeError, "`phi_rf` can not be"):
            shc.phi_rf = 0

    def test__get_gap_voltage_per_harmonic(self):
        def calc_rf_waveform(
            _time_arr, _omega, _phi, _voltage, _v_corr=1, _phi_corr=0
        ):
            return (
                _voltage
                * _v_corr
                * np.sin(_omega * _time_arr + _phi + _phi_corr)
            )

        ts = np.linspace(0, 20, 100)
        phi_rf = np.array([0, 1, 2, 3])
        omega_rf = np.array([4, 5, 6, 7])
        voltage = np.array([10, 20, 30, 40])
        harmonic_index = np.array([0, 1, 2, 3])

        mhc = MultiHarmonicRFStation(
            voltage=voltage,
            n_harmonics=len(harmonic_index),
            phi_rf=phi_rf,
            main_harmonic_idx=0,
            harmonic=np.zeros(len(harmonic_index)),
        )
        mhc.omega_rf_design = omega_rf

        for harm_ind in harmonic_index:
            np.testing.assert_allclose(
                mhc._get_gap_voltage_per_harmonic(ts, harm_ind),
                calc_rf_waveform(
                    ts, omega_rf[harm_ind], phi_rf[harm_ind], voltage[harm_ind]
                ),
            )

        cav_fb_0 = Mock(spec=LocalFeedback)
        cav_fb_0.relative_voltage_correction = np.ones(len(ts)) * 5
        cav_fb_0.phase_correction = np.arange(0.1, 0.7, 100)
        cav_fb_2 = Mock(spec=LocalFeedback)
        cav_fb_2.relative_voltage_correction = np.ones(len(ts)) * 10
        cav_fb_2.phase_correction = np.arange(0.7, 1.7, len(ts))
        mhc.cavity_feedback_list = [cav_fb_0, None, cav_fb_2, None]
        cav_fb_0.profile = Mock(spec=StaticProfile)
        cav_fb_0.profile.n_bins = len(ts)
        cav_fb_0.profile.hist_x = ts

        sol = calc_rf_waveform(
            ts,
            omega_rf[0],
            phi_rf[0],
            voltage[0],
            _v_corr=np.ones(len(ts)) * 5,
            _phi_corr=np.arange(0.1, 0.7, 100),
        )
        sol += calc_rf_waveform(
            ts,
            omega_rf[2],
            phi_rf[2],
            voltage[2],
            _v_corr=np.ones(len(ts)) * 10,
            _phi_corr=np.arange(0.7, 1.7, 100),
        )
        for harm_ind in [1, 3]:
            sol += calc_rf_waveform(
                ts, omega_rf[harm_ind], phi_rf[harm_ind], voltage[harm_ind]
            )

        np.testing.assert_allclose(mhc.calc_gap_voltage_with_feedbacks(), sol)

    def test_attach_cavity_feedback(self):
        cavity_feedback_good = Mock(spec=LocalFeedback)

        mhc = MultiHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            voltage=np.array([6e6]),
            harmonic=np.array([25000]),
            n_harmonics=1,
            main_harmonic_idx=0,
            phi_rf=np.array([0]),
        )
        mhc.attach_cavity_feedback(cavity_feedback_good)

        mhc = MultiHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            voltage=np.array([6e6, 6e6]),
            harmonic=np.array([25000, 2]),
            n_harmonics=2,
            main_harmonic_idx=0,
            phi_rf=np.array([0, 2]),
        )
        with self.assertRaisesRegex(
            ValueError,
            "If a single feedback is provided, "
            "the harmonic_index needs to be provided as well",
        ):
            mhc.attach_cavity_feedback(cavity_feedback_good)
        mhc.attach_cavity_feedback(cavity_feedback_good, 0)
        assert mhc.any_feedback_not_none
        assert mhc.cavity_feedback_list[1] is None
        mhc.attach_cavity_feedback(cavity_feedback_good, 1)
        assert mhc.any_feedback_not_none
        with self.assertRaisesRegex(
            ValueError, "must be less than the number of RF stations."
        ):
            mhc.attach_cavity_feedback(cavity_feedback_good, 2)

        mhc = MultiHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            voltage=np.array([6e6, 6e6]),
            harmonic=np.array([25000, 2]),
            n_harmonics=2,
            main_harmonic_idx=0,
            phi_rf=np.array([0, 2]),
        )
        with self.assertRaisesRegex(TypeError, "Invalid input type"):
            mhc.attach_cavity_feedback(Mock("not_a_fdbk"))

        with self.assertRaisesRegex(ValueError, "incorrect length"):
            mhc.attach_cavity_feedback([cavity_feedback_good, None, None], 0)

        with self.assertWarnsRegex(UserWarning, "will be ignored"):
            mhc.attach_cavity_feedback([cavity_feedback_good, None], 0)
        with self.assertWarnsRegex(UserWarning, "are being overridden"):
            mhc.attach_cavity_feedback([cavity_feedback_good, None], 0)

    def test_single_cavity_feedback_allowed(self):
        self.track_called = False

        def dummy_track(beam: BeamBaseClass):
            self.track_called = True
            return

        prof = StaticProfile.from_cutoff(0, 1e-9, 3e9)

        cavity_feedback_good = Mock(spec=LocalFeedback)
        cavity_feedback_good.track = dummy_track
        cavity_feedback_good.profile = prof
        cavity_feedback_good.phase_correction = 0
        cavity_feedback_good.relative_voltage_correction = 0

        mhc = SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            voltage=6e6,
            harmonic=25000,
            phi_rf=0,
            cavity_feedback=cavity_feedback_good,
        )

        mhc._turn_i = 1
        mhc._ring = Mock(Ring)
        mhc._ring.circumference = 456

        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(1)
        simulation.ring.circumference = 456
        simulation.ring.section_lengths = np.array(
            [simulation.ring.circumference]
        )
        simulation.magnetic_cycle = Mock(ConstantMagneticCycle)
        simulation.magnetic_cycle.get_target_total_energy.return_value = 1.0

        self.beam.ratio = 0.01

        mhc.on_init_simulation(simulation=simulation)
        mhc.on_run_simulation(
            simulation=simulation,
            beam=self.beam,
            n_turns=100,
        )
        cavity_feedback_good.on_run_simulation(
            simulation=simulation,
            beam=self.beam,
            n_turns=100,
        )

        mhc.track(beam=self.beam)

        assert self.track_called

    def test_single_cavity_feedbacks_allowed_mhc(self):
        self.track_called = False

        def dummy_track(beam: BeamBaseClass):
            self.track_called = True
            return

        prof = StaticProfile.from_cutoff(0, 1e-9, 3e9)

        cavity_feedback_good = Mock(spec=LocalFeedback)
        cavity_feedback_good.track = dummy_track
        cavity_feedback_good.profile = prof
        cavity_feedback_good.phase_correction = 0
        cavity_feedback_good.relative_voltage_correction = 0

        mhc = MultiHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            voltage=np.array([6e6]),
            harmonic=np.array([25000]),
            n_harmonics=1,
            main_harmonic_idx=0,
            phi_rf=np.array([0]),
            cavity_feedback=cavity_feedback_good,
        )

        mhc._turn_i = 1
        mhc._ring = Mock(Ring)
        mhc._ring.circumference = 456

        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(1)
        simulation.ring.circumference = 456
        simulation.ring.section_lengths = np.array(
            [simulation.ring.circumference]
        )
        simulation.magnetic_cycle = Mock(ConstantMagneticCycle)
        simulation.magnetic_cycle.get_target_total_energy.return_value = 1.0

        self.beam.ratio = 0.01

        mhc.on_init_simulation(simulation=simulation)
        mhc.on_run_simulation(
            simulation=simulation,
            beam=self.beam,
            n_turns=100,
        )
        cavity_feedback_good.on_run_simulation(
            simulation=simulation,
            beam=self.beam,
            n_turns=100,
        )

        mhc.track(beam=self.beam)

        assert self.track_called

    def test_track_with_feedbacks(self):
        SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        # prof = StaticProfile.from_cutoff(0, 1e-9, 3e9)
        beam_feedback_good = Mock(spec=BeamFeedbackBase)
        cavity_feedback_good = Mock(spec=IQCavityFeedback)
        cavity_feedback_good.info_string.return_value = (
            "Unnamed-LocalFeedback-000"
        )

        # TODO: remove this, once cavity feedback setup is fixed
        mhc_feedbacks = MultiHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            main_harmonic_idx=0,
            n_harmonics=1,
            voltage=np.array([1]),
            phi_rf=np.array([1]),
            harmonic=np.array([1]),
            cavity_feedback=[
                cavity_feedback_good,
            ],
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
            simulation=simulation,
            beam=self.beam,
            n_turns=100,
        )

        with self.assertRaises(TypeError):
            mhc_feedbacks.track(beam=self.beam)

            cavity_feedback_good.track.assert_called_once()

        info_str = mhc_feedbacks.info_string()
        assert "Feedback" in info_str
        # TODO: here a test should be added which checks for the correct ordering of the calls with Mocks

    def test_with_wakefields(self):
        wf = Mock(WakeField)
        shc = SingleHarmonicRFStation(
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

    def test_tune_main_harmonic(self):
        mhc = MultiHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            voltage=np.array([2 * np.pi * 1e6, 6e6]),
            harmonic=np.array([1, 2]),
            n_harmonics=2,
            main_harmonic_idx=0,
            phi_rf=np.array([0, 2]),
        )
        mhc._ring = Mock(spec=Ring)
        mhc._ring.calc_average_eta_0.return_value = 1

        def phi_s(beam):
            return np.pi / 2

        mhc.calc_phi_s_main_harmonic = phi_s

        beam = Mock(spec=BeamBaseClass)
        beam.particle_type = Mock(spec=ParticleType)
        beam.particle_type.charge = 2
        beam.reference = Mock(spec=ReferenceCoordinates)
        beam.reference.beta = 1
        beam.reference.total_energy = 1e6
        self.assertAlmostEqual(
            mhc.calc_synchrotron_tune_main_harmonic(beam),
            calc_synchrotron_tune_single_harmonic(
                charge=2,
                voltage=2 * np.pi * 1e6,
                beta=1,
                energy=1e6,
                phi_s=np.pi / 2,
                harmonic=1,
                eta_0=1,
            ),
        )

        mhc = MultiHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            voltage=np.array([2 * np.pi * 1e6, 6e6]),
            harmonic=np.array([1, 2]),
            n_harmonics=2,
            main_harmonic_idx=0,
            phi_rf=np.array([0, 2]),
        )
        mhc._ring = Mock(spec=Ring)
        mhc._ring.calc_average_eta_0.return_value = 1

        def phi_s(beam):
            return 0

        mhc.calc_phi_s_main_harmonic = phi_s
        self.assertEqual(
            mhc.calc_synchrotron_tune_main_harmonic(beam), np.sqrt(2)
        )

        mhc = MultiHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            voltage=np.array([6e6, 9e6]),
            harmonic=np.array([35640, 2]),
            n_harmonics=2,
            main_harmonic_idx=0,
            phi_rf=np.array([0, 2]),
        )
        alpha = 1 / 55.759505**2
        gamma = 450e9 / proton.mass
        eta = alpha - (1 / (gamma**2))
        mhc._ring = Mock(spec=Ring)
        mhc._ring.calc_average_eta_0.return_value = eta

        def phi_s(beam):
            return 0

        mhc.calc_phi_s_main_harmonic = phi_s

        beam = Mock(spec=BeamBaseClass)
        beam.particle_type = Mock(spec=ParticleType)
        beam.particle_type.charge = 1
        beam.reference = Mock(spec=ReferenceCoordinates)
        beam.reference.beta = 1
        beam.reference.total_energy = 450e9
        self.assertAlmostEqual(
            mhc.calc_synchrotron_tune_main_harmonic(beam), 0.00489862554460765
        )

        assert calc_synchrotron_tune_single_harmonic(
            2, 2 * np.pi * 1e6, 1, 1e6, 0, 1, 1
        ) == np.sqrt(2)
        self.assertAlmostEqual(
            calc_synchrotron_tune_single_harmonic(
                2, 2 * np.pi * 1e6, 1, 1e6, np.pi / 2, 1, 1
            ),
            0,
        )

        # LHC flat bottom
        alpha = 1 / 55.759505**2
        gamma = 450e9 / proton.mass
        eta = alpha - (1 / (gamma**2))
        assert (
            calc_synchrotron_tune_single_harmonic(
                1, 6e6, 1, 450e9, 0, 35640, eta
            )
            == 0.00489862554460765
        )


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
        beam.dE = backend.linspace(-1e6, 1e6, 10, dtype=backend.float)  #
        # delta E  in eV
        beam.dt = backend.linspace(-1e-6, 1e-6, 10, dtype=backend.float)  #
        # delta t in s
        beam.read_partial_dt.return_value = beam.dt
        beam.write_partial_dE.return_value = beam.dE

        self.beam = beam

        self.multi_harmonic_cavity = MultiHarmonicRFStation.headless(
            section_index=0,
            voltage=np.array([1e6, 2e6], dtype=float),
            phi_rf=np.array([0.1 * np.pi, np.pi], dtype=float),
            harmonic=np.array([1, 5], dtype=float),
            circumference=456,
            local_wakefield=None,
            cavity_feedback=None,
            total_energy=939,
            main_harmonic_idx=0,
            beam_reference_beta=1,
        )
        self.multi_harmonic_cavity._ring.section_lengths = [1, 2, 3]

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_track_increments(self) -> None:
        self.multi_harmonic_cavity
        self.multi_harmonic_cavity.delta_omega_rf = (
            0.1 * self.multi_harmonic_cavity.omega_rf_design
        )
        phi_a = self.multi_harmonic_cavity.delta_phi_rf.copy()
        self.multi_harmonic_cavity.track(beam=self.beam)
        phi_b = self.multi_harmonic_cavity.delta_phi_rf.copy()
        self.multi_harmonic_cavity.track(beam=self.beam)
        phi_c = self.multi_harmonic_cavity.delta_phi_rf.copy()
        print(phi_a, phi_b, phi_c)
        self.assertTrue(phi_a[0] == phi_b[0] < phi_c[0])
        # since the change will act on the next turn, the first two will be equivalent

    def test_track(self) -> None:
        self.multi_harmonic_cavity.track(beam=self.beam)

        self.assertEqual(self.beam.reference.total_energy, 939)  # incremented
        self.assertEqual(self.beam.reference.time, 0)  # unchanged

        # print(self.beam.dE.tolist())
        np.testing.assert_allclose(  # changer/ test pinned to some value
            copy_to_cpu(self.beam.dE),
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
            copy_to_cpu(self.beam.dt),
            np.linspace(-1e-6, 1e-6, 10),
        )

    def test_wrong_array(self) -> None:
        local_cav = MultiHarmonicRFStation(
            n_harmonics=2,
            main_harmonic_idx=0,
            voltage=np.array([1, 2]),
            phi_rf=np.array([3, 4]),
            harmonic=np.array([5, 6]),
        )
        np.testing.assert_allclose(
            copy_to_cpu(local_cav.voltage), np.array([1, 2])
        )
        np.testing.assert_allclose(
            copy_to_cpu(local_cav.phi_rf), np.array([3, 4])
        )
        np.testing.assert_allclose(
            copy_to_cpu(local_cav.harmonic), np.array([5, 6])
        )

        with self.assertRaises(ValueError):
            _ = MultiHarmonicRFStation(
                n_harmonics=2,
                main_harmonic_idx=0,
                voltage=np.array([1]),
                phi_rf=np.array([3, 4]),
                harmonic=np.array([5, 6]),
            )
        with self.assertRaises(ValueError):
            _ = MultiHarmonicRFStation(
                n_harmonics=2,
                main_harmonic_idx=0,
                voltage=np.array([1, 2]),
                phi_rf=np.array([3]),
                harmonic=np.array([5, 6]),
            )
        with self.assertRaises(ValueError):
            _ = MultiHarmonicRFStation(
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
            self.multi_harmonic_cavity.get_main_harmonic_t_rf()
            == 2
            * np.pi
            / self.multi_harmonic_cavity.get_main_harmonic_omega_rf()
        )
        assert (
            self.multi_harmonic_cavity.calc_main_harmonic_t_rf(
                beam_beta=self.beam.reference.beta, ring_circumference=456
            )
            == self.multi_harmonic_cavity.get_main_harmonic_t_rf()
        )

        self.multi_harmonic_cavity.cavity_feedback_list = [
            Mock(spec=LocalFeedback),
        ]
        with self.assertWarnsRegex(
            UserWarning,
            "`get_main_harmonic_voltage` returns unperturbed voltage",
        ):
            self.multi_harmonic_cavity.get_main_harmonic_voltage()

    def test_on_init_simulation_fails2(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.multi_harmonic_cavity.phi_rf_design = None
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
        beam.reference.time = float(0)
        beam.reference.beta = 0.5
        beam.reference.velocity = float(beam.reference.beta * c0)
        beam.reference.gamma = float(np.sqrt(1 - 0.25))  # beta**2
        beam.reference.total_energy = float(938)
        beam.dE = backend.linspace(
            -1e6, 1e6, 10, dtype=backend.float
        )  # delta E in eV
        beam.dt = backend.linspace(
            -1e-6, 1e-6, 10, dtype=backend.float
        )  # delta t in s
        beam.read_partial_dt.return_value = beam.dt
        beam.write_partial_dE.return_value = beam.dE

        self.beam = beam

        self.single_harmonic_cavity = SingleHarmonicRFStation.headless(
            section_index=0,
            voltage=1e6,
            phi_rf=np.pi * 0.3,
            harmonic=3.5,
            circumference=456,
            local_wakefield=None,
            cavity_feedback=None,
            total_energy=939.0,
            beam_reference_beta=beam.reference.beta,
        )
        self.single_harmonic_cavity._ring.section_lengths = [1, 2, 3]

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_track(self) -> None:
        self.single_harmonic_cavity.track(beam=self.beam)

        self.assertEqual(939, self.beam.reference.total_energy)  # incremented
        self.assertEqual(self.beam.reference.time, 0)  # unchanged
        np.testing.assert_allclose(  # test pinned to some value
            copy_to_cpu(self.beam.dE),
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
            copy_to_cpu(self.beam.dt),
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
            self.single_harmonic_cavity.get_main_harmonic_t_rf()
            == 2 * np.pi / self.single_harmonic_cavity.omega_rf
        )
        assert (
            self.single_harmonic_cavity.calc_main_harmonic_t_rf(
                beam_beta=self.beam.reference.beta, ring_circumference=456
            )
            == self.single_harmonic_cavity.get_main_harmonic_t_rf()
        )
        self.single_harmonic_cavity.cavity_feedback_list = [
            Mock(spec=LocalFeedback),
        ]
        with self.assertWarnsRegex(
            UserWarning,
            "`get_main_harmonic_voltage` returns unperturbed voltage",
        ):
            self.single_harmonic_cavity.get_main_harmonic_voltage()

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
            self.single_harmonic_cavity.phi_rf_design = None
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


if __name__ == "__main__":
    unittest.main()
