import unittest
import warnings
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import sympy
from matplotlib import pyplot as plt
from numpy import ndarray as NumpyArray
from scipy.constants import speed_of_light as c0

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    StaticProfile,
    make_multibunch_beam,
    mu_minus,
    mu_plus,
    positron,
)
from blond.acc_math.analytic.hamilton import (
    calc_phi_s_single_harmonic,
    calc_synchrotron_tune_single_harmonic,
)
from blond.acc_math.analytic.simple_math import (
    _assert_purely_real_or_imaginary,
)
from blond.acc_math.analytic.synchrotron_radiation.synchrotron_radiation_maths import (
    calculate_energy_loss_per_turn,
)
from blond.core.backends.backend import backend
from blond.core.base import DynamicParameter
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.beams import ProbeBeam
from blond.core.beam.particle_types import ParticleType, lead_82, proton
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.experimental import PooledInterpolationKick
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.cavities import (
    MultiHarmonicRFStation,
    SingleHarmonicRFStation,
)
from blond.physics.drifts import DriftSimple
from blond.physics.feedbacks.base import LocalFeedback
from blond.physics.feedbacks.beam_feedback import BeamFeedbackBase
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.impedances.base import WakeField
from blond.physics.profiles_sparse import EquidistantMultiProfile
from blond.testing.backend_testing import multi_backend_testcase
from blond.testing.helpers import allclose_tolerances


def _fixed_total_energy_cycle(total_energy: float):
    """Minimal stand-in for a magnetic cycle returning a constant total energy.

    Used by headless RF-station tests whose beams carry hand-picked (often
    unphysical) reference energies that a real ``ConstantMagneticCycle`` would
    reject. ``headless(magnetic_cycle=...)`` only needs the
    ``get_target_total_energy`` interface.
    """

    return SimpleNamespace(get_target_total_energy=lambda **_: total_energy)


class TestRFStationBaseClass(unittest.TestCase):
    def setUp(self) -> None:
        self.beam = Mock(BeamBaseClass)
        self.beam.reference = Mock(ReferenceCoordinates)
        # `ReferenceCoordinates` carries the pending-gain ledger that
        # reframing elements fill and the station consumes; a spec'd
        # Mock hands back a child Mock, which is not summable.
        self.beam.reference.pending_rf_energy_gain = 0.0
        self.beam.reference.take_pending_rf_energy_gain.return_value = 0.0

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
        self.beam.read_partial_dt.return_value = self.beam.dt
        self.beam.write_partial_dE.return_value = self.beam.dE
        self.beam.signed_charge_with_direction.return_value = proton._charge

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

    def test_delta_omega_rf_warns_on_post_init_change(self):
        shc = SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        # Re-assigning the same value (including the init value) must not warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            shc.delta_omega_rf = shc.delta_omega_rf
        # Changing it after initialisation warns: downstream cavity feedbacks
        # sample it once per turn and treat it as constant within a turn.
        with self.assertWarnsRegex(
            UserWarning, "delta_omega_rf changed after"
        ):
            shc.delta_omega_rf = 1.0e3

    def test_delta_omega_rf_change_raises_in_multi_station_ring(self):
        shc = SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        # Simulate run start in a ring with more than one RF station AND a
        # cavity feedback somewhere in it -- the reference tracking of that
        # feedback is the whole reason the offset cannot move mid-run.
        shc._n_rf_stations_in_ring = 2
        shc._cavity_feedback_in_ring = True
        # Re-assigning the same value stays silent.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            shc.delta_omega_rf = shc.delta_omega_rf
        # Changing it is forbidden: the cavity-feedback reference tracking
        # spans the other stations' sections, so a change during the turn
        # would be applied inconsistently across segments.
        with self.assertRaisesRegex(RuntimeError, "more than one RF station"):
            shc.delta_omega_rf = 1.0e3

    def test_delta_omega_rf_change_allowed_without_cavity_feedback(self):
        """A multi-station ring with no cavity feedback is unconstrained.

        The guard exists because a cavity feedback reconstructs the reference
        across the OTHER stations' sections, so a mid-run change cannot be
        applied consistently.  With no cavity feedback anywhere in the ring
        there is no such reconstruction, and a beam feedback (phase or radial
        loop) must be free to write the offset every turn -- which is exactly
        what ``BeamFeedbackBase`` does.
        """
        shc = SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        shc._n_rf_stations_in_ring = 2
        shc._cavity_feedback_in_ring = False

        shc.delta_omega_rf = 1.0e3

        self.assertEqual(shc.delta_omega_rf, 1.0e3)

    def test_on_run_simulation_detects_cavity_feedback_in_ring(self):
        """The guard arms off the ring's cavity feedbacks, not the count."""
        from blond.physics.cavities import RFStationBaseClass

        plain = SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        other = SingleHarmonicRFStation(
            section_index=2,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        simulation = Mock()
        simulation.ring.elements.get_elements.return_value = (plain, other)
        plain._update_n_rf_stations_in_ring(simulation)
        self.assertFalse(plain._cavity_feedback_in_ring)

        with_feedback = SingleHarmonicRFStation(
            section_index=2,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=Mock(spec=LocalFeedback),
        )
        simulation.ring.elements.get_elements.return_value = (
            plain,
            with_feedback,
        )
        plain._update_n_rf_stations_in_ring(simulation)
        self.assertTrue(plain._cavity_feedback_in_ring)

    def test_on_run_simulation_counts_rf_stations_in_ring(self):
        from blond.physics.cavities import RFStationBaseClass

        shc = SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        other = SingleHarmonicRFStation(
            section_index=2,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        simulation = Mock()
        simulation.ring.elements.get_elements.return_value = (shc, other)
        shc._update_n_rf_stations_in_ring(simulation)
        self.assertEqual(shc._n_rf_stations_in_ring, 2)
        simulation.ring.elements.get_elements.assert_called_once_with(
            RFStationBaseClass
        )

    def test_beam_feedback_presence_detection(self):
        from blond.physics.cavities import RFStationBaseClass
        from blond.physics.feedbacks.beam_feedback import (
            BeamFeedbackBase,
        )

        shc = SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )

        def ring_with(stations, loops):
            simulation = Mock()

            def fake_get_elements(class_):
                if class_ is RFStationBaseClass:
                    return stations
                if class_ is BeamFeedbackBase:
                    return loops
                return ()

            simulation.ring.elements.get_elements.side_effect = (
                fake_get_elements
            )
            return simulation

        # No loop anywhere -> clock does not need to lead.
        shc._update_beam_feedback_presence(ring_with((shc,), ()))
        self.assertFalse(shc._beam_feedback_in_simulation)

        # Loop as a ring element -> present.
        shc._update_beam_feedback_presence(
            ring_with((shc,), (Mock(BeamFeedbackBase),))
        )
        self.assertTrue(shc._beam_feedback_in_simulation)

        # Loop only attached to (another) station -> present.
        other = SingleHarmonicRFStation(
            section_index=2,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        other._beam_feedback = Mock(BeamFeedbackBase)
        shc._update_beam_feedback_presence(ring_with((shc, other), ()))
        self.assertTrue(shc._beam_feedback_in_simulation)

    def test_delta_omega_rf_phase_slip_uses_actual_elapsed_time(self):
        shc = SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        delta_omega = 2.0 * np.pi * 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shc.delta_omega_rf = delta_omega

        # First passage only initialises the phase-slip clock.
        shc._update_delta_phi_rf_from_beam_feedback(reference_time=1.0e-6)
        self.assertEqual(shc.phase_correction_frequency_offset, 0.0)

        # Second passage: slip is delta_omega times the actually elapsed
        # reference time (one revolution).
        shc._update_delta_phi_rf_from_beam_feedback(reference_time=3.0e-6)
        np.testing.assert_allclose(
            shc.phase_correction_frequency_offset,
            delta_omega * 2.0e-6,
            rtol=1e-15,
        )

        # Third passage with a *different* revolution time (acceleration):
        # the accumulated slip stays exact, delta_omega * total elapsed time.
        shc._update_delta_phi_rf_from_beam_feedback(reference_time=4.5e-6)
        np.testing.assert_allclose(
            shc.phase_correction_frequency_offset,
            delta_omega * 3.5e-6,
            rtol=1e-15,
        )

    def test_delta_omega_rf_phase_slip_clock_advances_while_offset_is_zero(
        self,
    ):
        shc = SingleHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            beam_feedback=None,
            cavity_feedback=None,
        )
        # delta_omega_rf is 0 -> no slip, but the clock must still follow.
        shc._update_delta_phi_rf_from_beam_feedback(reference_time=1.0e-6)
        shc._update_delta_phi_rf_from_beam_feedback(reference_time=2.0e-6)
        self.assertEqual(shc.phase_correction_frequency_offset, 0.0)

        # Switching the offset on later must accumulate only from the last
        # passage, not from the (stale) first one.
        delta_omega = 2.0 * np.pi * 50.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shc.delta_omega_rf = delta_omega
        shc._update_delta_phi_rf_from_beam_feedback(reference_time=3.0e-6)
        np.testing.assert_allclose(
            shc.phase_correction_frequency_offset,
            delta_omega * 1.0e-6,
            rtol=1e-15,
        )

    def test__get_gap_voltage_per_harmonic(self):
        def calc_rf_waveform(
            _time_arr, _omega, _phi, _voltage, _v_corr=1, _phi_corr=0
        ) -> NumpyArray:
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
            harmonic=np.ones(len(harmonic_index)),
        )
        mhc.omega_rf_design = omega_rf

        for harm_ind in harmonic_index:
            desired = calc_rf_waveform(
                ts,
                omega_rf[harm_ind],
                phi_rf[harm_ind],
                voltage[harm_ind],
            )
            np.testing.assert_allclose(
                copy_to_cpu(mhc._get_gap_voltage_per_harmonic(ts, harm_ind)),
                copy_to_cpu(desired),
                **allclose_tolerances(desired),
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

        np.testing.assert_allclose(
            copy_to_cpu(mhc.calc_gap_voltage_with_feedbacks()),
            sol,
            **allclose_tolerances(sol),
        )

        with self.assertRaisesRegex(
            ValueError, "If no `harmonic_index` is provided"
        ):
            mhc._get_gap_voltage_per_harmonic(ts=ts)

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

        self.assertIs(
            cavity_feedback_good, mhc.get_main_harmonic_cavity_feedback()
        )

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

        self.assertIs(
            cavity_feedback_good, mhc.get_main_harmonic_cavity_feedback()
        )

        mhc._turn_counter = 1
        mhc._ring = Mock(Ring)
        mhc._ring.circumference = 456

        simulation = Mock(Simulation)
        simulation.turn_counter = DynamicParameter(1)
        simulation.ring.circumference = 456
        simulation.ring.section_lengths = np.array(
            [simulation.ring.circumference]
        )
        simulation.ring.elements.get_elements.return_value = (mhc,)
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

        self.assertTrue(self.track_called)

    def test_single_cavity_feedbacks_allowed_mhc(self):
        self.track_called = False

        def dummy_track(beam: BeamBaseClass):
            self.track_called = True

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

        mhc._turn_counter = 1
        mhc._ring = Mock(Ring)
        mhc._ring.circumference = 456

        simulation = Mock(Simulation)
        simulation.turn_counter = DynamicParameter(1)
        simulation.ring.circumference = 456
        simulation.ring.section_lengths = np.array(
            [simulation.ring.circumference]
        )
        simulation.ring.elements.get_elements.return_value = (mhc,)
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

        self.assertTrue(self.track_called)

    # ``test_track_with_feedbacks`` was removed: it exercised the purged
    # experimental ``IQCavityFeedbackOld`` cavity feedback.

    def test_with_wakefields(self):
        wf = Mock(WakeField)
        # The station reads the local wakefield's profile time axis to evaluate
        # the gap voltage for the interpolated kick, so give the mock a real
        # time base spanning the (mocked) beam coordinates instead of a bare
        # Mock (otherwise ``omega_rf * profile.hist_x`` is float * Mock).
        wf.profile.hist_x = np.linspace(-2e-6, 2e-6, 64)
        shc = SingleHarmonicRFStation.headless(
            section_index=0,
            harmonic=1,
            voltage=1,
            phi_rf=1,
            circumference=456,
            beam_reference_beta=self.beam.reference.beta,
            local_wakefield=wf,
        )
        # With ``magnetic_cycle=None`` no reference-energy update is performed,
        # but the local wakefield must still be tracked.
        shc.track(beam=self.beam)
        wf.track.assert_called_once()

    def test_track_reference_without_magnetic_cycle_returns_zero(self):
        # A headless station with ``magnetic_cycle=None`` (e.g. when an
        # external code such as xsuite owns the reference) leaves the
        # reference untouched and applies no acceleration kick.
        shc = SingleHarmonicRFStation.headless(
            section_index=0,
            harmonic=1,
            voltage=1,
            phi_rf=1,
            circumference=456,
            beam_reference_beta=self.beam.reference.beta,
            magnetic_cycle=None,
        )
        original_total_energy = self.beam.reference.total_energy
        reference_energy_change = shc.track_reference(
            reference=self.beam.reference
        )
        self.assertEqual(reference_energy_change, 0.0)
        self.assertEqual(
            self.beam.reference.total_energy, original_total_energy
        )

    def test_calc_phi_s_main_harmonic_without_magnetic_cycle_raises(self):
        # The synchronous phase requires an energy program; a headless station
        # created without one (``magnetic_cycle=None``) must raise rather than
        # silently return a meaningless value.
        shc = SingleHarmonicRFStation.headless(
            section_index=0,
            harmonic=1,
            voltage=1,
            phi_rf=1,
            circumference=456,
            beam_reference_beta=self.beam.reference.beta,
            magnetic_cycle=None,
        )
        with self.assertRaisesRegex(ValueError, "magnetic_cycle"):
            shc.calc_phi_s_main_harmonic(beam=self.beam)

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
        # Co-rotating: direction-signed charge equals the raw charge.
        beam.signed_charge_with_direction.return_value = 2
        beam.reference = Mock(spec=ReferenceCoordinates)
        # `ReferenceCoordinates` carries the pending-gain ledger that
        # reframing elements fill and the station consumes; a spec'd
        # Mock hands back a child Mock, which is not summable.
        beam.reference.pending_rf_energy_gain = 0.0
        beam.reference.take_pending_rf_energy_gain.return_value = 0.0
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
        # Co-rotating: direction-signed charge equals the raw charge.
        beam.signed_charge_with_direction.return_value = 1
        beam.reference = Mock(spec=ReferenceCoordinates)
        # `ReferenceCoordinates` carries the pending-gain ledger that
        # reframing elements fill and the station consumes; a spec'd
        # Mock hands back a child Mock, which is not summable.
        beam.reference.pending_rf_energy_gain = 0.0
        beam.reference.take_pending_rf_energy_gain.return_value = 0.0
        beam.reference.beta = 1
        beam.reference.total_energy = 450e9
        self.assertAlmostEqual(
            mhc.calc_synchrotron_tune_main_harmonic(beam), 0.00489862554460765
        )

        self.assertEqual(
            calc_synchrotron_tune_single_harmonic(
                2, 2 * np.pi * 1e6, 1, 1e6, 0, 1, 1
            ),
            np.sqrt(2),
        )
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
        self.assertEqual(
            calc_synchrotron_tune_single_harmonic(
                1, 6e6, 1, 450e9, 0, 35640, eta
            ),
            0.00489862554460765,
        )


class TestAttachCavityFeedbackIndexValidation(unittest.TestCase):
    """
    ``attach_cavity_feedback`` slot handling.

    The station applies each feedback's corrections at its LIST slot,
    while a feedback computes them from the RF parameters at its own
    ``harmonic_index``. The slot argument is authoritative: attaching
    SETS the feedback's ``harmonic_index`` from the slot, so the two
    cannot disagree through this path. A slot that does not exist is
    rejected, and a fractional slot is a hard error -- a harmonic index
    is a list slot, not a physical quantity to be rounded.
    """

    @staticmethod
    def _make_station(n_harmonics: int = 3) -> MultiHarmonicRFStation:
        return MultiHarmonicRFStation(
            section_index=1,
            local_wakefield=None,
            voltage=np.full(n_harmonics, 6e6),
            harmonic=np.arange(1, n_harmonics + 1) * 25000.0,
            n_harmonics=n_harmonics,
            main_harmonic_idx=0,
            phi_rf=np.zeros(n_harmonics),
        )

    @staticmethod
    def _make_feedback(harmonic_index: int | None = None):
        feedback = Mock(spec=LocalFeedback)
        if harmonic_index is not None:
            feedback.harmonic_index = harmonic_index
        return feedback

    def test_negative_harmonic_index_raises(self):
        # -1 passes ``> n_rf - 1`` and would write to the LAST slot.
        station = self._make_station()
        with self.assertRaisesRegex(ValueError, "harmonic_index"):
            station.attach_cavity_feedback(self._make_feedback(), -1)
        self.assertFalse(station.any_feedback_not_none)

    def test_integral_float_harmonic_index_is_coerced(self):
        # Reaches the list assignment as a float today and dies there
        # with "list indices must be integers or slices, not float".
        station = self._make_station()
        feedback = self._make_feedback()
        station.attach_cavity_feedback(feedback, 2.0)
        self.assertIs(station.cavity_feedback_list[2], feedback)

    def test_fractional_harmonic_index_raises(self):
        # A harmonic index is a list slot, not a physical quantity, so
        # there is nothing meaningful to round 1.5 to: hard reject.
        station = self._make_station()
        with self.assertRaisesRegex(ValueError, "1.5"):
            station.attach_cavity_feedback(self._make_feedback(), 1.5)
        self.assertFalse(station.any_feedback_not_none)

    def test_numpy_integer_harmonic_index_is_accepted(self):
        # ``np.int64`` is a perfectly good list index but not an ``int``,
        # so the coercion must not reject it.
        station = self._make_station()
        feedback = self._make_feedback()
        station.attach_cavity_feedback(feedback, np.int64(2))
        self.assertIs(station.cavity_feedback_list[2], feedback)

    def test_non_numeric_harmonic_index_raises(self):
        station = self._make_station()
        with self.assertRaises(TypeError):
            station.attach_cavity_feedback(self._make_feedback(), "1")
        self.assertFalse(station.any_feedback_not_none)

    def test_mismatched_feedback_harmonic_index_is_set_from_slot(self):
        # The slot argument is authoritative: attaching overwrites the
        # feedback's own index, so a mismatch cannot survive the attach.
        station = self._make_station()
        feedback = self._make_feedback(harmonic_index=1)
        station.attach_cavity_feedback(feedback, 0)
        self.assertIs(station.cavity_feedback_list[0], feedback)
        self.assertEqual(feedback.harmonic_index, 0)

    def test_feedback_without_harmonic_index_still_attaches(self):
        # Duck-typed feedbacks that never declare a harmonic must keep
        # working exactly as before -- and must not have one grafted on.
        station = self._make_station()
        feedback = self._make_feedback()
        station.attach_cavity_feedback(feedback, 2)
        self.assertIs(station.cavity_feedback_list[2], feedback)
        self.assertFalse(hasattr(feedback, "harmonic_index"))

    def test_matching_feedback_harmonic_index_attaches(self):
        station = self._make_station()
        feedback = self._make_feedback(harmonic_index=2)
        station.attach_cavity_feedback(feedback, 2)
        self.assertIs(station.cavity_feedback_list[2], feedback)

    def test_list_path_sets_feedback_harmonic_index_from_position(self):
        station = self._make_station()
        feedback = self._make_feedback(harmonic_index=2)
        station.attach_cavity_feedback([None, feedback, None])
        self.assertIs(station.cavity_feedback_list[1], feedback)
        self.assertEqual(feedback.harmonic_index, 1)

    def test_list_path_accepts_matching_feedback_harmonic_index(self):
        station = self._make_station()
        feedback = self._make_feedback(harmonic_index=1)
        station.attach_cavity_feedback([None, feedback, None])
        self.assertIs(station.cavity_feedback_list[1], feedback)


class TestCallables(unittest.TestCase):
    def test_valid_purely_real_or_imaginary(self):
        """Test that purely real, purely imaginary, and zero pass."""
        for val in [5 + 0j, 0 + 3j, 0j]:
            _assert_purely_real_or_imaginary(val)  # Should not raise

    def test_invalid_purely_real_or_imaginary(self):
        with self.assertRaises(ValueError):
            _assert_purely_real_or_imaginary(5 + 1j)  # Should  raise

    def test_invalid_purely_real_or_imaginary_array(self):
        _assert_purely_real_or_imaginary(np.array([1, 1j, 1]))
        _assert_purely_real_or_imaginary(np.array([1, 1, 1]))
        _assert_purely_real_or_imaginary(np.array([1j, 1j, 1j]))
        with self.assertRaises(ValueError):
            _assert_purely_real_or_imaginary(
                np.array([1j, 1j + 1, 1j])
            )  # Should  raise

    @pytest.mark.cupy
    def test_invalid_purely_real_or_imaginary_array(self):
        try:
            import cupy as cp  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        _assert_purely_real_or_imaginary(cp.array([1, 1j, 1]))
        _assert_purely_real_or_imaginary(cp.array([1, 1, 1]))
        _assert_purely_real_or_imaginary(cp.array([1j, 1j, 1j]))
        with self.assertRaises(ValueError):
            _assert_purely_real_or_imaginary(
                cp.array([1j, 1j + 1, 1j])
            )  # Should  raise


class TestMultiHarmonicCavity(unittest.TestCase):
    def setUp(self) -> None:
        from blond.core.beam.base import BeamBaseClass

        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        # `ReferenceCoordinates` carries the pending-gain ledger that
        # reframing elements fill and the station consumes; a spec'd
        # Mock hands back a child Mock, which is not summable.
        beam.reference.pending_rf_energy_gain = 0.0
        beam.reference.take_pending_rf_energy_gain.return_value = 0.0
        beam.common_array_size = 1
        beam.particle_type = proton
        beam._particle_type = beam.particle_type
        beam.reference.time = 0
        beam.reference.beta = 0.5
        beam.reference.velocity = beam.reference.beta * c0
        beam.reference.gamma = np.sqrt(1 - 0.25)  # beta**2
        beam.reference.total_energy = 938
        beam.reference._total_energy = beam.reference.total_energy
        beam.dE = backend.linspace(-1e6, 1e6, 10, dtype=backend.float)
        # delta E  in eV
        beam.dt = backend.linspace(-1e-6, 1e-6, 10, dtype=backend.float)
        # delta t in s
        beam.read_partial_dt.return_value = beam.dt
        beam.write_partial_dE.return_value = beam.dE
        beam.signed_charge_with_direction.return_value = proton._charge

        self.beam = beam

        self.multi_harmonic_cavity = MultiHarmonicRFStation.headless(
            section_index=0,
            voltage=np.array([1e6, 2e6], dtype=float),
            phi_rf=np.array([0.1 * np.pi, np.pi], dtype=float),
            harmonic=np.array([1, 5], dtype=float),
            circumference=456,
            local_wakefield=None,
            cavity_feedback=None,
            magnetic_cycle=_fixed_total_energy_cycle(939),
            main_harmonic_idx=0,
            beam_reference_beta=1,
        )
        self.multi_harmonic_cavity._ring.section_lengths = [1, 2, 3]

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_delayed_kick_with_feedback_requires_time_axis(self):
        cavity_feedback = Mock(spec=LocalFeedback)
        with self.assertRaises(AssertionError):
            MultiHarmonicRFStation.headless(
                section_index=0,
                voltage=np.array([1e6], dtype=float),
                phi_rf=np.array([0.1 * np.pi], dtype=float),
                harmonic=np.array([1], dtype=float),
                circumference=456,
                local_wakefield=None,
                cavity_feedback=cavity_feedback,  # trigger
                main_harmonic_idx=0,
                beam_reference_beta=1,
                delayed_kick=Mock(PooledInterpolationKick),  # trigger
                # trigger because it shouldn't coexist with cavity feedback
                delayed_kick_time_axis=np.linspace(1, 2, 10),
            )

    def test_delayed_kick_without_feedback_list_requires_time_axis2(self):
        with self.assertRaises(
            AssertionError,
        ):
            self.multi_harmonic_cavity = MultiHarmonicRFStation.headless(
                section_index=0,
                voltage=np.array([1e6, 2e6], dtype=float),
                phi_rf=np.array([0.1 * np.pi, np.pi], dtype=float),
                harmonic=np.array([1, 5], dtype=float),
                circumference=456,
                local_wakefield=None,
                cavity_feedback=None,
                main_harmonic_idx=0,
                beam_reference_beta=1,
                delayed_kick=Mock(PooledInterpolationKick),
                delayed_kick_time_axis=None,  # trigger Exception
            )

    def test_delayed_kick_registers(self):
        delayed_kick_mock = Mock(PooledInterpolationKick)
        multi_harmonic_cavity = MultiHarmonicRFStation.headless(
            section_index=0,
            voltage=np.array([1e6, 2e6], dtype=float),
            phi_rf=np.array([0.1 * np.pi, np.pi], dtype=float),
            harmonic=np.array([1, 5], dtype=float),
            circumference=456,
            local_wakefield=None,
            cavity_feedback=None,
            main_harmonic_idx=0,
            beam_reference_beta=1,
            delayed_kick=delayed_kick_mock,
            delayed_kick_time_axis=np.linspace(1, 2, 10),  # trigger Exception
        )
        multi_harmonic_cavity.track(self.beam)
        self.assertTrue(delayed_kick_mock.register.called)

    def test_local_wakefield_uses_the_interpolated_kick(self):
        """A local wakefield selects the interpolated kick, as on SingleHarmonic.

        ``SingleHarmonicRFStation`` routes a station carrying a local
        wakefield through ``_track_interp`` -- the profile the wakefield
        already maintains makes the interpolated kick cheaper than one
        ``sin`` per macroparticle.  The multi-harmonic station has to make
        the same choice: otherwise the identical lattice tracks through a
        different kick path depending only on how the RF was declared.
        """
        wakefield = Mock()
        wakefield.profile = Mock(spec=StaticProfile)
        wakefield.profile.hist_x = np.linspace(-1e-6, 1e-6, 12)

        cavity = MultiHarmonicRFStation.headless(
            section_index=0,
            voltage=np.array([1e6, 5e5]),
            phi_rf=np.array([0.1, 0.2]),
            harmonic=np.array([5, 10]),
            main_harmonic_idx=0,
            circumference=456,
            local_wakefield=wakefield,
            cavity_feedback=None,
            beam_reference_beta=1,
            delayed_kick=None,
            delayed_kick_time_axis=None,
        )

        with (
            patch.object(cavity, "_track_interp") as interp,
            patch.object(cavity, "_track_no_interp") as no_interp,
        ):
            cavity.track(self.beam)

        interp.assert_called_once()
        no_interp.assert_not_called()
        # The wakefield's own grid is what the kick is interpolated on.
        np.testing.assert_array_equal(
            interp.call_args.kwargs["time_axis"], wakefield.profile.hist_x
        )

    def test_delayed_kick_with_feedback(self):
        delayed_kick_mock = Mock(PooledInterpolationKick)
        cavity_feedback = Mock(spec=LocalFeedback)
        cavity_feedback.profile = Mock(spec=StaticProfile)
        cavity_feedback.profile.n_bins = 10
        cavity_feedback.profile.hist_x = np.linspace(1, 2, 10)
        cavity_feedback.relative_voltage_correction = np.linspace(1, 2, 10)
        cavity_feedback.phase_correction = np.linspace(1, 2, 10)
        multi_harmonic_cavity = MultiHarmonicRFStation.headless(
            section_index=0,
            voltage=np.array([1e6, 2e6], dtype=float),
            phi_rf=np.array([0.1 * np.pi, np.pi], dtype=float),
            harmonic=np.array([1, 5], dtype=float),
            circumference=456,
            local_wakefield=None,
            cavity_feedback=[cavity_feedback, None],
            main_harmonic_idx=0,
            beam_reference_beta=1,
            delayed_kick=delayed_kick_mock,
            delayed_kick_time_axis=None,
        )
        multi_harmonic_cavity.track(self.beam)
        self.assertTrue(delayed_kick_mock.register.called)

    def test_track_increments(self) -> None:
        delta_omega_rf = 0.1 * self.multi_harmonic_cavity.omega_rf_design
        self.multi_harmonic_cavity.delta_omega_rf = delta_omega_rf
        # The slip accumulates from the actually elapsed reference time, so
        # advance the (mocked) reference clock between the passages as the
        # drifts of a real ring would.
        t_rev = 1.0e-6
        phi_a = self.multi_harmonic_cavity.delta_phi_rf.copy()
        # First passage initialises the slip clock, no slip yet.
        self.multi_harmonic_cavity.track(beam=self.beam)
        phi_b = self.multi_harmonic_cavity.delta_phi_rf.copy()
        # One revolution elapses; its slip is accumulated at the end of the
        # second passage and applied to the phase of the third.
        self.beam.reference.time += t_rev
        self.multi_harmonic_cavity.track(beam=self.beam)
        phi_c = self.multi_harmonic_cavity.delta_phi_rf.copy()
        self.beam.reference.time += t_rev
        self.multi_harmonic_cavity.track(beam=self.beam)
        phi_d = self.multi_harmonic_cavity.delta_phi_rf.copy()
        self.assertTrue(phi_a[0] == phi_b[0] == phi_c[0] < phi_d[0])
        np.testing.assert_allclose(
            phi_d, phi_c + delta_omega_rf * t_rev, rtol=1e-15
        )
        # since the change will act on the next turn, the first two will be equivalent

    def test_slip_clock_only_runs_with_beam_feedback_or_offset(self) -> None:
        # delta_omega_rf is zero and no beam feedback exists in the
        # simulation: nothing can introduce a phase slip, so the slip
        # bookkeeping is skipped entirely.
        self.multi_harmonic_cavity.track(beam=self.beam)
        self.assertIsNone(
            self.multi_harmonic_cavity._last_reference_time_phase_slip
        )
        # With a beam feedback present, the clock must lead even while the
        # offset is still zero (it may be switched on any turn).
        self.multi_harmonic_cavity._beam_feedback_in_simulation = True
        self.beam.reference.time += 1.0e-6
        self.multi_harmonic_cavity.track(beam=self.beam)
        self.assertEqual(
            self.multi_harmonic_cavity._last_reference_time_phase_slip,
            self.beam.reference.time,
        )

    def test_track(self) -> None:
        self.multi_harmonic_cavity.track(beam=self.beam)

        self.assertEqual(self.beam.reference.total_energy, 939)  # incremented
        self.assertEqual(self.beam.reference.time, 0)  # unchanged

        # print(self.beam.dE.tolist())
        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

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
            rtol=1e-12,
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
        simulation.turn_counter = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.multi_harmonic_cavity.voltage = None
            self.multi_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_general_getters(self) -> None:
        self.multi_harmonic_cavity._update_reference_based_attributes(
            reference=self.beam.reference
        )
        # TODO: Maybe better to use self.assertEqual here instead
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
            self.multi_harmonic_cavity.get_main_harmonic_t_rf()
            == 2
            * np.pi
            / self.multi_harmonic_cavity.get_main_harmonic_omega_rf_design()
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
        simulation.turn_counter = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.multi_harmonic_cavity.phi_rf_design = None
            self.multi_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_on_init_simulation_fails3(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_counter = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.multi_harmonic_cavity.harmonic = None
            self.multi_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_info_string(self):
        self.multi_harmonic_cavity.info_string()  # just hope it executes.

    @pytest.mark.backend_mutation
    def test_interp_kick_single_harmonic(self):
        beam = ProbeBeam(
            particle_type=lead_82,
            dt=np.linspace(0, 1, 100),
            reference_total_energy=1e12,
        )
        rf_station_smooth = SingleHarmonicRFStation.headless(
            section_index=0,
            voltage=1e3,
            phi_rf=np.deg2rad(33),
            harmonic=5,
            circumference=123,
            beam_reference_beta=beam.reference.beta,
            local_wakefield=None,
            cavity_feedback=None,
            delayed_kick=None,
            delayed_kick_time_axis=None,
        )
        beam1_smooth = deepcopy(beam)
        rf_station_smooth.track(beam1_smooth)
        result_smooth = beam1_smooth.dE.copy_as_numpy()

        pool = PooledInterpolationKick()
        beam_interp = deepcopy(beam)
        rf_station_interp = SingleHarmonicRFStation.headless(
            section_index=0,
            voltage=1e3,
            phi_rf=np.deg2rad(33),
            harmonic=5,
            circumference=123,
            beam_reference_beta=beam.reference.beta,
            local_wakefield=None,
            cavity_feedback=None,
            delayed_kick=pool,
            delayed_kick_time_axis=beam_interp.dt.copy_as_numpy(),
        )
        rf_station_interp.track(beam_interp)
        pool.track(beam_interp)

        result_interp = beam_interp.dE.copy_as_numpy()
        offset = 2 * np.min(result_smooth)
        result_smooth += offset
        result_interp += offset
        DEV_PLOT = False
        if DEV_PLOT:
            plt.plot(result_smooth, "o", label="result_smooth")
            plt.plot(result_interp, "x", label="result_interp")
            plt.show()
        self.assertTrue(not np.any(np.isnan(result_smooth)))
        self.assertTrue(not np.any(np.isnan(result_interp)))
        np.testing.assert_allclose(
            result_smooth[:-1],
            result_interp[:-1],
            **allclose_tolerances(result_smooth[:-1], 1e-6),
            # FIXME
            #  this tolerance is so low because of the GPU
            #  backend. Reason unknown for now.
            #  Use `test_kick_interpolated_bug` to resolve this issue.
        )

    @pytest.mark.backend_mutation
    def test_interp_kick_multi_harmonic(self):
        beam = ProbeBeam(
            particle_type=lead_82,
            dt=np.linspace(0, 1, 100, dtype=backend.float),
            reference_total_energy=1e12,
        )
        rf_station_smooth = MultiHarmonicRFStation.headless(
            section_index=0,
            voltage=np.array([1e3, 2e3]),
            phi_rf=np.array([np.deg2rad(33), np.deg2rad(63)]),
            harmonic=np.array([5, 7.3]),
            circumference=123,
            main_harmonic_idx=1,
            beam_reference_beta=beam.reference.beta,
            local_wakefield=None,
            cavity_feedback=None,
            delayed_kick=None,
            delayed_kick_time_axis=None,
        )
        beam1_smooth = deepcopy(beam)
        rf_station_smooth.track(beam1_smooth)
        result_smooth = beam1_smooth.dE.copy_as_numpy()

        pool = PooledInterpolationKick()
        beam_interp = deepcopy(beam)
        rf_station_interp = MultiHarmonicRFStation.headless(
            section_index=0,
            voltage=np.array([1e3, 2e3]),
            phi_rf=np.array([np.deg2rad(33), np.deg2rad(63)]),
            harmonic=np.array([5, 7.3]),
            circumference=123,
            main_harmonic_idx=1,
            beam_reference_beta=beam.reference.beta,
            local_wakefield=None,
            cavity_feedback=None,
            delayed_kick=pool,
            delayed_kick_time_axis=beam_interp.dt.copy_as_numpy(),
        )
        rf_station_interp.track(beam_interp)
        pool.track(beam_interp)

        result_interp = beam_interp.dE.copy_as_numpy()
        offset = 2 * np.min(result_smooth)
        result_smooth += offset
        result_interp += offset
        DEV_PLOT = False
        if DEV_PLOT:
            plt.plot(result_smooth, "o", label="result_smooth")
            plt.plot(result_interp, "x", label="result_interp")
            plt.show()
        self.assertTrue(not np.any(np.isnan(result_smooth)))
        self.assertTrue(not np.any(np.isnan(result_interp)))

        np.testing.assert_allclose(
            result_smooth[:-1],
            result_interp[:-1],
            **allclose_tolerances(result_smooth[:-1], 1e-6),
            # FIXME
            #  this tolerance is so low because of the GPU
            #  backend. Reason unknown for now.
            #  Use `test_kick_interpolated_bug` to resolve this issue.
        )

    @multi_backend_testcase("Numpy64Bit", "Cupy64Bit")
    @pytest.mark.backend_mutation
    def test_compare_track_ham(self):
        """The tracker's dE change must equal ``-dH/d(dt)`` from
        ``get_hamilton_symbolic`` — by Hamilton's equation,
        ``Delta dE = -dH/d(dt)``.

        Exercised both with ``reference_energy_change == 0`` and with a
        non-zero acceleration to cover the ``Delta E_ref * dt`` term in
        the multi-harmonic Hamiltonian.
        """

        dt_s, q_s = sympy.symbols("dt q", real=True)

        for ref_energy_change in (-1e5, 0.0, 1e5):
            with self.subTest(reference_energy_change=ref_energy_change):
                beam = ProbeBeam(
                    dt=np.linspace(-5e-10, 5e-10, 11),
                    particle_type=proton,
                    reference_total_energy=1e9,
                )
                rf = MultiHarmonicRFStation.headless(
                    section_index=0,
                    voltage=np.array([1e6, 5e5]),
                    phi_rf=np.array([np.pi * 0.3, np.pi * 0.7]),
                    harmonic=np.array([10, 30]),
                    main_harmonic_idx=0,
                    circumference=2 * np.pi * 100.0,
                    magnetic_cycle=ConstantMagneticCycle(
                        reference_particle=proton,
                        value=beam.reference.total_energy + ref_energy_change,
                        in_unit="total energy",
                    ),
                    beam_reference_beta=beam.reference.beta,
                    local_wakefield=None,
                    cavity_feedback=None,
                    delayed_kick=None,
                    delayed_kick_time_axis=None,
                )
                dE_before = beam.dE.copy_as_numpy()
                dt_values = beam.dt.copy_as_numpy()

                rf.track(beam=beam)
                actual = beam.dE.copy_as_numpy() - dE_before

                dH_ddt = sympy.lambdify(
                    (dt_s, q_s),
                    sympy.diff(rf.get_hamilton_symbolic(), dt_s),
                    modules="numpy",
                )
                predicted = -dH_ddt(
                    dt_values, float(beam.signed_charge_with_direction())
                )

                np.testing.assert_allclose(actual, predicted, rtol=1e-12)

    def test_get_hamilton_symbolic_replace_symbols_false_keeps_rf_symbols(
        self,
    ):
        """With ``replace_symbols=False`` every harmonic's voltage, RF
        angular frequency and phase must stay as the free symbols
        ``V_j``, ``omega_j`` and ``phi_j``; resubstituting their numeric
        values must reproduce the ``replace_symbols=True`` Hamiltonian.
        """
        beam = ProbeBeam(
            dt=np.linspace(-5e-10, 5e-10, 11),
            particle_type=proton,
            reference_total_energy=1e9,
        )
        rf = MultiHarmonicRFStation.headless(
            section_index=0,
            voltage=np.array([1e6, 5e5]),
            phi_rf=np.array([np.pi * 0.3, np.pi * 0.7]),
            harmonic=np.array([10, 30]),
            main_harmonic_idx=0,
            circumference=2 * np.pi * 100.0,
            beam_reference_beta=beam.reference.beta,
            local_wakefield=None,
            cavity_feedback=None,
            delayed_kick=None,
            delayed_kick_time_axis=None,
        )
        # Populate ``design_energy_gain``, the tilt the Hamiltonian uses.
        rf.track(beam=beam)

        ham_sym = rf.get_hamilton_symbolic(replace_symbols=False)
        free_names = {s.name for s in ham_sym.free_symbols}

        substitutions = {}
        for rf_idx in range(rf.n_rf):
            for name in (f"V_{rf_idx}", f"omega_{rf_idx}", f"phi_{rf_idx}"):
                self.assertIn(name, free_names)
            substitutions[sympy.Symbol(f"V_{rf_idx}")] = float(
                rf.voltage[rf_idx]
            )
            substitutions[sympy.Symbol(f"omega_{rf_idx}", positive=True)] = (
                float(rf.omega_rf_design[rf_idx])
            )
            substitutions[sympy.Symbol(f"phi_{rf_idx}", real=True)] = float(
                rf.phi_rf_design[rf_idx]
            )

        ham_num = rf.get_hamilton_symbolic(replace_symbols=True)
        self.assertEqual(
            sympy.simplify(ham_sym.subs(substitutions) - ham_num), 0
        )


class TestSingleHarmonicRFStation(unittest.TestCase):
    def setUp(self) -> None:
        from blond.core.beam.base import BeamBaseClass

        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        # `ReferenceCoordinates` carries the pending-gain ledger that
        # reframing elements fill and the station consumes; a spec'd
        # Mock hands back a child Mock, which is not summable.
        beam.reference.pending_rf_energy_gain = 0.0
        beam.reference.take_pending_rf_energy_gain.return_value = 0.0
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
        beam.signed_charge_with_direction.return_value = proton._charge

        self.beam = beam

        self.single_harmonic_cavity = SingleHarmonicRFStation.headless(
            section_index=0,
            voltage=1e6,
            phi_rf=np.pi * 0.3,
            harmonic=3.5,
            circumference=456,
            local_wakefield=None,
            cavity_feedback=None,
            magnetic_cycle=_fixed_total_energy_cycle(939.0),
            beam_reference_beta=beam.reference.beta,
        )
        self.single_harmonic_cavity._ring.section_lengths = [1, 2, 3]

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_delayed_kick_with_feedback_requires_time_axis(self):
        cavity_feedback = Mock(spec=LocalFeedback)
        with self.assertRaises(AssertionError):
            SingleHarmonicRFStation.headless(
                section_index=0,
                voltage=1e6,
                phi_rf=0.1 * np.pi,
                harmonic=1,
                circumference=456,
                local_wakefield=None,
                cavity_feedback=cavity_feedback,  # trigger
                beam_reference_beta=1,
                delayed_kick=Mock(PooledInterpolationKick),  # trigger
                # trigger because it shouldn't coexist with cavity feedback
                delayed_kick_time_axis=np.linspace(1, 2, 10),
            )

    def test_delayed_kick_without_feedback_list_requires_time_axis2(self):
        with self.assertRaises(
            AssertionError,
        ):
            self.multi_harmonic_cavity = SingleHarmonicRFStation.headless(
                section_index=0,
                voltage=1e6,
                phi_rf=0.1,
                harmonic=5,
                circumference=456,
                local_wakefield=None,
                cavity_feedback=None,
                beam_reference_beta=1,
                delayed_kick=Mock(PooledInterpolationKick),
                delayed_kick_time_axis=None,  # trigger Exception
            )

    def test_delayed_kick_registers(self):
        delayed_kick_mock = Mock(PooledInterpolationKick)
        multi_harmonic_cavity = SingleHarmonicRFStation.headless(
            section_index=0,
            voltage=1e6,
            phi_rf=0.1,
            harmonic=5,
            circumference=456,
            local_wakefield=None,
            cavity_feedback=None,
            beam_reference_beta=1,
            delayed_kick=delayed_kick_mock,
            delayed_kick_time_axis=np.linspace(1, 2, 10),  # trigger Exception
        )
        multi_harmonic_cavity.track(self.beam)
        self.assertTrue(delayed_kick_mock.register.called)

    def test_interpolated_kick_warns_once_outside_the_profile(self):
        """Particles outside the interpolation window must be reported.

        ``kick_interpolated`` only kicks a particle whose bin index lands in
        ``[0, n_slices - 1)``.  Everything outside that window is skipped --
        and because ``acceleration_kick`` is folded into the same per-bin
        term, such a particle receives neither the RF voltage nor the
        design energy gain, so it silently decouples from the ramp.  That
        behaviour is deliberate (the exact kick is only available on the
        non-interpolated path), but it must not be silent: the station warns
        the first time it happens, once, naming how many particles are
        affected.
        """
        # The fixture's beam spans dt = -1e-6 .. +1e-6 s.  A window that
        # covers only the positive half leaves half the beam outside.
        time_axis = np.linspace(0.0, 1e-6, 8)
        voltage = np.zeros_like(time_axis)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.single_harmonic_cavity._track_interp(
                beam=self.beam,
                reference_energy_change=0.0,
                time_axis=time_axis,
                voltage=voltage,
            )
        offending = [
            str(entry.message)
            for entry in caught
            if "outside the interpolation window" in str(entry.message)
        ]
        self.assertEqual(
            len(offending), 1, msg=f"{[str(c.message) for c in caught]}"
        )
        # Actionable: say how many, and what it costs them.
        self.assertIn("acceleration", offending[0])

        # Reported ONCE: a per-turn repeat would drown the log.
        with warnings.catch_warnings(record=True) as caught_again:
            warnings.simplefilter("always")
            self.single_harmonic_cavity._track_interp(
                beam=self.beam,
                reference_energy_change=0.0,
                time_axis=time_axis,
                voltage=voltage,
            )
        self.assertEqual(
            [
                str(entry.message)
                for entry in caught_again
                if "outside the interpolation window" in str(entry.message)
            ],
            [],
        )

    def test_interpolated_kick_silent_when_beam_fits(self):
        """A beam wholly inside the window must not warn."""
        time_axis = np.linspace(-2e-6, 2e-6, 8)
        voltage = np.zeros_like(time_axis)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.single_harmonic_cavity._track_interp(
                beam=self.beam,
                reference_energy_change=0.0,
                time_axis=time_axis,
                voltage=voltage,
            )
        self.assertEqual(
            [
                str(entry.message)
                for entry in caught
                if "outside the interpolation window" in str(entry.message)
            ],
            [],
        )

    def test_delayed_kick_with_feedback(self):
        delayed_kick_mock = Mock(PooledInterpolationKick)
        cavity_feedback = Mock(spec=LocalFeedback)
        cavity_feedback.profile = Mock(spec=StaticProfile)
        cavity_feedback.profile.n_bins = 10
        cavity_feedback.profile.hist_x = np.linspace(1, 2, 10)
        cavity_feedback.relative_voltage_correction = np.linspace(1, 2, 10)
        cavity_feedback.phase_correction = np.linspace(1, 2, 10)
        multi_harmonic_cavity = SingleHarmonicRFStation.headless(
            section_index=0,
            voltage=1e6,
            phi_rf=0.1,
            harmonic=5,
            circumference=456,
            local_wakefield=None,
            cavity_feedback=cavity_feedback,
            beam_reference_beta=1,
            delayed_kick=delayed_kick_mock,
            delayed_kick_time_axis=None,
        )
        multi_harmonic_cavity.track(self.beam)
        self.assertTrue(delayed_kick_mock.register.called)

    def test_track(self) -> None:
        self.single_harmonic_cavity.track(beam=self.beam)

        self.assertEqual(939, self.beam.reference.total_energy)  # incremented
        self.assertEqual(self.beam.reference.time, 0)  # unchanged

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

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
            rtol=1e-12,
        )

        np.testing.assert_allclose(  # unchanged
            copy_to_cpu(self.beam.dt),
            np.linspace(-1e-6, 1e-6, 10),
        )

    def test_general_getters(self) -> None:
        self.single_harmonic_cavity._update_reference_based_attributes(
            reference=self.beam.reference
        )
        # TODO: Maybe better to use self.assertEqual here instead
        assert (
            self.single_harmonic_cavity.get_main_harmonic()
            == self.single_harmonic_cavity.harmonic
        )
        assert (
            self.single_harmonic_cavity.get_main_harmonic_t_rf()
            == 2 * np.pi / self.single_harmonic_cavity.omega_rf
        )
        assert (
            self.single_harmonic_cavity.get_main_harmonic_t_rf()
            == 2
            * np.pi
            / self.single_harmonic_cavity.get_main_harmonic_omega_rf_design()
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
        simulation.turn_counter = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.single_harmonic_cavity.voltage = None
            self.single_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_on_init_simulation_fails2(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_counter = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.single_harmonic_cavity.phi_rf_design = None
            self.single_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_on_init_simulation_fails3(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_counter = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.single_harmonic_cavity.harmonic = None
            self.single_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_calculate_synchronous_phase_with_synchrotron_radiation(self):
        radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.71368060083e-11,
            ]
        )
        shc = SingleHarmonicRFStation(
            section_index=0,
            harmonic=242400,
            voltage=51e6,
            phi_rf=0,
        )

        SR_ring = Ring(
            90.65874532 * 1e3,
            radiation_integrals=radiation_integrals,
        )
        drift1 = DriftSimple(
            orbit_length=90.65874532 * 1e3,
        )
        drift1.momentum_compaction_factor = 0.646747216157 / 90.65874532 / 1e3
        SR_ring.add_elements([drift1, shc])
        simulation = Simulation(
            SR_ring,
            MagneticCyclePerTurn(
                value_init=20e9,
                values_after_turn=np.linspace(20e9, 20e9, 1),
                reference_particle=positron,
                in_unit="total energy",
            ),
        )
        shc.on_init_simulation(simulation=simulation)

        positron_beam = Mock(BeamBaseClass)
        positron_beam.reference = Mock(ReferenceCoordinates)
        # `ReferenceCoordinates` carries the pending-gain ledger that
        # reframing elements fill and the station consumes; a spec'd
        # Mock hands back a child Mock, which is not summable.
        positron_beam.reference.pending_rf_energy_gain = 0.0
        positron_beam.reference.take_pending_rf_energy_gain.return_value = 0.0

        positron_beam.particle_type = positron
        # Co-rotating: direction-signed charge equals the raw charge.
        positron_beam.signed_charge_with_direction.return_value = (
            positron.charge
        )
        positron_beam.reference.total_energy = 20e9
        positron_beam.reference.time = 0
        positron_beam.reference.gamma = 40000

        phi_s_calculated = shc.calc_phi_s_main_harmonic(beam=positron_beam)
        energy_loss_per_turn = calculate_energy_loss_per_turn(
            energy=20e9,
            radiation_integrals=radiation_integrals,
            particle_type=positron,
        )
        expected_energy_change = +energy_loss_per_turn
        expected_phi_s = np.pi - np.arcsin(expected_energy_change / 51e6)
        self.assertEqual(phi_s_calculated, expected_phi_s)

    @multi_backend_testcase("Numpy64Bit", "Cupy64Bit")
    @pytest.mark.backend_mutation
    def test_compare_track_ham(self):
        """The tracker's dE change must equal ``-dH/d(dt)`` from
        ``get_hamilton_symbolic`` — by Hamilton's equation,
        ``Delta dE = -dH/d(dt)``.

        We exercise both ``reference_energy_change == 0`` and
        ``reference_energy_change != 0`` to cover the acceleration term
        ``Delta E_ref * dt`` in the Hamiltonian.
        """
        import sympy

        from blond.core.beam.particle_types import proton

        dt_s, q_s = sympy.symbols("dt q", real=True)

        for ref_energy_change in (-1e5, 0.0, 1e5):
            with self.subTest(reference_energy_change=ref_energy_change):
                beam = ProbeBeam(
                    dt=np.linspace(-5e-10, 5e-10, 11),
                    particle_type=proton,
                    reference_total_energy=1e9,
                )
                rf = SingleHarmonicRFStation.headless(
                    section_index=0,
                    voltage=1e6,
                    phi_rf=np.pi * 0.3,
                    harmonic=10,
                    circumference=2 * np.pi * 100.0,
                    magnetic_cycle=ConstantMagneticCycle(
                        reference_particle=proton,
                        value=beam.reference.total_energy + ref_energy_change,
                        in_unit="total energy",
                    ),
                    beam_reference_beta=beam.reference.beta,
                    local_wakefield=None,
                    cavity_feedback=None,
                    delayed_kick=None,
                    delayed_kick_time_axis=None,
                )
                dE_before = beam.dE.copy_as_numpy()
                dt_values = beam.dt.copy_as_numpy()

                rf.track(beam=beam)
                actual = beam.dE.copy_as_numpy() - dE_before

                dH_ddt = sympy.lambdify(
                    (dt_s, q_s),
                    sympy.diff(rf.get_hamilton_symbolic(), dt_s),
                    modules="numpy",
                )
                predicted = -dH_ddt(
                    dt_values, float(beam.signed_charge_with_direction())
                )

                np.testing.assert_allclose(actual, predicted, rtol=1e-12)

    def test_get_hamilton_symbolic_replace_symbols_false_keeps_rf_symbols(
        self,
    ):
        """With ``replace_symbols=False`` the voltage, RF angular
        frequency and phase must stay as the free symbols ``V``,
        ``omega_rf`` and ``phi_rf``; resubstituting their numeric values
        must reproduce the ``replace_symbols=True`` Hamiltonian.
        """
        beam = ProbeBeam(
            dt=np.linspace(-5e-10, 5e-10, 11),
            particle_type=proton,
            reference_total_energy=1e9,
        )
        rf = SingleHarmonicRFStation.headless(
            section_index=0,
            voltage=1e6,
            phi_rf=np.pi * 0.3,
            harmonic=10,
            circumference=2 * np.pi * 100.0,
            beam_reference_beta=beam.reference.beta,
            local_wakefield=None,
            cavity_feedback=None,
            delayed_kick=None,
            delayed_kick_time_axis=None,
        )
        # Populate ``design_energy_gain``, the tilt the Hamiltonian uses.
        rf.track(beam=beam)

        ham_sym = rf.get_hamilton_symbolic(replace_symbols=False)
        free_names = {s.name for s in ham_sym.free_symbols}
        for name in ("V", "omega_rf", "phi_rf"):
            self.assertIn(name, free_names)

        ham_num = rf.get_hamilton_symbolic(replace_symbols=True)
        resubstituted = ham_sym.subs(
            {
                sympy.Symbol("V"): float(rf.voltage),
                sympy.Symbol("omega_rf", positive=True): float(
                    rf.omega_rf_design
                ),
                sympy.Symbol("phi_rf", real=True): float(rf.phi_rf_design),
            }
        )
        self.assertEqual(sympy.simplify(resubstituted - ham_num), 0)


class TestCavityFeedbackSparseProfileIntegration(unittest.TestCase):
    @pytest.mark.skip
    @pytest.mark.backend_mutation
    @multi_backend_testcase("Numpy64Bit")
    def test_cavity_feedback_kick_with_gapped_filling_pattern_does_not_raise(
        self,
    ):
        # Real (non-mocked) `SingleHarmonicRFStation` with a real
        # `IQCavityFeedback` whose `.profile` is an
        # `EquidistantMultiProfile` with a genuine internal gap
        # (fill/fill/empty/empty, repeated -- not merely a shorter
        # contiguous run followed by trailing zeros). This exercises the
        # `any_feedback_not_none` branch of `_track`, i.e.
        # `time_axis = self.cavity_feedback_list[0].profile.hist_x`
        # followed by `_track_interp`, which previously passed the gapped
        # (non-uniform) `hist_x` straight into `kick_interpolated` with no
        # sparse metadata.
        #
        # Uses `multi_backend_testcase` (rather than a manual
        # `backend.change_backend(...)`) so the original active
        # backend is restored after the test, instead of leaking into
        # whichever test happens to run next.
        sync_momentum = 25.92e9  # [eV / c]
        harmonic = 4620

        ring = Ring(circumference=6911.56)
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=sync_momentum,
            in_unit="momentum",
        )
        _bunch = Beam(intensity=1e10, particle_type=proton)
        drift = DriftSimple(
            momentum_compaction_factor=0.05,
            orbit_length=ring.circumference,
        )

        # Genuine internal gap: two filled buckets, two empty buckets,
        # repeated across the whole ring -- as opposed to one contiguous
        # run of filled buckets followed by zero-padding at the end.
        bucket_index = np.arange(harmonic)
        filling_pattern = (bucket_index % 4) < 2

        profile = EquidistantMultiProfile(
            filling_pattern=filling_pattern,
            bins_per_profile=2**8,
            offset=0,
        )
        cavity_feedback = IQCavityFeedbackTimingClass(
            profile=profile,
            n_cavities=1,
            R_over_Q=1,
            Q_L=1,
            generator_current_bias=1,
            n_rf_periods_per_coarse_grid=1,
            harmonic_index=0,
        )

        rf_station = SingleHarmonicRFStation(
            harmonic=harmonic,
            voltage=0.9e6,
            phi_rf=0.31,
            cavity_feedback=cavity_feedback,
        )

        t_rf = (
            magnetic_cycle.get_t_rev_init(
                ring.circumference,
                particle_type=proton,
            )
            / harmonic
        )

        ring.add_elements((profile, rf_station, drift))
        sim = Simulation(
            ring=ring,
            magnetic_cycle=magnetic_cycle,
        )
        # `profile.hist_x` is only allocated once `on_init_simulation`
        # has run (triggered above by constructing `Simulation`); set
        # the feedback's correction arrays on the real, gapped time
        # axis rather than on the pre-init `None`.
        cavity_feedback.phase_correction = np.zeros_like(profile.hist_x)
        cavity_feedback.relative_voltage_correction = np.ones_like(
            profile.hist_x
        )

        sim.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=2e-9 / 4,
                seed=1,
                n_macroparticles=1e4,
            ),
            beam=_bunch,
        )

        beam = make_multibunch_beam(
            beam=_bunch,
            n_times=int(harmonic // 10),
            t_distance=t_rf * 10,
        )
        dE_before = beam.dE.copy_as_numpy()

        drift.orbit_length = 0.0
        sim.check_circumference = "ignore"
        sim.run_simulation(beams=beam, n_turns=1)

        dE_after = beam.dE.copy_as_numpy()
        self.assertFalse(
            np.allclose(dE_before, dE_after),
            "Expected the sparse-aware cavity feedback kick to change "
            "dE, but dE was unchanged.",
        )


class TestCounterRotatingSynchronousPhase(unittest.TestCase):
    """Analytic ``phi_s`` must respect the collider's CR charge symmetry.

    Every acceleration kick in ``cavities.py`` uses
    ``beam.signed_charge_with_direction()``, which flips the sign of the
    charge for a counter-rotating beam. A counter-rotating ``mu-`` beam is
    therefore accelerated with effective charge ``+1`` -- identical to a
    co-rotating ``mu+`` beam. The analytic synchronous phase must use the
    same direction-signed charge so that it mirrors the kick, otherwise the
    CR beam's ``phi_s`` disagrees with the ``mu+`` beam it should match.
    """

    def _build_accelerating_case(self, particle_type, is_counter_rotating):
        """Return ``(rf_station, ring, cycle, beam)`` for an accelerating ring.

        A single-section ring is used so the counter-rotating section index
        (``len(section_lengths) - section_index - 1``) coincides with the
        co-rotating one and both directions target the same energy program.
        The gentle ramp keeps ``energy_gain / (voltage * charge)`` inside the
        ``arcsin`` domain and the high energy keeps the beam above transition.
        """
        n_turns = 10000
        ring = Ring(circumference=20e3)
        rf_station = SingleHarmonicRFStation(
            voltage=1e6, harmonic=10, phi_rf=np.deg2rad(-90)
        )
        drift = DriftSimple(
            orbit_length=ring.circumference,
            momentum_compaction_factor=1.0,  # positive -> above transition
        )
        energy_init = particle_type.mass + 1e12
        ramp = np.linspace(energy_init, 1.005 * energy_init, n_turns + 1)
        cycle = MagneticCyclePerTurn(
            reference_particle=particle_type,
            value_init=float(ramp[0]),
            values_after_turn=ramp[1:],
            in_unit="total energy",
        )
        ring.add_elements((rf_station, drift), reorder=False, section_index=0)
        simulation = Simulation(ring=ring, magnetic_cycle=cycle)
        _ = simulation  # wires _ring / _magnetic_cycle / _turn_counter
        beam = Beam(
            intensity=12,
            particle_type=particle_type,
            is_counter_rotating=is_counter_rotating,
        )
        beam.setup_beam(
            dt=np.array([0.0]),
            dE=np.array([0.0]),
            reference_total_energy=float(ramp[0]),
        )
        return rf_station, ring, cycle, beam

    def test_cr_symmetry_phi_s_matches_co_rotating(self):
        # Co-rotating mu+ and counter-rotating mu- see the same effective
        # charge (+1) in the kick, so their analytic phi_s must be equal.
        rf_co, _, _, beam_co = self._build_accelerating_case(
            mu_plus, is_counter_rotating=False
        )
        rf_cr, _, _, beam_cr = self._build_accelerating_case(
            mu_minus, is_counter_rotating=True
        )

        phi_s_co = rf_co.calc_phi_s_main_harmonic(beam=beam_co)
        phi_s_cr = rf_cr.calc_phi_s_main_harmonic(beam=beam_cr)

        self.assertTrue(np.isfinite(phi_s_co))
        self.assertTrue(np.isfinite(phi_s_cr))
        # The pre-fix code passes the raw (unsigned) charge -1 for the CR
        # beam, which lands on the wrong arcsin branch: this assertion is
        # the RED step and only passes once both call sites use
        # ``signed_charge_with_direction()``.
        np.testing.assert_allclose(phi_s_cr, phi_s_co, rtol=0, atol=0)

    def test_co_rotating_phi_s_unchanged(self):
        # For a co-rotating beam signed charge == raw charge, so switching
        # the call site to the direction-signed charge must leave the value
        # bit-identical. Reconstruct the pre-fix computation with the raw
        # charge and confirm the method still reproduces it.
        rf_co, ring, cycle, beam_co = self._build_accelerating_case(
            mu_plus, is_counter_rotating=False
        )
        phi_s_co = rf_co.calc_phi_s_main_harmonic(beam=beam_co)

        self.assertEqual(
            beam_co.signed_charge_with_direction(),
            beam_co.particle_type.charge,
        )

        turn_i = (
            rf_co._turn_counter.value
            if rf_co._turn_counter is not None
            else None
        )
        target_total_energy = cycle.get_target_total_energy(
            turn_i=turn_i,
            section_i=0,
            reference_time=float(beam_co.reference.time),
            particle_type=beam_co.particle_type,
        )
        energy_gain = target_total_energy - beam_co.reference.total_energy
        phi_s_raw = calc_phi_s_single_harmonic(
            charge=beam_co.particle_type.charge,  # raw, pre-fix behaviour
            voltage=float(rf_co.get_main_harmonic_voltage()),
            energy_gain=energy_gain,
            above_transition=not ring.is_below_transition(beam=beam_co),
        )
        np.testing.assert_allclose(phi_s_co, phi_s_raw, rtol=0, atol=0)

    def test_synchrotron_tune_sign_robust(self):
        # The tune uses ``np.abs(charge)`` and ``np.abs(eta_0 * cos(phi_s))``,
        # so the raw-vs-signed charge (and the phi_s branch flip it induces)
        # cancels: the co- and counter-rotating tunes are identical. This is
        # the invariant the fix must preserve.
        rf_co, _, _, beam_co = self._build_accelerating_case(
            mu_plus, is_counter_rotating=False
        )
        rf_cr, _, _, beam_cr = self._build_accelerating_case(
            mu_minus, is_counter_rotating=True
        )

        tune_co = rf_co.calc_synchrotron_tune_main_harmonic(beam=beam_co)
        tune_cr = rf_cr.calc_synchrotron_tune_main_harmonic(beam=beam_cr)

        self.assertTrue(np.isfinite(tune_co))
        np.testing.assert_allclose(tune_cr, tune_co, rtol=1e-12, atol=0)


if __name__ == "__main__":
    unittest.main()
