import json
import os
import unittest
import warnings
from collections import deque
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c, e
from scipy.fft import next_fast_len

from blond import (
    Beam,
    ConstantMagneticCycle,
    Cupy64Bit,
    Numpy64Bit,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    WakeField,
    mu_plus,
    uranium_29,
)
from blond.core.backends.backend import backend
from blond.core.beam.base import BeamBaseClass
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.generals.cupy.no_cupy_import import is_cupy_array
from blond.handle_results.helpers import callers_relative_path
from blond.physics.impedances.solvers import (
    ContinuousMultiTurnTimeDomainSolver,
    InductiveImpedance,
    InductiveImpedanceSolver,
    MultiPassResonatorSolver,
    PeriodicFreqSolver,
    SingleTurnResonatorConvolutionSolver,
    TimeDomainFftSolver,
)
from blond.physics.impedances.sources import ImpedanceTableFreq, Resonators
from blond.physics.profiles import (
    DynamicProfileConstCutoff,
    DynamicProfileConstNBins,
    StaticProfile,
)


class TestTimeDomainFftSolver(unittest.TestCase):
    def setUp(self):
        self.resonators = Resonators(
            shunt_impedances=np.array([1, 2, 3]),
            center_frequencies=np.array([500e6, 750e6, 1.5e9]),
            quality_factors=np.array([5, 5, 5]),
        )
        self.time_domain_fft_solver = TimeDomainFftSolver()
        self.left_edge, self.right_edge, self.hist_step = -2e-9, 1e-9, 0.01e-10
        self.hist_x = np.linspace(
            self.left_edge,
            self.right_edge,
            int(np.round((self.right_edge - self.left_edge) / self.hist_step))
            + 1,
            endpoint=True,
        )

        self.time_domain_fft_solver._parent_wakefield = Mock(WakeField)
        self.time_domain_fft_solver._parent_wakefield.profile = Mock(
            spec=StaticProfile
        )
        self.time_domain_fft_solver._parent_wakefield.profile.hist_step = (
            self.hist_step
        )
        self.time_domain_fft_solver._parent_wakefield.profile.hist_x = (
            self.hist_x
        )

        profile = np.zeros_like(
            self.time_domain_fft_solver._parent_wakefield.profile.hist_x
        )
        profile[9:12] = 1  # symmetric profile around centerpoint
        profile /= np.sum(profile)
        self.time_domain_fft_solver._parent_wakefield.profile.hist_y = profile

        self.time_domain_fft_solver._parent_wakefield.sources = (
            self.resonators,
        )

        self.beam = Mock(BeamBaseClass)

        self.beam.intensity = int(1e9)
        self.beam.particle_type.charge = 1
        self.beam.n_macroparticles_partial.return_value = int(1e3)
        self.beam.ratio = (
            self.beam.intensity / self.beam.n_macroparticles_partial()
        )

        self.time_domain_fft_solver._parent_wakefield.profile.hist_y_to_density_factor = (
            1 / self.beam.n_macroparticles_partial()
        )

        self.time_domain_fft_solver._parent_wakefield.profile.beam_spectrum.return_value = np.fft.rfft(
            self.time_domain_fft_solver._parent_wakefield.profile.hist_y,
            n=next_fast_len(
                len(
                    self.time_domain_fft_solver._parent_wakefield.profile.hist_y
                )
                * 2
            ),
        )

    @unittest.skip
    def test__ind_voltage_calculation(self):
        self.time_domain_fft_solver._wake_imp_y_needs_update = True
        ind_volt = self.time_domain_fft_solver.calc_induced_voltage(
            beam=self.beam
        )

        assert len(ind_volt) == len(
            self.time_domain_fft_solver._parent_wakefield.profile.hist_y
        )

    def test_error_throwing_warning_throwing(self):
        local_solver = deepcopy(self.time_domain_fft_solver)
        local_solver._parent_wakefield.sources = (ImpedanceTableFreq,)

        with self.assertRaisesRegex(
            Exception, "Can only accept impedance that support"
        ):
            local_solver._update_impedance_sources(beam=self.beam)

        # local_solver._parent_wakefield.sources = (self.resonators,)
        # local_solver._update_impedance_sources(beam=self.beam)
        # local_solver._wake_imp_y = np.array([0])
        # local_solver._update_impedance_sources(beam=self.beam)
        # assert local_solver._wake_imp_y == np.array([0])  # check that nothing gets changed without flag
        #
        # local_solver._wake_imp_y_needs_update = True
        # local_solver._wake_imp_y = np.ones_like(local_solver._parent_wakefield.profile.hist_x, dtype=complex)
        # local_solver._update_impedance_sources(beam=self.beam)
        # assert np.sum(local_solver._wake_imp_y) != 0

    def test_on_wakefield_init_simulation_error_throwing(self):
        simulation = Mock(Simulation)
        parent_wakefield = Mock(WakeField)
        profile = Mock(DynamicProfileConstNBins)
        profile.n_bins = 10
        parent_wakefield.profile = profile
        parent_wakefield.profile.hist_step = 1
        resonators = Mock(Resonators)
        resonators.is_dynamic = False
        parent_wakefield.sources = (resonators,)
        resonators.get_impedance.return_value = np.linspace(1, 2, 6)

        with warnings.catch_warnings(record=True) as w:
            self.time_domain_fft_solver.expect_profile_change = False
            self.time_domain_fft_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )
            self.assertIn("Because you are using a", str(w[0].message))

        with self.assertRaisesRegex(NotImplementedError, "Unrecognized type"):
            profile = Mock(BeamBaseClass)
            parent_wakefield.profile = profile
            parent_wakefield.profile.hist_step = 1
            parent_wakefield.profile.n_bins = 10
            self.time_domain_fft_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )

        with self.assertRaises(Exception):
            parent_wakefield.profile = None
            self.time_domain_fft_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )

    def test_dynamic_profile_integration(self):
        beam = Beam(
            intensity=21,
            particle_type=uranium_29,
        )
        cavity = SingleHarmonicRfStation(
            harmonic=1,
            voltage=0,
            phi_rf=0,
        )
        rng = np.random.default_rng()
        dt = rng.standard_normal(1000)

        # truncate and shift center to 1
        dt[dt > 1] = 0
        dt[dt < -1] = 0
        dt += 1

        beam.setup_beam(dt=dt, dE=np.linspace(0, 1, 1000))
        profile = DynamicProfileConstNBins(n_bins=200)
        profile.update_attributes(beam=beam)
        wf = WakeField(
            sources=(
                Resonators(
                    shunt_impedances=np.array([1, 2, 3]),
                    center_frequencies=np.array([500e6, 750e6, 1.5e9]),
                    quality_factors=np.array([5e10, 5, 5]),
                ),
            ),
            solver=TimeDomainFftSolver(),
            profile=profile,
        )
        ring = Ring(circumference=123)
        cycle = ConstantMagneticCycle(
            reference_particle=uranium_29,
            value=1e12,
        )
        sim = Simulation.from_locals(locals=locals())
        profile.track(beam=beam)

        self.time_domain_fft_solver._wake_imp_y_needs_update = True
        self.time_domain_fft_solver._wake_imp_y = np.ones_like(
            self.time_domain_fft_solver._parent_wakefield.profile.hist_x,
            dtype=complex,
        )
        self.time_domain_fft_solver._update_impedance_sources(beam=self.beam)
        assert np.sum(self.time_domain_fft_solver._wake_imp_y) != 0

        profile_a = profile.hist_y
        induced_voltage_a = wf.calc_induced_voltage(beam=beam)

        dt = beam.write_partial_dt()
        dt -= 1
        dt /= 2
        dt += 1
        profile.update_attributes(beam=beam)
        profile.track(beam=beam)
        profile_b = profile.hist_y
        induced_voltage_b = wf.calc_induced_voltage(beam=beam)

        DEV_PLOT = False
        if DEV_PLOT:
            plt.subplot(2, 1, 1)
            plt.plot(profile_a)
            plt.plot(profile_b)
            plt.subplot(2, 1, 2)
            plt.plot(induced_voltage_a)
            plt.plot(induced_voltage_b)
            plt.show()


class TestInductiveImpedanceSolver(unittest.TestCase):
    def setUp(self):
        self.inductive_impedance_solver = InductiveImpedanceSolver()
        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        beam.intensity = 1e12
        beam.n_macroparticles_partial.return_value = 128
        beam.particle_type.charge = 1
        beam.ratio = 1

        beam.reference.velocity = 123
        self.inductive_impedance_solver._beam = beam
        self.inductive_impedance_solver._Z_over_n = 12
        _parent_wakefield = Mock(WakeField)
        _parent_wakefield.profile.hist_step = 1
        _parent_wakefield.profile.hist_y_to_density_factor = beam.ratio

        self.inductive_impedance_solver._parent_wakefield = _parent_wakefield
        simulation = Mock(Simulation)
        simulation.ring.circumference = 123
        self.inductive_impedance_solver._simulation = simulation
        _parent_wakefield.profile.gradient_hist_y = np.linspace(1, 3)

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_calc_induced_voltage(self):
        self.inductive_impedance_solver.calc_induced_voltage(
            self.inductive_impedance_solver._beam
        )  # TODO Pin Physics case here!

    def test_on_wakefield_init_simulation(self):
        simulation = Mock(Simulation)
        simulation.turn_i = 0
        parent_wakefield = Mock(WakeField)
        indcutive_impedance = Mock(InductiveImpedance)
        indcutive_impedance.Z_over_n = 1
        parent_wakefield.sources = (indcutive_impedance,)
        self.inductive_impedance_solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=parent_wakefield
        )


class TestPeriodicFreqSolver(unittest.TestCase):
    def setUp(self):
        self.inductive_impedance = InductiveImpedance(
            Z_over_n=34.6669349520904 / 10e9 * 11e3
        )
        self.resonators = Resonators(
            shunt_impedances=np.array([500, 1e6, 1e9]),
            center_frequencies=np.array([400e6, 600e6, 1.2e9]),
            quality_factors=np.array([1, 2, 3]),
        )
        self.periodic_freq_solver = PeriodicFreqSolver(t_periodicity=10)

        self.periodic_freq_solver._parent_wakefield = Mock(WakeField)
        self.periodic_freq_solver._parent_wakefield.profile.beam_spectrum.return_value = np.linspace(
            0, 1, 6
        )
        self.periodic_freq_solver._parent_wakefield.profile.hist_step = 1
        self.periodic_freq_solver._parent_wakefield.profile.n_bins = 8

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test__update_internal_data(self):
        self.periodic_freq_solver._parent_wakefield.sources = (
            self.resonators,
        )
        self.periodic_freq_solver._update_internal_data()
        self.assertEqual(self.periodic_freq_solver._n_time, 10)

    def test__update_internal_data2(self):
        self.periodic_freq_solver._parent_wakefield.sources = (
            self.resonators,
        )
        self.periodic_freq_solver._parent_wakefield.profile.hist_step = 0.5e-9
        self.periodic_freq_solver.t_periodicity = 1e-8

    def _test_calc_induced_voltage(self, backend_class):
        from blond import backend

        backend.change_backend(backend_class)
        self.periodic_freq_solver._parent_wakefield.profile.beam_spectrum.return_value = backend.linspace(
            0, 1, 11
        )
        self.periodic_freq_solver._parent_wakefield.sources = (
            self.resonators,
        )
        self.periodic_freq_solver._parent_wakefield.profile.hist_step = 0.5e-9
        self.periodic_freq_solver._parent_wakefield.profile.n_bins = 20
        self.periodic_freq_solver.t_periodicity = 1e-8
        self.periodic_freq_solver._update_internal_data()
        beam = Mock(BeamBaseClass)
        beam.intensity = int(11e3)
        beam.n_macroparticles_partial.return_value = int(3e6)
        self.periodic_freq_solver._parent_wakefield.profile.hist_y_to_density_factor = (
            1 / beam.n_macroparticles_partial.return_value
        )
        beam.particle_type.charge = 1
        beam.ratio = 1e5
        induced_voltage = self.periodic_freq_solver.calc_induced_voltage(
            beam=beam,
        )
        induced_voltage = self.periodic_freq_solver.calc_induced_voltage(
            beam=beam,
        )
        if is_cupy_array(induced_voltage):
            induced_voltage = induced_voltage.get()
        pinned_values = np.load(
            callers_relative_path(
                "resources/induced_voltage_periodic_freq_solver.npz",
                stacklevel=1,
            )
        )["induced_voltage"]

        np.testing.assert_allclose(pinned_values, induced_voltage, rtol=1e-5)

        # np.savez(callers_relative_path("resources/induced_voltage_periodic_freq_solver.npz", stacklevel=1), induced_voltage=induced_voltage)
        DEV_PLOT = False
        if DEV_PLOT:
            plt.plot(induced_voltage)
            plt.show()

        assert self.periodic_freq_solver._freq_y_needs_update == False
        self.periodic_freq_solver._parent_wakefield.sources = (
            Mock(BeamBaseClass),
        )
        # if this were to be checked, it would error, but since
        # we are on _freq_y_needs_update == False, it skips
        self.periodic_freq_solver._update_impedance_sources(beam=beam)

        with self.assertRaises(Exception):
            self.periodic_freq_solver._freq_y_needs_update = True
            # update is now forced, which should force the error
            self.periodic_freq_solver._update_impedance_sources(beam=beam)

    def test_calc_induced_voltage_gpu(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))

        from blond import backend

        backend_org = type(backend)
        self._test_calc_induced_voltage(backend_class=Cupy64Bit)
        backend.change_backend(backend_org)

    def test_calc_induced_voltage_cpu(self):
        from blond import backend

        backend_org = type(backend)
        self._test_calc_induced_voltage(backend_class=Numpy64Bit)
        backend.change_backend(backend_org)

    def test_on_wakefield_init_simulation(self):
        simulation = Mock(Simulation)
        parent_wakefield = Mock(WakeField)
        profile = Mock(StaticProfile)
        profile.n_bins = 10
        parent_wakefield.profile = profile
        parent_wakefield.profile.hist_step = 1
        resonators = Mock(Resonators)
        resonators.is_dynamic = False
        parent_wakefield.sources = (resonators,)
        resonators.get_impedance.return_value = np.linspace(1, 2, 6)

        self.periodic_freq_solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=parent_wakefield
        )

    def test_on_wakefield_init_simulation_error_throwing(self):
        simulation = Mock(Simulation)
        parent_wakefield = Mock(WakeField)
        profile = Mock(DynamicProfileConstNBins)
        profile.n_bins = 10
        parent_wakefield.profile = profile
        parent_wakefield.profile.hist_step = 1
        resonators = Mock(Resonators)
        resonators.is_dynamic = False
        parent_wakefield.sources = (resonators,)
        resonators.get_impedance.return_value = np.linspace(1, 2, 6)

        with warnings.catch_warnings(record=True) as w:
            self.periodic_freq_solver.expect_profile_change = False
            self.periodic_freq_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )
            self.assertIn("Because you are using a", str(w[0].message))

        with self.assertRaisesRegex(NotImplementedError, "Unrecognized type"):
            profile = Mock(BeamBaseClass)
            parent_wakefield.profile = profile
            parent_wakefield.profile.hist_step = 1
            parent_wakefield.profile.n_bins = 10
            self.periodic_freq_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )

        with self.assertRaises(Exception):
            parent_wakefield.profile = None
            self.periodic_freq_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )

    def test_dynamic_profile_integration(self):
        beam = Beam(
            intensity=21,
            particle_type=uranium_29,
        )
        cavity = SingleHarmonicRfStation(
            harmonic=1,
            voltage=0,
            phi_rf=0,
        )

        rng = np.random.default_rng()
        dt = rng.standard_normal(1000)

        # truncate and shift center to 1
        dt[dt > 1] = 0
        dt[dt < -1] = 0
        dt += 1

        beam.setup_beam(dt=dt, dE=np.linspace(0, 1, 1000))
        profile = DynamicProfileConstCutoff(timestep=0.1)
        profile.update_attributes(beam=beam)
        wf = WakeField(
            sources=(
                Resonators(
                    shunt_impedances=np.array([1, 2, 3]),
                    center_frequencies=np.array([500e6, 750e6, 1.5e9]),
                    quality_factors=np.array([5e10, 5, 5]),
                ),
            ),
            solver=PeriodicFreqSolver(
                t_periodicity=4.0, allow_next_fast_len=True
            ),
            profile=profile,
        )
        ring = Ring(circumference=123)
        cycle = ConstantMagneticCycle(
            reference_particle=uranium_29,
            value=1e12,
        )
        sim = Simulation.from_locals(locals=locals())
        profile.track(beam=beam)

        induced_voltage_a = wf.calc_induced_voltage(beam=beam)

        dt = beam.write_partial_dt()
        dt -= 1
        dt /= 2
        dt += 1
        profile.update_attributes(beam=beam)
        profile.track(beam=beam)

        induced_voltage_b = wf.calc_induced_voltage(beam=beam)

        DEV_PLOT = False
        if DEV_PLOT:
            plt.plot(induced_voltage_a)
            plt.plot(induced_voltage_b)
            plt.show()


class TestAnalyticSingleTurnResonatorSolver(unittest.TestCase):
    def setUp(self):
        self.resonators = Resonators(
            shunt_impedances=np.array([1, 2, 3]),
            center_frequencies=np.array([500e6, 750e6, 1.5e9]),
            quality_factors=np.array([5, 5, 5]),
        )
        self.single_turn_resonator_convolution_solver = (
            SingleTurnResonatorConvolutionSolver()
        )
        self.left_edge, self.right_edge, self.hist_step = -2e-9, 1e-9, 1e-10
        self.hist_x = np.linspace(
            self.left_edge,
            self.right_edge,
            int(np.round((self.right_edge - self.left_edge) / self.hist_step))
            + 1,
            endpoint=True,
        )

        self.beam = Mock(BeamBaseClass)

        self.beam.intensity = int(1e9)
        self.beam.particle_type.charge = 1
        self.beam.n_macroparticles_partial.return_value = int(1e3)

        self.single_turn_resonator_convolution_solver._parent_wakefield = Mock(
            WakeField
        )
        self.single_turn_resonator_convolution_solver._parent_wakefield.profile = Mock(
            spec=StaticProfile
        )
        self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_step = self.hist_step
        self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_x = self.hist_x
        self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_y_to_density_factor = (
            1 / self.beam.n_macroparticles_partial()
        )

        profile = np.zeros_like(
            self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_x
        )
        profile[9:12] = 1  # symmetric profile around centerpoint
        profile /= np.sum(profile)
        self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_y = profile

        self.single_turn_resonator_convolution_solver._parent_wakefield.sources = (
            self.resonators,
        )

    def test_compare_with_fft_solver(self):
        analy_solver = deepcopy(self.single_turn_resonator_convolution_solver)
        left_edge, right_edge, hist_step = (
            -2e-9,
            1e-9,
            0.01e-10,
        )  # finer profile, otherwise FFT solver fails
        hist_x = np.linspace(
            left_edge,
            right_edge,
            int(np.round((right_edge - left_edge) / hist_step)) + 1,
            endpoint=True,
        )

        analy_solver._parent_wakefield.profile = Mock(spec=StaticProfile)
        analy_solver._parent_wakefield.profile.hist_step = hist_step
        analy_solver._parent_wakefield.profile.hist_x = hist_x

        profile = np.zeros_like(analy_solver._parent_wakefield.profile.hist_x)
        profile[9:12] = 1  # symmetric profile around centerpoint
        profile /= np.sum(profile)
        analy_solver._parent_wakefield.profile.hist_y = profile
        analy_solver._parent_wakefield.profile.hist_y_to_density_factor = (
            1 / self.beam.n_macroparticles_partial()
        )

        analy_solver._parent_wakefield.profile.hist_y_to_density_factor = (
            1 / self.beam.n_macroparticles_partial.return_value
        )

        analy_solver._update_potential_sources(zero_pinning=True)
        initial_wake_pot = analy_solver._wake_function_vals
        initial_wake_pot_time = analy_solver._wake_function_time
        assert len(initial_wake_pot) == len(initial_wake_pot_time)
        initial_voltage = analy_solver.calc_induced_voltage(beam=self.beam)

        td_fft_solver = TimeDomainFftSolver()
        td_fft_solver._parent_wakefield = Mock(WakeField)
        td_fft_solver._parent_wakefield.profile = Mock(StaticProfile)
        td_fft_solver._parent_wakefield.profile.hist_step = hist_step
        td_fft_solver._parent_wakefield.profile.hist_x = hist_x
        td_fft_solver._parent_wakefield.profile.hist_y_to_density_factor = (
            1 / self.beam.n_macroparticles_partial()
        )

        td_fft_solver._parent_wakefield.profile.hist_y = (
            analy_solver._parent_wakefield.profile.hist_y
        )
        td_fft_solver._parent_wakefield.profile.hist_y_to_density_factor = (
            1 / self.beam.n_macroparticles_partial.return_value
        )

        td_fft_solver._parent_wakefield.sources = (self.resonators,)

        td_fft_solver._parent_wakefield.profile.beam_spectrum.return_value = (
            np.fft.rfft(
                analy_solver._parent_wakefield.profile.hist_y,
                n=next_fast_len(
                    len(analy_solver._parent_wakefield.profile.hist_y) * 2
                ),
            )
        )

        td_solver = td_fft_solver.calc_induced_voltage(beam=self.beam)
        np.testing.assert_allclose(
            initial_voltage,
            td_solver[0 : len(initial_voltage)],
            atol=1e-10,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test__update_potential_sources_profile_changes_array_lengths(
        self,
    ):  # in principle, this is a test for the dynamic profile, currently not implemented
        """
        Ensure that the profile does not change on application of different profile lengths with 0-padding
        """
        self.single_turn_resonator_convolution_solver._update_potential_sources(
            zero_pinning=True
        )
        initial_wake_pot = (
            self.single_turn_resonator_convolution_solver._wake_function_vals
        )
        initial_wake_pot_time = (
            self.single_turn_resonator_convolution_solver._wake_function_time
        )
        assert len(initial_wake_pot) == len(initial_wake_pot_time)
        initial_voltage = (
            self.single_turn_resonator_convolution_solver.calc_induced_voltage(
                beam=self.beam
            )
        )
        initial_profile_len = len(
            self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_x
        )
        assert initial_profile_len == len(initial_voltage)

        # extend profile with 0s towards the back, should not change the values, which are before the 0s
        new_right_edge = 2.0e-9
        self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_x = np.append(
            self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_x,
            np.arange(
                self.right_edge + self.hist_step,
                new_right_edge + self.hist_step,
                self.hist_step,
            ),
        )
        num_to_append = len(
            self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_x
        ) - len(
            self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_y
        )
        self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_y = np.append(
            self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_y,
            np.zeros(int(num_to_append)),
        )

        self.single_turn_resonator_convolution_solver._wake_function_vals_needs_update = True
        self.single_turn_resonator_convolution_solver._update_potential_sources(
            zero_pinning=True
        )
        updated_voltage = (
            self.single_turn_resonator_convolution_solver.calc_induced_voltage(
                beam=self.beam
            )
        )
        # check for correct length of profiles and voltages
        profile_len = len(
            self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_x
        )
        assert profile_len == len(updated_voltage)
        assert len(initial_wake_pot_time) != len(
            self.single_turn_resonator_convolution_solver._wake_function_time
        )

        # check for unchanging of voltage, which should not change
        shift_index = int(profile_len - initial_profile_len)
        assert np.allclose(
            self.single_turn_resonator_convolution_solver._wake_function_vals[
                shift_index : shift_index + len(initial_wake_pot)
            ],
            initial_wake_pot,
        )
        assert np.allclose(
            updated_voltage[: len(initial_voltage)],
            initial_voltage,
        )

    def test__correct_time_diff_on_internal_array(self):
        _ = self.single_turn_resonator_convolution_solver.calc_induced_voltage(
            beam=self.beam
        )
        np.allclose(
            np.diff(
                self.single_turn_resonator_convolution_solver._wake_function_time
            ),
            np.ones(
                len(
                    self.single_turn_resonator_convolution_solver._wake_function_time
                )
                - 1
            )
            * self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_step,
        )

    def test__update_potential_sources_location_of_calculation_matching(self):
        _ = self.single_turn_resonator_convolution_solver.calc_induced_voltage(
            beam=self.beam
        )
        first_time = self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_x
        found = False
        for run_ind in range(
            len(
                self.single_turn_resonator_convolution_solver._wake_function_time
            )
            - len(
                self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_x
            )
        ):
            if np.allclose(
                self.single_turn_resonator_convolution_solver._wake_function_time[
                    run_ind : run_ind
                    + len(
                        self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_x
                    )
                ],
                self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_x,
                atol=self.hist_step / 100,
            ):
                found = True
                break
        assert found

        local_copy = deepcopy(self.single_turn_resonator_convolution_solver)
        local_copy._wake_function_vals_needs_update = True
        local_copy._parent_wakefield.profile.hist_x = (
            local_copy._parent_wakefield.profile.hist_x + 1e-10 / 2
        )
        _ = local_copy.calc_induced_voltage(beam=self.beam)

        found = False
        for run_ind in range(
            len(local_copy._wake_function_time)
            - len(local_copy._parent_wakefield.profile.hist_x)
        ):
            if np.allclose(
                local_copy._wake_function_time[
                    run_ind : run_ind
                    + len(local_copy._parent_wakefield.profile.hist_x)
                ],
                first_time,
                atol=self.hist_step / 100,
            ):
                found = True
                break
        assert found

    def test__update_potential_sources_result_values(self):
        beam = Mock(BeamBaseClass)
        beam.intensity = int(1e2)
        beam.particle_type.charge = 1
        beam.n_macroparticles_partial.return_value = int(1e2)
        self.single_turn_resonator_convolution_solver._parent_wakefield.profile.hist_y_to_density_factor = (
            1 / beam.n_macroparticles_partial()
        )

        self.single_turn_resonator_convolution_solver._update_potential_sources(
            zero_pinning=True
        )
        profile_width = int(
            (self.right_edge - self.left_edge) / self.hist_step
        )
        self.single_turn_resonator_convolution_solver._wake_function_vals = (
            np.zeros(profile_width * 2 + 1)
        )
        self.single_turn_resonator_convolution_solver._wake_function_vals[
            profile_width - 1 : profile_width + 2
        ] = 1 / 3 / e
        calced_voltage = (
            self.single_turn_resonator_convolution_solver.calc_induced_voltage(
                beam=beam
            )
        )

        min_voltage = np.min(calced_voltage)  # negative due to positive charge
        assert np.isclose(min_voltage, -1 / 3)
        assert np.isclose(
            np.abs(calced_voltage - min_voltage).argmin(), profile_width // 3
        )
        assert np.sum(calced_voltage[0 : profile_width // 3 - 3]) == 0
        assert np.sum(calced_voltage[profile_width // 3 + 3 :]) == 0

        # same check, but with self.hist_step/2 shifted histogram, should have same values
        local_res = deepcopy(self.single_turn_resonator_convolution_solver)
        local_res._parent_wakefield.profile.hist_x = (
            self.hist_x + self.hist_step / 2
        )

        local_res._update_potential_sources(zero_pinning=True)

        calced_voltage = local_res.calc_induced_voltage(beam=beam)
        min_voltage = np.min(calced_voltage)

        assert np.isclose(min_voltage, -1 / 3)
        assert np.isclose(
            np.abs(calced_voltage - min_voltage).argmin(), profile_width // 3
        )
        assert np.sum(calced_voltage[0 : profile_width // 3 - 3]) == 0
        assert np.sum(calced_voltage[profile_width // 3 + 3 :]) == 0

    def test_against_CST_results(self):
        # TODO: fix this, not very close to CST atm
        # CST settings: open BC at z, magnetic symmetry planes, ec1 parameters from https://cds.cern.ch/record/533324, f_cutoff = 2.5GHz, WF length = 5m
        # create bunch with sigma of 40mm --> set this as profile, convolute with potential to get wake for the first 5 meters
        sigma_z = 40e-3
        # R_over_Q = np.array([51.94, 13.7312, 0.0915, 2.638805, 2.132499, 2.712645, 4.064])
        # q_factor = np.array([4.15e8, 4.416e5, 38791, 70.629, 59.224, 35.6335, 23.2348])
        # freq = np.array([1.30192e9, 2.4508e9, 2.70038e9, 3.0675e9, 3.083e9, 3.34753e9, 3.42894e9])
        print("cwd =", os.getcwd())
        print("__file__ dir =", Path(__file__).parent)
        with open(
            str(Path(__file__).parent) + r"/resources/TESLA_until_4.5GHz.json",
            encoding="utf-8",
        ) as cst_modes_EM_file:
            cst_modes_dict = json.load(cst_modes_EM_file)
        freq, q_factor, R_over_Q = [], [], []
        for mode in cst_modes_dict:
            if cst_modes_dict[mode]["Qext"] < 200:
                continue
            freq.append(cst_modes_dict[mode]["freq"])
            q_factor.append(cst_modes_dict[mode]["Qext"])
            R_over_Q.append(cst_modes_dict[mode]["R/Q_||"])
        freq = np.array(freq)
        q_factor = np.array(q_factor)
        R_over_Q = np.array(R_over_Q)

        R_shunt = R_over_Q * q_factor

        res = Resonators(
            quality_factors=q_factor,
            shunt_impedances=R_shunt,
            center_frequencies=freq,
        )
        analy = SingleTurnResonatorConvolutionSolver()

        bunch_time = np.linspace(
            -sigma_z * 8.54 / c, 8.54 * sigma_z / c, 2**12
        )
        bunch = np.exp(-0.5 * (bunch_time / (sigma_z / c)) ** 2)

        analy._parent_wakefield = Mock(WakeField)
        analy._parent_wakefield.profile.hist_step = (
            bunch_time[1] - bunch_time[0]
        )
        analy._parent_wakefield.profile.__
        analy._parent_wakefield.profile.hist_x = bunch_time
        analy._parent_wakefield.profile.hist_y = bunch / np.sum(bunch)

        analy._parent_wakefield.sources = (res,)

        beam = Mock(BeamBaseClass)
        beam.intensity = int(1e3)
        beam.particle_type.charge = 1 / e
        beam.n_macroparticles_partial.return_value = int(1e3)
        # intensity == n_macroparticles, integrated bunch is 1 --> all normalized to 1C

        analy._wake_function_vals_needs_update = True
        analy._parent_wakefield.profile.hist_y_to_density_factor = (
            1 / beam.n_macroparticles_partial()
        )

        calced_voltage = analy.calc_induced_voltage(beam=beam)

        # cst_result = np.load(
        #     str(Path(__file__).parent) + r"/resources/TESLA_ec1_WF_pot.npz"
        # )
        # time_axis = cst_result["s_axis"] / c
        # pot_axis = cst_result["pot_axis"] * 1e12  # pC
        # plt.plot(np.interp(bunch_time, time_axis, pot_axis)[: len(calced_voltage)])
        # plt.plot(calced_voltage[: len(calced_voltage)])
        # plt.show()

        # assert np.allclose(np.interp(bunch_time, time_axis, pot_axis)[len(calced_voltage) // 2:], calced_voltage[len(calced_voltage) // 2:], atol=1e10)

    def test_calc_induced_voltage(self):
        beam = Mock(BeamBaseClass)
        beam.intensity = int(1e9)
        beam.particle_type.charge = 1
        beam.n_macroparticles_partial.return_value = int(1e3)
        initial = (
            self.single_turn_resonator_convolution_solver.calc_induced_voltage(
                beam=beam
            )
        )
        first_nonzero_index = np.abs(initial).argmax() - 1
        beam.intensity = int(1e4)
        assert (
            self.single_turn_resonator_convolution_solver.calc_induced_voltage(
                beam=beam
            )[first_nonzero_index:]
            != initial[first_nonzero_index:]
        ).all()

    def test__on_wakefield_simulation_init(self):
        parent_wakefield = Mock(WakeField)
        profile = Mock(StaticProfile)
        simulation = Mock(Simulation)
        profile.n_bins = 10
        parent_wakefield.profile = profile
        parent_wakefield.profile.hist_step = 1

        resonators = Mock(Resonators)
        resonators.is_dynamic = False
        parent_wakefield.sources = (resonators,)
        self.single_turn_resonator_convolution_solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=parent_wakefield
        )

        with self.assertRaises(RuntimeError):
            profile_wrong = Mock(DynamicProfileConstCutoff)
            parent_wakefield.profile = profile_wrong
            self.single_turn_resonator_convolution_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )
        with self.assertRaises(RuntimeError):
            profile_wrong = Mock(DynamicProfileConstNBins)
            parent_wakefield.profile = profile_wrong
            self.single_turn_resonator_convolution_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )
        parent_wakefield.profile = profile
        with self.assertRaises(RuntimeError):
            wrong_source = Mock(InductiveImpedance)
            wrong_source.is_dynamic = False
            parent_wakefield.sources = (wrong_source, resonators)
            self.single_turn_resonator_convolution_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )
        with self.assertRaises(RuntimeError):
            wrong_source.is_dynamic = True
            parent_wakefield.sources = (wrong_source, resonators)
            self.single_turn_resonator_convolution_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )
        with self.assertRaises(ValueError):
            parent_wakefield.profile = None
            parent_wakefield.sources = (resonators,)
            self.single_turn_resonator_convolution_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )


class TestMultiPassResonatorSolver(unittest.TestCase):
    def setUp(self):
        self.resonators = Resonators(
            shunt_impedances=np.array([1, 2, 3]),
            center_frequencies=np.array([500e6, 750e6, 1.5e9]),
            quality_factors=np.array([10e3, 10e3, 10e3]),
            shunt_impedances_counter_rotating=np.array([-1, -2, -3]),
        )
        self.multi_pass_resonator_solver = MultiPassResonatorSolver()
        self.hist_step, self.hist_x = (
            1e-10,
            np.arange(-1e-9, 1e-9 + 1e-10, 1e-10),
        )

        self.multi_pass_resonator_solver._parent_wakefield = Mock(WakeField)
        self.multi_pass_resonator_solver._parent_wakefield.profile = Mock(
            StaticProfile
        )
        self.multi_pass_resonator_solver._parent_wakefield.profile.hist_step = self.hist_step
        self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x = (
            self.hist_x
        )
        self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y_to_density_factor = 1

        self.profile = np.zeros_like(
            self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x
        )
        self.profile[9:12] = 1  # symmetric profile around centerpoint
        self.profile /= np.sum(self.profile)
        self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y = (
            self.profile
        )

        self.multi_pass_resonator_solver._parent_wakefield.sources = (
            self.resonators,
        )

        self.beam = Mock(BeamBaseClass)
        self.beam.reference = Mock(ReferenceCoordinates)

        self.beam.intensity = int(1e2)
        self.beam.particle_type.charge = 1
        self.beam.n_macroparticles_partial.return_value = int(1e2)
        self.beam.reference.time = 0
        self.beam.is_counter_rotating = False

    def test_info_string_with_RF_station(self):
        shc = SingleHarmonicRfStation(
            section_index=0,
            harmonic=1,
            voltage=1,
            phi_rf=1,
            local_wakefield=WakeField(
                profile=StaticProfile.from_cutoff(0, 1e-9, 3e9),
                sources=(self.resonators,),
                solver=self.multi_pass_resonator_solver,
            ),
        )
        assert "WakeField" in shc.info_string()

    def test_determine_storage_time_single_res(self):
        simulation = Mock(Simulation)
        single_resonator = Resonators(
            shunt_impedances=np.array([1]),
            center_frequencies=np.array([500e6]),
            quality_factors=np.array([10e3]),
        )
        local_solv = deepcopy(self.multi_pass_resonator_solver)
        local_solv.on_wakefield_init_simulation(
            simulation=simulation,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_solv._parent_wakefield.sources = (single_resonator,)
        local_solv._determine_storage_time()
        assert np.isclose(
            local_solv._maximum_storage_time,
            -np.log(local_solv._decay_fraction_threshold)
            / single_resonator._alpha[0],
        )

        with self.assertRaises(RuntimeError):
            local_solv._parent_wakefield = None
            local_solv._determine_storage_time()

    def test_determine_storage_time_multi_res(self):
        # Check for mixing with multiple resonators
        simulation = Mock(Simulation)
        single_resonator = Resonators(
            shunt_impedances=np.array([1, 10]),
            center_frequencies=np.array([500e6, 500e6]),
            quality_factors=np.array([10e3, 10e6]),
        )  # 2nd one should be way later, but similar amplitude
        local_solv = deepcopy(self.multi_pass_resonator_solver)
        local_solv.on_wakefield_init_simulation(
            simulation=simulation,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_solv._parent_wakefield.sources = (single_resonator,)
        local_solv._determine_storage_time()
        assert not np.isclose(
            local_solv._maximum_storage_time,
            -np.log(local_solv._decay_fraction_threshold)
            / single_resonator._alpha[0],
        )
        assert not np.isclose(
            local_solv._maximum_storage_time,
            -np.log(local_solv._decay_fraction_threshold)
            / single_resonator._alpha[1],
        )  # mixing of signals

        # check if one properly overshadows the other with high R_shunt
        single_resonator = Resonators(
            shunt_impedances=np.array([1, 1e9]),
            center_frequencies=np.array([500e6, 500e6]),
            quality_factors=np.array([10e3, 10e6]),
        )  # 2nd one should be way later
        local_solv._parent_wakefield.sources = (single_resonator,)
        local_solv._determine_storage_time()
        assert not np.isclose(
            local_solv._maximum_storage_time,
            -np.log(local_solv._decay_fraction_threshold)
            / single_resonator._alpha[0],
        )
        assert np.isclose(
            local_solv._maximum_storage_time,
            -np.log(local_solv._decay_fraction_threshold)
            / single_resonator._alpha[1],
        )  # no mixing due to 2nd one with way higher shunt impedance

    def test_remove_fully_decayed_wake_profiles(self):
        self.multi_pass_resonator_solver._wake_function_vals = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )
        self.multi_pass_resonator_solver._wake_function_time = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )  # technically not correct length but doesnt matter here
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )
        self.multi_pass_resonator_solver._past_profiles = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )
        self.multi_pass_resonator_solver._past_profiles_counter_rotation_flag = deque(
            [False, False, False]
        )

        self.multi_pass_resonator_solver._maximum_storage_time = 1.0
        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(
            indexes_to_check=1
        )

        assert (
            len(self.multi_pass_resonator_solver._wake_function_vals)
            == len(self.multi_pass_resonator_solver._wake_function_time)
            == len(self.multi_pass_resonator_solver._past_profile_times)
            == len(self.multi_pass_resonator_solver._past_profiles)
            == 2
        )
        # check correct values in both elements --> to ensure last one got kicked
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_function_vals[0]), 3
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_function_time[0]),
            0.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profile_times[0]),
            0.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profiles[0]), 3
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_function_vals[1]), 6
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_function_time[1]),
            3.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profile_times[1]),
            3.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profiles[1]), 6
        )

        # check that we don't crash for the empty array --> only one entry present
        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(
            indexes_to_check=2
        )

        self.multi_pass_resonator_solver._wake_function_vals = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )
        self.multi_pass_resonator_solver._wake_function_time = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )  # technically not correct length but doesnt matter here
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )
        self.multi_pass_resonator_solver._past_profiles = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )
        self.multi_pass_resonator_solver._past_profiles_counter_rotation_flag = deque(
            [False, False, False]
        )

        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(
            indexes_to_check=2
        )
        assert (
            len(self.multi_pass_resonator_solver._wake_function_vals)
            == len(self.multi_pass_resonator_solver._wake_function_time)
            == len(self.multi_pass_resonator_solver._past_profile_times)
            == len(self.multi_pass_resonator_solver._past_profiles)
            == 1
        )
        # check correct values in both elements --> to ensure last one got kicked
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_function_vals[0]), 3
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_function_time[0]),
            0.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profile_times[0]),
            0.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profiles[0]), 3
        )

        self.multi_pass_resonator_solver._wake_function_vals = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )
        self.multi_pass_resonator_solver._wake_function_time = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )  # technically not correct length but doesnt matter here
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )
        self.multi_pass_resonator_solver._past_profiles = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )
        self.multi_pass_resonator_solver._past_profiles_counter_rotation_flag = deque(
            [False, False, False]
        )

        self.multi_pass_resonator_solver._maximum_storage_time = 2.0
        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(
            indexes_to_check=2
        )
        assert (
            len(self.multi_pass_resonator_solver._wake_function_vals)
            == len(self.multi_pass_resonator_solver._wake_function_time)
            == len(self.multi_pass_resonator_solver._past_profile_times)
            == len(self.multi_pass_resonator_solver._past_profiles)
            == 2
        )
        # check correct values in both elements --> to ensure last one got kicked
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_function_vals[0]), 3
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_function_time[0]),
            0.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profile_times[0]),
            0.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profiles[0]), 3
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_function_vals[1]), 6
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_function_time[1]),
            3.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profile_times[1]),
            3.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profiles[1]), 6
        )

        self.multi_pass_resonator_solver._past_profile_times = deque(
            np.add(
                self.multi_pass_resonator_solver._past_profile_times,
                self.multi_pass_resonator_solver._maximum_storage_time + 1,
            )
        )
        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(
            indexes_to_check=2
        )

        assert len(self.multi_pass_resonator_solver._past_profile_times) == 0

        # check immediate return
        self.multi_pass_resonator_solver._wake_function_vals = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )
        self.multi_pass_resonator_solver._wake_function_time = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )  # technically not correct length but doesnt matter here
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )
        self.multi_pass_resonator_solver._past_profiles = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )
        self.multi_pass_resonator_solver._past_profiles_counter_rotation_flag = deque(
            [False, False, False]
        )

        self.multi_pass_resonator_solver._maximum_storage_time = 0.0
        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(
            indexes_to_check=4
        )

    def test_remove_fully_decayed_wake_profiles_physics(self):
        simulation = Mock(Simulation)
        single_resonator = Resonators(
            shunt_impedances=np.array([1]),
            center_frequencies=np.array([500e6]),
            quality_factors=np.array([10e3]),
        )  # 2nd one should be way later, but similar amplitude
        local_solv = deepcopy(self.multi_pass_resonator_solver)
        local_solv._parent_wakefield.sources = (single_resonator,)
        local_solv.on_wakefield_init_simulation(
            simulation=simulation,
            parent_wakefield=local_solv._parent_wakefield,
        )
        assert np.isclose(
            local_solv._maximum_storage_time,
            1
            / -single_resonator._alpha[0]
            * np.log(local_solv._decay_fraction_threshold),
        )

        with self.assertRaises(ValueError):
            local_solv._parent_wakefield.profile = None
            local_solv.on_wakefield_init_simulation(
                simulation=simulation,
                parent_wakefield=local_solv._parent_wakefield,
            )

        with self.assertRaises(RuntimeError):
            local_solv._parent_wakefield.profile = DynamicProfileConstCutoff(
                timestep=0
            )
            local_solv.on_wakefield_init_simulation(
                simulation=simulation,
                parent_wakefield=local_solv._parent_wakefield,
            )

        with self.assertRaises(RuntimeError):
            local_solv._parent_wakefield.sources = (
                ImpedanceTableFreq(np.array([0]), np.array([0])),
            )
            local_solv.on_wakefield_init_simulation(
                simulation=simulation,
                parent_wakefield=local_solv._parent_wakefield,
            )

    def test_update_past_profile_times_wake_times(self):
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )
        self.multi_pass_resonator_solver._wake_function_time = deque(
            [
                np.array([4.1, 4.2, 4.3]),
                np.array([5.1, 5.2, 5.3]),
                np.array([6.1, 6.2, 6.3]),
            ]
        )
        sum_before_shift_prof = np.sum(
            self.multi_pass_resonator_solver._past_profile_times
        )
        sum_before_shift_wake = np.sum(
            self.multi_pass_resonator_solver._wake_function_time
        )
        orig_ref = 1
        self.multi_pass_resonator_solver._last_reference_time = orig_ref
        delta_t = 1
        self.multi_pass_resonator_solver._update_past_profile_times_wake_times(
            current_time=self.multi_pass_resonator_solver._last_reference_time
            + delta_t
        )
        assert np.isclose(
            sum_before_shift_prof + 9,
            np.sum(self.multi_pass_resonator_solver._past_profile_times),
        )
        assert np.isclose(
            sum_before_shift_wake + 9,
            np.sum(self.multi_pass_resonator_solver._wake_function_time),
        )
        assert (
            self.multi_pass_resonator_solver._last_reference_time
            == orig_ref + delta_t
        )

        with self.assertRaises(AssertionError):
            self.multi_pass_resonator_solver._update_past_profile_times_wake_times(
                current_time=self.multi_pass_resonator_solver._last_reference_time
                - delta_t
            )

    def test__update_past_profile_potentials_new_arr_init(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_res._past_profile_times.appendleft(
            self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x
        )
        local_res._past_profiles.appendleft(
            self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y
        )
        local_res._past_profiles_counter_rotation_flag.appendleft(False)
        local_res._update_past_profile_wake_functions(zero_pinning=True)

        assert len(local_res._wake_function_time) == 1
        assert len(local_res._wake_function_vals) == 1
        assert len(local_res._past_profile_times) == 1
        assert len(local_res._past_profiles) == 1

        assert len(local_res._wake_function_vals[0]) == len(
            local_res._wake_function_time[0]
        )

        assert np.allclose(local_res._past_profile_times[0], self.hist_x)
        assert np.allclose(local_res._past_profiles[0], self.profile)

    def test__update_past_profile_potentials_pushback_of_2nd_array(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_res._past_profile_times.appendleft(
            deepcopy(
                self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x
            )
        )
        local_res._past_profiles.appendleft(
            deepcopy(
                self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y
            )
        )
        local_res._past_profiles_counter_rotation_flag.appendleft(False)
        local_res._update_past_profile_wake_functions(zero_pinning=True)
        local_res._update_past_profile_times_wake_times(1e-8)
        local_res._past_profile_times.appendleft(
            deepcopy(
                self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x
            )
        )
        local_res._past_profiles.appendleft(
            deepcopy(
                self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y
            )
        )
        local_res._past_profiles_counter_rotation_flag.appendleft(False)
        local_res._update_past_profile_wake_functions(zero_pinning=True)

        # should have been pushed back --> [1] is the oder profile, [0] is the newest
        assert len(local_res._wake_function_time) == 2
        assert len(local_res._wake_function_vals) == 2
        assert len(local_res._past_profile_times) == 2
        assert len(local_res._past_profiles) == 2

        assert len(local_res._wake_function_vals[0]) == len(
            local_res._wake_function_time[0]
        )
        assert len(local_res._wake_function_vals[1]) == len(
            local_res._wake_function_time[1]
        )

        assert np.allclose(local_res._past_profile_times[0], self.hist_x)
        assert np.allclose(local_res._past_profiles[0], self.profile)
        assert np.allclose(
            local_res._past_profile_times[1], self.hist_x + 1e-8
        )
        assert np.allclose(local_res._past_profiles[1], self.profile + 1e-8)

        assert np.not_equal(
            local_res._wake_function_vals[0], local_res._wake_function_vals[1]
        ).any()
        assert np.allclose(
            local_res._wake_function_time[0],
            local_res._wake_function_time[1] - 1e-8,
            atol=1e-10,
        )

    def test__update_potential_sources(self):
        """
        Test presence of arrays and correct shifting of timing
        """
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_res._update_potential_sources(self.beam)

        local_res._wake_function_vals_needs_update = True
        tsteps = [0.5, 1.0, 1.6]
        local_res._maximum_storage_time = 1.5
        beam = deepcopy(self.beam)
        beam.reference.time = tsteps[0]
        local_res._update_potential_sources(beam=beam)

        assert (
            len(local_res._wake_function_time)
            == len(local_res._wake_function_vals)
            == len(local_res._past_profile_times)
            == len(local_res._past_profiles)
            == 2
        )
        np.testing.assert_allclose(
            local_res._wake_function_time[1],
            local_res._wake_function_time[0] + tsteps[0],
        )
        np.testing.assert_allclose(
            local_res._past_profile_times[1],
            local_res._past_profile_times[0] + tsteps[0],
        )
        np.testing.assert_allclose(
            local_res._past_profiles[0], local_res._past_profiles[1]
        )

        # repeat another time, first array should be kicked out due to delay
        local_res._wake_function_vals_needs_update = True
        beam.reference.time = tsteps[1]
        local_res._update_potential_sources(beam=beam)
        assert (
            len(local_res._wake_function_time)
            == len(local_res._wake_function_vals)
            == len(local_res._past_profile_times)
            == len(local_res._past_profiles)
            == 3
        )
        np.testing.assert_allclose(
            local_res._wake_function_time[1],
            local_res._wake_function_time[0] + tsteps[1] - tsteps[0],
        )
        np.testing.assert_allclose(
            local_res._past_profile_times[1],
            local_res._past_profile_times[0] + tsteps[1] - tsteps[0],
        )
        np.testing.assert_allclose(
            local_res._past_profiles[0], local_res._past_profiles[1]
        )

        # kick out oldest profile
        local_res._wake_function_vals_needs_update = True
        beam.reference.time = tsteps[2]
        local_res._update_potential_sources(beam=beam)
        assert (
            len(local_res._wake_function_time)
            == len(local_res._wake_function_vals)
            == len(local_res._past_profile_times)
            == len(local_res._past_profiles)
            == 3
        )
        np.testing.assert_allclose(
            np.mean(local_res._wake_function_time[1]),
            np.mean(local_res._wake_function_time[0] + tsteps[2] - tsteps[1]),
        )
        np.testing.assert_allclose(
            np.mean(local_res._past_profile_times[1]),
            np.mean(local_res._past_profile_times[0] + tsteps[2] - tsteps[1]),
        )
        np.testing.assert_allclose(
            np.mean(local_res._wake_function_time[2]),
            np.mean(local_res._wake_function_time[1] + tsteps[1] - tsteps[0]),
        )
        np.testing.assert_allclose(
            np.mean(local_res._past_profile_times[2]),
            np.mean(local_res._past_profile_times[1] + tsteps[1] - tsteps[0]),
        )

        np.testing.assert_allclose(
            local_res._past_profiles[0], local_res._past_profiles[1]
        )
        np.testing.assert_allclose(
            local_res._past_profiles[1], local_res._past_profiles[2]
        )

    def test__update_potential_sources_hist_step(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        beam = deepcopy(self.beam)
        local_res._update_potential_sources(beam)

        local_res._maximum_storage_time = 1.5

        local_res._parent_wakefield.profile.hist_x *= 2
        local_res._wake_function_vals_needs_update = True
        beam.reference.time += 1
        with self.assertRaises(
            AssertionError,
            msg="profile bin size needs to be constant: hist_step might be too small with casting to delta_t precision",
        ):
            local_res._update_potential_sources(beam)

    def test_calc_induced_voltage_array_lengths(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        ind_volt = local_res.calc_induced_voltage(beam=self.beam)

        assert len(ind_volt) == len(local_res._parent_wakefield.profile.hist_x)

        local_res._maximum_storage_time = 1.5
        local_res._wake_function_vals_needs_update = True
        beam = deepcopy(self.beam)
        beam.reference.time = 1
        local_res._update_potential_sources(beam=beam)

        assert len(ind_volt) == len(local_res._parent_wakefield.profile.hist_x)

    def test_calc_induced_voltage_two_passages(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        ind_volt = local_res.calc_induced_voltage(beam=self.beam)

        assert len(ind_volt) == len(local_res._parent_wakefield.profile.hist_x)

        local_res._maximum_storage_time = 1.5
        local_res._wake_function_vals_needs_update = True
        beam = deepcopy(self.beam)
        beam.reference.time = 1
        local_res._update_potential_sources(beam=beam)

        assert len(ind_volt) == len(local_res._parent_wakefield.profile.hist_x)

    def test_calc_induced_voltage_counter_rotation(self):
        sim = Mock(Simulation)

        local_res_corot = deepcopy(self.multi_pass_resonator_solver)
        local_res_corot.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        beam = deepcopy(self.beam)
        beam.is_counter_rotating = False
        ind_volt_corot = local_res_corot.calc_induced_voltage(beam=beam)

        assert len(ind_volt_corot) == len(
            local_res_corot._parent_wakefield.profile.hist_x
        )

        local_res_counterrot = deepcopy(self.multi_pass_resonator_solver)
        local_res_counterrot.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        beam = deepcopy(self.beam)
        beam.is_counter_rotating = True
        ind_volt_corot = local_res_counterrot.calc_induced_voltage(beam=beam)
        np.testing.assert_allclose(
            ind_volt_corot, ind_volt_corot
        )  # first one needs to be the same as this is the self-field

        local_res_counterrot._parent_wakefield.profile.hist_y = np.zeros_like(
            local_res_counterrot._parent_wakefield.profile.hist_y
        )
        # avoid interference from current profile
        local_res_counterrot_corot = deepcopy(local_res_counterrot)
        local_res_counterrot_counterrot = deepcopy(local_res_counterrot)
        beam.is_counter_rotating = False
        beam.reference.time += np.finfo(float).eps
        counterrot_corot_ind_volt = (
            local_res_counterrot_corot.calc_induced_voltage(beam)
        )
        beam.is_counter_rotating = True
        counterrot_counterrot_ind_volt = (
            local_res_counterrot_counterrot.calc_induced_voltage(beam)
        )

        np.testing.assert_allclose(
            counterrot_corot_ind_volt, -counterrot_counterrot_ind_volt
        )
        # should be inverted as all shunt impedances are inverted

    def test_calc_induced_voltage_counter_rotation_opposite_charge(self):
        sim = Mock(Simulation)

        local_res_counterrot = deepcopy(self.multi_pass_resonator_solver)
        local_res_counterrot.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        beam = deepcopy(self.beam)
        beam.particle_type.charge = -1
        beam.is_counter_rotating = True
        ind_volt_corot = local_res_counterrot.calc_induced_voltage(beam=beam)
        np.testing.assert_allclose(
            ind_volt_corot, ind_volt_corot
        )  # first one needs to be the same as this is the self-field

        local_res_counterrot._parent_wakefield.profile.hist_y = np.zeros_like(
            local_res_counterrot._parent_wakefield.profile.hist_y
        )
        # avoid interference from current profile
        local_res_counterrot_corot = deepcopy(local_res_counterrot)
        local_res_counterrot_counterrot = deepcopy(local_res_counterrot)
        beam.is_counter_rotating = False
        beam.reference.time += np.finfo(float).eps
        beam.read_partial_dt.return_value = np.linspace(
            local_res_counterrot._parent_wakefield.profile.hist_x[0],
            local_res_counterrot._parent_wakefield.profile.hist_x[-1],
            num=100,
        )
        beam.dE = np.zeros_like(beam.read_partial_dt())

        beam_corot = deepcopy(beam)
        counterrot_corot_ind_volt = (
            local_res_counterrot_corot.calc_induced_voltage(beam_corot)
        )
        backend.set_specials("python")
        backend.specials.kick_induced_voltage(
            dt=beam_corot.read_partial_dt(),
            dE=beam_corot.dE,
            voltage=counterrot_corot_ind_volt,
            bin_centers=self.hist_x,  # base for induced voltage
            charge=backend.float(beam.particle_type.charge),
            acceleration_kick=backend.float(
                0.0
            ),  # TODO was this ever required??
        )

        beam_corot_dt = beam_corot.read_partial_dt()
        beam_corot_dE = beam_corot.dE

        beam.is_counter_rotating = True
        beam.particle_type.charge = 1
        counterrot_counterrot_ind_volt = (
            local_res_counterrot_counterrot.calc_induced_voltage(beam)
        )

        # opposite charge and opposite direction --> should be same kick but not same voltage
        np.testing.assert_allclose(
            counterrot_corot_ind_volt, -counterrot_counterrot_ind_volt
        )

        backend.specials.kick_induced_voltage(
            dt=beam.read_partial_dt(),
            dE=beam.dE,
            voltage=counterrot_counterrot_ind_volt,
            bin_centers=self.hist_x,  # base for induced voltage
            charge=beam.particle_type.charge,
            acceleration_kick=backend.float(0.0),
        )
        np.testing.assert_allclose(beam.read_partial_dt(), beam_corot_dt)
        np.testing.assert_allclose(beam.dE, beam_corot_dE)
        # the resulting kick should be the same

    def test_calc_induced_voltage_vals(self):
        resonators = Resonators(
            shunt_impedances=np.array([1e12]),
            center_frequencies=np.array([500e6]),
            quality_factors=np.array([10e5]),
        )

        local_res = MultiPassResonatorSolver()

        sigma_z = 40e-3
        sigma_length = 15
        bunch_time = np.linspace(
            -sigma_z * sigma_length / c, sigma_length * sigma_z / c, 2**10
        )
        bunch = np.exp(-0.5 * (bunch_time / (sigma_z / c)) ** 2)

        local_res._parent_wakefield = Mock(WakeField)
        local_res._parent_wakefield.profile = Mock(spec=StaticProfile)
        local_res._parent_wakefield.profile.hist_step = (
            bunch_time[1] - bunch_time[0]
        )
        local_res._parent_wakefield.profile.hist_x = bunch_time
        local_res._parent_wakefield.profile.hist_y = bunch / np.sum(bunch)
        local_res._parent_wakefield.profile.hist_y_to_density_factor = (
            1 / self.beam.n_macroparticles_partial()
        )

        local_res._parent_wakefield.sources = (resonators,)

        sim = Mock(Simulation)

        local_res.on_wakefield_init_simulation(
            simulation=sim, parent_wakefield=local_res._parent_wakefield
        )
        beam = deepcopy(self.beam)
        beam.reference.time = 0
        ind_volt_init = local_res.calc_induced_voltage(beam=beam)

        t_rf = 1 / resonators._center_frequencies[0]
        delay_time = (
            np.floor((1 / resonators._alpha[0]) / t_rf) * t_rf
        )  # multiple of t_r to ensure in-phase correctness
        beam = deepcopy(self.beam)
        beam.reference.time = delay_time

        ind_volt = local_res.calc_induced_voltage(beam=beam)

        # ensure perfect addition of in-phase component
        assert not np.allclose(ind_volt, ind_volt_init)
        assert np.argmax(ind_volt) == np.argmax(ind_volt_init)
        assert not np.isclose(ind_volt[0], 0)
        assert np.isclose(
            np.min(ind_volt), np.min(ind_volt_init) * (1 + 1 / np.exp(1))
        )

        # assert equality for fully decayed case
        local_res._wake_function_vals_needs_update = True
        local_res._maximum_storage_time = (
            local_res._maximum_storage_time * 1000
        )
        delay_time = (
            np.floor((1 / resonators._alpha[0]) / t_rf) * t_rf * 100
        )  # multiple of t_r

        beam = deepcopy(self.beam)
        beam.reference.time = delay_time
        ind_volt = local_res.calc_induced_voltage(beam=beam)

        # ensure perfect addition of in-phase component
        assert np.allclose(ind_volt, ind_volt_init)
        assert np.argmax(ind_volt) == np.argmax(ind_volt_init)

    def test_on_wakefield_init_simulation_wrong_source(self):
        src = ImpedanceTableFreq(
            freq_x=np.array([1e12]), freq_y=np.array([500e6])
        )
        parent_wakefield = WakeField(
            sources=(src,), solver=None, profile=Mock(StaticProfile)
        )
        simulation = Mock(Simulation)
        with self.assertRaisesRegex(
            RuntimeError, "Expected `Resonators` and not "
        ):
            self.multi_pass_resonator_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )
        res = Resonators(
            shunt_impedances=1.0, center_frequencies=1.0, quality_factors=1.0
        )
        res.is_dynamic = True
        parent_wakefield = WakeField(
            sources=(res,), solver=None, profile=Mock(StaticProfile)
        )
        with self.assertRaisesRegex(
            RuntimeError, "Expected `Resonators` and not "
        ):
            self.multi_pass_resonator_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )

    def test_compare_to_analytical_resonator_solver_for_results(self):
        resonators = Resonators(
            shunt_impedances=np.array([1e12, 1e10]),
            center_frequencies=np.array([500e6, 1000e6]),
            quality_factors=np.array([10e5, 10e4]),
        )

        sigma_z = 40e-3
        sigma_length = 8.54
        for delta_t in [0, 0.5e-9, -0.5e-9]:
            bunch_time = np.linspace(
                -sigma_z * sigma_length / c + delta_t,
                sigma_length * sigma_z / c + delta_t,
                2**10,
            )
            bunch = np.exp(-0.5 * (bunch_time / (sigma_z / c)) ** 2)

            local_res = MultiPassResonatorSolver(
                decay_fraction_threshold=0.999
            )

            local_res._parent_wakefield = Mock(WakeField)
            local_res._parent_wakefield.profile = Mock(spec=StaticProfile)
            local_res._parent_wakefield.profile.hist_step = (
                bunch_time[1] - bunch_time[0]
            )
            local_res._parent_wakefield.profile.hist_x = bunch_time
            local_res._parent_wakefield.profile.hist_y = bunch / np.sum(bunch)
            local_res._parent_wakefield.profile.hist_y_to_density_factor = (
                1 / self.beam.n_macroparticles_partial()
            )

            local_res._parent_wakefield.sources = (resonators,)
            local_res._wake_function_vals_needs_update = True

            sim = Mock(Simulation)

            local_res.on_wakefield_init_simulation(
                simulation=sim, parent_wakefield=local_res._parent_wakefield
            )

            local_res_analy = SingleTurnResonatorConvolutionSolver()
            local_res_analy._parent_wakefield = Mock(WakeField)
            local_res_analy._parent_wakefield.profile.hist_step = (
                bunch_time[1] - bunch_time[0]
            )
            local_res_analy._parent_wakefield.profile.hist_x = bunch_time
            local_res_analy._parent_wakefield.profile.hist_y = bunch / np.sum(
                bunch
            )
            local_res_analy._parent_wakefield.profile.hist_y_to_density_factor = (
                1 / self.beam.n_macroparticles_partial()
            )
            local_res_analy._parent_wakefield.sources = (resonators,)

            local_res_analy._wake_function_vals_needs_update = True

            local_res._last_reference_time = -np.finfo(float).eps

            ind_volt_mtw = local_res.calc_induced_voltage(beam=self.beam)

            np.testing.assert_allclose(
                ind_volt_mtw,
                local_res_analy.calc_induced_voltage(beam=self.beam),
            )


beam_spectrum = np.array(
    [
        (1000000 + 0j),
        (546055 - 830191.3125j),
        (-386175.90625 - 895154.0625j),
        (-929950.4375 - 163693.03125j),
        (-618882.25 + 657077.6875j),
        (197810.15625 + 828036.75j),
        (744962.75 + 268384.8125j),
        (579837.125 - 436622.25j),
        (-43704.27734375 - 653752.3125j),
        (-507581 - 284416.96875j),
        (-449277.90625 + 236233.046875j),
        (-38983.41796875 + 432552.5j),
        (286995.71875 + 223259.0625j),
        (279745.40625 - 99642.65625j),
        (51381.61328125 - 229906.671875j),
        (-129055.453125 - 126061.1875j),
        (-127790.234375 + 33567.01171875j),
        (-23345.23828125 + 88123.484375j),
        (43304.59375 + 38279.46875j),
        (28336.650390625 - 15921.2724609375j),
        (-9165.607421875 - 14923.294921875j),
        (-10844.8662109375 + 12870.37890625j),
        (14614.5625 + 16428.9765625j),
        (24028.693359375 - 8325.3125j),
        (3728.75439453125 - 25826.583984375j),
        (-19382.13671875 - 14781.4970703125j),
        (-19275.111328125 + 8182.13916015625j),
        (-1829.8526611328125 + 16374.3193359375j),
        (9600.4970703125 + 6545.54345703125j),
        (5864.1181640625 - 3728.43994140625j),
        (-1555.8551025390625 - 2825.3115234375j),
        (-929.931396484375 + 2653.229736328125j),
        (4469.30908203125 + 1772.39111328125j),
        (4453.5341796875 - 4563.33984375j),
        (-2260.787109375 - 6772.86376953125j),
        (-7032.0302734375 - 1237.1505126953125j),
        (-4036.97998046875 + 5117.88671875j),
        (2285.41259765625 + 4964.3994140625j),
        (4198.69921875 - 53.27255630493164j),
        (852.5298461914062 - 2852.020263671875j),
        (-1995.01171875 - 785.3765258789062j),
        (-645.3869018554688 + 1920.93359375j),
        (2127.16748046875 + 1064.1029052734375j),
        (1950.819091796875 - 1896.6151123046875j),
        (-946.5963134765625 - 2676.17431640625j),
        (-2658.9443359375 - 354.8549499511719j),
        (-1315.0810546875 + 1855.872314453125j),
        (789.0394287109375 + 1476.8363037109375j),
        (963.0291748046875 - 131.25628662109375j),
        (-197.47085571289062 - 361.48529052734375j),
        (-267.4827575683594 + 729.0340576171875j),
        (1110.9569091796875 + 843.406494140625j),
        (1714.2562255859375 - 836.8746337890625j),
        (126.95709991455078 - 2252.027587890625j),
        (-2022.1378173828125 - 1328.2230224609375j),
        (-2149.22119140625 + 1070.2410888671875j),
        (-133.6682586669922 + 2215.332275390625j),
        (1607.8917236328125 + 1025.0438232421875j),
        (1300.6856689453125 - 759.4834594726562j),
        (-137.00125122070312 - 1062.3828125j),
        (-680.1707763671875 - 52.202335357666016j),
        (59.46891784667969 + 503.0387268066406j),
        (630.0478515625 - 156.08868408203125j),
        (7.446657180786133 - 885.3383178710938j),
        (-983.9039306640625 - 437.452392578125j),
        (-919.9992065429688 + 749.2589721679688j),
        (229.85089111328125 + 1186.7908935546875j),
        (1097.0821533203125 + 348.54083251953125j),
        (739.1185302734375 - 714.1712646484375j),
        (-243.70486450195312 - 821.800537109375j),
        (-651.0577392578125 - 104.2130126953125j),
        (-230.5679168701172 + 393.1328430175781j),
        (212.8994140625 + 176.57608032226562j),
        (69.81470489501953 - 187.07212829589844j),
        (-284.4610290527344 - 38.782466888427734j),
        (-149.28424072265625 + 403.2663269042969j),
        (428.4289245605469 + 384.4917297363281j),
        (658.8712158203125 - 279.1316833496094j),
        (61.28057098388672 - 847.6890869140625j),
        (-824.9246826171875 - 529.1279296875j),
        (-981.2300415039062 + 511.59515380859375j),
        (-71.99556732177734 + 1226.9716796875j),
        (1099.7730712890625 + 776.24169921875j),
        (1348.27099609375 - 548.6397705078125j),
        (298.8646240234375 - 1525.852294921875j),
        (-1166.5372314453125 - 1153.245361328125j),
        (-1673.1678466796875 + 343.2100830078125j),
        (-658.93896484375 + 1624.4730224609375j),
        (1003.30224609375 + 1458.7991943359375j),
        (1755.3701171875 - 48.670570373535156j),
        (870.01513671875 - 1468.204833984375j),
        (-763.1235961914062 - 1430.8338623046875j),
        (-1503.0745849609375 - 49.374874114990234j),
        (-680.2874755859375 + 1178.7718505859375j),
        (683.5153198242188 + 994.6351928710938j),
        (1038.5364990234375 - 230.2852783203125j),
        (91.0697250366211 - 958.4382934570312j),
        (-876.1680908203125 - 326.85821533203125j),
        (-580.2130737304688 + 804.28271484375j),
        (651.9199829101562 + 902.2933349609375j),
        (1229.41455078125 - 308.60955810546875j),
        (255.07472229003906 - 1405.1473388671875j),
        (-1269.451416015625 - 936.3026123046875j),
        (-1528.2242431640625 + 757.5350952148438j),
        (-48.672462463378906 + 1808.0867919921875j),
        (1635.09521484375 + 940.2272338867188j),
        (1655.5782470703125 - 1010.9423828125j),
        (-83.06259155273438 - 1971.1259765625j),
        (-1772.9078369140625 - 900.5798950195312j),
        (-1662.837890625 + 1092.2197265625j),
        (103.03956604003906 + 1974.75927734375j),
        (1725.0592041015625 + 916.4118041992188j),
        (1655.7408447265625 - 967.3838500976562j),
        (74.237060546875 - 1869.1888427734375j),
        (-1471.683837890625 - 1059.4622802734375j),
        (-1643.707275390625 + 593.0517578125j),
        (-447.6824035644531 + 1614.2926025390625j),
        (990.9194946289062 + 1255.69970703125j),
        (1522.913818359375 - 33.67795944213867j),
        (859.4178466796875 - 1165.8160400390625j),
        (-366.68115234375 - 1325.9505615234375j),
        (-1201.4239501953125 - 509.562255859375j),
        (-1085.7430419921875 + 588.2033081054688j),
        (-204.4443817138672 + 1145.1346435546875j),
        (719.5697631835938 + 816.6748046875j),
        (1008.0802612304688 - 52.485050201416016j),
        (536.1304321289062 - 755.3461303710938j),
        (-235.342529296875 - 806.5758056640625j),
        (-697.5813598632812 - 285.0380554199219j),
        (-585.1900634765625 + 325.027099609375j),
        (-102.58599090576172 + 582.0130615234375j),
        (339.4863586425781 + 395.673828125j),
        (462.59283447265625 - 8.486898422241211j),
        (258.45654296875 - 325.7151184082031j),
        (-83.0448226928711 - 370.9265441894531j),
        (-319.6766357421875 - 151.38063049316406j),
        (-294.9234313964844 + 156.8343963623047j),
        (-40.29768371582031 + 315.8642883300781j),
        (228.7417449951172 + 200.93203735351562j),
        (279.23443603515625 - 78.20695495605469j),
        (78.79337310791016 - 261.405517578125j),
        (-165.60009765625 - 189.73187255859375j),
        (-222.21824645996094 + 35.22499084472656j),
        (-76.15454864501953 + 175.79055786132812j),
        (83.13419342041016 + 126.52083587646484j),
        (105.09873962402344 + 3.9750986099243164j),
        (40.21259689331055 - 38.385005950927734j),
        (23.517005920410156 - 10.885547637939453j),
        (59.750431060791016 - 34.42404556274414j),
        (22.050045013427734 - 121.93507385253906j),
        (-127.9256362915039 - 123.51632690429688j),
        (-221.2574462890625 + 57.13376235961914j),
        (-75.03158569335938 + 265.0582275390625j),
        (224.9461212158203 + 226.7318115234375j),
        (348.04742431640625 - 99.4266128540039j),
        (88.53955841064453 - 395.8073425292969j),
        (-340.2792053222656 - 298.523681640625j),
        (-475.7969665527344 + 169.363037109375j),
        (-101.6201171875 + 554.31640625j),
        (470.820556640625 + 415.5801086425781j),
        (669.1730346679688 - 196.3659210205078j),
        (225.537109375 - 736.6522827148438j),
        (-526.5823974609375 - 661.7786254882812j),
        (-920.8875122070312 + 49.392215728759766j),
        (-546.9371337890625 + 835.9398193359375j),
        (365.3968811035156 + 1011.1128540039062j),
        (1095.865234375 + 347.1644592285156j),
        (1004.6427612304688 - 693.3190307617188j),
        (77.2365951538086 - 1283.8448486328125j),
        (-1003.653564453125 - 892.463623046875j),
        (-1366.3907470703125 + 242.4560546875j),
        (-671.6649169921875 + 1246.890625j),
        (561.606689453125 + 1309.5821533203125j),
        (1359.483154296875 + 375.66497802734375j),
        (1114.74853515625 - 797.5819702148438j),
        (88.83451843261719 - 1302.199462890625j),
        (-871.38671875 - 847.8641967773438j),
        (-1104.3768310546875 + 80.4963607788086j),
        (-628.3318481445312 + 763.6647338867188j),
        (61.4375114440918 + 873.9108276367188j),
        (548.4217529296875 + 571.0171508789062j),
        (752.5946044921875 + 123.45515441894531j),
        (713.718505859375 - 370.7041015625j),
        (351.87591552734375 - 836.5725708007812j),
        (-371.8904113769531 - 980.2443237304688j),
        (-1111.89990234375 - 459.451171875j),
        (-1210.77685546875 + 605.7813110351562j),
        (-331.25152587890625 + 1451.19775390625j),
        (1000.4151000976562 + 1247.2117919921875j),
        (1679.4927978515625 - 29.737947463989258j),
        (1022.3040771484375 - 1393.864013671875j),
        (-502.4727783203125 - 1670.935546875j),
        (-1624.1373291015625 - 595.9730224609375j),
        (-1415.8056640625 + 917.6967163085938j),
        (-115.42440795898438 + 1616.76513671875j),
        (1153.1307373046875 + 1015.6717529296875j),
        (1414.6177978515625 - 274.04791259765625j),
        (612.1968994140625 - 1192.517333984375j),
        (-508.0769348144531 - 1133.395751953125j),
        (-1112.1336669921875 - 299.4683837890625j),
        (-876.7816162109375 + 620.5961303710938j),
        (-79.81355285644531 + 1008.7623901367188j),
        (692.0337524414062 + 672.2603149414062j),
        (923.7800903320312 - 108.94757080078125j),
        (472.4669189453125 - 770.1814575195312j),
        (-319.0427551269531 - 819.75537109375j),
        (-825.1046142578125 - 218.18653869628906j),
        (-626.4759521484375 + 530.4212036132812j),
        (87.59724426269531 + 773.9874267578125j),
        (651.5792236328125 + 320.2908630371094j),
        (558.5313720703125 - 355.27630615234375j),
        (-26.97935676574707 - 587.2075805664062j),
        (-458.6560363769531 - 215.13861083984375j),
        (-323.70916748046875 + 273.42041015625j),
        (120.67063903808594 + 327.2365417480469j),
        (295.50274658203125 - 34.519020080566406j),
        (12.472862243652344 - 287.5738220214844j),
        (-312.7098083496094 - 78.12947082519531j),
        (-202.18341064453125 + 326.228515625j),
        (261.9542236328125 + 370.3259582519531j),
        (513.22119140625 - 81.42315673828125j),
        (190.2266387939453 - 542.4237670898438j),
        (-403.1576232910156 - 464.5584411621094j),
        (-628.4400024414062 + 113.3181381225586j),
        (-234.74403381347656 + 600.7507934570312j),
        (376.28515625 + 512.245849609375j),
        (612.210205078125 - 33.02763748168945j),
        (300.6817626953125 - 497.7712707519531j),
        (-217.0853271484375 - 501.297607421875j),
        (-499.81805419921875 - 119.6686019897461j),
        (-386.4754638671875 + 304.4532470703125j),
        (-8.927909851074219 + 487.6376647949219j),
        (386.03851318359375 + 327.9058837890625j),
        (539.5643920898438 - 108.8580093383789j),
        (263.0180358886719 - 559.9129638671875j),
        (-354.3292541503906 - 613.6788940429688j),
        (-816.2152099609375 - 48.68413543701172j),
        (-551.6635131835938 + 764.4580688476562j),
        (412.2121887207031 + 998.5927124023438j),
        (1211.3245849609375 + 190.70692443847656j),
        (886.4723510742188 - 1051.5980224609375j),
        (-489.75494384765625 - 1440.9505615234375j),
        (-1621.3782958984375 - 353.4176025390625j),
        (-1223.034423828125 + 1295.376953125j),
        (506.2715759277344 + 1812.50048828125j),
        (1884.8863525390625 + 518.6773071289062j),
        (1442.802490234375 - 1379.538818359375j),
        (-449.64068603515625 - 1951.3525390625j),
        (-1880.881103515625 - 593.681396484375j),
        (-1408.7034912109375 + 1284.2669677734375j),
        (398.9777526855469 + 1762.167724609375j),
        (1613.2137451171875 + 464.91522216796875j),
        (1059.488037109375 - 1104.08349609375j),
        (-468.767333984375 - 1289.0528564453125j),
        (-1214.9393310546875 - 86.9875717163086j),
        (-469.2613830566406 + 982.4129638671875j),
        (719 + 703j),
        (868.576171875 - 465.49859619140625j),
        (-177.3074188232422 - 1014.2913818359375j),
        (-1102.683837890625 - 210.6896514892578j),
        (-701.121826171875 + 1025.41943359375j),
        (678.4178466796875 + 1189.08740234375j),
        (1488.288330078125 - 51.3181266784668j),
        (724.996826171875 - 1419.1453857421875j),
        (-912.3438110351562 - 1406.827392578125j),
        (-1734.18017578125 + 69.95088958740234j),
        (-853.1304931640625 + 1549.5250244140625j),
        (879.3045654296875 + 1543.7845458984375j),
        (1758.7855224609375 + 63.83644485473633j),
        (961.4549560546875 - 1426.7130126953125j),
        (-676.7559204101562 - 1516.0364990234375j),
        (-1566.7027587890625 - 217.44122314453125j),
        (-951.3502807617188 + 1143.4931640625j),
        (439.9523620605469 + 1308.3087158203125j),
        (1231.9859619140625 + 278.393310546875j),
        (784.6526489257812 - 825.4915161132812j),
        (-285.23907470703125 - 970.409912109375j),
        (-863.5292358398438 - 193.4509735107422j),
        (-492.74755859375 + 584.5517578125j),
        (271.3939208984375 + 598.0994873046875j),
        (569.4842529296875 - 14.173060417175293j),
        (168.1858367919922 - 482.6431579589844j),
        (-378.22344970703125 - 304.1585388183594j),
        (-420.8644104003906 + 248.274169921875j),
        (64.37186431884766 + 506.00762939453125j),
        (507.66192626953125 + 175.49806213378906j),
        (416.97412109375 - 373.3819885253906j),
        (-102.6439208984375 - 561.9661254882812j),
        (-521.9799194335938 - 225.365966796875j),
        (-474.2726135253906 + 281.3768615722656j),
        (-71.60576629638672 + 518.2411499023438j),
        (318.74188232421875 + 373.2025451660156j),
        (464.4137878417969 + 37.19683837890625j),
        (367.1384582519531 - 281.807373046875j),
        (94.50050354003906 - 483.183349609375j),
        (-295.8297424316406 - 469.1922607421875j),
        (-627.8256225585938 - 127.0148696899414j),
        (-583.3247680664062 + 451.62396240234375j),
        (-12.205272674560547 + 835.037353515625j),
        (730.3890991210938 + 565.7655639648438j),
        (955.7614135742188 - 284.5089416503906j),
        (325.6766662597656 - 998.472412109375j),
        (-667.6729125976562 - 848.8599853515625j),
        (-1080.62353515625 + 100.2799072265625j),
        (-478.8802185058594 + 953.4017944335938j),
        (550.7349243164062 + 867.74267578125j),
        (971.7245483398438 - 46.257694244384766j),
        (390.56304931640625 - 821.3099975585938j),
        (-523.7376708984375 - 665.9739990234375j),
        (-775.0570678710938 + 187.95166015625j),
        (-127.28072357177734 + 760.64013671875j),
        (655.8616943359375 + 413.4488220214844j),
        (669.895263671875 - 454.85986328125j),
        (-132.73406982421875 - 857.7469482421875j),
        (-891.1356811523438 - 301.13720703125j),
        (-758.6077880859375 + 680.4332275390625j),
        (204.68463134765625 + 1077.440673828125j),
        (1086.0107421875 + 434.26806640625j),
        (1019.0123901367188 - 699.5228881835938j),
        (5.998669147491455 - 1295.8741455078125j),
        (-1094.8546142578125 - 790.7841796875j),
        (-1334.421142578125 + 430.7089538574219j),
        (-471.70184326171875 + 1374.6011962890625j),
        (838.28955078125 + 1251.29736328125j),
        (1559.367431640625 + 100.52810668945312j),
        (1072.203369140625 - 1219.249267578125j),
        (-322.8441467285156 - 1658.1134033203125j),
        (-1572.9678955078125 - 786.2866821289062j),
        (-1644.64501953125 + 802.1537475585938j),
        (-371.5653076171875 + 1864.2593994140625j),
        (1308.318359375 + 1471.89990234375j),
        (2025.5660400390625 - 166.74468994140625j),
        (1108.5147705078125 - 1769.1793212890625j),
        (-774.2650146484375 - 1987.6527099609375j),
        (-2090.1806640625 - 570.140869140625j),
        (-1716.0771484375 + 1354.80517578125j),
        (73.10182189941406 + 2190.296630859375j),
        (1799.483642578125 + 1232.2833251953125j),
        (2032.714111328125 - 713.4841918945312j),
        (612.2112426757812 - 2020.60205078125j),
        (-1235.2659912109375 - 1638.8162841796875j),
        (-1977.297607421875 + 32.23268127441406j),
        (-1085.078125 + 1545.3419189453125j),
        (580.6546020507812 + 1688.5220947265625j),
        (1598.873779296875 + 486.4198913574219j),
        (1232.4154052734375 - 935.58837890625j),
        (-32.25548553466797 - 1415.9495849609375j),
        (-1053.2720947265625 - 729.284423828125j),
        (-1082.1085205078125 + 373.9307556152344j),
        (-305.974853515625 + 963.8233032226562j),
        (506.7047424316406 + 724.440185546875j),
        (766.4578857421875 + 48.469215393066406j),
        (463.76641845703125 - 481.63800048828125j),
        (-37.261085510253906 - 591.7978515625j),
        (-414.2210693359375 - 360.22100830078125j),
        (-541.7673950195312 + 31.44391441345215j),
        (-381.5037841796875 + 429.8420104980469j),
        (62.22858428955078 + 636.4482421875j),
        (597.6738891601562 + 416.90411376953125j),
        (800.1161499023438 - 237.76010131835938j),
        (335.22943115234375 - 890.0291748046875j),
        (-585.0151977539062 - 898.8955078125j),
        (-1192.9080810546875 - 56.53915023803711j),
        (-811.11376953125 + 1031.153076171875j),
        (403.0852966308594 + 1363.2027587890625j),
        (1438.1492919921875 + 490.4816589355469j),
        (1299.3638916015625 - 938.29833984375j),
        (-12.590837478637695 - 1668.88037109375j),
        (-1405.097412109375 - 985.8861083984375j),
        (-1645.8026123046875 + 578.5010375976562j),
        (-491.81744384765625 + 1682.5198974609375j),
        (1072.12255859375 + 1373.24658203125j),
        (1712.009521484375 - 66.240966796875j),
        (921.6016845703125 - 1389.68798828125j),
        (-568.3739013671875 - 1502.4246826171875j),
        (-1479.7525634765625 - 394.1646423339844j),
        (-1112.8404541015625 + 920.3233032226562j),
        (100.93111419677734 + 1342.301025390625j),
        (1064.920654296875 + 632.95068359375j),
        (1021.9495849609375 - 467.44622802734375j),
        (169.08358764648438 - 988.1494140625j),
        (-635.845703125 - 603.50634765625j),
        (-727.8802490234375 + 172.27691650390625j),
        (-201.88320922851562 + 584.6067504882812j),
        (313.58135986328125 + 376.9619445800781j),
        (360.065185546875 - 66.9429931640625j),
        (63.16919708251953 - 242.04905700683594j),
        (-132.58493041992188 - 73.14615631103516j),
        (-27.849571228027344 + 99.71968841552734j),
        (141.71051025390625 + 10.149182319641113j),
        (69.1510238647461 - 199.6261749267578j),
        (-199.30995178222656 - 192.77391052246094j),
        (-316.2856750488281 + 98.86055755615234j),
        (-83.21517181396484 + 360.5453186035156j),
        (279.3336486816406 + 273.898681640625j),
        (383.2545166015625 - 90.28992462158203j),
        (128.6056365966797 - 354.58306884765625j),
        (-199.72076416015625 - 277.6192626953125j),
        (-289.86474609375 - 2.1300747394561768j),
        (-144.86575317382812 + 171.61199951171875j),
        (3.0831351280212402 + 155.20407104492188j),
        (37.86552047729492 + 104.5933837890625j),
        (74.46469116210938 + 122.4101791381836j),
        (207.14752197265625 + 86.66552734375j),
        (285.7414245605469 - 134.5946502685547j),
        (86.82210540771484 - 393.2875061035156j),
        (-318.82440185546875 - 355.91986083984375j),
        (-531.7201538085938 + 67.47595977783203j),
        (-257.4722900390625 + 512.5247192382812j),
        (294.28314208984375 + 509.4446716308594j),
        (578.9171142578125 + 27.17749786376953j),
        (311.37677001953125 - 451.0508117675781j),
        (-206.43350219726562 - 451.6064453125j),
        (-428.27203369140625 - 33.4087028503418j),
        (-179.38262939453125 + 305.8042297363281j),
        (178.8179931640625 + 219.9225311279297j),
        (210.8596649169922 - 105.90128326416016j),
        (-76.04891967773438 - 220.80545043945312j),
        (-272.5445556640625 + 28.99479866027832j),
        (-87.20144653320312 + 322.6208801269531j),
        (293.8397521972656 + 264.0213928222656j),
        (424.7808837890625 - 137.71669006347656j),
        (117.95597076416016 - 469.24951171875j),
        (-341.45703125 - 371.9692687988281j),
        (-504.2298889160156 + 74.21741485595703j),
        (-221.46697998046875 + 447.39508056640625j),
        (228.46949768066406 + 417.1265869140625j),
        (438.83392333984375 + 47.22177505493164j),
        (258.8426513671875 - 304.09686279296875j),
        (-101.19734191894531 - 337.834228515625j),
        (-296.6463623046875 - 72.47125244140625j),
        (-169.4198760986328 + 202.3028564453125j),
        (117.29533386230469 + 208.3036651611328j),
        (238.85415649414062 - 52.62094497680664j),
        (31.919893264770508 - 284.4852600097656j),
        (-308.45050048828125 - 180.21630859375j),
        (-381.2581787109375 + 233.24208068847656j),
        (4.1608662605285645 + 546.8287963867188j),
        (550.142822265625 + 346.69525146484375j),
        (685.6243286132812 - 307.7081604003906j),
        (146.9575653076172 - 832.4761962890625j),
        (-660.6621704101562 - 650.3602294921875j),
        (-975.3693237304688 + 183.76220703125j),
        (-427.3472595214844 + 946.4254760742188j),
        (539.9969482421875 + 914.9953002929688j),
        (1059.668701171875 + 89.17404174804688j),
        (675.4036865234375 - 792.6438598632812j),
        (-234.20790100097656 - 969.7981567382812j),
        (-861.1040649414062 - 364.2635803222656j),
        (-743.7108154296875 + 426.2837219238281j),
        (-113.80331420898438 + 761.2048950195312j),
        (451.1782531738281 + 507.4854431152344j),
        (593.728759765625 + 3.370931625366211j),
        (369.4726257324219 - 371.15631103515625j),
        (15.120086669921875 - 478.8143615722656j),
        (-301.4694519042969 - 355.24371337890625j),
        (-479.189453125 - 52.57445526123047j),
        (-397.25140380859375 + 332.4573059082031j),
        (-9.055815696716309 + 562.68359375j),
        (470.4050598144531 + 384.6501770019531j),
        (627.0022583007812 - 159.92111206054688j),
        (238.8196563720703 - 634.379638671875j),
        (-406.5791320800781 - 567.9391479492188j),
        (-707.8330078125 + 34.475711822509766j),
        (-345.68914794921875 + 619.0547485351562j),
        (346.5264892578125 + 609.1443481445312j),
        (685.738525390625 + 9.395039558410645j),
        (337.4090881347656 - 574.5166625976562j),
        (-329.5458068847656 - 554.30126953125j),
        (-623.8472900390625 + 29.260467529296875j),
        (-253.54043579101562 + 552.9640502929688j),
        (373.869873046875 + 468.0729675292969j),
        (585.5476684570312 - 126.09699249267578j),
        (151.23049926757812 - 590.210205078125j),
        (-473.2769470214844 - 415.8533020019531j),
        (-615.9526977539062 + 237.61651611328125j),
        (-88.73763275146484 + 692.4515991210938j),
        (596.22265625 + 441.3269958496094j),
        (723.5203247070312 - 315.3165588378906j),
        (102.44583892822266 - 832.3944091796875j),
        (-699.4541015625 - 547.91650390625j),
        (-878.0409545898438 + 328.29730224609375j),
        (-190.92926025390625 + 965.5521240234375j),
        (751.3858642578125 + 701.7022094726562j),
        (1031.74658203125 - 276.47454833984375j),
        (322.8696594238281 - 1055.48828125j),
        (-743.5953369140625 - 856.6619873046875j),
        (-1145.433837890625 + 179.6462860107422j),
        (-463.2474365234375 + 1083.5306396484375j),
        (678.7537841796875 + 978.2498168945312j),
        (1194.263671875 - 54.107879638671875j),
        (589.9885864257812 - 1035.783203125j),
        (-552.4454345703125 - 1041.921875j),
        (-1152.5289306640625 - 93.7615966796875j),
        (-682.515380859375 + 890.6288452148438j),
        (358.1138610839844 + 1014.2964477539062j),
        (986.4109497070312 + 245.4470672607422j),
        (699.4478759765625 - 634.484130859375j),
        (-118.2936782836914 - 851.3685302734375j),
        (-679.7514038085938 - 346.6108703613281j),
        (-583.920654296875 + 299.97259521484375j),
        (-93.38773345947266 + 534.4118041992188j),
        (277.4806213378906 + 322.108154296875j),
        (310.6400146484375 + 16.418848037719727j),
        (179.2066650390625 - 118.01221466064453j),
        (102.47124481201172 - 135.18826293945312j),
        (61.63823699951172 - 194.25399780273438j),
        (-88.59552764892578 - 264.07635498046875j),
        (-321.46783447265625 - 156.2101593017578j),
        (-390.343994140625 + 170.92481994628906j),
        (-123.35735321044922 + 462.2459411621094j),
        (309.9789733886719 + 406.2012939453125j),
        (522 + 0j),
    ]
)
beam_profile = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0,
        9.0,
        25.0,
        83.0,
        117.0,
        254.0,
        380.0,
        487.0,
        656.0,
        824.0,
        1063.0,
        1337.0,
        1605.0,
        1920.0,
        2218.0,
        2628.0,
        2978.0,
        3472.0,
        3842.0,
        4301.0,
        4764.0,
        5093.0,
        5778.0,
        6406.0,
        6780.0,
        7268.0,
        7842.0,
        8502.0,
        9056.0,
        9824.0,
        10385.0,
        10886.0,
        11322.0,
        12172.0,
        12579.0,
        13208.0,
        13622.0,
        13888.0,
        14733.0,
        15470.0,
        15726.0,
        16297.0,
        16701.0,
        16855.0,
        17430.0,
        17885.0,
        17950.0,
        18644.0,
        18813.0,
        18874.0,
        19355.0,
        19493.0,
        19459.0,
        19639.0,
        19891.0,
        19691.0,
        19988.0,
        19831.0,
        19753.0,
        19393.0,
        19532.0,
        19216.0,
        18621.0,
        18480.0,
        18464.0,
        18111.0,
        17736.0,
        17201.0,
        16787.0,
        16410.0,
        15674.0,
        15109.0,
        14674.0,
        14177.0,
        13361.0,
        12983.0,
        12363.0,
        11932.0,
        11056.0,
        10309.0,
        9796.0,
        9058.0,
        8253.0,
        7714.0,
        7182.0,
        6548.0,
        5907.0,
        5398.0,
        4889.0,
        4307.0,
        3728.0,
        3202.0,
        2726.0,
        2368.0,
        1893.0,
        1512.0,
        1179.0,
        941.0,
        716.0,
        448.0,
        305.0,
        172.0,
        78.0,
        31.0,
        5.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)


class TestHeadlessSolvers(unittest.TestCase):
    def test_comp(self):
        t_rf = 7.706144104735e-10
        prof_ = StaticProfile.from_rad(
            1e-10 / t_rf * 2 * np.pi, 2 * np.pi, 2**9, t_rf
        )
        prof = Mock(StaticProfile)
        prof.beam_spectrum.return_value = beam_spectrum
        prof.hist_y = beam_profile
        prof.cut_left = prof_.cut_left
        prof.cut_right = prof_.cut_right
        prof.hist_x = prof_.hist_x
        prof.hist_step = prof_.hist_step
        Q_factor = 1.76e6
        beam = Mock(BeamBaseClass)
        beam.n_macroparticles_partial.return_value = 1e6
        prof.hist_y_to_density_factor = 1 / beam.n_macroparticles_partial()
        beam.intensity = 2.4e12
        beam.ratio = (
            beam.intensity / beam.n_macroparticles_partial.return_value
        )
        beam.particle_type = mu_plus

        wf_td = WakeField.headless(
            sources=(
                Resonators(
                    shunt_impedances=np.array([518 * Q_factor]),
                    center_frequencies=np.array([1 / t_rf]),
                    quality_factors=np.array([Q_factor]),
                ),
            ),
            solver=TimeDomainFftSolver(),
            profile=prof,
            beam=beam,
        )

        ind_voltage_td = wf_td.solver.calc_induced_voltage(beam=beam)
        wf_convol = WakeField.headless(
            sources=(
                Resonators(
                    shunt_impedances=np.array([518 * Q_factor]),
                    center_frequencies=np.array([1 / t_rf]),
                    quality_factors=np.array([Q_factor]),
                ),
            ),
            solver=SingleTurnResonatorConvolutionSolver(),
            profile=prof,
            beam=beam,
        )
        ind_voltage_res = wf_convol.solver.calc_induced_voltage(beam=beam)

        DEBUG_PLOT = False
        if DEBUG_PLOT:
            fig, ax = plt.subplots(1, 1)
            ax.plot(
                ind_voltage_td[: len(wf_td.profile.hist_y)],
                label="fft time domain",
            )
            ax.plot(ind_voltage_res, label="resonator convolution", ls="--")
            ax.legend()

            ax.plot(
                wf_td.profile.hist_y
                / max(wf_td.profile.hist_y)
                * np.min(ind_voltage_res)
            )
            plt.show()

        assert np.allclose(
            ind_voltage_res,
            ind_voltage_td[: len(wf_td.profile.hist_y)],
            atol=15,
        )


class TestContinuousMultiTurnTimeDomainSolver(unittest.TestCase):
    def test_update_wake_kernel_fails(self):
        from blond.testing.mocks import beam_mock, static_profile_mock

        prof = StaticProfile(cut_left=-1e-9, cut_right=1e-9, n_bins=128)

        prof.hist_y_to_density_factor = 0.3
        prof._hist_y = np.array(np.exp(-((np.arange(128) - 64) ** 2) / 1e2))

        beam_mock.particle_type = uranium_29
        beam_mock.intensity = 1e-13

        class FaultyResonators:
            def get_wake(self):  # emulate wroing implementation
                return

        wf_mutli = WakeField.headless(
            sources=(FaultyResonators(),),
            solver=ContinuousMultiTurnTimeDomainSolver(n_turns=10),
            profile=prof,
            beam=beam_mock,
        )
        with self.assertRaises(TypeError):
            wf_mutli.solver._update_wake_kernel()

        class FaultyResonators2:
            def get_cake(self):  # emulate wroing implementation
                return

        with self.assertRaisesRegex(
            AttributeError, "should implement `TimeDomain.get_wake`"
        ):
            wf_mutli = WakeField.headless(
                sources=(FaultyResonators2(),),
                solver=ContinuousMultiTurnTimeDomainSolver(n_turns=10),
                profile=prof,
                beam=beam_mock,
            )

    def test_calc_induced_voltage_assert_profile_length_correct(self):
        t_rf = 7.706144104735e-10
        Q_factor = 1.76e6
        from blond.testing.mocks import beam_mock, static_profile_mock

        prof = StaticProfile(cut_left=-1e-9, cut_right=1e-9, n_bins=128)

        prof.hist_y_to_density_factor = 0.3
        prof._hist_y = np.array(np.exp(-((np.arange(128) - 64) ** 2) / 1e2))

        beam_mock.particle_type = uranium_29
        beam_mock.intensity = 1e-13

        wf_mutli = WakeField.headless(
            sources=(
                Resonators(
                    shunt_impedances=np.array([518 * Q_factor]),
                    center_frequencies=np.array([1 / t_rf]),
                    quality_factors=np.array([Q_factor]),
                ),
            ),
            solver=ContinuousMultiTurnTimeDomainSolver(n_turns=10),
            profile=prof,
            beam=beam_mock,
        )
        wf_mutli.solver._simulation.magnetic_cycle.get_t_rev_init.return_value = 1e12
        with self.assertRaises(AssertionError):
            wf_mutli.solver._assert_profile_length_correct()

    def test_calc_induced_voltage_assert_warns_profile(self):
        t_rf = 7.706144104735e-10
        Q_factor = 1.76e6
        from blond.testing.mocks import beam_mock, static_profile_mock

        prof = DynamicProfileConstNBins(n_bins=128)
        prof.cut_left = -1e-9
        prof.cut_right = 1e-9
        prof._hist_x = np.linspace(prof.cut_left, prof.cut_right, 128)

        prof.hist_y_to_density_factor = 0.3
        prof._hist_y = np.array(np.exp(-((np.arange(128) - 64) ** 2) / 1e2))

        beam_mock.particle_type = uranium_29
        beam_mock.intensity = 1e-13
        with self.assertWarnsRegex(UserWarning, "Expected StaticProfile"):
            wf_mutli = WakeField.headless(
                sources=(
                    Resonators(
                        shunt_impedances=np.array([518 * Q_factor]),
                        center_frequencies=np.array([1 / t_rf]),
                        quality_factors=np.array([Q_factor]),
                    ),
                ),
                solver=ContinuousMultiTurnTimeDomainSolver(n_turns=10),
                profile=prof,
                beam=beam_mock,
            )

    def test_calc_induced_voltage_single_turn(self):
        t_rf = 7.706144104735e-10
        Q_factor = 1.76e6
        from blond.testing.mocks import beam_mock, static_profile_mock

        prof = StaticProfile(cut_left=-1e-9, cut_right=1e-9, n_bins=128)

        prof.hist_y_to_density_factor = 0.3
        prof._hist_y = np.array(np.exp(-((np.arange(128) - 64) ** 2) / 1e2))

        beam_mock.particle_type = uranium_29
        beam_mock.intensity = 1e-13

        wf_mutli = WakeField.headless(
            sources=(
                Resonators(
                    shunt_impedances=np.array([518 * Q_factor]),
                    center_frequencies=np.array([1 / t_rf]),
                    quality_factors=np.array([Q_factor]),
                ),
            ),
            solver=ContinuousMultiTurnTimeDomainSolver(n_turns=10),
            profile=prof,
            beam=beam_mock,
        )
        wf_single = WakeField.headless(
            sources=(
                Resonators(
                    shunt_impedances=np.array([518 * Q_factor]),
                    center_frequencies=np.array([1 / t_rf]),
                    quality_factors=np.array([Q_factor]),
                ),
            ),
            solver=TimeDomainFftSolver(allow_next_fast_len=False),
            profile=prof,
            beam=beam_mock,
        )
        offset = 1.8e-17
        wf_single.calc_induced_voltage(beam=beam_mock)
        wf_mutli.calc_induced_voltage(beam=beam_mock)
        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.subplot(3, 1, 1)
            plt.plot(prof.hist_y)
            plt.subplot(3, 1, 2)
            plt.plot(wf_mutli._induced_voltage, label="wf_mutli")
            plt.plot(wf_single._induced_voltage, "--", label="wf_single")
            plt.subplot(3, 1, 3)
            plt.plot(wf_mutli.induced_voltage - wf_single.induced_voltage)
            plt.legend()
            plt.show()
        np.testing.assert_allclose(
            wf_mutli.induced_voltage + offset,
            wf_single.induced_voltage + offset,
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )

    def test_calc_induced_voltage_multi_turn(self):
        t_rf = 7.706144104735e-10
        Q_factor = 1.76e6
        sources = (
            Resonators(
                shunt_impedances=np.array([518 * Q_factor]),
                center_frequencies=np.array([1 / t_rf]),
                quality_factors=np.array([Q_factor]),
            ),
            Resonators(
                shunt_impedances=np.array([518 * Q_factor]),
                center_frequencies=np.array([1 / t_rf]),
                quality_factors=np.array([Q_factor]),
            ),
        )

        from blond.testing.mocks import beam_mock, static_profile_mock

        prof_single = StaticProfile(cut_left=-1e-9, cut_right=1e-9, n_bins=128)

        prof_single.hist_y_to_density_factor = 0.3
        prof_single._hist_y = np.array(
            np.exp(-((np.arange(128) - 64) ** 2) / 1e2)
        )

        prof_two_turns = StaticProfile(
            cut_left=-1e-9, cut_right=1e-9 + 2e-9, n_bins=2 * 128
        )

        prof_two_turns.hist_y_to_density_factor = 0.3
        prof_two_turns._hist_y = np.concatenate(
            (prof_single.hist_y, 0.5 * prof_single.hist_y)
        )

        beam_mock.particle_type = uranium_29
        beam_mock.intensity = 1e-13

        wf_mutli = WakeField.headless(
            sources=sources,
            solver=ContinuousMultiTurnTimeDomainSolver(n_turns=10),
            profile=prof_single,
            beam=beam_mock,
        )
        wf_single = WakeField.headless(
            sources=sources,
            solver=TimeDomainFftSolver(allow_next_fast_len=False),
            profile=prof_two_turns,
            beam=beam_mock,
        )
        wf_mutli.solver._simulation.magnetic_cycle.get_t_rev_init.return_value = (
            prof_single.cut_right - prof_single.cut_left
        )
        offset = 1.8e-17
        wf_single.calc_induced_voltage(beam=beam_mock)

        wf_mutli.calc_induced_voltage(beam=beam_mock)
        prof_single._hist_y *= 0.5
        wf_mutli.calc_induced_voltage(beam=beam_mock)  # second turn

        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.figure()
            plt.plot(
                np.fft.irfft(wf_single.solver._wake_imp_y), label="wf_single"
            )
            plt.plot(wf_mutli.solver._wake_kernel, label="wf_mutli")
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(prof_single.hist_y * 2)
            plt.plot(prof_single.hist_y)
            plt.plot(prof_two_turns.hist_y)
            plt.subplot(3, 1, 2)
            plt.plot(wf_mutli._induced_voltage, label="wf_mutli")
            plt.plot(
                wf_single._induced_voltage[-128:], "--", label="wf_single"
            )
            plt.subplot(3, 1, 3)
            plt.plot(
                wf_mutli.induced_voltage - wf_single.induced_voltage[-128:]
            )
            plt.legend()
            plt.show()
        np.testing.assert_allclose(
            wf_mutli.induced_voltage + offset,
            wf_single.induced_voltage[-128:] + offset,
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )
