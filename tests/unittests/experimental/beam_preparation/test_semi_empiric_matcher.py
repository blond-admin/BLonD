import contextlib
import doctest
import io
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    BeamStatisticsOncePerTurn,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    WakeField,
    backend,
    momentum_compaction_factor,
    proton,
)
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    _POTENTIAL_WELL_OVERSAMPLING,
    SemiEmpiricMatcher,
    _curvature_at_minimum,
    get_grid_correction_from_one_turn_map,
    get_hamilton_semi_analytic,
    get_one_turn_map_empiric,
)
from blond.experimental.simulation.warmup import warmup
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.testing.backend_testing import BLonDTestCase
from blond.testing.simulation import (
    SimulationTwoRFStations,
    SimulationTwoRFStationsWithWake,
)


class TestSemiEmpiricMatcher(BLonDTestCase):
    def test_roughly_correct_no_intensity_above_transition(self):
        # check if the mean and the 10% and 90% percentiles are correct

        # pinned values
        expected_dt = {
            10: 9.357760148747992e-10,
            50: 1.2487530725048837e-09,
            90: 1.5592096439245792e-09,
        }
        expected_dE = {
            10: -203033613.1606067,
            50: -497237.79234934977,
            90: 200762720.55608103,
        }
        sim = SimulationTwoRFStations()
        self._test_matching(sim)

        DEV_PLOT = False
        if DEV_PLOT:
            idx = np.argmax(sim.beam1.read_partial_dt())
            data = np.ones((1000, 2))
            data[:, :] = np.nan

            def my_callback(simulation: Simulation, beam: Beam):
                if simulation.turn_counter.value % 1 != 0:
                    return
                plt.subplot(2, 1, 1)
                plt.cla()
                beam.plot_hist2d(range=((0.7e-9, 1.8e-9), (-3.5e8, 3.5e8)))
                data[simulation.turn_counter.value % data.shape[0], 0] = (
                    sim.beam1.read_partial_dt()[idx]
                )
                data[simulation.turn_counter.value % data.shape[0], 1] = (
                    sim.beam1.read_partial_dE()[idx]
                )
                plt.plot(data[:, 0], data[:, 1], ".")
                plt.axhline(beam.read_partial_dE().mean())
                plt.axvline(beam.read_partial_dt().mean())
                plt.subplot(2, 1, 2)
                if simulation.turn_counter.value == 0:
                    plt.cla()
                plt.hist(
                    beam.read_partial_dt(),
                    bins=256,
                    histtype="step",
                    density=True,
                )
                plt.draw()
                plt.pause(0.1)

            sim.simulation.run_simulation(
                beams=(sim.beam1,),
                callbacks=my_callback,
                n_turns=1e6,
            )
        for percentile in (10, 50, 90):
            percentile_dt = float(
                backend.percentile(sim.beam1.read_partial_dt(), percentile)
            )
            percentile_dE = float(
                backend.percentile(sim.beam1.read_partial_dE(), percentile)
            )
            np.testing.assert_allclose(
                expected_dt[percentile],
                percentile_dt,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                expected_dE[percentile],
                percentile_dE,
                rtol=1e-4,
            )

    def test_roughly_correct_intensity_above_transition(self):

        sim = SimulationTwoRFStationsWithWake()
        wakefield = sim.simulation.ring.elements.get_element(WakeField)

        wakefield.track_profile = False
        sim.simulation.print_one_turn_execution_order()
        wakefield.track_profile = True
        sim.simulation.print_one_turn_execution_order()

        self._test_matching(sim)
        DEV_PLOT = False
        DEV_PLOT2 = False
        if DEV_PLOT:

            def my_callback(simulation: Simulation, beam: Beam):
                if simulation.turn_counter.value % 10 != 0:
                    return
                plt.subplot(2, 1, 1)
                plt.cla()
                beam.plot_hist2d(range=((0.7e-9, 1.8e-9), (-3.5e8, 3.5e8)))
                plt.axhline(float(beam.read_partial_dE().mean()))
                plt.axvline(float(beam.read_partial_dt().mean()))
                plt.subplot(2, 1, 2)
                plt.hist(
                    copy_to_cpu(beam.read_partial_dt()),
                    bins=256,
                    histtype="step",
                    density=True,
                )
                plt.draw()
                plt.pause(0.1)

            sim.simulation.turn_counter.value = 0
            my_callback(simulation=sim.simulation, beam=sim.beam1)
            sim.simulation.run_simulation(
                beams=(sim.beam1,), callbacks=my_callback
            )
        # pinned values
        expected_dt = {
            10: 8.93158991410339e-10,
            50: 1.1922249824269493e-09,
            90: 1.487705227890184e-09,
        }
        expected_dE = {
            10: -202860014.82034442,
            50: -491660.89176954154,
            90: 200443301.97634044,
        }
        if DEV_PLOT2:
            sim.beam1.plot_scatter()
            plt.show()
        for percentile in (10, 50, 90):
            percentile_dt = float(
                backend.percentile(sim.beam1.read_partial_dt(), percentile)
            )
            percentile_dE = float(
                backend.percentile(sim.beam1.read_partial_dE(), percentile)
            )

            np.testing.assert_allclose(
                expected_dt[percentile],
                percentile_dt,
                rtol=2e-4,
            )
            np.testing.assert_allclose(
                expected_dE[percentile],
                percentile_dE,
                rtol=2e-4,
            )

    def test_roughly_correct_no_intensity_below_transition(self):
        # check if the mean and the 10% and 90% percentiles are correct

        # pinned values
        expected_dt = {
            10: 2.183409806558131e-09,
            50: 2.496288490756453e-09,
            90: 2.8067353359529354e-09,
        }
        expected_dE = {
            10: -2094341953.5976465,
            50: -5129130.852868866,
            90: 2070917134.525886,
        }
        sim = SimulationTwoRFStations(below_transition_crossing=True)
        self._test_matching(sim, below_transition_crossing=True)

        DEV_PLOT = False
        if DEV_PLOT:
            idx = np.argmax(sim.beam1.read_partial_dt())
            data = np.ones((1000, 2))
            data[:, :] = np.nan

            def my_callback(simulation: Simulation, beam: Beam):
                if simulation.turn_counter.value % 1 != 0:
                    return
                plt.subplot(2, 1, 1)
                plt.cla()
                beam.plot_hist2d()
                data[simulation.turn_counter.value % data.shape[0], 0] = (
                    sim.beam1.read_partial_dt()[idx]
                )
                data[simulation.turn_counter.value % data.shape[0], 1] = (
                    sim.beam1.read_partial_dE()[idx]
                )
                plt.plot(data[:, 0], data[:, 1], ".")
                plt.axhline(beam.read_partial_dE().mean())
                plt.axvline(beam.read_partial_dt().mean())
                plt.subplot(2, 1, 2)
                if simulation.turn_counter.value == 0:
                    plt.cla()
                plt.hist(
                    beam.read_partial_dt(),
                    bins=256,
                    histtype="step",
                    density=True,
                )
                plt.draw()
                plt.pause(0.1)

            sim.simulation.run_simulation(
                beams=(sim.beam1,),
                callbacks=my_callback,
                n_turns=1e6,
            )
        for percentile in (10, 50, 90):
            percentile_dt = float(
                backend.percentile(sim.beam1.read_partial_dt(), percentile)
            )
            percentile_dE = float(
                backend.percentile(sim.beam1.read_partial_dE(), percentile)
            )
            np.testing.assert_allclose(
                expected_dt[percentile],
                percentile_dt,
                rtol=1e-4,
            )

            np.testing.assert_allclose(
                expected_dE[percentile],
                percentile_dE,
                rtol=1e-4,
            )

    def test_roughly_correct_intensity_below_transition(self):
        sim = SimulationTwoRFStationsWithWake(below_transition_crossing=True)
        self._test_matching(sim, below_transition_crossing=True)
        DEV_PLOT = False
        if DEV_PLOT:

            def my_callback(simulation: Simulation, beam: Beam):
                if simulation.turn_counter.value % 10 != 0:
                    return
                plt.subplot(2, 1, 1)
                plt.cla()
                beam.plot_hist2d()
                plt.axhline(beam.read_partial_dE().mean())
                plt.axvline(beam.read_partial_dt().mean())
                plt.subplot(2, 1, 2)
                plt.hist(
                    beam.read_partial_dt(),
                    bins=256,
                    histtype="step",
                    density=True,
                )
                plt.draw()
                plt.draw()
                plt.pause(0.1)

            sim.simulation.turn_counter.value = 0
            my_callback(simulation=sim.simulation, beam=sim.beam1)
            sim.simulation.run_simulation(
                beams=(sim.beam1,), callbacks=my_callback
            )
        # pinned values
        expected_dt = {
            10: 2.197717254972563e-09,
            50: 2.5039150672533025e-09,
            90: 2.8102421167028965e-09,
        }
        expected_dE = {
            10: -2092347616.032768,
            50: -5220587.534548046,
            90: 2066870065.0492826,
        }
        for percentile in (10, 50, 90):
            percentile_dt = float(
                backend.percentile(sim.beam1.read_partial_dt(), percentile)
            )
            percentile_dE = float(
                backend.percentile(sim.beam1.read_partial_dE(), percentile)
            )
            np.testing.assert_allclose(
                expected_dt[percentile],
                percentile_dt,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                expected_dE[percentile],
                percentile_dE,
                rtol=1e-4,
            )

    def _test_matching(self, sim, below_transition_crossing=False):
        simulation = sim.simulation
        beam = sim.beam1
        t_rev = simulation.magnetic_cycle.get_t_rev_init(
            simulation.ring.circumference,
            particle_type=beam.particle_type,
        )
        ts = (
            np.linspace(
                0 + (t_rev / 2 if below_transition_crossing else 0),
                t_rev + (t_rev / 2 if below_transition_crossing else 0),
            )
            / 36540
        )
        # actively change the harmonic off the revolution time.
        # matching should still work
        # cav = sim.simulation.ring.elements.get_element(MultiHarmonicCavity)
        # cav.harmonic = 10*33000 * np.ones(len(cav.harmonic), backend.float)
        # cav = sim.simulation.ring.elements.get_element(SingleHarmonicRFStation)
        # cav.harmonic = 10*33000

        matcher = SemiEmpiricMatcher(
            time_limit=(ts.min(), ts.max()),
            hamilton_to_density_kwargs=dict(
                hamilton_max=100,
                density_modifier=4,
            ),
            n_macroparticles=1e5,
            internal_grid_shape=(512 - 1, 512 - 1),
            increment_intensity_effects_until_iteration_i=10,
            maxiter_intensity_effects=1000,
            tolerance_potential_well=0.000001,
            animate=False,
            debug=True,
        )
        sim.simulation.prepare_beam(
            beam=sim.beam1,
            preparation_routine=matcher,
        )
        return matcher


class TestSemiEmpiricMatcherPotentialWellGrid(BLonDTestCase):
    def test_oversampled_potential_well_downsamples_onto_ts(self):
        # The potential well is evaluated on an oversampled grid and then
        # down-sampled with ``[::_POTENTIAL_WELL_OVERSAMPLING]``. The
        # down-sampled points must coincide with the matcher grid ``ts``,
        # otherwise the Hamiltonian is shifted in time and the matched
        # bunch performs a dipole oscillation.
        sim = SimulationTwoRFStations()
        t_rev = sim.simulation.magnetic_cycle.get_t_rev_init(
            sim.simulation.ring.circumference,
            particle_type=sim.beam1.particle_type,
        )
        n_time = 128 - 1
        ts = np.linspace(0, t_rev) / 36540
        expected_dt = np.linspace(ts.min(), ts.max(), n_time)
        matcher = SemiEmpiricMatcher(
            time_limit=(ts.min(), ts.max()),
            hamilton_to_density_kwargs=dict(
                hamilton_max=100,
                density_modifier=4,
            ),
            n_macroparticles=1e3,
            internal_grid_shape=(n_time, n_time),
            verbose=False,
        )
        get_potential_well_empiric = Simulation.get_potential_well_empiric
        with mock.patch.object(
            Simulation,
            "get_potential_well_empiric",
            autospec=True,
            side_effect=get_potential_well_empiric,
        ) as spy:
            sim.simulation.prepare_beam(
                beam=sim.beam1,
                preparation_routine=matcher,
            )
        self.assertGreater(spy.call_count, 0)
        for call in spy.call_args_list:
            dt = copy_to_cpu(call.kwargs["dt"])
            np.testing.assert_allclose(
                dt[::_POTENTIAL_WELL_OVERSAMPLING],
                expected_dt,
                rtol=0,
                atol=1e-9 * (expected_dt[1] - expected_dt[0]),
            )


def _lhc_kick_then_drift(voltage=6e6):
    """LHC at 450 GeV/c, one RF kick per turn followed by the whole drift."""
    ring = Ring(circumference=26658.883)
    ring.add_elements(
        [
            SingleHarmonicRFStation(harmonic=35640, voltage=voltage, phi_rf=0),
            DriftSimple(
                orbit_length=26658.883,
                momentum_compaction_factor=momentum_compaction_factor(
                    transition_gamma=55.759505
                ),
            ),
        ],
        reorder=False,
    )
    simulation = Simulation(
        ring=ring,
        magnetic_cycle=ConstantMagneticCycle(
            value=450e9, reference_particle=proton
        ),
    )
    return simulation, Beam(intensity=1e9, particle_type=proton)


class TestSemiEmpiricMatcherOneTurnMap(BLonDTestCase):
    """Correction of the matched beam to the tracked one-turn map."""

    drift = 3e-16  # drift per turn, [s/eV]
    kick = 1e14  # slope of the RF kick per turn, [eV/s]

    def _assert_correction(self, one_turn_map, expected_shear):
        mu = np.arccos(0.5 * np.trace(one_turn_map))
        shear_dt_per_dE, deltaE_scale = get_grid_correction_from_one_turn_map(
            one_turn_map=one_turn_map,
            beta_hamiltonian=np.sqrt(self.drift / self.kick),
        )
        np.testing.assert_allclose(shear_dt_per_dE, expected_shear, atol=1e-30)
        # the discrete map has a larger energy spread than the smooth
        # Hamiltonian, by 1 / cos(mu / 2) for all three element orders
        np.testing.assert_allclose(
            deltaE_scale, 1 / np.cos(mu / 2), rtol=1e-12
        )

    def test_grid_correction_kick_then_drift(self):
        drift, kick = self.drift, self.kick
        self._assert_correction(
            np.array([[1 - drift * kick, drift], [-kick, 1]]), drift / 2
        )

    def test_grid_correction_drift_then_kick(self):
        drift, kick = self.drift, self.kick
        self._assert_correction(
            np.array([[1, drift], [-kick, 1 - drift * kick]]), -drift / 2
        )

    def test_grid_correction_symmetric(self):
        drift, kick = self.drift, self.kick
        one_turn_map = np.array(
            [
                [1 - drift * kick / 2, drift * (1 - drift * kick / 4)],
                [-kick, 1 - drift * kick / 2],
            ]
        )
        self._assert_correction(one_turn_map, 0.0)

    def test_get_one_turn_map_empiric_kick_then_drift(self):
        voltage, harmonic = 6e6, 35640
        simulation, _ = _lhc_kick_then_drift(voltage=voltage)
        t_rev = simulation.magnetic_cycle.get_t_rev_init(
            simulation.ring.circumference, particle_type=proton
        )
        omega_rf = 2 * np.pi * harmonic / t_rev

        t_fixed, dE_fixed, one_turn_map = get_one_turn_map_empiric(
            simulation=simulation,
            t_guess=0.45 * t_rev / harmonic,  # off the fixed point
            delta_t=1e-13,
            delta_E=1e4,
            particle_type=proton,
        )

        # above transition, the bucket centre is at omega_rf * t = pi
        np.testing.assert_allclose(t_fixed, np.pi / omega_rf, rtol=1e-9)
        np.testing.assert_allclose(dE_fixed, 0, atol=1e-3)
        # kick, then drift: [[1 - drift kick, drift], [-kick, 1]]
        energy = np.sqrt(450e9**2 + proton.mass**2)
        beta = 450e9 / energy
        alpha = momentum_compaction_factor(transition_gamma=55.759505)
        drift = (
            t_rev * (alpha - (proton.mass / energy) ** 2) / (beta**2 * energy)
        )
        kick = voltage * omega_rf
        np.testing.assert_allclose(
            one_turn_map,
            [[1 - drift * kick, drift], [-kick, 1]],
            rtol=1e-6,
        )

    def test_grid_correction_unstable_raises(self):
        with self.assertRaises(ValueError):
            get_grid_correction_from_one_turn_map(
                one_turn_map=np.array([[1.0, self.drift], [self.kick, 1.0]]),
                beta_hamiltonian=np.sqrt(self.drift / self.kick),
            )

    def _bunch_length_oscillation(self, correct_to_one_turn_map):
        simulation, beam = _lhc_kick_then_drift()
        simulation.prepare_beam(
            beam=beam,
            preparation_routine=SemiEmpiricMatcher(
                time_limit=(0, 2.5e-9),
                n_macroparticles=1e5,
                hamilton_to_density_function=lambda hamilton_2D, **_: np.exp(
                    -hamilton_2D / 1.7
                ),
                hamilton_to_density_kwargs={},
                seed=1,
                verbose=False,
                correct_to_one_turn_map=correct_to_one_turn_map,
            ),
        )
        statistics = BeamStatisticsOncePerTurn(each_turn_i=1)
        simulation.run_simulation(
            beams=(beam,),
            n_turns=400,  # two synchrotron periods
            observe=(statistics,),
            show_progressbar=False,
        )
        bunch_length = copy_to_cpu(statistics.bunch_length)
        return np.ptp(bunch_length) / bunch_length.mean()

    def test_matched_bunch_length_does_not_oscillate(self):
        # With the RF kick before the drift, the beam matched to the smooth
        # Hamiltonian is tilted against the invariant ellipse of the
        # one-turn map, and its length oscillates by about 1.4 % (the
        # synchrotron phase advance is mu = 0.031 per turn).
        self.assertGreater(self._bunch_length_oscillation(False), 0.01)
        self.assertLess(self._bunch_length_oscillation(True), 0.005)

    def _match_small_beam(
        self,
        time_limit_in_rf_periods,
        correct_to_one_turn_map,
        voltage=6e6,
    ):
        simulation, beam = _lhc_kick_then_drift(voltage=voltage)
        rf_period = (
            simulation.magnetic_cycle.get_t_rev_init(
                simulation.ring.circumference, particle_type=proton
            )
            / 35640
        )
        simulation.prepare_beam(
            beam=beam,
            preparation_routine=SemiEmpiricMatcher(
                time_limit=(
                    time_limit_in_rf_periods[0] * rf_period,
                    time_limit_in_rf_periods[1] * rf_period,
                ),
                n_macroparticles=1e3,
                hamilton_to_density_function=lambda hamilton_2D, **_: np.exp(
                    -hamilton_2D / 1.7
                ),
                hamilton_to_density_kwargs={},
                internal_grid_shape=(128 - 1, 128 - 1),
                verbose=False,
                correct_to_one_turn_map=correct_to_one_turn_map,
            ),
        )
        return (
            copy_to_cpu(beam.read_partial_dt()),
            copy_to_cpu(beam.read_partial_dE()),
        )

    def _assert_falls_back_to_uncorrected_beam(
        self, warning_regex, time_limit_in_rf_periods, voltage=6e6
    ):
        with self.assertWarnsRegex(UserWarning, warning_regex):
            dt, dE = self._match_small_beam(
                time_limit_in_rf_periods, True, voltage=voltage
            )
        dt_uncorrected, dE_uncorrected = self._match_small_beam(
            time_limit_in_rf_periods, False, voltage=voltage
        )
        np.testing.assert_array_equal(dt, dt_uncorrected)
        np.testing.assert_array_equal(dE, dE_uncorrected)

    def test_correction_skipped_if_time_limit_misses_bucket_centre(self):
        # Above transition, t = 0 is the unstable fixed point and the bucket
        # centre is half an RF period later. Around t = 0, the potential
        # well curves downwards and is lowest at the edges of `time_limit`.
        self._assert_falls_back_to_uncorrected_beam(
            "does not curve upwards at its minimum", (-0.2, 0.2)
        )

    def test_correction_skipped_if_time_limit_clips_bucket_centre(self):
        # The potential well is lowest at the end of `time_limit`, but the
        # fixed point of the one-turn map is half an RF period after t = 0.
        self._assert_falls_back_to_uncorrected_beam(
            "no fixed point near the potential minimum", (0, 0.4)
        )

    def test_correction_skipped_if_one_turn_map_unstable(self):
        # At 30 GV, drift * kick > 4: the bucket centre is still a fixed
        # point, but the kick-drift map is linearly unstable there.
        self._assert_falls_back_to_uncorrected_beam(
            "not linearly stable", (0, 1), voltage=30e9
        )

    def test_energy_scale_from_analytic_well_on_coarse_grid(self):
        # The curvature of the well must be estimated well below the
        # O(mu**2) energy scale it corrects, also on a coarse grid. An
        # analytic well keeps the normalisation of the empiric one out.
        voltage, harmonic, n_grid = 6e6, 35640, 128 - 1
        simulation, beam = _lhc_kick_then_drift(voltage=voltage)
        beam.reference.total_energy = (
            simulation.magnetic_cycle.get_total_energy_init(
                particle_type=proton
            )
        )
        t_rev = simulation.magnetic_cycle.get_t_rev_init(
            simulation.ring.circumference, particle_type=proton
        )
        omega_rf = 2 * np.pi * harmonic / t_rev
        ts = np.linspace(
            0,
            2 * np.pi / omega_rf,
            (n_grid - 1) * _POTENTIAL_WELL_OVERSAMPLING + 1,
        )
        # -integral(kick dt) / t_rev for the kick voltage * sin(omega_rf t)
        potential_well = voltage * np.cos(omega_rf * ts) / (omega_rf * t_rev)
        eta = float(simulation.ring.calc_average_eta_0(beam.reference.gamma))
        drift_term = abs(eta) / (
            beam.reference.beta**2 * beam.reference.total_energy
        )
        matcher = SemiEmpiricMatcher(
            time_limit=(ts.min(), ts.max()),
            n_macroparticles=1e3,
            hamilton_to_density_kwargs={},
            internal_grid_shape=(n_grid, n_grid),
            verbose=False,
            correct_to_one_turn_map=True,
        )

        _, deltaE_scale = matcher._measure_grid_correction(
            beam=beam,
            simulation=simulation,
            ts=ts,
            potential_well=potential_well,
            eta=eta,
            deltaE_max=float(
                np.sqrt(np.ptp(potential_well) / (0.5 * drift_term))
            ),
        )

        drift_times_kick = t_rev * drift_term * voltage * omega_rf
        mu = np.arccos(1 - drift_times_kick / 2)
        np.testing.assert_allclose(deltaE_scale, 1 / np.cos(mu / 2), rtol=1e-5)


class TestSemiEmpiricMatcherWarmup(BLonDTestCase):
    """Wiring of the frozen-beam ``warmup`` into the matcher iteration."""

    def _run_matching(self, **matcher_kwargs):
        sim = SimulationTwoRFStationsWithWake()
        t_rev = sim.simulation.magnetic_cycle.get_t_rev_init(
            sim.simulation.ring.circumference,
            particle_type=sim.beam1.particle_type,
        )
        ts = np.linspace(0, t_rev) / 36540
        matcher_kwargs = {
            # Exactly two intensity iterations: the convergence error is
            # only computed for `i_intensity > 1`, so with maxiter=2 the
            # loop can never break early.
            "maxiter_intensity_effects": 2,
            "increment_intensity_effects_until_iteration_i": 0,
            "tolerance_potential_well": 1e-6,
            "verbose": False,
            **matcher_kwargs,
        }
        matcher = SemiEmpiricMatcher(
            time_limit=(ts.min(), ts.max()),
            hamilton_to_density_kwargs=dict(
                hamilton_max=100,
                density_modifier=4,
            ),
            n_macroparticles=1e4,
            internal_grid_shape=(128 - 1, 128 - 1),
            **matcher_kwargs,
        )
        sim.simulation.prepare_beam(
            beam=sim.beam1,
            preparation_routine=matcher,
        )
        return matcher

    def test_warmup_off_by_default(self):
        with mock.patch(
            "blond.experimental.beam_preparation.semi_empiric_matcher.warmup"
        ) as warmup_spy:
            self._run_matching()
        warmup_spy.assert_not_called()

    def test_warmup_called_once_per_intensity_iteration(self):
        with mock.patch(
            "blond.experimental.beam_preparation.semi_empiric_matcher.warmup",
            wraps=warmup,
        ) as warmup_spy:
            self._run_matching(n_warmup_turns=2)
        # one call per intensity-effects iteration (see maxiter above)
        self.assertEqual(warmup_spy.call_count, 2)
        for call in warmup_spy.call_args_list:
            self.assertEqual(call.kwargs["n_turns"], 2)


class TestSemiEmpiricMatcherAnimation(BLonDTestCase):
    def test_animate_pauses_for_animate_pause_time(self):
        with mock.patch("matplotlib.pyplot.pause") as pause:
            TestSemiEmpiricMatcherWarmup._run_matching(
                self, animate=True, animate_pause_time=0.0
            )

        self.assertGreater(pause.call_count, 0)
        for call in pause.call_args_list:
            self.assertEqual(call, mock.call(0.0))

    def test_plot_before_first_potential_well_shows_only_histogram(self):
        from blond.core.beam.beams import ProbeBeam

        ts = np.linspace(0, 1e-9, 16 - 1)
        matcher = SemiEmpiricMatcher(
            time_limit=(ts.min(), ts.max()),
            n_macroparticles=1e3,
            hamilton_to_density_kwargs={},
            internal_grid_shape=(16 - 1, 16 - 1),
            verbose=False,
        )
        beam = ProbeBeam(dt=np.linspace(0, 1e-9, 10), particle_type=proton)

        matcher._plot_current_state(beam=beam, i=0, scalar=0.0, ts=ts)

        histogram_axes, potential_well_axes = plt.figure(
            "SemiEmpiricMatcher"
        ).axes
        self.assertGreater(len(histogram_axes.patches), 0)
        self.assertEqual(len(potential_well_axes.lines), 0)


class TestSemiEmpiricMatcherStagnation(BLonDTestCase):
    def test_stall_rolls_back_silently_if_not_verbose(self):
        rematch = SemiEmpiricMatcher._rematch_beam_to_potential_well
        stdout = io.StringIO()
        with (
            mock.patch.object(
                SemiEmpiricMatcher,
                "_rematch_beam_to_potential_well",
                autospec=True,
                side_effect=rematch,
            ) as rematch_spy,
            contextlib.redirect_stdout(stdout),
        ):
            TestSemiEmpiricMatcherWarmup._run_matching(
                self,
                # The first error is computed in the third iteration. It
                # never reaches the tolerance, and without patience the
                # iteration stops at once.
                maxiter_intensity_effects=5,
                tolerance_potential_well=1e-15,
                stagnation_patience=0,
                verbose=False,
            )
        rematch_spy.assert_called_once()
        self.assertEqual(stdout.getvalue(), "")

    def test_stall_rolls_back_to_correction_of_best_iteration(self):
        # Each measurement returns its own, negligibly small correction, so
        # the correction applied in the rollback tells which potential well
        # it was measured on.
        measurements = []  # (potential_well, grid_correction)

        def measure(matcher, **kwargs):
            grid_correction = (0.0, 1.0 + 1e-12 * len(measurements))
            measurements.append(
                (copy_to_cpu(kwargs["potential_well"]), grid_correction)
            )
            return grid_correction

        applied_corrections = []
        populate = SemiEmpiricMatcher._populate_beam_from_hamiltonian

        def populate_and_record(matcher, **kwargs):
            applied_corrections.append(matcher._grid_correction)
            return populate(matcher, **kwargs)

        rematch = SemiEmpiricMatcher._rematch_beam_to_potential_well
        with (
            mock.patch.object(
                SemiEmpiricMatcher,
                "_measure_grid_correction",
                autospec=True,
                side_effect=measure,
            ),
            mock.patch.object(
                SemiEmpiricMatcher,
                "_populate_beam_from_hamiltonian",
                autospec=True,
                side_effect=populate_and_record,
            ),
            mock.patch.object(
                SemiEmpiricMatcher,
                "_rematch_beam_to_potential_well",
                autospec=True,
                side_effect=rematch,
            ) as rematch_spy,
        ):
            TestSemiEmpiricMatcherWarmup._run_matching(
                self,
                # With a patience of one, the stall is detected one
                # iteration after the best one, so best and last differ.
                maxiter_intensity_effects=20,
                tolerance_potential_well=1e-15,
                stagnation_patience=1,
                correct_to_one_turn_map=True,
            )

        rematch_spy.assert_called_once()
        best_potential_well = copy_to_cpu(
            rematch_spy.call_args.kwargs["avg_potential_well"]
        )
        # the correction is measured on the oversampled potential well
        (best_correction,) = (
            grid_correction
            for potential_well, grid_correction in measurements
            if np.allclose(
                potential_well[::_POTENTIAL_WELL_OVERSAMPLING],
                best_potential_well,
                rtol=1e-12,
            )
        )
        self.assertNotEqual(best_correction, measurements[-1][1])
        # the rollback populates the beam last
        self.assertEqual(applied_corrections[-1], best_correction)


class TestCurvatureAtMinimum(BLonDTestCase):
    rf_period = 2.5e-9  # [s]
    n_grid = 128 - 1  # coarse grid of the matcher

    def _cosine_well(self):
        ts = np.linspace(
            0,
            self.rf_period,
            (self.n_grid - 1) * _POTENTIAL_WELL_OVERSAMPLING + 1,
        )
        omega_rf = 2 * np.pi / self.rf_period
        return ts, np.cos(omega_rf * ts), omega_rf

    def test_quartic_term_does_not_bias_curvature(self):
        ts, potential_well, omega_rf = self._cosine_well()
        coarse_step = self.rf_period / (self.n_grid - 1)

        curvature = _curvature_at_minimum(
            ts=ts, potential_well=potential_well, half_width=3 * coarse_step
        )

        # V'' = omega_rf**2 at the minimum, omega_rf * t = pi. A parabola
        # over the same width is off by 1.6e-3.
        np.testing.assert_allclose(curvature, omega_rf**2, rtol=1e-5)

    def test_minimum_at_edge_of_window(self):
        ts, potential_well, omega_rf = self._cosine_well()
        coarse_step = self.rf_period / (self.n_grid - 1)
        before_minimum = ts <= 0.5 * self.rf_period

        curvature = _curvature_at_minimum(
            ts=ts[before_minimum],
            potential_well=potential_well[before_minimum],
            half_width=3 * coarse_step,
        )

        # one-sided fit, still the curvature of the well
        np.testing.assert_allclose(curvature, omega_rf**2, rtol=1e-3)


class TestCallables(BLonDTestCase):
    def test_get_hamiltonian_semi_analytic_energy_range(self):
        ts = np.linspace(0, 1, 100)
        energy_range = (-2.0, 3.0)  # not the symmetric automatic range

        deltaE_grid, _, _ = get_hamilton_semi_analytic(
            ts=ts,
            potential_well=0.5 * ts**2,
            reference_total_energy=1.0,
            eta=0.1,
            beta=1,
            shape=(100, 50),
            energy_range=energy_range,
        )

        for deltaE_row in deltaE_grid:
            np.testing.assert_allclose(
                deltaE_row, np.linspace(*energy_range, 50), rtol=1e-15
            )

    def test_get_hamiltonian_semi_analytic(self):
        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        # Define simple test inputs
        ts = np.linspace(0, 1, 100)  # Time or spatial grid
        eta = 0.1  # Some parameter (perhaps perturbation strength or scaling factor)
        shape = (100, 100)  # Shape of Hamiltonian matrix

        # Define a simple potential well — e.g., harmonic potential
        def gen_potential_well(x):
            return 0.5 * x**2

        # Use the same ts as x-values
        potential_values = gen_potential_well(ts)

        # Reference energy for testing (hypothetical or computed elsewhere)
        reference_total_energy = 1.0  # Placeholder

        # Call the function
        deltaE_grid, time_grid, hamilton_2D = get_hamilton_semi_analytic(
            ts=ts,
            potential_well=potential_values,
            reference_total_energy=reference_total_energy,
            eta=eta,
            beta=1,
            shape=shape,
        )

        hamilton_2D_expected = (
            0.5 * eta / reference_total_energy * deltaE_grid.T * deltaE_grid.T
            + potential_values
        ).T  # [eV]
        DEV_PLOT = False
        if DEV_PLOT:
            plt.figure()
            plt.imshow(hamilton_2D_expected)
            plt.title("hamilton_2D_expected")
            plt.figure()
            plt.imshow(hamilton_2D)
            plt.title("hamilton_2D")
            plt.show()
        np.testing.assert_allclose(
            hamilton_2D_expected,
            hamilton_2D,
            rtol=1e-12,
        )


class TestSemiEmpiricMatcherDocstringExample(BLonDTestCase):
    def test_docstring_example_runs(self):
        """The ``Examples`` section of the class docstring stays runnable."""
        finder = doctest.DocTestFinder(recurse=False)
        (docstring_test,) = finder.find(SemiEmpiricMatcher, globs={})
        runner = doctest.DocTestRunner()
        runner.run(docstring_test)
        results = runner.summarize(verbose=False)
        self.assertGreater(results.attempted, 0)
        self.assertEqual(results.failed, 0)


if __name__ == "__main__":
    unittest.main()
