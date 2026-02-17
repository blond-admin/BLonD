import unittest

import numpy as np
from blond.handle_results.observables import (
    DynamicProfileConstNBinsObservation,
    RFStationPhaseObservation,
    StaticMultiProfileObservation,
    StaticProfileObservation,
    WakeFieldObservation,
)
from blond.physics.impedances.solvers import TimeDomainFftSolver
from blond.physics.impedances.sources import Resonators
from blond.physics.profiles import DynamicProfileConstNBins

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    uranium_29,
)


class TestObservables(unittest.TestCase):
    def test_run_all(self):
        ring = Ring(circumference=123.4)
        beam = Beam(intensity=12, particle_type=uranium_29)
        beam.setup_beam(
            dt=np.linspace(-2, 2, 512),
            dE=np.linspace(-2, 2, 512),
            reference_total_energy=uranium_29.mass + 1.4e6 * 238,
        )
        cycle = ConstantMagneticCycle(
            reference_particle=beam.particle_type,
            value=beam.reference.total_energy,
            in_unit="total energy",
        )

        static_profile_01 = StaticProfile(cut_left=-1, cut_right=1, n_bins=128)
        static_profile_02 = StaticProfile(cut_left=-1, cut_right=2, n_bins=128)
        dynamic_profile = DynamicProfileConstNBins(n_bins=128)
        wakefield = WakeField(
            sources=(Resonators(1, 2, 3),),
            solver=TimeDomainFftSolver(),
            profile=static_profile_01,
        )
        drift = DriftSimple(
            orbit_length=ring.circumference, transition_gamma=1.2
        )
        rf_station = SingleHarmonicRFStation(voltage=1e6, phi_rf=0, harmonic=1)

        each_turn_i = 1
        # define all available observations
        cavity_phase_observation = RFStationPhaseObservation(
            each_turn_i=each_turn_i, rf_station=rf_station
        )
        static_orofile_observation = StaticProfileObservation(
            each_turn_i=each_turn_i, profile=static_profile_01
        )
        static_multi_profile_pbservation = StaticMultiProfileObservation(
            each_turn_i=each_turn_i,
            profiles=[static_profile_01, static_profile_02],
        )
        wake_field_observation = WakeFieldObservation(
            each_turn_i=each_turn_i, wakefield=wakefield
        )
        dynamic_profile_const_n_bins_observation = (
            DynamicProfileConstNBinsObservation(
                each_turn_i=each_turn_i, profile=dynamic_profile
            )
        )
        all_observables = (
            cavity_phase_observation,
            static_orofile_observation,
            static_multi_profile_pbservation,
            wake_field_observation,
            dynamic_profile_const_n_bins_observation,
        )
        simulation = Simulation.from_locals(locals())
        simulation.print_one_turn_execution_order()
        n_turns = 12
        n_turns_half = n_turns // each_turn_i
        simulation.run_simulation(
            beams=(beam,), n_turns=n_turns, observe=all_observables
        )

        for obs in all_observables:
            for attribute, rec in obs.get_recorders():
                print(rec)
                array_shape = rec.get_valid_entries().shape
                print(array_shape)
                assert array_shape[0] == n_turns_half, " ".join(
                    [str(obs), str(attribute), str(array_shape)]
                )
