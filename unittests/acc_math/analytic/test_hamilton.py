import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    proton,
)
from blond.acc_math.analytic.hamilton import (
    calc_synchrotron_tune_single_harmonic,
    phase_modulo_above_transition,
    phase_modulo_below_transition,
    separatrix_single_rf_blond,
    separatrix_single_rf_calculation,
)
from blond.experimental.beam_preparation.filamentation_matcher import (
    FilamentationMatcher,
)
from blond.handle_results.observables_as_elements import (
    BeamObservationInRingElement,
)


class TestPhaseModuloBelowTransition(unittest.TestCase):
    def test_scalar_values(self):
        self.assertAlmostEqual(phase_modulo_below_transition(0.5), 0.5)
        self.assertAlmostEqual(
            phase_modulo_below_transition(-np.pi / 2), -np.pi / 2
        )
        self.assertAlmostEqual(
            phase_modulo_below_transition(3 * np.pi), -np.pi
        )
        self.assertAlmostEqual(
            phase_modulo_below_transition(-3 * np.pi / 2), np.pi / 2
        )

    def test_array_values(self):
        phi = np.linspace(-10, 10)
        result = phase_modulo_below_transition(phi)
        DEV_PLOT = False
        if DEV_PLOT:
            plt.plot(phi)
            plt.plot(result)
            plt.show()
        self.assertTrue(np.all(result < np.pi))
        self.assertTrue(np.all(result >= -np.pi))


class TestPhaseModuloAboveTransition(unittest.TestCase):
    def test_scalar_values(self):
        # 0 stays 0
        self.assertAlmostEqual(phase_modulo_above_transition(0.0), 0.0)

        # Positive values below 2π remain unchanged
        self.assertAlmostEqual(
            phase_modulo_above_transition(np.pi / 2), np.pi / 2
        )

        # Values above 2π wrap around
        self.assertAlmostEqual(phase_modulo_above_transition(3 * np.pi), np.pi)

        # Negative values wrap into the positive range
        self.assertAlmostEqual(
            phase_modulo_above_transition(-np.pi / 2), 3 * np.pi / 2
        )

    def test_array_values(self):
        phi = np.linspace(-10, 10)
        result = phase_modulo_above_transition(phi)

        # All results should be within [0, 2π)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result < 2 * np.pi))

    def test_periodicity(self):
        # Check that adding 2π doesn't change the result
        vals = np.linspace(-5, 5, 10)
        self.assertTrue(
            np.allclose(
                phase_modulo_above_transition(vals),
                phase_modulo_above_transition(vals + 2 * np.pi),
            )
        )


class TestSynchrotronTune(unittest.TestCase):
    def test_tune(self):
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


def test_single_harmonic_separatrix_blond():
    p_s = 450.0e9  # Synchronous momentum [eV]
    harmonic_number = 35640  # Harmonic number
    voltage1 = 2e6  # RF voltage, station 1 [eV]
    phi_rf = 0  # Phase modulation/offset
    transition_gamma = 55.759505  # Transition gamma

    energy_cycle = ConstantMagneticCycle(
        value=p_s,
        reference_particle=proton,
    )
    ring = Ring(
        circumference=26658.883,
    )
    beam = Beam(
        intensity=1.0e9,
        particle_type=proton,
    )

    observation = BeamObservationInRingElement(
        each_turn_i=1,
        section_index=0,
        n_turns=10,
        folder="./",
    )

    one_turn_execution_order = (
        DriftSimple(
            transition_gamma=transition_gamma,
            orbit_length=ring.circumference,
            section_index=0,
        ),
        SingleHarmonicRFStation(
            harmonic=harmonic_number,
            phi_rf=phi_rf,
            voltage=voltage1,
            section_index=0,
        ),
        observation,
    )

    ring.add_elements(one_turn_execution_order, reorder=False)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

    dt = np.arange(0, 5e-9, 1e-11)

    sim.prepare_beam(
        preparation_routine=FilamentationMatcher(
            time_limit=[0.1e-9, 4e-9],
            energy_limit=[-4e8, 4e8],
            n_macroparticles=3000,
            n_iter=2000,
            every_iter_to_plot=10,
            animate=False,
            purge_limit_time=[0.1e-9, 4e-9],
            purge_limit_energy=[-5e8, 5e8],
            purge=True,
        ),
        beam=beam,
    )

    phi, separatrix_calc = separatrix_single_rf_blond(
        one_turn_execution_order[1], beam, energy_cycle, ring, dt, 0
    )

    test_inside_separatrix = 0
    dt_array = beam.read_partial_dt()
    dE_array = beam.read_partial_dE()

    for particle in range(len(dt_array)):
        dt_value = dt_array[particle]

        sep_dt = np.argmin(np.absolute(phi - dt_value))

        if (
            np.absolute(separatrix_calc[sep_dt])
            - np.absolute(dE_array[particle])
        ) > 0:
            test_inside_separatrix += 1

    np.testing.assert_almost_equal(test_inside_separatrix, len(dt_array))


def test_single_harmonic_separatrix_magentic_cycle_per_turn_blond():
    p_s = 450.0e9  # Synchronous momentum [eV]
    harmonic_number = 35640  # Harmonic number
    voltage1 = 2e6  # RF voltage, station 1 [eV]
    phi_rf = 0  # Phase modulation/offset
    transition_gamma = 55.759505  # Transition gamma

    N_TURNS = int(1e3)

    energy_cycle = MagneticCyclePerTurn(
        value_init=p_s,
        values_after_turn=np.linspace(p_s, p_s, N_TURNS),
        reference_particle=proton,
    )
    ring = Ring(
        circumference=26658.883,
    )
    beam = Beam(
        intensity=1.0e9,
        particle_type=proton,
    )

    observation = BeamObservationInRingElement(
        each_turn_i=1,
        section_index=0,
        n_turns=10,
        folder="./",
    )

    one_turn_execution_order = (
        DriftSimple(
            transition_gamma=transition_gamma,
            orbit_length=ring.circumference,
            section_index=0,
        ),
        SingleHarmonicRFStation(
            harmonic=harmonic_number,
            phi_rf=phi_rf,
            voltage=voltage1,
            section_index=0,
        ),
        observation,
    )

    ring.add_elements(one_turn_execution_order, reorder=False)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

    dt = np.arange(0, 5e-9, 1e-11)

    sim.prepare_beam(
        preparation_routine=FilamentationMatcher(
            time_limit=[0.1e-9, 4e-9],
            energy_limit=[-4e8, 4e8],
            n_macroparticles=3000,
            n_iter=2000,
            every_iter_to_plot=10,
            animate=False,
            purge_limit_time=[0.1e-9, 4e-9],
            purge_limit_energy=[-5e8, 5e8],
            purge=True,
        ),
        beam=beam,
    )

    phi, separatrix_calc = separatrix_single_rf_blond(
        one_turn_execution_order[1], beam, energy_cycle, ring, dt, 0
    )

    test_inside_separatrix = 0
    dt_array = beam.read_partial_dt()
    dE_array = beam.read_partial_dE()

    for particle in range(len(dt_array)):
        dt_value = dt_array[particle]

        sep_dt = np.argmin(np.absolute(phi - dt_value))

        if (
            np.absolute(separatrix_calc[sep_dt])
            - np.absolute(dE_array[particle])
        ) > 0:
            test_inside_separatrix += 1

    np.testing.assert_almost_equal(test_inside_separatrix, len(dt_array))


def test_single_harmonic_separatrix_calculation():
    p_s = 450.0e9  # Synchronous momentum [eV]
    harmonic_number = 35640  # Harmonic number
    voltage1 = 2e6  # RF voltage, station 1 [eV]
    phi_rf = 0  # Phase modulation/offset
    transition_gamma = 55.759505  # Transition gamma

    energy_cycle = ConstantMagneticCycle(
        value=p_s,
        reference_particle=proton,
    )
    ring = Ring(
        circumference=26658.883,
    )
    beam = Beam(
        intensity=1.0e9,
        particle_type=proton,
    )

    observation = BeamObservationInRingElement(
        each_turn_i=1,
        section_index=0,
        n_turns=10,
        folder="./",
    )

    one_turn_execution_order = (
        DriftSimple(
            transition_gamma=transition_gamma,
            orbit_length=ring.circumference,
            section_index=0,
        ),
        SingleHarmonicRFStation(
            harmonic=harmonic_number,
            phi_rf=phi_rf,
            voltage=voltage1,
            section_index=0,
        ),
        observation,
    )

    ring.add_elements(one_turn_execution_order, reorder=False)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

    sim.prepare_beam(
        preparation_routine=FilamentationMatcher(
            time_limit=[0.1e-9, 4e-9],
            energy_limit=[-4e8, 4e8],
            n_macroparticles=3000,
            n_iter=2000,
            every_iter_to_plot=10,
            animate=False,
            purge_limit_time=[0.1e-9, 4e-9],
            purge_limit_energy=[-5e8, 5e8],
            purge=True,
        ),
        beam=beam,
    )

    reference_gamma = beam.reference.gamma

    eta = 1 / (transition_gamma * transition_gamma) - 1 / (
        reference_gamma * reference_gamma
    )

    dt = np.arange(0, 5e-9, 1e-11)

    reference_velocity = beam.reference.velocity
    circumference = ring.circumference
    t_rev = circumference / reference_velocity

    omega_rf = (2 * np.pi) * harmonic_number / t_rev

    phi, separatrix_calc = separatrix_single_rf_calculation(
        voltage1,
        harmonic_number,
        proton,
        energy_gain=0,
        omega_rf=omega_rf,
        eta=eta,
        energy=p_s,
        dt_array=dt,
    )

    test_inside_separatrix = 0
    dt_array = beam.read_partial_dt()
    dE_array = beam.read_partial_dE()

    for particle in range(len(dt_array)):
        dt_value = dt_array[particle]

        sep_dt = np.argmin(np.absolute(phi - dt_value))

        if (
            np.absolute(separatrix_calc[sep_dt])
            - np.absolute(dE_array[particle])
        ) > 0:
            test_inside_separatrix += 1

    np.testing.assert_almost_equal(test_inside_separatrix, len(dt_array))


if __name__ == "__main__":
    unittest.main()
