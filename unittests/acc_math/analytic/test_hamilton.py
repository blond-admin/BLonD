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
    calc_phi_s_single_harmonic,
    is_in_separatrix,
    phase_modulo_above_transition,
    phase_modulo_below_transition,
    separatrix_single_rf,
    single_rf_sin_hamiltonian,
)
from blond.experimental.beam_preparation.filamentation_matcher import (
    FilamentationMatcher,
)
from blond.handle_results.observables_as_elements import (
    BeamObservationInRingElement,
)


class TestPhiS(unittest.TestCase):
    def test_phi_s_1(self):
        xs = np.linspace(-1, 1, 200)
        voltage = 1.5
        omega = 5
        phi = 1.5
        energy_gain = 4
        DEV_PLOT = False
        for charge in (-5, 5):
            for above_transition in (1, 0):
                phi_s = calc_phi_s_single_harmonic(
                    charge,
                    voltage,
                    energy_gain,
                    above_transition=above_transition,
                )
                t_s = (phi_s - phi) / (omega)
                if DEV_PLOT:
                    ys = charge * voltage * np.sin(omega * xs + phi)
                    plt.plot(xs, ys)
                    plt.axhline(energy_gain)
                    plt.axvline(t_s)
                    plt.show()
                self.assertAlmostEqual(
                    charge * voltage * np.sin(omega * t_s + phi), energy_gain
                )


class TestFunctions(unittest.TestCase):
    def test_phase_modulo_above_transition(self):
        upper_limit = 2 * np.pi
        lower_limit = 0
        phis = np.linspace(-100, 100, 200)
        phis_corrected = phase_modulo_above_transition(phis)
        DEV_PLOT = False
        if DEV_PLOT:
            plt.title("phase_modulo_above_transition")
            plt.plot(phis, "o")
            plt.plot(phis_corrected, "o")
            plt.axhline(upper_limit)
            plt.axhline(lower_limit)
            plt.show()
        self.assertTrue(
            np.all(phis_corrected < upper_limit),
            msg=f"{phis_corrected.max()=}",
        )
        self.assertTrue(
            np.all(phis_corrected >= lower_limit),
            msg=f"{phis_corrected.min()=}",
        )

    def test_phase_modulo_below_transition(self):
        upper_limit = np.pi
        lower_limit = -np.pi
        phis = np.linspace(-100, 100, 200)
        phis_corrected = phase_modulo_below_transition(phis)
        DEV_PLOT = False
        if DEV_PLOT:
            plt.title("phase_modulo_below_transition")
            plt.plot(phis, "o")
            plt.plot(phis_corrected, "o")
            plt.axhline(upper_limit)
            plt.axhline(lower_limit)
            plt.show()
        self.assertTrue(
            np.all(phis_corrected <= upper_limit),
            msg=f"{phis_corrected.max()=}",
        )
        self.assertTrue(
            np.all(phis_corrected >= lower_limit),
            msg=f"{phis_corrected.min()=}",
        )


class TestSingleRFSinHamiltonian(unittest.TestCase):
    def setUp(self):
        self.charge = 1.0  # elementary charge units
        self.harmonic = 10
        self.voltage = 1e6  # V
        self.omega_rf = 2 * np.pi * 1e6  # rad/s
        self.phi_rf_d = 0.0  # rad
        self.phi_s = np.pi / 6  # stable phase, rad
        self.etas = [-0.01]  # below transition
        self.beta = 0.9
        self.total_energy = 1e9  # eV
        self.ring_circumference = 100.0  # m

    def test_hamiltonian_at_separatrix_max(self):
        # Max point of separatrix in phase: phi_b = π - phi_s
        dt_sep_max = (np.pi - self.phi_s - self.phi_rf_d) / self.omega_rf
        dE_sep_max = 0.0  # maximum in phase, energy = 0
        for sign in (-1, 1):
            H = single_rf_sin_hamiltonian(
                charge=self.charge,
                harmonic=self.harmonic,
                voltage=self.voltage,
                omega_rf=self.omega_rf,
                phi_rf_d=self.phi_rf_d,
                phi_s=self.phi_s,
                etas=sign * np.array(self.etas),
                beta=self.beta,
                total_energy=self.total_energy,
                ring_circumference=self.ring_circumference,
                dt=dt_sep_max,
                dE=dE_sep_max,
            )
            H_pinned = (
                -184782456987.43494
            )  # guarantee that result doesnt change
            # physics might be still wrong. In that case, H_pinned might need to
            # b changed.
            self.assertEqual(H, H_pinned)

    @unittest.skip("TODO")  # TODO
    def test_is_in_separatrix(self):
        self.fail()  # TODO
        is_in_separatrix(
            charge=self.charge,
            harmonic=self.harmonic,
            voltage=self.voltage,
            omega_rf=self.omega_rf,
            phi_rf_d=self.phi_rf_d,
            phi_s=self.phi_s,
            etas=self.etas,
            beta=self.beta,
            total_energy=self.total_energy,
            ring_circumference=self.ring_circumference,
            dt=dt_sep_max,
            dE=dE_sep_max,
        )


class TestIsInSeparatrix(unittest.TestCase):
    @unittest.skip("TODO")  # TODO
    def test1(self):
        is_in_separatrix
        self.fail()  # TODO


def test_single_harmonic_separatrix():
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
    phi, separatrix_calc = separatrix_single_rf(
        one_turn_execution_order[1], energy_cycle, ring, dt, 0
    )

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


def test_single_harmonic_separatrix_magentic_cycle_per_turn():
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
    phi, separatrix_calc = separatrix_single_rf(
        one_turn_execution_order[1], energy_cycle, ring, dt, 0
    )

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
