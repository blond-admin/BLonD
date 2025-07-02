import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import speed_of_light as c0

from blond3 import (
    EnergyCycleByTime,
    EnergyCyclePerTurn,
    EnergyCyclePerTurnAllCavities,
    ConstantEnergyCycle,
    proton,
)
from blond3.cycles.energy_cycle import (
    _to_momentum,
    beta_by_momentum,
    EnergyCycleBase,
    derive_time,
    calc_beta,
    calc_gamma,
    calc_total_energy,
    calc_energy_kin,
)
from blond3.testing.simulation import ExampleSimulation01, SimulationTwoRfStations

simulation_ex1 = ExampleSimulation01().simulation


class TestRelativisticFunctions(unittest.TestCase):
    def setUp(self):
        self.mass = 938e6  # Proton mass in eV/c²
        self.p = 1e9  # 1 GeV/c

    def test_calc_beta(self):
        beta = calc_beta(self.mass, self.p)
        expected = self.p / np.sqrt(self.p**2 + self.mass**2)
        self.assertAlmostEqual(beta, expected, places=8)

    def test_calc_gamma(self):
        gamma = calc_gamma(self.mass, self.p)
        expected = 1 / np.sqrt(1 - (self.p**2 / (self.p**2 + self.mass**2)))
        self.assertAlmostEqual(gamma, expected, places=8)

    def test_calc_energy(self):
        energy = calc_total_energy(self.mass, self.p)
        expected = np.sqrt(self.p**2 + self.mass**2)
        self.assertAlmostEqual(energy, expected, places=8)

    def test_calc_energy_kin(self):
        kin_energy = calc_energy_kin(self.mass, self.p)
        expected = np.sqrt(self.p**2 + self.mass**2) - self.mass
        self.assertAlmostEqual(kin_energy, expected, places=8)


class TestFunctions(unittest.TestCase):
    def test__to_momentum_momentum(self):
        data = np.array([1e6, 2e6, 3e6])  # [eV/c]
        result = _to_momentum(
            data=data,
            mass=proton.mass,
            charge=proton.charge,
            convert_from="momentum",
        )
        np.testing.assert_allclose(
            result, data, rtol=1e-8, err_msg="Momentum input should return unchanged"
        )

    def test__to_momentum_total_energy(self):
        energy = np.array([1e9, 2e9, 3e9])  # [eV]
        mass = proton.mass  # ~938 MeV/c²
        expected_momentum = np.sqrt(energy**2 - mass**2)
        result = _to_momentum(
            data=energy,
            mass=mass,
            charge=proton.charge,
            convert_from="total energy",
        )
        np.testing.assert_allclose(result, expected_momentum, rtol=1e-6)

    def test__to_momentum_kinetic_energy(self):
        kinetic_energy = np.array([1e6, 5e6, 1e7])  # [eV]
        mass = proton.mass
        total_energy = kinetic_energy + mass
        expected_momentum = np.sqrt(total_energy**2 - mass**2)
        result = _to_momentum(
            data=kinetic_energy,
            mass=mass,
            charge=proton.charge,
            convert_from="kinetic energy",
        )
        np.testing.assert_allclose(result, expected_momentum, rtol=1e-6)

    def test__to_momentum_field(self):
        magnetic_field = np.array([1.0, 2.0, 3.0])  # [T]
        radius = 27e3 / (2 * np.pi)  # bending radius [m]
        charge = proton.charge
        expected_momentum = (
            magnetic_field * radius * charge * 299792458
        )  # eBr, result in eV/c

        result = _to_momentum(
            data=magnetic_field,
            mass=proton.mass,
            charge=charge,
            convert_from="bending field",
            bending_radius=radius,
        )
        np.testing.assert_allclose(result, expected_momentum, rtol=1e-6)

    def test_beta_by_momentum(self):
        # Example momenta in eV/c
        momenta = np.array([1e6, 1e9, 5e9])  # eV/c
        mass = proton.mass  # ~938e6 eV/c² for a proton

        # Expected beta: beta = p / sqrt(p^2 + m^2)
        expected_beta = momenta / np.sqrt(momenta**2 + mass**2)

        result = beta_by_momentum(momentum=momenta, mass=mass)

        np.testing.assert_allclose(result, expected_beta, rtol=1e-8)

    @unittest.skip  # TODO
    def test_derive_time_glue(self):
        pass

    def test_derive_time(self):
        c = c0  # total circumference in meters
        # using 299 792 458 m, so one turn must take more than 1s

        # Structure: drift [m], rf 0, drift [fraction], rf 1, drift [fraction], rf 2
        fast_execution_order = [1 * c, int(0), 1 * c, int(1), 2 * c, int(2)]
        # expecting flight time about 1s, 1s, 2s...

        mass = proton.mass  # ~938e6 eV/c²
        n_turns = 2
        n_sections = 3  # same as number of RF stations

        # momentum and time program as a function of time
        program_time = np.array([0.0, 1e-6, 2e-6, 5e9])
        program_momentum = np.array([1e9, 2e9, 3e9, 4e9])

        times = derive_time(
            fast_execution_order=fast_execution_order,
            interpolate=np.interp,
            mass=mass,
            n_sections=n_sections,
            n_turns=n_turns,
            program_momentum=program_momentum,
            program_time=program_time,
            t0=0.0,
        )

        # Allow small numerical tolerance
        np.testing.assert_allclose(
            times[:, 0],
            np.array([1.371260191867345, 2.419027899042942, 4.514563313394136]),
            rtol=1e-8,
        )

    def test_derive_time_const_momentum(self):
        c = c0  # total circumference in meters
        # using 299 792 458 m, so one turn must take more than 1s

        # Structure: drift [m], rf 0, drift [fraction], rf 1, drift [fraction], rf 2
        fast_execution_order = [1 * c, int(0), 1 * c, int(1), 1 * c, int(2)]
        # expecting flight time about 1s, 1s, 2s...

        mass = proton.mass  # ~938e6 eV/c²
        n_turns = 2
        n_sections = 3  # same as number of RF stations

        # momentum and time program as a function of time
        program_time = np.array([0.0, 1e-6, 2e-6, 5e9])
        program_momentum = np.array([1e9, 1e9, 1e9, 1e9])  # CONSTANT

        times = derive_time(
            fast_execution_order=fast_execution_order,
            interpolate=np.interp,
            mass=mass,
            n_sections=n_sections,
            n_turns=n_turns,
            program_momentum=program_momentum,
            program_time=program_time,
            t0=0.0,
        )

        # Allow small numerical tolerance
        np.testing.assert_allclose(
            np.diff(times[:, 0]),
            np.array([1.371260191867345, 1.371260191867345]),
            rtol=1e-8,
        )


class TestConstantEnergyCycle(unittest.TestCase):
    def setUp(self):
        self.constant_energy_cycle = ConstantEnergyCycle(
            value=11, max_turns=11, in_unit="momentum",
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        self.constant_energy_cycle.on_init_simulation(simulation=simulation_ex1)
        self.assertEqual(self.constant_energy_cycle.momentum.min(), 11)
        self.assertEqual(self.constant_energy_cycle.momentum.max(), 11)


class TestEnergyCycleBase(unittest.TestCase):
    def setUp(self):
        self.energy_cycle_base = EnergyCycleBase()

        self.energy_cycle_base._momentum = np.array(
            [[1e9, 2e9, 3e9], [4e9, 5e9, 6e9]]
        ).T
        # eV/c]
        self.energy_cycle_base._section_lengths = np.array([100, 200, 300])  # [m]
        self.energy_cycle_base._particle = proton

    def test___init__(self):
        self.assertIsInstance(self.energy_cycle_base, EnergyCycleBase)

    def test_beta(self):
        beta = self.energy_cycle_base.beta
        expected = self.energy_cycle_base._momentum / np.sqrt(
            self.energy_cycle_base._momentum**2
            + self.energy_cycle_base._particle.mass**2
        )
        assert_allclose(beta, expected, rtol=1e-8)

    def test_gamma(self):
        gamma = self.energy_cycle_base.gamma
        expected = 1 / np.sqrt(1 - self.energy_cycle_base.beta**2)
        assert_allclose(gamma, expected, rtol=1e-8)

    def test_energy(self):
        energy = self.energy_cycle_base.total_energy
        expected = np.sqrt(
            self.energy_cycle_base._momentum**2
            + self.energy_cycle_base._particle.mass**2
        )
        assert_allclose(energy, expected, rtol=1e-8)

    def test_kin_energy(self):
        kin_energy = self.energy_cycle_base.kin_energy
        expected = (
            self.energy_cycle_base.total_energy - self.energy_cycle_base._particle.mass
        )
        assert_allclose(kin_energy, expected, rtol=1e-8)

    def test_delta_E(self):
        delta_e = self.energy_cycle_base.delta_E
        energy = self.energy_cycle_base.kin_energy
        expected = np.diff(energy.flatten("K"))
        assert_allclose(delta_e.flatten("K")[1:], expected, rtol=1e-8)

    def test_t_section(self):
        beta = self.energy_cycle_base.beta[:, 0]
        lengths = self.energy_cycle_base._section_lengths
        expected_time = lengths / (beta * c0)
        actual_time = self.energy_cycle_base.t_section
        assert actual_time.shape == (3, 2), f"{actual_time.shape=}"
        assert_allclose(actual_time[:, 0], expected_time, rtol=1e-8)

    def test_t_rev(self):
        for turn_i in range(2):
            beta = self.energy_cycle_base.beta[:, turn_i]
            lengths = self.energy_cycle_base._section_lengths
            c = 299_792_458  # m/s
            expected_time = np.sum(lengths / (beta * c))
            actual_time = self.energy_cycle_base.t_rev[turn_i]
            self.assertEqual(actual_time, expected_time)

    def test_cycle_time(self):
        beta = self.energy_cycle_base.beta[:, 0]
        lengths = self.energy_cycle_base._section_lengths
        c = 299_792_458  # m/s
        expected_time = np.cumsum(lengths / (beta * c))
        actual_time = self.energy_cycle_base.cycle_time[:, 0]
        assert_allclose(actual_time, expected_time, rtol=1e-8)

    # @unittest.skip
    # def test_omega_rev(self):
    #    # TODO: implement test for `omega_rev`
    #    self.energy_cycle_base.omega_rev()

    def test_f_rev(self):
        expected_freq = 1 / self.energy_cycle_base.t_rev
        actual_freq = self.energy_cycle_base.f_rev
        assert_allclose(actual_freq, expected_freq, rtol=1e-8)

    def test_invalidate_cache(self):
        # This is a placeholder; add actual cache-clearing verification if applicable
        try:
            self.energy_cycle_base._invalidate_cache()
        except Exception as e:
            self.fail(f"_invalidate_cache() raised an exception: {e}")

    def test_on_init_simulation(self):
        self.energy_cycle_base.on_init_simulation(simulation=simulation_ex1)

    def test_on_run_simulation(self):
        self.energy_cycle_base.on_run_simulation(
            simulation=simulation_ex1, n_turns=1, turn_i_init=10
        )


class TestEnergyCycleByTime(unittest.TestCase):
    def setUp(self):
        self.energy_cycle_by_time = EnergyCycleByTime(
            t0=1.0,
            max_turns=10,
            base_time=np.linspace(1, 12, 12),
            base_values=np.linspace(1e9, 5e9, 12),
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        self.energy_cycle_by_time.on_init_simulation(simulation=simulation_ex1)


class TestEnergyCyclePerTurn(unittest.TestCase):
    def setUp(self):
        self.momentum = np.linspace(1, 10, 11)
        self.energy_cycle_per_turn = EnergyCyclePerTurn(
            value_init=float(self.momentum[0]), values_after_turn=self.momentum[1:]
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        self.energy_cycle_per_turn.on_init_simulation(simulation=simulation_ex1)
        turn_i = 0
        assert_allclose(
            self.energy_cycle_per_turn._momentum[turn_i, :], self.momentum[1:]
        )

    def test_two_rf(self):
        simulation = SimulationTwoRfStations().simulation
        self.energy_cycle_per_turn.on_init_simulation(simulation=simulation)
        cavity_i = 1
        assert_allclose(
            self.energy_cycle_per_turn._momentum[cavity_i, :], self.momentum[1:]
        )


class TestEnergyCyclePerTurnAllCavities(unittest.TestCase):
    def setUp(self):
        self.momentum = np.ones((1, 10))
        self.energy_cycle_per_turn_all_cavities = EnergyCyclePerTurnAllCavities(
            values_after_cavity_per_turn=self.momentum
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_wrong_cavity_count(self):
        # simulation has only one cavity, but give program for 10 cavities
        with self.assertRaises(AssertionError):
            self.energy_cycle_per_turn_all_cavities = EnergyCyclePerTurnAllCavities(
                values_after_cavity_per_turn=np.ones((10, 10))
            )
            self.energy_cycle_per_turn_all_cavities.on_init_simulation(
                simulation=simulation_ex1
            )

    def test_on_init_simulation(self):
        self.energy_cycle_per_turn_all_cavities.on_init_simulation(
            simulation=simulation_ex1
        )
        assert_allclose(self.energy_cycle_per_turn_all_cavities.momentum, self.momentum)


if __name__ == "__main__":
    unittest.main()
