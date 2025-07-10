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


class TestConstantEnergyCycle(unittest.TestCase):
    def setUp(self):
        self.constant_energy_cycle = ConstantEnergyCycle(
            value=11,
            in_unit="momentum",
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        self.constant_energy_cycle.on_init_simulation(simulation=simulation_ex1)
        self.assertEqual(self.constant_energy_cycle._momentum, 11)

    def test_headless(self):
        cec = ConstantEnergyCycle.headless(
            value=1,
            in_unit="momentum",
            bending_radius=None,
            mass=1,
            charge=2,
        )
        self.assertEqual(
            cec.get_target_total_energy(0, 0, 0),
            cec.get_target_total_energy(111, 111, 111),
        )


class EnergyCycleBaseHelper(EnergyCycleBase):
    def get_target_total_energy(
        self, turn_i: int, section_i: int, reference_time: float
    ):
        pass

    @staticmethod
    def headless(*args, **kwargs):
        pass


class TestEnergyCycleBase(unittest.TestCase):
    def setUp(self):
        self.energy_cycle_base = EnergyCycleBaseHelper()
        self.energy_cycle_base._momentum_init = 1e9
        self.energy_cycle_base._momentum = np.array(
            [[1e9, 2e9, 3e9], [4e9, 5e9, 6e9]]
        ).T
        # eV/c]
        self.energy_cycle_base._section_lengths = np.array([100, 200, 300])  # [m]
        self.energy_cycle_base._mass = proton.mass

    def test___init__(self):
        self.assertIsInstance(self.energy_cycle_base, EnergyCycleBase)

    def test_energy(self):
        energy = self.energy_cycle_base.total_energy_init
        expected = np.sqrt(
            self.energy_cycle_base._momentum_init**2 + self.energy_cycle_base._mass**2
        )
        assert_allclose(energy, expected, rtol=1e-8)

    def test_invalidate_cache(self):
        # This is a placeholder; add actual cache-clearing verification if applicable
        try:
            self.energy_cycle_base.invalidate_cache()
        except Exception as e:
            self.fail(f"_invalidate_cache() raised an exception: {e}")

    def test_on_init_simulation(self):
        self.energy_cycle_base.on_init_simulation(
            simulation=simulation_ex1,
            momentum_init=11,
            n_turns=10,
        )

    def test_on_run_simulation(self):
        self.energy_cycle_base.on_run_simulation(
            simulation=simulation_ex1, n_turns=1, turn_i_init=10
        )


class TestEnergyCycleByTime(unittest.TestCase):
    def setUp(self):
        self.energy_cycle_by_time = EnergyCycleByTime(
            t0=1.0,
            base_time=np.linspace(1, 12, 12),
            base_values=np.linspace(1e9, 5e9, 12),
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        self.energy_cycle_by_time.on_init_simulation(simulation=simulation_ex1)

    def test_headless(self):
        ebt = EnergyCycleByTime.headless(
            t0=131,
            base_time=np.linspace(0, 10, endpoint=True),
            base_values=np.linspace(0, 10, endpoint=True),
            mass=1,
            charge=2,
            in_unit="total energy",
        )
        self.assertEqual(ebt.total_energy_init, 10)


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

    def test_headless(self):
        evpt = EnergyCyclePerTurn.headless(
            mass=1,
            charge=1,
            value_init=0,
            values_after_turn=np.ones(10),
            section_lengths=np.array([0.5, 1]),
        )
        self.assertEqual(evpt._momentum.shape, (2, 10))


class TestEnergyCyclePerTurnAllCavities(unittest.TestCase):
    def setUp(self):
        self.momentum = np.ones((1, 10))
        self.energy_cycle_per_turn_all_cavities = EnergyCyclePerTurnAllCavities(
            values_after_cavity_per_turn=self.momentum,
            value_init=1,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_wrong_cavity_count(self):
        # simulation has only one cavity, but give program for 10 cavities
        with self.assertRaises(AssertionError):
            self.energy_cycle_per_turn_all_cavities = EnergyCyclePerTurnAllCavities(
                values_after_cavity_per_turn=np.ones((10, 10)),
                value_init=10,
            )
            self.energy_cycle_per_turn_all_cavities.on_init_simulation(
                simulation=simulation_ex1
            )

    def test_on_init_simulation(self):
        self.energy_cycle_per_turn_all_cavities.on_init_simulation(
            simulation=simulation_ex1,
            momentum_init=11,
            momentum=np.ones(
                (1, 10),
            ),
        )
        assert_allclose(
            self.energy_cycle_per_turn_all_cavities._momentum_after_cavity_per_turn,
            self.momentum,
        )

    def test_headless(self):
        ecptac = EnergyCyclePerTurnAllCavities.headless(
            value_init=10,
            values_after_cavity_per_turn=np.ones((2, 20)),
            mass=1,
            charge=1,
            section_lengths=np.array([0.1, 0.3]),
        )
        self.assertEqual(ecptac._momentum_after_cavity_per_turn.shape, (2, 20))


if __name__ == "__main__":
    unittest.main()
