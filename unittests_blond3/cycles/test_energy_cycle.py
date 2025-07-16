from __future__ import annotations

import unittest
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import speed_of_light as c0
from blond3 import (
    MagneticCycleByTime,
    MagneticCyclePerTurn,
    MagneticCyclePerTurnAllCavities,
    ConstantMagneticCycle,
    proton,
)
from blond3._core.beam.base import BeamBaseClass
from blond3._core.beam.particle_types import ParticleType, uranium_29
from blond3.acc_math.analytic.simple_math import (
    calc_beta,
    calc_gamma,
    calc_total_energy,
    calc_energy_kin,
    beta_by_momentum,
)
from blond3.cycles.magnetic_cycle import (
    MagneticCycleBase,
    _to_magnetic_rigidity,
    magnetic_rigidity_to_momentum,
)
from blond3.testing.simulation import ExampleSimulation01, SimulationTwoRfStations

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray as NumpyArray
    from blond3.cycles.magnetic_cycle import SynchronousDataTypes
simulation_ex1 = ExampleSimulation01().simulation


def _to_momentum(
    data: int | float | NumpyArray,
    mass: float,
    charge: float,
    convert_from: SynchronousDataTypes = "momentum",
    bending_radius: Optional[float] = None,
) -> NumpyArray:
    magnetic_rigidity = _to_magnetic_rigidity(
        data=data,
        mass=mass,
        charge=charge,
        convert_from=convert_from,
        bending_radius=bending_radius,
    )
    return magnetic_rigidity_to_momentum(
        magnetic_rigidity=magnetic_rigidity, charge=charge
    )


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
        self.constant_magnetic_cycle = ConstantMagneticCycle(
            value=2000e6,
            in_unit="total energy",
            reference_particle=proton,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        self.constant_magnetic_cycle.on_init_simulation(simulation=simulation_ex1)
        self.assertEqual(
            2000e6,
            self.constant_magnetic_cycle.get_total_energy_init(
                turn_i_init=0,
                t_init=0,
                particle_type=proton,
            ),
        )

    def test_headless(self):
        cec = ConstantMagneticCycle.headless(
            value=1,
            in_unit="momentum",
            bending_radius=None,
            particle_type=proton,
        )
        self.assertEqual(
            cec.get_target_total_energy(0, 0, 0, proton),
            cec.get_target_total_energy(111, 111, 111, proton),
        )


class MagneticCycleBaseHelper(MagneticCycleBase):
    def get_target_total_energy(
        self,
        turn_i: int,
        section_i: int,
        reference_time: float,
        particle_type: ParticleType,
    ):
        pass

    @staticmethod
    def headless(*args, **kwargs):
        pass


class TestEnergyCycleBase(unittest.TestCase):
    def setUp(self):
        self.magnetic_cycle_base = MagneticCycleBaseHelper(reference_particle=proton)
        self.momentum_init = 16

        self.magnetic_cycle_base._magnetic_rigidity_before_turn_0 = (
            self.momentum_init / (proton.charge * c0)
        )

    def test___init__(self):
        self.assertIsInstance(self.magnetic_cycle_base, MagneticCycleBase)

    def test_energy(self):
        energy = self.magnetic_cycle_base.get_total_energy_init(0, 0, proton)
        expected = np.sqrt(self.momentum_init**2 + proton.mass**2)
        assert_allclose(energy, expected, rtol=1e-8)

    def test_invalidate_cache(self):
        # This is a placeholder; add actual cache-clearing verification if applicable
        try:
            self.magnetic_cycle_base.invalidate_cache()
        except Exception as e:
            self.fail(f"_invalidate_cache() raised an exception: {e}")

    def test_on_init_simulation(self):
        self.magnetic_cycle_base.on_init_simulation(
            simulation=simulation_ex1,
            magnetic_rigidity_init=11,
            n_turns_max=10,
        )

    def test_on_run_simulation(self):
        self.magnetic_cycle_base.on_run_simulation(
            simulation=simulation_ex1,
            n_turns=1,
            turn_i_init=10,
            beam=Mock(BeamBaseClass),
        )


class TestEnergyCycleByTime(unittest.TestCase):
    def setUp(self):
        self.magnetic_cycle_by_time = MagneticCycleByTime(
            base_time=np.linspace(1, 12, 12),
            base_values=np.linspace(1e9, 5e9, 12),
            reference_particle=proton,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        self.magnetic_cycle_by_time.on_init_simulation(simulation=simulation_ex1)

    def test_headless(self):
        ebt = MagneticCycleByTime.headless(
            base_time=np.linspace(1e12, 1e12, endpoint=True),
            base_values=np.linspace(1e12, 1e12, endpoint=True),
            reference_particle=uranium_29,
            in_unit="total energy",
        )
        self.assertEqual(
            ebt.get_total_energy_init(0, 0, particle_type=uranium_29), 1e12
        )


class TestEnergyCyclePerTurn(unittest.TestCase):
    def setUp(self):
        self.momentum = np.linspace(1, 10, 11)
        self.magnetic_cycle_per_turn = MagneticCyclePerTurn(
            value_init=float(self.momentum[0]),
            values_after_turn=self.momentum[1:],
            reference_particle=uranium_29,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        self.magnetic_cycle_per_turn.on_init_simulation(simulation=simulation_ex1)
        turn_i = 0
        assert_allclose(
            magnetic_rigidity_to_momentum(
                self.magnetic_cycle_per_turn._magnetic_rigidity[turn_i, :],
                uranium_29.charge,
            ),
            self.momentum[1:],
        )

    def test_two_rf(self):
        simulation = SimulationTwoRfStations().simulation
        self.magnetic_cycle_per_turn.on_init_simulation(simulation=simulation)
        cavity_i = 1
        assert_allclose(
            magnetic_rigidity_to_momentum(
                self.magnetic_cycle_per_turn._magnetic_rigidity[cavity_i, :],
                uranium_29.charge,
            ),
            self.momentum[1:],
        )

    def test_headless(self):
        evpt = MagneticCyclePerTurn.headless(
            reference_particle=uranium_29,
            value_init=0,
            n_cavities=2,
            values_after_turn=np.ones(10),
        )
        self.assertEqual(evpt._magnetic_rigidity.shape, (2, 10))


class TestEnergyCyclePerTurnAllCavities(unittest.TestCase):
    def setUp(self):
        self.momentum = np.ones((1, 10))
        self.magnetic_cycle_per_turn_all_cavities = MagneticCyclePerTurnAllCavities(
            values_after_cavity_per_turn=self.momentum,
            value_init=1,
            reference_particle=uranium_29,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_wrong_cavity_count(self):
        # simulation has only one cavity, but give program for 10 cavities
        with self.assertRaises(AssertionError):
            self.magnetic_cycle_per_turn_all_cavities = MagneticCyclePerTurnAllCavities(
                values_after_cavity_per_turn=np.ones((10, 10)),
                value_init=10,
                reference_particle=uranium_29,
            )
            self.magnetic_cycle_per_turn_all_cavities.on_init_simulation(
                simulation=simulation_ex1
            )

    def test_on_init_simulation(self):
        self.magnetic_cycle_per_turn_all_cavities.on_init_simulation(
            simulation=simulation_ex1,
            momentum_init=11,
            momentum=np.ones(
                (1, 10),
            ),
        )
        assert_allclose(
            self.momentum,
            magnetic_rigidity_to_momentum(
                self.magnetic_cycle_per_turn_all_cavities._magnetic_rigidity_after_cavity_per_turn,
                uranium_29.charge,
            ),
        )

    def test_headless(self):
        ecptac = MagneticCyclePerTurnAllCavities.headless(
            value_init=10,
            values_after_cavity_per_turn=np.ones((2, 20)),
            reference_particle=uranium_29,
        )
        self.assertEqual(
            (2, 20),
            magnetic_rigidity_to_momentum(
                ecptac._magnetic_rigidity_after_cavity_per_turn, proton.charge
            ).shape,
        )


if __name__ == "__main__":
    unittest.main()
