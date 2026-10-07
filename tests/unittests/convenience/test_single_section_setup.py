import unittest

import numpy as np

from blond import (
    ConstantMagneticCycle,
    Resonators,
    TimeDomainFftSolver,
    WakeField,
    proton,
)
from blond.convenience.single_section_setup import single_section_simulation
from blond.core.scheduling import ScheduledArray
from blond.testing.backend_testing import BLonDTestCase


class TestCallables(BLonDTestCase):
    def test_executes(self):
        for cycle in (1e12, 1e12 * np.ones(10)):
            single_section_simulation(
                ring_circumference=123,
                cycle_values=cycle,
                cycle_unit="momentum",
                particle_type=proton,
                ring_momentum_compaction_factor=12,
                cavity_voltage=1e3,
                cavity_phi_rf=0,
                cavity_harmonic=123,
                cavity_n_harmonics=1,
                wakefield_impedance_sources=None,
                wakefield_solver=None,
                wakefield_cutoff_frequency=None,
            )
        for ring_momentum_compaction_factor in (
            12,
            ScheduledArray(12 * np.ones(10)),
        ):
            single_section_simulation(
                ring_circumference=123,
                cycle_values=cycle,
                cycle_unit="momentum",
                particle_type=proton,
                ring_momentum_compaction_factor=ring_momentum_compaction_factor,
                cavity_voltage=1e3,
                cavity_phi_rf=0,
                cavity_harmonic=123,
                cavity_n_harmonics=1,
                wakefield_impedance_sources=None,
                wakefield_solver=None,
                wakefield_cutoff_frequency=None,
            )

        for param in (1.0, 1.0 * np.ones(3)):
            single_section_simulation(
                ring_circumference=123,
                cycle_values=cycle,
                cycle_unit="momentum",
                particle_type=proton,
                ring_momentum_compaction_factor=12,
                cavity_voltage=param,
                cavity_phi_rf=param,
                cavity_harmonic=param,
                cavity_n_harmonics=1
                if isinstance(param, float)
                else len(param),
                wakefield_impedance_sources=None,
                wakefield_solver=None,
                wakefield_cutoff_frequency=None,
            )
        single_section_simulation(
            ring_circumference=123,
            cycle_values=cycle,
            cycle_unit="momentum",
            particle_type=proton,
            ring_momentum_compaction_factor=12,
            cavity_voltage=ScheduledArray(12 * np.ones(10)),
            cavity_phi_rf=ScheduledArray(12 * np.ones(10)),
            cavity_harmonic=ScheduledArray(12 * np.ones(10)),
            cavity_n_harmonics=1,
            wakefield_impedance_sources=None,
            wakefield_solver=None,
            wakefield_cutoff_frequency=None,
        )

    def test_executes_raises(self):
        with self.assertRaises(TypeError):
            single_section_simulation(
                ring_circumference=123,
                cycle_values=None,
                cycle_unit="momentum",
                particle_type=proton,
                ring_momentum_compaction_factor=12,
                cavity_voltage=1e3,
                cavity_phi_rf=0,
                cavity_harmonic=123,
                cavity_n_harmonics=1,
                wakefield_impedance_sources=None,
                wakefield_solver=None,
                wakefield_cutoff_frequency=None,
            )

    def test_executes_with_wakes(self):
        single_section_simulation(
            ring_circumference=123,
            cycle_values=1e12,
            cycle_unit="momentum",
            particle_type=proton,
            ring_momentum_compaction_factor=12,
            cavity_voltage=1e3,
            cavity_phi_rf=0,
            cavity_harmonic=123,
            cavity_n_harmonics=1,
            wakefield_impedance_sources=(
                Resonators(
                    shunt_impedances=np.array([1, 2, 3]),
                    center_frequencies=np.array([500e6, 750e6, 1.5e9]),
                    quality_factors=np.array([5, 5, 5]),
                ),
            ),
            wakefield_solver=TimeDomainFftSolver(),
            wakefield_cutoff_frequency=2e9,
        )


class TestSingleSectionSimulationTypeCheckFindings(BLonDTestCase):
    """Red tests for bugs surfaced while introducing ``ty``."""

    def test_int_cycle_values(self):
        """A plain ``int`` is a valid constant cycle value, like ``float``."""
        simulation = single_section_simulation(
            ring_circumference=123,
            cycle_values=10**12,
            cycle_unit="momentum",
            particle_type=proton,
            ring_momentum_compaction_factor=12,
            cavity_voltage=1e3,
            cavity_phi_rf=0,
            cavity_harmonic=123,
            cavity_n_harmonics=1,
        )
        self.assertIsInstance(simulation.magnetic_cycle, ConstantMagneticCycle)

    def test_multi_harmonic_with_wakes(self):
        """With several harmonics the wakefield profile must span one
        main-harmonic bucket (``t_rev / harmonic[main_harmonic_idx]``)
        instead of receiving an array as ``cut_right``."""
        sources = (
            Resonators(
                shunt_impedances=np.array([1, 2, 3]),
                center_frequencies=np.array([500e6, 750e6, 1.5e9]),
                quality_factors=np.array([5, 5, 5]),
            ),
        )
        simulation = single_section_simulation(
            ring_circumference=123,
            cycle_values=1e12,
            cycle_unit="momentum",
            particle_type=proton,
            ring_momentum_compaction_factor=12,
            cavity_voltage=np.array([1e3, 5e2]),
            cavity_phi_rf=np.array([0.0, 0.0]),
            cavity_harmonic=np.array([123.0, 246.0]),
            cavity_n_harmonics=2,
            wakefield_impedance_sources=sources,
            wakefield_solver=TimeDomainFftSolver(),
            wakefield_cutoff_frequency=2e9,
        )
        t_rev = simulation.magnetic_cycle.get_t_rev_init(
            circumference=123, particle_type=proton
        )
        profile = simulation.ring.elements.get_element(WakeField).profile
        self.assertAlmostEqual(
            float(profile.cut_right), t_rev / 123.0, delta=1e-3 * t_rev
        )


if __name__ == "__main__":
    unittest.main()
