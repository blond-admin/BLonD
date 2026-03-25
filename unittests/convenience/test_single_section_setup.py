import unittest

import numpy as np

from blond import Resonators, TimeDomainFftSolver, proton
from blond.convenience.single_section_setup import single_section_simulation
from blond.core.base import ScheduledArray


class TestCallables(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
