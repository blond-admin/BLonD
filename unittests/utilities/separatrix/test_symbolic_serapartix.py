# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/
import unittest
from unittest.mock import Mock

import numpy as np
import pytest
import sympy
from matplotlib import pyplot as plt

from blond import (
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)
from blond.handle_results.helpers import callers_relative_path
from blond.physics.drifts import DriftExact
from blond.testing.backend_testing import multi_backend_testcase
from blond.testing.helpers import allclose_tolerances
from blond.utilities.separatrix.symbolic_serapartix import (
    SymbolicSeparatrixHelper,
)


class TestSymbolicSeparatrixHelper:
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @multi_backend_testcase("Cupy64Bit", "Numpy64Bit")
    @pytest.mark.backend_mutation
    def test_integration(self):
        DEV_DRAW = False
        ring = Ring(26658.883)

        rf_station1 = MultiHarmonicRFStation(
            section_index=0, n_harmonics=2, main_harmonic_idx=0
        )
        base_harmonic = 35640
        rf_station1.harmonic = np.array([base_harmonic, 4 * base_harmonic])
        rf_station1.voltage = np.array([6e6, 6e6 / 2])
        rf_station1.phi_rf_design = np.array([0, 0])
        N_TURNS = int(1e3)

        energy_cycle = MagneticCyclePerTurn.init_from_linspace(
            values=np.linspace(450e9, 451e9, N_TURNS + 1),
            reference_particle=proton,
        )

        drift1 = DriftSimple(
            section_index=0,
            orbit_length=ring.circumference / 2,
        )
        drift1.momentum_compaction_factor = momentum_compaction_factor(
            transition_gamma=55.759505
        )

        drift2 = DriftExact(
            orbit_length=ring.circumference / 2,
            section_index=1,
            momentum_compaction_factor=drift1.momentum_compaction_factor,
            higher_order_alpha=np.array(
                [drift1.alpha_0 * 2, drift1.alpha_0 * (-3)]
            ),
        )

        rf_station2 = SingleHarmonicRFStation(
            section_index=1,
            harmonic=base_harmonic,
            voltage=6e6,
            phi_rf=np.deg2rad(20),
        )

        ring.add_elements(
            (drift1, rf_station1), deepcopy=True, section_index=0
        )
        ring.add_elements(
            (drift2, rf_station2), deepcopy=True, section_index=1
        )

        sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

        t_rf = sim.get_t_rev_init() / base_harmonic

        beam1 = Beam.simple_gaussian(
            n_macroparticles=1e5,
            intensity=1e9,
            particle_type=proton,
            dt_scale=t_rf / 4,
            dE_scale=0.1e9 / 2,
            dt_offset=t_rf / 2,
            seed=1,
        )
        t0 = beam1.dt.min()
        t1 = beam1.dt.max()
        r = t1 - t0
        trange0_ = (t0 - 2 * r, t1 + 2 * r)
        plt.figure("Dynamic beam")
        plt.xlim(trange0_)

        def custom_action(
            simulation: Simulation, beam: Beam
        ):  # pragma: no cover
            plt.figure("Dynamic beam")
            if simulation.turn_i.value % 10 != 0:
                return

            dt = beam.read_partial_dt()
            beam.plot_scatter()
            separatrix_dE = SymbolicSeparatrixHelper.from_simulation(
                simulation=sim
            ).get_separatrix(
                beam=beam,
                dt=np.linspace(*trange0_, 1000),
            )
            if simulation.turn_i.value == 0:
                separatrix_dE_pinned = np.loadtxt(
                    callers_relative_path(
                        "resources/separatrix_dE_pinned.txt", stacklevel=1
                    ),
                )
                np.testing.assert_allclose(
                    separatrix_dE,
                    separatrix_dE_pinned,
                    **allclose_tolerances(separatrix_dE_pinned),
                )
            if DEV_DRAW:
                sim.plot_separatrix(
                    beam=beam,
                    dt=np.linspace(*trange0_, 1000),
                )
                plt.xlim(trange0_)
                plt.ylim(-2e9, 2e9)

                plt.draw()
                plt.pause(0.1)
                plt.cla()

        sim.run_simulation(
            beams=(beam1,),
            n_turns=N_TURNS if DEV_DRAW else 1,
            callbacks=custom_action,
        )


class TestSymbolicSeparatrixInternals:
    """Cover edge-case branches of the private helpers."""

    OMEGA_MIN = 2.0 * np.pi  # canonical period of 1 s

    def _helper(self) -> SymbolicSeparatrixHelper:
        return SymbolicSeparatrixHelper(
            hamiltonian=sympy.Integer(0), omega_min=self.OMEGA_MIN
        )

    def test_interior_extrema_negative_a_finds_local_minima(self):
        values = np.array([1.0, 0.0, 1.0])
        idx = SymbolicSeparatrixHelper._interior_extrema(values, a=-1.0)
        np.testing.assert_array_equal(idx, np.array([1]))

    def test_interior_extrema_negative_a_ignores_local_maxima(self):
        values = np.array([0.0, 1.0, 0.0])
        idx = SymbolicSeparatrixHelper._interior_extrema(values, a=-1.0)
        assert idx.size == 0

    def test_find_canonical_bucket_no_extremum_returns_none(self):
        helper = self._helper()
        bucket = helper._find_canonical_bucket(
            period_start=0.0,
            a=1.0,
            potential=lambda dt: np.asarray(dt, dtype=float),
        )
        assert bucket is None

    def test_find_canonical_bucket_negative_a_picks_local_minimum(self):
        helper = self._helper()

        def potential(dt):
            return np.sin(2 * np.pi * dt) - 0.1 * dt

        bucket = helper._find_canonical_bucket(
            period_start=0.0, a=-1.0, potential=potential
        )
        assert bucket is not None
        assert 0.7 < bucket.ufp_dt < 0.8
        np.testing.assert_allclose(bucket.ufp_potential, -1.075, atol=0.01)
        np.testing.assert_allclose(bucket.shift_per_period, -0.1, atol=1e-9)

    def test_H_sep_per_dt_zero_a_returns_all_nan(self):
        helper = self._helper()
        dt = np.linspace(0.0, 1.0, 11)
        H_sep = helper._H_sep_per_dt(
            dt, a=0.0, potential=lambda x: np.cos(2 * np.pi * x)
        )
        assert H_sep.shape == dt.shape
        assert np.all(np.isnan(H_sep))

    def test_H_sep_per_dt_no_canonical_bucket_returns_all_nan(self):
        helper = self._helper()
        dt = np.linspace(0.0, 1.0, 11)
        H_sep = helper._H_sep_per_dt(
            dt,
            a=1.0,
            potential=lambda x: np.asarray(x, dtype=float),
        )
        assert np.all(np.isnan(H_sep))

    def test_H_sep_per_dt_negative_a_uses_maximum_branch(self):
        helper = self._helper()

        def potential(dt):
            return np.sin(2 * np.pi * dt) - 0.1 * dt

        dt = np.array([0.6, 0.9, 1.2])
        H_sep = helper._H_sep_per_dt(dt, a=-1.0, potential=potential)

        bucket = helper._find_canonical_bucket(
            period_start=float(np.min(dt)), a=-1.0, potential=potential
        )
        assert bucket is not None
        bucket_index = helper._bucket_index(dt, bucket)
        left = bucket.ufp_potential + bucket_index * bucket.shift_per_period
        right = left + bucket.shift_per_period
        # Sanity: shift_per_period != 0 so np.maximum and np.minimum diverge.
        assert not np.allclose(left, right)
        np.testing.assert_allclose(H_sep, np.maximum(left, right))


class TestSymbolicSeparatrixHelperFromSimulation:
    """Cover `SymbolicSeparatrixHelper.from_simulation`."""

    def test_raises_when_no_symbolic_hamiltonian_elements(self):
        simulation = Mock()
        simulation.ring.elements.get_elements.return_value = []

        with pytest.raises(
            ValueError,
            match="No elements with `HasSymbolicHamiltonian` found.",
        ):
            SymbolicSeparatrixHelper.from_simulation(simulation=simulation)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
