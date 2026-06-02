# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Verify that :class:`~blond.beam_preparation.bigaussian.BiGaussian` centres the
bunch on the stable fixed point of the RF bucket.

The :class:`~blond.utilities.separatrix.symbolic_separatrix.SymbolicSeparatrixHelper`
builds the *actual* Hamiltonian the particles experience from each ring
element's ``get_hamilton_symbolic()``. Its stable fixed point (SFP) in ``dt``
is, by definition, where a matched bunch must centre -- a convention-independent
oracle. ``BiGaussian`` independently derives the centre from
``(phi_s - phi_rf) / omega_rf``; a correct implementation puts
``beam.dt.mean()`` on the SFP (modulo the RF period).

The cases below exercise the regimes where the historical ``BiGaussian`` bugs
surface: below transition (extra ``pi`` shift), ``phi_rf != 0`` (double
``phi_rf`` subtraction), and the ``sigma_dE=None`` matching path (below-transition
NaN). Both muon charge signs are covered (mass != proton, ``+1`` and ``-1``).
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pytest
from matplotlib import pyplot as plt

from blond import (
    Beam,
    BiGaussian,
    momentum_compaction_factor,
    mu_minus,
    mu_plus,
)
from blond.convenience.single_section_setup import single_section_simulation
from blond.utilities.separatrix.symbolic_separatrix import (
    SymbolicSeparatrixHelper,
)

# Applies to every test in this module (including the unittest methods), so
# ``pytest -m "not integration"`` still excludes them.
pytestmark = pytest.mark.integration

_DEV_DRAW = False


class TestBiGaussianCenterOnStableFixedPoint(unittest.TestCase):
    """BiGaussian must centre the bunch on the separatrix stable fixed point."""

    # --- Machine constants (a muon ring; mass != proton, both charge signs) -
    CIRCUMFERENCE = 26658.883  # [m]
    P0 = 5e9  # reference momentum [eV/c] -> gamma ~ 47 for muons
    VOLTAGE = 6e6  # [V]
    HARMONIC = 35640
    PHI_RF = 0.3  # [rad], non-zero -> exercises the phi_rf double-subtraction
    N_TURNS = 10
    DP_PER_TURN = 2e6  # [eV/c] gentle ramp -> sin(phi_s) ~ 1/3, real bucket
    SIGMA_DT = 5e-11  # [s], comfortably inside the bucket
    N_MACROPARTICLES = int(2e4)

    # Tolerance on the centre-vs-SFP alignment, as a fraction of T_rf. Both
    # known bug magnitudes (phi_rf/omega ~ 0.048 T_rf and the pi-shift's
    # 0.5 T_rf) are far larger; the matched-bunch centre lands within ~1e-4.
    CENTER_ATOL_FRACTION = 0.01

    @staticmethod
    def _beam_gamma(particle, momentum: float) -> float:
        """Relativistic gamma of ``particle`` at ``momentum`` [eV/c]."""
        return float(np.sqrt(momentum**2 + particle.mass**2) / particle.mass)

    def _build_sim(self, particle, above_transition: bool):
        """A single-section accelerating muon simulation in the chosen regime.

        ``transition_gamma`` is placed below the beam ``gamma`` for the
        above-transition case and above it for the below-transition case, so
        the sign of ``eta`` (and hence the bucket orientation) is selected
        purely by the optics.
        """
        gamma = self._beam_gamma(particle, self.P0)
        transition_gamma = 0.5 * gamma if above_transition else 2.0 * gamma
        cycle_values = np.linspace(
            self.P0,
            self.P0 + self.DP_PER_TURN * self.N_TURNS,
            self.N_TURNS + 1,
        )
        return single_section_simulation(
            ring_circumference=self.CIRCUMFERENCE,
            cycle_values=cycle_values,
            cycle_unit="momentum",
            particle_type=particle,
            ring_momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=transition_gamma
            ),
            cavity_voltage=self.VOLTAGE,
            cavity_phi_rf=self.PHI_RF,
            cavity_harmonic=self.HARMONIC,
            cavity_n_harmonics=1,
        )

    @staticmethod
    def _t_rf(simulation) -> float:
        """RF period [s] of the (single) main-harmonic cavity."""
        from blond.physics.cavities import RFStationBaseClass

        rf_station = simulation.ring.elements.get_elements(RFStationBaseClass)[
            0
        ]
        return 2.0 * np.pi / float(rf_station.omega_rf_design)

    @staticmethod
    def _wrap(x: float, period: float) -> float:
        """Map ``x`` into ``[-period/2, period/2)`` for a modulo comparison."""
        return (x + period / 2.0) % period - period / 2.0

    def _prepare_and_measure(
        self, particle, above_transition: bool, reinsertion: bool
    ):
        """Prepare a BiGaussian bunch and return what is needed to judge it.

        ``sigma_dE=None`` is deliberate: it forces the dt->dE matching path
        (``_get_dE_from_dt``), exercising the below-transition NaN bug rather
        than bypassing it with an explicit ``sigma_dE``.
        """
        simulation = self._build_sim(
            particle, above_transition=above_transition
        )
        beam = Beam(intensity=1e9, particle_type=particle)
        simulation.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                sigma_dt=self.SIGMA_DT,
                sigma_dE=None,
                reinsertion=reinsertion,
                seed=1,
                n_macroparticles=self.N_MACROPARTICLES,
            ),
        )
        helper = SymbolicSeparatrixHelper.from_simulation(
            simulation=simulation
        )
        sfp_dt = helper.get_stable_fixed_point(beam=beam)
        return simulation, beam, helper, sfp_dt

    def _draw(self, simulation, beam, helper, sfp_dt, title):
        """DEV_DRAW visual: hist2d + separatrix + stable-fixed-point marker."""
        t_rf = self._t_rf(simulation)
        dt = np.asarray(beam.read_partial_dt())
        dE = np.asarray(beam.read_partial_dE())
        plt.figure(title)
        plt.hist2d(dt, dE, bins=120, cmin=1)
        dt_grid = np.linspace(sfp_dt - t_rf, sfp_dt + t_rf, 1000)
        helper.plot_separatrix(beam=beam, dt=dt_grid, label="separatrix")
        plt.axvline(
            sfp_dt, color="k", linestyle=":", label="stable fixed point"
        )
        plt.axvline(float(np.mean(dt)), color="lime", label="bunch centre")
        plt.title(title)
        plt.legend()
        if _DEV_DRAW:
            plt.show()

    def _assert_center_on_sfp(
        self, simulation, beam, helper, sfp_dt, title
    ) -> None:
        """Assert the bunch centre lands on the SFP (modulo T_rf)."""
        t_rf = self._t_rf(simulation)
        self.assertTrue(np.isfinite(sfp_dt), "oracle found no bucket")
        center = float(np.mean(np.asarray(beam.read_partial_dt())))
        self._draw(simulation, beam, helper, sfp_dt, title)
        offset = self._wrap(center - sfp_dt, t_rf)
        np.testing.assert_allclose(
            offset,
            0.0,
            atol=self.CENTER_ATOL_FRACTION * t_rf,
            err_msg=(
                f"{title}: bunch centre {center:.3e}s is off the stable fixed "
                f"point {sfp_dt:.3e}s by {offset:.3e}s "
                f"({offset / t_rf:.3f} of T_rf={t_rf:.3e}s)"
            ),
        )

    def test_center_on_stable_fixed_point(self):
        """mu+/- above and below transition land on the SFP (reinsertion off)."""
        for particle, particle_id in ((mu_plus, "mu+"), (mu_minus, "mu-")):
            for above_transition in (True, False):
                with self.subTest(
                    particle=particle_id, above_transition=above_transition
                ):
                    simulation, beam, helper, sfp_dt = (
                        self._prepare_and_measure(
                            particle,
                            above_transition=above_transition,
                            reinsertion=False,
                        )
                    )
                    regime = "above" if above_transition else "below"
                    self._assert_center_on_sfp(
                        simulation,
                        beam,
                        helper,
                        sfp_dt,
                        f"BiGaussian {particle_id} {regime} transition",
                    )

    def test_reinsertion_keeps_center_and_stays_in_bucket(self):
        """``reinsertion=True``: whole bunch inside the separatrix and its
        centre still on the SFP (reinsertion must not bias the centre)."""
        for above_transition in (True, False):
            with self.subTest(above_transition=above_transition):
                simulation, beam, helper, sfp_dt = self._prepare_and_measure(
                    mu_plus,
                    above_transition=above_transition,
                    reinsertion=True,
                )
                self.assertTrue(np.isfinite(sfp_dt), "oracle found no bucket")

                dt = np.asarray(beam.read_partial_dt())
                dE = np.asarray(beam.read_partial_dE())
                # Every particle must lie between the two separatrix branches.
                upper, lower = helper.get_separatrix(beam=beam, dt=dt)
                inside = (
                    np.isfinite(upper)
                    & np.isfinite(lower)
                    & (dE < upper)
                    & (dE > lower)
                )
                self.assertTrue(
                    inside.all(),
                    f"{(~inside).sum()} / {dt.size} reinserted particles fall "
                    f"outside the separatrix",
                )

                regime = "above" if above_transition else "below"
                self._assert_center_on_sfp(
                    simulation,
                    beam,
                    helper,
                    sfp_dt,
                    f"BiGaussian reinsertion mu+ {regime} transition",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
