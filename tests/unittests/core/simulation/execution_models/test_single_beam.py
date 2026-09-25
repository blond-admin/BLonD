# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""The main loop must flush kernels a deferring backend has queued."""

from __future__ import annotations

import numpy as np
import pytest

from blond import (
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
    proton,
)
from blond.core.backends.backend import Numpy64Bit
from blond.physics.impedances.solvers import PeriodicFreqSolver
from blond.testing.backend_testing import BLonDTestCase

N_TURNS = 20
PROFILE_LENGTH = 2.1e-6  # s


def _run(specials_mode: str, callback=None) -> tuple[np.ndarray, ...]:
    """Track a small PSB-like ring, return the final dt, dE, profile."""
    backend.change_backend(Numpy64Bit)
    backend.set_specials(specials_mode)
    ring = Ring(circumference=2 * np.pi * 100)
    magnetic_cycle = MagneticCyclePerTurn(
        value_init=310e6,
        values_after_turn=np.linspace(310e6, 320e6, N_TURNS),
        reference_particle=proton,
    )
    rf_station = SingleHarmonicRFStation()
    rf_station.harmonic = 1
    rf_station.voltage = 8e3
    rf_station.phi_rf_design = 0.0
    drift = DriftSimple(orbit_length=2 * np.pi * 100)
    drift.momentum_compaction_factor = 1 / 4.4**2
    profile = StaticProfile(cut_left=0, cut_right=PROFILE_LENGTH, n_bins=128)
    wakefield = WakeField(
        sources=(
            Resonators(np.array([1e4]), np.array([4e6]), np.array([3.0])),
        ),
        solver=PeriodicFreqSolver(PROFILE_LENGTH, allow_next_fast_len=True),
        profile=profile,
    )
    ring.add_elements(
        (rf_station, drift, wakefield), reorder=False, section_index=0
    )
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    rng = np.random.default_rng(seed=1)
    beam = Beam(intensity=1e13, particle_type=proton)
    beam.setup_beam(
        dt=rng.normal(1.0e-6, 1e-7, 10_000),
        dE=rng.normal(0, 1e6, 10_000),
        reference_total_energy=magnetic_cycle.get_total_energy_init(
            particle_type=beam.particle_type
        ),
    )
    profile.track(beam=beam)
    simulation.finalize(beams=(beam,), n_turns=N_TURNS)
    simulation.mainloop(
        beams=(beam,),
        n_turns=N_TURNS,
        show_progressbar=False,
        callbacks=callback,
    )
    return (
        beam.read_partial_dt().copy(),
        beam.read_partial_dE().copy(),
        profile.hist_y.copy(),
    )


class TestMainloopSingleBeamDeferred(BLonDTestCase):
    def tearDown(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

    @pytest.mark.backend_mutation
    def test_deferred_matches_eager(self):
        for eager, deferred in zip(
            _run("cpp"), _run("cpp_deferred"), strict=True
        ):
            np.testing.assert_allclose(deferred, eager, rtol=1e-12)
        self.assertEqual(backend.specials.n_pending(), 0)

    @pytest.mark.backend_mutation
    def test_callback_sees_flushed_beam(self):
        mean_dE: dict[str, list[float]] = {"cpp": [], "cpp_deferred": []}
        for mode, means in mean_dE.items():

            def record(simulation, beam, means=means):
                means.append(float(np.mean(beam.read_partial_dE())))

            record.each_turn_i = 1
            _run(mode, callback=record)
        np.testing.assert_allclose(
            mean_dE["cpp_deferred"], mean_dE["cpp"], rtol=1e-12
        )
