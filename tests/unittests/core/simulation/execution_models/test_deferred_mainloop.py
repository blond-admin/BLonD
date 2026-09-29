import warnings
from copy import deepcopy

import numpy as np
import pytest

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)
from blond.core.backends.backend import Numpy64Bit, backend
from blond.core.simulation.execution_models.conterrotating_beams import (
    MainloopCounterRotatingBeams,
)
from blond.testing.backend_testing import BLonDTestCase


def _build_two_section_sim() -> tuple[Simulation, Beam]:
    """Build a minimal two-section ring (RF + drift per section)."""
    circumference = 26658.883
    ring = Ring(circumference=circumference)

    cavity0 = SingleHarmonicRFStation(section_index=0)
    cavity0.harmonic = 35640
    cavity0.voltage = 6e6
    cavity0.phi_rf_design = 0
    drift0 = DriftSimple(orbit_length=circumference / 2, section_index=0)
    drift0.momentum_compaction_factor = momentum_compaction_factor(
        transition_gamma=55.76
    )

    cavity1 = SingleHarmonicRFStation(section_index=1)
    cavity1.harmonic = 35640
    cavity1.voltage = 6e6
    cavity1.phi_rf_design = 0
    drift1 = DriftSimple(orbit_length=circumference / 2, section_index=1)
    drift1.momentum_compaction_factor = momentum_compaction_factor(
        transition_gamma=55.76
    )

    magnetic_cycle = ConstantMagneticCycle(
        value=450e9, reference_particle=proton
    )

    beam = Beam(intensity=1e9, particle_type=proton)
    beam.setup_beam(
        dt=np.linspace(1e-9, 10e-9, 1000),
        dE=np.linspace(-1e6, 1e6, 1000),
        reference_time=0,
        reference_total_energy=450e9,
    )

    sim = Simulation.from_locals(locals())
    return sim, beam


def _run_until_section_index(mode):
    backend.change_backend(Numpy64Bit)
    backend.set_specials(mode)
    sim, beam = _build_two_section_sim()
    sim.run_simulation(
        beams=(beam,),
        n_turns=1,
        until_section_index=1,
        show_progressbar=False,
    )
    # Read the queue state *before* touching any flushing accessor
    # (`read_partial_dt`/`read_partial_dE` flush as a side effect and
    # would hide a missing flush at the early `return`).
    n_bytes = (
        backend.specials.kernel_call_queue.n_bytes
        if mode == "cpp_deferred"
        else 0
    )
    return (
        n_bytes,
        beam.read_partial_dt().copy(),
        beam.read_partial_dE().copy(),
    )


def _run_until_section_index_counterrotating(mode):
    backend.change_backend(Numpy64Bit)
    backend.set_specials(mode)
    with warnings.catch_warnings():
        # `MainloopCounterRotatingBeams` is marked untested code.
        warnings.simplefilter("ignore")
        sim, beam = _build_two_section_sim()
        beam_cr = deepcopy(beam)
        beam_cr._is_counter_rotating = True
        sim.execution_model = MainloopCounterRotatingBeams()
        sim.run_simulation(
            beams=(beam, beam_cr),
            n_turns=1,
            until_section_index=1,
            show_progressbar=False,
        )
    n_bytes = (
        backend.specials.kernel_call_queue.n_bytes
        if mode == "cpp_deferred"
        else 0
    )
    return (
        n_bytes,
        beam.read_partial_dt().copy(),
        beam.read_partial_dE().copy(),
    )


def _run(mode):
    backend.change_backend(Numpy64Bit)
    backend.set_specials(mode)
    from blond.examples.scripts import EX_23_Main_long_ps_booster as ex

    seen = []

    def record(simulation, beam):
        seen.append(beam.read_partial_dE().copy())

    record.each_turn_i = 1
    sim, beam = ex.build(n_macroparticles=10_000, n_bins=1000)  # Step 3
    sim.run_simulation(beams=(beam,), n_turns=5, callbacks=[record])
    return seen, beam.read_partial_dt().copy()


@pytest.mark.backend_mutation
class TestDeferredMainloop(BLonDTestCase):
    def tearDown(self) -> None:
        backend.specials.flush()
        backend.set_specials("python")
        backend.change_backend(Numpy64Bit)

    def test_deferred_matches_eager_and_callbacks_see_flushed_beam(self):
        eager_turns, eager_dt = _run("cpp")
        deferred_turns, deferred_dt = _run("cpp_deferred")
        self.assertEqual(len(eager_turns), len(deferred_turns))
        for eager, deferred in zip(eager_turns, deferred_turns):
            np.testing.assert_allclose(deferred, eager, rtol=1e-12)
        np.testing.assert_allclose(deferred_dt, eager_dt, rtol=1e-12)
        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)

    def test_end_of_loop_flushes_pending_calls_with_no_readout(self):
        # A callback that does not fire on the last turn (and no
        # observables) leaves nothing to trigger
        # `flush_before_readout` during the loop. Only the
        # end-of-loop flush (Step 4, point 3) can empty the queue
        # before `run_simulation` returns.
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp_deferred")
        from blond.examples.scripts import EX_23_Main_long_ps_booster as ex

        def record(simulation, beam):
            pass

        record.each_turn_i = 3
        sim, beam = ex.build(n_macroparticles=10_000, n_bins=1000)
        sim.run_simulation(beams=(beam,), n_turns=5, callbacks=[record])
        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)

    def test_early_return_flushes_pending_calls(self):
        # `until_section_index` makes the loop `return` after the
        # first section's kernel calls (RF kick + drift) are queued
        # but before the second section is reached. Only the flush
        # at the early `return` (Step 4, point 1) empties the queue
        # before `run_simulation` returns.
        _, eager_dt, eager_dE = _run_until_section_index("cpp")
        n_bytes, deferred_dt, deferred_dE = _run_until_section_index(
            "cpp_deferred"
        )
        np.testing.assert_allclose(deferred_dt, eager_dt, rtol=1e-12)
        np.testing.assert_allclose(deferred_dE, eager_dE, rtol=1e-12)
        self.assertEqual(n_bytes, 0)

    def test_early_return_flushes_pending_calls_counterrotating(self):
        # Same as `test_early_return_flushes_pending_calls`, but for
        # `MainloopCounterRotatingBeams`, which has its own early
        # `return`/flush point in `conterrotating_beams.py`.
        _, eager_dt, eager_dE = _run_until_section_index_counterrotating("cpp")
        n_bytes, deferred_dt, deferred_dE = (
            _run_until_section_index_counterrotating("cpp_deferred")
        )
        np.testing.assert_allclose(deferred_dt, eager_dt, rtol=1e-12)
        np.testing.assert_allclose(deferred_dE, eager_dE, rtol=1e-12)
        self.assertEqual(n_bytes, 0)
