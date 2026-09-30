import threading
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


def _run_in_worker_thread(mode):
    backend.change_backend(Numpy64Bit)
    backend.set_specials(mode)
    from blond.examples.scripts import EX_23_Main_long_ps_booster as ex

    sim, beam = ex.build(n_macroparticles=10_000, n_bins=1000)
    worker = threading.Thread(
        target=sim.run_simulation,
        kwargs=dict(beams=(beam,), n_turns=5, show_progressbar=False),
    )
    worker.start()
    worker.join()
    return beam.read_partial_dt().copy(), beam.read_partial_dE().copy()


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
        # Direct storage access: flushes by itself, no accessor needed.
        seen.append(beam._dE.array_local.copy())

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
        # observables) reads nothing at the end of the loop; the queue
        # is still empty when `run_simulation` returns, because
        # `Simulation.mainloop` flushes once after the execution model.
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp_deferred")
        from blond.examples.scripts import EX_23_Main_long_ps_booster as ex

        def record(simulation, beam):
            pass

        record.each_turn_i = 3
        sim, beam = ex.build(n_macroparticles=10_000, n_bins=1000)
        sim.run_simulation(beams=(beam,), n_turns=5, callbacks=[record])
        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)

    def test_worker_thread_run_leaves_nothing_queued(self):
        # The queue is per thread and dies with it: a run in a worker
        # thread must not leave its last turn queued there, or the main
        # thread would read stale coordinates.
        eager_dt, eager_dE = _run_in_worker_thread("cpp")
        deferred_dt, deferred_dE = _run_in_worker_thread("cpp_deferred")
        np.testing.assert_allclose(deferred_dt, eager_dt, rtol=1e-12)
        np.testing.assert_allclose(deferred_dE, eager_dE, rtol=1e-12)

    def test_early_return_flushes_pending_calls(self):
        # `until_section_index` makes the loop `return` after the
        # first section's kernel calls (RF kick + drift) are queued
        # but before the second section is reached. The flush in
        # `Simulation.mainloop` covers this return path too.
        _, eager_dt, eager_dE = _run_until_section_index("cpp")
        n_bytes, deferred_dt, deferred_dE = _run_until_section_index(
            "cpp_deferred"
        )
        np.testing.assert_allclose(deferred_dt, eager_dt, rtol=1e-12)
        np.testing.assert_allclose(deferred_dE, eager_dE, rtol=1e-12)
        self.assertEqual(n_bytes, 0)

    def test_worker_thread_exception_still_applies_queued_kick(self):
        # If the execution model raises mid-turn, the queued kernel
        # calls from earlier in that turn must still be applied (the
        # end-of-run flush runs in `finally`), and the original
        # exception must still propagate, not be masked by the flush.
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp_deferred")
        sim, beam = _build_two_section_sim()
        sim.finalize(beams=(beam,), n_turns=1)
        dt_before = beam.read_partial_dt().copy()

        def _mainloop_that_queues_then_raises(**kwargs) -> None:
            queued_beam = kwargs["beams"][0]
            backend.specials.drift_simple(
                dt=queued_beam.write_partial_dt(),
                dE=queued_beam.read_partial_dE(),
                T=1e-6,
                eta_0=0.01,
                beta=0.9,
                energy=450e9,
            )
            raise RuntimeError("boom")

        sim.execution_model.mainloop = _mainloop_that_queues_then_raises
        with self.assertRaisesRegex(RuntimeError, "boom"):
            sim.mainloop(beams=(beam,), n_turns=1, show_progressbar=False)
        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
        dt_after = beam.read_partial_dt().copy()
        self.assertFalse(np.array_equal(dt_before, dt_after))

    def test_early_return_flushes_pending_calls_counterrotating(self):
        # Same as `test_early_return_flushes_pending_calls`, but for
        # `MainloopCounterRotatingBeams`, which has its own early
        # `return` in `conterrotating_beams.py`.
        _, eager_dt, eager_dE = _run_until_section_index_counterrotating("cpp")
        n_bytes, deferred_dt, deferred_dE = (
            _run_until_section_index_counterrotating("cpp_deferred")
        )
        np.testing.assert_allclose(deferred_dt, eager_dt, rtol=1e-12)
        np.testing.assert_allclose(deferred_dE, eager_dE, rtol=1e-12)
        self.assertEqual(n_bytes, 0)
