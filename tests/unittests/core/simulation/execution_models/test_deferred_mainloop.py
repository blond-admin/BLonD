import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.testing.backend_testing import BLonDTestCase


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
