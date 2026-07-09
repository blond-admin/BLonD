"""BLonD2 vs BLonD3 regression: RF + drift only (single macroparticle).

Authors: Oliver Muller Smedt, Simon Lauber
"""

import os
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from blond.core.backends.backend import (
    Cupy64Bit,
    Numpy64Bit,
    backend,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu

_DEV_DRAW = os.getenv("DEV_DRAW", "False").lower() == "true"
_RESOURCES = Path(__file__).parent / "resources"


@pytest.mark.integration
class TestKickDrift(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_kickdrift_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        self._run_kickdrift()

    @pytest.mark.backend_mutation
    def test_kickdrift_cuda64(self):
        try:
            import cupy  # type: ignore # noqa: F401
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        self._run_kickdrift()

    def _run_kickdrift(self):
        phi_rf = np.genfromtxt(_RESOURCES / "phase.txt", delimiter=",")
        transition_gamma = np.genfromtxt(
            _RESOURCES / "gamma.txt", delimiter=","
        )
        momentum = np.genfromtxt(_RESOURCES / "momentum.txt", delimiter=",")

        # The BLonD2-vs-BLonD3 comparison runs a Python-level per-turn
        # tracking loop on *both* frameworks, so its cost scales with the
        # number of turns. The full ~7000-turn programme made this the single
        # slowest test on CI. A kick/drift discrepancy diverges within a
        # handful of turns, so the first SIM_TURNS turns exercise the same
        # machinery at a fraction of the cost. Slicing all three per-turn
        # programmes identically keeps the BLonD2 and BLonD3 inputs in
        # lock-step.
        SIM_TURNS = 100
        phi_rf = phi_rf[: SIM_TURNS + 1]
        transition_gamma = transition_gamma[: SIM_TURNS + 1]
        momentum = momentum[: SIM_TURNS + 1]

        from blond import momentum_compaction_factor

        momentum_compaction_factor_ = momentum_compaction_factor(
            transition_gamma=transition_gamma
        )

        N_TURNS = len(momentum) - 1
        CIRCUMFERENCE = 2 * np.pi * 100
        VOLTAGE = 200e3
        HARMONIC = 8
        INTENSITY = 1
        INITIAL_E = np.array([25e6])
        INITIAL_T = np.array([0.4e-6])
        N_MACROS = len(INITIAL_E)

        # ── BLonD 3 ──────────────────────────────────────────────────────
        from blond import (
            Beam,
            BeamObservationOncePerTurn,
            DriftSimple,
            MagneticCyclePerTurn,
            MultiHarmonicRFStation,
            RFStationPhaseObservation,
            Ring,
            Simulation,
            proton,
        )

        ring = Ring(circumference=CIRCUMFERENCE)
        energy_cycle = MagneticCyclePerTurn(
            value_init=float(momentum[0]),
            values_after_turn=momentum[1:].copy(),
            reference_particle=proton,
        )

        rf_station1 = MultiHarmonicRFStation(
            n_harmonics=1, main_harmonic_idx=0
        )
        rf_station1.harmonic = np.array([HARMONIC])
        rf_station1.voltage = np.array([VOLTAGE])
        rf_station1.schedule(
            "phi_rf_design", phi_rf[:-1].copy()[:, np.newaxis]
        )

        drift1 = DriftSimple(orbit_length=CIRCUMFERENCE)
        drift1.schedule(
            "momentum_compaction_factor",
            momentum_compaction_factor_[1:].copy(),
        )

        beam1 = Beam(intensity=INTENSITY, particle_type=proton)
        beam1.setup_beam(dt=INITIAL_T.copy(), dE=INITIAL_E.copy())
        ring.add_elements(
            (rf_station1, drift1), reorder=False, section_index=0
        )
        sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

        bunch_observation = BeamObservationOncePerTurn(each_turn_i=1)
        cavity_obs = RFStationPhaseObservation(
            each_turn_i=1, rf_station=rf_station1
        )

        def _blond3_callback(simulation: Simulation):
            if not _DEV_DRAW or not simulation.turn_counter.value:
                return
            plt.scatter(beam1._dt[-100:], beam1._dE[-100:], marker="x", c="C1")
            plt.scatter(
                beam1._dt[:100],
                beam1._dE[:100],
                marker="x",
                c="C1",
                label="blond3",
            )
            plt.draw()
            plt.pause(0.1)
            plt.cla()

        sim.run_simulation(
            beams=(beam1,),
            n_turns=N_TURNS,
            observe=[cavity_obs, bunch_observation],
            callbacks=_blond3_callback if _DEV_DRAW else None,
        )

        # observation buffers may live on the GPU (Cupy64Bit); pull to host
        # before comparing against the CPU-only BLonD 2 reference.
        time_history_blond3 = copy_to_cpu(bunch_observation.dts)[:, 0]
        energy_history_blond3 = copy_to_cpu(bunch_observation.dEs)[:, 0]

        # ── BLonD 2 ──────────────────────────────────────────────────────
        from blond.legacy.blond2.beam.beam import Beam, Proton
        from blond.legacy.blond2.input_parameters.rf_parameters import (
            RFStation,
        )
        from blond.legacy.blond2.input_parameters.ring import Ring
        from blond.legacy.blond2.trackers.tracker import (
            FullRingAndRF,
            RingAndRFTracker,
        )
        from blond.legacy.blond2.utils import bmath

        # Force BLonD 2 onto its fastest available CPU backend (cpp > numba >
        # python). The legacy ``bm`` singleton is a mutable global; without
        # this the per-turn reference loop could inherit the pure-python
        # backend left active by an earlier test and run ~50x slower.
        bmath.use_cpu()

        ring = Ring(
            CIRCUMFERENCE,
            1 / transition_gamma.copy() ** 2,
            momentum.copy(),
            Proton(),
            n_turns=N_TURNS,
        )
        beam2 = Beam(
            ring, N_MACROS, INTENSITY, dt=INITIAL_T.copy(), dE=INITIAL_E.copy()
        )
        rf = RFStation(ring, HARMONIC, VOLTAGE, phi_rf.copy())

        full_tracker = FullRingAndRF(
            [RingAndRFTracker(rf, beam2, solver="simple")]
        )

        time_history_blond2 = np.empty(N_TURNS + 1)
        energy_history_blond2 = np.empty(N_TURNS + 1)
        time_history_blond2[0] = beam2.dt[0]
        energy_history_blond2[0] = beam2.dE[0]

        for turn in range(N_TURNS):
            full_tracker.track()
            time_history_blond2[turn + 1] = beam2.dt[0]
            energy_history_blond2[turn + 1] = beam2.dE[0]

            if _DEV_DRAW:
                plt.figure(11)
                plt.scatter(beam2.dt[-100:], beam2.dE[-100:], c="C0")
                plt.scatter(beam2.dt[:100], beam2.dE[:100], c="C0")
                plt.draw()
                plt.pause(0.1)
                plt.cla()

        if _DEV_DRAW:
            plt.figure()
            plt.plot(
                time_history_blond3,
                energy_history_blond3,
                "x-",
                label="blond3",
            )
            plt.plot(
                time_history_blond2[1:],
                energy_history_blond2[1:],
                ".-",
                label="blond2",
            )
            plt.legend()
            plt.show()

        # ── Assertions ────────────────────────────────────────────────────
        np.testing.assert_allclose(
            time_history_blond3,
            time_history_blond2[1:],
            rtol=1e-9,
            err_msg="BLonD3 dt history diverges from BLonD2 reference",
        )
        np.testing.assert_allclose(
            energy_history_blond3,
            energy_history_blond2[1:],
            rtol=1e-9,
            err_msg="BLonD3 dE history diverges from BLonD2 reference",
        )


if __name__ == "__main__":
    unittest.main()
