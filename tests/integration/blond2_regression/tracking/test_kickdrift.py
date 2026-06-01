"""BLonD2 vs BLonD3 regression: RF + drift only (single macroparticle).

Authors: Oliver Muller Smedt, Simon Lauber
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

_DEV_DRAW = os.getenv("DEV_DRAW", "False").lower() == "true"
_RESOURCES = Path(__file__).parent / "resources"


@pytest.mark.integration
def test_kickdrift():
    phi_rf = np.genfromtxt(_RESOURCES / "phase.txt", delimiter=",")
    transition_gamma = np.genfromtxt(_RESOURCES / "gamma.txt", delimiter=",")
    momentum = np.genfromtxt(_RESOURCES / "momentum.txt", delimiter=",")

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

    # ── BLonD 3 ──────────────────────────────────────────────────────────────
    from blond import Numpy64Bit, backend

    backend.change_backend(Numpy64Bit)
    backend.set_specials("numba")

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

    rf_station1 = MultiHarmonicRFStation(n_harmonics=1, main_harmonic_idx=0)
    rf_station1.harmonic = np.array([HARMONIC])
    rf_station1.voltage = np.array([VOLTAGE])
    rf_station1.schedule("phi_rf_design", phi_rf[:-1].copy()[:, np.newaxis])

    drift1 = DriftSimple(orbit_length=CIRCUMFERENCE)
    drift1.schedule(
        "momentum_compaction_factor",
        momentum_compaction_factor_[1:].copy(),
    )

    beam1 = Beam(intensity=INTENSITY, particle_type=proton)
    beam1.setup_beam(dt=INITIAL_T.copy(), dE=INITIAL_E.copy())
    ring.add_elements((rf_station1, drift1), reorder=False, section_index=0)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

    bunch_observation = BeamObservationOncePerTurn(each_turn_i=1)
    cavity_obs = RFStationPhaseObservation(
        each_turn_i=1, rf_station=rf_station1
    )

    def _blond3_callback(simulation: Simulation):
        if not _DEV_DRAW or not simulation.turn_i.value:
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

    time_history_blond3 = bunch_observation.dts[:, 0]
    energy_history_blond3 = bunch_observation.dEs[:, 0]

    # ── BLonD 2 ──────────────────────────────────────────────────────────────
    from blond.legacy.blond2.beam.beam import Beam, Proton
    from blond.legacy.blond2.input_parameters.rf_parameters import RFStation
    from blond.legacy.blond2.input_parameters.ring import Ring
    from blond.legacy.blond2.trackers.tracker import (
        FullRingAndRF,
        RingAndRFTracker,
    )

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
            time_history_blond3, energy_history_blond3, "x-", label="blond3"
        )
        plt.plot(
            time_history_blond2[1:],
            energy_history_blond2[1:],
            ".-",
            label="blond2",
        )
        plt.legend()
        plt.show()

    # ── Assertions ───────────────────────────────────────────────────────────
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
