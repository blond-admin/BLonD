"""Integration test of BLonD 2 vs 3 only with RF + drift.

Notes
-----
Authors:
Oliver Muller Smedt
Simon Lauber
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from blond import momentum_compaction_factor


# noqa
def main():  # noqa
    blond2_anim = False
    blond3_anim = False
    profile_blond2 = False
    profile_blond3 = False
    phi_rf = np.genfromtxt("phase.txt", delimiter=",")
    transition_gamma = np.genfromtxt("gamma.txt", delimiter=",")
    momentum_compaction_factor_ = momentum_compaction_factor(
        transition_gamma=transition_gamma
    )
    momentum = np.genfromtxt("momentum.txt", delimiter=",")

    N_TURNS = len(momentum) - 1
    N_TURNS_SIM = N_TURNS
    CIRCUMFERENCE = 2 * np.pi * 100
    VOLTAGE = 200e3
    HARMONIC = 8
    INTENSITY = 1
    test = "accuracy"
    if test == "performance":
        n_macro = int(1e6)
        rnd = np.random.default_rng()
        distr = rnd.standard_normal((n_macro, 2))
        INITIAL_E = distr[:, 1].flatten() * 25e6
        INITIAL_T = distr[:, 0].flatten() * 1e-8 + 0.35e-6
    elif test == "accuracy":
        INITIAL_E = np.array([25e6])
        INITIAL_T = np.array([0.4e-6])
    else:
        raise Exception()
    N_MACROS = len(INITIAL_E)

    ################################## BLOND 3 Implementation ##############################################################
    from blond import (
        Numpy64Bit,
        backend,
    )

    backend.change_backend(Numpy64Bit)
    backend.set_specials("numba")

    from blond import MultiHarmonicRFStation  # NOQA
    from blond import (  # NOQA
        Beam,
        BeamObservationOncePerTurn,
        DriftSimple,
        MagneticCyclePerTurn,
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
    rf_station1.schedule(
        "phi_rf",
        phi_rf[:-1].copy()[:, np.newaxis],
    )

    drift1 = DriftSimple(
        orbit_length=CIRCUMFERENCE,
    )
    drift1.schedule(
        "momentum_compaction_factor",
        momentum_compaction_factor_[1:].copy(),
    )

    beam1 = Beam(intensity=INTENSITY, particle_type=proton)
    beam1.setup_beam(
        dt=INITIAL_T.copy(),
        dE=INITIAL_E.copy(),
    )
    ring.add_elements((rf_station1, drift1), reorder=False, section_index=0)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    sim.print_one_turn_execution_order()

    bunch_observation = BeamObservationOncePerTurn(each_turn_i=1)
    cavity_obs = RFStationPhaseObservation(
        each_turn_i=1, rf_station=rf_station1
    )
    # sim.profiling(0, 0, N_TURNS )

    def my_callback(simulation: Simulation):
        if simulation.turn_i.value:
            plt.scatter(
                beam1._dt[-100:],
                beam1._dE[-100:],
                marker="x",
                c="C1",
            )
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

    import cProfile
    import io
    import pstats
    from pstats import SortKey

    if profile_blond3:
        print("-" * 79)
        pr = cProfile.Profile()
        pr.enable()
    t0 = time.time()
    sim.run_simulation(
        beams=(beam1,),
        n_turns=N_TURNS_SIM,
        observe=[cavity_obs, bunch_observation] if test == "accuracy" else (),
        callbacks=my_callback if blond3_anim else None,
    )

    print(f"runtime blond3 {time.time() - t0}")
    if profile_blond3:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    if test == "accuracy":
        time_history_blond3 = bunch_observation.dts[:, 0]
        energy_history_blond3 = bunch_observation.dEs[:, 0]

    ################################## BLOND 2 Implementation ##############################################################

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

    rf_section_trackers = []
    rf_section_trackers.append(RingAndRFTracker(rf, beam2, solver="simple"))
    # TODO EXACT
    full_tracker = FullRingAndRF(rf_section_trackers)

    time_history_blond2 = np.empty(N_TURNS + 1)
    energy_history_blond2 = np.empty(N_TURNS + 1)
    tot_energy_history_blond2 = np.empty(N_TURNS + 1)
    phase_history_blond2 = np.empty(N_TURNS + 1)
    omega_history_blond2 = np.empty(N_TURNS + 1)
    voltage_history_blond2 = np.empty(N_TURNS + 1)

    time_history_blond2[0] = beam2.dt[0]
    energy_history_blond2[0] = beam2.dE[0]
    tot_energy_history_blond2[0] = beam2.energy
    phase_history_blond2[0] = rf_section_trackers[0].rf_params.phi_rf[0, 0]
    omega_history_blond2[0] = rf_section_trackers[0].rf_params.omega_rf[0, 0]
    voltage_history_blond2[0] = rf_section_trackers[0].rf_params.voltage[0, 0]

    t0 = time.time()
    if profile_blond2:
        import cProfile
        import io
        import pstats
        from pstats import SortKey

        print("-" * 79)
        pr = cProfile.Profile()
        pr.enable()
    print("BLOND2")
    for turn in tqdm(range(N_TURNS_SIM)):
        full_tracker.track()

        if test == "accuracy":
            time_history_blond2[turn + 1] = beam2.dt[0]
            energy_history_blond2[turn + 1] = beam2.dE[0]
            tot_energy_history_blond2[turn + 1] = beam2.energy
            phase_history_blond2[turn + 1] = rf_section_trackers[
                0
            ].rf_params.phi_rf[0, turn + 1]
            omega_history_blond2[turn + 1] = rf_section_trackers[
                0
            ].rf_params.omega_rf[0, turn + 1]
            voltage_history_blond2[turn + 1] = rf_section_trackers[
                0
            ].rf_params.voltage[0, turn + 1]
        if blond2_anim:
            plt.figure(11)
            plt.scatter(
                beam2.dt[-100:],
                beam2.dE[-100:],
                c="C0",
            )
            plt.scatter(
                beam2.dt[:100],
                beam2.dE[:100],
                c="C0",
            )
            plt.draw()
            plt.pause(0.1)
            plt.cla()
    print(f"runtime blond2 {time.time() - t0}")
    if profile_blond2:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    if test == "performance":
        plt.scatter(
            beam2.dt[-100:],
            beam2.dE[-100:],
            c="C0",
        )
        plt.scatter(
            beam2.dt[:100],
            beam2.dE[:100],
            c="C0",
            label="blond2",
        )
        plt.scatter(
            beam1._dt[-100:],
            beam1._dE[-100:],
            marker="x",
            c="C1",
        )
        plt.scatter(
            beam1._dt[:100],
            beam1._dE[:100],
            marker="x",
            c="C1",
            label="blond3",
        )
        plt.legend()
        plt.show()
        sys.exit()
    else:
        plt.figure()
        plt.plot(
            time_history_blond3, energy_history_blond3, "x-", label="blond3"
        )
        plt.plot(
            time_history_blond2[:N_TURNS_SIM],
            energy_history_blond2[:N_TURNS_SIM],
            ".-",
            label="blond2",
        )
    plt.legend()
    plt.figure()
    plt.plot(
        tot_energy_history_blond2[:N_TURNS_SIM],
        label=str(tot_energy_history_blond2.shape),
    )
    plt.plot(bunch_observation.reference_total_energy)
    plt.legend()

    plt.figure()
    ax = plt.subplot(3, 1, 1)
    plt.plot(phase_history_blond2, ".-", label="blond 2")
    plt.plot(cavity_obs.phases, ".-", label="blond 3")
    plt.subplot(3, 1, 2, sharex=ax)
    plt.plot(omega_history_blond2, ".-", label="blond 2")
    plt.plot(cavity_obs.omegas, ".-", label="blond 3")
    plt.subplot(3, 1, 3, sharex=ax)
    plt.plot(voltage_history_blond2, ".-", label="blond 2")
    plt.plot(cavity_obs.voltages, ".-", label="blond 3")
    plt.legend()
    plt.xlim(0, 10)
    plt.legend()
    plt.show()
    ############################################## See if the two are equivelant ###########################################

    if np.all(time_history_blond3 == time_history_blond2) and np.all(
        energy_history_blond3 == energy_history_blond2
    ):
        print("test passed successfully")
    else:
        print("Test Failed")
        print(
            "final_coordinates BLonD2:   ",
            time_history_blond2[-1],
            "   ",
            energy_history_blond2[-1],
        )
        print(
            "final_coordinates BLonD3:   ",
            time_history_blond3[-1],
            "   ",
            energy_history_blond3[-1],
        )


if __name__ == "__main__":
    main()
