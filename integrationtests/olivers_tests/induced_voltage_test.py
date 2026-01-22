"""Integration test of BLonD 2 vs 3 with induced voltage.

Notes
-----
Authors:
Oliver Muller Smedt
Simon Lauber
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from blond.handle_results.observables import (
    StaticProfileObservation,
    WakeFieldObservation,
)
from blond.legacy.blond2.impedances.impedance import TotalInducedVoltage


# noqa
def main():  # noqa
    profile_blond2 = False
    phi_rf = np.genfromtxt("phase.txt", delimiter=",")
    transition_gamma = np.genfromtxt("gamma.txt", delimiter=",")
    momentum = np.genfromtxt("momentum.txt", delimiter=",")

    N_TURNS = len(momentum) - 1
    SIM_TURNS = N_TURNS
    CIRCUMFERENCE = 2 * np.pi * 100  # Meters
    VOLTAGE = 200e3  # Volts
    HARMONIC = 8
    INTENSITY = 600e10  # roughly TOF intensity

    # resonator params
    Q = 3
    F_RES = 4e6  # Hz
    R_SH = 10000  # Ohms

    PROFILE_LENGTH = 2.124873604201372e-06

    n_macro = int(1e4)
    rnd = np.random.default_rng()
    distr = rnd.standard_normal((n_macro, 2))
    INITIAL_E = distr[:, 1].flatten() * 25e6
    INITIAL_T = distr[:, 0].flatten() * 1e-8 + 0.35e-6

    N_MACROS = len(INITIAL_E)

    N_BINS = 1000

    ################################## BLOND 3 Implementation ##############################################################
    from blond import Numpy64Bit, backend

    backend.change_backend(Numpy64Bit)
    backend.set_specials("cpp")
    # backend.set_specials("fortran")

    from blond import (
        Beam,
        BeamObservationOncePerTurn,
        DriftSimple,
        MagneticCyclePerTurn,
        RFStationPhaseObservation,
        Ring,
        Simulation,
        SingleHarmonicRFStation,
        StaticProfile,
        WakeField,
        proton,
    )
    from blond.physics.impedances.solvers import PeriodicFreqSolver
    from blond.physics.impedances.sources import Resonators

    ring = Ring(circumference=CIRCUMFERENCE)

    magnetic_cycle = MagneticCyclePerTurn(
        value_init=float(momentum[0]),
        values_after_turn=momentum[1:].copy(),
        reference_particle=proton,
    )

    rf_station1 = SingleHarmonicRFStation()
    rf_station1.harmonic = HARMONIC
    rf_station1.voltage = VOLTAGE
    rf_station1.schedule(
        "phi_rf",
        phi_rf[:-1].copy(),
    )

    drift1 = DriftSimple(orbit_length=CIRCUMFERENCE)
    drift1.schedule("transition_gamma", transition_gamma[1:].copy())

    beam1 = Beam(intensity=INTENSITY, particle_type=proton)

    profile = StaticProfile(0, PROFILE_LENGTH, N_BINS)
    wakefield = WakeField(
        sources=(
            Resonators(np.array([R_SH]), np.array([F_RES]), np.array([Q])),
        ),
        solver=PeriodicFreqSolver(PROFILE_LENGTH, allow_next_fast_len=True),
        profile=profile,
    )

    ring.add_elements(
        (
            rf_station1,
            drift1,
            profile,
            wakefield,
        ),
        reorder=False,
        section_index=0,
    )
    sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    beam1.setup_beam(
        dt=INITIAL_T.copy(),
        dE=INITIAL_E.copy(),
        reference_total_energy=magnetic_cycle.get_total_energy_init(
            particle_type=beam1.particle_type
        ),
    )
    sim.print_one_turn_execution_order()
    profile.track(beam=beam1)

    bunch_observation = BeamObservationOncePerTurn(each_turn_i=1)
    rf_station_obs = RFStationPhaseObservation(
        each_turn_i=1, rf_station=rf_station1
    )
    static_profile_observation = StaticProfileObservation(
        each_turn_i=1, profile=profile
    )
    wakefield_observation = WakeFieldObservation(
        each_turn_i=1, wakefield=wakefield
    )

    def my_callback(simulation: Simulation, beam: Beam):
        return
        wakefield = simulation.ring.elements.get_element(WakeField)
        plt.figure(123)
        plt.twinx()
        plt.plot(
            wakefield.induced_voltage,
            label=f"BLonD3 {simulation.turn_i.value=}",
        )

    """sim.profiling(
        beams=(beam1,),
        turn_i_init=0,
        profile_start_turn_i=0,
        profile_n_turns=N_TURNS,
        sortby=SortKey.TIME,
    )"""

    try:
        sim.load_results(
            n_turns=N_TURNS,
            observe=[
                rf_station_obs,
                bunch_observation,
                static_profile_observation,
                wakefield_observation,
            ],
        )
    except FileNotFoundError:
        t0 = time.time()

        sim.run_simulation(
            beams=(beam1,),
            n_turns=SIM_TURNS,
            observe=[
                rf_station_obs,
                bunch_observation,
                static_profile_observation,
                wakefield_observation,
            ],
            callbacks=my_callback,
        )
        t1 = time.time()
        print(f"{t1 - t0}s")

    ################################## BLOND 2 Implementation ##############################################################

    from blond.legacy.blond2.beam.beam import Beam, Proton
    from blond.legacy.blond2.beam.profile import CutOptions, Profile
    from blond.legacy.blond2.impedances.impedance import (
        InducedVoltageFreq,
    )
    from blond.legacy.blond2.impedances.impedance_sources import Resonators
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

    resonator = Resonators(R_SH, F_RES, Q)
    cut_options = CutOptions(0, PROFILE_LENGTH, N_BINS)
    profile = Profile(beam2, cut_options)
    induced_voltage = InducedVoltageFreq(beam2, profile, [resonator])
    total_induced_voltage = TotalInducedVoltage(
        beam2, profile, [induced_voltage]
    )

    rf_section_trackers = []
    rf_section_trackers.append(RingAndRFTracker(rf, beam2, solver="simple"))
    # TODO EXACT
    full_tracker = FullRingAndRF(rf_section_trackers)

    time_history_blond2 = np.empty(N_TURNS + 1)
    energy_history_blond2 = np.empty(N_TURNS + 1)
    profile_history_blond2 = np.empty((N_BINS, N_TURNS + 1))
    induced_history_blond2 = np.empty((N_BINS, N_TURNS + 1))
    profile_history_blond2[:, 0] = profile.n_macroparticles
    induced_history_blond2[:, 0] = total_induced_voltage.induced_voltage
    time_history_blond2[0] = beam2.dt[0]
    energy_history_blond2[0] = beam2.dE[0]

    if profile_blond2:
        import cProfile
        import io
        import pstats
        from pstats import SortKey

        print("-" * 79)
        pr = cProfile.Profile()
        pr.enable()
    t0 = time.time()
    for turn in tqdm(range(SIM_TURNS)):
        full_tracker.track()
        profile.track()
        total_induced_voltage.track()
        if False:
            plt.figure(123)
            plt.plot(
                total_induced_voltage.induced_voltage,
                "--",
                label=f"BLonD2 {turn=}",
            )
        time_history_blond2[turn + 1] = beam2.dt[0]
        energy_history_blond2[turn + 1] = beam2.dE[0]
        profile_history_blond2[:, turn + 1] = profile.n_macroparticles
        induced_history_blond2[:, turn + 1] = (
            total_induced_voltage.induced_voltage
        )
    t1 = time.time()
    print(f"{t1 - t0}s")
    if profile_blond2:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    time_history_blond3 = bunch_observation.dts[:, 0]
    energy_history_blond3 = bunch_observation.dEs[:, 0]
    if False:
        plt.legend()
        plt.show()
    plt.figure()
    plt.subplot(5, 1, 1)
    plt.title("C-time")
    plt.plot(bunch_observation.reference_time[1:], ".-", label="BLonD3")
    plt.plot(ring.cycle_time[:SIM_TURNS], "--", label="BLonD2")
    plt.legend()
    plt.subplot(5, 1, 2)
    plt.title("E tot")
    plt.plot(bunch_observation.reference_total_energy[:], ".-")
    plt.plot(ring.energy[0, :SIM_TURNS], "--")
    plt.figure()
    print(static_profile_observation.hist_y[turn, :].shape)
    print(profile_history_blond2[:, turn].shape)
    """for turn in tqdm(range(SIM_TURNS + 1)):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(static_profile_observation.hist_y[turn, :], "o-", label="BLonD3")
        plt.plot(profile_history_blond2[:, turn], "x-", label="BLonD2")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(wakefield_observation.induced_voltage[turn, :], "o-", label="BLonD3")
        plt.plot(induced_history_blond2[:, turn], "x-", label="BLonD2")
        plt.legend()"""

    plt.figure()
    ############################################## See if the two are equivelant ###########################################
    plt.figure(999)
    plt.plot(time_history_blond3, energy_history_blond3, "x-", label="blond3")
    plt.plot(
        time_history_blond2[:SIM_TURNS],
        energy_history_blond2[:SIM_TURNS],
        ".-",
        label="blond2",
    )
    plt.legend()
    plt.show()

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
        print(
            "initial_coordinates BLonD2:   ",
            time_history_blond2[0],
            "   ",
            energy_history_blond2[0],
        )
        print(
            "initial_coordinates BLonD3:   ",
            time_history_blond3[0],
            "   ",
            energy_history_blond3[0],
        )


if __name__ == "__main__":
    main()
