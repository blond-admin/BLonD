"""BLonD2 vs BLonD3 regression: RF + drift + induced voltage.

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
def test_induced_voltage():
    phi_rf = np.genfromtxt(_RESOURCES / "phase.txt", delimiter=",")
    transition_gamma = np.genfromtxt(_RESOURCES / "gamma.txt", delimiter=",")
    momentum = np.genfromtxt(_RESOURCES / "momentum.txt", delimiter=",")

    from blond import momentum_compaction_factor

    N_TURNS = len(momentum) - 1
    SIM_TURNS = N_TURNS
    CIRCUMFERENCE = 2 * np.pi * 100
    VOLTAGE = 200e3
    HARMONIC = 8
    INTENSITY = 600e10

    Q = 3
    F_RES = 4e6
    R_SH = 10000

    PROFILE_LENGTH = 2.124873604201372e-06
    n_macro = int(1e4)
    N_BINS = 1000

    rnd = np.random.default_rng(seed=42)
    distr = rnd.standard_normal((n_macro, 2))
    INITIAL_E = distr[:, 1].flatten() * 25e6
    INITIAL_T = distr[:, 0].flatten() * 1e-8 + 0.35e-6
    N_MACROS = len(INITIAL_E)

    # ── BLonD 3 ──────────────────────────────────────────────────────────────
    from blond import Numpy64Bit, backend

    backend.change_backend(Numpy64Bit)
    backend.set_specials("cpp")

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
    from blond.handle_results.observables import (
        StaticProfileObservation,
        WakeFieldObservation,
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
    rf_station1.schedule("phi_rf_design", phi_rf[:-1].copy())

    drift1 = DriftSimple(orbit_length=CIRCUMFERENCE)
    drift1.schedule(
        "momentum_compaction_factor",
        momentum_compaction_factor(transition_gamma[1:].copy()),
    )

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
        (rf_station1, drift1, wakefield), reorder=False, section_index=0
    )
    sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    beam1.setup_beam(
        dt=INITIAL_T.copy(),
        dE=INITIAL_E.copy(),
        reference_total_energy=magnetic_cycle.get_total_energy_init(
            particle_type=beam1.particle_type
        ),
    )
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

    def _blond3_callback(simulation: Simulation, beam: Beam):
        if not _DEV_DRAW:
            return
        wf = simulation.ring.elements.get_element(WakeField)
        plt.figure(123)
        plt.twinx()
        plt.plot(
            wf.induced_voltage,
            label=f"BLonD3 {simulation.turn_counter .value=}",
        )

    sim.run_simulation(
        beams=(beam1,),
        n_turns=SIM_TURNS,
        observe=(
            rf_station_obs,
            bunch_observation,
            static_profile_observation,
            wakefield_observation,
        ),
        callbacks=_blond3_callback,
    )

    # ── BLonD 2 ──────────────────────────────────────────────────────────────
    from blond.legacy.blond2.beam.beam import Beam, Proton
    from blond.legacy.blond2.beam.profile import CutOptions, Profile
    from blond.legacy.blond2.impedances.impedance import (
        InducedVoltageFreq,
        TotalInducedVoltage,
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
    profile2 = Profile(beam2, cut_options)
    induced_voltage = InducedVoltageFreq(beam2, profile2, [resonator])
    total_induced_voltage = TotalInducedVoltage(
        beam2, profile2, [induced_voltage]
    )
    full_tracker = FullRingAndRF(
        [RingAndRFTracker(rf, beam2, solver="simple")]
    )

    time_history_blond2 = np.empty(N_TURNS + 1)
    energy_history_blond2 = np.empty(N_TURNS + 1)
    profile_history_blond2 = np.empty((N_BINS, N_TURNS + 1))
    induced_history_blond2 = np.empty((N_BINS, N_TURNS + 1))
    profile_history_blond2[:, 0] = profile2.n_macroparticles
    induced_history_blond2[:, 0] = total_induced_voltage.induced_voltage
    time_history_blond2[0] = beam2.dt[0]
    energy_history_blond2[0] = beam2.dE[0]

    for turn in range(SIM_TURNS):
        full_tracker.track()
        profile2.track()
        total_induced_voltage.track()
        time_history_blond2[turn + 1] = beam2.dt[0]
        energy_history_blond2[turn + 1] = beam2.dE[0]
        profile_history_blond2[:, turn + 1] = profile2.n_macroparticles
        induced_history_blond2[:, turn + 1] = (
            total_induced_voltage.induced_voltage
        )

    if _DEV_DRAW:
        plt.figure()
        plt.plot(
            bunch_observation.dts[:, 0],
            bunch_observation.dEs[:, 0],
            "x-",
            label="blond3",
        )
        plt.plot(
            time_history_blond2[:SIM_TURNS],
            energy_history_blond2[:SIM_TURNS],
            ".-",
            label="blond2",
        )
        plt.legend()
        plt.show()

    # ── Assertions ───────────────────────────────────────────────────────────
    np.testing.assert_allclose(
        bunch_observation.dts[:, 0],
        time_history_blond2[1:],
        rtol=1e-9,
        err_msg="BLonD3 dt history diverges from BLonD2 reference",
    )
    np.testing.assert_allclose(
        bunch_observation.dEs[:, 0],
        energy_history_blond2[1:],
        rtol=1e-9,
        err_msg="BLonD3 dE history diverges from BLonD2 reference",
    )
    np.testing.assert_allclose(
        static_profile_observation.hist_y,
        profile_history_blond2[:, 1:].T,
        rtol=1e-10,
        err_msg="BLonD3 profile history diverges from BLonD2 reference",
    )
    np.testing.assert_allclose(
        wakefield_observation.induced_voltage,
        induced_history_blond2[:, 1:].T,
        rtol=1e-5,
        err_msg="BLonD3 induced voltage history diverges from BLonD2 reference",
    )
