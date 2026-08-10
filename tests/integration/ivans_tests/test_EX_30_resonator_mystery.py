"""Low-Q broadband resonator: time- and frequency-domain solvers agree.

End-to-end regression for the ``InducedVoltageTime`` vs
``InducedVoltageFreq`` discrepancy, driven through a full
``Simulation.run_simulation`` rather than a bare ``calc_induced_voltage``
call (cf. the unit-level ``test_low_q_resonator_time_matches_freq``).

The resonator sits at 20x the RF harmonic, so its wake oscillates several
times within a handful of profile bins. Point-sampling the wake aliased
badly (~230 % error) and silently produced a wrong induced voltage;
bin-integrating the wake (exact for a histogram beam) makes the two solvers
overlap.

Set ``DEV_DRAW=true`` in the environment to plot the profile and both
induced-voltage curves for visual inspection.
"""

import os

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRFStation,
    PeriodicFreqSolver,
    Resonators,
    Ring,
    Simulation,
    StaticProfile,
    TimeDomainFftSolver,
    WakeField,
    copy_to_cpu,
    proton,
)

_DEV_DRAW = os.getenv("DEV_DRAW", "False").lower() == "true"


def test_low_q_resonator_time_matches_freq_end_to_end():  # NOQA: PLR0915
    """Time- and frequency-domain solvers agree for a low-Q resonator.

    Both solvers see the same tracked profile, so the only difference is the
    wake representation. With bin-integration the relative deviation stays
    well below the point-sampling regime (~2.3).
    """
    n_particles = int(3e11)
    n_macroparticles = int(5e5)
    bunch_length = 0.83162555241781e-9

    sync_momentum = 6800e9  # [eV]

    # Machine and RF parameters
    radius = 4242.89
    gamma_transition = 55.759505
    circumference = 2 * np.pi * radius  # [m]
    momentum_compaction = float(1 / gamma_transition**2)

    # Cavity parameters
    n_rf_systems = 2
    main_harmonic = 35640.0
    voltage_program = 6e6

    ring = Ring(circumference=circumference)
    cycle = ConstantMagneticCycle(
        reference_particle=proton, value=sync_momentum, in_unit="momentum"
    )
    rf_station = MultiHarmonicRFStation(
        n_harmonics=n_rf_systems,
        main_harmonic_idx=0,
        harmonic=np.array([main_harmonic, 2 * main_harmonic]),
        voltage=np.array([voltage_program, 0.0]),
        phi_rf=np.array([0.0, 0.0]),
    )
    drift = DriftSimple(
        orbit_length=ring.circumference,
        momentum_compaction_factor=momentum_compaction,
    )

    t_rev = cycle.get_t_rev_init(circumference=ring.circumference)
    f_rev = 1 / t_rev
    t_rf = float(t_rev / rf_station.get_main_harmonic())
    bucket_length = t_rf

    beam = Beam.simple_gaussian(
        n_macroparticles=n_macroparticles,
        intensity=n_particles,
        particle_type=proton,
        dE_scale=1,
        dt_scale=bunch_length / 2,
        dt_offset=2.5e-9,
    )

    number_of_slices = 256
    beam_profile = StaticProfile(
        cut_left=0,
        cut_right=2 * bucket_length,
        n_bins=number_of_slices,
    )

    # Broadband, low-Q resonator at 20x the RF harmonic (Z/n = 0.7 Ohm, Q = 1)
    z_over_n = 0.7  # Ohm
    frequency_res = 20.0 / t_rf
    quality_factor = 1.0
    shunt_impedance = z_over_n * quality_factor * frequency_res / f_rev

    resonator_freq = Resonators(shunt_impedance, frequency_res, quality_factor)
    resonator_time = Resonators(shunt_impedance, frequency_res, quality_factor)

    wake_freq = WakeField(
        sources=(resonator_freq,),
        solver=PeriodicFreqSolver(),
        profile=beam_profile,
    )
    wake_freq.track_profile = True
    wake_time = WakeField(
        sources=(resonator_time,),
        solver=TimeDomainFftSolver(),
        profile=beam_profile,
    )
    wake_time.track_profile = True

    ring.add_elements((drift, rf_station, wake_freq, wake_time))
    simulation = Simulation(ring=ring, magnetic_cycle=cycle)
    simulation.run_simulation(beams=beam, n_turns=1)

    induced_voltage_freq = np.asarray(copy_to_cpu(wake_freq.induced_voltage))
    induced_voltage_time = np.asarray(copy_to_cpu(wake_time.induced_voltage))

    max_rel_dev = np.max(
        np.abs(induced_voltage_time - induced_voltage_freq)
    ) / np.max(np.abs(induced_voltage_freq))

    if _DEV_DRAW:
        plt.figure("total_induced_voltage")
        plt.subplot(2, 1, 1)
        plt.plot(beam_profile.hist_x * 1e9, beam_profile.hist_y, "b")
        plt.ylabel("Profile")
        plt.subplot(2, 1, 2)
        plt.plot(
            beam_profile.hist_x * 1e9,
            induced_voltage_freq,
            "b",
            label="induced_voltage_freq",
        )
        plt.plot(
            beam_profile.hist_x * 1e9,
            induced_voltage_time,
            "g",
            label="induced_voltage_time (bin-integrated)",
        )
        plt.xlabel("Time (ns)")
        plt.ylabel("Induced voltage (V)")
        plt.legend()
        plt.show()

    # Point-sampling the wake gave ~2.3 here; bin-integration keeps the two
    # solvers close on this deliberately under-resolved grid.
    assert max_rel_dev < 0.15, max_rel_dev
