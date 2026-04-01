"""Shows the problem with the resonator."""

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
    backend,
    proton,
)


def main():  # NOQA: PLR0915
    """Shows the problem with the resonator."""
    backend.set_specials("cpp")
    n_particles = int(3e11)
    n_macroparticles = int(5e6)
    bunch_length = 0.83162555241781e-9

    sync_momentum = 6800e9  # [eV]

    # Machine and RF parameters
    radius = 4242.89
    gamma_transition = 55.759505  # 55.76
    C = 2 * np.pi * radius  # [m]

    # Derived parameters
    momentum_compaction = float(1 / gamma_transition**2)

    # Cavities parameters
    n_rf_systems = 2
    harmonic_numbers = 35640.0
    n = 2
    vr = 0.0

    voltage_program = 6e6
    phi_offset = 0

    ring = Ring(circumference=C)
    cycle = ConstantMagneticCycle(
        reference_particle=proton, value=sync_momentum, in_unit="momentum"
    )
    rf_station = MultiHarmonicRFStation(
        n_harmonics=n_rf_systems,
        main_harmonic_idx=0,
        harmonic=np.array([harmonic_numbers, n * harmonic_numbers]),
        voltage=np.array([voltage_program, vr * voltage_program]),
        phi_rf=np.array([phi_offset, phi_offset]),
    )
    drift = DriftSimple(
        orbit_length=ring.circumference,
        momentum_compaction_factor=momentum_compaction,
    )

    t_rev = cycle.get_t_rev_init(circumference=ring.circumference)
    f_rev = 1 / t_rev
    t_rf = float(t_rev / rf_station.get_main_harmonic())
    bucket_length = t_rf

    # DEFINE BEAM------------------------------------------------------------------
    # beam = Beam(ring, n_macroparticles, n_particles)
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

    Z_over_n = 0.7  # Ohm

    frequency_R = 20.0 / t_rf
    Q = 1.0
    R_S = Z_over_n * Q * frequency_R / f_rev

    Zres_low = Resonators(R_S, frequency_R, Q)
    Zres_high = Resonators(R_S, frequency_R, Q)
    Zres_high.supersampling = 1000

    wake1 = WakeField(
        sources=(Zres_low,), solver=PeriodicFreqSolver(), profile=beam_profile
    )
    wake1.track_profile = True
    wake2 = WakeField(
        sources=(Zres_low,), solver=TimeDomainFftSolver(), profile=beam_profile
    )
    wake2.track_profile = True
    wake3 = WakeField(
        sources=(Zres_high,),
        solver=TimeDomainFftSolver(),
        profile=beam_profile,
    )
    wake3.track_profile = True
    ring.add_elements((drift, rf_station, wake1, wake2, wake3))
    simulation = Simulation(ring=ring, magnetic_cycle=cycle)
    simulation.run_simulation(beams=beam, n_turns=1)

    plt.figure("total_induced_voltage")
    plt.subplot(2, 1, 1)
    plt.plot(
        beam_profile.hist_x * 1e9,
        beam_profile.hist_y,
        "b",
    )
    plt.subplot(2, 1, 2)
    plt.plot(
        beam_profile.hist_x * 1e9,
        wake1.induced_voltage,
        "b",
        label="induced_voltage_freq",
    )
    plt.plot(
        beam_profile.hist_x * 1e9,
        wake2.induced_voltage,
        "r",
        label="induced_voltage_time (vanilla)",
    )
    plt.plot(
        beam_profile.hist_x * 1e9,
        wake3.induced_voltage,
        "g",
        label="induced_voltage_time (supersampling)",
    )
    plt.xlabel("Time (ns)")
    plt.ylabel("Induced voltage (V)")
    plt.legend()

    plt.figure("impedance real part")
    # plt.plot(
    #    np.real(ind_volt_freq.total_impedance), "b", label="induced_voltage_freq"
    # )
    # plt.plot(
    #    np.real(ind_volt_time.total_impedance), "r", label="induced_voltage_time"
    # )
    plt.xlabel("Maybe frequency (arb. units)")
    plt.ylabel("BLonD2 impedance (arb. units)")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
