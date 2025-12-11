"""Simplified simulation of bunch-to-bucket transfer into the LHC."""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRfStation,
    Ring,
    Simulation,
    StaticProfile,
    backend,
    proton,
)
from blond.core.backends.backend import Numpy64Bit
from blond.experimental.physics.feedbacks.accelerators.lhc.beam_feedback import (
    LHCBeamControl,
)
from blond.experimental.physics.feedbacks.accelerators.lhc.cavity_feedback import (
    LHCCavityLoop,
    LHCCavityLoopCommissioning,
)


def lhc_flatbottom_settings():
    """Settings for LHC flat bottom."""
    # LHC parameters
    circumference = 26658.8832  # [m]
    energy = 450e9
    n_bunches = 36
    intensity = 2.3e11 * n_bunches
    n_turns = 2000
    voltage = 5e6
    h = 35640
    gamma_t = 53.8
    delta_f = -3480

    rel_gamma = energy / proton.mass
    rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

    return (
        circumference,
        energy,
        n_bunches,
        intensity,
        n_turns,
        voltage,
        h,
        gamma_t,
        delta_f,
        rel_beta,
    )


def main():
    """Runs LHC simulation.

    Will be compared to BLonD2 equivalent simulation.
    """
    # Use 64 bit numpy and CPP backend
    backend.change_backend(Numpy64Bit)
    backend.set_specials("cpp")

    fig_scale = 0.75

    # LHC parameters
    (
        circumference,
        energy,
        n_bunches,
        intensity,
        n_turns,
        voltage,
        h,
        gamma_t,
        delta_f,
        rel_beta,
    ) = lhc_flatbottom_settings()

    # Beam object
    beam = Beam(intensity, proton)
    cycle = ConstantMagneticCycle(proton, energy, in_unit="total energy")
    lattice = DriftSimple(orbit_length=circumference, transition_gamma=gamma_t)

    # RF station
    cavity = MultiHarmonicRfStation(
        voltage=np.array([voltage]),
        phi_rf=np.array([0.0]),
        harmonic=np.array([h]),
        n_harmonics=1,
        main_harmonic_idx=0,
    )

    f_rf = cavity.get_main_harmonic_omega_rf_design(
        rel_beta, lattice.orbit_length
    ) / (2 * np.pi)
    f_rev = f_rf / h
    t_rf = 1 / f_rf
    t_rev = 1 / f_rev

    profile = StaticProfile(
        cut_left=0,
        cut_right=(240 + n_bunches * 10) / f_rf,
        n_bins=2**6 * (240 + n_bunches * 10),
    )

    # LHC cavity feedback
    commissioning = LHCCavityLoopCommissioning(
        G_a=6.79e-6, G_d=10, G_o=10, tau_a=170e-6, tau_d=400e-6, tau_o=110e-6
    )
    cavity_control = LHCCavityLoop(
        profile=profile,
        tau_otfb=1.2e-6,
        f_c=f_rf + delta_f,
        RFFB=commissioning,
    )
    cavity.attach_cavity_feedback(
        cavity_control,
    )

    # LHC beam feedback
    beam_control = LHCBeamControl(
        profile,
        pl_gain=1 / (5 * t_rev) * 1,
        sl_gain=1 / (5 * t_rev) / 10 * 1,
        current_thres=0.5,
    )
    cavity.attach_beam_feedback(beam_control)

    bigaussian = BiGaussian(1_000_000, sigma_dt=1.2e-9 / 4)
    ring = Ring(circumference)
    ring.add_elements(
        [profile, lattice, beam_control, cavity],
    )

    simulation = Simulation(ring, cycle)
    simulation.prepare_beam(beam, bigaussian)

    _dt_tmp = beam._dt
    _dE_tmp = beam._dE
    _flags_tmp = beam._flags
    _ids_tmp = beam._ids

    for i in range(1, n_bunches):
        beam._dt = np.append(beam._dt, _dt_tmp + 10 * t_rf * i)
        beam._dE = np.append(beam._dE, _dE_tmp)
        beam._flags = np.append(beam._flags, _flags_tmp)
        beam._ids = np.append(beam._ids, _ids_tmp)

    beam._dt += 100 * t_rf + 40 / 360 * t_rf

    i_beam = np.zeros((n_turns, h // 10), dtype=complex)
    rf_power = np.zeros((n_turns, h // 10), dtype=complex)

    simulation.finalize(
        (beam,),
        n_turns,
    )

    for i in tqdm(range(n_turns)):
        simulation.turn_i.value = i

        for element in ring.elements.elements:
            element.track(beam)

        i_beam[i, :] = cavity_control.I_BEAM_COARSE[-h // 10 :]
        rf_power[i, :] = cavity_control.generator_power()[-h // 10 :]

    fig, ax = plt.subplots(figsize=(10 * fig_scale, 6 * fig_scale))
    ax.plot(np.abs(i_beam).T)

    fig, ax = plt.subplots(figsize=(10 * fig_scale, 6 * fig_scale))
    ax.plot(np.abs(rf_power).T)

    fig, ax = plt.subplots(figsize=(10 * fig_scale, 6 * fig_scale))
    ax.plot(np.max(np.abs(rf_power), axis=1) / 1e3)

    plt.show()


if __name__ == "__main__":
    main()
