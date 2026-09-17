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
import unittest

import numpy as np
import pytest
from matplotlib import pyplot as plt

from blond import (
    AllowPlotting,
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRFStation,
    PeriodicFreqSolver,
    Resonators,
    Ring,
    Simulation,
    StaticProfile,
    WakeField,
    copy_to_cpu,
    proton,
)
from blond.physics.impedances.solvers import (
    MultiPassResonatorSolver,
    MultiPoleSparseSolve,
    SingleTurnResonatorConvolutionSolver,
    TimeDomainFftSolver,
)

_DEV_DRAW = os.getenv("DEV_DRAW", "False").lower() == "true"

# Every time-domain resonator solver must reproduce the frequency-domain
# reference on this deliberately under-resolved grid (see module docstring).
_TIME_SOLVER_FACTORIES = (
    TimeDomainFftSolver,
    SingleTurnResonatorConvolutionSolver,
    MultiPoleSparseSolve,
    MultiPassResonatorSolver,
)


@pytest.mark.integration
class TestLowQResonatorEndToEnd(unittest.TestCase):
    """Every time-domain solver matches the freq solver end to end."""

    def test_low_q_resonator_time_matches_freq_end_to_end(  # NOQA: PLR0915
        self,
    ):
        """Every time-domain solver agrees with the freq solver (low-Q).

        All solvers see the same tracked profile, so the only
        difference is the wake representation. With bin-integration
        each time-domain solver's relative deviation from the
        frequency-domain reference stays well below the
        point-sampling regime (~2.3).
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
            seed=0,
        )

        number_of_slices = 256
        beam_profile = StaticProfile(
            cut_left=0,
            cut_right=2 * bucket_length,
            n_bins=number_of_slices,
        )

        # Broadband, low-Q resonator at 20x the RF harmonic
        # (Z/n = 0.7 Ohm, Q = 1)
        z_over_n = 0.7  # Ohm
        frequency_res = 20.0 / t_rf
        quality_factor = 1.0
        shunt_impedance = z_over_n * quality_factor * frequency_res / f_rev

        def make_resonators():
            return Resonators(shunt_impedance, frequency_res, quality_factor)

        wake_freq = WakeField(
            sources=(make_resonators(),),
            solver=PeriodicFreqSolver(),
            profile=beam_profile,
        )
        wake_freq.track_profile = True

        wakes_time = {}
        for make_solver in _TIME_SOLVER_FACTORIES:
            wake_time = WakeField(
                sources=(make_resonators(),),
                solver=make_solver(),
                profile=beam_profile,
            )
            wake_time.track_profile = True
            wakes_time[make_solver.__name__] = wake_time

        ring.add_elements((drift, rf_station, wake_freq, *wakes_time.values()))
        simulation = Simulation(ring=ring, magnetic_cycle=cycle)
        simulation.run_simulation(beams=beam, n_turns=1)

        induced_voltage_freq = np.asarray(
            copy_to_cpu(wake_freq.induced_voltage)
        )
        induced_voltage_time = {
            name: np.asarray(copy_to_cpu(wake.induced_voltage))
            for name, wake in wakes_time.items()
        }

        peak_freq = np.max(np.abs(induced_voltage_freq))
        max_rel_dev = {
            name: np.max(np.abs(voltage - induced_voltage_freq)) / peak_freq
            for name, voltage in induced_voltage_time.items()
        }

        if _DEV_DRAW:
            with AllowPlotting():
                plt.figure("total_induced_voltage")
                plt.subplot(2, 1, 1)
                plt.plot(beam_profile.hist_x * 1e9, beam_profile.hist_y, "b")
                plt.ylabel("Profile")
                plt.subplot(2, 1, 2)
                plt.plot(
                    beam_profile.hist_x * 1e9,
                    induced_voltage_freq,
                    "k",
                    lw=2,
                    label="induced_voltage_freq (ref)",
                )
                for name, voltage in induced_voltage_time.items():
                    plt.plot(beam_profile.hist_x * 1e9, voltage, label=name)
                plt.xlabel("Time (ns)")
                plt.ylabel("Induced voltage (V)")
                plt.legend()
                plt.show()

        # Point-sampling the wake gave ~2.3 here; bin-integration keeps every
        # time-domain solver close to the freq reference on this deliberately
        # under-resolved grid.
        for name, dev in max_rel_dev.items():
            with self.subTest(name=name):
                self.assertLess(dev, 0.15, msg=f"{name}: {dev}")


if __name__ == "__main__":
    unittest.main()
