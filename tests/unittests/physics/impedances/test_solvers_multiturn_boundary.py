"""Turn-boundary tests for `MultiPoleSparseSolve`.

A profile that covers the whole revolution period leaves no room between the
turns: the first bin of turn ``n + 1`` follows the last bin of turn ``n`` by
exactly one bin width. The solver's recursion reads a state that lags by two
bins, so the state handed from one call to the next must be referenced two
bins back -- not one -- and the charge of the previous call's trailing bin
must reach the new call's leading bin through the near-lag terms.

These tests cover that from three sides: a full-turn profile has to run at
all, its wake has to wrap around the turn boundary with the amplitude a
continuous multi-turn convolution gives, and the kernel alone has to give the
same answer whether a profile is fed to it in one call or two.
"""

import unittest

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
    make_multibunch_beam,
    momentum_compaction_factor,
    proton,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.solvers import (
    ContinuousMultiTurnTimeDomainSolver,
    MultiPoleSparseSolve,
)

# LHC-like machine
CIRCUMFERENCE = 26658.883
TRANSITION_GAMMA = 55.759505
HARMONIC = 35640
RF_VOLTAGE = 6e6  # [V]
SYNC_MOMENTUM = 450e9  # [eV/c]

SIGMA_DT = 2e-10  # [s]
N_MACROPARTICLES = 20000
SEED = 42

DEV_DRAW = False


def _make_simulation(solver, sources, n_bins):
    """Assemble an LHC-like ring whose profile spans the whole turn."""
    ring = Ring(circumference=CIRCUMFERENCE)
    magnetic_cycle = ConstantMagneticCycle(
        reference_particle=proton, value=SYNC_MOMENTUM, in_unit="momentum"
    )
    t_rev = magnetic_cycle.get_t_rev_init(
        ring.circumference, particle_type=proton
    )
    profile = StaticProfile(cut_left=0.0, cut_right=t_rev, n_bins=n_bins)
    wakefield = WakeField(sources=sources, solver=solver, profile=profile)
    drift = DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=TRANSITION_GAMMA
        ),
        orbit_length=ring.circumference,
    )
    rf_station = SingleHarmonicRFStation(
        harmonic=HARMONIC, voltage=RF_VOLTAGE, phi_rf=0.0
    )
    ring.add_elements((wakefield, drift, rf_station))
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    return simulation, wakefield, drift, rf_station, t_rev


class TestMultiPoleSparseSolveFullTurnProfile(unittest.TestCase):
    """A profile spanning one full revolution must track several turns."""

    def test_full_turn_profile_runs_several_turns(self):
        """No turn-boundary state is left closer than the recursion allows."""
        n_bins = 512
        simulation, wakefield, _, _, _ = _make_simulation(
            solver=MultiPoleSparseSolve(),
            sources=(Resonators(1e5, 1e9, 5.0),),
            n_bins=n_bins,
        )
        beam = Beam(intensity=1e11, particle_type=proton)
        simulation.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=SIGMA_DT,
                n_macroparticles=N_MACROPARTICLES,
                seed=SEED,
            ),
            beam=beam,
        )
        simulation.run_simulation(beams=beam, n_turns=3)

        induced_voltage = copy_to_cpu(wakefield.induced_voltage)
        self.assertEqual(len(induced_voltage), n_bins)
        self.assertTrue(np.all(np.isfinite(induced_voltage)))


class TestMultiPoleSparseSolveTurnBoundaryWake(unittest.TestCase):
    """The wake must cross the turn boundary with the right amplitude.

    A bunch sits in the very last bin of the turn, so its wake reaches the
    first bins of the next turn through the state handed over between calls
    and through the near-lag term the solver adds for the trailing bin. The
    reference is `ContinuousMultiTurnTimeDomainSolver`, which convolves the
    same bin-averaged wake over a continuous two-turn history.
    """

    n_bins = 512
    # Narrow-band enough that the wake still lives a few bins after the
    # bunch, so the turn boundary actually carries signal.
    shunt_impedance = 1e6  # [Ohm]
    center_frequency = 1e9  # [Hz]
    quality_factor = 1e4

    def _run(self, solver, n_turns):
        sources = (
            Resonators(
                self.shunt_impedance,
                self.center_frequency,
                self.quality_factor,
            ),
        )
        simulation, wakefield, drift, rf_station, t_rev = _make_simulation(
            solver=solver, sources=sources, n_bins=self.n_bins
        )
        bin_dt = t_rev / self.n_bins
        bunch = Beam(intensity=1e11, particle_type=proton)
        simulation.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=SIGMA_DT,
                n_macroparticles=N_MACROPARTICLES,
                seed=SEED,
            ),
            beam=bunch,
        )
        # One bunch in the centre of bin 4, one in the centre of the last
        # bin. Bin 0 stays empty, so the (deliberately unhandled) non-causal
        # tap from the next turn's first bin carries no charge and the
        # comparison is not polluted by it.
        beam = make_multibunch_beam(
            beam=bunch,
            n_times=2,
            t_distance=t_rev - 5.0 * bin_dt,
            common_offset=4.5 * bin_dt,
        )

        def _advance_reference_time(simulation, beam):
            # `orbit_length = 0` freezes the beam but also stops
            # reference.time from advancing; this callback restores the
            # correct t_rev increment, which the pole solver needs to place
            # the previous turn's state.
            beam.reference.time += CIRCUMFERENCE / beam.reference.velocity

        # Freeze the beam so every turn sees the identical profile.
        drift.orbit_length = 0.0
        rf_station.voltage = 0.0
        simulation.check_circumference = "ignore"
        simulation.run_simulation(
            beams=beam, n_turns=n_turns, callbacks=_advance_reference_time
        )
        return wakefield

    def test_wake_wraps_around_the_turn_boundary(self):
        n_turns = 4
        wakefield_poles = self._run(MultiPoleSparseSolve(), n_turns=n_turns)
        wakefield_reference = self._run(
            ContinuousMultiTurnTimeDomainSolver(n_turns=2), n_turns=n_turns
        )

        voltage_poles = copy_to_cpu(wakefield_poles.induced_voltage)
        voltage_reference = copy_to_cpu(wakefield_reference.induced_voltage)

        if DEV_DRAW:  # pragma: no cover
            hist_x = copy_to_cpu(wakefield_poles.profile.hist_x)
            plt.figure("turn_boundary")
            plt.plot(hist_x, voltage_poles, label="MultiPoleSparseSolve")
            plt.plot(
                hist_x, voltage_reference, "--", label="continuous multi-turn"
            )
            plt.legend()
            plt.show()

        scale = np.max(np.abs(voltage_reference))
        self.assertGreater(scale, 0.0)
        # The wake of the trailing bunch must still be visible in the first
        # bins of the turn -- otherwise the test would pass on a solver that
        # simply forgets the previous turn.
        self.assertGreater(np.max(np.abs(voltage_reference[:4])), 0.01 * scale)
        np.testing.assert_allclose(
            voltage_poles,
            voltage_reference,
            rtol=1e-3,
            atol=1e-3 * scale,
        )


class TestPoleKernelCallBoundary(unittest.TestCase):
    """Splitting a profile over two calls must change nothing.

    `wake_from_pole_residue` hands its state to the next call through
    `states`, so two calls over consecutive halves of a profile have to
    reproduce the single call over the whole of it, bin for bin. This is the
    kernel-level form of a turn boundary that a full-turn profile puts one
    single bin apart.
    """

    def test_split_call_matches_single_call(self):
        n_bins = 64
        split = 32
        rng = np.random.default_rng(7)
        profile = backend.array(rng.random(n_bins), dtype=backend.float)
        profile_dts = backend.array(
            np.linspace(0.0, 6.3e-9, n_bins), dtype=backend.float
        )
        bin_dt = float(profile_dts[1] - profile_dts[0])
        poles = backend.array(
            np.array(
                [-2e8 + 2j * np.pi * 1e9, -5e8 + 2j * np.pi * 2e9],
                dtype=complex,
            ),
            dtype=backend.complex,
        )
        residues = backend.array(
            np.array([1.0 + 0.5j, 0.3 - 0.2j], dtype=complex),
            dtype=backend.complex,
        )
        n_poles = len(poles)
        update_on_bin = backend.array(np.zeros(1, dtype=np.int32))
        pole_signs = backend.ones_like(poles, dtype=backend.float)

        def _new_states():
            states = backend.zeros(2 * n_poles + 2, dtype=backend.complex)
            states[-1] = profile_dts[0] - bin_dt
            states[-2] = profile_dts[0] - 2.0 * bin_dt
            return states

        def _call(profile_part, dts_part, states):
            voltage = backend.zeros(len(profile_part), dtype=backend.float)
            backend.specials.wake_from_pole_residue(
                profile=profile_part,
                profile_dts=dts_part,
                poles=poles,
                residues=residues,
                is_counterrotating_beam=False,
                counterrotating_pole_signs=pole_signs,
                states=states,
                voltage=voltage,
                voltage_threaded=backend.zeros(
                    (backend.specials.get_max_threads(), len(profile_part)),
                    dtype=backend.float,
                ),
                update_on_bin=update_on_bin,
                factor=1.0,
            )
            return copy_to_cpu(voltage)

        voltage_single = _call(profile, profile_dts, _new_states())

        states = _new_states()
        voltage_first = _call(profile[:split], profile_dts[:split], states)
        voltage_second = _call(profile[split:], profile_dts[split:], states)
        voltage_split = np.concatenate((voltage_first, voltage_second))

        np.testing.assert_allclose(
            voltage_split,
            voltage_single,
            rtol=1e-10,
            atol=1e-10 * np.max(np.abs(voltage_single)),
        )


if __name__ == "__main__":
    unittest.main()
