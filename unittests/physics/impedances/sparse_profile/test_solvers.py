import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

import blond
from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MagneticCyclePerTurn,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    TimeDomainFftSolver,
    WakeField,
    backend,
    make_multibunch_beam,
    momentum_compaction_factor,
    proton,
)
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.profiles_sparse import EquidistantMultiProfile

resonator_data = np.loadtxt(
    os.path.join(
        os.path.dirname(blond.__file__),
        "examples/scripts/resources/EX_05_new_HQ_table.txt",
    ),
    comments="!",
)

R_shunt = resonator_data[:, 2] * 10**6
f_res = resonator_data[:, 0] * 10**9
Q_factor = resonator_data[:, 1] * 100

# ── machine constants ─────────────────────────────────────────────────────────

CIRCUMFERENCE = 6911.56
TRANSITION_GAMMA = 22.82177322938192
HARMONIC = 4620
SYNC_MOMENTUM = 25.92e9  # [eV/c]
BUNCH_INTENSITY = 1e10
SIGMA_DT = 2e-9 / 4
SEED = 1
BINS_PER_PROFILE = 2**8
BUNCH_SPACING = 10  # fill every Nth slot

n_macroparticles = int(1e4)
DEV_DRAW = False


# ── machine element factories ─────────────────────────────────────────────────


def _make_ring():
    return Ring(circumference=CIRCUMFERENCE)


def _make_drift(ring):
    return DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=TRANSITION_GAMMA
        ),
        orbit_length=ring.circumference,
    )


def _make_rf_station():
    return SingleHarmonicRFStation(
        harmonic=HARMONIC, voltage=0.9e6, phi_rf=0.0
    )


def _make_bunch():
    return Beam(intensity=BUNCH_INTENSITY, particle_type=proton)


# ── simulation runners ────────────────────────────────────────────────────────


def _run_sparse(filling_pattern, n_turns=1, callbacks=None):
    """Run MultiPoleSparseSolve with a frozen beam (orbit_length=0, voltage=0).

    Parameters
    ----------
    filling_pattern : bool array of length HARMONIC
    n_turns : number of turns to simulate
    callbacks : optional callback passed to run_simulation (used for the
                multi-turn test to advance reference.time manually)

    Returns
    -------
    (WakeField, t_rev)
    """
    backend.set_specials("cpp")  # TODO remove
    ring = _make_ring()
    magnetic_cycle = ConstantMagneticCycle(
        reference_particle=proton, value=SYNC_MOMENTUM, in_unit="momentum"
    )
    bunch = _make_bunch()
    drift = _make_drift(ring)
    rf_station = _make_rf_station()
    t_rev = magnetic_cycle.get_t_rev_init(
        ring.circumference, particle_type=proton
    )
    t_rf = t_rev / HARMONIC

    profile = EquidistantMultiProfile(
        bins_per_profile=BINS_PER_PROFILE,
        filling_pattern=filling_pattern,
        offset=0,
    )
    wakefield = WakeField(
        sources=(Resonators(R_shunt, f_res, Q_factor),),
        solver=MultiPoleSparseSolve(),
        profile=profile,
    )
    ring.add_elements((wakefield, drift, rf_station))
    sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    sim.prepare_beam(
        preparation_routine=BiGaussian(
            sigma_dt=SIGMA_DT, seed=SEED, n_macroparticles=n_macroparticles
        ),
        beam=bunch,
    )
    beam = make_multibunch_beam(
        beam=bunch,
        n_times=int(np.sum(filling_pattern)),
        t_distance=t_rf * BUNCH_SPACING,
    )
    drift.orbit_length = 0  # freeze beam profile across turns
    rf_station.voltage = 0.0
    sim.check_circumference = "ignore"
    sim.run_simulation(beams=beam, n_turns=n_turns, callbacks=callbacks)
    return wakefield, t_rev


def _run_reference(n_rad, n_bins, n_bunches):
    """Run TimeDomainFftSolver on a StaticProfile as ground-truth reference.

    Parameters
    ----------
    n_rad     : profile width in units of full turns
                (1.0 = one turn, 1/HARMONIC = one RF bucket, 2.0 = two turns)
    n_bins    : total number of bins in the StaticProfile
    n_bunches : number of bunches spread uniformly over the profile
    """
    ring = _make_ring()
    magnetic_cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        values_after_turn=np.linspace(SYNC_MOMENTUM, SYNC_MOMENTUM, 2),
        value_init=SYNC_MOMENTUM,
        in_unit="momentum",
    )
    bunch = _make_bunch()
    drift = _make_drift(ring)
    rf_station = _make_rf_station()
    t_rev = magnetic_cycle.get_t_rev_init(
        ring.circumference, particle_type=proton
    )
    t_rf = t_rev / HARMONIC

    profile = StaticProfile.from_rad(0, 2 * np.pi * n_rad, n_bins, t_rev)
    wakefield = WakeField(
        sources=(Resonators(R_shunt, f_res, Q_factor),),
        solver=TimeDomainFftSolver(allow_next_fast_len=False),
        profile=profile,
    )
    ring.add_elements((wakefield, drift, rf_station))
    sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    sim.prepare_beam(
        preparation_routine=BiGaussian(
            sigma_dt=SIGMA_DT, seed=SEED, n_macroparticles=n_macroparticles
        ),
        beam=bunch,
    )
    beam = make_multibunch_beam(
        beam=bunch, n_times=n_bunches, t_distance=t_rf * BUNCH_SPACING
    )
    sim.run_simulation(beams=beam, n_turns=1)
    return wakefield


# ── shared assertion ──────────────────────────────────────────────────────────


def _assert_matches_reference(sparse_wf, ref_wf, x_shift=0.0):
    """Assert that the sparse induced voltage matches the interpolated reference.

    x_shift : offset added to sparse x-coordinates before interpolating into
              the reference (for the multi-turn test, x_shift=t_rev maps the
              turn-2 sparse bins into [t_rev, 2*t_rev] in the reference).

    Only bins whose shifted x falls within the reference's domain are compared.
    This handles the single-bunch test where the sparse profile spans a full
    turn but the reference only covers one RF bucket — the remaining bunches
    are simply not tested (they are out of the reference's scope).
    """
    x = sparse_wf.profile.hist_x + x_shift
    mask = (x >= ref_wf.profile.hist_x[0]) & (x <= ref_wf.profile.hist_x[-1])

    v_ref_at_sparse = np.interp(
        x[mask],
        ref_wf.profile.hist_x,
        ref_wf.induced_voltage,
    )
    # atol covers zero-signal bins where the reference carries floating-point
    # noise but the sparse solver correctly returns 0.
    atol = 0.01 * np.max(np.abs(v_ref_at_sparse))
    np.testing.assert_allclose(
        sparse_wf.induced_voltage[mask],
        v_ref_at_sparse,
        rtol=1e-6,
        atol=atol,
    )


# ── test classes ──────────────────────────────────────────────────────────────


class TestMultiPoleSparseMultiBunch(unittest.TestCase):
    """Multi-bunch: filling pattern every 10th slot (HARMONIC // BUNCH_SPACING = 462 bunches).

    Most important test case: verifies MultiPoleSparseSolve against a dense
    single-turn reference (TimeDomainFftSolver on a StaticProfile spanning the
    full turn).
    """

    def test_induced_voltage_matches_reference(self):
        filling_pattern = np.zeros(HARMONIC, bool)
        filling_pattern[::BUNCH_SPACING] = 1

        wakefield_sparse, _ = _run_sparse(filling_pattern, n_turns=1)
        wakefield_ref = _run_reference(
            n_rad=1,
            n_bins=BINS_PER_PROFILE * HARMONIC,
            n_bunches=int(np.sum(filling_pattern)),
        )

        if DEV_DRAW:
            profile_sparse = wakefield_sparse.profile
            profile_ref = wakefield_ref.profile
            fig, (ax1, ax2) = plt.subplots(
                2, 1, sharex=True, num="compare_multibunch"
            )
            plt.sca(ax1)
            profile_sparse.plot(marker="x")
            ax1.set_xlim(4e-8, 6e-8)
            ax2.plot(
                profile_sparse.hist_x,
                wakefield_sparse.induced_voltage,
                label="sparse",
            )
            ax2.plot(
                profile_ref.hist_x,
                wakefield_ref.induced_voltage,
                "--",
                label="reference",
            )
            ax2.set_xlim(4e-8, 6e-8)
            ax2.legend()
            plt.show()

        _assert_matches_reference(wakefield_sparse, wakefield_ref)


class TestMultiPoleSparseSingleBunch(unittest.TestCase):
    """Single-bunch edge case: only slot 0 in filling pattern.

    Sanity-checks that MultiPoleSparseSolve reduces correctly to the single-bunch
    regime, compared to a TimeDomainFftSolver on a single-bucket StaticProfile.
    """

    def test_induced_voltage_matches_reference(self):
        filling_pattern = np.zeros(HARMONIC, bool)
        filling_pattern[::BUNCH_SPACING] = 1

        wakefield_sparse, _ = _run_sparse(filling_pattern, n_turns=1)
        wakefield_ref = _run_reference(
            n_rad=1 / HARMONIC,
            n_bins=BINS_PER_PROFILE,
            n_bunches=1,
        )

        if DEV_DRAW:
            profile_sparse = wakefield_sparse.profile
            profile_ref = wakefield_ref.profile
            fig, (ax1, ax2) = plt.subplots(
                2, 1, sharex=True, num="compare_single_bunch"
            )
            plt.sca(ax1)
            profile_sparse.plot()
            ax1.plot(profile_ref.hist_x, profile_ref.hist_y, "--")
            ax2.plot(
                profile_sparse.hist_x,
                wakefield_sparse.induced_voltage,
                label="sparse",
            )
            ax2.plot(
                profile_ref.hist_x,
                wakefield_ref.induced_voltage,
                "--",
                label="reference",
            )
            ax2.legend()
            plt.show()

        _assert_matches_reference(wakefield_sparse, wakefield_ref)


class TestMultiPoleSparseMultiBunchMultiTurn(unittest.TestCase):
    """Multi-turn: verifies that wake contributions from previous turns are accumulated.

    Runs MultiPoleSparseSolve for 2 turns (same 462-bunch filling pattern).
    Compares turn-2 induced voltage against a fake-2-turn reference:
    a StaticProfile spanning 2 revolutions loaded with 2x the bunches,
    solved with TimeDomainFftSolver in a single pass. The turn-2 sparse bins
    map to positions [t_rev, 2*t_rev] in the reference.
    """

    N_TURNS = 2

    def test_sparse_multibunch_multiturn(self):
        filling_pattern = np.zeros(HARMONIC, bool)
        filling_pattern[::BUNCH_SPACING] = 1

        def _advance_reference_time(simulation, beam):
            # orbit_length=0 freezes the beam but also stops reference.time from
            # advancing; this callback restores the correct t_rev increment so
            # the solver's inter-turn state tracking stays valid.
            beam.reference.time += CIRCUMFERENCE / beam.reference.velocity

        wakefield_sparse, t_rev = _run_sparse(
            filling_pattern,
            n_turns=self.N_TURNS,
            callbacks=_advance_reference_time,
        )
        wakefield_ref = _run_reference(
            n_rad=self.N_TURNS,
            n_bins=BINS_PER_PROFILE * HARMONIC * self.N_TURNS,
            n_bunches=int(np.sum(filling_pattern)) * self.N_TURNS,
        )

        if DEV_DRAW:
            profile_sparse = wakefield_sparse.profile
            profile_ref = wakefield_ref.profile
            fig, (ax1, ax2) = plt.subplots(
                2, 1, sharex=True, num="compare_multiturn"
            )
            # Reference spans [0, 2*t_rev]; sparse profile is frozen so turn 1
            # and turn 2 are identical — show both at their respective positions.
            ax1.plot(
                profile_ref.hist_x,
                profile_ref.hist_y,
                color="C1",
                label="reference",
            )
            plt.sca(ax1)
            profile_sparse.plot(marker="x", color="C0")  # turn 1
            ax1.plot(  # turn 2 (shifted)
                profile_sparse.hist_x + t_rev,
                profile_sparse.hist_y,
                marker="x",
                color="C0",
                linestyle=":",
            )
            ax1.legend(["reference", "sparse (both turns)"])
            ax2.plot(
                profile_sparse.hist_x + t_rev,
                wakefield_sparse.induced_voltage,
                label="sparse turn 2",
            )
            ax2.plot(
                profile_ref.hist_x,
                wakefield_ref.induced_voltage,
                "--",
                label="reference (fake 2-turn)",
            )
            ax2.legend()
            plt.show()

        # Turn-2 sparse bins sit at hist_x ∈ [0, t_rev]; in the reference they
        # appear at hist_x + t_rev ∈ [t_rev, 2*t_rev].
        _assert_matches_reference(
            wakefield_sparse, wakefield_ref, x_shift=t_rev
        )


if __name__ == "__main__":
    unittest.main()
