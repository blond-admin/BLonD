"""The tiled-GEMM far field of the pole-residue wake, against BLonD's kernel.

`PoleTileGemm` evaluates `Specials.wake_from_pole_residue` as two GEMMs per
slab of tiles and a parallel scan over the tiles, and `MultiPoleGemmSolve`
uses it as the far field of `MultiPoleSparseSolve`. Both must reproduce the
recursion: voltage and hand-over state, call after call.
"""

import unittest

import numpy as np

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
from blond.experimental.physics.impedances.pole_tile_gemm import (
    MultiPoleGemmSolve,
    PoleTileGemm,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.testing.backend_testing import BLonDTestCase

BIN_DT = 2.5e-11


def _grid(rng, n_runs, run_length, last_run_length):
    """Runs of `run_length` bins (the last `last_run_length`), random gaps."""
    lengths = [run_length] * (n_runs - 1) + [last_run_length]
    offsets, position = [], 0
    for length in lengths:
        offsets.append(position + np.arange(length))
        position += length + int(rng.integers(1, 300))
    return np.concatenate(offsets)


def _poles(rng, n_poles):
    frequency = rng.uniform(1e8, 5e9, n_poles)
    quality = 10 ** rng.uniform(-0.2, 4.5, n_poles)  # down to fast poles
    poles = -2 * np.pi * frequency / (2 * quality) + 2j * np.pi * frequency
    real = np.arange(n_poles) % 5 == 0
    poles[real] = poles[real].real
    residues = rng.normal(size=n_poles) + 1j * rng.normal(size=n_poles)
    residues[real] = residues[real].real
    signs = np.where(np.arange(n_poles) % 3 == 0, -1.0, 1.0)
    return poles, residues, signs


class TestPoleTileGemm(BLonDTestCase):
    """`far_field` equals `Specials.wake_from_pole_residue`, call by call."""

    def _reference_call(
        self, offsets, profile, poles, residues, signs, states, call
    ):
        voltage = backend.zeros(len(profile), dtype=backend.float)
        backend.specials.wake_from_pole_residue(
            profile_time=backend.array(3e-6 + offsets * BIN_DT),
            profile=backend.array(profile, dtype=backend.float),
            carried_charge=backend.array([call["carried"]]),
            carried_is_counterrotating=call["carried_cr"],
            state_lag_dt=call["state_lag"] * BIN_DT,
            carried_lag_dt=call["carried_lag"] * BIN_DT,
            poles=backend.array(poles, dtype=backend.complex),
            residues=backend.array(residues, dtype=backend.complex),
            is_counterrotating_beam=call["cr"],
            counterrotating_pole_signs=backend.array(signs),
            factor=call["factor"],
            bin_dt=BIN_DT,
            states=states,
            voltage=voltage,
            voltage_threaded=backend.zeros(
                (backend.specials.get_max_threads(), len(profile))
            ),
        )
        return copy_to_cpu(voltage)

    def _check(self, offsets, n_poles, seed, max_slab_bytes=None):
        rng = np.random.default_rng(seed)
        poles, residues, signs = _poles(rng, n_poles)
        engine = PoleTileGemm(
            bin_offsets=offsets,
            poles=backend.array(poles, dtype=backend.complex),
            residues=backend.array(residues, dtype=backend.complex),
            counterrotating_pole_signs=backend.array(signs),
            bin_dt=BIN_DT,
            **(
                {}
                if max_slab_bytes is None
                else {"max_slab_bytes": max_slab_bytes}
            ),
        )
        states_in = rng.normal(size=n_poles) + 1j * rng.normal(size=n_poles)
        reference_states = backend.array(states_in, dtype=backend.complex)
        gemm_states = backend.array(states_in, dtype=backend.complex)
        for call_i in range(3):
            carried_lag = float(rng.choice([1.0, 1.4, 2.0, 7.3, 300.0]))
            call = {
                "carried": float(rng.normal()),
                "carried_cr": bool(rng.random() < 0.5),
                "cr": bool(rng.random() < 0.5),
                "carried_lag": carried_lag,
                "state_lag": carried_lag - 1.0,
                "factor": float(rng.uniform(0.5, 2.0)),
            }
            profile = rng.random(len(offsets)) * 1e3
            expected = self._reference_call(
                offsets,
                profile,
                poles,
                residues,
                signs,
                reference_states,
                call,
            )
            voltage = copy_to_cpu(
                engine.far_field(
                    profile=backend.array(profile, dtype=backend.float),
                    carried_charge=backend.array([call["carried"]]),
                    carried_is_counterrotating=call["carried_cr"],
                    state_lag_dt=call["state_lag"] * BIN_DT,
                    carried_lag_dt=call["carried_lag"] * BIN_DT,
                    is_counterrotating_beam=call["cr"],
                    factor=call["factor"],
                    states=gemm_states,
                )
            )
            scale = float(np.max(np.abs(expected)))
            np.testing.assert_allclose(
                voltage,
                expected,
                rtol=1e-9,
                atol=1e-9 * scale,
                err_msg=f"{call_i=} {call}",
            )
            np.testing.assert_allclose(
                copy_to_cpu(gemm_states),
                copy_to_cpu(reference_states),
                rtol=1e-9,
                atol=1e-9
                * float(np.max(np.abs(copy_to_cpu(reference_states)))),
                err_msg=f"states {call_i=} {call}",
            )

    def test_bunches_with_gaps(self):
        rng = np.random.default_rng(1)
        self._check(_grid(rng, 20, 128, 128), n_poles=37, seed=2)

    def test_partial_last_run(self):
        rng = np.random.default_rng(3)
        self._check(_grid(rng, 9, 64, 23), n_poles=21, seed=4)

    def test_single_run(self):
        self._check(np.arange(700), n_poles=16, seed=5)

    def test_single_bin(self):
        self._check(np.arange(1), n_poles=5, seed=6)

    def test_slabs_equal_one_pass(self):
        """A tiny memory budget forces many slabs; the result is the same."""
        rng = np.random.default_rng(7)
        self._check(
            _grid(rng, 30, 64, 64), n_poles=33, seed=8, max_slab_bytes=1
        )

    def test_grid_without_common_tile_raises(self):
        offsets = np.concatenate((np.arange(10), 50 + np.arange(64)))
        with self.assertRaises(ValueError):
            PoleTileGemm(
                bin_offsets=offsets,
                poles=backend.array([-1e8 + 1e9j], dtype=backend.complex),
                residues=backend.array([1.0 + 0j], dtype=backend.complex),
                counterrotating_pole_signs=backend.array([1.0]),
                bin_dt=BIN_DT,
            )


class TestMultiPoleGemmSolve(unittest.TestCase):
    """The solver equals `MultiPoleSparseSolve` over several turns."""

    n_bins = 512

    def _run(self, solver, n_turns=4):
        circumference = 26658.883
        ring = Ring(circumference=circumference)
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton, value=450e9, in_unit="momentum"
        )
        t_rev = magnetic_cycle.get_t_rev_init(
            circumference, particle_type=proton
        )
        profile = StaticProfile(
            cut_left=0.0, cut_right=t_rev, n_bins=self.n_bins
        )
        wakefield = WakeField(
            sources=(Resonators(1e6, 1e9, 1e4), Resonators(3e5, 2.3e9, 3.0)),
            solver=solver,
            profile=profile,
        )
        drift = DriftSimple(
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=55.759505
            ),
            orbit_length=circumference,
        )
        rf_station = SingleHarmonicRFStation(
            harmonic=35640, voltage=6e6, phi_rf=0.0
        )
        ring.add_elements((wakefield, drift, rf_station))
        simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
        bin_dt = t_rev / self.n_bins
        bunch = Beam(intensity=1e11, particle_type=proton)
        simulation.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=2e-10, n_macroparticles=20000, seed=42
            ),
            beam=bunch,
        )
        beam = make_multibunch_beam(
            beam=bunch,
            n_times=3,
            t_distance=t_rev / 3.1,
            common_offset=4.5 * bin_dt,
        )

        def advance_reference_time(simulation, beam):
            beam.reference.time += circumference / beam.reference.velocity

        # a frozen beam: every turn sees the same profile
        drift.orbit_length = 0.0
        rf_station.voltage = 0.0
        simulation.check_circumference = "ignore"
        simulation.run_simulation(
            beams=beam, n_turns=n_turns, callbacks=advance_reference_time
        )
        return copy_to_cpu(wakefield.induced_voltage)

    def test_matches_multi_pole_sparse_solve(self):
        expected = self._run(MultiPoleSparseSolve())
        voltage = self._run(MultiPoleGemmSolve())
        scale = float(np.max(np.abs(expected)))
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(
            voltage, expected, rtol=1e-9, atol=1e-9 * scale
        )


if __name__ == "__main__":
    unittest.main()
