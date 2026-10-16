"""`MultiPoleSparseSolve` with the GEMM far field of the CUDA backend.

On a GPU the solver evaluates its far field with
`CudaSpecials.no_parity.wake_from_pole_residue_gemm` when the grid tiles
and there are enough poles; the result must equal the kernel's.
"""

from unittest import mock

import numpy as np
import pytest

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    Cupy64Bit,
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
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.testing.backend_testing import BLonDTestCase

CIRCUMFERENCE = 26658.883
N_BINS = 512


def _run(
    solver: MultiPoleSparseSolve,
    n_turns: int = 4,
    profile_fraction: float = 1.0,
) -> np.ndarray:
    """
    Induced voltage after `n_turns` of a frozen three-bunch beam.

    The profile spans `profile_fraction` of the revolution period, from
    its start: 1 makes the calls seamless, less leaves a gap between them.
    """
    ring = Ring(circumference=CIRCUMFERENCE)
    magnetic_cycle = ConstantMagneticCycle(
        reference_particle=proton, value=450e9, in_unit="momentum"
    )
    t_rev = magnetic_cycle.get_t_rev_init(CIRCUMFERENCE, particle_type=proton)
    n_bins = round(N_BINS * profile_fraction)
    profile = StaticProfile(
        cut_left=0.0, cut_right=t_rev * n_bins / N_BINS, n_bins=n_bins
    )
    rng = np.random.default_rng(0)
    n_resonators = 30  # 30 complex poles: enough for the GEMM
    wakefield = WakeField(
        sources=(
            Resonators(
                rng.uniform(1e4, 1e6, n_resonators),
                rng.uniform(2e8, 3e9, n_resonators),
                10 ** rng.uniform(0.5, 4.0, n_resonators),
            ),
        ),
        solver=solver,
        profile=profile,
    )
    drift = DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=55.759505
        ),
        orbit_length=CIRCUMFERENCE,
    )
    rf_station = SingleHarmonicRFStation(
        harmonic=35640, voltage=6e6, phi_rf=0.0
    )
    ring.add_elements((wakefield, drift, rf_station))
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    bin_dt = t_rev / N_BINS
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
        beam.reference.time += CIRCUMFERENCE / beam.reference.velocity

    # a frozen beam: every turn sees the same profile
    drift.orbit_length = 0.0
    rf_station.voltage = 0.0
    simulation.check_circumference = "ignore"
    simulation.run_simulation(
        beams=beam, n_turns=n_turns, callbacks=advance_reference_time
    )
    return copy_to_cpu(wakefield.induced_voltage)


@pytest.mark.cupy
@pytest.mark.backend_mutation
class TestMultiPoleSparseSolveGemm(BLonDTestCase):
    """The GEMM far field gives the kernel's voltage over several turns."""

    def setUp(self) -> None:
        try:
            from blond.core.backends.cuda.callables import CudaSpecials
        except ImportError as exc:
            if "cupy" in str(exc):
                self.skipTest("Cupy not available")
            raise
        self.cuda_specials = CudaSpecials
        original_backend = type(backend)
        original_mode = backend.specials_mode
        self.addCleanup(backend.set_specials, original_mode)
        self.addCleanup(backend.change_backend, original_backend)
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")

    def test_uses_gemm_and_matches_kernel(self) -> None:
        solver = MultiPoleSparseSolve()
        voltage = _run(solver)
        self.assertIsNotNone(solver._far_field.gemm_plan)
        with mock.patch.object(self.cuda_specials, "no_parity", None):
            kernel_solver = MultiPoleSparseSolve()
            expected = _run(kernel_solver)
        self.assertIsNone(kernel_solver._far_field.gemm_plan)
        scale = float(np.max(np.abs(expected)))
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(
            voltage, expected, rtol=1e-9, atol=1e-9 * scale
        )

    def test_near_field_kernel_matches_python(self) -> None:
        """`pole_near_field` gives the near field and carried charge."""
        no_parity = self.cuda_specials.no_parity
        for fraction in (1.0, 0.75):  # seamless calls, then a gap
            with self.subTest(profile_fraction=fraction):
                with mock.patch.object(
                    no_parity,
                    "pole_near_field",
                    wraps=no_parity.pole_near_field,
                ) as near_field:
                    voltage = _run(MultiPoleSparseSolve(), 4, fraction)
                self.assertEqual(near_field.call_count, 4)
                with mock.patch.object(self.cuda_specials, "no_parity", None):
                    expected = _run(MultiPoleSparseSolve(), 4, fraction)
                scale = float(np.max(np.abs(expected)))
                np.testing.assert_allclose(
                    voltage, expected, rtol=1e-9, atol=1e-9 * scale
                )
