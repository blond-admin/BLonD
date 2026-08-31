"""
Whole-feedback equivalence of the numba and CUDA backends.

``tests/unittests/core/backends`` pins the backends against each other
kernel by kernel; nothing checks that a *cavity feedback in a tracking
loop* produces the same physics on the GPU. This module closes that gap
with the cheapest end-to-end fixture in the package: the one-section
RCS1-like ring of
:mod:`test_beam_loading_sign_vs_design_rf_phase` (h = 2590, one drift +
one RF station carrying an :class:`IQCavityFeedbackTimingClass`), shrunk
to 2000 macroparticles and run for a handful of turns so the loop closes
(profile -> beam current -> cavity -> kick -> profile) several times.

The same fixture is built and tracked twice in one process, once on
``Numpy64Bit`` with the ``numba`` specials and once on ``Cupy64Bit`` with
the ``cuda`` specials, from bit-identical initial coordinates. Bit
identity of the *results* is not available and is not asserted: the numba
kick evaluates BLonD's ``fast_sin`` Cody-Waite polynomial (< 2 ULP) while
the CUDA kick calls libdevice ``sin``, and the two histogram kernels
accumulate the profile in a different order.

Measured relative L2 error, CUDA vs numba, on the settings below::

    dt                              2.4576e-17
    dE                              1.6159e-15
    beam current, fine grid         8.1295e-16
    antenna voltage, coarse grid    6.9020e-18
    beam-induced antenna voltage    1.9590e-17

and the same fixture at 5000 macroparticles, swept in turns, shows the
error is essentially flat -- it is float64 round-off, not accumulation::

    turns   dE         dt         V_ant,coarse   I_beam,fine
        1   1.5e-15    0.0        3.0e-18        8.2e-16
        5   1.5e-15    2.4e-17    0.0            8.4e-16
       20   1.6e-15    7.1e-17    1.1e-17        7.4e-16
       50   3.6e-15    1.4e-16    7.3e-18        7.0e-16

``fast_sin`` is not the dominant term: numba vs the plain ``python``
specials is 1.2e-15 on ``dE``, the same order as CUDA vs numba (1.1e-15).

``TOLERANCE = 1e-11`` therefore keeps roughly four orders of margin over
the worst observed value, while still failing hard on any real
divergence. On this metric a sign error is 2.0e0, a one-sample shift of
the beam current is 2.2e-1, and even a 1 ppm scale error is 1.0e-6.
"""

import unittest

import numpy as np
import pytest

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    Cupy64Bit,
    DriftSimple,
    Numpy64Bit,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    backend,
    mu_plus,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.testing.backend_testing import cupy_available

from .support import rel_err

#: RCS1-like cavity and ring, shared with the sibling beam-loading tests.
R_OVER_Q = 518.0
Q_L = 1.29e4
ALPHA_P = 10.395e-4
CIRCUMFERENCE = 5990.0
ENERGY = 63e9
HARMONIC = 2590
INTENSITY = 2.7e12
V_DESIGN = 30e6
N_SLICES = 1024
#: Deliberately small: the check is about backend agreement, not statistics.
N_MACROPARTICLES = 2000
#: Enough turns for the feedback loop to close on its own output.
N_TURNS = 5
#: Four orders of margin over the measured 50-turn round-off (see module
#: docstring); still tight enough that any physics divergence fails.
TOLERANCE = 1e-11


@pytest.mark.cupy
@unittest.skipIf(not cupy_available, "Cupy not found")
class TestCavityFeedbackNumbaVsCudaEquivalence(unittest.TestCase):
    """The cavity feedback tracks the same physics on CPU and on GPU."""

    @classmethod
    def setUpClass(cls):
        """Track the fixture once per backend and keep the host copies."""
        cls._entry_backend = type(backend)
        cls._entry_specials = backend.specials_mode
        try:
            cls.cpu = cls._run(Numpy64Bit, "numba")
            cls.gpu = cls._run(Cupy64Bit, "cuda")
        finally:
            backend.change_backend(cls._entry_backend)
            backend.set_specials(cls._entry_specials)

    @classmethod
    def tearDownClass(cls):
        """Hand the session back the backend it arrived with."""
        backend.change_backend(cls._entry_backend)
        backend.set_specials(cls._entry_specials)

    @staticmethod
    def _initial_coordinates(t_rf: float):
        """
        Bunch coordinates, generated on the host so both runs start equal.

        Parameters
        ----------
        t_rf
            RF period of the station [s].

        Returns
        -------
        dt
            Arrival times of the macroparticles [s].
        dE
            Energy offsets of the macroparticles [eV].
        """
        rng = np.random.default_rng(7)
        dt = 1.5 * t_rf + 0.06 * t_rf * rng.standard_normal(N_MACROPARTICLES)
        dE = 1.5e7 * rng.standard_normal(N_MACROPARTICLES)
        return dt, dE

    @classmethod
    def _run(cls, backend_class, specials: str) -> dict:
        """
        Build and track the fixture on one backend.

        Parameters
        ----------
        backend_class
            Backend to activate, e.g. ``Numpy64Bit`` or ``Cupy64Bit``.
        specials
            Specials mode to activate, e.g. ``"numba"`` or ``"cuda"``.

        Returns
        -------
        state
            Host copies of the beam coordinates and of the feedback's
            beam current and antenna voltage after ``N_TURNS`` turns.
        """
        backend.change_backend(backend_class)
        backend.set_specials(specials)

        cycle = ConstantMagneticCycle(
            reference_particle=mu_plus, value=ENERGY, in_unit="total energy"
        )
        t_rev = cycle.get_t_rev_init(CIRCUMFERENCE, particle_type=mu_plus)
        t_rf = t_rev / HARMONIC

        profile = StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, N_SLICES, t_rf
        )
        # Operating-point cavity: V_init = V_design held by the matched
        # generator current, as in the sibling beam-loading tests.
        feedback = IQCavityFeedbackTimingClass(
            profile=profile,
            R_over_Q=R_OVER_Q,
            Q_L=Q_L,
            generator_current_bias=V_DESIGN / (2.0 * R_OVER_Q * Q_L),
            n_cavities=1,
            initial_voltage=V_DESIGN,
            n_rf_periods_per_coarse_grid=1,
            delta_omega=0.0,
        )
        rf_station = SingleHarmonicRFStation(
            voltage=V_DESIGN,
            phi_rf=0.0,
            harmonic=HARMONIC,
            cavity_feedback=feedback,
            profile=profile,
        )
        ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
        drift = DriftSimple(
            orbit_length=CIRCUMFERENCE, momentum_compaction_factor=ALPHA_P
        )
        # Drift first, so the feedback's station is not the first
        # reference-altering element.
        ring.add_elements([drift, rf_station], reorder=False)
        simulation = Simulation(ring=ring, magnetic_cycle=cycle)

        beam = Beam(intensity=INTENSITY, particle_type=mu_plus)
        beam.reference.total_energy = ENERGY
        simulation.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                n_macroparticles=N_MACROPARTICLES,
                sigma_dt=0.06 * t_rf,
                sigma_dE=1.5e7,
                seed=7,
                reinsertion=True,
            ),
        )
        dt, dE = cls._initial_coordinates(t_rf)
        beam.setup_beam(dt=dt, dE=dE)

        simulation.run_simulation(
            (beam,), n_turns=N_TURNS, show_progressbar=False
        )
        return {
            "dt": copy_to_cpu(beam.dt.array_local).copy(),
            "dE": copy_to_cpu(beam.dE.array_local).copy(),
            "beam current, fine grid": copy_to_cpu(
                feedback.beam_current_fine_grid
            ).copy(),
            "antenna voltage, coarse grid": copy_to_cpu(
                feedback.antenna_voltage_coarse_grid
            ).copy(),
            "beam-induced antenna voltage": copy_to_cpu(
                feedback.antenna_voltage_beam_coarse_grid
            ).copy(),
        }

    def test_gpu_reproduces_the_cpu_feedback_and_beam(self):
        """Every tracked and every feedback array agrees to round-off."""
        for key, reference in self.cpu.items():
            with self.subTest(quantity=key):
                error = rel_err(self.gpu[key], reference)
                self.assertLess(
                    error,
                    TOLERANCE,
                    msg=(
                        f"the CUDA backend and the numba backend disagree "
                        f"on '{key}' after {N_TURNS} turns: relative L2 "
                        f"error {error:.3e} > {TOLERANCE:.0e}"
                    ),
                )
