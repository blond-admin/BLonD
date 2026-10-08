"""
Direct and inductive terms of a vector-fitted model in
`MultiPoleSparseSolve`.

A vector fit of an impedance gives, besides its poles, a constant term
:math:`d` and a term :math:`e\\,s` linear in :math:`s = 2\\pi i f`:
:math:`Z(s) = \\sum_k \\rho_k / (s - p_k) + d + e\\,s`. Neither has a
pole-residue representation; their wakes are :math:`d\\,\\delta(t)` and
:math:`e\\,\\delta'(t)`, so `MultiPoleSparseSolve` applies them to the
binned profile directly. They are checked against `PeriodicFreqSolver`
with the equivalent frequency-domain sources, on a noise-free profile.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.constants import e
from scipy.special import erf

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    PeriodicFreqSolver,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
    copy_to_cpu,
    momentum_compaction_factor,
    proton,
)
from blond.physics.impedances.base import (
    FreqDomain,
    SupportsVectorFittedModel,
    VectorFit,
    WakeFieldSource,
)
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.impedances.sources import InductiveImpedance, Resonators
from blond.testing.backend_testing import BLonDTestCase

CIRCUMFERENCE = 6911.56  # m
TRANSITION_GAMMA = 22.8
HARMONIC = 4620
SYNC_MOMENTUM = 26e9  # eV/c
INTENSITY = 1e11
N_BINS = 128
SIGMA_DT = 0.5e-9  # s

DIRECT_TERM = 2.5e3  # Ohm
INDUCTIVE_TERM = 3e-8  # Ohm s


class _VectorFittedSource(WakeFieldSource, SupportsVectorFittedModel):
    """A source given directly by its vector fit."""

    def __init__(
        self, poles=(), residues=(), direct_term=0.0, inductive_term=0.0
    ):
        super().__init__(is_dynamic=False)
        self._fit = VectorFit(
            poles=np.asarray(poles, dtype=complex),
            residues=np.asarray(residues, dtype=complex),
            counterrotation_signs=np.ones(len(poles)),
            direct_term=direct_term,
            inductive_term=inductive_term,
        )

    def get_vectorfit(self) -> VectorFit:
        return self._fit


class _LegacyVectorFittedSource(WakeFieldSource, SupportsVectorFittedModel):
    """A source still returning the plain three-tuple of poles."""

    def __init__(self, resonators):
        super().__init__(is_dynamic=False)
        self._resonators = resonators

    def get_vectorfit(self):
        fit = self._resonators.get_vectorfit()
        return fit.poles, fit.residues, fit.counterrotation_signs


class _ConstantImpedance(WakeFieldSource, FreqDomain):
    """:math:`Z(f) = d` at every frequency."""

    def __init__(self, direct_term):
        super().__init__(is_dynamic=False)
        self.direct_term = direct_term

    def get_impedance(self, freq_x, simulation, beam, hist_step=None):
        return self.direct_term * backend.ones(
            len(freq_x), dtype=backend.complex
        )


def _t_rev():
    """Revolution period, in [s]."""
    cycle = ConstantMagneticCycle(
        reference_particle=proton, value=SYNC_MOMENTUM, in_unit="momentum"
    )
    return cycle.get_t_rev_init(CIRCUMFERENCE, particle_type=proton)


def _bin_integrated_gaussian(hist_x, hist_step, center, sigma_dt):
    """Fraction of a Gaussian bunch in each bin, free of shot noise."""
    scale = np.sqrt(2.0) * sigma_dt
    return 0.5 * (
        erf((hist_x + 0.5 * hist_step - center) / scale)
        - erf((hist_x - 0.5 * hist_step - center) / scale)
    )


def _run_one_turn(sources, solver):
    """One turn on an analytic Gaussian line density, one RF bucket long.

    Returns
    -------
    induced_voltage
        The induced voltage on the profile bins, in [V].
    """
    ring = Ring(circumference=CIRCUMFERENCE)
    cycle = ConstantMagneticCycle(
        reference_particle=proton, value=SYNC_MOMENTUM, in_unit="momentum"
    )
    drift = DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=TRANSITION_GAMMA
        ),
        orbit_length=CIRCUMFERENCE,
    )
    rf_station = SingleHarmonicRFStation(
        harmonic=HARMONIC, voltage=1e6, phi_rf=0.0
    )
    t_rf = _t_rev() / HARMONIC
    profile = StaticProfile(cut_left=0.0, cut_right=t_rf, n_bins=N_BINS)
    wakefield = WakeField(sources=sources, solver=solver, profile=profile)
    wakefield.track_profile = False
    ring.add_elements([drift, rf_station, wakefield], reorder=True)

    simulation = Simulation(ring=ring, magnetic_cycle=cycle)
    beam = Beam(intensity=INTENSITY, particle_type=proton)
    simulation.prepare_beam(
        beam=beam,
        preparation_routine=BiGaussian(
            sigma_dt=SIGMA_DT, n_macroparticles=1_000, seed=42
        ),
    )
    hist_x = copy_to_cpu(profile.hist_x)
    profile.hist_y[:] = backend.array(
        _bin_integrated_gaussian(
            hist_x, profile.hist_step, 0.5 * t_rf, SIGMA_DT
        ),
        dtype=backend.float,
    )
    profile.hist_y_to_density_factor = 1.0
    simulation.run_simulation(beams=(beam,), n_turns=1)
    return copy_to_cpu(wakefield.induced_voltage)


class TestVectorFit(unittest.TestCase):
    def test_terms_default_to_zero(self):
        fit = VectorFit(poles=[], residues=[], counterrotation_signs=[])
        self.assertEqual(fit.direct_term, 0.0)
        self.assertEqual(fit.inductive_term, 0.0)

    def test_resonators_have_no_direct_or_inductive_term(self):
        fit = Resonators(1e6, 1e9, 10.0).get_vectorfit()
        self.assertIsInstance(fit, VectorFit)
        self.assertEqual(fit.direct_term, 0.0)
        self.assertEqual(fit.inductive_term, 0.0)


class TestMultiPoleSparseSolveMatchesFreqDomain(BLonDTestCase):
    """The terms against `PeriodicFreqSolver`, which applies them exactly."""

    def _assert_matches(self, voltage, voltage_reference):
        peak = np.max(np.abs(voltage_reference))
        self.assertGreater(peak, 0.0)
        np.testing.assert_allclose(
            voltage, voltage_reference, rtol=0.0, atol=1e-9 * peak
        )

    def test_direct_term(self):
        voltage = _run_one_turn(
            (_VectorFittedSource(direct_term=DIRECT_TERM),),
            MultiPoleSparseSolve(),
        )
        voltage_reference = _run_one_turn(
            (_ConstantImpedance(DIRECT_TERM),), PeriodicFreqSolver()
        )
        self._assert_matches(voltage, voltage_reference)

    def test_inductive_term(self):
        """`InductiveImpedance` uses the same central difference."""
        t_rev = _t_rev()
        voltage = _run_one_turn(
            (_VectorFittedSource(inductive_term=INDUCTIVE_TERM),),
            MultiPoleSparseSolve(),
        )
        voltage_reference = _run_one_turn(
            (InductiveImpedance(Z_over_n=2 * np.pi * INDUCTIVE_TERM / t_rev),),
            PeriodicFreqSolver(),
        )
        self._assert_matches(voltage, voltage_reference)

    def test_terms_add_to_the_poles(self):
        """Poles, direct and inductive term superpose."""
        resonators = Resonators(1e5, 1.2e9, 5.0)
        fit = resonators.get_vectorfit()
        combined = _run_one_turn(
            (
                _VectorFittedSource(
                    fit.poles, fit.residues, DIRECT_TERM, INDUCTIVE_TERM
                ),
            ),
            MultiPoleSparseSolve(),
        )
        separate = (
            _run_one_turn((resonators,), MultiPoleSparseSolve())
            + _run_one_turn(
                (_VectorFittedSource(direct_term=DIRECT_TERM),),
                MultiPoleSparseSolve(),
            )
            + _run_one_turn(
                (_VectorFittedSource(inductive_term=INDUCTIVE_TERM),),
                MultiPoleSparseSolve(),
            )
        )
        self._assert_matches(combined, separate)

    def test_terms_of_several_sources_add(self):
        voltage = _run_one_turn(
            (
                _VectorFittedSource(direct_term=0.25 * DIRECT_TERM),
                _VectorFittedSource(direct_term=0.75 * DIRECT_TERM),
            ),
            MultiPoleSparseSolve(),
        )
        voltage_reference = _run_one_turn(
            (_VectorFittedSource(direct_term=DIRECT_TERM),),
            MultiPoleSparseSolve(),
        )
        self._assert_matches(voltage, voltage_reference)


class TestMultiPoleSparseSolveOnHandMadeAxes(BLonDTestCase):
    """The terms on bin axes with gaps, as an `EquidistantMultiProfile` has."""

    BIN_DT = 1e-10

    def _voltage(self, sources, hist_x_in_bins, hist_y):
        solver = MultiPoleSparseSolve()
        parent = Mock(WakeField)
        parent.sources = sources
        hist_x = backend.array(
            self.BIN_DT * np.asarray(hist_x_in_bins, dtype=float),
            dtype=backend.float,
        )
        profile = Mock(spec=StaticProfile)
        profile.hist_x = hist_x
        profile.hist_y = backend.array(hist_y, dtype=backend.float)
        profile.hist_step = self.BIN_DT
        profile.hist_y_to_density_factor = 1.0
        parent.profile = profile
        solver._parent_wakefield = parent
        solver._profile = profile
        beam = SimpleNamespace(
            reference=SimpleNamespace(time=0.0),
            particle_type=SimpleNamespace(charge=1.0),
            intensity=1.0 / e,
            is_counter_rotating=False,
        )
        return copy_to_cpu(solver.calc_induced_voltage(beam=beam))

    def test_gapped_axis_matches_dense_axis(self):
        """No difference is taken across a gap; the bins beyond are empty."""
        sources = (
            _VectorFittedSource(
                direct_term=DIRECT_TERM, inductive_term=INDUCTIVE_TERM
            ),
        )
        gapped_x = [0, 1, 2, 3, 7, 8, 9, 10]
        # charge right at both edges of the gap
        gapped_y = [0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 1.0, 0.0]
        dense_x = list(range(11))
        dense_y = [0.0, 1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 5.0, 4.0, 1.0, 0.0]

        voltage_gapped = self._voltage(sources, gapped_x, gapped_y)
        voltage_dense = self._voltage(sources, dense_x, dense_y)
        np.testing.assert_allclose(
            voltage_gapped,
            voltage_dense[gapped_x],
            rtol=1e-12,
            atol=1e-12 * np.max(np.abs(voltage_dense)),
        )

    def test_plain_three_tuple_is_still_accepted(self):
        resonators = Resonators(1e5, 1.2e9, 5.0)
        hist_x = list(range(8))
        hist_y = [0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 1.0, 0.0]
        np.testing.assert_allclose(
            self._voltage(
                (_LegacyVectorFittedSource(resonators),), hist_x, hist_y
            ),
            self._voltage((resonators,), hist_x, hist_y),
            rtol=1e-12,
        )

    def test_rejects_source_without_vector_fit(self):
        with self.assertRaisesRegex(TypeError, "SupportsVectorFittedModel"):
            self._voltage(
                (_ConstantImpedance(DIRECT_TERM),), list(range(4)), [1.0] * 4
            )


if __name__ == "__main__":
    unittest.main()
