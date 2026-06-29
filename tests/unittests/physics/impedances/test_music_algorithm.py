# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the :class:`blond.physics.impedances.music_algorithm.Music`."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.special as scisp
from scipy.constants import elementary_charge as e

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)
from blond.core.backends.backend import Numpy64Bit, backend
from blond.core.beam.particle_types import uranium_29
from blond.legacy.blond2.impedances.music import Music as LegacyMusic
from blond.physics.impedances.music_algorithm import Music
from blond.physics.impedances.sources import Resonators


@pytest.fixture(autouse=True)
def _numpy_backend():
    """Pin the numpy backend and restore state (tests are order-agnostic)."""
    backend_org = backend.__class__
    specials_org = backend.specials_mode
    backend.change_backend(Numpy64Bit)
    backend.set_specials("python")
    yield
    backend.change_backend(backend_org)
    backend.set_specials(specials_org)


R_S = 1e6
FREQ_R = 1e9
Q = 1.0


def _resonator():
    return Resonators(
        shunt_impedances=R_S,
        center_frequencies=FREQ_R,
        quality_factors=Q,
    )


def _beam(n=64, intensity=1e11, seed=0):
    rng = np.random.default_rng(seed)
    dt = backend.array(rng.random(n) * 1e-9, dtype=backend.float)
    dE = backend.array(rng.standard_normal(n) * 1e6, dtype=backend.float)
    beam = Beam(intensity=intensity, particle_type=proton)
    beam.setup_beam(dt=dt, dE=dE)
    return beam, dt, dE


def test_requires_single_resonator():
    """Music supports a single resonance only."""
    multi = Resonators(
        shunt_impedances=[1e6, 2e6],
        center_frequencies=[1e9, 2e9],
        quality_factors=[1.0, 1.0],
    )
    with pytest.raises(ValueError):
        Music(source=multi)


def test_rejects_non_resonators_source():
    """`source` must be a `Resonators` instance."""
    with pytest.raises(TypeError):
        Music(source=object())


def test_raises_on_cuda_backend(monkeypatch):
    """MuSiC is unsupported on the cuda backend."""
    beam, _, _ = _beam()
    # pretend the cuda backend is active (no GPU needed for this check)
    monkeypatch.setattr(backend, "specials_mode", "cuda")
    with pytest.raises(NotImplementedError):
        Music.headless(beam=beam, source=_resonator())


def test_single_turn_matches_legacy_and_sorts():
    """One turn reproduces legacy ``track_py`` and leaves beam consistent."""
    beam, dt, dE = _beam()
    n = beam.n_macroparticles_partial()
    intensity = beam.intensity

    # legacy oracle
    lbeam = SimpleNamespace(dt=np.asarray(dt).copy(), dE=np.asarray(dE).copy())
    legacy = LegacyMusic(
        lbeam,
        [R_S, 2 * np.pi * FREQ_R, Q],
        n,
        intensity,
        t_rev=2e-6,
    )
    legacy.track_py()

    music = Music.headless(beam=beam, source=_resonator())
    ids_before = np.asarray(beam.read_partial_ids()).copy()
    music.track(beam=beam)

    sorted_dt = np.asarray(beam.read_partial_dt())
    sorted_dE = np.asarray(beam.read_partial_dE())
    sorted_ids = np.asarray(beam.read_partial_ids())

    assert np.all(np.diff(sorted_dt) >= 0)  # sorted ascending
    np.testing.assert_allclose(sorted_dt, lbeam.dt, rtol=1e-12)
    np.testing.assert_allclose(sorted_dE, lbeam.dE, rtol=1e-9)
    # ids must follow the same permutation as dt (particle identity intact)
    order = np.argsort(np.asarray(dt))
    np.testing.assert_array_equal(sorted_ids, ids_before[order])
    np.testing.assert_allclose(
        music.induced_voltage, legacy.induced_voltage, rtol=1e-9
    )


def test_element_raises_under_mpi(monkeypatch):
    """The Music element itself refuses to set up under MPI (fail-fast)."""
    beam, _, _ = _beam()
    monkeypatch.setattr(
        "blond.physics.impedances.music_algorithm.mpi_is_distributed",
        lambda: True,
    )
    with pytest.raises(NotImplementedError):
        Music.headless(beam=beam, source=_resonator())


def test_multiturn_matches_legacy():
    """A trackable headless Music reproduces legacy single/multi-turn."""
    n = 64
    rng = np.random.default_rng(1)
    dt_np = (rng.random(n) * 1e-9).astype(np.float64)
    dE_np = (rng.standard_normal(n) * 1e6).astype(np.float64)
    intensity = 1e11
    n_turns = 4
    t_rev = 2e-6

    beam = Beam(intensity=intensity, particle_type=proton)
    beam.setup_beam(
        dt=backend.array(dt_np, dtype=backend.float),
        dE=backend.array(dE_np, dtype=backend.float),
    )
    music = Music.headless(beam=beam, source=_resonator())
    # No drift here, so dt stays static across turns. The element reads the
    # elapsed time from the reference clock, so advance it by one t_rev
    # before each subsequent turn (a real simulation does this via drifts).
    for turn in range(n_turns):
        if turn > 0:
            beam.reference.time += t_rev
        music.track(beam=beam)

    # legacy oracle driven by hand with the same fixed dt every turn
    lbeam = SimpleNamespace(dt=dt_np.copy(), dE=dE_np.copy())
    legacy = LegacyMusic(
        lbeam, [R_S, 2 * np.pi * FREQ_R, Q], n, intensity, t_rev=t_rev
    )
    legacy.track_py()
    for _ in range(1, n_turns):
        legacy.track_py_multi_turn()

    np.testing.assert_allclose(
        np.asarray(beam.read_partial_dE()), lbeam.dE, rtol=1e-9
    )


def test_runs_in_full_simulation():
    """Smoke test: Music tracks inside the real main loop and kicks dE."""
    beam = Beam(intensity=1e11, particle_type=proton)
    rng = np.random.default_rng(5)
    dt = backend.array(rng.random(256) * 1e-9, dtype=backend.float)
    dE = backend.array(rng.standard_normal(256) * 1e6, dtype=backend.float)
    beam.setup_beam(dt=dt, dE=dE)
    dE_before = np.asarray(beam.read_partial_dE()).copy()

    ring = Ring(circumference=2 * np.pi * 100)
    rf = SingleHarmonicRFStation(harmonic=1, voltage=0, phi_rf=0)
    drift = DriftSimple(
        orbit_length=2 * np.pi * 100,
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=20.0
        ),
    )
    music = Music(source=_resonator())
    ring.add_elements([rf, drift, music])
    cycle = ConstantMagneticCycle(reference_particle=proton, value=25.92e9)
    sim = Simulation(ring=ring, magnetic_cycle=cycle)
    sim.run_simulation(beams=(beam,), n_turns=3)

    assert music.induced_voltage is not None
    assert len(music.induced_voltage) == beam.n_macroparticles_partial()
    # the beam was actually perturbed by the induced voltage
    assert not np.allclose(np.asarray(beam.read_partial_dE()), dE_before)
    # dt remains sorted after the final turn
    assert np.all(np.diff(np.asarray(beam.read_partial_dt())) >= 0)


def test_const_uses_intensity_and_macroparticles():
    """The MuSiC prefactor folds in intensity / n_macroparticles."""
    beam, _, _ = _beam(n=32, intensity=3e11)
    music = Music.headless(beam=beam, source=_resonator())
    # proton charge == 1, so charge**2 == 1
    expected = -e * R_S * (2 * np.pi * FREQ_R) * 3e11 / (32 * Q)
    np.testing.assert_allclose(music._const, expected, rtol=1e-12)


def test_rejects_non_singly_charged_particles():
    """MuSiC only supports singly-charged particles for now."""
    n, intensity = 32, 3e11
    rng = np.random.default_rng(0)
    dt = backend.array(rng.random(n) * 1e-9, dtype=backend.float)
    dE = backend.array(rng.standard_normal(n) * 1e6, dtype=backend.float)
    beam = Beam(intensity=intensity, particle_type=uranium_29)
    beam.setup_beam(dt=dt, dE=dE)
    with pytest.raises(AssertionError):
        Music.headless(beam=beam, source=_resonator())


def test_each_turn_i_must_be_one():
    """Multi-turn bridging assumes consecutive turns -> each_turn_i == 1."""
    from unittest.mock import Mock

    from blond.core.simulation.simulation import Simulation

    beam, _, _ = _beam()
    music = Music(source=_resonator())
    music.each_turn_i = 2
    sim = Mock(Simulation)
    music.on_init_simulation(simulation=sim)
    with pytest.raises(AssertionError):
        music.on_run_simulation(simulation=sim, beam=beam, n_turns=1)


def _analytical_gaussian_resonator(sigma_t, Q, R_s, omega_r, tau, n_particles):
    """Closed-form induced voltage of a Gaussian bunch in a resonator.

    Independent (non-BLonD) ground truth; see
    `blond.legacy.blond2.impedances.induced_voltage_analytical`.
    """
    alpha = omega_r / (2 * Q)
    ombar = np.sqrt(omega_r**2 - alpha**2)
    A = (alpha * sigma_t**2 - tau + 1j * ombar * sigma_t**2) / (
        np.sqrt(2) * sigma_t
    )
    B = alpha * ombar * sigma_t**2 - ombar * tau
    result = (
        R_s
        * alpha
        / ombar
        * np.e ** (0.5 * (alpha**2 - ombar**2) * sigma_t**2 - alpha * tau)
        * (
            scisp.erfc(A).real * (ombar * np.cos(B) + alpha * np.sin(B))
            + scisp.erfc(A).imag * (alpha * np.cos(B) - ombar * np.sin(B))
        )
    )
    return -n_particles * e * result


def test_matches_analytical_gaussian_resonator():
    """MuSiC reproduces the closed-form Gaussian-bunch resonator voltage.

    Uses a slowly-varying-wake regime (``omega_R * sigma << 1``) so the
    smooth analytical curve is well sampled; the bin-averaged MuSiC voltage
    then matches it to shot-noise level — an independent (non-BLonD) check.
    """
    R_S, freq_R, Q = 1e7, 1e6, 1.0  # omega_R*sigma ~ 0.19
    omega_R = 2 * np.pi * freq_R
    intensity = 1e12
    sigma = 3e-8
    n = 200_000

    rng = np.random.default_rng(1000)
    beam = Beam(intensity=intensity, particle_type=proton)
    beam.setup_beam(
        dt=backend.array(sigma * rng.standard_normal(n), dtype=backend.float),
        dE=backend.array(np.zeros(n), dtype=backend.float),
    )
    music = Music.headless(beam=beam, source=Resonators(R_S, freq_R, Q))
    music.track(beam=beam)  # single turn (Q=1 -> inter-turn wake damped)

    dt = np.asarray(beam.read_partial_dt())
    iv = np.asarray(music.induced_voltage)
    edges = np.linspace(-2.5 * sigma, 2.5 * sigma, 26)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(dt, edges)
    binned = np.array([iv[which == b + 1].mean() for b in range(len(centers))])
    analytical = _analytical_gaussian_resonator(
        sigma, Q, R_S, omega_R, centers, intensity
    )
    peak = np.max(np.abs(analytical))
    rel = np.abs(binned - analytical) / peak
    assert rel.max() < 1e-2, f"max rel err {rel.max():.4f}"
