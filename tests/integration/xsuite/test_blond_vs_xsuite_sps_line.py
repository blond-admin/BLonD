# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
``WrapBlond4Xsuite`` inside a *real* SPS lattice, vs a pure-xsuite reference.

Every other BLonD<->xsuite test here uses a trivial one-element
``LineSegmentMap`` line. This one loads a real SPS line (thousands of elements,
six cavities, full transverse optics) so the wrapper is exercised interleaved
with genuine transverse transport and arbitrary element ordering.

Both runs share the lattice and the initial distribution. The reference keeps
the real accelerating cavity (``acta.31637``); the run under test zeroes that
cavity and inserts a ``WrapBlond4Xsuite`` around an equivalent
``SingleHarmonicRFStation`` at the same position. The two must agree on the
longitudinal phase space, and the transverse plane (``x``/``px``/``y``/``py``)
must agree to near machine precision. Unlike the ``LineSegmentMap`` tests,
bit-identity cannot be required here: the BLonD kick and xsuite's cavity kick
differ at ULP level (numpy vs compiled C), and in a real chromatic, dispersive
lattice those ULPs couple into the transverse coordinates.

No energy program is attached: this is a stationary (constant-energy) check, so
the agreement is far tighter than the ramped LHC tests.
"""

from __future__ import annotations

import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")
pytest.importorskip("xpart")

from blond import SingleHarmonicRFStation, copy_to_cpu  # noqa: E402
from blond.interfaces.xsuite.elements.wrap_blond_element import (  # noqa: E402
    WrapBlond4Xsuite,
)
from tests.integration.xsuite._lhc_harness import (  # noqa: E402
    matched_context,
)

# SPS-like RF parameters; the only accelerating cavity in the loaded line is
# ``acta.31637`` (V = 3 MV, lag = 180 deg); the other five are zero-voltage.
SPS_LINE = (
    Path(__file__).parent / "resources/line_no_spacecharge_and_particle.json"
)
P0C = 26e9
RF_VOLTAGE = 3e6
HARMONIC = 4620
CAVITY_NAME = "acta.31637"
N_TURNS = 20
N_PART = 50
SEED = 0

# xsuite ``lag`` is in degrees; the wrapper's convention is
# ``phi_rf [rad] = lag [deg] * pi/180`` (see test_ramp_wrapped_blond_vs_xsuite).
# The real cavity runs at lag = 180 deg, so the BLonD station uses phi_rf = pi.
PHI_RF = np.pi

# Turn-by-turn coordinates, shape ``(n_part, n_turns)`` each.
_Track = namedtuple("_Track", ["zeta", "ptau", "x", "px", "y", "py"])


def _load_sps_line() -> xt.Line:
    line = xt.load(str(SPS_LINE))
    line.set_particle_ref("proton", p0c=P0C)
    return line


def _rf_frequency(line: xt.Line) -> float:
    """Design RF frequency [Hz] = harmonic * f_rev, matching BLonD's design omega."""
    beta0 = float(line.particle_ref.beta0[0])
    return HARMONIC * beta0 * c / line.get_length()


def _initial_distribution() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return {
        "x": rng.uniform(-1e-3, 1e-3, N_PART),
        "px": rng.uniform(-1e-5, 1e-5, N_PART),
        "y": rng.uniform(-2e-3, 2e-3, N_PART),
        "py": rng.uniform(-3e-5, 3e-5, N_PART),
        "zeta": rng.uniform(-2e-2, 2e-2, N_PART),
        "delta": rng.uniform(-1e-4, 1e-4, N_PART),
    }


def _record(line: xt.Line, dist: dict[str, np.ndarray]) -> _Track:
    particles = line.build_particles(**dist)
    line.track(
        particles=particles, num_turns=N_TURNS, turn_by_turn_monitor=True
    )
    rec = line.record_last_track
    return _Track(
        copy_to_cpu(rec.zeta).copy(),
        copy_to_cpu(rec.ptau).copy(),
        copy_to_cpu(rec.x).copy(),
        copy_to_cpu(rec.px).copy(),
        copy_to_cpu(rec.y).copy(),
        copy_to_cpu(rec.py).copy(),
    )


def _run_pure_xsuite(dist: dict[str, np.ndarray]) -> _Track:
    """Reference: the real SPS cavity, frequency pinned to harmonic * f_rev."""
    line = _load_sps_line()
    line[CAVITY_NAME].frequency = _rf_frequency(line)
    line.build_tracker(_context=matched_context())
    return _record(line, dist)


def _run_wrapped_blond(dist: dict[str, np.ndarray]) -> _Track:
    """Under test: real cavity zeroed, a wrapped BLonD station at its position."""
    line = _load_sps_line()
    blond_cavity = SingleHarmonicRFStation.headless(
        section_index=0,
        voltage=RF_VOLTAGE,
        phi_rf=PHI_RF,
        harmonic=HARMONIC,
        circumference=line.get_length(),
        beam_reference_beta=float(line.particle_ref.beta0[0]),
        magnetic_cycle=None,
    )
    line[CAVITY_NAME].voltage = 0.0
    line.insert_element(
        at=CAVITY_NAME,
        element=WrapBlond4Xsuite(element=blond_cavity),
        name="blond_cavity",
    )
    line.build_tracker(_context=matched_context())
    return _record(line, dist)


class TestWrappedBlondInSpsLine(unittest.TestCase):
    """The wrapper must behave inside a real multi-element SPS lattice."""

    def test_matches_pure_xsuite(self):
        """Wrapped BLonD cavity matches the real SPS cavity, transverse untouched."""
        dist = _initial_distribution()

        x = _run_pure_xsuite(dist)
        w = _run_wrapped_blond(dist)

        self.assertEqual(w.zeta.shape, x.zeta.shape)
        np.testing.assert_allclose(w.zeta, x.zeta, atol=1e-9, rtol=0.0)
        np.testing.assert_allclose(w.ptau, x.ptau, atol=1e-12, rtol=0.0)

        # The wrapper only acts on zeta/ptau, but the ptau tolerance above
        # admits ULP-level kick differences (numpy vs compiled C), and
        # dispersion/chromaticity carry those into the transverse plane —
        # observed up to ~4e-15 across FP environments. Bit-identity of the
        # untouched transverse arrays is asserted on a delta-decoupled line
        # in test_ramp_wrapped_blond_vs_xsuite instead.
        np.testing.assert_allclose(w.x, x.x, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(w.px, x.px, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(w.y, x.y, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(w.py, x.py, atol=1e-14, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
