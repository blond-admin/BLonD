# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Support helpers for the LHC blond2-comparison suite.

The blond2 half of every comparison runs the *frozen* legacy code
(``blond.legacy.blond2``) with hard-coded seeds, so its outputs cannot
legitimately change. :func:`blond2_reference` therefore pins those outputs to
committed ``.npz`` files (raw IEEE float64/complex128 bytes -- bit-exact) and
loads them on later runs instead of re-simulating the legacy code, which
removes most of the suite's wall time while every compared value and tolerance
stays byte-identical.

Regenerate the references (e.g. after intentionally changing a comparison's
parameters) with::

    BLOND_REGEN_BLOND2_REFERENCE=1 python -m pytest <this directory>

which re-runs the legacy simulations and overwrites the ``.npz`` files.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

REGEN_ENV_VAR = "BLOND_REGEN_BLOND2_REFERENCE"

_RESOURCES_DIR = Path(__file__).parent / "resources"


def blond2_reference(
    name: str,
    builder: Callable[[], Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """
    Load (or generate and pin) the blond2 reference arrays for one test class.

    On the first run -- or whenever the environment variable
    ``BLOND_REGEN_BLOND2_REFERENCE`` is set to a non-empty value -- the
    ``builder`` runs the full frozen-legacy simulation and its outputs are
    written to ``resources/<name>_blond2_reference.npz`` (compressed;
    ``np.savez`` stores float64/complex128 losslessly as raw IEEE bytes).
    On every later run the arrays are loaded from that file instead, so the
    comparisons see byte-identical reference data without paying for the
    legacy simulation.

    Parameters
    ----------
    name
        Base name of the reference file (one per test class).
    builder
        Zero-argument callable running the blond2 simulation and returning
        the reference arrays as a ``{attribute_name: array}`` mapping.

    Returns
    -------
    dict
        The reference arrays, keyed like the mapping the builder returns.
    """
    path = _RESOURCES_DIR / f"{name}_blond2_reference.npz"
    if path.exists() and not os.environ.get(REGEN_ENV_VAR):
        with np.load(path) as archive:
            return {key: archive[key] for key in archive.files}

    arrays = dict(builder())
    _RESOURCES_DIR.mkdir(exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays
