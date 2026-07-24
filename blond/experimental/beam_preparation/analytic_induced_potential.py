# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Induced voltage/potential from a smooth analytic line density.

Building blocks for the intensity-effect matching iteration (BLonD 2
philosophy): the candidate line density is a *smooth analytic* function
on its own grid — no macroparticles are sampled — and the induced
voltage is computed by **deep copies** of the ring's
:class:`~blond.physics.impedances.base.WakeField` elements, re-initialised
on that smooth profile (the BLonD 2 ``reprocess()`` analog). The induced
potential is then a plain array added to the RF potential well before
the separatrix cut.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.impedances.base import WakeField
from blond.physics.profiles import StaticProfile

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


def clone_wakefields_on_smooth_profile(
    simulation: Simulation,
    time_array: NumpyArray,
) -> tuple[list[WakeField], StaticProfile | None]:
    """
    Deep-copy the ring's wakefields onto a smooth-profile grid.

    Each :class:`WakeField` of the ring is deep-copied (sources and
    solver included, the `Simulation` itself is shared, not copied) and
    re-initialised on a fresh :class:`StaticProfile` spanning
    ``time_array`` — so the candidate line density can have its own
    resolution, independent of the tracked profile (BLonD 2
    philosophy).

    Parameters
    ----------
    simulation
        `Simulation` context manager holding the ring.
    time_array
        Time grid of the candidate line density (bin centres), in [s].
        Must be uniformly spaced.

    Returns
    -------
    wakefield_clones
        Deep-copied wakefields attached to the smooth profile
        (empty list if the ring has none).
    smooth_profile
        The shared smooth profile, or None if the ring has no
        wakefields.
    """
    wakefields = simulation.ring.elements.get_elements(
        WakeField, recursive=False
    )
    if len(wakefields) == 0:
        return [], None

    wakefield_clones = []
    for wakefield in wakefields:
        # Share the Simulation instance instead of deep-copying the
        # whole simulation graph through solver back-references.
        memo = {id(simulation): simulation}
        clone = copy.deepcopy(wakefield, memo)
        clone.track_profile = False
        clone.update_induced_voltage = True
        wakefield_clones.append(clone)
    smooth_profile = reattach_smooth_profile(
        wakefield_clones, simulation, time_array
    )
    return wakefield_clones, smooth_profile


def reattach_smooth_profile(
    wakefield_clones: list[WakeField],
    simulation: Simulation,
    time_array: NumpyArray,
) -> StaticProfile:
    """
    Attach a fresh smooth profile spanning ``time_array`` to the clones.

    The separatrix-cut frame moves while the induced potential
    converges, so the smooth profile is rebuilt every iteration and
    the solvers re-initialised on it — the BLonD 2 ``reprocess()``
    analog (the deep copy itself is done only once).

    Parameters
    ----------
    wakefield_clones
        Wakefields from :func:`clone_wakefields_on_smooth_profile`.
    simulation
        `Simulation` context manager.
    time_array
        Time grid of the candidate line density (bin centres), in [s].
        Must be uniformly spaced.

    Returns
    -------
    smooth_profile
        The freshly attached smooth profile.
    """
    time_step = float(time_array[1] - time_array[0])
    smooth_profile = StaticProfile(
        cut_left=float(time_array[0]) - 0.5 * time_step,
        cut_right=float(time_array[-1]) + 0.5 * time_step,
        n_bins=len(time_array),
    )
    for clone in wakefield_clones:
        # `profile` is a read-only property protecting ring elements;
        # the clone is private to the matching and never tracked, so
        # writing the backing attribute is safe here.
        clone._profile = smooth_profile
        clone.solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=clone
        )
    return smooth_profile


def induced_voltage_from_line_density(
    wakefield_clones: list[WakeField],
    smooth_profile: StaticProfile,
    line_density_values: NumpyArray,
    beam: BeamBaseClass,
) -> NumpyArray:
    """
    Total induced voltage of a smooth candidate line density.

    Parameters
    ----------
    wakefield_clones
        Wakefields from :func:`clone_wakefields_on_smooth_profile`.
    smooth_profile
        The shared smooth profile of the clones.
    line_density_values
        Candidate line density on the smooth-profile grid (any
        normalization; it is scaled to the beam intensity internally).
    beam
        Beam being matched — provides intensity and particle charge
        (works before the beam is populated).

    Returns
    -------
    induced_voltage
        Total induced voltage on the smooth-profile grid, in [V].
    """
    # The profile's histogram lives on the active backend (a CuPy
    # device array under CUDA); convert the host line density at this
    # boundary — CuPy rejects slice-assignment from a host array
    # ("non-scalar numpy.ndarray cannot be used for fill").
    smooth_profile._hist_y[:] = backend.array(
        line_density_values, dtype=backend.float
    )
    total = float(np.sum(line_density_values))
    assert total > 0.0, "The candidate line density is empty."
    # Same semantics as the framework: hist_y * factor = beam fraction
    # per bin (the framework sets 1 / n_macroparticles).
    smooth_profile.hist_y_to_density_factor = 1.0 / total
    smooth_profile.invalidate_cache()

    induced_voltage = np.zeros(len(line_density_values), dtype=float)
    for wakefield_clone in wakefield_clones:
        induced_voltage += copy_to_cpu(
            wakefield_clone.calc_induced_voltage(beam=beam)
        )
    return induced_voltage
