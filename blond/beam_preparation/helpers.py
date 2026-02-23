# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions for beam creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blond import Beam


def make_multibunch_beam(
    beam: Beam, n_times: int, t_distance: float, common_offset: float = 0.0
) -> Beam:
    """
    Add a bunch repeatedly with different time offset.

    Parameters
    ----------
    beam
        The beam object that is used as a reference.
    n_times
        Number of times the beam should be repeatedly added.
    t_distance
        Distance between each beam that is added, in [s].
    common_offset
        Offset that is applied to all added bunches equally, in[s].

    Returns
    -------
    full_beam
        Beam with many ``dt``-shifted copies of the input beam.

    Examples
    --------
    >>> from blond import make_multibunch_beam, Beam, proton
    >>>
    >>> beam = Beam(
    ...     intensity=1, particle_type=proton
    ... )
    >>> beam.setup_beam(dt=[1, 2, 3], dE=[1e3, 2e3, 3e3])
    >>> beam = make_multibunch_beam(
    ...     beam=beam,
    ...     n_times=3,
    ...     t_distance=222,
    ...     common_offset=111,
    ... )
    """
    from blond import Beam, backend

    assert beam.is_set_up(), (
        "Please set up beam correctly, e.g. using ``beam.setup_beam(...)``."
    )
    full_beam = Beam(
        intensity=n_times * beam.intensity,
        particle_type=beam.particle_type,
        is_counter_rotating=beam.is_counter_rotating,
    )
    # np.repeat([1,2], 2)
    # array([1, 1, 2, 2])
    full_dE = backend.repeat(beam._dE.array_local, n_times)

    full_dt = backend.repeat(beam._dt.array_local, n_times)
    for i in range(n_times):
        t_offset = t_distance * i + common_offset
        sel = slice(i, None, n_times)
        full_dt[sel] += t_offset

    full_beam.setup_beam(
        dt=full_dt,
        dE=full_dE,
        mpi_mode="all-ranks",
    )
    return full_beam
