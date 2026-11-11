# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Convenience functions for testing BLonD."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, Mock, patch

from blond.core.beam.beams import Beam
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.core.simulation.simulation import Simulation
from blond.cycles.magnetic_cycle import ConstantMagneticCycle
from blond.physics.drifts import DriftSimple
from blond.physics.impedances.base import WakeField
from blond.physics.profiles import StaticProfile
from blond.physics.rf_station import SingleHarmonicRFStation

simulation_mock = Mock(Simulation)


def new_simulation_mock() -> Mock:
    """
    Create a fresh `Simulation` mock.

    Unlike the shared module-level `simulation_mock`, each call returns an
    independent mock, so state set by one caller cannot leak into another.

    Returns
    -------
    simulation
        Mock with the spec of
        :class:`~blond.core.simulation.simulation.Simulation`.
    """
    return Mock(Simulation)


beam_mock = Mock(Beam)
beam_mock.reference = Mock(ReferenceCoordinates)
static_profile_mock = Mock(StaticProfile)
wakefield_profile_mock = Mock(WakeField)
cycle_const_mock = Mock(ConstantMagneticCycle)

drift_simple_mock = Mock(DriftSimple)
single_harmonic_rf_station_mock = Mock(SingleHarmonicRFStation)


@contextmanager
def identical_mpi_ranks(n_ranks: int = 2) -> Iterator[MagicMock]:
    """
    Run as one of `n_ranks` MPI ranks that all hold the same data.

    Patches the MPI module used by
    `blond.core.backends.mpi_distributed.callables` with a fake whose
    ``COMM_WORLD.Allreduce`` reduces over `n_ranks` copies of the
    buffer, so MPI code paths run without `mpirun`.

    Parameters
    ----------
    n_ranks
        Number of ranks to emulate.

    Yields
    ------
    fake_mpi
        The fake MPI module; its ``COMM_WORLD`` records every call.
    """
    from blond.core.backends.mpi_distributed import callables

    fake_mpi = MagicMock()

    def allreduce(_in_place, buffer, op):
        if op is fake_mpi.SUM:  # MIN and MAX of identical copies: no-op
            buffer *= n_ranks

    fake_mpi.COMM_WORLD.Allreduce.side_effect = allreduce
    with (
        patch.object(callables, "MPI", fake_mpi),
        patch.object(callables, "mpi_is_distributed", return_value=True),
    ):
        yield fake_mpi
