# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from cupy.typing import NDArray as CupyArray
from numpy._typing import NDArray as NumpyArray

from blond import Beam
from blond.core.beam.particle_types import ParticleType
from blond.generals.distributed.distributed_array import DistributedArray

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal


class WeightenedBeam(Beam):
    def __init__(
        self,
        intensity: int | float,
        particle_type: ParticleType,
    ) -> None:
        raise NotImplementedError  # todo
        super().__init__(intensity, particle_type)
        self._weights: DistributedArray | None = None

    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray | None = None,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
        mpi_mode: Literal["root-distributes", "all-ranks"] = "all-ranks",
        weights: NumpyArray | CupyArray = None,
    ) -> None:
        """
        Sets beam array attributes for simulation

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s]
        dE
            Macro-particle energy coordinates, in [eV]
        flags
            Macro-particle flags
        reference_time
            Time of the reference frame (global time), in [s]
        reference_total_energy
            Time of the reference frame (global total energy), in [eV]
        mpi_mode
            - "root-distributes": The array is distributed from the root node to all ranks.
            - "all-ranks":  All ranks setup the beam independently.
        weights
            Weight per macro-particle
        """
        assert weights is not None
        assert len(dt) == len(weights)
        super().setup_beam(dt=dt, dE=dE, flags=flags)
        self._weights = DistributedArray(weights.astype(np.int32))

        if mpi_mode == "root-distributes":
            self._weights.mpi_scatter()
        elif mpi_mode == "all-ranks":
            pass
        else:
            raise NameError(f"Unknown {mpi_mode=}")

    @staticmethod
    def from_beam(beam: Beam):
        pass
