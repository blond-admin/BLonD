# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from blond import backend
from blond.core.base import BeamPhysicsRelevant
from blond.physics.impedances.sources import get_hash

if TYPE_CHECKING:  # pragma: no cover
    from typing import TypeVar

    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation

    T = TypeVar("T")

logger = logging.getLogger(__name__)


class PooledInterpolationKick(BeamPhysicsRelevant):
    def __init__(
        self, section_index: int = 0, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(section_index, name)
        self.buffer_energy_change = {}
        self.buffer_time_axis = {}

    def on_init_simulation(self, simulation: Simulation) -> None:
        self.wipe_buffer()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        self.wipe_buffer()

    def wipe_buffer(self):
        self.buffer_energy_change = {}
        self.buffer_time_axis = {}

    def register(self, time_axis: NumpyArray, energy_change: NumpyArray):
        key = get_hash(time_axis)
        if key in self.buffer_energy_change:
            self.buffer_energy_change[key] += energy_change
        else:
            self.buffer_energy_change[key] = energy_change.copy()
            self.buffer_time_axis[key] = time_axis.copy()

    def _track(self, beam: BeamBaseClass) -> None:
        for key in self.buffer_energy_change.keys():
            voltage = self.buffer_energy_change[key]
            time = self.buffer_time_axis[key]
            backend.specials.change_dE_interpolated(
                dt=beam.read_partial_dt(),
                dE=beam.write_partial_dE(),
                bin_centers=time,
                voltage=voltage,
                charge=beam.particle_type.charge,
                acceleration_kick=0.0,
            )
        self._set_buffer_zero()

    def _set_buffer_zero(self):
        for key in self.buffer_energy_change.keys():
            self.buffer_energy_change[key][:] = 0.0
