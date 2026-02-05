# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Class to generate coasting distributions.

Authors
-------
Simon Albright
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond.beam_preparation import base
from blond.core import helpers as core_help
from blond.core.backends.backend import backend
from blond.generals.cupy import no_cupy_import
from blond.generals.distributed import helpers as mpi_help

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class Coasting(base.BeamPreparationRoutine):
    def __init__(
        self,
        n_macroparticles: int,
        energy_bins: ArrayLike,
        energy_profile: ArrayLike,
        start_time: float = 0,
        stop_time: float | None = None,
        energy_offset: float = 0,
        seed: int = 0,
    ):
        super().__init__()

        self._n_macroparticles_local = mpi_help.mpi_local_size(
            core_help.int_from_float_with_warning(
                n_macroparticles, warning_stacklevel=2
            ),
            warning_hint="n_macroparticles",
        )

        self.energy_bins = energy_bins
        self.energy_profile = energy_profile
        self.start_time = start_time
        self.stop_time = stop_time
        self.energy_offset = energy_offset

        self._seed = seed

    def prepare_beam(self, simulation: Simulation, beam: BeamBaseClass):
        super().prepare_beam(simulation, beam)

        rng = mpi_help.mpi_aware_random_generator_cpu(
            seed=(self._seed + 1) if self._seed is not None else None,
            n_forward_per_rank=self._n_macroparticles_local,
        )

        dE = backend.cast_arr_float_if_needed(
            rng.choice(
                self.energy_bins,
                self._n_macroparticles_local,
                p=no_cupy_import.copy_to_cpu(self.energy_profile),
            )
        )

        # Generated distribution is discrete at values defined in
        # self.energy_bins.  An offset is applied to make each bin be
        # sampled uniformly.
        bin_width = self.energy_bins[1] - self.energy_bins[0]
        e_shift = rng.uniform(
            low=0, high=bin_width / 2, size=self._n_macroparticles_local
        )
        # Generated offsets go from 0 -> binwidth/2, multiply every
        # other value by -1 to go from -bin_width/2 -> +bin_width/2.
        # The generation is of the form [low, high), so setting low to
        # -bin_width/2 might slightly bias the result.
        e_shift[::2] *= -1

        dE += e_shift

        if self.stop_time is None:
            circ = simulation.ring.circumference
            particle = beam.particle_type
            self.stop_time = simulation.magnetic_cycle.get_t_rev_init(
                circ, particle
            )

        t_width = self.stop_time - self.start_time

        dt = backend.cast_arr_float_if_needed(
            rng.uniform(low=0, high=t_width, size=self._n_macroparticles_local)
        )
        dt += self.start_time

        beam.setup_beam(dt=dt, dE=dE)
