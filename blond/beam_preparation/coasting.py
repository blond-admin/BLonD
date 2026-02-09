# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Class to generate coasting distributions.

Notes
-----
Authors:
S. Albright
"""

from __future__ import annotations

import warnings
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
    """
    Routines to generate a coasting-like beam distribution.

    Generate a beam with given energy distribution and uniform time
    distribution.  By default, the beam will be generated from 0 to
    t_rev, but different start and stop times can be specified.  An
    energy offset, either constant or time-varying can be optionally
    be added as well.

    Parameters
    ----------
    n_macroparticles
        Number of macroparticles to be generated.
    energy_bins
        The energy bins of the required energy distribution, in [eV].
    energy_profile
        The required energy distribution corresponding to `energy_bins`.
    start_time
        The start time of the distribution, in [s].
    stop_time
        The stop time of the distribution, in [s].
    energy_offset
        The energy offset to be applied after generating the
        distribution.
        If this is a float, a global offset is introduced, in [eV].
        If this an array, it should take the form [time, energy] and
        will be interpolated along the generated distribution.  The
        units are [s, eV].
    seed
        The seed for the random generator.
    """

    def __init__(
        self,
        n_macroparticles: int,
        energy_bins: ArrayLike,
        energy_profile: ArrayLike,
        start_time: float = 0,
        stop_time: float | None = None,
        energy_offset: float | ArrayLike = 0,
        seed: int | None = None,
    ):
        super().__init__()

        self._n_macroparticles_local = mpi_help.mpi_local_size(
            core_help.int_from_float_with_warning(
                n_macroparticles, warning_stacklevel=2
            ),
            warning_hint="n_macroparticles",
        )

        self.energy_bins = energy_bins

        # Automatically cast energy profile to the correct array type
        # to allow for any ArrayLike to work with any backend.
        energy_profile = backend.cast_arr_float_if_needed(energy_profile)
        profile_sum = backend.sum(energy_profile)

        if profile_sum != 1:
            warnings.warn(
                "Energy profile does not sum to 1 and will be"
                " automatically normalised.",
                stacklevel=2,
            )
            energy_profile /= profile_sum

        self.energy_profile = energy_profile

        if (stop_time is not None) and (stop_time < start_time):
            raise ValueError(
                "`start_time` must be less than `stop_time`,"
                f" but got {start_time=} and {stop_time=}."
            )

        self.start_time = start_time
        self.stop_time = stop_time
        self.energy_offset = backend.cast_arr_float_if_needed(energy_offset)

        if self.energy_offset.shape == ():
            self.energy_offset = float(self.energy_offset)

        self._seed = seed

    def prepare_beam(self, simulation: Simulation, beam: BeamBaseClass):
        """
        Populate the beam with the defined distribution.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond.core.beam.beam.Beam` object.
        """
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

        # Set stop time to t_rev if not defined
        if self.stop_time is None:
            circ = simulation.ring.circumference
            particle = beam.particle_type
            self.stop_time = simulation.magnetic_cycle.get_t_rev_init(
                circ, particle
            )

        dt = backend.cast_arr_float_if_needed(
            rng.uniform(
                low=self.start_time,
                high=self.stop_time,
                size=self._n_macroparticles_local,
            )
        )

        if isinstance(self.energy_offset, backend.ndarray):
            dE += backend.interp(
                dt, self.energy_offset[0], self.energy_offset[1]
            )
        else:
            dE += self.energy_offset

        beam.setup_beam(dt=dt, dE=dE, mpi_mode="all-ranks")
