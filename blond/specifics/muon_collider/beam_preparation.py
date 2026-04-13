# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Helper functions to initialize the beam.

Notes
-----
Authors:
Leonard Thiele
Simon Lauber
"""

from copy import deepcopy
from os import PathLike

import numpy as np

from blond.core.beam.base import BeamBaseClass


def load_beam_coordinates_counterrot_from_file(
    filename: PathLike | str,
    beam: BeamBaseClass,
    beam_counterrot: BeamBaseClass,
) -> None:
    """
    Load single file to initialize beam coordinates.

    Parameters
    ----------
    filename
        File that was saved with ``np.save(...)``
        that holds the dt and dE coordinates.
    beam
        Simulation :class:`~blond._cycles_core.beam.beam.Beam` object.
    beam_counterrot
        Simulation :class:`~blond._cycles_core.beam.beam.Beam` object.

    Notes
    -----
    Both beams will be initialized with the same coordinates.
    """
    loaded_dict = np.load(filename, allow_pickle=True)
    beam.setup_beam(
        dt=loaded_dict["dt"],
        dE=loaded_dict["dE"],
        mpi_mode="root-distributes",
    )
    beam_counterrot.setup_beam(
        dt=loaded_dict["dt"],
        dE=loaded_dict["dE"],
        mpi_mode="root-distributes",
    )


def load_beam_coordinates_from_file(
    filename: PathLike | str,
    beam: BeamBaseClass,
) -> None:
    """
    Load single file to initialize beam coordinates.

    Parameters
    ----------
    filename
        File that was saved with ``np.save(...)``
        that holds the dt and dE coordinates.
    beam
        Simulation :class:`~blond._cycles_core.beam.beam.Beam` object.

    Notes
    -----
    Both beams will be initialized with the same coordinates.
    """
    loaded_dict = np.load(filename)
    beam.setup_beam(
        dt=loaded_dict["dt"],
        dE=loaded_dict["dE"],
        mpi_mode="root-distributes",
    )


def copy_beam_data_from_other_beam(
    to_beam: BeamBaseClass,
    other_beam: BeamBaseClass,
):
    """
    Copy beam data from one beam to another.

    Parameters
    ----------
    to_beam
        Beam to copy parameters onto.
    other_beam
        Beam to copy parameters from.
    """
    if other_beam._is_distributed:
        raise RuntimeError("Copying is not supported with distributed beams.")

    to_beam._dt = deepcopy(other_beam._dt)
    to_beam._dE = deepcopy(other_beam._dE)
    to_beam._flags = deepcopy(other_beam._flags)
    to_beam._ids = deepcopy(other_beam._ids)

    to_beam.intensity = deepcopy(other_beam.intensity)
    to_beam._is_distributed = False
