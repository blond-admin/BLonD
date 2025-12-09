# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions to initialize the beam.

Authors
-------
Leonard Thiele
Simon Lauber
"""

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
    loaded_dict = np.load(filename)
    beam.setup_beam(
        dt=loaded_dict["dt"],
        dE=loaded_dict["dE"],
    )
    beam_counterrot.setup_beam(
        dt=loaded_dict["dt"],
        dE=loaded_dict["dE"],
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
    )
