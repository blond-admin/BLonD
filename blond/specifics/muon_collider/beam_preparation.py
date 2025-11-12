"""Helper functions to initialize the beam.

Authors
-------
§
"""

from os import PathLike

import numpy as np

from blond._core.beam.base import BeamBaseClass


def load_beam_coordinates_counterrot_from_file(
    filename: PathLike | str,
    beam: BeamBaseClass,
    beam_counterrot: BeamBaseClass,
) -> None:
    """Load single file to initialize beam coordinates.

    Notes
    -----
    Both beams will be initialized with the same coordinates.

    Parameters
    ----------
    filename
        File that was saved with ``np.save(...)``
        that holds the dt and dE coordinates
    beam
        Simulation :class:`~blond._cycles_core.beam.beam.Beam` object
    beam_counterrot
        Simulation :class:`~blond._cycles_core.beam.beam.Beam` object

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
    """Load single file to initialize beam coordinates.

    Notes
    -----
    Both beams will be initialized with the same coordinates.

    Parameters
    ----------
    filename
        File that was saved with ``np.save(...)``
        that holds the dt and dE coordinates
    beam
        Simulation :class:`~blond._cycles_core.beam.beam.Beam` object

    """
    loaded_dict = np.load(filename)
    beam.setup_beam(
        dt=loaded_dict["dt"],
        dE=loaded_dict["dE"],
    )
