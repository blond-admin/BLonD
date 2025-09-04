from os import PathLike

import numpy as np

from blond._core.beam.base import BeamBaseClass


def load_beam_data_counterrot_from_file(
    filename: PathLike | str,
    beam: BeamBaseClass,
    beam_counterrot: BeamBaseClass,
) -> None:
    beam.setup_beam(
        dt=np.load(filename)["dt"],
        dE=np.load(filename)["dE"],
    )
    beam_counterrot.setup_beam(
        dt=np.load(filename)["dt"],
        dE=np.load(filename)["dE"],
    )
