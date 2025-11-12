import os
import unittest

import numpy as np

from blond import Beam, proton
from blond.specifics.muon_collider.beam_preparation import (
    load_beam_coordinates_counterrot_from_file,
    load_beam_coordinates_from_file,
)


class TestBeamPreparationMuCol(unittest.TestCase):
    def test_load_beam_coordinates_counterrot_from_file(self):
        beam = Beam(
            intensity=1,
            particle_type=proton,
            is_counter_rotating=True
        )

        beam_cr = Beam(
            intensity=1,
            particle_type=proton,
            is_counter_rotating=True)

        filename="testfile.npz"
        dt = np.linspace(-50e-9, 50e-9, num=100)
        dE = np.linspace(-50e9, 50e9, num=100)

        np.savez(filename, dt=dt, dE=dE, allow_pickle=True)

        load_beam_coordinates_counterrot_from_file(filename,
                                           beam, beam_cr,)

        os.remove(filename)
    def test_load_beam_coordinates_from_file(self):
        beam = Beam(
            intensity=1,
            particle_type=proton,
            is_counter_rotating=True
        )

        filename="testfile.npz"
        dt = np.linspace(-50e-9, 50e-9, num=100)
        dE = np.linspace(-50e9, 50e9, num=100)

        np.savez(filename, dt=dt, dE=dE, allow_pickle=True)

        load_beam_coordinates_counterrot_from_file(filename, beam)

        os.remove(filename)
