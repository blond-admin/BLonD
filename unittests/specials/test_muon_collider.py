import os
import unittest

import numpy as np

from blond import Beam, proton
from blond.specifics.muon_collider.beam_preparation import (
    copy_beam_data_from_other_beam,
    load_beam_coordinates_counterrot_from_file,
    load_beam_coordinates_from_file,
)


class TestBeamPreparationMuCol(unittest.TestCase):
    def test_load_beam_coordinates_counterrot_from_file(self):
        beam = Beam(
            intensity=1, particle_type=proton, is_counter_rotating=True
        )

        beam_cr = Beam(
            intensity=1, particle_type=proton, is_counter_rotating=True
        )

        filename = "testfile.npz"
        dt = np.linspace(-50e-9, 50e-9, num=100)
        dE = np.linspace(-50e9, 50e9, num=100)

        np.savez(filename, dt=dt, dE=dE, allow_pickle=True)

        load_beam_coordinates_counterrot_from_file(
            filename,
            beam,
            beam_cr,
        )

        os.remove(filename)

    def test_load_beam_coordinates_from_file(self):
        beam = Beam(
            intensity=1, particle_type=proton, is_counter_rotating=True
        )

        filename = "testfile.npz"
        dt = np.linspace(-50e-9, 50e-9, num=100)
        dE = np.linspace(-50e9, 50e9, num=100)

        np.savez(filename, dt=dt, dE=dE, allow_pickle=True)

        load_beam_coordinates_from_file(filename, beam)

        os.remove(filename)

    def test_copy_beam_data_from_other_beam(self):
        beam = Beam(
            intensity=1, particle_type=proton, is_counter_rotating=True
        )

        filename = "testfile.npz"
        beam._dt = np.linspace(-50e-9, 50e-9, num=100)
        beam._dE = np.linspace(-50e9, 50e9, num=100)
        beam._flags = np.ones_like(beam._dE)
        beam._ids = np.arange(len(beam._dE))

        beam_CR = Beam(
            intensity=2, particle_type=proton, is_counter_rotating=False
        )

        copy_beam_data_from_other_beam(beam_CR, beam)

        assert np.allclose(beam._dE, beam_CR._dE)
        assert np.allclose(beam._flags, beam_CR._flags)
        assert np.allclose(beam._ids, beam_CR._ids)
        assert np.allclose(beam._dt, beam_CR._dt)
        assert beam.intensity == beam_CR.intensity

        beam._is_distributed = True

        with self.assertRaisesRegex(
            RuntimeError, "Copying is not supported with distributed beams."
        ):
            copy_beam_data_from_other_beam(beam_CR, beam)
