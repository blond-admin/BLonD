import os
import unittest

import numpy as np

from blond import Beam, copy_to_cpu, proton
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
        beam.setup_beam(
            dt=np.linspace(-50e-9, 50e-9, num=100),
            dE=np.linspace(-50e9, 50e9, num=100),
        )

        beam_CR = Beam(
            intensity=2, particle_type=proton, is_counter_rotating=False
        )

        copy_beam_data_from_other_beam(beam_CR, beam)

        assert np.allclose(
            copy_to_cpu(beam._dE.array_local),
            copy_to_cpu(beam_CR._dE.array_local),
        )
        assert np.allclose(
            copy_to_cpu(beam._flags.array_local),
            copy_to_cpu(beam_CR._flags.array_local),
        )
        assert np.allclose(
            copy_to_cpu(beam._ids.array_local),
            copy_to_cpu(beam_CR._ids.array_local),
        )
        assert np.allclose(
            copy_to_cpu(beam._dt.array_local),
            copy_to_cpu(beam_CR._dt.array_local),
        )
        assert beam.intensity == beam_CR.intensity

        beam._is_distributed = True

        with self.assertRaisesRegex(
            RuntimeError, "Copying is not supported with distributed beams."
        ):
            copy_beam_data_from_other_beam(beam_CR, beam)
