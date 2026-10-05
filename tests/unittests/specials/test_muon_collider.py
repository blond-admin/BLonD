import os
import unittest

import numpy as np

from blond import Beam, proton
from blond.core.backends.mpi_distributed.distributed_array import (
    DistributedArray,
)
from blond.specifics.muon_collider.beam_preparation import (
    copy_beam_data_from_other_beam,
    load_beam_coordinates_counterrot_from_file,
    load_beam_coordinates_from_file,
)
from blond.testing.backend_testing import BLonDTestCase


class TestBeamPreparationMuCol(BLonDTestCase):
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
        beam._dt = DistributedArray(np.linspace(-50e-9, 50e-9, num=100))
        beam._dE = DistributedArray(np.linspace(-50e9, 50e9, num=100))
        beam._flags = DistributedArray(np.ones(100, dtype=np.int32))
        beam._ids = DistributedArray(np.arange(100))

        beam_CR = Beam(
            intensity=2, particle_type=proton, is_counter_rotating=False
        )

        copy_beam_data_from_other_beam(beam_CR, beam)

        for name in ("_dt", "_dE", "_flags", "_ids"):
            np.testing.assert_allclose(
                getattr(beam, name).array_local,
                getattr(beam_CR, name).array_local,
            )
        self.assertEqual(beam.intensity, beam_CR.intensity)

        beam._is_distributed = True

        with self.assertRaisesRegex(
            RuntimeError, "Copying is not supported with distributed beams."
        ):
            copy_beam_data_from_other_beam(beam_CR, beam)
