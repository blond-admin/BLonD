import unittest

import numpy as np

from blond.examples.scripts.EX_28_Multiturn_sparse_sps import (
    VectorFittedModel,
)


class TestVectorFittedModelPlot(unittest.TestCase):
    """A real pole has no implicit complex conjugate (vector-fitting
    convention): `VectorFittedModel.plot` must reconstruct its frequency
    response without doubling it via a conjugate term."""

    def test_real_pole_not_doubled(self):
        pole = -3e7
        residue = 2.5
        model = VectorFittedModel(
            poles=np.array([pole], dtype=complex),
            residues=np.array([residue], dtype=complex),
        )

        freq = np.linspace(1e5, 1e9, 1000)
        h = model.plot(freq=freq)

        s = 1j * 2 * np.pi * freq
        expected = residue / (s - pole)
        np.testing.assert_allclose(h, expected)

    def test_complex_conjugate_pair_still_doubled(self):
        pole = -2e7 + 1e9j
        residue = 1.0 - 0.5j
        model = VectorFittedModel(
            poles=np.array([pole], dtype=complex),
            residues=np.array([residue], dtype=complex),
        )

        freq = np.linspace(1e5, 1e9, 1000)
        h = model.plot(freq=freq)

        s = 1j * 2 * np.pi * freq
        expected = residue / (s - pole) + np.conjugate(residue) / (
            s - np.conjugate(pole)
        )
        np.testing.assert_allclose(h, expected)


if __name__ == "__main__":
    unittest.main()
