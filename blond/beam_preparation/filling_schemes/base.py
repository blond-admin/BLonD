# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Utility functions base class for filling schemes."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray

    NumpyArray = NDArray[Any]


class FillingScheme:
    """
    Utility object for filling schemes.

    Parameters
    ----------
    filling_scheme
        Array with entries to indicate which bukcet should be filled,
        e.g. ``[1,0,0,1,]``.
    harmonic
        Harmonic that this filling scheme is made for.
    """

    def __init__(
        self,
        filling_scheme: NumpyArray,
        harmonic: int | None = None,
    ):
        self.filling_scheme = self._resize(filling_scheme, harmonic).astype(
            bool
        )
        assert self.harmonic == harmonic

    def _resize(
        self, filling_scheme: NumpyArray, harmonic: int | None
    ) -> NumpyArray:
        if harmonic is not None:
            if len(filling_scheme) == harmonic:
                pass
            elif len(filling_scheme) < harmonic:
                filling_scheme = self.zero_padd(filling_scheme, harmonic)
            else:
                filling_scheme = filling_scheme[:harmonic]
        return filling_scheme

    @property
    def harmonic(self) -> int:
        """
        Harmonic that this filling scheme is made for.

        Returns
        -------
        harmonic
            Harmonic that this filling scheme is made for.
        """
        return len(self.filling_scheme)

    def fits_harmonic(self, harmonic: int):
        """
        Wether this filling scheme fits the harmonic.

        Parameters
        ----------
        harmonic
            Harmonic that this filling scheme is made for.

        Returns
        -------
        fits_harmonic
            True if the filling scheme fits the harmonic, False otherwise.
        """
        return self.harmonic == harmonic

    @staticmethod
    def zero_padd(array: NumpyArray, n: int):
        """
        Add zeros to the end of the array.

        Parameters
        ----------
        array
            Array to be padded.
        n
            The array length after padding.

        Returns
        -------
        array_new
            The prolongated array.
        """
        return np.concatenate((array, np.zeros(n - len(array))))


class FillingSchemeByTurn(FillingScheme):
    """
    Utility object for filling schemes.

    Parameters
    ----------
    filling_scheme
        Array with entries to indicate which bucket should be filled at
        which turn, e.g. ``[0, np.nan, 0, 1, 1]``.
    harmonic
        Harmonic that this filling scheme is made for.
    """

    def __init__(
        self,
        filling_scheme: NumpyArray,
        harmonic: int | None = None,
    ):
        super().__init__(
            filling_scheme=~np.isnan(filling_scheme),
            harmonic=harmonic,
        )
        at_turn = self._resize(filling_scheme, harmonic)
        assert np.all(at_turn[self.filling_scheme] >= 0), (
            f"Turn indication must be >= 0, not {np.nanmin(at_turn)}"
        )
        self.at_turn = at_turn
