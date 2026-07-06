# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Filling patterns and schemes, and bunch trains definition."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from blond.core.beam.beams import Beam


class FillingPattern:
    """
    Class defining a filling pattern.

    Parameters
    ----------
    harmonic
        Main RF harmonic number, defines number of buckets.
    bitmask
        The bitmask defining the filling pattern, by default None.
        If None, the pattern will be created as all zeroes.
    """

    def __init__(self, harmonic: int, bitmask: list[int] | None = None):

        self.bitmask = [0 for _ in harmonic] if bitmask is None else bitmask

        if len(self.bitmask) != harmonic:
            raise ValueError(
                "The bitmask must have as many entries as there are buckets"
            )

    def create_filling_scheme(
        self, train_starts: Iterable[int]
    ) -> FillingScheme:
        """
        Convert the filling pattern to a FillingScheme.

        Parameters
        ----------
        train_starts
            Which bucket numbers will be filled by the first bunch of
            each train.
        """
        ...

    def create_sparse_profile(self, n_bins: int, width: float):
        """
        Create a sparse profile matching the filling pattern.

        Parameters
        ----------
        n_bins
            How many bins each profile should have.
        width
            The width of each profile in seconds.
        """
        ...


class FillingScheme:
    """
    Class defining the trains of a filling scheme.

    Parameters
    ----------
    trains
        The bunch trains that make up this filling scheme.
    """

    def __init__(self, trains: Iterable[BunchTrain]):
        self.trains = trains

    def create_filling_pattern(self) -> FillingPattern:
        """Convert the filling scheme to a filling pattern."""
        ...


class BunchTrain:
    """
    Class defining the bunch parameters for a train.

    Parameters
    ----------
    bucket_numbers
        The bucket numbers (relative to the first bunch) of this train.
    intensities
        The intensity of each bunch.
    emittances
        The longitudinal emittance of each bunch.
    """

    def __init__(
        self,
        bucket_numbers: Iterable[int],
        intensities: Iterable[float],
        emittances: Iterable[float],
    ):

        self.bucket_numbers = bucket_numbers
        self.intensities = intensities
        self.emittances = emittances

    def match_beam(self) -> Beam:
        """Create a new beam object with the defined train parameters."""
        ...
