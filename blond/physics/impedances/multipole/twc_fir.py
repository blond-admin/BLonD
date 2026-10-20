# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
The travelling-wave-cavity wake `MultiPoleSparseSolve` applies by FIR.

Its finite-support wake has no pole-residue representation, so sources
provide it through `SupportsTWCFIRModel` instead of a vector fit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


class SupportsTWCFIRModel(ABC):
    r"""
    Mixin for sources with a finite-support travelling-wave-cavity wake.

    The wake :math:`W(t) = (4 R / \tilde a)(1 - t / \tilde a)
    \cos(\omega_r t)` for :math:`0 < t < \tilde a` has no pole-residue
    representation; `MultiPoleSparseSolve` applies it with the
    `Specials.wake_from_twc_fir` recursion instead.

    See Also
    --------
    blond.physics.impedances.multipole.solver.MultiPoleSparseSolve : The corresponding wakefield solver.
    """

    @abstractmethod  # pragma: no cover
    def get_twc_fir(
        self,
    ) -> tuple[
        NumpyArray | CupyArray, NumpyArray | CupyArray, NumpyArray | CupyArray
    ]:
        """
        Provide the travelling-wave-cavity wake parameters per mode.

        Returns
        -------
        r_shunt
            Shunt impedance per mode, in [Ohm].
        a_tilde
            Wake support (filling) time per mode, in [s].
        omega_r
            Angular resonant frequency per mode, in [rad/s].
        """
        pass


class TravelingWaveCavityTWCFIR(SupportsTWCFIRModel):
    """
    Wake parameters of a travelling-wave cavity for `MultiPoleSparseSolve`.

    Mix-in of :class:`~blond.physics.impedances.sources.TravelingWaveCavity`.
    """

    def get_twc_fir(
        self,
    ) -> tuple[
        NumpyArray | CupyArray, NumpyArray | CupyArray, NumpyArray | CupyArray
    ]:
        """
        Provide the travelling-wave-cavity wake parameters per mode.

        Returns
        -------
        r_shunt
            Shunt impedance per mode, in [Ohm].
        a_tilde
            Wake support (filling) time per mode, in [s].
        omega_r
            Angular resonant frequency per mode, in [rad/s].
        """
        return (
            self.R_S,
            self.a_factor / (2 * np.pi),
            2 * np.pi * self.frequency_R,
        )
