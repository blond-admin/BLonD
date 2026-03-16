# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from typing import TYPE_CHECKING

from blond.physics.feedbacks.base import LocalFeedback
from blond.physics.profiles import StaticProfile

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray

    NumpyArray = NDArray[Any]

    from blond.core.beam.base import BeamBaseClass

# TODO rewrite all docstrings


class IQCavityFeedback(LocalFeedback):
    """
    Base class to design cavity feedbacks.

    Parameters
    ----------
    profile
        Beam profile the feedback acts on
    n_cavities
        Number of cavities the feedback controls
    n_periods_coarse
        Number of periods for the coarse grid
    harmonic_index
        Index of the RF harmonic that should be controlled by the feedback
    use_lowpass_filter
        Whether to apply a lowpass filter when calculating the beam current
    section_index
        # TODO might be removed?
    name
        # TODO might be removed

    Attributes
    ----------
    n_cavities
        Number of cavities the feedback is working on
    use_lowpass_filter
        Apply a low-pass filter to the RF beam current
    harmonic_index
        The harmonic index the cavity feedback is working on
    n_periods_coarse
        Sampling time in the model and the number of samples per turn
    T_s
        xxx # TODO
    n_coarse
        xxx # TODO
    omega_carrier
        xxx # TODO
    omega_rf
        xxx # TODO
    dT
        xxx # TODO
    V_SET
        xxx # TODO
    I_BEAM_COARSE
        xxx # TODO
    I_BEAM_FINE
        xxx # TODO
    V_ANT_COARSE
        xxx # TODO
    V_ANT_FINE
        xxx # TODO
    I_GEN_COARSE
        xxx # TODO
    I_GEN_FINE
        xxx # TODO
    V_corr
        xxx # TODO
    alpha_sum
        xxx # TODO
    phi_corr
        xxx # TODO
    omega_carrier_prev
        xxx # TODO
    T_s_prev
        xxx # TODO
    rf_centers_prev
        xxx # TODO

    """

    # TODO docstring

    def __init__(
        self,
        profile: StaticProfile,
        n_cavities: int,
        n_periods_coarse: int | float,
        harmonic_index: int,
        use_lowpass_filter: bool = False,
        name: str | None = None,
    ):
        """
        Base class to design cavity feedbacks.

        Parameters
        ----------
        profile
            Beam profile the feedback acts on
        n_cavities
            Number of cavities the feedback controls
        n_periods_coarse
            Number of rf periods the coarse grid sampling period corresponds to
        harmonic_index
            Index of the RF harmonic that should be controlled by the feedback
        use_lowpass_filter
            Whether to apply a lowpass filter when calculating the beam current
        section_index
            # TODO migh be removed?
        name
            # TODO might be removed
        """
        super().__init__(
            profile=profile,
            name=name,
        )

        self.V_corr: NumpyArray | None = None
        self.alpha_sum: NumpyArray | None = None
        self.phi_corr: NumpyArray | None = None

        self.gap_voltage_phase: NumpyArray | None = None

    def _track(self, beam: BeamBaseClass) -> None:
        r"""
        Tracking method of the cavity feedback.

        Parameters
        ----------
        beam
            Simulation `Beam` object

        """
        pass
