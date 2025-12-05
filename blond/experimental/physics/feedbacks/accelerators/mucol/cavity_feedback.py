# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Cavity feedback stubs for the muon collider."""

from warnings import warn

import numpy as np

from blond import StaticProfile
from blond.experimental.physics.feedbacks.cavity_feedback import (
    IQCavityFeedback,
)

MINIMUM_QL_FEEDBACK_MODEL = 0.5

class PassiveCavity(IQCavityFeedback):
    r"""
    Passive Cavity, implementing the beam-cavity interaction formulas without a feedback involved.

    Parameters
    ----------
        profile
            profile on which the feedback should act
        R_over_Q
            shunt impedance over quality factor of one cavity [$$\Omega$$]
        Q_L
            Loaded quality factor of one cavity [1]
        f_center
            center frequency of the cavity [Hz]
        f_detuning
            detuning of the cavity [Hz]
        n_cavities
            number of cavities
        generator_current
            given in [A]
        generator_phase
            given in [rad]
        injection_phase
            In :func:`xxxx` the cavity will optimise the phase at injection towards
            this value by adjusting the initial parameters at the beginning of the internal tracking given in [rad]
        injection_voltage
            In :func:`xxxx` the cavity will optimise the voltage at injection towards
            this value by adjusting the initial parameters at the beginning of the internal tracking given in [V]
        harmonic_index
            only the default of 0 is allowed
        n_periods_coarse
            number of RF periods, one coarse grid corresponds to
        section_index
            section which the feedback belongs to
        use_lowpass_filter
            Used in :func:xxx
        name
            If not given, is automatically chosen
    """

    def __init__(self,
                 profile: StaticProfile,  # is this stricly necessary?
                 R_over_Q: float,
                 Q_L: float,
                 f_center: float,
                 f_detuning: float,
                 n_cavities: int,
                 generator_current: float,
                 generator_phase: float = 0,
                 injection_phase: float = -1,
                 injection_voltage: float = -1,
                 harmonic_index: int = 0,
                 n_periods_coarse: int = 1,
                 section_index: int = 0,
                 use_lowpass_filter: bool = False,
                 name: str | None = None) -> None:
        if harmonic_index != 0:
            raise NotImplementedError("harmonic indices other than 0 are not supported with this module")

        assert R_over_Q >= 0, "R_over_Q must be >= 0"
        self.R_over_Q = R_over_Q

        assert Q_L >= MINIMUM_QL_FEEDBACK_MODEL, "Q_L must be >= 0.5"
        self.Q_L = Q_L

        assert f_center >= 0, "f_center must be >= 0"  # TODO: does this make sense here?
        self.f_center = f_center

        assert f_detuning >= 0, "fset must be >= 0"
        self.f_detuning = f_detuning
        self.omega_center = 2 * np.pi * self.f_center - 2 * np.pi * self.f_detuning

        assert n_cavities > 0, "n_cavities must be > 0"
        self.n_cavities = n_cavities

        self.generator_current = generator_current
        self.generator_phase = generator_phase
        self.injection_phase = injection_phase
        self.injection_voltage = injection_voltage

        if use_lowpass_filter:
            warn("lowpass filter is not used in this class", stacklevel=2)

        super().__init__(profile=profile,
                         n_cavities=n_cavities,
                         section_index=section_index,
                         # TODO: this should not be necessary or? The parent cavity already has this information
                         name=name,
                         n_periods_coarse=n_periods_coarse,
                         harmonic_index=harmonic_index)
