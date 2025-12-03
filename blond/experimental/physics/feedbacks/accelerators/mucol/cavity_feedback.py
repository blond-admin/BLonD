# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from warnings import warn

from blond.experimental.physics.feedbacks.cavity_feedback import (
    IQCavityFeedback,
)
from blond.physics.profiles import ProfileBaseClass


class PassiveCavity(IQCavityFeedback):

    def __init__(self,
                 profile: ProfileBaseClass,  # is this stricly necessary?
                 harmonic_index: int = 0,
                 section_index: int = 0,
                 use_lowpass_filter: bool = False,
                 name: str | None = None) -> None:
        if use_lowpass_filter:
            warn("lowpass filter is not used in this class")

        super().__init__(profile=profile,
                         _parent_cavity=None,
                         n_cavities=None,
                         section_index=section_index,
                         name=name)
