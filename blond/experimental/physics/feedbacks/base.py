# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from blond.physics.cavities import (
    MultiHarmonicRFStation,
    SingleHarmonicRFStation,
)
from blond.physics.feedbacks.base import FeedbackBaseClass
from blond.physics.profiles import ProfileBaseClass


# TODO: Remove
class GroupedFeedback(FeedbackBaseClass):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavities: list[SingleHarmonicRFStation | MultiHarmonicRFStation],
        section_index: int = 0,
        name: str | None = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.profile = profile
        self.cavities = cavities
