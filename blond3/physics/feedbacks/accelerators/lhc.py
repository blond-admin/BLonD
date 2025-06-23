from ..base import LocalFeedback, GlobalFeedback
from ...cavities import SingleHarmonicCavity, MultiHarmonicCavity
from ...profiles import ProfileBaseClass


class LhcBeamFeedBack(GlobalFeedback):
    def __init__(self, profile: ProfileBaseClass, section_index: int = 0):
        super().__init__(profile=profile, section_index=section_index)


class LhcRfFeedback(LocalFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavity: SingleHarmonicCavity | MultiHarmonicCavity,
        section_index: int = 0,
    ):
        super().__init__(profile=profile, cavity=cavity, section_index=section_index)
