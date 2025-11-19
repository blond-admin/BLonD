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
