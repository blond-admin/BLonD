class LhcBeamFeedBack(GlobalFeedback):
    def __init__(self, profile: ProfileBaseClass, group: int = 0):
        super().__init__(profile=profile, group=group)


class LhcRfFeedback(LocalFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavity: SingleHarmonicCavity | MultiHarmonicCavity,
        group: int = 0,
    ):
        super().__init__(profile=profile, cavity=cavity, group=group)
