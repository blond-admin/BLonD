from enum import IntEnum

class BeamFlags(IntEnum):
    """Flags that define the beam state."""

    # Please mind that the LOST flag is hardcoded in all backends
    # for loss_box
    LOST = -500  # by convention with XSuite team.
    ACTIVE = 1