"""Init-ordering (``@requires``) regression tests for the cavity feedback.

The simulation's init-ordering walk collects ``requires`` metadata from
every method found along the MRO (see ``blond.core.ring.helpers``). The
timing class overrides ``on_run_simulation`` without calling ``super()``,
so the decorated base method is dead code: the override that actually
runs must carry its own ``@requires`` metadata, otherwise deleting the
base method would silently drop the init-ordering constraint.
"""

from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedbackTimingClass,
)


def test_timing_on_run_simulation_carries_own_requires() -> None:
    """The executed override itself declares its init-ordering deps."""
    on_run_simulation = IQCavityFeedbackTimingClass.__dict__[
        "on_run_simulation"
    ]
    assert getattr(on_run_simulation, "requires", None) == [
        "RFStationBaseClass",
        "BeamBaseClass",
    ]
