import matplotlib.pyplot as plt
import pytest


def _skip(*args, **kwargs) -> None:
    """Do nothing in place of a pyplot call that renders or waits."""


@pytest.fixture(autouse=True)
def skip_plot_rendering(monkeypatch):
    """Keep the examples from rendering, saving or animating their plots.

    The example tests only check that the scripts run. Drawing the figures
    and the `plt.pause` waiting of the animations is wasted time; the
    BLonD plotting methods the examples call have their own unit tests.
    """
    for name in ("draw", "pause", "savefig", "show"):
        monkeypatch.setattr(plt, name, _skip)
    yield
    plt.close("all")
