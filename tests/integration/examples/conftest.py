import matplotlib.pyplot as plt
import pytest


def _pause_without_sleep(interval: float) -> None:
    """Draw the active figure like `plt.pause`, but skip the waiting."""
    if plt.get_fignums() and plt.gcf().stale:
        plt.gcf().canvas.draw_idle()


@pytest.fixture(autouse=True)
def pause_without_sleep(monkeypatch):
    """Keep the examples' `plt.pause` animations from sleeping in tests.

    With the non-interactive test backend `plt.pause` renders the figure
    and then only waits, which added seconds of idle time to the examples.
    """
    monkeypatch.setattr(plt, "pause", _pause_without_sleep)
