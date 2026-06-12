import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt

from blond import Batch, PatternSegment, Train
from blond.cycles.filling_patterns import plot


@pytest.fixture(autouse=True)
def _agg_backend():
    matplotlib.use("Agg")
    yield
    plt.close("all")


class TestPlot:
    def test_smoke_with_conventional_labels(self):
        train = Train(Batch(4, 1), n_copies=2, copy_gap=5)
        ax = plot(train)
        assert ax is not None

    def test_default_labels_tolerate_handbuilt_pattern(self):
        # No 'batch'/'train' labels defined: the conventional defaults
        # must render (as unassigned), not raise.
        segment = PatternSegment(bucket_indices=np.array([0, 2]), n_buckets=5)
        plot(segment)

    def test_explicit_unknown_label_raises(self):
        train = Train(Batch(4, 1), n_copies=2, copy_gap=5)
        with pytest.raises(KeyError, match="injction"):
            plot(train, face_label="injction")

    def test_explicit_valid_label_works(self):
        labeled = Train(Batch(4, 1), n_copies=2, copy_gap=5).with_label(
            "injection"
        )
        plot(labeled, face_label="injection", edge_label="injection")

    def test_color_array_length_mismatch_raises(self):
        train = Train(Batch(4, 1), n_copies=2, copy_gap=5)
        with pytest.raises(ValueError, match="length"):
            plot(train, face=["red", "blue"])
