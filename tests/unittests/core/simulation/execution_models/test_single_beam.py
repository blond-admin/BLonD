import importlib
from unittest.mock import MagicMock, patch

from blond.core.simulation.execution_models.single_beam import (
    MainloopSingleBeam,
)
from blond.testing.backend_testing import BLonDTestCase

MODULE = "blond.core.simulation.execution_models.single_beam"


class TestMainloopSingleBeamProgressbar(BLonDTestCase):
    """Under MPI, only rank 0 shows the progress bar."""

    def _run_mainloop_as_rank(self, rank: int) -> MagicMock:
        helpers = importlib.import_module(
            "blond.core.backends.mpi_distributed.helpers"
        )
        simulation = MagicMock()
        simulation.turn_counter.value = 0
        simulation._ring.elements.elements = []
        simulation._sanitize_callbacks.return_value = []
        beams = (MagicMock(),)
        with (
            patch.object(helpers, "MPI_RANK", rank),
            patch(f"{MODULE}.tqdm", side_effect=lambda it, **_: it) as tqdm,
        ):
            MainloopSingleBeam().mainloop(
                simulation=simulation,
                beams=beams,
                n_turns=0,
                show_progressbar=True,
            )
        return tqdm

    def test_root_rank_shows_progressbar(self):
        self._run_mainloop_as_rank(rank=0).assert_called_once()

    def test_non_root_rank_hides_progressbar(self):
        self._run_mainloop_as_rank(rank=1).assert_not_called()
