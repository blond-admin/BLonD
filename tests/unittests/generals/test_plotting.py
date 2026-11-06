from blond.generals.plotting_ import DEFAULT_CMAP
from blond.testing.backend_testing import BLonDTestCase


class TestDefaultCmap(BLonDTestCase):
    def test_default_cmap_is_turbo(self) -> None:
        self.assertEqual(DEFAULT_CMAP, "turbo")
