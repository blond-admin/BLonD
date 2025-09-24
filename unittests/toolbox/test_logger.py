import unittest

from blond.toolbox.logger import Logger


class TestLogger(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.logger = Logger(debug=None)

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_disable(self):
        # TODO: implement test for `disable`
        self.logger.disable()
