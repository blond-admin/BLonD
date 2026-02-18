import unittest

from blond.utils.turn_counter import get_turn_counter, TurnCounter


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_get_turn_counter(self):
        # TODO: implement test for `get_turn_counter`
        get_turn_counter(name=None)


class TestTurnCounter(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.turn_counter = TurnCounter(name=None)

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test___iter__(self):
        # TODO: implement test for `__iter__`
        self.turn_counter.__iter__()

    @unittest.skip
    def test___next__(self):
        # TODO: implement test for `__next__`
        self.turn_counter.__next__()

    @unittest.skip
    def test___repr__(self):
        # TODO: implement test for `__repr__`
        self.turn_counter.__repr__()

    @unittest.skip
    def test___str__(self):
        # TODO: implement test for `__str__`
        self.turn_counter.__str__()

    @unittest.skip
    def test_initialise(self):
        # TODO: implement test for `initialise`
        self.turn_counter.initialise(max_turns=None, n_sections=None)
