import unittest



import unittest


@unittest.skip("Not Implemented")
class TestFunctions(unittest.TestCase):
    def test_my_callback(self):
        from blond3.examples import EX_MuonCollider # NOQA will run the
        # full script. just checking if it crashes