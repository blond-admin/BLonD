import unittest




@unittest.skip
class TestMain(unittest.TestCase):
    @unittest.skip
    def test_describe_accelerator(self):
        # TODO: implement test for `describe_accelerator`
        self.main.describe_accelerator()

    @unittest.skip
    def test_ready_simulation_and_beam(self):
        # TODO: implement test for `ready_simulation_and_beam`
        self.main.ready_simulation_and_beam(my_accelerator=None)

    @unittest.skip
    def test_run_simulation(self):
        # TODO: implement test for `run_simulation`
        self.main.run_simulation(simulation=None)
