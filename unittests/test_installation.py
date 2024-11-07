import os
import pathlib
import shutil
import subprocess
import sys
import unittest
import blond

this_directory = pathlib.Path(__file__).parent.resolve()
blond_path = pathlib.Path(blond.__file__).parent.resolve()



def execute_command(commands):
    print(commands)
    command = f"bash -i -c '{'; '.join(commands)}'"
    prc = subprocess.Popen(
        command,
        cwd=this_directory,
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=True
    )
    prc.communicate()
    assert prc.returncode == 0, f"{prc.returncode=}"



venv_tmp = "venv_tmp"


class TestInstallation(unittest.TestCase):
    def test_setup(self):
        commands = ["pip install --upgrade pip", f'python -m venv {venv_tmp}']
        execute_command(commands)

        commands = [f"source {venv_tmp}/bin/activate",  # activate virtual environment to install fresh blond
                    f"cd ..",  # # change to BLonD folder
                    "pip install .[all]"  # run installation of BLonD
                    ]
        execute_command(commands)

        commands = [
            f"source {venv_tmp}/bin/activate",  # activate virtual environment to install fresh blond
            'python -c "from blond import test; test()"',
        ]
        execute_command(commands)

    def tearDown(self):
        execute_command([f"rm -rf {venv_tmp}"])
