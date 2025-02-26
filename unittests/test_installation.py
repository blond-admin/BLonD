import pathlib
import platform
import subprocess
import sys
import unittest

this_directory = pathlib.Path(__file__).parent.resolve()


def execute_command(commands):
    if platform.system() == 'Windows':
        # For Windows, we use PowerShell to run the commands
        command = f"powershell -Command \"{'; '.join(commands)}\""
    else:
        # For Linux (or Unix-like systems), we can use bash
        command = f"bash -i -c '{'; '.join(commands)}'"
    print(f"Running command in subprocess: {command}")
    # Use subprocess.run to execute the command
    result = subprocess.run(
        command,
        cwd=this_directory,
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=True
    )

    # This will raise a CalledProcessError if the command failed (returns non-zero returncode).
    # If you actually want a specific string returned when failing, using assert as you did before also works fine.
    result.check_returncode()


venv_tmp = "venv_tmp"


class TestInstallation(unittest.TestCase):
    def test_installation_in_venv(self):
        """Integration test: run pip install in a temporary directory and ensure it works."""

        if platform.system() == 'Windows':
            activate_venv = f"{venv_tmp}\\Scripts\\activate"
        else:  # For Linux/Unix-based systems
            activate_venv = f"source {venv_tmp}/bin/activate"
        commands = [f"python -m pip install --upgrade pip",
                    f'python -m venv {venv_tmp}',
                    activate_venv,  # activate virtual environment to install fresh blond
                    f"cd ..",  # # change to BLonD folder
                    "pip install .[all]",  # run installation of BLonD
                    "python -c 'from blond import test; test()'",
                    ]
        execute_command(commands)

    def tearDown(self):
        # Define the command based on the OS
        if platform.system() == 'Windows':
            execute_command([f'Remove-Item "{venv_tmp}" -Recurse -Force'])
        else:
            execute_command([f"rm -rf {venv_tmp}"])  # For Linux/Unix-based systems
