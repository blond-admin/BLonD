import os
import subprocess
import time

from pathlib import Path


def main():
    target = "benchmark_cpp.py"
    print(f"Running {target}")
    branches = ["develop", "performance/rf_volt_comp"]
    this_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    blond_project_dir = this_dir.parent.parent
    compile_file = blond_project_dir / Path("blond/compile.py")
    assert compile_file.is_file()

    assert target.endswith(".py")
    compile_verbose = False
    for branch in branches:
        print(f"\nChecking out branch: {branch}")
        time.sleep(0.1)
        try:
            subprocess.run(["git", "checkout", branch], check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("Uncommited changes")
        time.sleep(0.1)
        print("Compiling...", end="")
        subprocess.run(
            ["python3", str(compile_file), "--optimize", "--parallel", "-gpu"],
            check=True,
            capture_output=False,
            stdout=None if compile_verbose else subprocess.PIPE,
        )
        print("Done!")

        result = subprocess.run(["python3", target], capture_output=True, text=True)
        print(f"{result.stdout[:-1]} on {branch}")  # stdout without last \n


if __name__ == "__main__":
    main()
