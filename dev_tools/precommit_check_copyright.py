"""Checks the copyright notice to all files."""

import os
from pathlib import Path

EXCLUDE = "legacy"


def perform_check():
    """Check the copyright notice in all files."""
    this_dir = Path(__file__).parent
    ROOT = (this_dir / "../blond/").resolve()
    assert ROOT.exists(), str(ROOT)
    with open(this_dir / "copyright_notice.txt") as file:
        text_py = file.read() + "\n"
    text_cpp = text_py.replace("#", r"//")
    text_fortran = text_py.replace("#", r"!")

    for dirpath, _, filenames in os.walk(ROOT):
        if EXCLUDE in dirpath:
            continue
        for name in filenames:
            if (
                name == "_version.py"
            ):  # is dynamically written during pip install
                continue
            is_python_file = name.endswith(".py")
            is_cpp_file = (
                name.endswith(".h")
                or name.endswith(".cpp")
                or name.endswith(".cu")
            )
            is_fortran_file = name.endswith(".f90")
            if is_python_file or is_cpp_file or is_fortran_file:
                full_path = os.path.join(dirpath, name)
                if os.path.getsize(full_path) > 0:  # skip empty files
                    with open(full_path, encoding="utf-8") as f:
                        content = f.read()
                    if is_python_file:
                        copyright_message = text_py
                    elif is_fortran_file:
                        copyright_message = text_fortran
                    elif is_cpp_file:
                        copyright_message = text_cpp
                    else:
                        raise RuntimeError()
                    if not content.startswith(copyright_message):
                        raise ValueError(
                            f"{full_path}:1 does not start "
                            f"with a copyright message."
                            f" Execute `dev_tools/copy_copyright_to_all_files.py` "
                            f"to fix this problem."
                        )


if __name__ == "__main__":
    perform_check()
