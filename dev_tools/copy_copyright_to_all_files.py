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
                copyright_message = get_copyright_message(
                    is_cpp_file,
                    is_fortran_file,
                    is_python_file,
                    text_py,
                )
                full_path = os.path.join(dirpath, name)
                if os.path.getsize(full_path) > 0:  # skip empty files
                    with open(full_path) as f:
                        content = f.read()
                    if not content.startswith(copyright_message):
                        with open(full_path, "w") as f:
                            f.seek(0)
                            if (
                                is_python_file
                                or is_cpp_file
                                or is_fortran_file
                            ):
                                f.write(copyright_message + content)
                            else:
                                raise RuntimeError()


def get_copyright_message(
    is_cpp_file: bool,
    is_fortran_file: bool,
    is_python_file: bool,
    text_py: str,
):
    """
    Get the copyright message in the correct syntax for the different backends.

    Parameters
    ----------
    is_cpp_file
        Whether the file is C++.
    is_fortran_file
        Whether the file is FORTRAN.
    is_python_file
        Whether the file is Python.
    text_py
        The original text message to be converted.

    Returns
    -------
    copyright_message
        The copyright message in the correct syntax for the different backends.

    """
    text_cpp = text_py.replace("#", r"//")
    text_fortran = text_py.replace("#", r"!")
    if is_python_file:
        copyright_message = text_py
    elif is_fortran_file:
        copyright_message = text_fortran
    elif is_cpp_file:
        copyright_message = text_cpp
    else:
        raise RuntimeError()
    return copyright_message


if __name__ == "__main__":
    perform_check()
