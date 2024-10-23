"""
Script for BLonD developers to autogenerate docker files for different Python versions.
The script takes a template docker file that might be edited as desired.
"""
import os
from pathlib import Path
from warnings import warn


def main():
    PY_VERSIONS = ['3.10', '3.11', '3.12']
    read_py_version = PY_VERSIONS[0]
    folder_read = Path(f"python{without_dot(read_py_version)}-cuda/")

    if not folder_read.exists():
        raise FileNotFoundError(f"{folder_read} not found!")

    for py_version in PY_VERSIONS[1:]:
        populate_folders(folder_read, py_version, read_py_version)


def without_dot(s: str) -> str:
    return s.replace(".", "")


def populate_folders(folder_read: Path, py_version_new: str, py_version_old: str):
    """Copies all files from folder_read to folder_write whilst editing the file contents.

    Parameters
    ----------
    folder_read:
        The source folder to read all files from
    py_version_new
        The target Python version for the newly generated folder and files
    py_version_old
        The Python version of the read folder
    """
    folder_write = Path(f"python{without_dot(py_version_new)}-cuda/")
    Path(folder_write).mkdir(exist_ok=True)

    with open(folder_write / ".gitignore", "w") as f:
        pass

    # Check if the read folder is empty
    files = os.listdir(folder_read)
    if not files:
        warn(f"{folder_read} is empty! No files to copy.")

    for file in files:
        with open(folder_read / file, "r") as f:
            content = f.read()
        content = replace_python_versions(content, py_version_new, py_version_old)
        if file.endswith(".sh"):
            # First line should be the Shebang in .sh file
            insert_index = content.index("\n") + 1
        else:
            insert_index = 0

        content = (content[:insert_index] +
                   f"# This file is auto-generated using {folder_read / file}\n"
                   + content[insert_index:]
                   )

        with open(folder_write / file, "w") as f:
            f.write(content)


def replace_python_versions(content: str, py_version_new: str, py_version_old: str) -> str:
    """Replaces Python version in content

    Parameters
    ----------
    content
        A long text, e.g. the content of a file
    py_version_new:
        The Python version that should be inside the file
    py_version_old:
        The python version that is originally in content and should be replaced

    Returns
    -------
    A string with replaced python version in content
    """
    # replace known strings python310 and python:3.10
    content = content \
        .replace(f"python:{py_version_old}", f"python:{py_version_new}") \
        .replace(f"python{without_dot(py_version_old)}", f"python{without_dot(py_version_new)}")

    # discover unforeseen replacements, warn developer about it
    if py_version_old in content:
        content = content.replace(py_version_old, py_version_new)
        warn(f"Found another part with {py_version_old=}.. replacing it !")
    if without_dot(py_version_old) in content:
        content = content.replace(without_dot(py_version_old), without_dot(py_version_new))
        warn(f"Found another part with {without_dot(py_version_old)=}.. replacing it !")
    return content


if __name__ == "__main__":
    main()
