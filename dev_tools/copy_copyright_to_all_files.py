"""helper functions to add the copyright notice to all files."""

import os
from pathlib import Path


def insert_copyright_notes():
    """Adds the copyright notice to all files."""
    ROOT = Path("../blond/").resolve()
    assert ROOT.exists(), str(ROOT)
    with open("copyright_notice.txt") as file:
        text_py = file.read() + "\n"
    text_cpp = text_py.replace("#", r"//")
    text_fortran = text_py.replace("#", r"!")

    for dirpath, _, filenames in os.walk(ROOT):
        if "legacy" in dirpath:
            continue
        for name in filenames:
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
                    with open(full_path, "r+", encoding="utf-8") as f:
                        content = f.read()
                        if "copyright" in content.lower():
                            print("Skipped:")
                            print(f"{full_path}:1")
                            continue
                        f.seek(0)
                        if is_python_file:
                            f.write(text_py + content)
                        elif is_cpp_file:
                            f.write(text_cpp + content)
                        elif is_fortran_file:
                            f.write(text_fortran + content)
                        else:
                            raise RuntimeError()

                        print("Updated:")
                        print(f"{full_path}:1")


if __name__ == "__main__":
    insert_copyright_notes()
