import os
from os import mkdir
from warnings import warn

PY_VERSIONS = ['3.10', '3.11', '3.12']
py_version0 = PY_VERSIONS[0]
_pyversion0 = py_version0.replace(".", "")
folder_read = f"python{_pyversion0}-cuda/"
assert os.path.isdir(folder_read), f"{folder_read} not found!"

for py_version in PY_VERSIONS[1:]:
    _pyversion = py_version.replace(".", "")
    folder_write = f"python{_pyversion}-cuda/"
    if not os.path.isdir(folder_write):
        mkdir(folder_write)
    for file in os.listdir(folder_read):
        with open(folder_read + file, "r") as f:
            content = f.read()
        content = content \
            .replace(f"python:{py_version0}", f"python:{py_version}") \
            .replace(f"python{_pyversion0}", f"python{_pyversion}")  # python310
        if py_version0 in content:
            content = content.replace(py_version0, py_version)
            warn(f"Found another part with {py_version0=}.. replacing it !")
        if _pyversion0 in content:
            content = content.replace(_pyversion0, _pyversion)
            warn(f"Found another part with {_pyversion0=}.. replacing it !")

        content = f"# This file is auto-generated using {folder_read + file}\n" + content
        with open(folder_write + file, "w") as f:
            f.write(content)
        with open(folder_write + ".gitignore", "w") as f:
            pass
