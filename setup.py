"""Installation hook to build the backends.

Notes
-----
This is not conform with the `build_ext` process of `setuptools` (2025),
and thus hacky. Should be reworked when `setuptools` allows.
"""

import os.path
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

this_folder = os.path.dirname(os.path.abspath(__file__))

backends = (
    ("C++ backend", "blond/_core/backends/cpp/", "compile.py"),
    ("CUDA backend", "blond/_core/backends/cuda/", "compile.py"),
    ("Fortran backend", "blond/_core/backends/fortran/", "compile.py"),
)

copy_formats = (
    ".so",
    ".dll",
    ".cubin",
)


def _potentially_compile(
    backend_name: str, compile_file: str, location: str
) -> None:
    """
    Try to compile, catch exceptions as warnings.

    Parameters
    ----------
    backend_name
        Name of backend to print
    compile_file
        File that contains the compilation script
    location
        Location of the compilation script from content-root

    """
    try:
        subprocess.check_call(
            args=["python", compile_file],
            cwd=os.path.join(this_folder, location),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print(f"Compiled {backend_name}..")
    except Exception as exc:
        print(f"Failed to compile {backend_name} with error {exc}..")


def _move_compiled_libs(
    build_lib: str, relative_backend_location: str
) -> None:
    """
    Move DLLs etc. to build directory

    Parameters
    ----------
    build_lib
        Directory of the build
    relative_backend_location
        Path of the backend from content-root

    """
    for root, dirs, files in os.walk(
        os.path.join(this_folder, relative_backend_location)
    ):
        for file in files:
            if any([file.endswith(end) for end in copy_formats]):
                source = os.path.join(root, file)
                destination_dir = os.path.join(
                    build_lib,
                    os.path.relpath(root, this_folder),
                )
                os.makedirs(destination_dir, exist_ok=True)
                destination = os.path.join(destination_dir, file)
                print(f"moving {source} -> {destination}")
                shutil.move(source, destination)


class BuildDLL(build_ext):
    def run(self):
        for backend_name, location, compile_file in backends:
            _potentially_compile(
                backend_name=backend_name,
                compile_file=compile_file,
                location=location,
            )
            if not self.inplace:  # move compiled files only when necessary
                _move_compiled_libs(
                    build_lib=self.build_lib,
                    relative_backend_location=location,
                )

        # Dont execute super().run()
        # to skip trying to compile `_trigger_build_process` extension (which is not possible).
        # super().run() would need to be activated in case of actually adding
        # compiled python-extensions to the module.

        # super().run() # intended to commented out


# Empty extension needed to trigger building
# so that the scripts above are getting executed
trigger_build_process = Extension(
    # the actual build is done via compile.py
    name="_trigger_build_process",
    sources=[],  # intentionally empty, the standard extension builder should build nothing
    optional=True,  # install blond, even if compilation fails
)
setup(
    ext_modules=[
        trigger_build_process,
    ],  # trigger build_ext to run
    cmdclass={
        "build_ext": BuildDLL,
    },
)
