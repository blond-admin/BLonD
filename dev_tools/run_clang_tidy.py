"""Run clang-tidy on the C++ and CUDA backends (checks in `.clang-tidy`).

Lints exactly the sources `blond/core/backends/cpp/compile.py` and
`blond/core/backends/cuda/compile.py` compile, the C++ with the parallel
(OpenMP) code paths enabled. Needs `g++` for the standard headers and
clang-tidy, pinned in the `dev` extra (`pip install -e ".[dev]"`).

The CUDA sources are parsed by clang's own CUDA front end, which needs
the CUDA headers but no GPU. They are taken from `--cuda-path`,
`$CUDA_PATH`, the toolkit of the `nvcc` on `PATH`, or else the NVIDIA
header wheels (`pip install nvidia-cuda-runtime-cu12 nvidia-curand-cu12
nvidia-cuda-nvcc-cu12 nvidia-cuda-cccl-cu12`). Without any of them the
CUDA sources are skipped, unless `--require-cuda` is given.

Usage::

    python dev_tools/run_clang_tidy.py               # both backends
    python dev_tools/run_clang_tidy.py --diff blonder  # lines changed since

Exits non-zero if any finding is reported.
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPP_DIR = ROOT / "blond" / "core" / "backends" / "cpp"
CUDA_DIR = ROOT / "blond" / "core" / "backends" / "cuda"
# The kernel call records header (deferred specials), included
# by both the C++ and the CUDA sources.
RECORDS_INCLUDE = f"-I{ROOT / 'blond' / 'core' / 'backends' / 'deferred'}"
COMPILER_FLAGS = [
    "-std=c++11",
    "-D_USE_MATH_DEFINES",
    "-fopenmp",
    "-DPARALLEL",
    RECORDS_INCLUDE,
]
# Device code only: the kernels have no host side. `-nocudalib` skips
# libdevice, which only matters for code generation. The architecture is
# arbitrary: no kernel branches on it.
CUDA_FLAGS = [
    "-x",
    "cuda",
    "--cuda-device-only",
    "--cuda-gpu-arch=sm_75",
    "-nocudalib",
    RECORDS_INCLUDE,
]
# Header wheels that together provide what `kernels.cu` includes.
CUDA_HEADER_WHEELS = ("cuda_runtime", "curand", "cuda_nvcc", "cuda_cccl")


def cpp_sources() -> list[Path]:
    """Return the sources listed in `cpp_files` of the C++ compile script."""
    tree = ast.parse((CPP_DIR / "compile.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == (
            "cpp_files"
        ):
            return [CPP_DIR / name for name in ast.literal_eval(node.value)]
    raise RuntimeError("`cpp_files` not found in compile.py")


def cuda_sources() -> list[Path]:
    """Return the sources listed in `cuda_files` of the CUDA compile script.

    Each entry is `os.path.join(_basepath, "<name>")`: take the name.
    """
    tree = ast.parse((CUDA_DIR / "compile.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == (
            "cuda_files"
        ):
            return [
                CUDA_DIR / ast.literal_eval(element.args[-1])
                for element in node.value.elts
            ]
    raise RuntimeError("`cuda_files` not found in compile.py")


def cuda_path(explicit: str | None, tmp_dir: Path) -> Path | None:
    """Return a CUDA toolkit root for clang, or None if none is found.

    Without a toolkit, one is assembled in `tmp_dir` from the NVIDIA
    header wheels: clang only needs `include/` and an (empty) `bin/`.
    """
    if explicit:
        return Path(explicit)
    if os.environ.get("CUDA_PATH"):
        return Path(os.environ["CUDA_PATH"])
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    try:
        import nvidia  # noqa: PLC0415 (namespace package of the wheels)
    except ImportError:
        return None
    include_dirs = [
        Path(root) / wheel / "include"
        for root in nvidia.__path__
        for wheel in CUDA_HEADER_WHEELS
    ]
    include_dirs = [path for path in include_dirs if path.is_dir()]
    if len(include_dirs) < len(CUDA_HEADER_WHEELS):
        return None
    toolkit = tmp_dir / "cuda"
    (toolkit / "bin").mkdir(parents=True)
    for include_dir in include_dirs:
        shutil.copytree(include_dir, toolkit / "include", dirs_exist_ok=True)
    return toolkit


def omp_include_dir(tmp_dir: Path) -> list[str]:
    """Expose only gcc's `omp.h` to clang.

    Adding gcc's whole internal include dir would shadow clang's own
    builtin headers (`stddef.h`, intrinsics) and break parsing.
    """
    omp_header = subprocess.run(
        ["g++", "-print-file-name=include/omp.h"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not Path(omp_header).is_file():
        return []
    shutil.copy(omp_header, tmp_dir)
    return ["-isystem", str(tmp_dir)]


def changed_lines(base: str) -> list[dict]:
    """Return a clang-tidy `--line-filter` for lines changed since `base`."""
    diff = subprocess.run(
        ["git", "diff", "-U0", base, "--", str(CPP_DIR), str(CUDA_DIR)],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout
    ranges: dict[str, list[list[int]]] = {}
    file_name = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            file_name = None if line == "+++ /dev/null" else line[6:]
        elif line.startswith("@@") and file_name is not None:
            start, _, count = re.search(r"\+(\d+)(,(\d+))?", line).groups()
            n_lines = 1 if count is None else int(count)
            if n_lines:
                first = int(start)
                # clang-tidy matches names by suffix, so `kick.cpp` would
                # also match `linear_interp_kick.cpp`: use absolute paths.
                ranges.setdefault(str(ROOT / file_name), []).append(
                    [first, first + n_lines - 1]
                )
    return [{"name": name, "lines": lines} for name, lines in ranges.items()]


def main() -> int:
    """Run clang-tidy and print the findings with a per-check summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--diff",
        metavar="BASE",
        help="only report findings on lines changed since git ref BASE",
    )
    parser.add_argument("--clang-tidy", default="clang-tidy")
    parser.add_argument(
        "--cuda-path", help="CUDA toolkit root (default: auto-detect)"
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail instead of skipping CUDA when no CUDA headers are found",
    )
    args = parser.parse_args()

    tidy_args = [args.clang_tidy, "--quiet"]
    if args.diff:
        line_filter = changed_lines(args.diff)
        if not line_filter:
            print(f"No C++/CUDA backend changes since {args.diff}.")
            return 0
        tidy_args.append("--line-filter=" + json.dumps(line_filter))

    with tempfile.TemporaryDirectory() as tmp_dir:
        compiler_flags = COMPILER_FLAGS + omp_include_dir(Path(tmp_dir))
        if sys.platform == "win32":
            compiler_flags.append("--target=x86_64-w64-mingw32")
        jobs = [(source, compiler_flags) for source in cpp_sources()]

        toolkit = cuda_path(args.cuda_path, Path(tmp_dir))
        if toolkit is not None:
            cuda_flags = [*CUDA_FLAGS, f"--cuda-path={toolkit}"]
            jobs += [(source, cuda_flags) for source in cuda_sources()]
        elif args.require_cuda:
            print("No CUDA headers found (see --help).", file=sys.stderr)
            return 1
        else:
            print(
                "No CUDA headers found, skipping the CUDA backend "
                "(see --help).",
                file=sys.stderr,
            )

        def run(job: tuple[Path, list[str]]) -> str:
            source, flags = job
            return subprocess.run(
                [*tidy_args, str(source), "--", *flags],
                capture_output=True,
                text=True,
                check=False,
                cwd=ROOT,
            ).stdout

        with ThreadPoolExecutor() as pool:
            outputs = list(pool.map(run, jobs))

    # One block per finding, its notes included. A header's findings
    # repeat for every source that includes it: dedupe.
    finding_start = r"^.+:\d+:\d+: (?:warning|error): .*\[(?:[\w.,-]+)\]$"
    findings: dict[str, list[str]] = {}
    for output in outputs:
        for block in re.split(f"\n(?={finding_start})", output, flags=re.M):
            match = re.match(finding_start, block.strip(), flags=re.M)
            if match:
                checks = match.group(0).rsplit("[", 1)[1].rstrip("]")
                findings.setdefault(block.strip(), checks.split(","))

    for block in findings:
        print(block)
    per_check = Counter(
        check for checks in findings.values() for check in checks
    )
    print(f"\n{len(findings)} clang-tidy findings")
    for check, count in per_check.most_common():
        print(f"{count:6d}  {check}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
