"""Run clang-tidy on the C++ backend (checks configured in `.clang-tidy`).

Lints exactly the sources `blond/core/backends/cpp/compile.py` compiles,
with the parallel (OpenMP) code paths enabled. Needs `g++` for the
standard headers and `pip install clang-tidy==22.1.8`.

Usage::

    python dev_tools/run_clang_tidy.py               # whole backend
    python dev_tools/run_clang_tidy.py --diff blonder  # lines changed since

Exits non-zero if any finding is reported.
"""

import argparse
import ast
import json
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
COMPILER_FLAGS = [
    "-std=c++11",
    "-D_USE_MATH_DEFINES",
    "-fopenmp",
    "-DPARALLEL",
]


def compiled_sources() -> list[Path]:
    """Return the sources listed in `cpp_files` of the C++ compile script."""
    tree = ast.parse((CPP_DIR / "compile.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == (
            "cpp_files"
        ):
            return [CPP_DIR / name for name in ast.literal_eval(node.value)]
    raise RuntimeError("`cpp_files` not found in compile.py")


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
        ["git", "diff", "-U0", base, "--", str(CPP_DIR)],
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
    args = parser.parse_args()

    tidy_args = [args.clang_tidy, "--quiet"]
    if args.diff:
        line_filter = changed_lines(args.diff)
        if not line_filter:
            print(f"No C++ backend changes since {args.diff}.")
            return 0
        tidy_args.append("--line-filter=" + json.dumps(line_filter))

    with tempfile.TemporaryDirectory() as tmp_dir:
        compiler_flags = COMPILER_FLAGS + omp_include_dir(Path(tmp_dir))
        if sys.platform == "win32":
            compiler_flags.append("--target=x86_64-w64-mingw32")

        def run(source: Path) -> str:
            return subprocess.run(
                [*tidy_args, str(source), "--", *compiler_flags],
                capture_output=True,
                text=True,
                check=False,
                cwd=ROOT,
            ).stdout

        with ThreadPoolExecutor() as pool:
            outputs = list(pool.map(run, compiled_sources()))

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
