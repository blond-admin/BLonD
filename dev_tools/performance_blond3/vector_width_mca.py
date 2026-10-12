# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Estimate what 512-bit vectors would buy each C++ kernel, with llvm-mca.

Compiles every C++ backend source twice -- once as the build does it, once
forced to 512-bit vectors -- isolates the innermost vectorized loop of each
kernel, and has ``llvm-mca`` estimate its throughput. Reports cycles per
particle for both widths, normalized by the elements each loop iteration
handles, so the two widths are directly comparable.

What this does and does not tell you
------------------------------------
``llvm-mca`` models a steady-state pipeline with all data in L1 and no
frequency changes. It therefore estimates the *compute-bound ceiling*: it
is a good screen for which kernels could gain from wider vectors, and it
was within a few percent of measured single-core speedups for the kicks
and ``beam_phase``. It is blind to memory bandwidth and to AVX-512
downclocking, which is exactly why a kernel it likes can still fail to
improve in a threaded run on a large beam. Use it to pick candidates;
decide with `backends/scan_performance.py`.

``llvm-mca`` is not a BLonD dependency. Point ``BLOND_LLVM_MCA`` at the
binary, or have it on ``PATH``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

CPP_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "blond",
    "core",
    "backends",
    "cpp",
)

#: Flags the real build uses, minus the output options.
BUILD_FLAGS = [
    "-O3",
    "-std=c++11",
    "-funroll-loops",
    "-ftree-vectorize",
    "-march=native",
    "-ffast-math",
    "-fopenmp",
    "-DPARALLEL",
    "-D_GLIBCXX_PARALLEL",
    "-Wno-unknown-pragmas",
    "-D_USE_MATH_DEFINES",
    "-fPIC",
]

#: Bytes each vector register holds, as doubles.
ELEMENTS_PER_REGISTER = {"zmm": 8, "ymm": 4, "xmm": 2}


def find_llvm_mca() -> str:
    """
    Locate the ``llvm-mca`` binary.

    Returns
    -------
    path
        Path to ``llvm-mca``.

    Raises
    ------
    SystemExit
        If it can be found neither via ``BLOND_LLVM_MCA`` nor on ``PATH``.
    """
    path = os.environ.get("BLOND_LLVM_MCA") or shutil.which("llvm-mca")
    if not path:
        sys.exit(
            "llvm-mca not found. Install LLVM, or set BLOND_LLVM_MCA to the "
            "binary. On a machine without root, the rpm/deb can be unpacked "
            "into a scratch directory and pointed at with that variable."
        )
    return path


def compile_to_asm(source: str, out_dir: str, width_512: bool) -> str:
    """
    Compile one C++ source to assembly, optionally forcing 512-bit vectors.

    Parameters
    ----------
    source
        Path of the ``.cpp`` file to compile.
    out_dir
        Directory the assembly is written to.
    width_512
        Whether to add ``-mprefer-vector-width=512``.

    Returns
    -------
    asm_path
        Path of the generated assembly file.
    """
    name = os.path.basename(source).replace(".cpp", "")
    suffix = "512" if width_512 else "256"
    asm_path = os.path.join(out_dir, f"{name}_{suffix}.s")
    command = ["g++", *BUILD_FLAGS]
    if width_512:
        command.append("-mprefer-vector-width=512")
    else:
        # Neutralize the per-kernel macro too. Without this, a kernel that
        # already carries BLOND_PREFER_VECTOR_WIDTH_512 compiles to 512-bit
        # on both sides and is reported as 1.00x -- which reads like "no
        # benefit" for precisely the kernels that benefit most.
        command.append("-DBLOND_PREFER_VECTOR_WIDTH_512=")
    command += ["-I", CPP_DIR, "-S", "-o", asm_path, source]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return asm_path


def innermost_loops(asm_path: str) -> dict[str, list[str]]:
    """
    Return the innermost single-block loop of every function in `asm_path`.

    A loop qualifies only if it is one basic block: a label, straight-line
    instructions, and a conditional branch back to that label. Anything
    else -- a loop spanning several blocks, or a function epilogue caught
    by accident -- is not a meaningful unit to hand to ``llvm-mca``, which
    analyses straight-line code.

    Parameters
    ----------
    asm_path
        Path of an assembly file.

    Returns
    -------
    loops
        Instructions of the most heavily vectorized qualifying loop, per
        function symbol.
    """
    with open(asm_path, encoding="utf-8") as file:
        lines = file.read().split("\n")
    symbols = [
        (index, line.rstrip(":"))
        for index, line in enumerate(lines)
        if re.match(r"^[a-zA-Z_][\w.]*:$", line)
    ]
    loops: dict[str, list[str]] = {}
    for position, (start, name) in enumerate(symbols):
        end = (
            symbols[position + 1][0]
            if position + 1 < len(symbols)
            else len(lines)
        )
        body = lines[start:end]
        candidates = []
        for index, line in enumerate(body):
            match = re.match(r"^(\.L\d+):", line.strip())
            if not match:
                continue
            label = match.group(1)
            for offset in range(index + 1, len(body)):
                statement = body[offset].strip()
                if re.match(r"^\.L\d+:", statement) or statement.startswith(
                    "ret"
                ):
                    break
                if re.match(r"^j\w+\s+" + re.escape(label) + r"$", statement):
                    candidates.append(
                        [
                            text
                            for text in body[index + 1 : offset + 1]
                            if text.strip()
                            and not text.strip().startswith(".")
                        ]
                    )
                    break
        if candidates:
            loops[name] = max(
                candidates,
                key=lambda body: sum(
                    "zmm" in line or "ymm" in line for line in body
                ),
            )
    return loops


def elements_per_iteration(loop: list[str]) -> int:
    """
    Return how many doubles one iteration of `loop` writes.

    Counted from the vector stores, which is what makes a 256-bit and a
    512-bit version of the same loop comparable: the wider one does the
    same work in half as many iterations.

    Parameters
    ----------
    loop
        Instructions of one loop body.

    Returns
    -------
    elements
        Number of doubles stored per iteration.
    """
    total = 0
    for line in loop:
        match = re.match(r"\s*vmov(?:up|ap)d\s+%([zyx]mm)\d+,\s+[^%]", line)
        if match:
            total += ELEMENTS_PER_REGISTER[match.group(1)]
    return total


def cycles_per_iteration(mca: str, loop: list[str], cpu: str) -> float | None:
    """
    Run ``llvm-mca`` on `loop` and return its cycles per iteration.

    Parameters
    ----------
    mca
        Path of the ``llvm-mca`` binary.
    loop
        Instructions of one loop body.
    cpu
        Value for ``-mcpu``.

    Returns
    -------
    cycles
        Estimated cycles per loop iteration, or None if llvm-mca gave no
        cycle count.
    """
    iterations = 100
    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as handle:
        handle.write("\n".join(loop) + "\n")
        path = handle.name
    try:
        result = subprocess.run(
            [mca, f"-mcpu={cpu}", f"-iterations={iterations}", path],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(path)
    for line in result.stdout.split("\n"):
        if "Total Cycles" in line:
            return int(line.split(":")[1]) / iterations
    return None


def main() -> None:
    """Report the 256- vs 512-bit throughput estimate of every kernel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu",
        default="native",
        help="-mcpu value for llvm-mca. Default: native.",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help="C++ sources to analyse. Default: all kernels in the backend.",
    )
    args = parser.parse_args()

    mca = find_llvm_mca()
    sources = args.sources or sorted(
        os.path.join(CPP_DIR, name)
        for name in os.listdir(CPP_DIR)
        if name.endswith(".cpp")
    )

    print(
        f"{'kernel':38} {'256 cyc/elem':>13} {'512 cyc/elem':>13} "
        f"{'speedup':>8}"
    )
    with tempfile.TemporaryDirectory() as out_dir:
        for source in sources:
            try:
                asm_256 = compile_to_asm(source, out_dir, width_512=False)
                asm_512 = compile_to_asm(source, out_dir, width_512=True)
            except subprocess.CalledProcessError:
                continue
            loops_256 = innermost_loops(asm_256)
            loops_512 = innermost_loops(asm_512)
            for name, loop_512 in loops_512.items():
                loop_256 = loops_256.get(name)
                if loop_256 is None:
                    continue
                if not any("zmm" in line for line in loop_512):
                    continue  # GCC never widens this loop
                elements_256 = elements_per_iteration(loop_256)
                elements_512 = elements_per_iteration(loop_512)
                if not (elements_256 and elements_512):
                    continue
                cycles_256 = cycles_per_iteration(mca, loop_256, args.cpu)
                cycles_512 = cycles_per_iteration(mca, loop_512, args.cpu)
                if cycles_256 is None or cycles_512 is None:
                    continue
                per_element_256 = cycles_256 / elements_256
                per_element_512 = cycles_512 / elements_512
                print(
                    f"{name[:38]:38} {per_element_256:13.3f} "
                    f"{per_element_512:13.3f} "
                    f"{per_element_256 / per_element_512:7.2f}x"
                )


if __name__ == "__main__":  # pragma: no cover
    main()
