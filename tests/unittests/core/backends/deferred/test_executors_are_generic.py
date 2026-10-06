"""The deferred executors know no kernel: every kernel lives with its kind.

The executors (``deferred.cpp``, the batch section of ``kernels.cu`` and
the launch glue in ``cuda/callables.py``) apply whatever records a batch
holds through `visit_kernel_call` and the hooks each record type
implements. A kernel name in them means kernel-specific code leaked into
the executor.
"""

import re
from pathlib import Path

from blond.testing.backend_testing import BLonDTestCase

_BACKENDS = Path(__file__).parents[5] / "blond" / "core" / "backends"
# A record type (`DriftSimpleArgs`), a kernel id (`KernelId::DriftSimple`)
# or a kernel by name.
_KERNEL_NAME = re.compile(
    r"\b(?!KernelCallArgs\b)[A-Z]\w*Args\b|KernelId::|(?i:histogram)"
)


def _kernel_names(text: str) -> list[str]:
    return sorted(set(_KERNEL_NAME.findall(text)))


def _executor_section_of_kernels_cu() -> str:
    """From the batch parameter type to the end of the executor kernel."""
    text = (_BACKENDS / "cuda" / "kernels.cu").read_text()
    start = text.index("struct KernelCallBatch {")
    end = text.index('extern "C" __global__ void\nhistogram_sparse(')
    return text[start:end]


class TestExecutorsAreGeneric(BLonDTestCase):
    def test_cpp_executor_names_no_kernel(self) -> None:
        text = (_BACKENDS / "cpp" / "deferred.cpp").read_text()
        self.assertEqual(_kernel_names(text), [])

    def test_cuda_executor_names_no_kernel(self) -> None:
        self.assertEqual(_kernel_names(_executor_section_of_kernels_cu()), [])

    def test_cuda_launch_names_no_kernel(self) -> None:
        text = (_BACKENDS / "cuda" / "callables.py").read_text()
        start = text.index("def _split_batch(")
        end = text.index("class CudaSpecials(")
        self.assertEqual(_kernel_names(text[start:end]), [])
