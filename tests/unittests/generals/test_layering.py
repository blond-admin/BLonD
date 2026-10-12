import ast
import importlib
from pathlib import Path

import blond.generals
from blond.testing.backend_testing import BLonDTestCase

GENERALS_DIR = Path(blond.generals.__file__).parent


def _imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    """Collect ``(line, module)`` for every import in ``tree``."""
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append((node.lineno, node.module))
    return imported


class TestGeneralsLayering(BLonDTestCase):
    def test_generals_does_not_import_core(self):
        """`generals` sits below `core` and must not depend on it."""
        offenders = []
        for path in sorted(GENERALS_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for line, module in _imported_modules(tree):
                if module == "blond.core" or module.startswith("blond.core."):
                    relative = path.relative_to(GENERALS_DIR.parent)
                    offenders.append(f"{relative}:{line} imports {module}")
        self.assertEqual(offenders, [])

    def test_distributed_lives_in_core_backends(self):
        for name in ("distributed_array", "helpers"):
            with self.subTest(module=name):
                importlib.import_module(
                    f"blond.core.backends.mpi_distributed.{name}"
                )
