import ast
import pathlib

import blond
from blond.testing.backend_testing import BLonDTestCase

# Names `typing` gained after Python 3.10, the oldest supported version.
_TYPING_AFTER_3_10 = {"Self", "Never", "LiteralString", "assert_never"}
_SKIPPED_FOLDERS = ("legacy", "experimental")


def _runtime_typing_imports(tree: ast.Module) -> set[str]:
    """Names imported from `typing` outside `if TYPE_CHECKING:` blocks."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            names |= {alias.name for alias in node.names}
    return names


class TestPython310Compat(BLonDTestCase):
    def test_no_runtime_typing_import_newer_than_3_10(self) -> None:
        root = pathlib.Path(blond.__file__).parent
        offenders = []
        for path in root.rglob("*.py"):
            if any(folder in path.parts for folder in _SKIPPED_FOLDERS):
                continue
            tree = ast.parse(path.read_text())
            too_new = _runtime_typing_imports(tree) & _TYPING_AFTER_3_10
            if too_new:
                offenders.append(f"{path.relative_to(root)}: {too_new}")
        self.assertEqual(offenders, [])
