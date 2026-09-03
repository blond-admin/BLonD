"""Check that all BLonD toplevel classes are categorized for the docs.

``docs/create_doc_blond_main_objects.py`` crashes by design when a class
exported in ``blond/__init__.py`` has no entry in ``ASSIGNED_CATEGORIES``.
That crash only shows up in the (slow) documentation build, so this
pre-commit hook performs the same check and reports exactly which line to
add where.

The check is purely static: both files are parsed with ``ast`` instead of
being imported. Importing ``blond`` would pull in the whole package (and
require it to be installed), which is far too slow for a pre-commit hook.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_SCRIPT = REPO_ROOT / "docs" / "create_doc_blond_main_objects.py"
INIT_SCRIPT = REPO_ROOT / "blond" / "__init__.py"

# Depth limit while following re-exports (``from x import Y``) to the
# module that actually defines a name.
MAX_REEXPORT_DEPTH = 10


def parse(path: Path) -> ast.Module:
    """Parse a Python file into an abstract syntax tree.

    Parameters
    ----------
    path
        File to read and parse.

    Returns
    -------
    ast.Module
        The parsed syntax tree.
    """
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def read_toplevel_exports() -> dict[str, str]:
    """Read ``__all__`` of ``blond/__init__.py`` and where names come from.

    Returns
    -------
    dict of str to str
        Maps each name in ``__all__`` to the module it is imported from.
        Names without a matching ``from ... import ...`` map to ``""``.
    """
    tree = parse(INIT_SCRIPT)
    exported = all_names(tree)
    origins = import_origins(tree)
    return {name: origins.get(name, "") for name in exported}


def all_names(tree: ast.Module) -> list[str]:
    """Collect the string entries of the ``__all__`` assignment.

    Parameters
    ----------
    tree
        Syntax tree of ``blond/__init__.py``.

    Returns
    -------
    list of str
        The exported names, in source order.
    """
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        if "__all__" not in targets:
            continue
        assert isinstance(node.value, (ast.List, ast.Tuple)), (
            "`__all__` must be a list or tuple literal"
        )
        return [
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
        ]
    raise AssertionError(f"No `__all__` found in {INIT_SCRIPT}")


def import_origins(tree: ast.Module) -> dict[str, str]:
    """Map each imported name to the module it is imported from.

    Parameters
    ----------
    tree
        Syntax tree of a module.

    Returns
    -------
    dict of str to str
        Bound name (respecting ``as`` aliases) to source module. Relative
        imports and ``import x`` statements are skipped.
    """
    origins: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level != 0 or node.module is None:
            continue
        for alias in node.names:
            origins[alias.asname or alias.name] = node.module
    return origins


def module_file(module_path: str) -> Path | None:
    """Locate the source file of a ``blond`` module.

    Parameters
    ----------
    module_path
        Dotted module path, e.g. ``blond.physics.drifts``.

    Returns
    -------
    Path or None
        The ``.py`` file, or ``None`` if it is not a file in this repo.
    """
    base = REPO_ROOT.joinpath(*module_path.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def is_class(name: str, module_path: str, depth: int = 0) -> bool:
    """Decide whether a toplevel export is a class, without importing.

    The defining module is parsed and searched for ``class <name>``;
    re-exports (``from ... import <name>``) are followed. Names that
    cannot be resolved statically fall back to the naming convention
    that classes are ``CamelCase`` and everything else is not.

    Parameters
    ----------
    name
        Name as bound in the module.
    module_path
        Dotted path of the module the name is imported from.
    depth
        Current recursion depth while following re-exports.

    Returns
    -------
    bool
        Whether ``name`` refers to a class.
    """
    path = module_file(module_path) if module_path else None
    if path is None or depth > MAX_REEXPORT_DEPTH:
        return name[:1].isupper()

    tree = parse(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return True
        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if is_function and node.name == name:
            return False
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return False

    origins = import_origins(tree)
    if name in origins:
        return is_class(name, origins[name], depth + 1)
    return name[:1].isupper()


def doc_script_categories() -> list[str]:
    """Read the member names of the ``Categories`` enum in the doc script.

    Returns
    -------
    list of str
        Enum member names, in source order.
    """
    for node in parse(DOC_SCRIPT).body:
        if isinstance(node, ast.ClassDef) and node.name == "Categories":
            return [
                target.id
                for statement in node.body
                if isinstance(statement, ast.Assign)
                for target in statement.targets
                if isinstance(target, ast.Name)
            ]
    raise AssertionError(f"No `Categories` enum found in {DOC_SCRIPT}")


def assigned_categories() -> list[str]:
    """Read the keys of ``ASSIGNED_CATEGORIES`` in the doc script.

    Returns
    -------
    list of str
        The categorized names, in source order.
    """
    for node in parse(DOC_SCRIPT).body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        if "ASSIGNED_CATEGORIES" not in targets:
            continue
        assert isinstance(node.value, ast.Dict), (
            "`ASSIGNED_CATEGORIES` must be a dict literal"
        )
        return [
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
    raise AssertionError(f"No `ASSIGNED_CATEGORIES` found in {DOC_SCRIPT}")


def suggest_category(categories: list[str], module_path: str) -> str:
    """Guess a fitting category from the module a class is defined in.

    Parameters
    ----------
    categories
        Available ``Categories`` member names.
    module_path
        Module the class is imported from.

    Returns
    -------
    str
        Name of the suggested ``Categories`` member.
    """
    rules = (
        ("handle_results", "DIAGNOSTICS"),
        ("observation", "DIAGNOSTICS"),
        ("plot", "PLOTTING"),
        ("cycles", "CYCLE"),
        ("beam_preparation", "BEAM"),
        ("distribution", "BEAM"),
        ("impedance", "WAKE"),
        ("wake", "WAKE"),
        ("backends", "BACKEND"),
        ("simulation", "DYNAMICS"),
        ("ring", "LATTICE"),
        ("physics", "LATTICE"),
    )
    for needle, category in rules:
        if needle in module_path and category in categories:
            return category
    return "MISC"


def dict_line_number() -> int:
    """Find the line of the ``ASSIGNED_CATEGORIES`` dict in the doc script.

    Returns
    -------
    int
        1-based line number, or 0 if the dict was not found.
    """
    lines = DOC_SCRIPT.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if line.startswith("ASSIGNED_CATEGORIES"):
            return number
    return 0


def perform_check() -> int:
    """Compare the toplevel exports against ``ASSIGNED_CATEGORIES``.

    Returns
    -------
    int
        Process exit code: 0 if everything is categorized, else 1.
    """
    categories = doc_script_categories()
    assigned = assigned_categories()

    exported_classes = {
        name: module_path
        for name, module_path in read_toplevel_exports().items()
        if is_class(name, module_path)
    }
    missing = [name for name in exported_classes if name not in assigned]
    stale = [name for name in assigned if name not in exported_classes]

    if not missing and not stale:
        return 0

    location = f"{DOC_SCRIPT.relative_to(REPO_ROOT)}:{dict_line_number()}"
    print("Toplevel classes and `ASSIGNED_CATEGORIES` are out of sync.")
    print(f"Both live in {location} (`ASSIGNED_CATEGORIES`).\n")

    if missing:
        print(
            "The following classes are exported in `blond/__init__.py` but"
            " have no category. The documentation build will fail on them."
            " Add to `ASSIGNED_CATEGORIES`:\n"
        )
        for name in sorted(missing):
            module_path = exported_classes[name]
            suggestion = suggest_category(categories, module_path)
            print(
                f'    "{name}": Categories.{suggestion}.value,'
                f"  # from {module_path}"
            )
        available = ", ".join(categories)
        print(f"\nThe suggested category is a guess. Available: {available}")

    if stale:
        print(
            "\nThe following names are in `ASSIGNED_CATEGORIES` but are no"
            " longer exported classes of `blond`. Remove their line from"
            f" {location}:\n"
        )
        for name in sorted(stale):
            print(f'    "{name}"')

    return 1


if __name__ == "__main__":
    sys.exit(perform_check())
