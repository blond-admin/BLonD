"""Check that all BLonD toplevel classes are categorized for the docs.

``docs/create_doc_blond_main_objects.py`` crashes by design when a class
reachable as ``blond.<Name>`` has no entry in ``ASSIGNED_CATEGORIES``.
That crash only shows up in the (slow) documentation build, so this
hook performs the same check in milliseconds and says what to add where.

Everything here is static: the sources are read with ``ast`` and one
regex instead of importing ``blond``, which a pre-commit run cannot
expect to be installed. The price is that a class defined *outside*
``blond/`` and re-exported at toplevel is not recognized as a class --
the documentation build stays the authority, this is only its fast,
early warning.
"""

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_SCRIPT = REPO_ROOT / "docs" / "create_doc_blond_main_objects.py"
INIT_SCRIPT = REPO_ROOT / "blond" / "__init__.py"

CLASS_DEFINITION = re.compile(r"^class\s+(\w+)", re.MULTILINE)

# Directory name in the defining path -> `Categories` member, first
# match wins. Only used to pre-fill the line the developer has to add.
CATEGORY_HINTS = (
    ("handle_results", "DIAGNOSTICS"),
    ("plotting", "PLOTTING"),
    ("cycles", "CYCLE"),
    ("beam_preparation", "BEAM"),
    ("impedances", "WAKE"),
    ("backends", "BACKEND"),
    ("simulation", "DYNAMICS"),
    ("ring", "DYNAMICS"),
    ("beam", "BEAM"),
    ("physics", "LATTICE"),
)


def blond_classes() -> dict[str, Path]:
    """Find every class defined anywhere in the ``blond`` package.

    Returns
    -------
    classes
        Class name to the file defining it. ``blond/legacy`` is skipped,
        as it is excluded from the documentation.
    """
    classes: dict[str, Path] = {}
    for path in sorted((REPO_ROOT / "blond").rglob("*.py")):
        if "legacy" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        for match in CLASS_DEFINITION.finditer(source):
            classes.setdefault(match.group(1), path)
    return classes


def toplevel_names() -> dict[str, str]:
    """Read the names that ``import blond`` binds at toplevel.

    Returns
    -------
    names
        For every ``from ... import ...`` in ``blond/__init__.py``, the
        name bound in ``blond`` (what the doc script sees in
        ``dir(blond)``) mapped to the name in the defining module. The
        two differ for ``import Original as Bound``.
    """
    tree = ast.parse(INIT_SCRIPT.read_text(encoding="utf-8"))
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names[alias.asname or alias.name] = alias.name
    return names


def categorized_names() -> list[str]:
    """Read the keys of ``ASSIGNED_CATEGORIES`` in the doc script.

    Returns
    -------
    names
        The class names that already have a documentation category.
    """
    tree = ast.parse(DOC_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        is_wanted = any(
            isinstance(target, ast.Name) and target.id == "ASSIGNED_CATEGORIES"
            for target in node.targets
        )
        if is_wanted and isinstance(node.value, ast.Dict):
            return [
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant)
            ]
    raise AssertionError(f"No `ASSIGNED_CATEGORIES` dict in {DOC_SCRIPT}")


def suggest_category(path: Path) -> str:
    """Guess a category from the file a class is defined in.

    Parameters
    ----------
    path
        File defining the class.

    Returns
    -------
    category
        Name of a ``Categories`` member; ``MISC`` if nothing fits.
    """
    for directory, category in CATEGORY_HINTS:
        if directory in path.parts:
            return category
    return "MISC"


def perform_check() -> int:
    """Compare the toplevel classes against ``ASSIGNED_CATEGORIES``.

    Returns
    -------
    exit_code
        0 if everything is categorized, else 1.
    """
    classes = blond_classes()
    exported = {
        bound: classes[defined]
        for bound, defined in toplevel_names().items()
        if defined in classes
    }
    assigned = categorized_names()
    missing = sorted(name for name in exported if name not in assigned)
    stale = sorted(name for name in assigned if name not in exported)
    if not missing and not stale:
        return 0

    doc_script = DOC_SCRIPT.relative_to(REPO_ROOT)
    print(f"`ASSIGNED_CATEGORIES` in {doc_script} is out of sync with the")
    print("classes exported by blond/__init__.py.\n")
    if missing:
        print("Not categorized (the doc build will fail on these). Add:\n")
        for name in missing:
            category = suggest_category(exported[name])
            print(f'    "{name}": Categories.{category}.value,')
        print("\nThe category is a guess -- correct it if it does not fit.")
    if stale:
        print("\nNo longer exported classes of blond. Remove:\n")
        for name in stale:
            print(f'    "{name}"')
    return 1


if __name__ == "__main__":
    sys.exit(perform_check())
