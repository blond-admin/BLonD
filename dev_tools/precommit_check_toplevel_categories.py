"""Check that all BLonD toplevel classes are categorized for the docs.

``docs/create_doc_blond_main_objects.py`` crashes by design when a class
exported in ``blond/__init__.py`` has no entry in ``ASSIGNED_CATEGORIES``.
That crash only shows up in the (slow) documentation build, so this
pre-commit hook performs the same check and reports exactly which line to
add where.
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_SCRIPT = REPO_ROOT / "docs" / "create_doc_blond_main_objects.py"


def load_doc_script() -> ModuleType:
    """Import ``docs/create_doc_blond_main_objects.py`` by file path.

    Returns
    -------
    ModuleType
        The imported module, holding ``Categories`` and
        ``ASSIGNED_CATEGORIES``.
    """
    spec = importlib.util.spec_from_file_location(
        "create_doc_blond_main_objects", DOC_SCRIPT
    )
    assert spec is not None and spec.loader is not None, str(DOC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def suggest_category(categories, module_path: str) -> str:
    """Guess a fitting category from the module a class is defined in.

    Parameters
    ----------
    categories
        The ``Categories`` enum of the documentation script.
    module_path
        The ``__module__`` of the class to categorize.

    Returns
    -------
    str
        Name of the suggested ``Categories`` member.
    """
    rules = (
        ("handle_results", categories.DIAGNOSTICS),
        ("observation", categories.DIAGNOSTICS),
        ("plot", categories.PLOTTING),
        ("cycles", categories.CYCLE),
        ("beam_preparation", categories.BEAM),
        ("distribution", categories.BEAM),
        ("impedance", categories.WAKE),
        ("wake", categories.WAKE),
        ("backends", categories.BACKEND),
        ("simulation", categories.DYNAMICS),
        ("ring", categories.LATTICE),
        ("physics", categories.LATTICE),
    )
    for needle, category in rules:
        if needle in module_path:
            return category.name
    return categories.MISC.name


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
    import blond

    doc_script = load_doc_script()
    categories = doc_script.Categories
    assigned = doc_script.ASSIGNED_CATEGORIES

    exported_classes = {
        name: getattr(blond, name)
        for name in dir(blond)
        if inspect.isclass(getattr(blond, name))
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
            module_path = exported_classes[name].__module__
            suggestion = suggest_category(categories, module_path)
            print(
                f'    "{name}": Categories.{suggestion}.value,'
                f"  # from {module_path}"
            )
        available = ", ".join(category.name for category in categories)
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
