"""Check that every ``BLOND_*`` environment variable is declared.

The variables BLonD reads are listed in ``pyproject.toml`` under
``[tool.blond] env-vars``, in the same ``"NAME: description"`` format as
the pytest ``markers``. Without a check that list drifts: a new variable
is added in code and never documented, or a removed one stays listed.

A variable counts as used when a string constant in ``blond/`` is
exactly its name, e.g. ``os.environ.get("BLOND_FOO")`` or
``ENV_KEY = "BLOND_FOO"``. Docstrings, f-string messages and Python
names such as ``BLOND_ROOT`` are not matched. ``blond/legacy`` and
``blond/experimental`` are skipped, like in the other hooks.

The sources are read with ``ast`` instead of importing ``blond``, which
a pre-commit run cannot expect to be installed.
"""

import ast
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ENV_VAR_NAME = re.compile(r"BLOND_[A-Z0-9_]+")
SKIPPED_DIRECTORIES = ("legacy", "experimental")


def env_vars_in_source(source: str) -> dict[str, int]:
    """Find the ``BLOND_*`` variable names used in Python source.

    Parameters
    ----------
    source
        Content of a Python file.

    Returns
    -------
    names
        Every string constant that is exactly a ``BLOND_*`` name, mapped
        to the first line it appears on.
    """
    names: dict[str, int] = {}
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and ENV_VAR_NAME.fullmatch(node.value)
        ):
            first_line = names.get(node.value, node.lineno)
            names[node.value] = min(first_line, node.lineno)
    return names


def env_vars_used(package_dir: Path) -> dict[str, list[str]]:
    """Find the ``BLOND_*`` variables used in a package.

    Parameters
    ----------
    package_dir
        Directory of the ``blond`` package.

    Returns
    -------
    usages
        Variable name to the ``file:line`` locations using it, relative
        to the parent of ``package_dir`` (the repository root).
    """
    usages: dict[str, list[str]] = {}
    for path in sorted(package_dir.rglob("*.py")):
        relative_parts = path.relative_to(package_dir).parts
        if relative_parts[0] in SKIPPED_DIRECTORIES:
            continue
        source = path.read_text(encoding="utf-8")
        location = path.relative_to(package_dir.parent).as_posix()
        for name, line in env_vars_in_source(source).items():
            usages.setdefault(name, []).append(f"{location}:{line}")
    return usages


def declared_entries(pyproject: Path) -> list[str] | None:
    """Read ``[tool.blond] env-vars`` from ``pyproject.toml``.

    Parameters
    ----------
    pyproject
        Path of ``pyproject.toml``.

    Returns
    -------
    entries
        The raw ``"NAME: description"`` entries, or None if the table
        is missing.
    """
    with pyproject.open("rb") as file:
        config = tomllib.load(file)
    return config.get("tool", {}).get("blond", {}).get("env-vars")


def entry_to_add(name: str, usages: dict[str, list[str]]) -> str:
    """Format a paste-ready ``env-vars`` entry for an undeclared name.

    Parameters
    ----------
    name
        Name of the environment variable.
    usages
        Output of :func:`env_vars_used`.

    Returns
    -------
    line
        The entry with a placeholder description and, as a comment,
        where the variable is read.
    """
    return f'  "{name}: <what it does>",  # {", ".join(usages[name])}'


def parse_entries(
    entries: list[str],
) -> tuple[dict[str, str], list[str], list[str]]:
    """Split the ``env-vars`` entries by whether they are well-formed.

    Parameters
    ----------
    entries
        Output of :func:`declared_entries`.

    Returns
    -------
    declared
        Variable name to its full entry, for every entry with a valid
        ``BLOND_*`` name (with or without a description).
    undescribed
        Names whose entry has no description after the colon.
    malformed
        Entries that do not start with a valid ``BLOND_*`` name.
    """
    declared: dict[str, str] = {}
    undescribed: list[str] = []
    malformed: list[str] = []
    for entry in entries:
        name, _, description = entry.partition(":")
        name = name.strip()
        if not ENV_VAR_NAME.fullmatch(name):
            malformed.append(entry)
            continue
        declared[name] = entry
        if not description.strip():
            undescribed.append(name)
    return declared, undescribed, malformed


def perform_check(package_dir: Path, pyproject: Path) -> int:
    """Compare the used variables against the declared ones.

    Parameters
    ----------
    package_dir
        Directory of the ``blond`` package.
    pyproject
        Path of ``pyproject.toml``.

    Returns
    -------
    exit_code
        0 if the declarations match the code, else 1.
    """
    usages = env_vars_used(package_dir)
    entries = declared_entries(pyproject)
    if entries is None:
        print(f"{pyproject.name} has no `[tool.blond] env-vars` list. Add:\n")
        print("[tool.blond]\nenv-vars = [")
        for name in sorted(usages):
            print(entry_to_add(name, usages))
        print("]")
        return 1

    declared, undescribed, malformed = parse_entries(entries)
    missing = sorted(name for name in usages if name not in declared)
    stale = sorted(name for name in declared if name not in usages)
    sections = (
        (
            "Undeclared. Add to `env-vars` and describe them:",
            [entry_to_add(name, usages) for name in missing],
        ),
        (
            "Without a description. Add one after the colon:",
            [f'  "{name}: <what it does>",' for name in undescribed],
        ),
        (
            "No longer read in blond/. Remove from `env-vars`:",
            [f'  "{declared[name]}",' for name in stale],
        ),
        (
            'Not in the "BLOND_NAME: description" format. Fix:',
            [f'  "{entry}",' for entry in malformed],
        ),
    )
    if not any(lines for _, lines in sections):
        return 0

    print(f"`[tool.blond] env-vars` in {pyproject.name} is out of sync")
    print("with the BLOND_* environment variables read in blond/.")
    for heading, lines in sections:
        if lines:
            print(f"\n{heading}\n")
            print("\n".join(lines))
    return 1


if __name__ == "__main__":
    sys.exit(perform_check(REPO_ROOT / "blond", REPO_ROOT / "pyproject.toml"))
