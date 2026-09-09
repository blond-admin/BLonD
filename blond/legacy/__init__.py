# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Access point for the legacy blond version, use ``from blond.legacy import blond2``."""

import importlib
import importlib.util
import pkgutil
from importlib.abc import MetaPathFinder

LEGACY_PACKAGE = "blond.legacy.blond2"

#: Subtrees of ``blond`` that are never redirected to BLonD 2:
#: ``blond.legacy`` *is* the v2 package, and ``blond.interfaces``
#: exists in both versions (too rare to disentangle).
NEVER_REDIRECTED = ("blond.legacy", "blond.interfaces")


def _legacy_module_name(module_name: str) -> str | None:
    """Name of the BLonD 2 twin of `module_name`, or None if it has none."""
    if module_name == "blond":
        return LEGACY_PACKAGE
    if not module_name.startswith("blond."):
        return None
    if module_name.startswith(NEVER_REDIRECTED):
        return None
    return f"{LEGACY_PACKAGE}.{module_name.removeprefix('blond.')}"


def _moved_hint(missing: str, kind: str, corrected_import: str) -> str:
    """Message telling the user that `missing` moved to BLonD 2.

    `missing` names what was looked up, `kind` is "module" or "object",
    and `corrected_import` holds the indented import line(s) that work.
    """
    return (
        f"{missing} does not exist in the current blond (v3) API.\n"
        f"The old blond (v2) package now lives at {LEGACY_PACKAGE}.\n"
        f"To keep using the v2 {kind}, update your import to:\n"
        f"{corrected_import}"
    )


def _defining_legacy_module(legacy_module_name: str, name: str) -> str | None:
    """Name of the module below `legacy_module_name` defining `name`.

    Searches `legacy_module_name` itself and, if it is a package, all of
    its submodules. Returns None if `name` is not found anywhere.
    """
    try:
        module = importlib.import_module(legacy_module_name)
    except ImportError:
        return None
    if hasattr(module, name):
        return legacy_module_name
    if not hasattr(module, "__path__"):
        return None
    submodules = pkgutil.walk_packages(
        module.__path__,
        prefix=f"{legacy_module_name}.",
        onerror=lambda _: None,
    )
    for _, submodule_name, _ in submodules:
        try:
            submodule = importlib.import_module(submodule_name)
        except Exception:  # noqa: BLE001  # legacy modules may be broken
            continue
        if hasattr(submodule, name):
            return submodule_name
    return None


class DynamicLegacyFinder(MetaPathFinder):
    """
    Import hook that explains imports of moved BLonD 2 modules.

    Registered on ``sys.meta_path`` after the default finders, so it is
    only consulted for module names that do not exist in BLonD 3. If
    the same relative path exists below ``blond.legacy.blond2``, the
    import fails with a message telling the user where the module went.
    """

    def find_spec(self, fullname: str, path=None, target=None):
        """Never provide a module; raise ImportError if it moved to v2.

        `path` and `target` are part of the ``MetaPathFinder`` protocol
        and unused here.
        """
        legacy_target = _legacy_module_name(fullname)
        if legacy_target is None:
            return None
        try:
            if importlib.util.find_spec(legacy_target) is None:
                return None
        except (ImportError, AttributeError, ValueError):
            return None
        raise ImportError(
            _moved_hint(
                f"Module '{fullname}'",
                "module",
                f"  import {legacy_target}\n"
                f"  # or: from {legacy_target} import ...",
            )
        )


def legacy_getattr(module_name: str, name: str):
    """
    Explain a missing attribute that still exists in BLonD 2.

    Intended to be called from a module-level ``__getattr__`` of a
    BLonD 3 module (PEP 562), so that ``from blond import bigaussian``
    tells the user where the name went instead of failing silently.
    Never returns: raises ImportError naming the corrected import if
    `name` lives in the BLonD 2 twin of `module_name`, and
    AttributeError if `name` is private or gone from v2 as well.
    """
    legacy_module_name = _legacy_module_name(module_name)
    if legacy_module_name is not None and not name.startswith("_"):
        defining_module = _defining_legacy_module(legacy_module_name, name)
        if defining_module is not None:
            raise ImportError(
                _moved_hint(
                    f"'{name}' in module '{module_name}'",
                    "object",
                    f"  from {defining_module} import {name}",
                )
            )
    raise AttributeError(f"module '{module_name}' has no attribute '{name}'")
