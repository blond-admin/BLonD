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


def _legacy_module_name(module_name: str) -> str | None:
    """
    Map a BLonD 3 module name onto its BLonD 2 twin, if any.

    Parameters
    ----------
    module_name
        Fully qualified module name, e.g. ``"blond.beam"``.

    Returns
    -------
    str or None
        ``"blond.legacy.blond2.<rest>"``, or None if `module_name` is
        not below ``blond``, is already inside ``blond.legacy``, or is
        below ``blond.interfaces`` (which exists in both versions).
    """
    if module_name == "blond":
        return LEGACY_PACKAGE
    if not module_name.startswith("blond."):
        return None
    if module_name.startswith("blond.legacy"):
        return None
    # blond.interfaces exists in both versions; too rare to disentangle.
    if module_name.startswith("blond.interfaces"):
        return None
    return f"{LEGACY_PACKAGE}.{module_name[len('blond.'):]}"


class DynamicLegacyFinder(MetaPathFinder):
    """
    Import hook that explains imports of moved BLonD 2 modules.

    Registered on ``sys.meta_path`` after the default finders, so it is
    only consulted for module names that do not exist in BLonD 3. If
    the same relative path exists below ``blond.legacy.blond2``, the
    import fails with a message telling the user where the module went.
    """

    def find_spec(self, fullname: str, path=None, target=None):
        """
        Raise a helpful ImportError for moved BLonD 2 modules.

        Parameters
        ----------
        fullname
            Fully qualified name of the module being imported.
        path
            Parent package ``__path__`` (unused).
        target
            Module object to reload (unused).

        Returns
        -------
        None
            Always; this finder never provides a module itself.

        Raises
        ------
        ImportError
            If `fullname` has no BLonD 3 module but a BLonD 2 twin.
        """
        legacy_target = _legacy_module_name(fullname)
        if legacy_target is None:
            return None
        try:
            found = importlib.util.find_spec(legacy_target) is not None
        except (ImportError, AttributeError, ValueError):
            found = False
        if found:
            raise ImportError(
                f"Module '{fullname}' does not exist in the current "
                f"blond (v3) API.\n"
                f"The old blond (v2) package now lives at "
                f"{LEGACY_PACKAGE}.\n"
                f"To keep using the v2 module, update your import to:\n"
                f"  import {legacy_target}\n"
                f"  # or: from {legacy_target} import ..."
            )
        return None


def _find_in_legacy_package(legacy_module_name: str, name: str):
    """
    Locate `name` in a BLonD 2 module or any of its submodules.

    Parameters
    ----------
    legacy_module_name
        Fully qualified name of a module below ``blond.legacy.blond2``.
    name
        Attribute name to look for.

    Returns
    -------
    str or None
        Name of the module that defines `name`, or None.
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


def legacy_getattr(module_name: str, name: str):
    """
    Explain a missing attribute that still exists in BLonD 2.

    Intended to be called from a module-level ``__getattr__`` of a
    BLonD 3 module (PEP 562), so that ``from blond import bigaussian``
    tells the user where the name went instead of failing silently.

    Parameters
    ----------
    module_name
        ``__name__`` of the BLonD 3 module the attribute was looked
        up on.
    name
        The missing attribute name.

    Returns
    -------
    None
        Never returns; it always raises.

    Raises
    ------
    ImportError
        If `name` is found in the BLonD 2 twin of `module_name` or one
        of its submodules. The message contains the corrected import.
    AttributeError
        If `name` is private, or not found in BLonD 2 either.
    """
    legacy_module_name = _legacy_module_name(module_name)
    if legacy_module_name is not None and not name.startswith("_"):
        defining_module = _find_in_legacy_package(legacy_module_name, name)
        if defining_module is not None:
            raise ImportError(
                f"'{name}' does not exist in '{module_name}' in the "
                f"current blond (v3) API.\n"
                f"The old blond (v2) package now lives at "
                f"{LEGACY_PACKAGE}.\n"
                f"To keep using the v2 object, update your import to:\n"
                f"  from {defining_module} import {name}"
            )
    raise AttributeError(f"module '{module_name}' has no attribute '{name}'")
