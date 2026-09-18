# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Guard against eagerly formatted logging calls in the BLonD 3 tree.

``logger.debug(f"...{value}")`` builds its message *before* the logging
call, so the formatting happens on every invocation even when the level
is disabled and the message is discarded. In per-turn code that is pure
waste: ``apply_schedules`` used to format a numpy array to a string on
every single turn, which cost roughly a third of the per-turn Python
time of a kick+drift simulation.

Passing the values as arguments -- ``logger.debug("... %s", value)`` --
defers formatting to the handler, so nothing is rendered unless the
record is actually emitted.

Authors: Simon Lauber
"""

from __future__ import annotations

import ast
from pathlib import Path

from blond.testing.backend_testing import BLonDTestCase

# Logging methods that take a message as their first positional argument.
_LOGGING_METHODS = frozenset(
    {
        "debug",
        "info",
        "warning",
        "error",
        "critical",
        "exception",
        "log",
    }
)

# ``blond/legacy`` follows BLonD 2 conventions on purpose and
# ``blond/experimental`` is deliberately excluded from the lint gate.
_EXCLUDED_PARTS = ("legacy", "experimental")

_BLOND_ROOT = Path(__file__).resolve().parents[2] / "blond"


def _is_logger_call(node: ast.Call) -> bool:
    """
    Whether a call node is a ``logger.<level>(...)`` call.

    Parameters
    ----------
    node
        The call node to inspect.

    Returns
    -------
    is_logger_call
        True if the call targets a logging method on a ``logger`` object.
    """
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in _LOGGING_METHODS:
        return False
    target = func.value
    if isinstance(target, ast.Name):
        return "logger" in target.id.lower()
    if isinstance(target, ast.Attribute):
        return "logger" in target.attr.lower()
    return False


def _is_eagerly_formatted(message: ast.expr) -> bool:
    """
    Whether a log message is rendered before the logging call.

    Parameters
    ----------
    message
        The first positional argument of the logging call.

    Returns
    -------
    is_eager
        True for f-strings, ``str.format`` calls and ``%`` formatting.
    """
    if isinstance(message, ast.JoinedStr):  # f-string
        return True
    if isinstance(message, ast.BinOp) and isinstance(message.op, ast.Mod):
        return True
    return bool(
        isinstance(message, ast.Call)
        and isinstance(message.func, ast.Attribute)
        and message.func.attr == "format"
    )


def _find_offenders() -> list[str]:
    """
    Collect every eagerly formatted logging call under ``blond/``.

    Returns
    -------
    offenders
        ``file:line`` locations, one per offending logging call.
    """
    offenders: list[str] = []
    for path in sorted(_BLOND_ROOT.rglob("*.py")):
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_logger_call(node):
                continue
            if not node.args:
                continue
            if _is_eagerly_formatted(node.args[0]):
                relative = path.relative_to(_BLOND_ROOT.parent)
                offenders.append(f"{relative}:{node.lineno}")
    return offenders


class TestLoggingIsLazy(BLonDTestCase):
    """Logging calls must defer message formatting to the handler."""

    def test_no_eagerly_formatted_logging_calls(self):
        """No ``logger`` call in ``blond/`` may pre-render its message."""
        offenders = _find_offenders()
        self.assertEqual(
            [],
            offenders,
            msg=(
                "Eagerly formatted logging calls found. Pass the values as "
                "logging arguments instead, e.g. "
                'logger.debug("Wrote %s", value), so that nothing is '
                "formatted while the level is disabled:\n  "
                + "\n  ".join(offenders)
            ),
        )
