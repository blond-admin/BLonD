# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Portable, JSON-exchangeable cycle definitions.

Notes
-----
Requires the optional `portable` extra (`pydantic`). Other tools need only
the JSON Schema from `CycleDefinition.model_json_schema()`, not BLonD.
"""

__all__ = [
    "SCHEMA_VERSION",
    "CycleDefinition",
    "DriftDefinition",
    "MagneticProgramme",
    "ParticleDefinition",
    "Programme",
    "RFHarmonicDefinition",
    "RFStationDefinition",
]

try:
    import pydantic  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "`blond.interfaces.portable` requires pydantic; install it with "
        "`pip install blond[portable]`."
    ) from exc

from blond.interfaces.portable.cycle_definition import (
    SCHEMA_VERSION,
    CycleDefinition,
    DriftDefinition,
    MagneticProgramme,
    ParticleDefinition,
    Programme,
    RFHarmonicDefinition,
    RFStationDefinition,
)
