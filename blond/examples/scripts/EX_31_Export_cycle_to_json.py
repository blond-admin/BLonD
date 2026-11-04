# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Convert BLonD objects to a portable definition and print it as JSON."""

from blond import (
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)
from blond.interfaces.portable import CycleDefinition

circumference = 26658.883


def main():
    ring = Ring(circumference)
    ring.add_elements(
        [
            SingleHarmonicRFStation(harmonic=35640, voltage=6e6, phi_rf=0),
            DriftSimple(
                orbit_length=circumference,
                momentum_compaction_factor=momentum_compaction_factor(
                    transition_gamma=55.759505
                ),
            ),
        ]
    )
    magnetic_cycle = ConstantMagneticCycle(
        value=450e9, reference_particle=proton
    )

    cycle = CycleDefinition.from_ring(ring, magnetic_cycle, name="LHC")
    print(cycle.model_dump_json(indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
