# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""BLonD beam dynamics software."""

from blond.beam_preparation.bigaussian import BiGaussian  # NOQA
from blond.core.backends.backend import (  # NOQA
    Cupy32Bit,
    Cupy64Bit,
    Numpy32Bit,
    Numpy64Bit,
    backend,
)
from blond.core.base import UserDefinedElement  # NOQA
from blond.core.beam.beams import Beam  # NOQA
from blond.core.beam.particle_types import electron  # NOQA
from blond.core.beam.particle_types import mu_minus  # NOQA
from blond.core.beam.particle_types import mu_plus  # NOQA
from blond.core.beam.particle_types import positron  # NOQA
from blond.core.beam.particle_types import proton  # NOQA
from blond.core.beam.particle_types import uranium_29  # NOQA
from blond.core.ring.ring import Ring  # NOQA
from blond.core.simulation.simulation import Simulation  # NOQA
from blond.cycles.magnetic_cycle import ConstantMagneticCycle  # NOQA
from blond.cycles.magnetic_cycle import MagneticCycleByTime  # NOQA
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn  # NOQA
from blond.cycles.magnetic_cycle import (
    MagneticCyclePerTurnAllRFStations,  # NOQA
)
from blond.generals.cupy.no_cupy_import import AllowPlotting  # NOQA
from blond.handle_results.observables import BeamObservationOncePerTurn  # NOQA
from blond.handle_results.observables import RFStationPhaseObservation  # NOQA
from blond.handle_results.observables import StaticProfileObservation  # NOQA
from blond.handle_results.observables_as_elements import (
    BeamObservationInRingElement,  # NOQA
)
from blond.physics.cavities import (  # NOQA
    MultiHarmonicRFStation,
    SingleHarmonicRFStation,
)
from blond.physics.drifts import DriftSimple  # NOQA
from blond.physics.energy_reference_kick import ReferenceEnergyChange  # NOQA
from blond.physics.impedances.base import WakeField  # NOQA
from blond.physics.losses import BoxLosses  # NOQA
from blond.physics.profiles import StaticProfile  # NOQA
