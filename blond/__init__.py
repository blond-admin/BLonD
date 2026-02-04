# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""BLonD beam dynamics software."""

__all__ = [
    "BiGaussian",
    "Cupy32Bit",
    "Cupy64Bit",
    "Numpy32Bit",
    "Numpy64Bit",
    "backend",
    "UserDefinedElement",
    "Beam",
    "EmptyBeam",
    "electron",
    "mu_minus",
    "mu_plus",
    "positron",
    "proton",
    "uranium_29",
    "Ring",
    "Simulation",
    "ConstantMagneticCycle",
    "MagneticCycleByTime",
    "MagneticCyclePerTurn",
    "MagneticCyclePerTurnAllRFStations",
    "AllowPlotting",
    "BeamObservationOncePerTurn",
    "RFStationPhaseObservation",
    "StaticProfileObservation",
    "BeamObservationInRingElement",
    "MultiHarmonicRFStation",
    "SingleHarmonicRFStation",
    "DriftSimple",
    "ReferenceEnergyChange",
    "WakeField",
    "InductiveImpedanceSolver",
    "PeriodicFreqSolver",
    "TimeDomainFftSolver",
    "ImpedanceTableFreq",
    "InductiveImpedance",
    "Resonators",
    "BoxLosses",
    "DynamicProfileConstNBins",
    "StaticProfile",
    "DriftObservation",
    "SimulationObservation",
]
from blond.beam_preparation.bigaussian import BiGaussian
from blond.core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    Numpy32Bit,
    Numpy64Bit,
    backend,
)
from blond.core.base import UserDefinedElement
from blond.core.beam.beams import Beam, EmptyBeam
from blond.core.beam.particle_types import (
    electron,
    mu_minus,
    mu_plus,
    positron,
    proton,
    uranium_29,
)
from blond.core.ring.ring import Ring
from blond.core.simulation.simulation import Simulation
from blond.cycles.magnetic_cycle import (
    ConstantMagneticCycle,
    MagneticCycleByTime,
    MagneticCyclePerTurn,
    MagneticCyclePerTurnAllRFStations,
)
from blond.generals.cupy.no_cupy_import import AllowPlotting
from blond.handle_results.observables import (
    BeamObservationOncePerTurn,
    DriftObservation,
    RFStationPhaseObservation,
    SimulationObservation,
    StaticProfileObservation,
)
from blond.handle_results.observables_as_elements import (
    BeamObservationInRingElement,
)
from blond.physics.cavities import (
    MultiHarmonicRFStation,
    SingleHarmonicRFStation,
)
from blond.physics.drifts import DriftSimple
from blond.physics.energy_reference_kick import ReferenceEnergyChange
from blond.physics.impedances.base import WakeField
from blond.physics.impedances.solvers import (
    InductiveImpedanceSolver,
    PeriodicFreqSolver,
    TimeDomainFftSolver,
)
from blond.physics.impedances.sources import (
    ImpedanceTableFreq,
    InductiveImpedance,
    Resonators,
)
from blond.physics.losses import BoxLosses
from blond.physics.profiles import (
    DynamicProfileConstNBins,
    StaticProfile,
)
