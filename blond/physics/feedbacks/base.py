# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of implementations to model longitudinal feedbacks.

Authors
-------
Birk Karlsen Baeck
Leonard Thiele
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from blond.core.base import BeamPhysicsRelevant
from blond.core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.physics.cavities import (
        MultiHarmonicRfStation,
        RfStationBaseClass,
        SingleHarmonicRfStation,
    )
    from blond.physics.profiles import ProfileBaseClass


class FeedbackBaseClass(BeamPhysicsRelevant):
    """
    Baseclass for implementation of feedback elements.

    Parameters
    ----------
    section_index
        section index of the feedback
    name
        name of the feedback
    """

    def __init__(
        self,
        section_index: int = 0,
        name: str | None = None,
    ):
        super().__init__(section_index=section_index, name=name)


class LocalFeedback(FeedbackBaseClass):
    """
    Baseclass for implementation of local feedback elements.

    Baseclass for implementation of local feedback elements.
    This will be attached to a specific multi- or single-harmonic
    cavity, which will act on the beam.

    Parameters
    ----------
    name
        name of the feedback

    Attributes
    ----------
    phase_correction
        correction to the rf phase, has to be defined on the
        profile time grid.
    relative_voltage_correction
        relative correction to the setpoint voltage
        stemming from the feedback,
        has to be defined on the profile time grid.
    """

    def __init__(
        self,
        profile: ProfileBaseClass,
        name: str | None = None,
    ):
        super().__init__(
            name=name,
        )
        self._parent_rf_station: (
            SingleHarmonicRfStation | MultiHarmonicRfStation | None
        ) = None

        self.relative_voltage_correction: NumpyArray | None = None
        self.phase_correction: NumpyArray | None = None

        self.profile = profile

    def set_parent_rf_station(
        self, rf_station: MultiHarmonicRfStation | SingleHarmonicRfStation
    ) -> None:
        """
        Sets the parent RF station on initialization of the rf_station.

        Parameters
        ----------
        rf_station
            cavity to be the parent rf station

        """
        from blond.physics.cavities import (  # no cyclic import
            MultiHarmonicRfStation,
            SingleHarmonicRfStation,
        )

        assert self._parent_rf_station is None, (
            "This feedback has already one owner!"
        )
        if not isinstance(
            rf_station, SingleHarmonicRfStation | MultiHarmonicRfStation
        ):
            raise ValueError(
                f"Local feedbacks can only be initialized with SingleHarmonicRfStation "
                f"or MultiHarmonicRfStation but not {type(rf_station)}"
            )
        self._parent_rf_station = rf_station
        self._section_index = self._parent_rf_station.section_index

    @abstractmethod  # pragma: no cover
    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass


class GlobalFeedback(FeedbackBaseClass):
    """
    Baseclass for implementation of global feedback elements.

    Parameters
    ----------
    profile
        profile on which the feedback should work, needs to be tracked before.
    section_index
        section index of the feedback
    name
        name of the feedback
    """

    def __init__(
        self,
        profile: ProfileBaseClass,
        section_index: int = 0,
        name: str | None = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.profile = profile
        self.cavities: list[RfStationBaseClass] | None = None

    # Use `requires` to automatically sort execution order of
    # `element.on_init_simulation` for all elements
    @requires(["RfStationBaseClass"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        simulation
            `Simulation` context manager
        """
        from blond.physics.cavities import RfStationBaseClass

        self.cavities = simulation.ring.elements.get_elements(
            RfStationBaseClass,
        )
