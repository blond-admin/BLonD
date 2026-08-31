# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Pin the ``scipy.constants`` values that BLonD imports.

``m_e``, ``m_p``, ``epsilon_0``, ``hbar`` and the muon mass are CODATA
measured values, not SI-defined constants, so a scipy upgrade that ships
an updated CODATA revision changes them silently. Since these feed
straight into particle masses and radiation constants
(``blond/core/beam/particle_types.py``), a silent shift here would
silently shift physics results.

This happened in practice: a scipy version bump changed a few digits
of these constants and broke the test suite in ways that took a long
time to track down, since nothing pointed at "scipy constants changed"
as the cause. These tests fail loudly and specifically if that ever
happens again, so the failure points straight at the changed constant
instead of surfacing as a confusing, hard-to-localize downstream
mismatch.
"""

import unittest

from scipy.constants import (
    c,
    e,
    elementary_charge,
    epsilon_0,
    hbar,
    m_e,
    m_p,
    physical_constants,
    speed_of_light,
)


class TestScipyConstants(unittest.TestCase):
    """Pin the values of scipy constants used across ``blond/``."""

    def test_speed_of_light(self):
        # SI-defined (2019 redefinition), exact.
        self.assertEqual(c, 299792458.0)
        self.assertEqual(speed_of_light, 299792458.0)

    def test_elementary_charge(self):
        # SI-defined (2019 redefinition), exact.
        self.assertEqual(e, 1.602176634e-19)
        self.assertEqual(elementary_charge, 1.602176634e-19)

    def test_electron_mass(self):
        # CODATA measured value, not SI-defined.
        self.assertEqual(m_e, 9.1093837139e-31)

    def test_proton_mass(self):
        # CODATA measured value, not SI-defined.
        self.assertEqual(m_p, 1.67262192595e-27)

    def test_vacuum_permittivity(self):
        # CODATA measured value, not SI-defined.
        self.assertEqual(epsilon_0, 8.8541878188e-12)

    def test_reduced_planck_constant(self):
        # CODATA measured value, not SI-defined.
        self.assertEqual(hbar, 1.0545718176461565e-34)

    def test_muon_mass(self):
        # CODATA measured value, not SI-defined.
        mass, unit, uncertainty = physical_constants["muon mass"]
        self.assertEqual(mass, 1.883531627e-28)
        self.assertEqual(unit, "kg")
        self.assertEqual(uncertainty, 4.2e-36)


if __name__ == "__main__":
    unittest.main()
