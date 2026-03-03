from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as cont

from blond.acc_math.analytic import conversions as conv
from blond.core.backends import backend
from blond.testing.backend_testing import multi_backend_testcase

if TYPE_CHECKING:
    ...


class TestConversionFunctions(unittest.TestCase):
    @multi_backend_testcase
    def test_magnetic_rigidity_to_momentum(self):
        rigidity = 1
        charge = 1

        momentum = conv.magnetic_rigidity_to_momentum(rigidity, charge)
        target = charge * rigidity * cont.c

        self.assertEqual(momentum, target)

        rigidity = backend.backend.array(
            [1, 2, 3], dtype=backend.backend.float
        )

        momentum = conv.magnetic_rigidity_to_momentum(rigidity, charge).astype(
            backend.backend.float
        )

        target = charge * rigidity * cont.c
        if isinstance(backend.backend, backend.CupyBackend):
            momentum = momentum.get()
            target = target.get()

        np.testing.assert_array_equal(momentum, target)

    @multi_backend_testcase
    def test_beta_to_gamma(self):
        beta = 0
        gamma = conv.beta_to_gamma(beta)
        self.assertEqual(gamma, 1)

        beta = 0.5
        gamma = conv.beta_to_gamma(beta)
        self.assertEqual(gamma, 1 / np.sqrt(1 - beta**2))

        beta = backend.backend.array([0, 0.25, 0.5, 0.9])
        gamma = conv.beta_to_gamma(beta)

        target = 1 / np.sqrt(1 - beta**2)
        if isinstance(backend.backend, backend.CupyBackend):
            gamma = gamma.get()
            target = target.get()

        np.testing.assert_array_equal(gamma, target)

    @multi_backend_testcase
    def test_gamma_to_beta(self):
        gamma = 1
        beta = conv.gamma_to_beta(gamma)
        self.assertEqual(beta, 0)

        gamma = 10
        beta = conv.gamma_to_beta(gamma)
        self.assertEqual(beta, np.sqrt(1 - 1 / gamma**2))

        gamma = backend.backend.array([1, 10, 100, 100])
        beta = conv.gamma_to_beta(gamma)

        target = np.sqrt(1 - 1 / gamma**2)
        if isinstance(backend.backend, backend.CupyBackend):
            beta = beta.get()
            target = target.get()

        np.testing.assert_array_equal(beta, target)

    @multi_backend_testcase
    def test_frev_to_beta(self):
        circ = 1e3
        frev = 100e3

        beta = conv.frev_to_beta(frev, circ)

        self.assertEqual(beta, frev * circ / cont.c)

        frev = backend.backend.array([50e3, 100e3, 200e3])
        beta = conv.frev_to_beta(frev, circ)

        target = circ * frev / cont.c
        if isinstance(backend.backend, backend.CupyBackend):
            beta = beta.get()
            target = target.get()

        np.testing.assert_array_equal(beta, target)

    @multi_backend_testcase
    def test_beta_to_frev(self):
        circ = 1e3
        beta = 1

        frev = conv.beta_to_frev(beta, circ)

        self.assertEqual(frev, cont.c / circ)

        beta = backend.backend.array([0.1, 0.5, 1])
        frev = conv.beta_to_frev(beta, circ)

        target = beta * cont.c / circ
        if isinstance(backend.backend, backend.CupyBackend):
            frev = frev.get()
            target = target.get()

        np.testing.assert_array_equal(frev, target)

    @multi_backend_testcase
    def test_beta_to_trev(self):
        circ = 1e3
        beta = 1

        trev = conv.beta_to_trev(beta, circ)

        self.assertEqual(trev, circ / cont.c)

        beta = backend.backend.array([0.1, 0.5, 1])
        trev = conv.beta_to_trev(beta, circ)

        target = circ / (beta * cont.c)
        if isinstance(backend.backend, backend.CupyBackend):
            trev = trev.get()
            target = target.get()

        np.testing.assert_array_equal(trev, target)

    @multi_backend_testcase
    def test_momentum_to_beta(self):
        proton_mass = cont.physical_constants[
            "proton mass energy equivalent in MeV"
        ][0]

        lead_mass = 208 * proton_mass  # Approximate Pb-208

        momentum = 1e9
        beta_proton = conv.momentum_to_beta(momentum, proton_mass)
        beta_lead = conv.momentum_to_beta(momentum, lead_mass)

        self.assertEqual(
            beta_proton, 1 / np.sqrt(1 + proton_mass**2 / momentum**2)
        )
        self.assertEqual(
            beta_lead, 1 / np.sqrt(1 + lead_mass**2 / momentum**2)
        )

        momentum = backend.backend.array([1e9, 100e9, 10e12])

        beta_proton = conv.momentum_to_beta(momentum, proton_mass)
        beta_lead = conv.momentum_to_beta(momentum, lead_mass)

        target_proton = 1 / np.sqrt(1 + proton_mass**2 / momentum**2)
        target_lead = 1 / np.sqrt(1 + lead_mass**2 / momentum**2)
        if isinstance(backend.backend, backend.CupyBackend):
            beta_proton = beta_proton.get()
            beta_lead = beta_lead.get()
            target_proton = target_proton.get()
            target_lead = target_lead.get()

        np.testing.assert_array_equal(beta_proton, target_proton)
        np.testing.assert_array_equal(beta_lead, target_lead)

    @multi_backend_testcase
    def test_momentum_to_gamma(self):
        proton_mass = cont.physical_constants[
            "proton mass energy equivalent in MeV"
        ][0]

        lead_mass = 208 * proton_mass  # Approximate Pb-208

        momentum = 1e9
        beta_proton = conv.momentum_to_gamma(momentum, proton_mass)
        beta_lead = conv.momentum_to_gamma(momentum, lead_mass)

        self.assertEqual(
            beta_proton, np.sqrt((momentum / proton_mass) ** 2 + 1)
        )
        self.assertEqual(beta_lead, np.sqrt((momentum / lead_mass) ** 2 + 1))

        momentum = backend.backend.array([1e9, 100e9, 10e12])

        beta_proton = conv.momentum_to_gamma(momentum, proton_mass)
        beta_lead = conv.momentum_to_gamma(momentum, lead_mass)

        target_proton = np.sqrt((momentum / proton_mass) ** 2 + 1)
        target_lead = np.sqrt((momentum / lead_mass) ** 2 + 1)
        if isinstance(backend.backend, backend.CupyBackend):
            beta_proton = beta_proton.get()
            beta_lead = beta_lead.get()
            target_proton = target_proton.get()
            target_lead = target_lead.get()

        np.testing.assert_array_equal(beta_proton, target_proton)
        np.testing.assert_array_equal(beta_lead, target_lead)

    def test_momentum_beta_gamma_circular(self):
        beta = 0.5
        gamma = conv.beta_to_gamma(beta)
        self.assertAlmostEqual(beta, conv.gamma_to_beta(gamma), places=9)

        gamma = 100
        beta = conv.gamma_to_beta(gamma)
        self.assertAlmostEqual(gamma, conv.beta_to_gamma(beta), places=9)

        momentum = 10e6
        mass = cont.physical_constants["proton mass energy equivalent in MeV"][
            0
        ]

        mom_gamma = conv.momentum_to_gamma(momentum, mass)
        mom_beta = conv.momentum_to_beta(momentum, mass)

        self.assertAlmostEqual(
            mom_beta, conv.gamma_to_beta(mom_gamma), delta=1e-4
        )
        self.assertAlmostEqual(
            mom_gamma, conv.beta_to_gamma(mom_beta), delta=1e-4
        )

    @multi_backend_testcase
    def test_momentum_to_frev_trev(self):
        beta = 0.5
        gamma = conv.beta_to_gamma(beta)
        mass = cont.physical_constants["proton mass energy equivalent in MeV"][
            0
        ]
        momentum = gamma * mass * beta

        circ = 1e3

        t_rev = circ / (cont.c * beta)
        f_rev = (beta * cont.c) / circ

        self.assertAlmostEqual(
            t_rev, conv.momentum_to_trev(momentum, circ, mass)
        )
        self.assertAlmostEqual(
            f_rev, conv.momentum_to_frev(momentum, circ, mass)
        )

        beta = backend.backend.array([0.1, 0.5, 1])
        gamma = conv.beta_to_gamma(beta)
        mass = cont.physical_constants["proton mass energy equivalent in MeV"][
            0
        ]
        momentum = gamma * mass * beta

        circ = 1e3

        t_rev_target = circ / (cont.c * beta)
        f_rev_target = (beta * cont.c) / circ
        t_rev = conv.momentum_to_trev(momentum, circ, mass)
        f_rev = conv.momentum_to_frev(momentum, circ, mass)

        if isinstance(backend.backend, backend.CupyBackend):
            t_rev_target = t_rev_target.get()
            f_rev_target = f_rev_target.get()
            t_rev = t_rev.get()
            f_rev = f_rev.get()

        np.testing.assert_array_almost_equal(t_rev_target, t_rev)
        np.testing.assert_array_almost_equal(f_rev_target, f_rev)

    @multi_backend_testcase
    def test_momentum_energy_relations(self):
        momentum = 1e9
        mass = cont.physical_constants["proton mass energy equivalent in MeV"][
            0
        ]

        total_energy = conv.momentum_to_total_energy(momentum, mass)
        kinetic_energy = conv.momentum_to_kinetic_energy(momentum, mass)

        self.assertAlmostEqual(total_energy, kinetic_energy + mass)
        self.assertAlmostEqual(total_energy, np.sqrt(momentum**2 + mass**2))

        momentum = backend.backend.array([1e9, 100e9, 10e12])
        total_energy = conv.momentum_to_total_energy(momentum, mass)
        kinetic_energy = conv.momentum_to_kinetic_energy(momentum, mass)

        if isinstance(backend.backend, backend.CupyBackend):
            momentum = momentum.get()
            total_energy = total_energy.get()
            kinetic_energy = kinetic_energy.get()

        np.testing.assert_array_almost_equal(
            total_energy, kinetic_energy + mass
        )
        np.testing.assert_array_almost_equal(
            total_energy, np.sqrt(momentum**2 + mass**2)
        )

    @multi_backend_testcase
    def test_momentum_to_magnetic_field(self):
        momentum = 1e9
        bend_rad = 100
        charge = 1

        field = conv.momentum_to_magnetic_field(momentum, bend_rad, charge)

        self.assertEqual(field, momentum / (bend_rad * charge * cont.c))

        momentum = backend.backend.array([1e9, 100e9, 10e12])

        field = conv.momentum_to_magnetic_field(momentum, bend_rad, charge)

        if isinstance(backend.backend, backend.CupyBackend):
            momentum = momentum.get()
            field = field.get()

        np.testing.assert_array_almost_equal(
            field, momentum / (bend_rad * charge * cont.c)
        )

    @multi_backend_testcase
    def test_total_energy_energy_momentum_relations(self):
        total_energy = 1e12
        mass = cont.physical_constants["proton mass energy equivalent in MeV"][
            0
        ]
        kinetic_energy = conv.total_energy_to_kinetic_energy(
            total_energy, mass
        )
        momentum = conv.total_energy_to_momentum(total_energy, mass)

        self.assertAlmostEqual(kinetic_energy, total_energy - mass)
        self.assertAlmostEqual(momentum, np.sqrt(total_energy**2 - mass**2))

        total_energy = backend.backend.array([2e9, 100e9, 10e12])
        kinetic_energy = conv.total_energy_to_kinetic_energy(
            total_energy, mass
        )
        momentum = conv.total_energy_to_momentum(total_energy, mass)

        if isinstance(backend.backend, backend.CupyBackend):
            total_energy = total_energy.get()
            kinetic_energy = kinetic_energy.get()
            momentum = momentum.get()

        np.testing.assert_array_almost_equal(
            total_energy, kinetic_energy + mass
        )
        np.testing.assert_array_almost_equal(
            total_energy, np.sqrt(momentum**2 + mass**2)
        )

    @multi_backend_testcase
    def test_total_energy_to_magnetic_field(self):
        total_energy = 1e12
        mass = cont.physical_constants["proton mass energy equivalent in MeV"][
            0
        ]
        bend_rad = 100
        charge = 1

        field = conv.total_energy_to_magnetic_field(
            total_energy, bend_rad, charge, mass
        )

        momentum = np.sqrt(total_energy**2 - mass**2)
        target_field = conv.momentum_to_magnetic_field(
            momentum, bend_rad, charge
        )

        self.assertAlmostEqual(field, target_field)

        total_energy = backend.backend.array([2e9, 100e9, 10e12])
        field = conv.total_energy_to_magnetic_field(
            total_energy, bend_rad, charge, mass
        )

        momentum = np.sqrt(total_energy**2 - mass**2)
        target_field = conv.momentum_to_magnetic_field(
            momentum, bend_rad, charge
        )

        if isinstance(backend.backend, backend.CupyBackend):
            field = field.get()
            target_field = target_field.get()

        np.testing.assert_array_almost_equal(field, target_field)

    @multi_backend_testcase
    def test_total_energy_beta_gamma_relations(self):
        total_energy = 1e12
        mass = cont.physical_constants["proton mass energy equivalent in MeV"][
            0
        ]

        beta = conv.total_energy_to_beta(total_energy, mass)
        gamma = conv.total_energy_to_gamma(total_energy, mass)

        momentum = np.sqrt(total_energy**2 - mass**2)

        self.assertAlmostEqual(
            beta, conv.momentum_to_beta(momentum, mass), delta=1e-5
        )
        self.assertAlmostEqual(
            gamma, conv.momentum_to_gamma(momentum, mass), delta=1e-5
        )

        total_energy = backend.backend.array([2e9, 100e9, 10e12])

        beta = conv.total_energy_to_beta(total_energy, mass)
        gamma = conv.total_energy_to_gamma(total_energy, mass)

        momentum = np.sqrt(total_energy**2 - mass**2)
        target_beta = conv.momentum_to_beta(momentum, mass)
        target_gamma = conv.momentum_to_gamma(momentum, mass)

        if isinstance(backend.backend, backend.CupyBackend):
            beta = beta.get()
            gamma = gamma.get()
            target_beta = target_beta.get()
            target_gamma = target_gamma.get()

        np.testing.assert_array_almost_equal(beta, target_beta, decimal=5)
        np.testing.assert_array_almost_equal(gamma, target_gamma, decimal=5)

    @multi_backend_testcase
    def test_kinetic_energy_momentum_energy_relations(self):
        kinetic_energy = 1e9
        mass = cont.physical_constants["proton mass energy equivalent in MeV"][
            0
        ]

        momentum = conv.kinetic_energy_to_momentum(kinetic_energy, mass)
        total_energy = conv.kinetic_energy_to_total_energy(
            kinetic_energy, mass
        )

        self.assertAlmostEqual(total_energy, kinetic_energy + mass)
        self.assertAlmostEqual(
            momentum, np.sqrt((kinetic_energy + mass) ** 2 - mass**2)
        )

        kinetic_energy = backend.backend.array([1e9, 100e9, 10e12])
        momentum = conv.kinetic_energy_to_momentum(kinetic_energy, mass)
        total_energy = conv.kinetic_energy_to_total_energy(
            kinetic_energy, mass
        )

        target_momentum = np.sqrt((kinetic_energy + mass) ** 2 - mass**2)
        target_total_energy = kinetic_energy + mass

        if isinstance(backend.backend, backend.CupyBackend):
            momentum = momentum.get()
            total_energy = total_energy.get()
            target_momentum = target_momentum.get()
            target_total_energy = target_total_energy.get()

        np.testing.assert_array_almost_equal(momentum, target_momentum)
        np.testing.assert_array_almost_equal(total_energy, target_total_energy)

    @multi_backend_testcase
    def test_kinetic_energy_to_magnetic_field(self):
        kinetic_energy = 1e9
        mass = cont.physical_constants["proton mass energy equivalent in MeV"][
            0
        ]
        bend_rad = 100
        charge = 1

        field = conv.kinetic_energy_to_magnetic_field(
            kinetic_energy, bend_rad, charge, mass
        )

        momentum = np.sqrt((kinetic_energy + mass) ** 2 - mass**2)
        target_field = conv.momentum_to_magnetic_field(
            momentum, bend_rad, charge
        )

        self.assertAlmostEqual(field, target_field)

        kinetic_energy = backend.backend.array([1e9, 100e9, 10e12])
        field = conv.kinetic_energy_to_magnetic_field(
            kinetic_energy, bend_rad, charge, mass
        )

        momentum = np.sqrt((kinetic_energy + mass) ** 2 - mass**2)
        target_field = conv.momentum_to_magnetic_field(
            momentum, bend_rad, charge
        )

        if isinstance(backend.backend, backend.CupyBackend):
            field = field.get()
            target_field = target_field.get()

        np.testing.assert_array_almost_equal(field, target_field)

    @multi_backend_testcase
    def test_field_momentum_energy_relations(self):
        field = 1
        mass = cont.physical_constants["proton mass energy equivalent in MeV"][
            0
        ]
        bend_rad = 100
        charge = 1

        momentum = conv.magnetic_field_to_momentum(field, bend_rad, charge)
        total_energy = conv.magnetic_field_to_total_energy(
            field, bend_rad, charge, mass
        )
        kinetic_energy = conv.magnetic_field_to_kinetic_energy(
            field, bend_rad, charge, mass
        )

        self.assertAlmostEqual(momentum, charge * field * bend_rad * cont.c)
        self.assertAlmostEqual(total_energy, np.sqrt(momentum**2 + mass**2))
        self.assertAlmostEqual(
            kinetic_energy, np.sqrt(momentum**2 + mass**2) - mass
        )

        field = backend.backend.array([0.5, 1, 5, 10])
        momentum = conv.magnetic_field_to_momentum(field, bend_rad, charge)
        total_energy = conv.magnetic_field_to_total_energy(
            field, bend_rad, charge, mass
        )
        kinetic_energy = conv.magnetic_field_to_kinetic_energy(
            field, bend_rad, charge, mass
        )

        target_momentum = charge * field * bend_rad * cont.c
        target_total_energy = np.sqrt(momentum**2 + mass**2)
        target_kinetic_energy = np.sqrt(momentum**2 + mass**2) - mass

        if isinstance(backend.backend, backend.CupyBackend):
            momentum = momentum.get()
            total_energy = total_energy.get()
            kinetic_energy = kinetic_energy.get()
            target_momentum = target_momentum.get()
            target_total_energy = target_total_energy.get()
            target_kinetic_energy = target_kinetic_energy.get()

        np.testing.assert_array_almost_equal(momentum, target_momentum)
        np.testing.assert_array_almost_equal(total_energy, target_total_energy)
        np.testing.assert_array_almost_equal(
            kinetic_energy, target_kinetic_energy
        )

    @multi_backend_testcase
    def test_delta_P_delta_E(self):
        momentum = 1e9
        mass = cont.physical_constants["proton mass energy equivalent in MeV"][
            0
        ]
        total_energy = np.sqrt(momentum**2 + mass**2)

        delta_P = 1e-3 * momentum
        delta_E = conv.delta_P_to_delta_E(delta_P, momentum, mass)

        self.assertAlmostEqual(
            delta_E,
            np.sqrt(mass**2 + (delta_P + momentum) ** 2) - total_energy,
            delta=1e-5,
        )

        delta_E = 1e-3 * total_energy
        delta_P = conv.delta_E_to_delta_P(delta_E, total_energy, mass)
        self.assertAlmostEqual(
            delta_P,
            np.sqrt((total_energy + delta_E) ** 2 - mass**2) - momentum,
            delta=1e-5,
        )

        momentum = backend.backend.array(
            [1e9, 100e9, 10e12, 1e9, 100e9, 10e12]
        )
        total_energy = np.sqrt(momentum**2 + mass**2)
        delta_P = (
            backend.backend.array([1e-4, 1e-3, 1e-2, 1e-2, 1e-3, 1e-4])
            * momentum
        )
        delta_E = conv.delta_P_to_delta_E(delta_P, momentum, mass)

        target_delta_E = (
            np.sqrt(mass**2 + (delta_P + momentum) ** 2) - total_energy
        )

        if isinstance(backend.backend, backend.CupyBackend):
            delta_E = delta_E.get()
            target_delta_E = target_delta_E.get()

        np.testing.assert_array_almost_equal(
            delta_E, target_delta_E, decimal=5
        )

        momentum = backend.backend.array(
            [1e9, 100e9, 10e12, 1e9, 100e9, 10e12]
        )
        total_energy = np.sqrt(momentum**2 + mass**2)
        delta_E = (
            backend.backend.array([1e-4, 1e-3, 1e-2, 1e-2, 1e-3, 1e-4])
            * total_energy
        )
        delta_P = conv.delta_E_to_delta_P(delta_P, momentum, mass)

        target_delta_P = (
            np.sqrt((total_energy + delta_E) ** 2 - mass**2) - momentum
        )

        if isinstance(backend.backend, backend.CupyBackend):
            delta_P = delta_P.get()
            target_delta_P = target_delta_P.get()

        np.testing.assert_array_almost_equal(
            delta_P, target_delta_P, decimal=5
        )


if __name__ == "__main__":
    unittest.main()
