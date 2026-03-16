# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


import sympy as sp


def poisson_bracket(F, G, q, p):
    """
    Poisson bracket expansion.

    Parameters
    ----------
    F
    G
    q
    p

    Returns
    -------

    """
    return sp.diff(F, p) * sp.diff(G, q) - sp.diff(F, q) * sp.diff(G, p)


def cbh_two(A, B, q, p, order=2):
    """
    Compute log(exp(A) exp(B)) using CBH expansion
    """

    PB = lambda X, Y: poisson_bracket(X, Y, q, p)

    H = A + B

    if order >= 2:
        H += sp.Rational(1, 2) * PB(A, B)

    if order >= 3:
        H += sp.Rational(1, 12) * PB(A, PB(A, B))
        H -= sp.Rational(1, 12) * PB(B, PB(A, B))

    if order >= 4:
        H -= sp.Rational(1, 24) * PB(B, PB(A, PB(A, B)))

    return H


def cbh_lattice(hamiltonians, q, p, order=2):
    """
    Combine a list of Hamiltonians into one effective generator
    using recursive CBH.
    """

    if len(hamiltonians) == 0:
        return 0

    H_eff = hamiltonians[0]

    for H_next in hamiltonians[1:]:
        H_eff = cbh_two(H_eff, H_next, q, p, order=order)

    return H_eff
