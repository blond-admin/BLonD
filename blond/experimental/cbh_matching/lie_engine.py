# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import sympy as sp


class LieEngine:
    def __init__(self, coordinates):
        self.coords = coordinates
        self.n = len(coordinates) // 2

        self.q = coordinates[: self.n]
        self.p = coordinates[self.n :]

    def poisson(self, A, B):
        pb = 0

        for qi, pi in zip(self.q, self.p):
            pb += sp.diff(A, qi) * sp.diff(B, pi) - sp.diff(A, pi) * sp.diff(
                B, qi
            )

        return sp.simplify(pb)

    def lie(self, H, F):
        return self.poisson(F, H)

    def lie_power(self, H, F, n):
        result = F

        for _ in range(n):
            result = self.lie(H, result)

        return result

    def lie_exp(self, H, F, t, order=6):
        result = F

        for k in range(1, order + 1):
            result += (t**k / sp.factorial(k)) * self.lie_power(H, F, k)

        return sp.simplify(result)

    def compile(self, expr, parameters=()):
        vars_all = list(self.coords) + list(parameters)

        return sp.lambdify(vars_all, expr, "numpy")
