# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for llrf.filters

:Authors: **Helga Timko**
"""

import unittest
import numpy as np
from llrf.impulse_response import rectangle, triangle



class TestRectangle(unittest.TestCase):

    def test_1(self):
        

        tau = 1.
        time = np.array([-1, -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1])
        rect_exp = np.array([0., 0., 0.5, 1., 1., 1., 0.5, 0., 0.])
        rect_act = rectangle(time, tau)
        
        rect_exp = np.around(rect_exp, 12)
        rect_act = np.around(rect_act, 12)
        self.assertSequenceEqual(rect_exp.tolist(), rect_act.tolist(),
            msg="In TestRectangle test 1: rectangle arrays differ")

    def test_2(self):

        tau = 1.
        time = np.array([-0.51, -0.26, 0.01, 0.26, 0.51, 0.76, 1.01])
        rect_exp = np.array([0.5, 1., 1., 1., 0.5, 0., 0.])
        rect_act = rectangle(time, tau)
        
        rect_exp = np.around(rect_exp, 12)
        rect_act = np.around(rect_act, 12)
        self.assertSequenceEqual(rect_exp.tolist(), rect_act.tolist(),
            msg="In TestRectangle test 2: rectangle arrays differ")



class TestTriangle(unittest.TestCase):

    def test_1(self):

        tau = 1.
        time = np.array([-0.5, -0.25, 0., 0.25, 0.5, 0.75, 1, 1.25, 1.5])
        tri_exp = np.array([0., 0., 0.5, 0.75, 0.5, 0.25, 0., 0., 0.])
        tri_act = triangle(time, tau)
        
        tri_exp = np.around(tri_exp, 12)
        tri_act = np.around(tri_act, 12)
        self.assertSequenceEqual(tri_exp.tolist(), tri_act.tolist(),
            msg="In TestTriangle test 1: triangle arrays differ")

    def test_2(self):

        tau = 1.
        time = np.array([-0.01, 0.26, 0.51, 0.76, 1.01, 1.26, 1.51])
        tri_exp = np.array([0.5, 0.74, 0.49, 0.24, 0., 0., 0.])
        tri_act = triangle(time, tau)
        
        tri_exp = np.around(tri_exp, 12)
        tri_act = np.around(tri_act, 12)
        self.assertSequenceEqual(tri_exp.tolist(), tri_act.tolist(),
            msg="In TestTriangle test 2: triangle arrays differ")



if __name__ == '__main__':

    unittest.main()



