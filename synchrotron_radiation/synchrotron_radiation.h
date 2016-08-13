/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3), 
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities 
granted to it by virtue of its status as an Intergovernmental Organization or 
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Author: Juan F. Esteban Mueller

extern "C" void synchrotron_radiation(double * __restrict__ beam_dE, const double U0, 
                                const int n_macroparticles, const double tau_z,
                                const int n_kicks);
