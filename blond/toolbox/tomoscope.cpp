/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Generation of particle distribution from probability density
// Author: Helga Timko

#include <stdlib.h>
using uint = unsigned int;

double randd() {
    return (double)rand() / (RAND_MAX + 1.0);
}


extern "C" void generate_distribution(double * __restrict__ dt,
                                      double * __restrict__ dE, const double * probDistr, const uint seed,
                                      const uint profLen, const double cutoff, const double x0, const double y0,
                                      const double dtBin, const double dEBin, const uint nPart) {


// Initialise random seed
    srand(seed);

// Initialise some variables
    uint i, k, m;
    double iPos, kPos;
    double cutoff2 = cutoff * cutoff;
    double dtMin = -1.*x0 * dtBin;
    double dEMin = -1.*y0 * dEBin;


// Calculate cumulative probability
    double totProb = 0.;
    uint profLen2 = uint(profLen * profLen);
    double cumulDistr [profLen2];

    for (m = 0; m < profLen2; m++) {
        cumulDistr[m] = probDistr[m] + totProb;
        totProb += probDistr[m];
    }


// Normalise probability distribution
    double invTotProb = 1. / totProb;

    for (m = 0; m < profLen2; m++) {
        cumulDistr[m] *= invTotProb;
    }


// Generate particle coordinates
    uint n = 0;
    double randProb;

    while ( n < nPart ) {
        randProb = randd();
        for (m = 0; m < profLen2; m++) {
            if (randProb < cumulDistr[m])
                break;
        }
        i = int(m / profLen);
        k = m % profLen;

        iPos = double(i) + randd() - 0.5;
        kPos = double(k) + randd() - 0.5;

        // Add particle if inside cutoff
        if ( double ((iPos - x0) * (iPos - x0) + (kPos - y0) * (kPos - y0)) < cutoff2 ) {
            dt[n] = dtMin + iPos * dtBin;
            dE[n] = dEMin + kPos * dEBin;
            n++;
        }
    }

}
