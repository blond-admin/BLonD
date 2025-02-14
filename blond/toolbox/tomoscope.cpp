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
#include "../cpp_routines/blond_common.h"

using uint = unsigned int;

real_t randd() {
    return (real_t)rand() / (RAND_MAX + 1.0);
}


extern "C" void generate_distribution(real_t * __restrict__ dt,
                                      real_t * __restrict__ dE, const real_t * probDistr, const uint seed,
                                      const uint profLen, const real_t cutoff, const real_t x0, const real_t y0,
                                      const real_t dtBin, const real_t dEBin, const uint nPart) {


// Initialise random seed
    srand(seed);

// Initialise some variables
    uint i, k, m;
    real_t iPos, kPos;
    real_t cutoff2 = cutoff * cutoff;
    real_t dtMin = -1.*x0 * dtBin;
    real_t dEMin = -1.*y0 * dEBin;


// Calculate cumulative probability
    real_t totProb = 0.;
    uint profLen2 = uint(profLen * profLen);
    real_t cumulDistr [profLen2];

    for (m = 0; m < profLen2; m++) {
        cumulDistr[m] = probDistr[m] + totProb;
        totProb += probDistr[m];
    }


// Normalise probability distribution
    real_t invTotProb = 1. / totProb;

    for (m = 0; m < profLen2; m++) {
        cumulDistr[m] *= invTotProb;
    }


// Generate particle coordinates
    uint n = 0;
    real_t randProb;

    while ( n < nPart ) {
        randProb = randd();
        for (m = 0; m < profLen2; m++) {
            if (randProb < cumulDistr[m])
                break;
        }
        i = int(m / profLen);
        k = m % profLen;

        iPos = real_t(i) + randd() - 0.5;
        kPos = real_t(k) + randd() - 0.5;

        // Add particle if inside cutoff
        if ( real_t ((iPos - x0) * (iPos - x0) + (kPos - y0) * (kPos - y0)) < cutoff2 ) {
            dt[n] = dtMin + iPos * dtBin;
            dE[n] = dEMin + kPos * dEBin;
            n++;
        }
    }

}
