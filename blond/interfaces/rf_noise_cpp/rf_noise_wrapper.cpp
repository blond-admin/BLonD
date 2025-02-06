#include <stdio.h>

#include <VariNoise.h>
#include <stdexcept>
#include <vector>
// hardcopy from https://gitlab.cern.ch/bbielaws/varinoisegen (30.10.2024)
// Commit SHA 9f32cdbffcdaed60e07d3ba3cdcc71b617dede82

#include "varigen.h"

extern "C"
{

    void rf_noise_wrapper(
        const double *fHigh, size_t fHighSize,
        const double *fLow, size_t fLowSize,
        const double *xs, size_t xsSize,
        const double *ys, size_t ysSize,
        double *result, size_t resultSize,
        const int nSource,
        const int nPntMin,
        const int rSeed,
        const double samplingRate,
        const double rms)
    {
        std::vector<double> xsVec(xs, xs + xsSize);
        std::vector<double> ysVec(ys, ys + ysSize);

        DiscreteShapeFunction shapeFunction(xsVec, ysVec);
        // fClockIn, rmsIn, GenericShapeFunc, nSourceMin,  nPntMin , iRandDoIn
        VariNoise vn(samplingRate, rms, &shapeFunction, nSource, nPntMin, rSeed);
        for (size_t i = 0; i < resultSize; i++)
        {
            vn.SetBandPosAndRelAmp(fLow[i], fHigh[i], 1.0);
            result[i] = vn.NextValue();
        }

        return;
    }
}