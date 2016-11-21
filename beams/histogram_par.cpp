/*
 Copyright 2016 CERN. This software is distributed under the
 terms of the GNU General Public Licence version 3 (GPL Version 3), 
 copied verbatim in the file LICENCE.md.
 In applying this licence, CERN does not waive the privileges and immunities 
 granted to it by virtue of its status as an Intergovernmental Organization or 
 submit itself to any jurisdiction.
 Project website: http://blond.web.cern.ch/
 */

// Optimised C++ routine that calculates the histogram
// Author: Danilo Quartullo, Alexandre Lasheen

#include <omp.h>
#include <cmath>
#include <algorithm>

extern "C" void histogram(const double * __restrict__ input,
		double * __restrict__ output, const double cut_left,
		const double cut_right, const int n_slices,
		const int n_macroparticles) {

	const double inv_bin_width = n_slices / (cut_right - cut_left);

	for (int i = 0; i < n_slices; i++) {
		output[i] = 0.0;
	}

	double h[omp_get_max_threads()][n_slices];
#pragma omp parallel
	{
		double a;
		double fbin;
		int ffbin;

		int id = omp_get_thread_num();
		int threads = omp_get_num_threads();
		int tile = std::ceil(1.0 * n_macroparticles / threads);
		int start = id * tile;
		int end = std::min(start + tile, n_macroparticles);

		for (int i = 0; i < n_slices; ++i) {
			h[id][i] = 0;
		}

		for (int i = start; i < end; i++) {
			a = input[i];

			if ((a < cut_left) || (a > cut_right))
				continue;
			fbin = (a - cut_left) * inv_bin_width;
			ffbin = (int) fbin;
			h[id][ffbin] = h[id][ffbin] + 1.0;
		}
#pragma omp barrier

#pragma omp single
		for (int i = 0; i < threads; ++i) {
			for (int j = 0; j < n_slices; ++j) {
				output[j] += h[i][j];
			}
		}
	}
}

extern "C" void smooth_histogram(const double * __restrict__ input,
		double * __restrict__ output, const double cut_left,
		const double cut_right, const int n_slices,
		const int n_macroparticles) {

	int i;
	double a;
	double fbin;
	double ratioffbin;
	double ratiofffbin;
	double distToCenter;
	int ffbin;
	int fffbin;
	const double inv_bin_width = n_slices / (cut_right - cut_left);
	const double bin_width = (cut_right - cut_left) / n_slices;

	for (i = 0; i < n_slices; i++) {
		output[i] = 0.0;
	}

	for (i = 0; i < n_macroparticles; i++) {
		a = input[i];
		if ((a < (cut_left + bin_width * 0.5))
				|| (a > (cut_right - bin_width * 0.5)))
			continue;
		fbin = (a - cut_left) * inv_bin_width;
		ffbin = (int) fbin;
		distToCenter = fbin - (double) (ffbin);
		if (distToCenter > 0.5)
			fffbin = (int) (fbin + 1.0);
		ratioffbin = 1.5 - distToCenter;
		ratiofffbin = 1 - ratioffbin;
		if (distToCenter < 0.5)
			fffbin = (int) (fbin - 1.0);
		ratioffbin = 0.5 - distToCenter;
		ratiofffbin = 1 - ratioffbin;
		output[ffbin] = output[ffbin] + ratioffbin;
		output[fffbin] = output[fffbin] + ratiofffbin;
	}
}

