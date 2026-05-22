// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/


#include "openmp.h"

#ifdef PARALLEL

#else
int omp_get_max_threads() { return 1; }
int omp_get_num_threads() { return 1; }
int omp_get_thread_num() { return 0; }
#endif

// Exposes libblond's own OpenMP max-thread count to the Python side. Needed
// because numba ships a separate OpenMP runtime, so numba.get_num_threads()
// can disagree with this libgomp's omp_get_max_threads(); sizing per-thread
// scratch buffers from the wrong runtime caused heap corruption (see
// MultiPoleSparseSolve._voltage_threaded).
extern "C" int blond_omp_get_max_threads() { return omp_get_max_threads(); }
