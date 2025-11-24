// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/


#ifndef _OPENMP_H_
#define _OPENMP_H_

#ifdef PARALLEL
#include <omp.h> // omp_get_thread_num(), omp_get_num_threads()
#else
int omp_get_max_threads();
int omp_get_num_threads();
int omp_get_thread_num();
#endif

#endif // _OPENMP_H_
