
#ifndef _OPENMP_H_
#define _OPENMP_H_

#ifdef PARALLEL
	#include <omp.h>        // omp_get_thread_num(), omp_get_num_threads()
#else
	int omp_get_max_threads();
	int omp_get_num_threads();
	int omp_get_thread_num();
#endif

#endif // _OPENMP_H_