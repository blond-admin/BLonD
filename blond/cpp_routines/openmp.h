
#ifndef OPENMP_H
#define OPENMP_H

#ifdef PARALLEL
#include <omp.h>        // omp_get_thread_num(), omp_get_num_threads()
#else
int omp_get_max_threads() {return 1;}
int omp_get_num_threads() {return 1;}
int omp_get_thread_num() {return 0;}
#endif

#endif // OPENMP_H