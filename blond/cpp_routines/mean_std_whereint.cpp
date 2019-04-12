#include <cmath>

#ifdef PARALLEL
#include <omp.h>
#endif

extern "C" double mean(const double * __restrict__ data, const int n)
{
   double m = 0;
   #pragma omp parallel for reduction(+:m)
   for (int i = 0; i < n; ++i) {
      m += data[i];
   }
   return m / n;
}

extern "C" double standard_deviation(const double * __restrict__ data, const int n,
                         const double mean)
{
   double sum_deviation = 0.0;
   #pragma omp parallel for reduction(+:sum_deviation)
   for (int i = 0; i < n; ++i)
      sum_deviation += (data[i] - mean) * (data[i] - mean);
   return sqrt(sum_deviation / n);
}


extern "C" int where(const double *__restrict__ dt, const int n_macroparticles,
                                    const double constant1, const double constant2)
{
   int s = 0;
   #pragma omp parallel for reduction(+:s)
   for (int i = 0; i < n_macroparticles; i++) {
      s += (dt[i] < constant2 && dt[i] > constant1) ? 1 : 0;
   }
   return s;
}



