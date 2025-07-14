f2py -c -m histogram_module histogram.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp

f2py -c -m kick_induced_module kick_induced.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp
f2py -c -m drift_module drift.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp
f2py -c -m kick_module kick.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp
