f2py -c -m beam_phase_module_64 beam_phase_64.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp
f2py -c -m beam_phase_module_32 beam_phase_32.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp

f2py -c -m histogram_module_64 histogram_64.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp
f2py -c -m histogram_module_32 histogram_32.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp


f2py -c -m kick_induced_module_64 kick_induced_64.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp
f2py -c -m kick_induced_module_32 kick_induced_32.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp

f2py -c -m drift_module_64 drift_64.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp
f2py -c -m drift_module_32 drift_32.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp

f2py -c -m kick_module_64 kick_64.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp
f2py -c -m kick_module_32 kick_32.f90 \
     --f90flags='-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp' \
          -lgomp
