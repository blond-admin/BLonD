import cython
import numpy
cimport numpy
from libc.math cimport sin

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef kick(int n_rf, double[::1] beam_dE, \
         double[::1] beam_theta, \
         double[:] voltage, \
         double[:] harmonic,\
         double[:] phi_offset,\
         ): 
    
    cdef double[:] cvoltage = numpy.ascontiguousarray(voltage,dtype=numpy.float64)
    cdef double[:] charmonic = numpy.ascontiguousarray(harmonic,dtype=numpy.float64)
    cdef double[:] cphi_offset = numpy.ascontiguousarray(phi_offset,dtype=numpy.float64)
    cdef long size = beam_dE.shape[0]
    cdef int j, i
    
    for j in xrange(n_rf):
        for i in xrange(size):
                beam_dE[i] = beam_dE[i] + cvoltage[j] * sin(charmonic[j] * beam_theta[i] + cphi_offset[j])
    
