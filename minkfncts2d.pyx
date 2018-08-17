from cython.parallel import prange
from libc.math cimport sqrt
import numpy as np
cimport numpy as np
cimport cython

# Implementation based on http://iopscience.iop.org/article/10.1088/1742-5468/2008/12/P12015
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def MF2D(np.ndarray[np.double_t, ndim=2] image, double threshold):
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef int y
    cdef int x
    
    cdef double[:,:] c_image = image
    
    cdef double f = 0.0
    cdef double u = 0.0
    cdef double chi = 0.0
    
    cdef double p00
    cdef double p10
    cdef double p01
    cdef double p11
    
    cdef int pattern
    
    cdef double a1
    cdef double a2
    cdef double a3
    cdef double a4

    for y in prange(height-1, nogil=True):
        p10 = c_image[y, 0]
        p11 = c_image[y+1, 0]
        for x in range(width-1):
            pattern = 0
            
            p00 = p10
            p01 = p11
            p10 = c_image[y, x+1]
            p11 = c_image[y+1, x+1]
            
            if p00 > threshold:
                pattern = pattern | <int>1
            if p10 > threshold:
                pattern = pattern | <int>2
            if p11 > threshold:
                pattern = pattern | <int>4
            if p01 > threshold:
                pattern = pattern | <int>8
                
            # a1 = (p00 - threshold) / (p00 - p10)
            # a2 = (p10 - threshold) / (p10 - p11)
            # a3 = (p01 - threshold) / (p01 - p11)
            # a4 = (p00 - threshold) / (p00 - p01)
            
            if pattern == 0:
                pass
            elif pattern == 1:
                a1 = (p00 - threshold) / (p00 - p10)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 0.5 * a1 * a4
                u += sqrt(a1*a1 + a4*a4)
                chi += 0.25
            elif pattern == 2:
                a1 = (p00 - threshold) / (p00 - p10)
                a2 = (p10 - threshold) / (p10 - p11)
                f += 0.5 * (1.0-a1)*a2
                u += sqrt((1.0-a1)*(1.0-a1) + a2*a2)
                chi += 0.25
            elif pattern == 3:
                a2 = (p10 - threshold) / (p10 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += a2 + 0.5*(a4-a2)
                u += sqrt(1.0 + (a4-a2)*(a4-a2))
            elif pattern == 4:
                a2 = (p10 - threshold) / (p10 - p11)
                a3 = (p01 - threshold) / (p01 - p11)
                f += 0.5 * (1.0-a2)*(1.0-a3)
                u += sqrt((1.0-a2)*(1.0-a2) + (1.0-a3)*(1.0-a3))
                chi += 0.25
            elif pattern == 5:
                a1 = (p00 - threshold) / (p00 - p10)
                a2 = (p10 - threshold) / (p10 - p11)
                a3 = (p01 - threshold) / (p01 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 1.0 - 0.5*(1.0-a1)*a2 - 0.5*a3*(1.0-a4)
                u += sqrt((1.0-a1)*(1.0-a1) + a2*a2) + sqrt(a3*a3 + (1.0-a4)*(1.0-a4))
                chi += 0.5
            elif pattern == 6:
                a1 = (p00 - threshold) / (p00 - p10)
                a3 = (p01 - threshold) / (p01 - p11)
                f += (1.0-a3) + 0.5*(a3-a1)
                u += sqrt(1.0 + (a3-a1)*(a3-a1))
            elif pattern == 7:
                a3 = (p01 - threshold) / (p01 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 1.0 - 0.5*a3*(1.0-a4)
                u += sqrt(a3*a3 + (1.0-a4)*(1.0-a4))
                chi += -0.25
            elif pattern == 8:
                a3 = (p01 - threshold) / (p01 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 0.5*a3*(1.0-a4)
                u += sqrt(a3*a3 + (1.0-a4)*(1.0-a4))
                chi += 0.25
            elif pattern == 9:
                a1 = (p00 - threshold) / (p00 - p10)
                a3 = (p01 - threshold) / (p01 - p11)
                f += a1 + 0.5*(a3-a1)
                u += sqrt(1.0 + (a3-a1)*(a3-a1))
            elif pattern == 10:
                a1 = (p00 - threshold) / (p00 - p10)
                a2 = (p10 - threshold) / (p10 - p11)
                a3 = (p01 - threshold) / (p01 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 1.0 - 0.5*a1*a4 + 0.5*(1.0-a2)*(1.0-a3)
                u += sqrt(a1*a1 + a4*a4) + sqrt((1.0-a2)*(1.0-a2) + (1.0-a3)*(1.0-a3))
                chi += 0.5
            elif pattern == 11:
                a2 = (p10 - threshold) / (p10 - p11)
                a3 = (p01 - threshold) / (p01 - p11)
                f += 1.0 - 0.5*(1.0-a2)*(1.0-a3)
                u += sqrt((1.0-a2)*(1.0-a2) + (1.0-a3)*(1.0-a3))
                chi += -0.25
            elif pattern == 12:
                a2 = (p10 - threshold) / (p10 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += (1.0-a2) + 0.5*(a2-a4)
                u += sqrt(1.0 + (a2-a4)*(a2-a4))
            elif pattern == 13:
                a1 = (p00 - threshold) / (p00 - p10)
                a2 = (p10 - threshold) / (p10 - p11)
                f += 1.0 - 0.5*(1.0-a1)*a2
                u += sqrt((1.0-a1)*(1.0-a1) + a2*a2)
                chi += -0.25
            elif pattern == 14:
                a1 = (p00 - threshold) / (p00 - p10)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 1.0 - 0.5*a1*a4
                u += sqrt(a1*a1 + a4*a4)
                chi += -0.25
            elif pattern == 15:
                f += 1.0
    return (f, u, chi)
