# cython: language_level=3

cimport numpy as cnp

cdef double multiplicity(cnp.ndarray[double,ndim=2] _cutout_mid)