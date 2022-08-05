# cython: language_level=3

cimport numpy as cnp

cdef cnp.ndarray[double,ndim=2] create_22_mat(double a00, double a01, double a10, double a11)

cdef double sum_1d_d(cnp.ndarray[double,ndim=1] array)

cdef double sum_2d_d(cnp.ndarray[double,ndim=2] array)

cdef int sum_1d_i(cnp.ndarray[int,ndim=1] array)

cdef cnp.ndarray[double,ndim=1] cumsum_1d_d(cnp.ndarray[double,ndim=1] array)