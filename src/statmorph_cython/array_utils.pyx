# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

cimport numpy as cnp

cnp.import_array()

cdef cnp.ndarray[double,ndim=2] create_22_mat(double a00, double a01, double a10, double a11):
	cdef double[:,:] new_array = cnp.PyArray_SimpleNew(2,[2,2],cnp.NPY_DOUBLE)
	new_array[0,0] = a00
	new_array[0,1] = a01
	new_array[1,0] = a10
	new_array[1,1] = a11
	return new_array.base

cdef double sum_1d_d(cnp.ndarray[double,ndim=1] array):
	return cnp.PyArray_Sum(array,0,cnp.NPY_DOUBLE,None)

cdef double sum_2d_d(cnp.ndarray[double,ndim=2] array):
	cdef cnp.ndarray sum_once = cnp.PyArray_Sum(array,0,cnp.NPY_DOUBLE,None)
	return cnp.PyArray_Sum(sum_once,0,cnp.NPY_DOUBLE,None)

cdef int sum_1d_i(cnp.ndarray[int,ndim=1] array):
	return cnp.PyArray_Sum(array,0,cnp.NPY_INT,None)

cdef cnp.ndarray[double,ndim=1] cumsum_1d_d(cnp.ndarray[double,ndim=1] array):
	return cnp.PyArray_CumSum(array,0,cnp.NPY_DOUBLE,None)