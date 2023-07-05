# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

######################
# Calculate multiplicity #
######################

"""
In the following part, we calculate the multiply statistic,
since the origin defination of this para is related to redshift z,
Here we choose the MID segmap as the

https://ui.adsabs.harvard.edu/abs/2007ApJ...656....1L
"""
import warnings

import numpy as np

cimport numpy as cnp
from libc.math cimport log10
import time

cnp.import_array()

cdef double multiply_calculate_help(cnp.ndarray[double,ndim=2] image, int timeout=5):
	cdef long start_time = time.time()
	cdef double cost_time
	cdef int i, j
	cdef cnp.ndarray multiply, distances

	cdef int x_size = image.shape[1]
	cdef int y_size = image.shape[0]
	cdef cnp.ndarray[cnp.npy_int64,ndim=3] indices = np.indices((y_size, x_size))
	multiply_list = []
	for i in range(y_size):
		for j in range(x_size):
			distances = np.sqrt((i - indices[0]) ** 2 + (j - indices[1]) ** 2)
			multiply = image[i, j] * image / distances
			#multiply = cnp.PyArray_Where(np.isfinite(multiply), multiply, 0)
			#multiply_list[i, j] = np.sum(multiply)
			multiply_list.append(multiply)
		cost_time = time.time() - start_time
		if cost_time > timeout:
			warnings.warn("[multiply] timeout (%d)" % timeout)
			return -99
	multiply_list = np.array(multiply_list)
	multiply_list = cnp.PyArray_Where(np.isfinite(multiply_list), multiply_list, 0)

	return np.sum(multiply_list)

cdef cnp.ndarray[double,ndim=2] image_reshape(cnp.ndarray[double,ndim=2] image):
	cdef int i
	cdef int x_size = image.shape[1]
	cdef int y_size = image.shape[0]
	cdef int x_center = int(x_size / 2)
	cdef int y_center =  int(y_size / 2)

	cdef double[:] image_sort = np.sort(cnp.PyArray_Ravel(image,cnp.NPY_CORDER))[::-1]
	cdef cnp.ndarray[cnp.npy_int64,ndim=3] indices = np.indices((y_size, x_size))
	cdef cnp.ndarray[double,ndim=2] distances = np.sqrt((y_center - indices[0]) ** 2 + (x_center - indices[1]) ** 2)
	cdef long[:] index = cnp.PyArray_ArgSort(cnp.PyArray_Ravel(distances,cnp.NPY_CORDER),-1,cnp.NPY_QUICKSORT)
	cdef cnp.ndarray[double,ndim=1] new_image = cnp.PyArray_ZEROS(1, [x_size * y_size], cnp.NPY_DOUBLE, 0)
	for i in range(x_size * y_size):
		new_image[index[i]] = image_sort[i]
	cdef cnp.ndarray[double,ndim=2] new_image_2d = new_image.reshape((y_size, x_size))
	return new_image_2d

cdef double multiplicity(cnp.ndarray[double,ndim=2] _cutout_mid):
	cdef cnp.ndarray[double,ndim=2] image = _cutout_mid
	cdef int x_size = image.shape[1]
	cdef int y_size = image.shape[0]
	cdef double temp_origin, temp_reshape, multiply_para

	if x_size > 300 or y_size > 300:
		return -99
	else:
		temp_origin = multiply_calculate_help(image)
		temp_reshape = multiply_calculate_help(image_reshape(image))
		multiply_para = 100 * log10(temp_reshape / temp_origin)
		return multiply_para