# cython: language_level=3

cimport numpy as cnp
from .statmorph cimport StampMorphology, G2Info

cnp.import_array()

cdef class G2Calculator:
	# gradients
	cdef double[:,:] gradient_asymmetric_x, gradient_asymmetric_y
	cdef double[:,:] gradient_x, gradient_y
	cdef double[:,:] image, modules, phases

	cdef double[:,:] phases_noise, modules_noise, modules_normalized

	cdef int contour_pixels_count, valid_pixels_count, assimetric_pixel_count
	cdef int height, width, center_x, center_y

	cdef double phase_tolerance, module_tolerance

	# debug method to add noise to phases
	cdef double add_phase_noise(self)


	# debug method to add noise to modules
	cdef double add_module_noise(self)

	# aux methods
	# add pi to phases, this is done to maintain phases 0-360 degrees
	cdef double add_pi(self)

	# modules normalization by max module
	cdef double normalize_modules(self)

	# find maximum gradient
	cdef double get_max_gradient(self)

	# function to find angle difference
	cdef double angle_difference(self,double a1,double a2)

	# function that converst 0 to nan in matrix
	cdef void convert_to_nan(self)

	# function that constructs asymmetric field
	cdef void get_asymmetryc_field(self)

	# function that return (if such) opposite pixel
	cdef (int,int) get_opposite_pixel(self, int pixel_x, int pixel_y)

	# getting list of pixels that have same distance from center
	cdef tuple get_pixels_same_distance_from_center(self, double[:,:] distance_from_center, int distance)

	# calculating confluence
	cdef double get_confluence(self)

	# entry point
	cdef tuple get_g2(self)

cdef int _get_contour_count(cnp.ndarray[double,ndim=2] image)
cdef G2Info get_G2(StampMorphology base_info, (double, double) _asymmetry_center)