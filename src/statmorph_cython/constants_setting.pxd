# cython: language_level=3

cdef class ConstantsSetting:
	cdef double cutout_extent
	cdef int min_cutout_size
	cdef int n_sigma_outlier
	cdef double annulus_width
	cdef double eta
	cdef double petro_fraction_gini
	cdef int skybox_size
	cdef double petro_extent_cas
	cdef double petro_fraction_cas
	cdef double petro_extent_flux
	cdef int simplified_rot_threshold
	cdef int fmin_maxiter
	cdef double boxcar_size_mid
	cdef double sigma_mid
	cdef int niter_bh_mid

	cdef double g2_modular_tolerance
	cdef double g2_phase_tolerance

	cdef bint verbose
