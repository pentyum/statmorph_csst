# cython: language_level=3

cdef class Flags:
	cdef int flags

	cdef bint get_flag(self, int bit)

	cdef void set_flag_true(self, int bit)

	cpdef int value(self)