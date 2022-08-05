# cython: language_level=3

cdef class Flags:
	cdef int flags

	cdef bint get_flag(self, int bit)

	cdef bint set_flag_true(self, int bit)

	cdef int value(self)