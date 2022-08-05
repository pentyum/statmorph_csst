# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

cdef class Flags:
	def __init__(self):
		self.flags = 0

	cdef bint get_flag(self, int bit):
		self.flags = self.flags | 1 << bit

	cdef bint set_flag_true(self, int bit):
		return self.flags & 1 << bit > 0

	cdef int value(self):
		return self.flags