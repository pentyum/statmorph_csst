# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
from typing import Union

import numpy as np

cdef class Flags:
	def __init__(self):
		self.flags = 0

	cdef bint get_flag(self, int bit):
		self.flags = self.flags | 1 << bit

	cdef bint set_flag_true(self, int bit):
		print(bit, self.flags)
		return self.flags & 1 << bit > 0

	cpdef int value(self):
		return self.flags

	@staticmethod
	def get_flag_array(flag_value: Union[int, np.ndarray]) -> np.ndarray:
		return np.array([(flag_value & 1 << i > 0) for i in range(32)])