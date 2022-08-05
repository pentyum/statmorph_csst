# cython: language_level=3
from .statmorph cimport BaseInfo, CASInfo

cdef CASInfo calc_cas(BaseInfo base_info)