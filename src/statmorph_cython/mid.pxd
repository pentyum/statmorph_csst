# cython: language_level=3

from .statmorph cimport BaseInfo, MIDInfo

cdef MIDInfo calc_mid(BaseInfo base_info)