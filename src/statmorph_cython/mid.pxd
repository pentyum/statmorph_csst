# cython: language_level=3

from .statmorph cimport StampMorphology, MIDInfo

cdef MIDInfo calc_mid(StampMorphology base_info)