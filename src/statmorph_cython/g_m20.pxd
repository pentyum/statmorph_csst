# cython: language_level=3

from .statmorph cimport StampMorphology, GiniM20Info

cdef GiniM20Info calc_g_m20(StampMorphology base_info, (double, double) asymmetry_center)