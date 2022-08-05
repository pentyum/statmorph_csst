# cython: language_level=3

from .statmorph cimport BaseInfo, GiniM20Info

cdef GiniM20Info calc_g_m20(BaseInfo base_info, (double, double) asymmetry_center)