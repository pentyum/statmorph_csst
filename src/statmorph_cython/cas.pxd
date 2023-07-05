# cython: language_level=3
from .statmorph cimport StampMorphology, CASInfo
cimport numpy as cnp

cdef cnp.ndarray simplified_rot180(cnp.ndarray image, (double, double) center)
cdef CASInfo calc_cas(StampMorphology base_info, (double,double) set_asym_center)