# cython: language_level=3

cimport numpy as cnp
from .statmorph cimport BaseInfo, CompareInfo

cdef CompareInfo calc_color_dispersion(BaseInfo base_info, cnp.ndarray[double, ndim=2] image_compare)
