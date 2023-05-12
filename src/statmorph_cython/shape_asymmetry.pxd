# cython: language_level=3
from .statmorph cimport BaseInfo, CASInfo, GiniM20Info, ShapeAsymmetryInfo

cdef ShapeAsymmetryInfo calc_shape_asymmetry(BaseInfo base_info, CASInfo cas, GiniM20Info g_m20)