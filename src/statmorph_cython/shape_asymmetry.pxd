# cython: language_level=3
from .statmorph cimport StampMorphology, CASInfo, GiniM20Info, ShapeAsymmetryInfo

cdef ShapeAsymmetryInfo calc_shape_asymmetry(StampMorphology base_info, CASInfo cas, GiniM20Info g_m20)