# cython: language_level=3

cimport numpy as cnp
from .flags cimport Flags
from .statmorph cimport ConstantsSetting

cdef double _rpetro_circ_generic(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, (double, double) center,
								 double _diagonal_distance, Flags flags, ConstantsSetting constants)

cdef double _rpetro_ellip_generic(cnp.ndarray[double,ndim=2] cutout_stamp_maskzeroed, (double,double) center, double elongation, double theta,
								  double _diagonal_distance, Flags flags, ConstantsSetting constants)