# cython: language_level=3

cimport numpy as cnp

cdef cnp.ndarray[cnp.npy_bool,ndim=2] segmap_color_dispersion(double rpetro, (double,double) center, int nx, int ny)

cdef (double,double) find_alpha_beta(cnp.ndarray _cutout_stamp_maskzeroed, cnp.ndarray _cutout_stamp_maskzeroed_compare)

cdef double _sky_dispersion(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed_compare,
							cnp.ndarray[double,ndim=2] _bkg, cnp.ndarray[double,ndim=2] _bkg_compare)