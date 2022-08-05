# cython: language_level=3

cimport numpy as cnp
from .flags cimport Flags

cdef class ConstantsSetting:
	cdef double cutout_extent
	cdef int min_cutout_size
	cdef int n_sigma_outlier
	cdef double annulus_width
	cdef double eta
	cdef double petro_fraction_gini
	cdef int skybox_size
	cdef double petro_extent_cas
	cdef double petro_fraction_cas
	cdef double petro_extent_flux
	cdef int simplified_rot_threshold
	cdef int fmin_maxiter
	cdef double boxcar_size_mid
	cdef double sigma_mid
	cdef int niter_bh_mid
	cdef bint verbose

cdef class MorphInfo:
	cdef readonly double runtime
	cdef readonly Flags flags

	@staticmethod
	cdef double get_duration_sec(long end, long start)

	cdef void calc_runtime(self, long start)

cdef class BaseInfo(MorphInfo):
	cdef cnp.ndarray _image
	cdef cnp.ndarray _segmap
	cdef tuple _segmap_slice
	cdef int label
	cdef cnp.ndarray _mask
	cdef cnp.ndarray _weightmap
	cdef double _gain

	cdef bint flag_catastrophic
	cdef bint _use_centroid

	cdef ConstantsSetting constants

	cdef tuple _slice_stamp
	cdef cnp.ndarray _cutout_stamp
	cdef cnp.ndarray _segmap_stamp
	cdef cnp.ndarray _mask_stamp_nan
	cdef int xmin_stamp
	cdef int ymin_stamp
	cdef int xmax_stamp
	cdef int ymax_stamp
	cdef int nx_stamp
	cdef int ny_stamp
	cdef int num_badpixels
	cdef cnp.ndarray _mask_stamp_badpixels
	cdef cnp.ndarray _mask_stamp
	cdef cnp.ndarray _mask_stamp_no_bg
	cdef cnp.ndarray _cutout_stamp_maskzeroed
	cdef cnp.ndarray _cutout_stamp_maskzeroed_no_bg
	cdef readonly int size
	cdef readonly double surface_brightness
	cdef (double, double) _centroid
	cdef double xc_centroid
	cdef double yc_centroid
	cdef double _xc_stamp
	cdef double _yc_stamp
	cdef double _diagonal_distance
	cdef readonly double _rpetro_circ_centroid

	cdef str output_image_dir
	cdef readonly bint calc_cas
	cdef readonly bint calc_g_m20
	cdef readonly bint calc_mid
	cdef readonly bint calc_multiply
	cdef readonly bint calc_color_dispersion

	cdef readonly CASInfo cas
	cdef readonly GiniM20Info g_m20
	cdef readonly MIDInfo mid
	cdef readonly double multiply
	cdef cnp.ndarray image_compare
	cdef readonly CompareInfo compare_info

	cdef tuple get_slice_stamp(self)

	cdef int get_xmin_stamp(self)

	cdef int get_ymin_stamp(self)

	cdef int get_xmax_stamp(self)

	cdef int get_ymax_stamp(self)

	cdef int get_nx_stamp(self)

	cdef int get_ny_stamp(self)

	cdef cnp.ndarray get_mask_stamp_nan(self)

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] _get_badpixels(self, cnp.ndarray[double, ndim=2] image)

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_badpixels(self)

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp(self)

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_no_bg(self)

	cdef cnp.ndarray get_cutout_stamp_maskzeroed(self)

	cdef cnp.ndarray get_cutout_stamp_maskzeroed_no_bg(self)

	cdef (double,double) get_centroid(self)

	cdef double get_xc_centroid(self)

	cdef double get_yc_centroid(self)

	cdef double get_diagonal_distance(self)

	cdef double get_rpetro_circ_centroid(self)

	cdef void _abort_calculations(self)

	cdef void save_image(self)

cdef class CASInfo(MorphInfo):
	cdef tuple _slice_skybox
	cdef cnp.ndarray _bkg
	cdef double _sky_asymmetry
	cdef double xc_asymmetry, yc_asymmetry
	cdef readonly double concentration, asymmetry, smoothness, rpetro_circ, r20, r80
	cdef (double,double) _asymmetry_center
	cdef double sky_mean, sky_sigma, _sky_smoothness

cdef class GiniM20Info(MorphInfo):
	cdef readonly double gini, m20, rpetro_ellip
	cdef cnp.ndarray _segmap_gini

cdef class MIDInfo(MorphInfo):
	cdef cnp.ndarray _cutout_mid
	cdef readonly double multimode, intensity, deviation

cdef class CompareInfo(MorphInfo):
	cdef cnp.ndarray _image_compare
	cdef BaseInfo base_info
	cdef int num_badpixels

	cdef cnp.ndarray _mask_stamp_nan_compare
	cdef cnp.ndarray _mask_stamp_badpixels_compare
	cdef cnp.ndarray _mask_stamp_compare
	cdef cnp.ndarray _mask_stamp_no_bg_compare

	cdef cnp.ndarray _cutout_stamp_maskzeroed_compare
	cdef cnp.ndarray _cutout_stamp_maskzeroed_no_bg_compare
	cdef cnp.ndarray _bkg_compare
	cdef readonly double color_dispersion

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_nan_compare(self)
	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_badpixels_compare(self)
	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_compare(self)
	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_no_bg_compare(self)
	cdef cnp.ndarray[double, ndim=2] get_cutout_stamp_maskzeroed_compare(self)
	cdef cnp.ndarray[double, ndim=2] get_cutout_stamp_maskzeroed_no_bg_compare(self)
	cdef double get_color_dispersion(self)