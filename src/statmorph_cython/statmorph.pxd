# cython: language_level=3

cimport numpy as cnp
from .flags cimport Flags
from .constants_setting cimport ConstantsSetting

cdef class MorphInfo:
	cdef readonly double runtime
	cdef readonly Flags flags

	@staticmethod
	cdef double get_duration_sec(double end, double start)

	cdef void calc_runtime(self, double start)

cdef class BaseInfo(MorphInfo):
	cdef cnp.ndarray _image
	cdef cnp.ndarray _segmap
	cdef tuple _segmap_slice
	cdef readonly int label
	cdef cnp.ndarray _mask
	cdef cnp.ndarray _weightmap
	cdef double _gain

	cdef readonly bint flag_catastrophic
	cdef bint _use_centroid

	cdef ConstantsSetting constants

	cdef tuple _slice_stamp
	cdef cnp.ndarray _cutout_stamp
	cdef cnp.ndarray _segmap_stamp
	cdef cnp.ndarray _weightmap_stamp_old
	cdef cnp.ndarray _weightmap_stamp
	cdef cnp.ndarray _mask_stamp_old
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
	cdef readonly (double, double) _centroid
	cdef double xc_centroid
	cdef double yc_centroid
	cdef double _xc_stamp
	cdef double _yc_stamp
	cdef double _diagonal_distance
	cdef readonly double _rpetro_circ_centroid
	cdef readonly double sn_per_pixel

	cdef str output_image_dir
	cdef str save_stamp_dir
	cdef readonly bint calc_cas
	cdef readonly bint calc_g_m20
	cdef readonly bint calc_mid
	cdef readonly bint calc_multiplicity
	cdef readonly bint calc_color_dispersion
	cdef readonly bint calc_g2

	cdef readonly CASInfo cas
	cdef readonly GiniM20Info g_m20
	cdef readonly ShapeAsymmetryInfo shape_asymmetry
	cdef readonly MIDInfo mid
	cdef readonly double multiplicity
	cdef cnp.ndarray image_compare
	cdef cnp.ndarray image_compare_stamp
	cdef readonly CompareInfo compare_info
	cdef readonly G2Info g2
	cdef readonly long global_start
	cdef readonly logger

	cdef bint check_total_flux_nonpositive(self)

	cpdef void calculate_morphology(self, bint calc_cas, bint calc_g_m20, bint calc_shape_asymmetry, bint calc_mid, bint calc_multiply,
				 bint calc_color_dispersion, bint calc_g2, (double,double) set_asym_center)

	cdef void _check_segmaps(self)

	cdef void _check_stamp_size(self)

	cdef tuple get_slice_stamp(self)

	cdef int get_xmin_stamp(self)

	cdef int get_ymin_stamp(self)

	cdef int get_xmax_stamp(self)

	cdef int get_ymax_stamp(self)

	cdef int get_nx_stamp(self)

	cdef int get_ny_stamp(self)

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_nan(self)

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] _get_badpixels(self, cnp.ndarray[double, ndim=2] image)

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_badpixels(self)

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp(self)

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_no_bg(self)

	cdef cnp.ndarray[double, ndim=2] get_cutout_stamp_maskzeroed(self)

	cdef cnp.ndarray[double, ndim=2] get_cutout_stamp_maskzeroed_no_bg(self)

	cdef (double,double) get_centroid(self)

	cdef double get_xc_centroid(self)

	cdef double get_yc_centroid(self)

	cdef double get_diagonal_distance(self)

	cdef double get_rpetro_circ_centroid(self)

	cdef cnp.ndarray get_weightmap_stamp(self)

	cdef double get_sn_per_pixel(self)

	cdef void _abort_calculations(self)

	cdef void save_image(self)

	cdef void save_stamp(self)

	cdef void dump_stamps(self)

cdef class IndividualBaseInfo(BaseInfo):
	cdef _image_fits
	cdef _mask_fits
	cdef _weightmap_fits
	cdef _image_compare_fits

	cpdef void close_all(self)

cdef class CASInfo(MorphInfo):
	cdef tuple _slice_skybox
	cdef cnp.ndarray _bkg
	cdef readonly double _sky_asymmetry
	cdef double xc_asymmetry, yc_asymmetry
	cdef readonly double concentration, asymmetry, smoothness, rpetro_circ, r20, r80
	cdef readonly (double,double) _asymmetry_center
	cdef readonly double sky_mean, sky_sigma, _sky_smoothness

cdef class GiniM20Info(MorphInfo):
	cdef double[:] eigvals_asymmetry
	cdef readonly double gini, m20, rpetro_ellip, f, s, elongation_asymmetry, orientation_asymmetry
	cdef cnp.ndarray _segmap_gini

cdef class MIDInfo(MorphInfo):
	cdef cnp.ndarray _cutout_mid, _segmap_mid
	cdef readonly double multimode, intensity, deviation

cdef class G2Info(MorphInfo):
	cdef cnp.ndarray segmented_image
	cdef double g2_modular_tolerance
	cdef double g2_phase_tolerance

	cdef public double result_g2

	cdef double[:,:] gradient_x
	cdef double[:,:] gradient_y
	cdef double[:,:] gradient_asymmetric_x
	cdef double[:,:] gradient_asymmetric_y
	cdef double[:,:] modules_normalized
	cdef double[:,:] phases

	cdef double calc_g2(self)
	cdef get_gradient_plot(self)
	cdef get_asymmetry_gradient_plot(self)

cdef class CompareInfo(MorphInfo):
	cdef cnp.ndarray _image_compare_stamp
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

cdef class ShapeAsymmetryInfo(MorphInfo):
	cdef double rhalf_circ
	cdef double rhalf_ellip
	cdef double shape_asymmetry

cdef class SersicInfo(MorphInfo):
	cdef double sersic_amplitude
	cdef double sersic_rhalf
	cdef double sersic_n
	cdef double sersic_xc
	cdef double sersic_yc
	cdef double sersic_ellip
	cdef double sersic_theta