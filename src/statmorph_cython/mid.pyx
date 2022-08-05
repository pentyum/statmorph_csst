# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import warnings
import numpy as np
cimport numpy as cnp
import scipy.ndimage as ndi
import skimage.feature
import skimage.segmentation
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import _ni_label
import scipy.optimize as opt
from numpy.math cimport INFINITY, isfinite
from libc.math cimport sqrt, pi

from .statmorph cimport ConstantsSetting, BaseInfo, MIDInfo
from .flags cimport Flags

cnp.import_array()

cdef double _quantile(cnp.ndarray[double,ndim=1] sorted_values, double q):
	"""
	For a sorted (in increasing order) 1-d array, return the value
	corresponding to the quantile ``q``.

	Notes
	-----
	The result is identical to np.percentile(..., interpolation='lower'),
	but the currently defined function is infinitely faster for sorted arrays.
	"""
	if q < 0 or q > 1:
		raise ValueError('Quantiles must be in the range [0, 1].')
	return sorted_values[int(q*(len(sorted_values)-1))]

##################
# MID statistics #
##################

cdef cnp.ndarray[double,ndim=2] get_cutout_stamp_maskzeroed_no_bg_nonnegative(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed_no_bg):
	"""
	Same as ``_cutout_stamp_maskzeroed_no_bg``, but masking
	negative pixels.
	"""
	cdef cnp.ndarray image = _cutout_stamp_maskzeroed_no_bg
	return cnp.PyArray_Where(image > 0, image, 0.0)

cdef cnp.ndarray[double,ndim=1] get_sorted_pixelvals_stamp_no_bg_nonnegative(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed_no_bg_nonnegative, cnp.ndarray _mask_stamp_no_bg):
	"""
	Same as ``_sorted_pixelvals_stamp_no_bg``, but masking
	negative pixels.
	"""
	cdef cnp.ndarray image = _cutout_stamp_maskzeroed_no_bg_nonnegative
	cdef cnp.ndarray image_no_bg_list = image[~_mask_stamp_no_bg]
	cnp.PyArray_Sort(image_no_bg_list,-1,cnp.NPY_QUICKSORT)
	return image_no_bg_list

cdef cnp.ndarray _segmap_mid_main_clump(double q, cnp.ndarray[double,ndim=1] _sorted_pixelvals_stamp_no_bg_nonnegative, cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed_no_bg_nonnegative,
										int _x_maxval_stamp, int _y_maxval_stamp):
	"""
	For a given quantile ``q``, return a boolean array indicating
	the locations of pixels above ``q`` (within the original segment)
	that are also part of the "main" clump.
	"""
	cdef double threshold = _quantile(_sorted_pixelvals_stamp_no_bg_nonnegative, q)
	cdef cnp.ndarray above_threshold = _cutout_stamp_maskzeroed_no_bg_nonnegative >= threshold

	# Instead of assuming that the main segment is at the center
	# of the stamp, use the position of the brightest pixel:
	cdef int ic = _y_maxval_stamp
	cdef int jc = _x_maxval_stamp

	# Grow regions using 8-connected neighbor "footprint"
	cdef cnp.ndarray s = ndi.generate_binary_structure(2, 2)
	cdef cnp.ndarray[int,ndim=2] labeled_array
	cdef int num_features
	labeled_array = cnp.PyArray_EMPTY(2, above_threshold.shape, cnp.NPY_INT, 0)
	num_features = _ni_label._label(above_threshold, s, labeled_array)
	# labeled_array, num_features = ndi.label(above_threshold, structure=s)

	# Sanity check (brightest pixel should be part of the main clump):
	assert labeled_array[ic, jc] != 0

	return labeled_array == labeled_array[ic, jc]

cpdef double _segmap_mid_function(double q, cnp.ndarray[double,ndim=1] _sorted_pixelvals_stamp_no_bg_nonnegative, cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed_no_bg_nonnegative,
								  int _x_maxval_stamp, int _y_maxval_stamp, Flags flags, ConstantsSetting constants):
	"""
	Helper function to calculate the MID segmap.

	For a given quantile ``q``, return the ratio of the mean flux of
	pixels at the level of ``q`` (within the main clump) divided by
	the mean of pixels above ``q`` (within the main clump).
	"""
	cdef cnp.ndarray locs_main_clump = _segmap_mid_main_clump(q, _sorted_pixelvals_stamp_no_bg_nonnegative, _cutout_stamp_maskzeroed_no_bg_nonnegative, _x_maxval_stamp, _y_maxval_stamp)

	cdef double mean_flux_main_clump = np.mean(
		_cutout_stamp_maskzeroed_no_bg_nonnegative[locs_main_clump])
	cdef double mean_flux_new_pixels = _quantile(
		_sorted_pixelvals_stamp_no_bg_nonnegative, q)

	cdef double ratio
	if mean_flux_main_clump == 0:
		warnings.warn('[segmap_mid] Zero flux sum.', AstropyUserWarning)
		ratio = 1.0
		flags.set_flag_true(15)
	else:
		ratio = mean_flux_new_pixels / mean_flux_main_clump

	return ratio - constants.eta

cdef cnp.ndarray get_segmap_mid(cnp.ndarray[double,ndim=1] _sorted_pixelvals_stamp_no_bg_nonnegative, cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed_no_bg_nonnegative,
								int _x_maxval_stamp, int _y_maxval_stamp, Flags flags, ConstantsSetting constants):
	"""
	Create a new segmentation map as described in Section 4.3 from
	Freeman et al. (2013).

	Notes
	-----
	This implementation improves upon previous ones by making
	the MID segmap independent from the number of quantiles
	used in the calculation, as well as other parameters.
	"""
	cdef int num_pixelvals = len(_sorted_pixelvals_stamp_no_bg_nonnegative)

	# In some rare cases (as a consequence of an erroneous
	# initial segmap, as in J095553.0+694048_g.fits.gz),
	# the MID segmap is technically undefined because the
	# mean flux of "newly added" pixels never reaches the
	# target value, at least within the original segmap.
	# In these cases we simply assume that the MID segmap
	# is equal to the Gini segmap.
	if _segmap_mid_function(0.0, _sorted_pixelvals_stamp_no_bg_nonnegative, _cutout_stamp_maskzeroed_no_bg_nonnegative, _x_maxval_stamp, _y_maxval_stamp, flags, constants) > 0.0:
		if constants.verbose:
			warnings.warn('segmap_mid is undefined; using segmap_gini instead.', AstropyUserWarning)
		return None

	# Find appropriate quantile using numerical solver
	cdef double q_min = 0.0
	cdef double q_max = 1.0
	cdef double xtol = 1.0 / float(num_pixelvals)
	cdef double q = opt.brentq(_segmap_mid_function, q_min, q_max, (_sorted_pixelvals_stamp_no_bg_nonnegative, _cutout_stamp_maskzeroed_no_bg_nonnegative, _x_maxval_stamp, _y_maxval_stamp, flags, constants), xtol=xtol)

	cdef cnp.ndarray locs_main_clump = _segmap_mid_main_clump(q, _sorted_pixelvals_stamp_no_bg_nonnegative, _cutout_stamp_maskzeroed_no_bg_nonnegative, _x_maxval_stamp, _y_maxval_stamp)

	# Regularize a bit the shape of the segmap:
	cdef cnp.ndarray segmap_float = ndi.uniform_filter(
		np.float64(locs_main_clump), size=constants.boxcar_size_mid)
	cdef cnp.ndarray[cnp.npy_bool,ndim=2] segmap = segmap_float > 0.5

	# Make sure that brightest pixel is in segmap
	cdef int ic = _y_maxval_stamp
	cdef int jc = _x_maxval_stamp
	if not segmap[ic, jc]:
		warnings.warn('[segmap_mid] Adding brightest pixel to segmap.', AstropyUserWarning)
		segmap[ic, jc] = True
		flags.set_flag_true(15)

	# Grow regions with 8-connected neighbor "footprint"
	cdef cnp.ndarray s = ndi.generate_binary_structure(2, 2)
	cdef cnp.ndarray[int, ndim=2] labeled_array
	cdef int num_features
	labeled_array = cnp.PyArray_EMPTY(2, segmap.shape, cnp.NPY_INT, 0)
	num_features = _ni_label._label(segmap, s, labeled_array)
	# labeled_array, num_features = ndi.label(segmap, structure=s)

	return labeled_array == labeled_array[ic, jc]

cdef cnp.ndarray[double,ndim=2] get_cutout_mid(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed_no_bg_nonnegative, cnp.ndarray _segmap_mid):
	"""
	Apply the MID segmap to the postage stamp cutout
	and set negative pixels to zero.
	"""
	cdef cnp.ndarray image = cnp.PyArray_Where(_segmap_mid, _cutout_stamp_maskzeroed_no_bg_nonnegative, 0.0)
	return image

cdef cnp.ndarray[double,ndim=1] get_sorted_pixelvals_mid(cnp.ndarray[double,ndim=2] _cutout_mid, cnp.ndarray _mask_stamp_no_bg):
	"""
	Just the sorted pixel values of the MID cutout.
	"""
	cdef cnp.ndarray[double,ndim=2] image = _cutout_mid
	cdef cnp.ndarray[double,ndim=1] image_no_bg_list = image[~_mask_stamp_no_bg]
	cnp.PyArray_Sort(image_no_bg_list, -1, cnp.NPY_QUICKSORT)
	return image_no_bg_list

cdef cnp.ndarray[cnp.npy_int64,ndim=1] _multimode_function(double q,
														   cnp.ndarray[double,ndim=1] _sorted_pixelvals_mid,
														   cnp.ndarray[double,ndim=2] _cutout_mid):
	"""
	Helper function to calculate the multimode statistic.
	Returns the sorted "areas" of the clumps at quantile ``q``.
	"""
	cdef double threshold = _quantile(_sorted_pixelvals_mid, q)
	cdef cnp.ndarray above_threshold = _cutout_mid >= threshold

	# Neighbor "footprint" for growing regions, including corners:
	cdef cnp.ndarray s = ndi.generate_binary_structure(2, 2)
	cdef cnp.ndarray[int, ndim=2] labeled_array
	cdef int num_features
	labeled_array = cnp.PyArray_EMPTY(2, above_threshold.shape, cnp.NPY_INT, 0)
	num_features = _ni_label._label(above_threshold, s, labeled_array)
	# labeled_array, num_features = ndi.label(above_threshold, structure=s)

	# Zero is reserved for non-labeled pixels:
	cdef cnp.ndarray[int,ndim=1] labeled_array_nonzero, labels
	cdef cnp.ndarray[cnp.npy_int64,ndim=1] counts, sorted_counts

	labeled_array_nonzero = labeled_array[labeled_array != 0]
	labels, counts = np.unique(labeled_array_nonzero, return_counts=True)
	cnp.PyArray_Sort(counts, -1, cnp.NPY_QUICKSORT)
	sorted_counts = counts[::-1]

	return sorted_counts

cpdef double _multimode_ratio(double q, cnp.ndarray[double,ndim=1] _sorted_pixelvals_mid, cnp.ndarray[double,ndim=2] _cutout_mid):
	"""
	For a given quantile ``q``, return the "ratio" (A2/A1)*A2
	multiplied by -1, which is used for minimization.
	"""
	cdef double invalid = np.sum(_cutout_mid)  # high "energy" for basin-hopping
	cdef double ratio
	cdef cnp.ndarray[cnp.npy_int64, ndim=1] sorted_counts
	if (q < 0) or (q > 1):
		ratio = invalid
	else:
		sorted_counts = _multimode_function(q, _sorted_pixelvals_mid, _cutout_mid)
		if len(sorted_counts) == 1:
			ratio = invalid
		else:
			ratio = -1.0 * float(sorted_counts[1]) ** 2 / float(sorted_counts[0])

	return ratio

cdef double get_multimode(cnp.ndarray[double,ndim=1] _sorted_pixelvals_mid, cnp.ndarray[double,ndim=2] _cutout_mid, ConstantsSetting constants):
	"""
	Calculate the multimode (M) statistic as described in
	Freeman et al. (2013) and Peth et al. (2016).

	Notes
	-----
	In the original publication, Freeman et al. (2013)
	recommends using the more robust quantity (A2/A1)*A2,
	while Peth et al. (2016) recommends using the
	size-independent quantity A2/A1. Here we take a mixed
	approach (which, incidentally, is also what Mike Peth's
	IDL implementation actually does): we maximize the
	quantity (A2/A1)*A2 (as a function of the brightness
	threshold) but ultimately define the M statistic
	as the corresponding A2/A1 value.

	The original IDL implementation only explores quantiles
	in the range [0.5, 1.0], at least with the default settings.
	While this might be useful in practice, in theory the
	maximum (A2/A1)*A2 value could also happen in the quantile
	range [0.0, 0.5], so here we take a safer, more general
	approach and search over [0.0, 1.0].

	In practice, the quantity (A2/A1)*A2 is tricky to optimize.
	We improve over previous implementations by doing so
	in two stages: starting with a brute-force search
	over a relatively coarse array of quantiles, as in the
	original implementation, followed by a finer search using
	the basin-hopping method. This should do a better job of
	finding the global maximum.

	"""
	cdef double q_min = 0.0
	cdef double q_max = 1.0

	# STAGE 1: brute-force

	# We start with a relatively coarse separation between the
	# quantiles, equal to the value used in the original IDL
	# implementation. If every calculated ratio is invalid, we
	# try a smaller size.
	cdef double mid_stepsize = 0.02
	cdef cnp.ndarray[double, ndim=1] quantile_array
	cdef cnp.ndarray[double, ndim=1] ratio_array
	cdef int k, k_min
	cdef double q, q0, ratio_min

	while True:
		quantile_array = cnp.PyArray_Arange(q_min, q_max, mid_stepsize, cnp.NPY_DOUBLE)
		ratio_array = cnp.PyArray_ZEROS(1, quantile_array.shape, cnp.NPY_DOUBLE, 0)
		# cdef cnp.ndarray ratio_array = np.zeros_like(quantile_array)
		for k in range(len(quantile_array)):
			q = quantile_array[k]
			ratio_array[k] = _multimode_ratio(q, _sorted_pixelvals_mid, _cutout_mid)
		k_min = np.argmin(ratio_array)
		q0 = quantile_array[k_min]
		ratio_min = ratio_array[k_min]
		if ratio_min < 0:  # valid "ratios" should be negative
			break
		elif mid_stepsize < 1e-3:
			if constants.verbose:
				warnings.warn(
					'[M statistic] Single clump!', AstropyUserWarning)
			# This sometimes happens when the source is a star, so the user
			# might want to discard M=0 cases, depending on the dataset.
			return 0.0
		else:
			mid_stepsize = mid_stepsize / 2.0
			if constants.verbose:
				warnings.warn('[M statistic] Reduced stepsize to %g.' % (
					mid_stepsize), AstropyUserWarning)

	# STAGE 2: basin-hopping method

	# The results seem quite robust to changes in this parameter,
	# so I leave it hardcoded for now:
	cdef double mid_bh_rel_temp = 0.5

	cdef double temperature = -1.0 * mid_bh_rel_temp * ratio_min

	res = opt.basinhopping(
		_multimode_ratio, q0, minimizer_kwargs={"method": "Nelder-Mead", "args": (_sorted_pixelvals_mid, _cutout_mid)},
		niter=constants.niter_bh_mid, T=temperature, stepsize=mid_stepsize,
		interval=constants.niter_bh_mid / 2, disp=False, seed=0)
	cdef double q_final = res.x[0]
	"""
	cdef OptimizeResult res = basinhopping.pyx(
		self._multimode_ratio, q0, self._niter_bh_mid, temperature, mid_stepsize, self._niter_bh_mid / 2)
	cdef double q_final = res.x
	"""
	# Finally, return A2/A1 instead of (A2/A1)*A2
	cdef cnp.ndarray[cnp.npy_int64,ndim=1] sorted_counts = _multimode_function(q_final, _sorted_pixelvals_mid, _cutout_mid)

	return float(sorted_counts[1]) / float(sorted_counts[0])

cdef cnp.ndarray[double,ndim=2] get_cutout_mid_smooth(cnp.ndarray[double,ndim=2] _cutout_mid, ConstantsSetting constants):
	"""
	Just a Gaussian-smoothed version of the zero-masked image used
	in the MID calculations.
	"""
	cdef cnp.ndarray image_smooth = ndi.gaussian_filter(_cutout_mid, constants.sigma_mid)
	return image_smooth

cdef tuple get_watershed_mid(cnp.ndarray[double,ndim=2] _cutout_mid_smooth):
	"""
	This replaces the "i_clump" routine from the original IDL code.
	The main difference is that we do not place a limit on the
	number of labeled regions (previously limited to 100 regions).
	This is also much faster, thanks to the highly optimized
	"peak_local_max" and "watershed" skimage routines.
	Returns a labeled array indicating regions around local maxima.
	"""
	cdef cnp.ndarray[cnp.npy_int64,ndim=2] peaks = skimage.feature.peak_local_max(
		_cutout_mid_smooth, num_peaks=INFINITY)
	cdef int num_peaks = peaks.shape[0]
	# The zero label is reserved for the background:
	cdef cnp.ndarray peak_labels = cnp.PyArray_Arange(1, num_peaks + 1, 1, cnp.NPY_INT64)
	cdef cnp.ndarray ypeak = peaks[:,0]
	cdef cnp.ndarray xpeak = peaks[:,1]
	# ypeak, xpeak = peaks.T

	cdef cnp.ndarray[cnp.npy_int64,ndim=2] markers = cnp.PyArray_ZEROS(2, _cutout_mid_smooth.shape, cnp.NPY_INT64, 0)
	markers[ypeak, xpeak] = peak_labels

	cdef cnp.ndarray mask = _cutout_mid_smooth > 0
	cdef cnp.ndarray labeled_array = skimage.segmentation.watershed(
		-_cutout_mid_smooth, markers, connectivity=2, mask=mask)

	return labeled_array, peak_labels, xpeak, ypeak

cdef tuple get_intensity_sums(cnp.ndarray[double,ndim=2] _cutout_mid_smooth, cnp.ndarray labeled_array, cnp.ndarray[cnp.npy_int64,ndim=1] peak_labels, cnp.ndarray xpeak, cnp.ndarray ypeak):
	"""
	Helper function to calculate the intensity (I) and
	deviation (D) statistics.
	"""
	cdef int k
	cdef cnp.npy_int64 label
	cdef cnp.ndarray locs

	cdef cnp.ndarray[double,ndim=1] flux_sums = cnp.PyArray_ZEROS(1,peak_labels.shape,cnp.NPY_DOUBLE,0)
	for k in range(len(peak_labels)):
		label = peak_labels[k]
		locs = labeled_array == label
		flux_sums[k] = np.sum(_cutout_mid_smooth[locs])
	cdef cnp.ndarray sid = cnp.PyArray_ArgSort(flux_sums,-1,cnp.NPY_QUICKSORT)[::-1]
	cdef cnp.ndarray sorted_flux_sums = flux_sums[sid]
	cdef cnp.ndarray sorted_xpeak = xpeak[sid]
	cdef cnp.ndarray sorted_ypeak = ypeak[sid]

	return sorted_flux_sums, sorted_xpeak, sorted_ypeak

cdef double get_intensity(cnp.ndarray[double,ndim=1] sorted_flux_sums):
	"""
	Calculate the intensity (I) statistic as described in
	Peth et al. (2016).
	"""
	if len(sorted_flux_sums) <= 1:
		# Unlike the M=0 cases, there seem to be some legitimate
		# I=0 cases, so we do not turn on the "bad measurement" flag.
		return 0.0
	else:
		return sorted_flux_sums[1] / sorted_flux_sums[0]

cdef double get_deviation(cnp.ndarray[double,ndim=2] _cutout_mid, cnp.ndarray _segmap_mid, cnp.ndarray[double,ndim=1] sorted_flux_sums,
						  cnp.ndarray[cnp.npy_int64,ndim=1] sorted_xpeak, cnp.ndarray[cnp.npy_int64,ndim=1] sorted_ypeak,
						  Flags flags):
	"""
	Calculate the deviation (D) statistic as described in
	Peth et al. (2016).
	"""
	cdef cnp.ndarray image = _cutout_mid  # skimage wants double

	if len(sorted_flux_sums) == 0:
		warnings.warn('[deviation] There are no peaks.', AstropyUserWarning)
		flags.set_flag_true(16)
		return -99.0  # invalid

	cdef int xp = sorted_xpeak[0]
	cdef int yp = sorted_ypeak[0]

	# Calculate centroid
	cdef double[:,:] M = skimage.measure.moments(image, order=1)
	if M[0, 0] <= 0:
		warnings.warn('[deviation] Nonpositive flux within MID segmap.',
					  AstropyUserWarning)
		flags.set_flag_true(16)
		return -99.0  # invalid
	cdef double yc = M[1, 0] / M[0, 0]
	cdef double xc = M[0, 1] / M[0, 0]

	cdef double area = np.sum(_segmap_mid)
	cdef double D = sqrt(pi / area) * sqrt((xp - xc) ** 2 + (yp - yc) ** 2)

	if not isfinite(D):
		warnings.warn('Invalid D-statistic.', AstropyUserWarning)
		flags.set_flag_true(17)
		return -99.0  # invalid

	return D

cdef MIDInfo calc_mid(BaseInfo base_info):
	cdef MIDInfo mid_info = MIDInfo()

	cdef cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed_no_bg_nonnegative = get_cutout_stamp_maskzeroed_no_bg_nonnegative(base_info._cutout_stamp_maskzeroed_no_bg)
	cdef cnp.ndarray[double,ndim=1] _sorted_pixelvals_stamp_no_bg_nonnegative = get_sorted_pixelvals_stamp_no_bg_nonnegative(_cutout_stamp_maskzeroed_no_bg_nonnegative, base_info._mask_stamp_no_bg)
	cdef double maxval = np.max(base_info._cutout_stamp_maskzeroed_no_bg)
	cdef long[:] maxval_stamp_pos = np.argwhere(base_info._cutout_stamp_maskzeroed_no_bg == maxval)[0]
	cdef int _x_maxval_stamp = maxval_stamp_pos[1]
	cdef int _y_maxval_stamp = maxval_stamp_pos[0]
	cdef cnp.ndarray _segmap_mid = get_segmap_mid(_sorted_pixelvals_stamp_no_bg_nonnegative, _cutout_stamp_maskzeroed_no_bg_nonnegative, _x_maxval_stamp, _y_maxval_stamp, mid_info.flags, base_info.constants)
	if _segmap_mid is None:
		if base_info.g_m20 is not None:
			_segmap_mid = base_info.g_m20._segmap_gini
		else:
			return None

	cdef cnp.ndarray[double,ndim=2] _cutout_mid = get_cutout_mid(_cutout_stamp_maskzeroed_no_bg_nonnegative, _segmap_mid)
	mid_info._cutout_mid = _cutout_mid

	cdef cnp.ndarray[double,ndim=1] _sorted_pixelvals_mid = get_sorted_pixelvals_mid(_cutout_mid, base_info._mask_stamp_no_bg)

	mid_info.multimode = get_multimode(_sorted_pixelvals_mid, _cutout_mid, base_info.constants)

	cdef cnp.ndarray[double,ndim=2] _cutout_mid_smooth = get_cutout_mid_smooth(_cutout_mid, base_info.constants)

	cdef tuple _watershed_mid = get_watershed_mid(_cutout_mid_smooth)

	cdef cnp.ndarray labeled_array = _watershed_mid[0]
	cdef cnp.ndarray[cnp.npy_int64,ndim=1] peak_labels = _watershed_mid[1]
	cdef cnp.ndarray xpeak = _watershed_mid[2]
	cdef cnp.ndarray ypeak = _watershed_mid[3]

	cdef tuple _intensity_sums = get_intensity_sums(_cutout_mid_smooth, labeled_array, peak_labels, xpeak, ypeak)

	cdef cnp.ndarray[double,ndim=1] sorted_flux_sums = _intensity_sums[0]
	cdef cnp.ndarray[cnp.npy_int64,ndim=1] sorted_xpeak = _intensity_sums[1]
	cdef cnp.ndarray[cnp.npy_int64,ndim=1] sorted_ypeak = _intensity_sums[2]

	mid_info.intensity = get_intensity(sorted_flux_sums)
	mid_info.deviation = get_deviation(_cutout_mid, _segmap_mid, sorted_flux_sums, sorted_xpeak, sorted_ypeak, mid_info.flags)

	return mid_info