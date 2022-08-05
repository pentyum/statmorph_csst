# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import warnings
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, atan2, log10
import scipy.ndimage as ndi
import skimage.measure
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import _ni_label

from .petrosian cimport _rpetro_ellip_generic
from .photutils_simplified cimport EllipticalAnnulus, _aperture_mean_nomask
from .array_utils cimport create_22_mat, sum_1d_d, cumsum_1d_d
from .flags cimport Flags
from .statmorph cimport ConstantsSetting, BaseInfo, GiniM20Info

cnp.import_array()

cdef double[:,:] _covariance_generic(cnp.ndarray[double,ndim=2] cutout_stamp_maskzeroed_no_bg,
													(double, double) asymmetry_center, Flags flags):
	"""
	The covariance matrix of a Gaussian function that has the same
	second-order moments as the source, with respect to ``(xc, yc)``.
	可能产生警告8
	"""
	# skimage wants double precision:
	# cdef cnp.ndarray image = np.float64(self._cutout_stamp_maskzeroed_no_bg)
	cdef cnp.ndarray image = cutout_stamp_maskzeroed_no_bg

	# Calculate moments w.r.t. given center
	cdef double xc = asymmetry_center[0]
	cdef double yc = asymmetry_center[1]
	cdef double[:,:] Mc = skimage.measure.moments_central(image, center=(yc, xc), order=2)
	assert Mc[0, 0] > 0

	#cdef cnp.ndarray covariance = np.array([
	#	[Mc[0, 2], Mc[1, 1]],
	#	[Mc[1, 1], Mc[2, 0]]])
	cdef cnp.ndarray[double,ndim=2] covariance = create_22_mat(Mc[0, 2], Mc[1, 1],
												Mc[1, 1], Mc[2, 0])
	covariance /= Mc[0, 0]  # normalize

	# If there are nonpositive second moments, we deal with them,
	# but we indicate that there's something wrong with the data.
	if (covariance[0, 0] <= 0) or (covariance[1, 1] <= 0):
		warnings.warn('Nonpositive second moment.', AstropyUserWarning)
		flags.set_flag_true(0)

	# Modify covariance matrix in case of "infinitely thin" sources
	# by iteratively increasing the diagonal elements (see SExtractor
	# manual, eq. 43). Note that we allow negative moments.
	cdef double rho = 1.0 / 12.0  # variance of 1 pixel-wide top-hat distribution
	cdef double x2 = covariance[0, 0]
	cdef double xy = covariance[0, 1]
	cdef double y2 = covariance[1, 1]

	while fabs(x2 * y2 - xy ** 2) < rho ** 2:
		x2 += (x2 >= 0) * rho - (x2 < 0) * rho  # np.sign(0) == 0 is no good
		y2 += (y2 >= 0) * rho - (y2 < 0) * rho
	#covariance = np.array([[x2, xy],
	#					   [xy, y2]])
	covariance[0, 0] = x2
	covariance[0, 1] = xy
	covariance[1, 0] = xy
	covariance[1, 1] = y2

	return covariance


cdef double[:] _eigvals_generic(double[:,:] covariance, Flags flags):
	"""
	The ordered (largest first) eigenvalues of the covariance
	matrix, which correspond to the *squared* semimajor and
	semiminor axes. Note that we allow negative eigenvalues.
	可能产生警告9
	"""
	cdef double[:] eigvals = np.linalg.eigvals(covariance)
	eigvals = np.sort(np.abs(eigvals))[::-1]  # largest first (by abs. value)

	# We deal with negative eigenvalues, but we indicate that something
	# is not OK with the data (eigenvalues cannot be exactly zero after
	# the SExtractor-like regularization routine).
	if np.any(eigvals.base < 0):
		warnings.warn('Some negative eigenvalues.', AstropyUserWarning)
		flags.set_flag_true(1)

	return eigvals


cdef double _ellipticity_generic(double[:] eigvals):
	"""
	The ellipticity of (the Gaussian function that has the same
	second-order moments as) the source. Note that we allow
	negative eigenvalues.
	"""
	cdef double a = sqrt(fabs(eigvals[0]))
	cdef double b = sqrt(fabs(eigvals[1]))

	return 1.0 - (b / a)


cdef double _elongation_generic(double[:] eigvals):
	"""
	The elongation of (the Gaussian function that has the same
	second-order moments as) the source. Note that we allow
	negative eigenvalues.
	"""
	cdef double a = sqrt(fabs(eigvals[0]))
	cdef double b = sqrt(fabs(eigvals[1]))

	return a / b


cdef double _orientation_generic(double[:,:] covariance):
	"""
	The orientation (in radians) of the source.
	"""
	cdef double x2 = covariance[0, 0]
	cdef double xy = covariance[0, 1]
	cdef double y2 = covariance[1, 1]
	# x2, xy, xy, y2 = covariance.flat

	# SExtractor manual, eq. (21):
	cdef double theta = 0.5 * atan2(2.0 * xy, x2 - y2)

	return theta


#######################
# Gini-M20 statistics #
#######################

cdef cnp.ndarray[cnp.npy_bool,ndim=2] get_segmap_gini(cnp.ndarray[double,ndim=2] cutout_stamp_maskzeroed, double rpetro_ellip,
													  double elongation_asymmetry, double orientation_asymmetry, (double,double) centroid,
													  Flags flags, ConstantsSetting constants):
	"""
	Create a new segmentation map (relative to the "postage stamp")
	based on the elliptical Petrosian radius.
	可能产生警告22,23,24
	"""
	# Smooth image
	cdef double petro_sigma = constants.petro_fraction_gini * rpetro_ellip
	cdef cnp.ndarray cutout_smooth = ndi.gaussian_filter(cutout_stamp_maskzeroed, petro_sigma)

	# Use mean flux at the Petrosian "radius" as threshold
	cdef double a_in = rpetro_ellip - 0.5 * constants.annulus_width
	cdef double a_out = rpetro_ellip + 0.5 * constants.annulus_width
	cdef double b_out = a_out / elongation_asymmetry
	cdef double theta = orientation_asymmetry
	cdef EllipticalAnnulus ellip_annulus = EllipticalAnnulus(
		centroid, a_in, a_out, b_out, theta=theta)
	cdef double ellip_annulus_mean_flux = _aperture_mean_nomask(
		ellip_annulus, cutout_smooth)

	cdef cnp.ndarray above_threshold = cutout_smooth >= ellip_annulus_mean_flux

	# Grow regions with 8-connected neighbor "footprint"
	cdef cnp.ndarray s = ndi.generate_binary_structure(2, 2)
	cdef cnp.ndarray[int, ndim=2] labeled_array
	cdef int num_features
	labeled_array = cnp.PyArray_EMPTY(2, above_threshold.shape, cnp.NPY_INT, 0)
	num_features = _ni_label._label(above_threshold, s, labeled_array)
	# labeled_array, num_features = ndi.label(above_threshold, structure=s)

	# In some rare cases (e.g., Pan-STARRS J020218.5+672123_g.fits.gz),
	# this results in an empty segmap, so there is nothing to do.
	if num_features == 0:
		warnings.warn('[segmap_gini] Empty Gini segmap!',
					  AstropyUserWarning)
		flags.set_flag_unusual_true(11)
		return above_threshold

	# In other cases (e.g., object 110 from CANDELS/GOODS-S WFC/F160W),
	# the Gini segmap occupies the entire image, which is also not OK.
	if np.sum(above_threshold) == cnp.PyArray_SIZE(cutout_smooth):
		warnings.warn('[segmap_gini] Full Gini segmap!',
					  AstropyUserWarning)
		flags.set_flag_true(10)
		return above_threshold

	# If more than one region, activate the "bad measurement" flag
	# and only keep segment that contains the brightest pixel.
	cdef cnp.ndarray segmap
	cdef int ic, jc
	if num_features > 1:
		warnings.warn('[segmap_gini] Disjoint features in Gini segmap.',
					  AstropyUserWarning)
		flags.set_flag_true(11)
		ic, jc = np.argwhere(cutout_smooth == np.max(cutout_smooth))[0]
		assert labeled_array[ic, jc] != 0
		segmap = labeled_array == labeled_array[ic, jc]
	else:
		segmap = above_threshold

	return segmap

cdef double get_gini(cnp.ndarray[double,ndim=2] cutout_stamp_maskzeroed, cnp.ndarray[cnp.npy_bool,ndim=2] segmap_gini, Flags flags):
	"""
	Calculate the Gini coefficient as described in Lotz et al. (2004).
	可能产生警告25
	"""
	cdef cnp.ndarray[double,ndim=1] image = cnp.PyArray_Flatten(cutout_stamp_maskzeroed,cnp.NPY_CORDER)
	cdef cnp.ndarray[cnp.npy_bool,ndim=1] segmap = cnp.PyArray_Flatten(segmap_gini,cnp.NPY_CORDER)

	cdef cnp.ndarray[double,ndim=1] sorted_pixelvals = np.sort(np.abs(image[segmap]))
	cdef int n = len(sorted_pixelvals)
	if n <= 1 or sum_1d_d(sorted_pixelvals) == 0:
		warnings.warn('[gini] Not enough data for Gini calculation.',
					  AstropyUserWarning)
		flags.set_flag_true(12)
		return -99.0  # invalid

	cdef cnp.ndarray indices = cnp.PyArray_Arange(1, n+1, 1, cnp.NPY_INT)  # start at i=1
	cdef double gini = (sum_1d_d((2 * indices - n - 1) * sorted_pixelvals) /
			(float(n - 1) * sum_1d_d(sorted_pixelvals)))

	return gini

cdef double get_m20(cnp.ndarray[double,ndim=2] cutout_stamp_maskzeroed, cnp.ndarray[cnp.npy_bool,ndim=2] segmap_gini, Flags flags):
	"""
	Calculate the M_20 coefficient as described in Lotz et al. (2004).
	可能产生警告26,27,28
	"""
	if np.sum(segmap_gini) == 0:
		return -99.0  # invalid

	# Use the same region as in the Gini calculation
	cdef cnp.ndarray image = cnp.PyArray_Where(segmap_gini, cutout_stamp_maskzeroed, 0.0)
	# image = np.float64(image)  # skimage wants double

	# Calculate centroid
	cdef double[:,:] M = skimage.measure.moments(image, order=1)
	if M[0, 0] <= 0:
		warnings.warn('[deviation] Nonpositive flux within Gini segmap.',
					  AstropyUserWarning)
		flags.set_flag_true(13)
		return -99.0  # invalid
	cdef double yc = M[1, 0] / M[0, 0]
	cdef double xc = M[0, 1] / M[0, 0]

	# Calculate second total central moment
	cdef double[:,:] Mc = skimage.measure.moments_central(image, center=(yc, xc), order=2)
	cdef double second_moment_tot = Mc[0, 2] + Mc[2, 0]

	# Calculate threshold pixel value
	cdef cnp.ndarray[double,ndim=1] sorted_pixelvals = np.sort(cnp.PyArray_Flatten(image,cnp.NPY_CORDER))
	cdef cnp.ndarray[double,ndim=1] flux_fraction = cumsum_1d_d(sorted_pixelvals) / sum_1d_d(sorted_pixelvals)
	cdef cnp.ndarray[double,ndim=1] sorted_pixelvals_20 = sorted_pixelvals[flux_fraction >= 0.8]
	if len(sorted_pixelvals_20) == 0:
		# This can happen when there are very few pixels.
		warnings.warn('[m20] Not enough data for M20 calculation.',
					  AstropyUserWarning)
		flags.set_flag_true(14)
		return -99.0  # invalid
	cdef double threshold = sorted_pixelvals_20[0]

	# Calculate second moment of the brightest pixels
	cdef cnp.ndarray image_20 = cnp.PyArray_Where(image >= threshold, image, 0.0)
	cdef double[:,:] Mc_20 = skimage.measure.moments_central(image_20, center=(yc, xc), order=2)
	cdef double second_moment_20 = Mc_20[0, 2] + Mc_20[2, 0]
	cdef double m20

	if (second_moment_20 <= 0) | (second_moment_tot <= 0):
		warnings.warn('[m20] Negative second moment(s).',
					  AstropyUserWarning)
		flags.set_flag_true(12)
		m20 = -99.0  # invalid
	else:
		m20 = log10(second_moment_20 / second_moment_tot)

	return m20

cdef GiniM20Info calc_g_m20(BaseInfo base_info, (double, double) asymmetry_center):
	cdef GiniM20Info g_m20_info = GiniM20Info()
	cdef double elongation_asymmetry, orientation_asymmetry
	cdef double[:,:] _covariance_asymmetry = _covariance_generic(base_info._cutout_stamp_maskzeroed_no_bg, asymmetry_center, g_m20_info.flags)
	cdef double[:] _eigvals_asymmetry = _eigvals_generic(_covariance_asymmetry, g_m20_info.flags)
	elongation_asymmetry = _elongation_generic(_eigvals_asymmetry)
	orientation_asymmetry = _orientation_generic(_covariance_asymmetry)
	g_m20_info.rpetro_ellip = _rpetro_ellip_generic(base_info._cutout_stamp_maskzeroed, asymmetry_center, elongation_asymmetry, orientation_asymmetry, base_info._diagonal_distance, g_m20_info.flags, base_info.constants)
	g_m20_info._segmap_gini = get_segmap_gini(base_info._cutout_stamp_maskzeroed, g_m20_info.rpetro_ellip, elongation_asymmetry, orientation_asymmetry, base_info._centroid, g_m20_info.flags, base_info.constants)
	g_m20_info.gini = get_gini(base_info._cutout_stamp_maskzeroed, g_m20_info._segmap_gini, g_m20_info.flags)
	g_m20_info.m20 = get_m20(base_info._cutout_stamp_maskzeroed, g_m20_info._segmap_gini, g_m20_info.flags)
	return g_m20_info
