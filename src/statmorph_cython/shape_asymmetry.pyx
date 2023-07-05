# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import warnings
cimport numpy as cnp
import photutils
import skimage
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyUserWarning
from libc.math cimport cos, sin
from numpy.math cimport isnan
import numpy as np
import scipy.ndimage as ndi

from .cas cimport simplified_rot180
from .constants_setting cimport ConstantsSetting
from .photutils_simplified cimport _radius_at_fraction_of_total_circ, _radius_at_fraction_of_total_ellip, CircularAnnulus, ApertureMask, do_photometry, CircularAperture
from .flags cimport Flags
from .statmorph cimport StampMorphology, ShapeAsymmetryInfo, CASInfo, GiniM20Info

cnp.import_array()

cdef double _shape_asymmetry_function((double, double) center, cnp.ndarray[double,ndim=2] image, cnp.ndarray[cnp.npy_bool,ndim=2] _mask_stamp, double rmax_circ, Flags flags, ConstantsSetting constants):
	cdef int ny = image.shape[0]
	cdef int nx = image.shape[1]
	cdef double xc = center[0]
	cdef double yc = center[1]
	cdef int image_size = nx * ny

	if xc < 0 or xc >= nx or yc < 0 or yc >= ny:
		warnings.warn('[asym_center] Minimizer tried to exit bounds.',
					  AstropyUserWarning)
		flags.set_flag_true(6)
		# Return high value to keep minimizer within range:
		return 100.0

	# Rotate around given center
	cdef cnp.ndarray[cnp.npy_bool,ndim=2] image_180
	# cdef cnp.ndarray image_180 = skimage.transform.rotate(image, 180.0, center=center)
	if 0 <= constants.simplified_rot_threshold < image_size:
		image_180 = simplified_rot180(image, center)
	else:
		image_180 = skimage.transform.rotate(image, 180.0, center=center)

	# Apply symmetric mask
	cdef cnp.ndarray[cnp.npy_bool, ndim=2] mask = _mask_stamp
	cdef cnp.ndarray[cnp.npy_bool, ndim=2] mask_180
	if 0 <= constants.simplified_rot_threshold < image_size:
		mask_180 = simplified_rot180(mask, center)
	else:
		mask_180 = skimage.transform.rotate(mask, 180.0, center=center) >= 0.5

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] mask_symmetric = mask | mask_180
	image = cnp.PyArray_Where(~mask_symmetric, image, 0.0)
	image_180 = cnp.PyArray_Where(~mask_symmetric, image_180, 0.0)

	# Create aperture for the chosen kind of asymmetry
	if isnan(rmax_circ) or (rmax_circ <= 0):
		warnings.warn('[shape_asym] Invalid rmax_circ value.',
					  AstropyUserWarning)
		flags.set_flag_true(7)
		return -99.0  # invalid
	cdef CircularAperture ap = CircularAperture(center, rmax_circ)

	# Apply eq. 10 from Lotz et al. (2004)
	cdef double ap_abs_sum = do_photometry(ap, np.abs(image))
	cdef double ap_abs_diff = do_photometry(ap, np.abs(image_180 - image))

	if ap_abs_sum == 0.0:
		warnings.warn('[shape_asymmetry_function] Zero flux sum.',
					  AstropyUserWarning)
		flags.set_flag_true(8)  # unusual
		return -99.0  # invalid

	# The shape asymmetry of the background is zero
	cdef double asym = ap_abs_diff / ap_abs_sum

	return asym

cdef cnp.ndarray[cnp.npy_bool,ndim=2] segmap_shape_asym(cnp.ndarray[double,ndim=2] cutout_stamp_maskzeroed, (double,double) asymmetry_center, double rpetro_ellip, cnp.ndarray mask_stamp, tuple slice_skybox, cnp.ndarray[cnp.npy_bool,ndim=2] mask_stamp_no_bg, Flags flags, ConstantsSetting constants):

	cdef int ny = cutout_stamp_maskzeroed.shape[0]
	cdef int nx = cutout_stamp_maskzeroed.shape[1]

	# Center at pixel that minimizes asymmetry
	cdef (double, double) center = asymmetry_center

	# Create a circular annulus around the center
	# that only contains background sky (hopefully).
	cdef double r_in = constants.petro_extent_flux * rpetro_ellip
	cdef double r_out = 2.0 * constants.petro_extent_flux * rpetro_ellip
	cdef CircularAnnulus circ_annulus = CircularAnnulus(center, r_in, r_out)

	# Convert circular annulus aperture to binary mask
	cdef ApertureMask circ_annulus_mask_obj = circ_annulus.to_mask_mode(0)

	# With the same shape as the postage stamp
	cdef cnp.ndarray[double,ndim=2] circ_annulus_mask = circ_annulus_mask_obj.to_image((ny, nx), cnp.NPY_DOUBLE)
	# Invert mask and exclude other sources
	cdef cnp.ndarray[cnp.npy_bool,ndim=2] total_mask = mask_stamp | np.logical_not(circ_annulus_mask)

	# If sky area is too small (e.g., if annulus is outside the
	# image), use skybox instead.
	if np.sum(~total_mask) < constants.skybox_size ** 2:
		if constants.verbose:
			warnings.warn('[shape_asym] Using skybox for background.',
						  AstropyUserWarning)
		total_mask = np.ones((ny, nx), dtype=np.bool_)
		total_mask[slice_skybox] = False
		# However, if skybox is undefined, there is nothing to do.
		if np.sum(~total_mask) == 0:
			warnings.warn('[shape_asym] Asymmetry segmap undefined.',
						  AstropyUserWarning)
			flags.set_flag_true(0)
			return ~mask_stamp_no_bg

	# Define the "mode" as in Bertin & Arnouts (1996):
	cdef bkg_estimator = photutils.background.ModeEstimatorBackground(median_factor=2.5, mean_factor=1.5)

	cdef double mean, median, std
	# Do sigma-clipping until convergence
	mean, median, std = sigma_clipped_stats(
		cutout_stamp_maskzeroed, mask=total_mask, sigma=3.0,
		maxiters=None, cenfunc=bkg_estimator)

	# Mode as defined in Bertin & Arnouts (1996)
	cdef double mode = 2.5 * median - 1.5 * mean
	cdef double threshold = mode + std

	# Smooth image slightly and apply 1-sigma threshold
	cdef cnp.ndarray[double,ndim=2] image_smooth = ndi.uniform_filter(
		cutout_stamp_maskzeroed, size=constants.boxcar_size_shape_asym)
	cdef cnp.ndarray[cnp.npy_bool,ndim=2] above_threshold = image_smooth >= threshold

	cdef int ic, jc
	# Make sure that brightest pixel (of smoothed image) is in segmap
	ic, jc = np.argwhere(image_smooth == np.max(image_smooth))[0]
	if ~above_threshold[ic, jc]:
		warnings.warn('[shape_asym] Adding brightest pixel to segmap.',
					  AstropyUserWarning)
		above_threshold[ic, jc] = True
		flags.set_flag_true(1)

	cdef cnp.ndarray s, labled_array
	cdef int num_features
	# Grow regions with 8-connected neighbor "footprint"
	s = ndi.generate_binary_structure(2, 2)
	labeled_array, num_features = ndi.label(above_threshold, structure=s)

	return labeled_array == labeled_array[ic, jc]

cdef double get_rmax_circ(cnp.ndarray cutout_stamp_maskzeroed, (double, double) asymmetry_center, cnp.ndarray segmap_shape_asym, Flags flags):
	"""
	Return the distance (in pixels) from the pixel that minimizes
	the asymmetry to the edge of the main source segment, similar
	to Pawlik et al. (2016).
	"""
	cdef cnp.ndarray image = cutout_stamp_maskzeroed
	cdef int ny = image.shape[0]
	cdef int nx = image.shape[1]

	# Center at pixel that minimizes asymmetry
	cdef double xc = asymmetry_center[0]
	cdef double yc = asymmetry_center[1]

	# Distances from all pixels to the center
	ypos, xpos = np.mgrid[0:ny, 0:nx]
	cdef cnp.ndarray distances = np.sqrt((ypos - yc) ** 2 + (xpos - xc) ** 2)

	# Only consider pixels within the segmap.
	cdef double rmax_circ = np.max(distances[segmap_shape_asym])

	if rmax_circ == 0:
		warnings.warn('[rmax_circ] rmax_circ = 0!', AstropyUserWarning)
		flags.set_flag_true(2)

	return rmax_circ

cdef double get_rmax_ellip(cnp.ndarray cutout_stamp_maskzeroed, (double, double) asymmetry_center, double orientation_asymmetry, double elongation_asymmetry, cnp.ndarray segmap_shape_asym, Flags flags):
	"""
	Return the semimajor axis of the minimal ellipse (with fixed
	center, elongation and orientation) that contains all of
	the main segment of the shape asymmetry segmap. In most
	cases this is almost identical to rmax_circ.
	"""
	cdef cnp.ndarray image = cutout_stamp_maskzeroed
	cdef int ny = image.shape[0]
	cdef int nx = image.shape[1]

	# Center at pixel that minimizes asymmetry
	cdef double xc = asymmetry_center[0]
	cdef double yc = asymmetry_center[1]

	cdef double theta = orientation_asymmetry
	y, x = np.mgrid[0:ny, 0:nx]

	cdef cnp.ndarray xprime = (x - xc) * cos(theta) + (y - yc) * sin(theta)
	cdef cnp.ndarray yprime = -(x - xc) * sin(theta) + (y - yc) * cos(theta)
	cdef cnp.ndarray r_ellip = np.sqrt(xprime ** 2 + (yprime * elongation_asymmetry) ** 2)

	# Only consider pixels within the segmap.
	cdef double rmax_ellip = np.max(r_ellip[segmap_shape_asym])

	if rmax_ellip == 0:
		warnings.warn('[rmax_ellip] rmax_ellip = 0!', AstropyUserWarning)
		flags.set_flag_true(3)

	return rmax_ellip

cdef double get_rhalf_circ(cnp.ndarray cutout_stamp_maskzeroed, (double, double) asymmetry_center, double rmax_circ, Flags flags):
	"""
	The radius of a circular aperture containing 50% of the light,
	assuming that the center is the point that minimizes the
	asymmetry and that the total is at ``rmax_circ``.
	"""
	cdef cnp.ndarray image = cutout_stamp_maskzeroed
	cdef (double,double) center = asymmetry_center
	cdef double r
	cdef bint flag

	if rmax_circ == 0:
		r = 0.0
	else:
		r, flag = _radius_at_fraction_of_total_circ(
			image, center, rmax_circ, 0.5)
		if flag:
			flags.set_flag_true(4)

	# In theory, this return value can also be NaN
	return r


cdef double get_rhalf_ellip(cnp.ndarray cutout_stamp_maskzeroed, (double, double) asymmetry_center, double rmax_ellip, double elongation_asymmetry, double orientation_asymmetry, Flags flags):
	"""
	The semimajor axis of an elliptical aperture containing 50% of
	the light, assuming that the center is the point that minimizes
	the asymmetry and that the total is at ``rmax_ellip``.
	"""
	cdef cnp.ndarray image = cutout_stamp_maskzeroed
	cdef (double,double) center = asymmetry_center
	cdef double r
	cdef bint flag

	if rmax_ellip == 0:
		r = 0.0
	else:
		r, flag = _radius_at_fraction_of_total_ellip(
			image, center, elongation_asymmetry,
			orientation_asymmetry, rmax_ellip, 0.5)
		if flag:
			flags.set_flag_true(5)

	# In theory, this return value can also be NaN
	return r

cdef double get_shape_asymmetry(cnp.ndarray[cnp.npy_bool,ndim=2] segmap_shape_asym, (double, double) asymmetry_center, _mask_stamp, rmax_circ, Flags flags, ConstantsSetting constants):
	"""
	Calculate shape asymmetry as described in Pawlik et al. (2016).
	Note that the center is the one used for the standard asymmetry.
	"""
	cdef cnp.ndarray[double,ndim=2] image = cnp.PyArray_Where(segmap_shape_asym, 1.0, 0.0)
	cdef double asym = _shape_asymmetry_function(asymmetry_center, image, _mask_stamp, rmax_circ, flags, constants)

	return asym

cdef ShapeAsymmetryInfo calc_shape_asymmetry(BaseInfo base_info, CASInfo cas, GiniM20Info g_m20):
	cdef ShapeAsymmetryInfo shape_asym_info = ShapeAsymmetryInfo()

	cdef cnp.ndarray segmap = segmap_shape_asym(base_info._cutout_stamp_maskzeroed, cas._asymmetry_center, g_m20.rpetro_ellip, base_info._mask_stamp, cas._slice_skybox, base_info._mask_stamp_no_bg, shape_asym_info.flags, base_info.constants)

	cdef rmax_circ = get_rmax_circ(base_info._cutout_stamp_maskzeroed, cas._asymmetry_center, segmap, shape_asym_info.flags)
	cdef rmax_ellip = get_rmax_ellip(base_info._cutout_stamp_maskzeroed, cas._asymmetry_center, g_m20.orientation_asymmetry, g_m20.elongation_asymmetry, segmap, shape_asym_info.flags)
	shape_asym_info.rhalf_circ = get_rhalf_circ(base_info._cutout_stamp_maskzeroed, cas._asymmetry_center, rmax_circ, shape_asym_info.flags)
	shape_asym_info.rhalf_ellip = get_rmax_ellip(base_info._cutout_stamp_maskzeroed, cas._asymmetry_center, g_m20.orientation_asymmetry, g_m20.elongation_asymmetry, segmap, shape_asym_info.flags)
	shape_asym_info.shape_asymmetry = get_shape_asymmetry(segmap, cas._asymmetry_center, base_info._mask_stamp, rmax_circ, shape_asym_info.flags, base_info.constants)

	return shape_asym_info