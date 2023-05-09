# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
import warnings

cimport numpy as cnp
import photutils
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyUserWarning
from libc.math cimport cos, sin
import numpy as np
import scipy.ndimage as ndi

from .photutils_simplified import _radius_at_fraction_of_total_circ, _radius_at_fraction_of_total_ellip, CircularAnnulus, ApertureMask
from .constants_setting cimport ConstantsSetting
from .flags import Flags

cnp.import_array()

cdef cnp.ndarray[cnp.npy_bool,ndim=2] segmap_shape_asym(cnp.ndarray[double,ndim=2] cutout_stamp_maskzeroed, (double,double) asymmetry_center, double rpetro_ellip,
					   cnp.ndarray mask_stamp, slice_skybox, cnp.ndarray[cnp.npy_bool,ndim=2] mask_stamp_no_bg, Flags flags, ConstantsSetting constants):

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

cdef double rmax_circ(cnp.ndarray cutout_stamp_maskzeroed, (double, double) asymmetry_center, cnp.ndarray segmap_shape_asym, Flags flags):
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

cdef double rmax_ellip(cnp.ndarray cutout_stamp_maskzeroed, (double, double) asymmetry_center, double orientation_asymmetry, double elongation_asymmetry, cnp.ndarray segmap_shape_asym, Flags flags):
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

cdef double rhalf_circ(cnp.ndarray cutout_stamp_maskzeroed, (double, double) asymmetry_center, double rmax_circ, Flags flags):
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


cdef double rhalf_ellip(cnp.ndarray cutout_stamp_maskzeroed, (double, double) asymmetry_center, double rmax_ellip, double elongation_asymmetry, double orientation_asymmetry, Flags flags):
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