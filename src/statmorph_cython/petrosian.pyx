# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import warnings
from astropy.utils.exceptions import AstropyUserWarning
import scipy.optimize as opt

cimport numpy as cnp
from libc.math cimport fabs
from .flags cimport Flags
from .photutils_simplified cimport CircularAnnulus, CircularAperture, EllipticalAnnulus, EllipticalAperture,  _aperture_mean_nomask
from .constants_setting cimport ConstantsSetting

cpdef double _petrosian_function_circ(double r, (double, double) center, cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, Flags flags,
									  ConstantsSetting constants):
	"""
	Helper function to calculate the circular Petrosian radius.
	可能产生警告2

	For a given radius ``r``, return the ratio of the mean flux
	over a circular annulus divided by the mean flux within the
	circle, minus "eta" (eq. 4 from Lotz et al. 2004). The root
	of this function is the Petrosian radius.
	"""
	cdef cnp.ndarray[double,ndim=2] image = _cutout_stamp_maskzeroed
	cdef double r_in = r - 0.5 * constants.annulus_width
	cdef double r_out = r + 0.5 * constants.annulus_width

	cdef CircularAnnulus circ_annulus = CircularAnnulus(center, r_in, r_out)
	cdef CircularAperture circ_aperture = CircularAperture(center, r)

	# Force mean fluxes to be positive:
	cdef double circ_annulus_mean_flux = fabs(_aperture_mean_nomask(
		circ_annulus, image))
	cdef double circ_aperture_mean_flux = fabs(_aperture_mean_nomask(
		circ_aperture, image))
	cdef double ratio

	if circ_aperture_mean_flux == 0:
		warnings.warn('[rpetro_circ] Mean flux is zero.', AstropyUserWarning)
		# If flux within annulus is also zero (e.g. beyond the image
		# boundaries), return zero. Otherwise return 1.0:
		ratio = float(circ_annulus_mean_flux != 0)
		flags.set_flag_true(2)
	else:
		ratio = circ_annulus_mean_flux / circ_aperture_mean_flux

	return ratio - constants.eta

cdef double _rpetro_circ_generic(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, (double, double) center,
								 double _diagonal_distance, Flags flags, ConstantsSetting constants):
	"""
	Compute the Petrosian radius for concentric circular apertures.
	可能产生警告2,3,4,5

	Notes
	-----
	The so-called "curve of growth" is not always monotonic,
	e.g., when there is a bright, unlabeled and unmasked
	secondary source in the image, so we cannot just apply a
	root-finding algorithm over the full interval.
	Instead, we proceed in two stages: first we do a coarse,
	brute-force search for an appropriate interval (that
	contains a root), and then we apply the root-finder.

	"""
	# Find appropriate range for root finder
	cdef int npoints = 100
	cdef double r_inner = constants.annulus_width
	cdef double r_outer = _diagonal_distance
	assert r_inner < r_outer
	cdef double dr = (r_outer - r_inner) / float(npoints - 1)
	cdef double r_min, r_mix
	r_min, r_max = -1, -1
	cdef double r = r_inner  # initial value
	cdef double curval

	while True:
		if r >= r_outer:
			warnings.warn('[rpetro_circ] rpetro larger than cutout.',
						  AstropyUserWarning)
			flags.set_flag_true(3)
		curval = _petrosian_function_circ(r, center, _cutout_stamp_maskzeroed, flags, constants)
		if curval >= 0:
			r_min = r
		elif curval < 0:
			if r_min == -1:
				warnings.warn('[rpetro_circ] r_min is not defined yet.',
							  AstropyUserWarning)
				flags.set_flag_true(4)
				if r >= r_outer:
					# If r_min is still undefined at this point, then
					# rpetro must be smaller than the annulus width.
					warnings.warn('rpetro_circ < annulus_width! ' +
								  'Setting rpetro_circ = annulus_width.',
								  AstropyUserWarning)
					flags.set_flag_true(5) # unusual
					return r_inner
			else:
				r_max = r
				break
		r += dr

	cdef double rpetro_circ = opt.brentq(_petrosian_function_circ, r_min, r_max,
										 args=(center, _cutout_stamp_maskzeroed, flags, constants), xtol=1e-6)

	return rpetro_circ

cpdef double _petrosian_function_ellip(double a, (double,double) center, cnp.ndarray[double,ndim=2] cutout_stamp_maskzeroed,
									   double elongation, double theta, Flags flags, ConstantsSetting constants):
	"""
	Helper function to calculate the Petrosian "radius".

	For the ellipse with semi-major axis ``a``, return the
	ratio of the mean flux over an elliptical annulus
	divided by the mean flux within the ellipse,
	minus "eta" (eq. 4 from Lotz et al. 2004). The root of
	this function is the Petrosian "radius".
	可能产生警告9
	"""
	cdef cnp.ndarray image = cutout_stamp_maskzeroed

	cdef double b = a / elongation
	cdef double a_in = a - 0.5 * constants.annulus_width
	cdef double a_out = a + 0.5 * constants.annulus_width

	cdef double b_out = a_out / elongation

	cdef EllipticalAnnulus ellip_annulus = EllipticalAnnulus(
		center, a_in, a_out, b_out, theta=theta)
	cdef EllipticalAperture ellip_aperture = EllipticalAperture(
		center, a, b, theta=theta)

	# Force mean fluxes to be positive:
	cdef double ellip_annulus_mean_flux = fabs(_aperture_mean_nomask(
		ellip_annulus, image))
	cdef double ellip_aperture_mean_flux = fabs(_aperture_mean_nomask(
		ellip_aperture, image))
	cdef double ratio

	if ellip_aperture_mean_flux == 0:
		warnings.warn('[rpetro_ellip] Mean flux is zero.', AstropyUserWarning)
		# If flux within annulus is also zero (e.g. beyond the image
		# boundaries), return zero. Otherwise return 1.0:
		ratio = float(ellip_annulus_mean_flux != 0)
		flags.set_flag_true(9)
	else:
		ratio = ellip_annulus_mean_flux / ellip_aperture_mean_flux

	return ratio - constants.eta

cdef double _rpetro_ellip_generic(cnp.ndarray[double,ndim=2] cutout_stamp_maskzeroed, (double,double) center, double elongation, double theta,
								  double _diagonal_distance, Flags flags, ConstantsSetting constants):
	"""
	Compute the Petrosian "radius" (actually the semi-major axis)
	for concentric elliptical apertures.

	Notes
	-----
	The so-called "curve of growth" is not always monotonic,
	e.g., when there is a bright, unlabeled and unmasked
	secondary source in the image, so we cannot just apply a
	root-finding algorithm over the full interval.
	Instead, we proceed in two stages: first we do a coarse,
	brute-force search for an appropriate interval (that
	contains a root), and then we apply the root-finder.
	可能产生警告9,10,11,12
	"""
	# Find appropriate range for root finder
	cdef int npoints = 100
	cdef double a_inner = constants.annulus_width
	cdef double a_outer = _diagonal_distance
	assert a_inner < a_outer
	cdef double da = (a_outer - a_inner) / float(npoints - 1)
	cdef double a_min, a_max
	a_min, a_max = -1, -1
	cdef double a = a_inner  # initial value
	cdef double curval
	# cdef int i = 0
	while True:
		# print("i=%d, a=%.2f"%(i,a))
		if a >= a_outer:
			warnings.warn('[rpetro_ellip] rpetro larger than cutout.',
						  AstropyUserWarning)
			flags.set_flag_true(10)
		curval = _petrosian_function_ellip(a, center, cutout_stamp_maskzeroed, elongation, theta, flags, constants)
		# print("a_min=%.2f, a_max=%.2f, curval=%.3f"%(a_min, a_max, curval))
		if curval >= 0:
			a_min = a
		elif curval < 0:
			if a_min == -1:
				warnings.warn('[rpetro_ellip] a_min is not defined yet.',
							  AstropyUserWarning)
				flags.set_flag_true(11)
				if a >= a_outer:
					# If a_min is still undefined at this point, then
					# rpetro must be smaller than the annulus width.
					warnings.warn('rpetro_ellip < annulus_width! ' +
								  'Setting rpetro_ellip = annulus_width.',
								  AstropyUserWarning)
					flags.set_flag_true(12) # unusual
					return a_inner
			else:
				a_max = a
				break
		a += da
		# i = i + 1

	rpetro_ellip = opt.brentq(_petrosian_function_ellip, a_min, a_max,
							  args=(center, cutout_stamp_maskzeroed, elongation, theta, flags, constants), xtol=1e-6)

	return rpetro_ellip