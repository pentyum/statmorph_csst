# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
import warnings

cimport numpy as cnp
import scipy
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
from libc.math cimport fabs, sqrt

from .flags cimport Flags
from .photutils_simplified cimport EllipticalAnnulus, _aperture_mean_nomask
from .statmorph cimport StampMorphology, SersicInfo, CASInfo, GiniM20Info, ShapeAsymmetryInfo
from .constants_setting cimport ConstantsSetting

import numpy as np

cnp.import_array()

cdef double _ellipticity_generic(double[:] eigvals):
	"""
	The ellipticity of (the Gaussian function that has the same
	second-order moments as) the source. Note that we allow
	negative eigenvalues.
	"""
	cdef double a = sqrt(fabs(eigvals[0]))
	cdef double b = sqrt(fabs(eigvals[1]))

	return 1.0 - (b / a)

class ConvolvedSersic2D(models.Sersic2D):
	"""
	Two-dimensional Sersic surface brightness profile, convolved with
	a PSF provided by the user as a numpy array.

	See Also
	--------
	astropy.modeling.models.Sersic2D

	"""
	psf = None

	@classmethod
	def set_psf(cls, psf):
		"""
		Specify the PSF to be convolved with the Sersic2D model.
		"""
		cls.psf = psf / np.sum(psf)  # make sure it's normalized

	@classmethod
	def evaluate(cls, x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta):
		"""
		Evaluate the ConvolvedSersic2D model.
		"""
		z_sersic = models.Sersic2D.evaluate(x, y, amplitude, r_eff, n, x_0, y_0,
											ellip, theta)
		if cls.psf is None:
			raise AssertionError('Must specify PSF using set_psf method.')

		# Apparently, scipy.signal also wants double:
		return scipy.signal.fftconvolve(
			np.float64(z_sersic), np.float64(cls.psf), mode='same')

cdef _sersic_model(cnp.ndarray[double,ndim=2] cutout_stamp_maskzeroed, cnp.ndarray[double,ndim=2] weightmap_stamp, cnp.ndarray psf,
				   double orientation_asymmetry, double elongation_asymmetry, double ellipticity_asymmetry, double rhalf_ellip, (double,double) asymmetry_center, double concentration,
				   Flags flags, ConstantsSetting constants):
	"""
	Fit a 2D Sersic profile using Astropy's model fitting library.
	Return the fitted model object.
	"""
	cdef cnp.ndarray[double,ndim=2] image = cutout_stamp_maskzeroed
	cdef int ny = image.shape[0]
	cdef int nx = image.shape[1]
	cdef (double,double) center = asymmetry_center
	cdef double theta = orientation_asymmetry

	# Get flux at the "effective radius"
	cdef double a_in = rhalf_ellip - 0.5 * constants.annulus_width
	cdef double a_out = rhalf_ellip + 0.5 * constants.annulus_width
	if a_in < 0:
		warnings.warn('[sersic] rhalf_ellip < annulus_width.',
					  AstropyUserWarning)
		flags.set_flag_true(0)
		a_in = rhalf_ellip
	cdef double b_out = a_out / elongation_asymmetry
	cdef EllipticalAnnulus ellip_annulus = EllipticalAnnulus(
		center, a_in, a_out, b_out, theta=theta)
	cdef double ellip_annulus_mean_flux = _aperture_mean_nomask(ellip_annulus, image)
	if ellip_annulus_mean_flux <= 0.0:
		warnings.warn('[sersic] Nonpositive flux at r_e.', AstropyUserWarning)
		flags.set_flag_true(1)
		ellip_annulus_mean_flux = fabs(ellip_annulus_mean_flux)

	# Prepare data for fitting
	cdef cnp.ndarray[double,ndim=2] z = image.copy()
	y, x = np.mgrid[0:ny, 0:nx]
	cdef cnp.ndarray weightmap = weightmap_stamp
	# Exclude pixels with image == 0 or weightmap == 0 from the fit.
	cdef cnp.ndarray[double,ndim=2] fit_weights = cnp.PyArray_ZEROS(2, z.shape, cnp.NPY_DOUBLE, 0)
	cdef cnp.ndarray[cnp.npy_bool,ndim=2] locs = (image != 0) & (weightmap != 0)
	# The sky background noise is already included in the weightmap:
	fit_weights[locs] = 1.0 / weightmap[locs]

	# Initial guess
	cdef float guess_n = 10.0 ** (-1.5) * concentration ** 3.5  # empirical
	guess_n = min(max(guess_n, 1.0), 3.5)  # limit to range [1.0, 3.5]
	cdef double xc = asymmetry_center[0]
	cdef double yc = asymmetry_center[1]

	if psf is None:
		sersic_init = models.Sersic2D(
			amplitude=ellip_annulus_mean_flux, r_eff=rhalf_ellip,
			n=guess_n, x_0=xc, y_0=yc, ellip=ellipticity_asymmetry, theta=theta)
	else:
		sersic_init = ConvolvedSersic2D(
			amplitude=ellip_annulus_mean_flux, r_eff=rhalf_ellip,
			n=guess_n, x_0=xc, y_0=yc, ellip=ellipticity_asymmetry, theta=theta)
		sersic_init.set_psf(psf)

	# The number of data points cannot be smaller than the number of
	# free parameters (7 in the case of Sersic2D)
	if z.size < sersic_init.parameters.size:
		warnings.warn('[sersic] Not enough data for fit.',
					  AstropyUserWarning)
		flags.set_flag_true(2)
		return sersic_init

	# Since model fitting can be computationally expensive (especially
	# with a large PSF), only do it when the other measurements are OK.
	#if skip:
	#	warnings.warn('[sersic] Skipping Sersic fit...',
	#				  AstropyUserWarning)
	#	flags.set_flag_true(1)
	#	return sersic_init

	# Try to fit model
	fit_sersic = fitting.LevMarLSQFitter()
	sersic_model = fit_sersic(sersic_init, x, y, z, weights=fit_weights,
							  maxiter=constants.sersic_maxiter, acc=1e-5)
	if fit_sersic.fit_info['ierr'] not in [1, 2, 3, 4]:
		warnings.warn("fit_info['message']: " + fit_sersic.fit_info['message'],
					  AstropyUserWarning)
		flags.set_flag_true(3)

	return sersic_model

cdef SersicInfo calc_sersic(StampMorphology base_info, CASInfo cas, GiniM20Info g_m20_info, ShapeAsymmetryInfo shape_asym_info):
	cdef SersicInfo sersic_info = SersicInfo()
	cdef double ellipticity_asymmetry = _ellipticity_generic(g_m20_info.eigvals_asymmetry)

	cdef model = _sersic_model(base_info._cutout_stamp_maskzeroed, base_info._weightmap_stamp, base_info.psf,
							   g_m20_info.orientation_asymmetry, g_m20_info.elongation_asymmetry, ellipticity_asymmetry,
							   shape_asym_info.rhalf_ellip, cas._asymmetry_center, cas.concentration,
							   sersic_info.flags, base_info.constants)
	sersic_info.sersic_amplitude = model.amplitude.value
	sersic_info.sersic_rhalf = model.r_eff.value
	sersic_info.sersic_n = model.n.value
	sersic_info.sersic_xc = model.x_0.value
	sersic_info.sersic_yc = model.y_0.value
	sersic_info.sersic_ellip = model.ellip.value
	return sersic_info