# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
import warnings

cimport numpy as cnp
from astropy.modeling import models
from astropy.utils.exceptions import AstropyUserWarning
from libc.math cimport fabs, pow, sqrt, pi, floor

from .flags cimport Flags
from .photutils_simplified cimport EllipticalAnnulus, _aperture_mean_nomask
from .statmorph cimport BaseInfo, SersicInfo, CASInfo, GiniM20Info
from .constants_setting cimport ConstantsSetting

import numpy as np

cnp.import_array()

def _sersic_model(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, cnp.ndarray[double,ndim=2] weightmap_stamp, psf,
				  double orientation_asymmetry, double elongation_asymmetry, double rhalf_ellip, CASInfo cas, Flags flags, ConstantsSetting constants):
	"""
	Fit a 2D Sersic profile using Astropy's model fitting library.
	Return the fitted model object.
	"""
	cdef cnp.ndarray[double,ndim=2] image = _cutout_stamp_maskzeroed
	cdef int ny = image.shape[0]
	cdef int nx = image.shape[1]
	cdef (double,double) center = cas._asymmetry_center
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
	cdef cnp.ndarray[double,ndim=2] fit_weights = np.zeros_like(z)
	cdef cnp.ndarray[cnp.npy_bool,ndim=2] locs = (image != 0) & (weightmap != 0)
	# The sky background noise is already included in the weightmap:
	fit_weights[locs] = 1.0 / weightmap[locs]

	# Initial guess
	guess_n = 10.0 ** (-1.5) * cas.concentration ** 3.5  # empirical
	guess_n = min(max(guess_n, 1.0), 3.5)  # limit to range [1.0, 3.5]
	cdef double xc = cas._asymmetry_center[0]
	cdef double yc = cas._asymmetry_center[1]

	if psf is None:
		sersic_init = models.Sersic2D(
			amplitude=ellip_annulus_mean_flux, r_eff=rhalf_ellip,
			n=guess_n, x_0=xc, y_0=yc, ellip=self.ellipticity_asymmetry, theta=theta)
	else:
		sersic_init = ConvolvedSersic2D(
			amplitude=ellip_annulus_mean_flux, r_eff=rhalf_ellip,
			n=guess_n, x_0=xc, y_0=yc, ellip=self.ellipticity_asymmetry, theta=theta)
		sersic_init.set_psf(self._psf)

	# The number of data points cannot be smaller than the number of
	# free parameters (7 in the case of Sersic2D)
	if z.size < sersic_init.parameters.size:
		warnings.warn('[sersic] Not enough data for fit.',
					  AstropyUserWarning)
		self.flag_sersic = 1
		return sersic_init

	# Since model fitting can be computationally expensive (especially
	# with a large PSF), only do it when the other measurements are OK.
	if self.flag == 2:
		warnings.warn('[sersic] Skipping Sersic fit...',
					  AstropyUserWarning)
		self.flag_sersic = 1
		return sersic_init

	# Try to fit model
	fit_sersic = fitting.LevMarLSQFitter()
	sersic_model = fit_sersic(sersic_init, x, y, z, weights=fit_weights,
							  maxiter=self._sersic_maxiter, acc=1e-5)
	if fit_sersic.fit_info['ierr'] not in [1, 2, 3, 4]:
		warnings.warn("fit_info['message']: " + fit_sersic.fit_info['message'],
					  AstropyUserWarning)
		self.flag_sersic = 1

	return sersic_model

def sersic_amplitude(self):
	"""
	The amplitude of the 2D Sersic fit at the effective (half-light)
	radius (`astropy.modeling.models.Sersic2D`).
	"""
	return self._sersic_model.amplitude.value

def sersic_rhalf(self):
	"""
	The effective (half-light) radius of the 2D Sersic fit
	(`astropy.modeling.models.Sersic2D`).
	"""
	return self._sersic_model.r_eff.value

def sersic_n(self):
	"""
	The Sersic index ``n`` (`astropy.modeling.models.Sersic2D`).
	"""
	return self._sersic_model.n.value


def sersic_xc(self):
	"""
	The x-coordinate of the center of the 2D Sersic fit
	(`astropy.modeling.models.Sersic2D`), relative to the
	original image.
	"""
	return self.xmin_stamp + self._sersic_model.x_0.value


def sersic_yc(self):
	"""
	The y-coordinate of the center of the 2D Sersic fit
	(`astropy.modeling.models.Sersic2D`), relative to the
	original image.
	"""
	return self.ymin_stamp + self._sersic_model.y_0.value


def sersic_ellip(self):
	"""
	The ellipticity of the 2D Sersic fit
	(`astropy.modeling.models.Sersic2D`).
	"""
	return self._sersic_model.ellip.value


def sersic_theta(self):
	"""
	The orientation (counterclockwise, in radians) of the
	2D Sersic fit (`astropy.modeling.models.Sersic2D`).
	"""
	theta = self._sersic_model.theta.value
	return theta - np.floor(theta / np.pi) * np.pi