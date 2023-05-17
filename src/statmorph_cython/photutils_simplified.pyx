# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import warnings

from astropy.utils.exceptions import AstropyUserWarning
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, floor, ceil, pi, cos, sin, asin, fabs, isnan
from photutils.geometry import elliptical_overlap_grid
import scipy.optimize as opt
from numpy.math cimport NAN
from .array_utils cimport sum_1d_d

cnp.import_array()

cdef double do_photometry(PixelAperture ap, cnp.ndarray[double,ndim=2] image):
	cdef (double,double) photometry_result = ap.do_photometry(image, None)
	return photometry_result[0]

cdef double _aperture_area(PixelAperture ap, cnp.ndarray[cnp.npy_bool,ndim=2] mask):
	"""
	Calculate the area of a photutils aperture object,
	excluding masked pixels.
	"""
	return do_photometry(ap, cnp.PyArray_Cast(~mask, cnp.NPY_DOUBLE))

cdef double _aperture_mean_nomask(PixelAperture ap, cnp.ndarray[double,ndim=2] image):
	"""
	Calculate the mean flux of an image for a given photutils
	aperture object. Note that we do not use ``_aperture_area``
	here. Instead, we divide by the full area of the
	aperture, regardless of masked and out-of-range pixels.
	This avoids problems when the aperture is larger than the
	region of interest.
	"""
	cdef double area = ap.area()
	return do_photometry(ap, image) / area

cpdef double _fraction_of_total_function_circ(double r, cnp.ndarray[double,ndim=2] image, (double,double) center, double fraction, double total_sum):
	"""
	Helper function to calculate ``_radius_at_fraction_of_total_circ``.
	"""
	assert (r >= 0), "_fraction_of_total_function_circ: 需要r>=0，r=%f"%r
	assert (fraction >= 0) & (fraction <= 1), "_fraction_of_total_function_circ: 需要0<=fraction<=1，fraction=%f"%fraction
	assert (total_sum > 0), "_fraction_of_total_function_circ: 需要total_sum>0，而total_sum=%f"%total_sum

	cdef double cur_fraction, ap_sum
	cdef CircularAperture ap
	if r == 0:
		cur_fraction = 0.0
	else:
		ap = CircularAperture(center, r)
		# Force flux sum to be positive:
		ap_sum = fabs(do_photometry(ap, image))
		cur_fraction = ap_sum / total_sum

	return cur_fraction - fraction

cdef (double, bint) _radius_at_fraction_of_total_circ(cnp.ndarray[double,ndim=2] image, (double,double) center, double r_total, double fraction):
	"""
	Return the radius (in pixels) of a concentric circle that
	contains a given fraction of the light within ``r_total``.
	"""
	cdef bint flag = False  # flag=1 indicates a problem

	cdef CircularAperture ap_total = CircularAperture(center, r_total)

	cdef double total_sum = do_photometry(ap_total, image)
	assert not isnan(total_sum), "_radius_at_fraction_of_total_circ: total_sum为nan"
	assert total_sum != 0, "_radius_at_fraction_of_total_circ: total_sum=%f，为0" % total_sum
	if total_sum < 0:
		warnings.warn('[r_circ] Total flux sum is negative.', AstropyUserWarning)
		flag = True
		total_sum = fabs(total_sum)

	# Find appropriate range for root finder
	cdef int npoints = 100
	cdef double[:] r_grid = np.linspace(0.0, r_total, num=npoints)
	cdef int i = 0  # initial value
	cdef double r, r_min, r_max, curval

	while True:
		assert i < npoints, 'Root not found within range. i=%d>=%d'%(i,npoints)
		r = r_grid[i]
		curval = _fraction_of_total_function_circ(
			r, image, center, fraction, total_sum)
		if curval <= 0:
			r_min = r
		elif curval > 0:
			r_max = r
			break
		i += 1

	r = opt.brentq(_fraction_of_total_function_circ, r_min, r_max,
				   args=(image, center, fraction, total_sum), xtol=1e-6)

	return r, flag

cpdef double _fraction_of_total_function_ellip(double a, cnp.ndarray[double,ndim=2] image, (double,double) center, double elongation, double theta,
									  double fraction, double total_sum):
	"""
	Helper function to calculate ``_radius_at_fraction_of_total_ellip``.
	"""
	assert (a >= 0)
	assert (fraction >= 0) & (fraction <= 1)
	assert (total_sum > 0)

	cdef double cur_fraction, ap_sum, b
	cdef EllipticalAperture ap
	if a == 0:
		cur_fraction = 0.0
	else:
		b = a / elongation
		ap = EllipticalAperture(center, a, b, theta)
		# Force flux sum to be positive:
		ap_sum = fabs(do_photometry(ap,image))
		cur_fraction = ap_sum / total_sum

	return cur_fraction - fraction


cdef (double, bint) _radius_at_fraction_of_total_ellip(cnp.ndarray[double,ndim=2] image, (double,double) center, double elongation, double theta,
									   double a_total, double fraction):
	"""
	Return the semimajor axis (in pixels) of a concentric ellipse that
	contains a given fraction of the light within a larger ellipse of
	semimajor axis ``a_total``.
	"""
	cdef bint flag = False

	cdef double b_total = a_total / elongation
	cdef EllipticalAperture ap_total = EllipticalAperture(
		center, a_total, b_total, theta)

	cdef double total_sum = do_photometry(ap_total,image)
	assert not isnan(total_sum), "_radius_at_fraction_of_total_ellip: total_sum为nan"
	assert total_sum != 0, "_radius_at_fraction_of_total_ellip: total_sum=%f，为0" % total_sum
	if total_sum < 0:
		warnings.warn('[r_ellip] Total flux sum is negative.', AstropyUserWarning)
		flag = True
		total_sum = fabs(total_sum)

	# Find appropriate range for root finder
	cdef int npoints = 100
	cdef double[:] a_grid = np.linspace(0.0, a_total, num=npoints)
	cdef int i = 0  # initial value
	cdef double a, a_min, a_max, curval

	while True:
		assert i < npoints, 'Root not found within range.'
		a = a_grid[i]
		curval = _fraction_of_total_function_ellip(
			a, image, center, elongation, theta, fraction, total_sum)
		if curval <= 0:
			a_min = a
		elif curval > 0:
			a_max = a
			break
		i += 1

	a = opt.brentq(_fraction_of_total_function_ellip, a_min, a_max,
				   args=(image, center, elongation, theta, fraction, total_sum),
				   xtol=1e-6)

	return a, flag

cdef class BoundingBox:
	def __init__(self, int ixmin, int ixmax, int iymin, int iymax):
		if ixmin > ixmax:
			raise ValueError('ixmin must be <= ixmax')
		if iymin > iymax:
			raise ValueError('iymin must be <= iymax')

		self.ixmin = ixmin
		self.ixmax = ixmax
		self.iymin = iymin
		self.iymax = iymax

	@staticmethod
	cdef BoundingBox from_float(double xmin, double xmax, double ymin, double ymax):
		cdef int ixmin = int(floor(xmin + 0.5))
		cdef int ixmax = int(ceil(xmax + 0.5))
		cdef int iymin = int(floor(ymin + 0.5))
		cdef int iymax = int(ceil(ymax + 0.5))

		return BoundingBox(ixmin, ixmax, iymin, iymax)

	def __eq__(self, other):
		if not isinstance(other, BoundingBox):
			raise TypeError('Can compare BoundingBox only to another '
							'BoundingBox.')

		return ((self.ixmin == other.ixmin)
				and (self.ixmax == other.ixmax)
				and (self.iymin == other.iymin)
				and (self.iymax == other.iymax))

	cdef (double, double) center(self):
		"""
		The ``(y, x)`` center of the bounding box.
		"""
		return (0.5 * (self.iymax - 1 + self.iymin),
				0.5 * (self.ixmax - 1 + self.ixmin))

	cdef (int, int) shape(self):
		"""
		The ``(ny, nx)`` shape of the bounding box.
		"""
		return self.iymax - self.iymin, self.ixmax - self.ixmin

	cdef tuple get_overlap_slices(self, (int, int) shape):
		cdef int xmin = self.ixmin
		cdef int xmax = self.ixmax
		cdef int ymin = self.iymin
		cdef int ymax = self.iymax

		if xmin >= shape[1] or ymin >= shape[0] or xmax <= 0 or ymax <= 0:
			# no overlap of the bounding box with the input shape
			return None, None

		cdef tuple slices_large = (slice(max(ymin, 0), min(ymax, shape[0])),
								   slice(max(xmin, 0), min(xmax, shape[1])))
		cdef tuple slices_small = (slice(max(-ymin, 0),
										 min(ymax - ymin, shape[0] - ymin)),
								   slice(max(-xmin, 0),
										 min(xmax - xmin, shape[1] - xmin)))

		return slices_large, slices_small

cdef class ApertureMask:
	def __init__(self, cnp.ndarray[double, ndim=2] data, BoundingBox bbox):
		self.data = data
		cdef (int, int) bbox_shape = bbox.shape()
		if (self.data.shape[0] != bbox_shape[0]) or (self.data.shape[1] != bbox_shape[1]):
			raise ValueError('mask data and bounding box must have the same '
							 'shape')
		self.bbox = bbox
		self._mask = (self.data == 0)

	cdef tuple get_overlap_slices(self, (int, int) shape):
		return self.bbox.get_overlap_slices(shape)

	cdef cnp.ndarray[double, ndim=1] get_values(self, cnp.ndarray[double, ndim=2] data):
		cdef tuple slc_large, slc_small
		slc_large, slc_small = self.get_overlap_slices((data.shape[0], data.shape[1]))
		if slc_large is None:
			return np.array([])
		cdef cnp.ndarray cutout = data[slc_large]
		cdef cnp.ndarray apermask = self.data[slc_small]
		cdef cnp.ndarray pixel_mask = (apermask > 0)  # good pixels

		# ignore multiplication with non-finite data values
		return (cutout * apermask)[pixel_mask]

	cdef cnp.ndarray to_image(self, (int, int) shape, int dtype):
		# find the overlap of the mask on the output image shape
		slices_large, slices_small = self.get_overlap_slices(shape)

		if slices_small is None:
			return None  # no overlap

		# insert the mask into the output image
		cdef cnp.ndarray image = cnp.PyArray_ZEROS(2, [shape[0], shape[1]], dtype, 0)
		image[slices_large] = self.data[slices_small]
		return image

cdef class Aperture:
	def __init__(self):
		self.position = (0.0, 0.0)
		self.theta = 0.0

cdef class PixelAperture(Aperture):
	cdef void set_bbox_and_centered_edges(self):
		self.bbox = self.get_bbox()
		self._centered_edges = self.get_centered_edges()

	cdef (double, double) _xy_extents(self):
		raise NotImplementedError('Needs to be implemented in a subclass.')

	cdef BoundingBox get_bbox(self):
		cdef double x_delta, y_delta
		x_delta, y_delta = self._xy_extents()
		cdef double xmin = self.position[0] - x_delta
		cdef double xmax = self.position[0] + x_delta
		cdef double ymin = self.position[1] - y_delta
		cdef double ymax = self.position[1] + y_delta

		cdef BoundingBox bbox = BoundingBox.from_float(xmin, xmax, ymin, ymax)

		return bbox

	cdef (double, double, double, double) get_centered_edges(self):
		cdef BoundingBox bbox = self.bbox
		cdef double xmin = bbox.ixmin - 0.5 - self.position[0]
		cdef double xmax = bbox.ixmax - 0.5 - self.position[0]
		cdef double ymin = bbox.iymin - 0.5 - self.position[1]
		cdef double ymax = bbox.iymax - 0.5 - self.position[1]

		return xmin, xmax, ymin, ymax

	cdef double area(self):
		raise NotImplementedError('Needs to be implemented in a subclass.')

	cdef ApertureMask to_mask(self):
		raise NotImplementedError('Needs to be implemented in a subclass.')

	cdef ApertureMask to_mask_mode(self, int use_exact):
		raise NotImplementedError('Needs to be implemented in a subclass.')

	cdef (double, double) _do_photometry(self, cnp.ndarray[double, ndim=2] data, cnp.ndarray[double, ndim=2] variance):
		print("start _do_photometry")
		cdef double aperture_sums
		cdef double aperture_sum_errs
		cdef double aper_var

		print("start to_mask")
		cdef ApertureMask apermask = self.to_mask()
		print("start get_values")
		cdef cnp.ndarray[double,ndim=1] values = apermask.get_values(data)
		# if the aperture does not overlap the data return np.nan
		print("start sum_1d_d")
		aperture_sums = sum_1d_d(values) if len(values) != 0 else NAN

		if variance is not None:
			values = apermask.get_values(variance)
			# if the aperture does not overlap the data return np.nan
			aper_var = sum_1d_d(values) if len(values) != 0 else NAN
			aperture_sum_errs = sqrt(aper_var)

		return aperture_sums, aperture_sum_errs

	cdef (double, double) do_photometry(self, cnp.ndarray[double, ndim=2] data, cnp.ndarray[double, ndim=2] error):
		cdef cnp.ndarray variance
		if error is not None:
			variance = error ** 2
		else:
			variance = None

		return self._do_photometry(data, variance)

cdef class CircularAperture(PixelAperture):
	def __init__(self, (double, double) position, double r):
		super().__init__()
		self.position = position
		self.r = r
		self.set_bbox_and_centered_edges()

	cdef (double, double) _xy_extents(self):
		return self.r, self.r

	cdef double area(self):
		return pi * self.r ** 2

	cdef ApertureMask to_mask(self):
		return self.to_mask_mode(1)

	cdef ApertureMask to_mask_mode(self, int use_exact):
		cdef int subpixels = 1

		cdef double radius = self.r

		cdef cnp.ndarray mask
		cdef BoundingBox bbox = self.bbox
		cdef (double, double, double, double) edges = self._centered_edges

		cdef int ny, nx
		ny, nx = bbox.shape()
		mask = circular_overlap_grid(edges[0], edges[1], edges[2],
									 edges[3], nx, ny, radius, use_exact,
									 subpixels)

cdef class CircularAnnulus(PixelAperture):
	def __init__(self, (double, double) position, double r_in, double r_out):
		super().__init__()
		if not r_out > r_in:
			raise ValueError('r_out must be greater than r_in')

		self.position = position
		self.r_in = r_in
		self.r_out = r_out
		self.set_bbox_and_centered_edges()

	cdef (double, double) _xy_extents(self):
		return self.r_out, self.r_out

	cdef double area(self):
		return pi * (self.r_out ** 2 - self.r_in ** 2)

	cdef ApertureMask to_mask(self):
		return self.to_mask_mode(1)

	cdef ApertureMask to_mask_mode(self, int use_exact):
		cdef int subpixels = 1

		cdef double radius = self.r_out

		cdef cnp.ndarray mask
		cdef BoundingBox bbox = self.bbox
		cdef (double, double, double, double) edges = self._centered_edges

		cdef int ny, nx
		ny, nx = bbox.shape()
		mask = circular_overlap_grid(edges[0], edges[1], edges[2],
									 edges[3], nx, ny, radius, use_exact,
									 subpixels)

		# subtract the inner circle for an annulus
		mask -= circular_overlap_grid(edges[0], edges[1], edges[2],
									  edges[3], nx, ny, self.r_in,
									  use_exact, subpixels)

		return ApertureMask(mask, bbox)

cdef class EllipticalMaskMixin:
	@staticmethod
	cdef (double, double) _calc_extents(double semimajor_axis, double semiminor_axis, double theta):
		cos_theta = cos(theta)
		sin_theta = sin(theta)
		semimajor_x = semimajor_axis * cos_theta
		semimajor_y = semimajor_axis * sin_theta
		semiminor_x = semiminor_axis * -sin_theta
		semiminor_y = semiminor_axis * cos_theta
		x_extent = sqrt(semimajor_x ** 2 + semiminor_x ** 2)
		y_extent = sqrt(semimajor_y ** 2 + semiminor_y ** 2)

		return x_extent, y_extent

cdef class EllipticalAperture(PixelAperture):
	def __init__(self, (double, double) position, double a, double b, double theta=0.):
		super().__init__()
		self.position = position
		self.a = a
		self.b = b
		self.theta = theta
		self.set_bbox_and_centered_edges()

	cdef (double, double) _xy_extents(self):
		return EllipticalMaskMixin._calc_extents(self.a, self.b, self.theta)

	cdef double area(self):
		return pi * self.a * self.b

	cdef ApertureMask to_mask(self):
		return self.to_mask_mode(1)

	cdef ApertureMask to_mask_mode(self, int use_exact):
		cdef int subpixels = 1

		cdef double a = self.a
		cdef double b = self.b

		cdef cnp.ndarray mask
		cdef BoundingBox bbox = self.bbox
		cdef (double, double, double, double) edges = self._centered_edges

		cdef int ny, nx
		ny, nx = bbox.shape()
		mask = elliptical_overlap_grid(edges[0], edges[1], edges[2],
									   edges[3], nx, ny, a, b, self.theta,
									   use_exact, subpixels)

		return ApertureMask(mask, bbox)

cdef class EllipticalAnnulus(PixelAperture):
	def __init__(self, (double, double) position, double a_in, double a_out, double b_out, double b_in=-1,
				 double theta=0.):
		super().__init__()
		if not a_out > a_in:
			raise ValueError('"a_out" must be greater than "a_in".')

		self.position = position
		self.a_in = a_in
		self.a_out = a_out
		self.b_out = b_out

		if b_in == -1:
			b_in = self.b_out * self.a_in / self.a_out
		else:
			if not b_out > b_in:
				raise ValueError('"b_out" must be greater than "b_in".')
		self.b_in = b_in

		self.theta = theta
		self.set_bbox_and_centered_edges()

	cdef (double, double) _xy_extents(self):
		return EllipticalMaskMixin._calc_extents(self.a_out, self.b_out, self.theta)

	cdef double area(self):
		return pi * (self.a_out * self.b_out - self.a_in * self.b_in)

	cdef ApertureMask to_mask(self):
		return self.to_mask_mode(1)

	cdef ApertureMask to_mask_mode(self, int use_exact):
		cdef int subpixels = 1

		cdef double a = self.a_out
		cdef double b = self.b_out

		cdef cnp.ndarray mask
		cdef BoundingBox bbox = self.bbox
		cdef (double, double, double, double) edges = self._centered_edges

		cdef int ny, nx
		ny, nx = bbox.shape()
		mask = elliptical_overlap_grid(edges[0], edges[1], edges[2],
									   edges[3], nx, ny, a, b, self.theta,
									   use_exact, subpixels)

		# subtract the inner ellipse for an annulus
		mask -= elliptical_overlap_grid(edges[0], edges[1], edges[2],
										edges[3], nx, ny, self.a_in,
										self.b_in, self.theta,
										use_exact, subpixels)

		return ApertureMask(mask, bbox)

cdef cnp.ndarray[double, ndim=2] circular_overlap_grid(double xmin, double xmax, double ymin, double ymax,
													   int nx, int ny, double r, int use_exact,
													   int subpixels):
	cdef int i, j
	cdef double x, y, dx, dy, d, pixel_radius
	cdef double bxmin, bxmax, bymin, bymax
	cdef double pxmin, pxcen, pxmax, pymin, pycen, pymax

	# Define output array
	# cdef cnp.ndarray[double, ndim=2] frac = np.zeros([ny, nx], dtype=np.float64)
	cdef cnp.ndarray[double, ndim=2] frac = cnp.PyArray_ZEROS(2, [ny, nx], cnp.NPY_DOUBLE, 0)

	# Find the width of each element in x and y
	dx = (xmax - xmin) / nx
	dy = (ymax - ymin) / ny

	# Find the radius of a single pixel
	pixel_radius = 0.5 * sqrt(dx * dx + dy * dy)

	# Define bounding box
	bxmin = -r - 0.5 * dx
	bxmax = +r + 0.5 * dx
	bymin = -r - 0.5 * dy
	bymax = +r + 0.5 * dy

	for i in range(nx):
		pxmin = xmin + i * dx  # lower end of pixel
		pxcen = pxmin + dx * 0.5
		pxmax = pxmin + dx  # upper end of pixel
		if pxmax > bxmin and pxmin < bxmax:
			for j in range(ny):
				pymin = ymin + j * dy
				pycen = pymin + dy * 0.5
				pymax = pymin + dy
				if pymax > bymin and pymin < bymax:

					# Distance from circle center to pixel center.
					d = sqrt(pxcen * pxcen + pycen * pycen)

					# If pixel center is "well within" circle, count full
					# pixel.
					if d < r - pixel_radius:
						frac[j, i] = 1.

					# If pixel center is "close" to circle border, find
					# overlap.
					elif d < r + pixel_radius:

						# Either do exact calculation or use subpixel
						# sampling:
						if use_exact:
							frac[j, i] = circular_overlap_single_exact(
								pxmin, pymin, pxmax, pymax, r) / (dx * dy)
						else:
							frac[j, i] = circular_overlap_single_subpixel(
								pxmin, pymin, pxmax, pymax, r, subpixels)

		# Otherwise, it is fully outside circle.
		# No action needed.

	return frac

cdef double circular_overlap_single_subpixel(double x0, double y0,
											 double x1, double y1,
											 double r, int subpixels):
	cdef int i, j
	cdef double x, y, dx, dy, r_squared
	cdef double frac = 0.  # Accumulator.

	dx = (x1 - x0) / subpixels
	dy = (y1 - y0) / subpixels
	r_squared = r ** 2

	x = x0 - 0.5 * dx
	for i in range(subpixels):
		x += dx
		y = y0 - 0.5 * dy
		for j in range(subpixels):
			y += dy
			if x * x + y * y < r_squared:
				frac += 1.

	return frac / (subpixels * subpixels)

cdef double circular_overlap_single_exact(double xmin, double ymin,
										  double xmax, double ymax,
										  double r):
	if 0. <= xmin:
		if 0. <= ymin:
			return circular_overlap_core(xmin, ymin, xmax, ymax, r)
		elif 0. >= ymax:
			return circular_overlap_core(-ymax, xmin, -ymin, xmax, r)
		else:
			return circular_overlap_single_exact(xmin, ymin, xmax, 0., r) \
				   + circular_overlap_single_exact(xmin, 0., xmax, ymax, r)
	elif 0. >= xmax:
		if 0. <= ymin:
			return circular_overlap_core(-xmax, ymin, -xmin, ymax, r)
		elif 0. >= ymax:
			return circular_overlap_core(-xmax, -ymax, -xmin, -ymin, r)
		else:
			return circular_overlap_single_exact(xmin, ymin, xmax, 0., r) \
				   + circular_overlap_single_exact(xmin, 0., xmax, ymax, r)
	else:
		if 0. <= ymin:
			return circular_overlap_single_exact(xmin, ymin, 0., ymax, r) \
				   + circular_overlap_single_exact(0., ymin, xmax, ymax, r)
		if 0. >= ymax:
			return circular_overlap_single_exact(xmin, ymin, 0., ymax, r) \
				   + circular_overlap_single_exact(0., ymin, xmax, ymax, r)
		else:
			return circular_overlap_single_exact(xmin, ymin, 0., 0., r) \
				   + circular_overlap_single_exact(0., ymin, xmax, 0., r) \
				   + circular_overlap_single_exact(xmin, 0., 0., ymax, r) \
				   + circular_overlap_single_exact(0., 0., xmax, ymax, r)

cdef double circular_overlap_core(double xmin, double ymin, double xmax, double ymax,
								  double r):
	cdef double area, d1, d2, x1, x2, y1, y2

	if xmin * xmin + ymin * ymin > r * r:
		area = 0.
	elif xmax * xmax + ymax * ymax < r * r:
		area = (xmax - xmin) * (ymax - ymin)
	else:
		area = 0.
		d1 = floor_sqrt(xmax * xmax + ymin * ymin)
		d2 = floor_sqrt(xmin * xmin + ymax * ymax)
		if d1 < r and d2 < r:
			x1, y1 = floor_sqrt(r * r - ymax * ymax), ymax
			x2, y2 = xmax, floor_sqrt(r * r - xmax * xmax)
			area = ((xmax - xmin) * (ymax - ymin) -
					area_triangle(x1, y1, x2, y2, xmax, ymax) +
					area_arc(x1, y1, x2, y2, r))
		elif d1 < r:
			x1, y1 = xmin, floor_sqrt(r * r - xmin * xmin)
			x2, y2 = xmax, floor_sqrt(r * r - xmax * xmax)
			area = (area_arc(x1, y1, x2, y2, r) +
					area_triangle(x1, y1, x1, ymin, xmax, ymin) +
					area_triangle(x1, y1, x2, ymin, x2, y2))
		elif d2 < r:
			x1, y1 = floor_sqrt(r * r - ymin * ymin), ymin
			x2, y2 = floor_sqrt(r * r - ymax * ymax), ymax
			area = (area_arc(x1, y1, x2, y2, r) +
					area_triangle(x1, y1, xmin, y1, xmin, ymax) +
					area_triangle(x1, y1, xmin, y2, x2, y2))
		else:
			x1, y1 = floor_sqrt(r * r - ymin * ymin), ymin
			x2, y2 = xmin, floor_sqrt(r * r - xmin * xmin)
			area = (area_arc(x1, y1, x2, y2, r) +
					area_triangle(x1, y1, x2, y2, xmin, ymin))

	return area

cdef double floor_sqrt(double x):
	if x > 0:
		return sqrt(x)
	else:
		return 0

cdef double distance(double x1, double y1, double x2, double y2):
	return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

cdef double area_arc(double x1, double y1, double x2, double y2, double r):
	cdef double a, theta
	a = distance(x1, y1, x2, y2)
	theta = 2. * asin(0.5 * a / r)
	return 0.5 * r * r * (theta - sin(theta))

cdef double area_triangle(double x1, double y1, double x2, double y2, double x3,
						  double y3):
	return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
