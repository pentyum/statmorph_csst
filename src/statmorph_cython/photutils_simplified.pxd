# cython: language_level=3

cimport numpy as cnp

cdef double do_photometry(PixelAperture ap, cnp.ndarray[double,ndim=2] image)

cdef double _aperture_area(PixelAperture ap, cnp.ndarray[cnp.npy_bool,ndim=2] mask)

cdef double _aperture_mean_nomask(PixelAperture ap, cnp.ndarray[double,ndim=2] image)

cdef (double, bint) _radius_at_fraction_of_total_circ(cnp.ndarray[double,ndim=2] image, (double,double) center, double r_total, double fraction)

cdef class BoundingBox:
	cdef int ixmin
	cdef int ixmax
	cdef int iymin
	cdef int iymax

	@staticmethod
	cdef BoundingBox from_float(double xmin, double xmax, double ymin, double ymax)

	cdef (double, double) center(self)

	cdef (int, int) shape(self)

	cdef tuple get_overlap_slices(self, (int, int) shape)

cdef class ApertureMask:
	cdef cnp.ndarray data
	cdef BoundingBox bbox
	cdef cnp.ndarray _mask

	cdef tuple get_overlap_slices(self, (int, int) shape)

	cdef cnp.ndarray[double, ndim=1] get_values(self, cnp.ndarray[double, ndim=2] data)
	# cdef cnp.ndarray[double, ndim=1] get_values(self, cnp.ndarray[double, ndim=2] data, cnp.ndarray mask)

cdef class Aperture:
	cdef (double, double) position
	cdef double theta

cdef class PixelAperture(Aperture):
	cdef BoundingBox bbox
	cdef (double, double, double, double) _centered_edges

	cdef void set_bbox_and_centered_edges(self)

	cdef (double, double) _xy_extents(self)

	cdef BoundingBox get_bbox(self)

	cdef (double, double, double, double) get_centered_edges(self)

	cdef double area(self)

	cdef ApertureMask to_mask(self)

	cdef (double, double) _do_photometry(self, cnp.ndarray[double, ndim=2] data, cnp.ndarray[double, ndim=2] variance)

	cdef (double, double) do_photometry(self, cnp.ndarray[double, ndim=2] data, cnp.ndarray[double, ndim=2] error)

cdef class CircularAperture(PixelAperture):
	cdef double r
	cdef (double, double) _xy_extents(self)

	cdef double area(self)

	cdef ApertureMask to_mask(self)

cdef class CircularAnnulus(PixelAperture):
	cdef double r_in
	cdef double r_out

	cdef (double, double) _xy_extents(self)

	cdef double area(self)

	cdef ApertureMask to_mask(self)

cdef class EllipticalMaskMixin:
	@staticmethod
	cdef (double, double) _calc_extents(double semimajor_axis, double semiminor_axis, double theta)

cdef class EllipticalAperture(PixelAperture):
	cdef double a
	cdef double b

	cdef (double, double) _xy_extents(self)

	cdef double area(self)

	cdef ApertureMask to_mask(self)

cdef class EllipticalAnnulus(PixelAperture):
	cdef double a_in
	cdef double a_out
	cdef double b_in
	cdef double b_out

	cdef (double, double) _xy_extents(self)

	cdef double area(self)

	cdef ApertureMask to_mask(self)