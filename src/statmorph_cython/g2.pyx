# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

cimport numpy as cnp
from numpy.math cimport NAN
from libc.math cimport isnan, pow, sqrt, pi, floor, fabs

from .statmorph cimport StampMorphology, G2Info

import numpy as np
from scipy import signal

cnp.import_array()

"""
https://ui.adsabs.harvard.edu/abs/2018MNRAS.477L.101R
"""

cdef class G2Calculator:
	# gradients
	def __cinit__(self, double[:,:] image, int contour_pixels_count, double module_tolerance, double phase_tolerance):
		self.image = image
		self.contour_pixels_count = contour_pixels_count
		self.phase_tolerance = phase_tolerance
		self.module_tolerance = module_tolerance
		self.height = len(image)
		self.width = len(image[0])
		self.center_x = <int>floor(self.height/2)
		self.center_y = <int>floor(self.width/2)
		self.valid_pixels_count = 0
		self.assimetric_pixel_count = 0


	# debug method to add noise to phases
	cdef double add_phase_noise(self):
		cdef int i, j
		for i in range(self.width):
			for j in range(self.height):
				self.phases[i,j] = self.phases[i,j] + self.phases_noise[i,j]


	# debug method to add noise to modules
	cdef double add_module_noise(self):
		cdef int i, j
		for i in range(self.width):
			for j in range(self.height):
				self.modules_normalized[i,j] = self.modules_normalized[i,j] + self.modules_noise[i,j]


	# aux methods
	# add pi to phases, this is done to maintain phases 0-360 degrees
	cdef double add_pi(self):
		cdef int i, j
		for i in range(self.width):
			for j in range(self.height):
				self.phases[i,j] = self.phases[i,j] + pi

	# modules normalization by max module
	cdef double normalize_modules(self):
		cdef int i, j
		self.modules_normalized = cnp.PyArray_ZEROS(2, [self.width, self.height], cnp.NPY_DOUBLE, 0)
		cdef double max_gradient = 0
		max_gradient = self.get_max_gradient()
		for i in range(self.width):
			for j in range(self.height):
				self.modules_normalized[i,j] = self.modules[i,j]/max_gradient

	# find maximum gradient
	cdef double get_max_gradient(self):
		cdef double max_gradient = -1.0
		cdef int i, j

		for i in range(self.width):
			for j in range(self.height):
				if not isnan(self.gradient_x[j,i]) and not isnan(self.gradient_y[j,i]):
					if (max_gradient<0.0) or (sqrt(pow(self.gradient_y[j, i], 2.0)+pow(self.gradient_x[j, i], 2.0)) > max_gradient):
						# modulo campo gradiente -> distancia euclidiana (maior modulo)
						max_gradient = sqrt(pow(self.gradient_y[j, i],2.0)+pow(self.gradient_x[j, i],2.0))

		return max_gradient

	# function to find angle difference
	cdef double angle_difference(self,double a1,double a2):
		# if it is in second quadrant - add pi
		if pi/2 <= a1 <= pi:
			return fabs(round((a1 - (a2 - pi)),4))
		# if it is in first quadrant - substract pi pi
		else:
			return fabs(round((a1 - (a2 + pi)),4))

	# function that converst 0 to nan in matrix
	cdef void convert_to_nan(self):
		cdef int i, j
		for i in range(self.height):
			for j in range(self.width):
				if self.image[i,j] == 0:
					self.image[i,j] = NAN


	# function that constructs asymmetric field
	cdef void get_asymmetryc_field(self):
		cdef int[:] x, y
		cdef double[:,:] distance_from_center
		cdef double[:] uniq_distance_from_center
		cdef int i, j, pixel_pairs_count, distance
		cdef (int,int) opposite_pixel

		self.gradient_asymmetric_x = self.gradient_x.copy()
		self.gradient_asymmetric_y = self.gradient_y.copy()

		# calculating phases
		self.phases = np.arctan2(self.gradient_x,self.gradient_y)
		# adding pi to maintaing everything in range 0 - 2pi (radians)
		self.add_pi()

		distance_from_center = np.array([[int(sqrt(pow(i-self.center_x, 2.0)+pow(j-self.center_y, 2.0))) for i in range(self.width)] for j in range(self.height)], dtype=np.float64)

		uniq_distance_from_center = np.unique(distance_from_center)

		# run for each distance from center
		for distance in range(len(uniq_distance_from_center)):
			# selects pixels with equal distances to see if they are symmetrical or not
			x, y = self.get_pixels_same_distance_from_center(distance_from_center, distance)
			pixel_pairs_count = len(x)


			# compare each point in the same distance
			# verifica se os pixeis sao simetricos ou nao
			for i in range(pixel_pairs_count):
				if isnan(self.image[x[i], y[i]]):
					continue

				if (self.modules_normalized[x[i], y[i]] <= self.module_tolerance):
					# if vector is too small, it is considered symmetric
					self.gradient_asymmetric_x[x[i], y[i]] = NAN
					self.gradient_asymmetric_y[x[i], y[i]] = NAN

				# getting opposite pixel
				opposite_pixel = self.get_opposite_pixel(x[i], y[i])

				if opposite_pixel[0] == -1:
					continue
				if isnan(self.image[opposite_pixel[0], opposite_pixel[1]]):
					continue

				if self.modules_normalized[opposite_pixel[0], opposite_pixel[1]] <= self.module_tolerance:
					# if vector is too small, it is considered symmetric
					self.gradient_asymmetric_x[opposite_pixel[0], opposite_pixel[1]] = NAN
					self.gradient_asymmetric_y[opposite_pixel[0], opposite_pixel[1]] = NAN

				if abs(self.modules_normalized[x[i], y[i]] - self.modules_normalized[opposite_pixel[0], opposite_pixel[1]]) <= self.module_tolerance:
					if (self.angle_difference(self.phases[x[i], y[i]], self.phases[opposite_pixel[0], opposite_pixel[1]])  <= self.phase_tolerance):

						self.gradient_asymmetric_x[x[i], y[i]] = NAN
						self.gradient_asymmetric_y[x[i], y[i]] = NAN
						self.gradient_asymmetric_x[opposite_pixel[0], opposite_pixel[1]] = NAN
						self.gradient_asymmetric_y[opposite_pixel[0], opposite_pixel[1]] = NAN

	# function that return (if such) opposite pixel
	cdef (int,int) get_opposite_pixel(self, int pixel_x, int pixel_y):
		#0 - x; 1 - y
		cdef int opposite_pixel_x=-1, opposite_pixel_y=-1
		cdef int distance_from_center_x, distance_from_center_y

		#check quadrand, ignore third and fourth as there will be located opposites
		if pixel_x > self.center_x and pixel_y < self.center_y:
			distance_from_center_x = self.center_x - pixel_x
			distance_from_center_y = self.center_y - pixel_y
			opposite_pixel_x = self.center_x + distance_from_center_x
			opposite_pixel_y = self.center_y + distance_from_center_y

			#print('first quadrant, opposite in third qudrant')
			#quadrand, opposite_quandrant = 1,4

		elif pixel_x < self.center_x and pixel_y < self.center_y:
			distance_from_center_x = self.center_x - pixel_x
			distance_from_center_y = self.center_y - pixel_y
			opposite_pixel_x = self.center_x + distance_from_center_x
			opposite_pixel_y = self.center_y + distance_from_center_y
		#	 #print('second quadrant, opposite in fourth qudrant')
		#	 #quadrand, opposite_quandrant = 2,3

		elif pixel_x < self.center_x and pixel_y > self.center_y:
			#print('third quadrant - ignore')
			pass

		elif pixel_x > self.center_x and pixel_y > self.center_y:
			#print('fourth quadrant - ignore')
			pass

		elif pixel_x == self.center_x and pixel_y > self.center_y:
			distance_from_center_x = self.center_x - pixel_x
			distance_from_center_y = self.center_y - pixel_y
			opposite_pixel_x = self.center_x + distance_from_center_x
			opposite_pixel_y = self.center_y + distance_from_center_y
			#print('on same x axis, but ontop of center')

		elif pixel_x == self.center_x and pixel_y < self.center_y:
			distance_from_center_x = self.center_x - pixel_x
			distance_from_center_y = self.center_y - pixel_y
			opposite_pixel_x = self.center_x + distance_from_center_x
			opposite_pixel_y = self.center_y + distance_from_center_y
			#print('on same x axis, but below of center')


		elif pixel_x > self.center_x and pixel_y == self.center_y:
			distance_from_center_x = self.center_x - pixel_x
			distance_from_center_y = self.center_y - pixel_y
			opposite_pixel_x = self.center_x + distance_from_center_x
			opposite_pixel_y = self.center_y + distance_from_center_y
			#print('on same y axis, but right of center')

		elif pixel_x < self.center_x and pixel_y == self.center_y:
			distance_from_center_x = self.center_x - pixel_x
			distance_from_center_y = self.center_y - pixel_y
			opposite_pixel_x = self.center_x + distance_from_center_x
			opposite_pixel_y = self.center_y + distance_from_center_y
			#print('on same y axis, but left of center')

		return opposite_pixel_x, opposite_pixel_y

	# getting list of pixels that have same distance from center
	cdef tuple get_pixels_same_distance_from_center(self, double[:,:] distance_from_center, int distance):
		cdef int px, py
		cdef int[:] x, y
		x2,y2 = [],[]

		for py in range(self.height):
			for px in range(self.width):
				# 0 means that take only pixels of specific distance from the center
				if (abs(distance_from_center[py, px] - distance) == 0):
					x2.append(px)
					y2.append(py)

		x, y = np.array(x2, dtype=np.int32), np.array(y2, dtype=np.int32)
		return x, y

	# calculating confluence
	cdef double get_confluence(self):
		cdef double sum_x_vectors = 0.0
		cdef double sum_y_vectors = 0.0
		cdef double sum_modules = 0.0
		cdef double aux_mod = 0.0
		cdef double confluence = 0.0

		for j in range(self.height):
			for i in range(self.width):
				if (not isnan(self.gradient_asymmetric_y[j,i])) and (not isnan(self.gradient_asymmetric_x[j,i])) and (not isnan(self.image[j,i])):
					aux_mod = self.modules[j,i]
					sum_x_vectors += self.gradient_asymmetric_x[j,i]
					sum_y_vectors += self.gradient_asymmetric_y[j,i]
					sum_modules += aux_mod

					self.assimetric_pixel_count = self.assimetric_pixel_count + 1
					self.valid_pixels_count = self.valid_pixels_count + 1
				elif not isnan(self.image[j,i]):
					self.valid_pixels_count = self.valid_pixels_count + 1

		# if valid pixels does not cover the whole image, substract the contour pixels
		if self.valid_pixels_count != self.height*self.width:
			self.valid_pixels_count = self.valid_pixels_count - self.contour_pixels_count

		# if there is no assimetric modules, confluence is 0
		if self.assimetric_pixel_count == 0:
			return 0.0

		confluence = sqrt(pow(sum_x_vectors, 2.0) + pow(sum_y_vectors, 2.0)) / sum_modules

		return confluence

	# entry point
	cdef tuple get_g2(self):
		cdef int i, j
		cdef double confluence, g2

		self.convert_to_nan()

		self.gradient_x, self.gradient_y = np.gradient(self.image)

		self.modules = np.array([[sqrt(pow(self.gradient_y[j, i],2.0)+pow(self.gradient_x[j, i],2.0)) for i in range(self.width) ] for j in range(self.height)], dtype=np.float64)
		self.normalize_modules()

		self.get_asymmetryc_field()

		confluence = self.get_confluence()
		if self.valid_pixels_count > 0:
			g2 = (<double> self.assimetric_pixel_count / <double> self.valid_pixels_count) * (2.0 - confluence)
		else:
			raise ValueError('Not enough valid pixels in image for g2 extraction')

		return g2, self.gradient_x, self.gradient_y, self.gradient_asymmetric_x, self.gradient_asymmetric_y, self.modules_normalized, self.phases


cdef cnp.ndarray[double,ndim=2] _recenter_image(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, (double, double) center):
	cdef cnp.ndarray[double,ndim=2] image = _cutout_stamp_maskzeroed
	#center = self._asymmetry_center
	cdef int ny = image.shape[0]
	cdef int nx = image.shape[1]
	cdef int xc = <int>center[0]
	cdef int yc = <int>center[1]

	cdef int x_left = xc
	cdef int x_right = nx - xc - 1
	cdef int y_up = yc
	cdef int y_low = ny - yc - 1
	cdef int x_range = max(x_left, x_right)
	cdef int y_range = max(y_low, y_up)
	cdef int range_use = max(x_range, y_range)
	cdef cnp.ndarray[double,ndim=2] image_temp = cnp.PyArray_ZEROS(2, [2 * range_use + 1, 2 * range_use + 1], cnp.NPY_DOUBLE, 0)
	image_temp[range_use - y_up:range_use + y_low + 1, range_use - x_left:range_use + x_right + 1] = image

	# image_use = image_temp

	return image_temp

cdef int _get_contour_count(cnp.ndarray[double,ndim=2] image):
	# function that counts the contour pixels
	filter = np.array([[0, 1, 0],
					   [1, 0, 1],
					   [0, 1, 0]])

	aux = (image.copy())
	aux[image != 0] = 1
	aux = aux.astype(int)
	conv = signal.convolve2d(aux, filter, mode='same')
	contourMask = aux * np.logical_and(conv > 0, conv < 4)

	return contourMask.sum()

cdef G2Info get_G2(StampMorphology base_info, (double,double) _asymmetry_center):
	cdef cnp.ndarray[double,ndim=2] image = _recenter_image(base_info._cutout_stamp_maskzeroed_no_bg, _asymmetry_center)
	# image = np.float32(image)

	g2 = G2Info(image, base_info.constants)
	g2.calc_g2()

	return g2