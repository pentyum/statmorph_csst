# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
import warnings

from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np
from astropy.convolution import convolve
import skimage.measure
import matplotlib.pyplot as plt

from libc.math cimport sqrt, round
from time import time
cimport numpy as cnp

from .petrosian cimport _rpetro_circ_generic
from .flags cimport Flags
cimport statmorph_cython.cas
cimport statmorph_cython.g_m20
cimport statmorph_cython.multiplicity
cimport statmorph_cython.mid
cimport statmorph_cython.shape_asymmetry
cimport statmorph_cython.color_dispersion
from .constants_setting cimport ConstantsSetting, CutoutConstants
from .g2 cimport G2Calculator, _get_contour_count, get_G2

cnp.import_array()

cdef class MorphInfo:
	def __init__(self):
		# These flags will be modified during the calculations:
		self.flags = Flags()  # attempts to flag bad measurements
		"""
		bitmask用于标记错误
		"""
		self.runtime = 0

	@staticmethod
	cdef double get_duration_sec(double end, double start):
		return <double> (end - start)

	cdef void calc_runtime(self, double start):
		self.runtime = MorphInfo.get_duration_sec(time(), start)

	def get_values(self):
		pass

	@staticmethod
	def get_value_names():
		pass

	@staticmethod
	def get_value_formats():
		pass


cdef class StampMorphology(MorphInfo):
	def __init__(self,  int label, cnp.ndarray[double,ndim=2] cutout_stamp,
				 cnp.ndarray[int,ndim=2] segmap_stamp=None,
				 cnp.ndarray[cnp.npy_bool,ndim=2] mask_stamp_old=None,
				 cnp.ndarray[double,ndim=2] weightmap_stamp_old=None,
				 double gain=-1,  cnp.ndarray[double,ndim=2] image_compare_stamp=None,
				 str output_image_dir="None", tuple set_centroid=(-1, -1)):
		super().__init__()
		self.logger = None
		self.constants = ConstantsSetting()
		self.constants.label = label

		self.output_image_dir = output_image_dir
		"""
		图像输出文件夹，None表示不输出
		"""

		self.label = label
		"""
		该星系在在segmap中的label
		"""

		self._gain = gain
		"""
		增益
		"""

		# Measure runtime
		self.global_start = time()

		# if not isinstance(self._segmap, photutils.SegmentationImage):
		#	self._segmap = photutils.SegmentationImage(self._segmap)

		# Check sanity of input data
		# self._segmap.check_labels([self.label])
		# assert (self._segmap.shape[0] == self._image.shape[0]) and (self._segmap.shape[1] == self._image.shape[1])

		self.flag_catastrophic = False  # this one is reserved for really bad cases
		"""
		严重错误
		"""

		# If something goes wrong, use centroid instead of asymmetry center
		# (better performance in some pathological cases, e.g. GOODS-S 32143):
		self._use_centroid = False
		"""
		是否用光度分布平均值而不是不对称中心，如果不计算cas则用光度分布平均值
		"""

		self._cutout_stamp = cutout_stamp
		"""
		原始图像在该星系处的切片
		"""

		self._segmap_stamp = segmap_stamp
		"""
		segmap在该星系处的切片
		"""

		self._mask_stamp_old = mask_stamp_old
		"""
		mask在该星系处的切片，原始值
		"""

		self._weightmap_stamp_old = weightmap_stamp_old
		"""
		weightmap在该星系处的切片，原始值
		"""

		self.image_compare_stamp = self.image_compare_stamp

		print("stamp设置完成")

		self._check_stamp_size()


		self.nx_stamp = self.get_nx_stamp()
		"""
		图像切片的宽度
		"""

		self.ny_stamp = self.get_ny_stamp()
		"""
		图像切片的高度
		"""

		if self.nx_stamp * self.ny_stamp > 10000000:
			warnings.warn('%d: Cutout size too big (%d*%d>10M), skip.' % (self.label, self.nx_stamp, self.ny_stamp),
						  AstropyUserWarning)
			self._abort_calculations()
			return

		self._mask_stamp_nan = self.get_mask_stamp_nan()
		"""
		图像切片中有哪些点是nan或者inf
		"""

		self._weightmap_stamp = self.get_weightmap_stamp()
		"""
		weightmap在该星系处的切片
		"""

		self.num_badpixels = -1
		"""
		图像切片中的坏点数量，也就是不满足abs(原始图像而切片-周围平均后的图像切片)<=n_sigma_outlier*std像素的数量
		"""

		self._mask_stamp_badpixels = self.get_mask_stamp_badpixels()
		"""
		图像切片中坏点存在位置
		"""

		self._mask_stamp = self.get_mask_stamp()
		"""
		图像切片的总mask，该mask是segmap中非label的源、输入的mask、nan或者inf位置和坏点位置的并集
		"""

		self._mask_stamp_no_bg = self.get_mask_stamp_no_bg()
		"""
		图像切片的包括背景的mask，即总mask和segmap中label=0位置(即背景)的并集
		"""

		self._cutout_stamp_maskzeroed = self.get_cutout_stamp_maskzeroed()
		"""
		获得星系本体+背景的图像切片，但是被mask的部分被置为0，该mask指的是mask_stamp中的总mask，包括其它label的源、输入的mask、nan、inf和坏点
		"""

		self._cutout_stamp_maskzeroed_no_bg = self.get_cutout_stamp_maskzeroed_no_bg()
		"""
		获得星系本体图像切片，但是被mask的部分被置为0，该mask指的是mask_stamp_no_bg中的包括背景的总mask。
		"""

		# Check that the labeled galaxy segment has a positive flux sum.
		# If not, this is bad enough to abort all calculations and return
		# an empty object.
		if self.check_total_flux_nonpositive():
			return

		cdef cnp.ndarray[double, ndim=1] cutout_not_in_mask = self._cutout_stamp[~self._mask_stamp_no_bg]

		self.size = len(cutout_not_in_mask)
		"""
		用于计算的星系本体的全部像素的数量
		"""

		self.surface_brightness = np.mean(cutout_not_in_mask)
		"""
		用于计算的星系本体的大小每个像素流量的平均值
		"""

		if set_centroid == (-1, -1):
			self._centroid = self.get_centroid()
		else:
			self._centroid = set_centroid
		"""
		星系本体图像切片的一阶矩，即光度分布的质心，依次为x和y，坐标是相对于切片的
		"""

		# Centroid of the source relative to the "postage stamp" cutout:
		self._xc_stamp = self._centroid[0]
		"""
		星系光度质心的x坐标，相对于切片的
		"""

		self._yc_stamp = self._centroid[1]
		"""
		星系光度质心的x坐标，相对于切片的
		"""

		self._diagonal_distance = self.get_diagonal_distance()
		"""
		图像切片的对角线长度
		"""

		self._rpetro_circ_centroid = self.get_rpetro_circ_centroid()
		"""
		以光度质心为中心的Petrosian圆形孔径半径
		"""

	cdef bint check_total_flux_nonpositive(self):
		if np.sum(self._cutout_stamp_maskzeroed_no_bg) <= 0:
			warnings.warn('%d: Total flux is nonpositive. Returning empty object.' % self.label,
						  AstropyUserWarning)
			self._abort_calculations()
			return True
		return False

	cpdef void calculate_morphology(self, bint calc_cas, bint calc_g_m20, bint calc_shape_asymmetry, bint calc_mid,
									bint calc_multiplicity,
									bint calc_color_dispersion, bint calc_g2, (double, double) set_asym_center):
		cdef (double, double) center_used
		cdef long start

		self.calc_cas = calc_cas
		self.calc_g_m20 = calc_g_m20
		self.calc_mid = calc_mid
		self.calc_multiplicity = calc_multiplicity
		self.calc_color_dispersion = calc_color_dispersion
		self.calc_g2 = calc_g2

		if calc_cas:
			start = time()
			self.cas = statmorph_cython.cas.calc_cas(self, set_asym_center)
			self.cas.calc_runtime(start)
			center_used = self.cas._asymmetry_center
		else:
			self._use_centroid = True
			center_used = self._centroid

		if calc_g_m20:
			start = time()
			self.g_m20 = statmorph_cython.g_m20.calc_g_m20(self, center_used)
			self.g_m20.calc_runtime(start)

		if calc_shape_asymmetry:
			if self.cas is not None and self.g_m20 is not None:
				start = time()
				self.shape_asymmetry = statmorph_cython.shape_asymmetry.calc_shape_asymmetry(self, self.cas, self.g_m20)
				self.shape_asymmetry.calc_runtime(start)

		if calc_mid:
			start = time()
			self.mid = statmorph_cython.mid.calc_mid(self)
			self.mid.calc_runtime(start)
			if calc_multiplicity:
				self.multiplicity = statmorph_cython.multiplicity.multiplicity(self.mid._cutout_mid)

		if calc_color_dispersion:
			if self.image_compare_stamp is not None:
				start = time()
				self.compare_info = statmorph_cython.color_dispersion.calc_color_dispersion(self,
																							self.image_compare_stamp)
				self.compare_info.calc_runtime(start)
			else:
				warnings.warn("%d: [Color dispersion] compare image not defined" % self.label)

		if calc_g2:
			start = time()
			self.g2 = get_G2(self, center_used)
			self.g2.calc_runtime(start)

		self.sn_per_pixel = self.get_sn_per_pixel()

		self._check_segmaps()

		# Save image
		if self.output_image_dir is not None:
			self.save_image()

		self.calc_runtime(self.global_start)

	cdef void calc_morphology_uncertainties(self, int times):
		cdef int i
		cdef cnp.ndarray[double,ndim=2] noise, new_stamp
		for i in range(times):
			noise = self.generate_noise_stamp()
			new_stamp = self._cutout_stamp + noise
		return

	cdef void _check_segmaps(self):
		"""
		Compare Gini segmap and MID segmap; set flag = 1 (suspect) if they are
		very different from each other.
		"""
		cdef double area_max, area_overlap, area_ratio

		if self.calc_g_m20 and self.calc_mid:
			area_max = max(np.sum(self.g_m20._segmap_gini),
						   np.sum(self.mid._segmap_mid))
			area_overlap = np.sum(self.g_m20._segmap_gini &
								  self.mid._segmap_mid)

			area_ratio = area_overlap / float(area_max)
			if area_ratio < self.constants.segmap_overlap_ratio:
				if self.constants.verbose:
					warnings.warn('%d: Gini and MID segmaps are quite different.' % self.label,
								  AstropyUserWarning)
				self.flags.set_flag_true(13)  # suspect

		elif self.calc_g_m20:
			area_max = np.sum(self.g_m20._segmap_gini)
		elif self.calc_mid:
			area_max = np.sum(self.mid._segmap_mid)
		else:
			return
		if area_max == 0:
			warnings.warn('%d: Segmaps are empty!' % self.label, AstropyUserWarning)
			self.flags.set_flag_true(14)  # bad
			return

	cdef void _check_stamp_size(self):
		assert cnp.PyArray_SAMESHAPE(self._segmap_stamp, self._cutout_stamp)
		if self._mask_stamp_old is not None:
			assert cnp.PyArray_SAMESHAPE(self._mask_stamp_old, self._cutout_stamp)
			assert cnp.PyArray_ISBOOL(self._mask_stamp_old)
		if self._weightmap_stamp_old is not None:
			assert cnp.PyArray_SAMESHAPE(self._weightmap_stamp_old, self._cutout_stamp)

	cdef int get_nx_stamp(self):
		"""
		Number of pixels in the 'postage stamp' along the ``x`` direction.
		"""
		return self._cutout_stamp.shape[1]

	cdef int get_ny_stamp(self):
		"""
		Number of pixels in the 'postage stamp' along the ``y`` direction.
		"""
		return self._cutout_stamp.shape[0]

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_nan(self):
			"""
			Flag any NaN or inf values within the postage stamp.
			"""
			cdef cnp.ndarray locs_invalid = ~np.isfinite(self._cutout_stamp)
			if self._weightmap_stamp_old is not None:
				locs_invalid |= ~np.isfinite(self._weightmap_stamp_old)
			return locs_invalid

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] _get_badpixels(self, cnp.ndarray[double, ndim=2] image):
		"""
		Detect outliers (bad pixels) as described in Lotz et al. (2004).

		Notes
		-----
		ndi.generic_filter(image, np.std, ...) is too slow,
		so we do a workaround using ndi.convolve.
		"""
		# Pixel weights, excluding central pixel.
		cdef cnp.ndarray[double, ndim=2] w = np.array([
			[1, 1, 1],
			[1, 0, 1],
			[1, 1, 1]], dtype=np.float64)
		w = w / np.sum(w)

		# Use the fact that var(x) = <x^2> - <x>^2.
		cdef cnp.ndarray[double, ndim=2] local_mean = convolve(image, w, boundary='extend',
															   normalize_kernel=False)
		cdef cnp.ndarray[double, ndim=2] local_mean2 = convolve(image ** 2, w, boundary='extend',
																normalize_kernel=False)
		cdef cnp.ndarray[double, ndim=2] local_std = np.sqrt(local_mean2 - local_mean ** 2)

		# Get "bad pixels"
		cdef cnp.ndarray[cnp.npy_bool, ndim=2] badpixels = (np.abs(image - local_mean) >
															self.constants.n_sigma_outlier * local_std)

		return badpixels

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_badpixels(self):
		"""
		Flag badpixels (outliers).
		"""
		# self.num_badpixels = -1
		# cdef cnp.ndarray badpixels = np.zeros((self.ny_stamp, self.nx_stamp), dtype=np.bool8)
		cdef cnp.ndarray[cnp.npy_bool, ndim=2] badpixels = cnp.PyArray_ZEROS(2, [self.ny_stamp, self.nx_stamp],
																			 cnp.NPY_BOOL, 0)
		if self.constants.n_sigma_outlier > 0:
			badpixels = self._get_badpixels(self._cutout_stamp)
			self.num_badpixels = np.sum(badpixels)
		return badpixels

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp(self):
		"""
		Create a total binary mask for the "postage stamp".
		Pixels belonging to other sources (as well as pixels masked
		using the ``mask`` keyword argument) are set to ``True``,
		but the background (segmap == 0) is left alone.
		"""
		# cdef cnp.ndarray segmap_stamp = self._segmap[self._slice_stamp]
		cdef cnp.ndarray mask_stamp = (self._segmap_stamp != 0) & (self._segmap_stamp != self.label)
		if self._mask_stamp_old is not None:
			mask_stamp |= self._mask_stamp_old
		mask_stamp |= self._mask_stamp_nan
		mask_stamp |= self._mask_stamp_badpixels
		return mask_stamp

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_no_bg(self):
		"""
		Similar to ``_mask_stamp``, but also mask the background.
		"""
		# cdef cnp.ndarray segmap_stamp = self._segmap[self._slice_stamp]
		return self._mask_stamp | (self._segmap_stamp == 0)

	cdef cnp.ndarray[double, ndim=2] get_cutout_stamp_maskzeroed(self):
		"""
		Return a data cutout with its shape and position determined
		by ``_slice_stamp``. Pixels belonging to other sources
		(as well as pixels where ``mask`` == 1) are set to zero,
		but the background is left alone.

		In addition, NaN or inf values are removed at this point,
		as well as badpixels (outliers).
		"""
		return cnp.PyArray_Where(~self._mask_stamp,
								 self._cutout_stamp, 0.0)

	cdef cnp.ndarray[double, ndim=2] get_cutout_stamp_maskzeroed_no_bg(self):
		"""
		Like ``_cutout_stamp_maskzeroed``, but also mask the
		background.
		"""
		return cnp.PyArray_Where(~self._mask_stamp_no_bg,
								 self._cutout_stamp, 0.0)

	cdef (double,double) get_centroid(self):
		"""
		The (yc, xc) centroid of the input segment, relative to
		``_slice_stamp``.
		可能产生Base警告0,1
		"""
		cdef int ny, nx, ic, jc
		cdef cnp.ndarray image = self._cutout_stamp_maskzeroed_no_bg
		# cdef cnp.ndarray image = np.float64(self._cutout_stamp_maskzeroed_no_bg)  # skimage wants double

		# Calculate centroid
		cdef double[:,:] M = skimage.measure.moments(image, order=1)
		assert M[0, 0] > 0  # already checked by constructor
		cdef double yc = M[1, 0] / M[0, 0]
		cdef double xc = M[0, 1] / M[0, 0]

		ny = self.ny_stamp
		nx = self.nx_stamp
		if (yc < 0) or (yc >= ny) or (xc < 0) or (xc >= nx):
			warnings.warn('%d: Centroid is out-of-range. Fixing at center of postage stamp (bad!).'%self.label, AstropyUserWarning)
			yc = ny / 2.0
			xc = nx / 2.0
			self.flags.set_flag_true(0) # unusual

		# Print warning if centroid is masked:
		ic = <int>(round(yc))
		jc = <int>(round(xc))
		if self._cutout_stamp_maskzeroed[ic][jc] == 0.0:
			warnings.warn('%d: Centroid (%d,%d) is masked.'%(self.label, jc, ic), AstropyUserWarning)
			self.flags.set_flag_true(1)

		#return np.array([xc, yc])
		return xc, yc


	cdef double get_diagonal_distance(self):
		"""
		Return the diagonal distance (in pixels) of the postage stamp.
		This is used as an upper bound in some calculations.
		"""
		return sqrt(self.nx_stamp ** 2 + self.ny_stamp ** 2)

	cdef double get_rpetro_circ_centroid(self):
		"""
		Calculate the Petrosian radius with respect to the centroid.
		This is only used as a preliminary value for the asymmetry
		calculation.
		可能产生Base警告7,8,9,10
		"""
		return _rpetro_circ_generic(self._cutout_stamp_maskzeroed, self._centroid, self._diagonal_distance, self.flags, self.constants)

	cdef cnp.ndarray get_weightmap_stamp(self):
		"""
		Return a cutout of the weight map over the "postage stamp" region.
		If a weightmap is not provided as input, it is created using the
		``gain`` argument.
		"""
		if self._weightmap_stamp_old is None:
			# Already checked that gain is not None:
			"""
			assert self._gain > 0
			weightmap_stamp = np.sqrt(
				np.abs(self._image[self._slice_stamp])/self._gain + self.sky_sigma**2)
			"""
			return None
		else:
			weightmap_stamp = self._weightmap_stamp_old.copy()

		weightmap_stamp[self._mask_stamp_nan] = 0.0
		return weightmap_stamp

	cdef double get_sn_per_pixel(self):
		"""
		Calculate the signal-to-noise per pixel using the Petrosian segmap.
		可能产生Base警告2
		"""
		cdef cnp.ndarray noisemap = self._weightmap_stamp
		if noisemap is None:
			return -99.0

		if np.any(noisemap < 0):
			warnings.warn('%d: [sn_per_pixel] Some negative weightmap values.'%self.label,
						  AstropyUserWarning)
			noisemap = np.abs(noisemap)

		cdef cnp.ndarray locs = ((self._cutout_stamp_maskzeroed >= 0) & (noisemap > 0))

		if self.calc_g_m20:
			locs = locs & self.g_m20._segmap_gini
		else:
			locs = locs & (self._segmap_stamp==self.label)

		cdef double snp
		if np.sum(locs) == 0:
			warnings.warn('%d: Invalid sn_per_pixel.'%self.label, AstropyUserWarning)
			self.flags.set_flag_true(2)
			snp = -99.0  # invalid
		else:
			pixelvals = self._cutout_stamp_maskzeroed[locs]
			# The sky background noise is already included in the weightmap:
			snp = np.mean(pixelvals / noisemap[locs])

		return snp

	cdef cnp.ndarray[double,ndim=2] generate_noise_stamp(self):
		return np.random.normal(0,self._weightmap_stamp_old)

	cdef void _abort_calculations(self):
		"""
		Some cases are so bad that basically nothing can be measured
		(e.g. a bunch of pixels with a nonpositive total sum). We
		deal with these cases by creating an "empty" object and
		interrupting the constructor.
		"""
		warnings.warn("%d: 强制停止计算，返回空的对象"%self.label, AstropyUserWarning)
		self.nx_stamp = -99
		self.ny_stamp = -99
		self.flag_catastrophic = True
		# 图像切片总和不是正数，或者图片过大，直接终止全部计算
		self.runtime = -99.0

		self.cas = CASInfo()
		self.g_m20 = GiniM20Info()
		self.mid = MIDInfo()
		self.multiplicity = -99
		self.compare_info = CompareInfo()
		self.g2 = G2Info(cnp.PyArray_ZEROS(2, [1,1], cnp.NPY_DOUBLE, 0), self.constants)

	cdef (int, int, int, int) get_image_extent(self):
		return (0, self.nx_stamp, 0, self.ny_stamp)

	cdef void save_image(self):
		"""
		其中绿色轮廓为原始segmentation map(_mask_stamp_no_bg)，黄色轮廓为计算G和M20的segmentation map(_segmap_gini)
		绿点为光度中心，白点为不对称中心
		黑色圆圈为以不对称中心为中心的1.5rp，白色圆圈为用于计算C的r20，灰色圆圈为r80
		黑色方框为背景区域
		"""
		cdef cnp.ndarray sm_all = cnp.PyArray_Where(self._segmap_stamp == self.label, 2,
													cnp.PyArray_Where(self._segmap_stamp == 0, 0, 1))

		cdef int xmin_stamp, xmax_stamp, ymin_stamp, ymax_stamp
		cdef (int, int, int, int) extent = self.get_image_extent()

		xmin_stamp = extent[0]
		xmax_stamp = extent[1]
		ymin_stamp = extent[2]
		ymax_stamp = extent[3]

		cdef int rec_x, rec_y, rec_x_length, rec_y_length
		if self.nx_stamp < 1000 and self.ny_stamp < 1000:
			plt.figure(figsize=(10, 5))
		elif self.nx_stamp < 2500 and self.ny_stamp < 2500:
			plt.figure(figsize=(20, 10))
		else:
			plt.figure(figsize=(40, 20))

		plt.subplot(1, 2, 1)
		plt.imshow(sm_all, origin="lower")

		plt.subplot(1, 2, 2)
		plt.imshow(self._cutout_stamp, cmap="gray", origin="lower", extent=extent)
		cdef double vmax = np.percentile(self._cutout_stamp[~self._mask_stamp_no_bg], 90)
		plt.clim(0, vmax)
		plt.xlim(xmin_stamp, xmax_stamp)
		plt.ylim(ymin_stamp, ymax_stamp)

		if self.cas is not None:
			if self.cas._slice_skybox is not None:
				rec_x = xmin_stamp + self.cas._slice_skybox[1].start
				rec_y = ymin_stamp + self.cas._slice_skybox[0].start
				rec_x_length = self.cas._slice_skybox[1].stop - self.cas._slice_skybox[1].start
				rec_y_length = self.cas._slice_skybox[0].stop - self.cas._slice_skybox[0].start

				rec_sky = plt.Rectangle((rec_x, rec_y), rec_x_length, rec_y_length, edgecolor="white", linewidth=2,
										fill=False)

			asym_center = (self.cas._asymmetry_center[0]+xmin_stamp, self.cas._asymmetry_center[1]+ymin_stamp)

			circ_15rp = plt.Circle(asym_center, self.constants.petro_extent_cas * self.cas.rpetro_circ,
								   edgecolor="white",
								   linewidth=2,
								   fill=False)
			circ_r80 = plt.Circle(asym_center, self.cas.r80, edgecolor="gray", linewidth=1, fill=False)
			circ_r20 = plt.Circle(asym_center, self.cas.r20, edgecolor="black", linewidth=1, fill=False)

			plt.scatter(*asym_center, s=10, color="red", label="asym c (%.1f,%.1f)" % tuple(self.cas._asymmetry_center))

			plt.gca().add_patch(rec_sky)
			plt.gca().add_patch(circ_15rp)
			plt.gca().add_patch(circ_r80)
			plt.gca().add_patch(circ_r20)

		cdef double xc_centroid = self._xc_stamp + xmin_stamp
		cdef double yc_centroid = self._yc_stamp + ymin_stamp

		plt.scatter(xc_centroid, yc_centroid, s=10, color="olive",
					label="centroid (%.1f,%.1f)" % tuple(self._centroid))
		plt.contour(self._mask_stamp_no_bg, colors="green", extent=extent, levels=[0.5])

		if self.g_m20 is not None:
			plt.contour(self.g_m20._segmap_gini, colors="yellow", extent=extent, levels=[0.5])

		plt.tight_layout()
		plt.legend()

		plt.savefig("%s/%d.png" % (self.output_image_dir, self.label))
		plt.close()

	cdef void dump_stamps(self):
		cdef stamps = [
			self._cutout_stamp,
			self._cutout_stamp_maskzeroed,
			self._cutout_stamp_maskzeroed_no_bg,
			self._weightmap_stamp_old,
			self._weightmap_stamp,
			self._segmap_stamp,
			self._mask_stamp_old,
			self._mask_stamp_badpixels,
			self._mask_stamp,
		]
		plt.figure(figsize=(12, 12))
		cdef int i = 0
		for i in range(9):
			plt.subplot(3, 3, i + 1)
			plt.imshow(stamps[i], origin="lower")
			plt.colorbar()

		plt.tight_layout()
		plt.savefig("./dump_stamps_%d.pdf" % self.label)
		plt.close()


cdef class BigImageMorphology(StampMorphology):
	def __init__(self, cnp.ndarray[double,ndim=2] image, cnp.ndarray[int,ndim=2] segmap, tuple segmap_slice,
				 int label, cnp.ndarray[cnp.npy_bool,ndim=2] mask=None, cnp.ndarray[double,ndim=2] weightmap=None,
				 double gain=-1, cnp.ndarray[double,ndim=2] image_compare=None,
				 str output_image_dir=None, str save_stamp_dir=None, tuple set_centroid=(-1, -1)):
		self.cutout_constants = CutoutConstants()
		self.save_stamp_dir = save_stamp_dir
		"""
		stamp保存文件夹，None表示不保存
		"""

		self._image = image
		"""
		输入的原始图像引用
		"""

		self._segmap = segmap
		"""
		输入的segmentation map应用
		"""

		self._segmap_slice = segmap_slice
		"""
		选区位置(由photutils生成)
		"""

		self._mask = mask
		"""
		需要遮蔽的区域
		"""

		self._weightmap = weightmap
		"""
		权重图
		"""

		self.image_compare = image_compare

		# if not isinstance(self._segmap, photutils.SegmentationImage):
		#	self._segmap = photutils.SegmentationImage(self._segmap)

		# Check sanity of input data
		# self._segmap.check_labels([self.label])
		# assert (self._segmap.shape[0] == self._image.shape[0]) and (self._segmap.shape[1] == self._image.shape[1])
		assert cnp.PyArray_SAMESHAPE(self._segmap, self._image)
		if self._mask is not None:
			assert cnp.PyArray_SAMESHAPE(self._mask, self._image)
			assert cnp.PyArray_ISBOOL(self._mask)
		if self._weightmap is not None:
			assert cnp.PyArray_SAMESHAPE(self._weightmap, self._image)

		self._slice_stamp = self.get_slice_stamp()
		"""
		星系区域切片下标，依次为y和x，相对整幅图像
		"""

		cdef cnp.ndarray[double, ndim=2] cutout_stamp, weightmap_stamp_old, image_compare_stamp
		cdef cnp.ndarray[int, ndim=2] segmap_stamp
		cdef cnp.ndarray[cnp.npy_bool, ndim=2] mask_stamp_old

		cutout_stamp = self._image[self._slice_stamp]
		"""
		原始图像在该星系处的切片
		"""

		segmap_stamp = self._segmap[self._slice_stamp]
		"""
		segmap在该星系处的切片
		"""

		if self._weightmap is not None:
			weightmap_stamp_old = self._weightmap[self._slice_stamp]
		else:
			weightmap_stamp_old = None
		"""
		weightmap在该星系处的切片，原始值
		"""

		if self._mask is not None:
			mask_stamp_old = self._mask[self._slice_stamp]
		else:
			mask_stamp_old = None
		"""
		mask在该星系处的切片，原始值
		"""

		if self.image_compare is not None:
			image_compare_stamp = self.image_compare[self._slice_stamp]
		else:
			image_compare_stamp = None

		self.xmin_stamp = self.get_xmin_stamp()
		"""
		图像切片的x起点在整幅图像中的下标
		"""

		self.ymin_stamp = self.get_ymin_stamp()
		"""
		图像切片的y起点在整幅图像中的下标
		"""

		self.xmax_stamp = self.get_xmax_stamp()
		"""
		图像切片的x终点在整幅图像中的下标
		"""

		self.ymax_stamp = self.get_ymax_stamp()
		"""
		图像切片的y终点在整幅图像中的下标
		"""

		StampMorphology.__init__(self, label, cutout_stamp, segmap_stamp, mask_stamp_old, weightmap_stamp_old, gain, image_compare_stamp, output_image_dir, set_centroid)

		self.xc_centroid = self.get_xc_centroid()
		"""
		星系光度质心的x坐标，相对于整个图像的
		"""

		self.yc_centroid = self.get_yc_centroid()
		"""
		星系光度质心的y坐标，相对于整个图像的
		"""


	cdef tuple get_slice_stamp(self):
		"""
		Attempt to create a square slice that is somewhat larger
		than the minimal bounding box containing the labeled segment.
		Note that the cutout may not be square when the source is
		close to a border of the original image.
		"""
		assert self.cutout_constants.cutout_extent >= 1.0
		assert self.cutout_constants.min_cutout_size >= 2
		# Get dimensions of original bounding box
		cdef tuple s = self._segmap_slice
		cdef slice y_slice = s[0]
		cdef slice x_slice = s[1]
		cdef int xmin = x_slice.start
		cdef int xmax = x_slice.stop - 1
		cdef int ymin = y_slice.start
		cdef int ymax = y_slice.stop - 1
		cdef int dx = xmax + 1 - xmin
		cdef int dy = ymax + 1 - ymin

		cdef int xc = xmin + dx // 2
		cdef int yc = ymin + dy // 2
		# Add some extra space in each dimension
		cdef int dist = int(max(dx, dy) * self.cutout_constants.cutout_extent / 2.0)
		# Make sure that cutout size is at least ``min_cutout_size``
		dist = max(dist, self.cutout_constants.min_cutout_size // 2)
		# Make cutout
		cdef int ny = self._image.shape[0]
		cdef int nx = self._image.shape[1]
		# ny, nx = self._image.shape
		cdef tuple slice_stamp = (slice(max(0, yc - dist), min(ny, yc + dist)),
								  slice(max(0, xc - dist), min(nx, xc + dist)))
		return slice_stamp

	cdef int get_xmin_stamp(self):
		"""
		The minimum ``x`` position of the 'postage stamp'.
		"""
		return self._slice_stamp[1].start

	cdef int get_ymin_stamp(self):
		"""
		The minimum ``y`` position of the 'postage stamp'.
		"""
		return self._slice_stamp[0].start

	cdef int get_xmax_stamp(self):
		"""
		The maximum ``x`` position of the 'postage stamp'.
		"""
		return self._slice_stamp[1].stop - 1

	cdef int get_ymax_stamp(self):
		"""
		The maximum ``y`` position of the 'postage stamp'.
		"""
		return self._slice_stamp[0].stop - 1

	cdef int get_nx_stamp(self):
		"""
		Number of pixels in the 'postage stamp' along the ``x`` direction.
		"""
		return self.xmax_stamp + 1 - self.xmin_stamp

	cdef int get_ny_stamp(self):
		"""
		Number of pixels in the 'postage stamp' along the ``y`` direction.
		"""
		return self.ymax_stamp + 1 - self.ymin_stamp

	cdef double get_xc_centroid(self):
		"""
		The x-coordinate of the centroid, relative to the original image.
		"""
		return self._centroid[0] + self.xmin_stamp

	cdef double get_yc_centroid(self):
		"""
		The y-coordinate of the centroid, relative to the original image.
		"""
		return self._centroid[1] + self.ymin_stamp

	cpdef void calculate_morphology(self, bint calc_cas, bint calc_g_m20, bint calc_shape_asymmetry, bint calc_mid,
									bint calc_multiplicity,
									bint calc_color_dispersion, bint calc_g2, (double, double) set_asym_center):
		super().calculate_morphology(calc_cas, calc_g_m20, calc_shape_asymmetry, calc_mid, calc_multiplicity,calc_color_dispersion, calc_g2,set_asym_center)
		if self.save_stamp_dir is not None:
			self.save_stamp()

	cdef void save_stamp(self):
		hdu1 = fits.PrimaryHDU(self._cutout_stamp)
		hdu2 = fits.ImageHDU(self._weightmap_stamp_old)
		hdu2.name = "NOISE"
		hdulist = fits.HDUList([hdu1, hdu2])
		hdulist.writeto("%s/%d.fits" % (self.save_stamp_dir, self.label), overwrite=True)
		hdulist.close()

	cdef (int, int, int, int) get_image_extent(self):
		cdef int stamp_x = self._slice_stamp[1].start
		cdef int stamp_y = self._slice_stamp[0].start
		return stamp_x, self._slice_stamp[1].stop, stamp_y, self._slice_stamp[0].stop


cdef class FileStampMorphology(StampMorphology):
	def __init__(self,  int label, str fits_file_name, int fits_hdu_index=0,
				 str segmap_file_name="None", int segmap_hdu_index=0,
				 str mask_file_name="None", int mask_hdu_index=0,
				 str weightmap="None", int weightmap_hdu_index=0,
				 double gain=-1, str image_compare_file_name="None", int image_compare_hdu_index=0,
				 str output_image_dir="None", tuple set_centroid=(-1, -1)):
		self._image_fits = fits.open(fits_file_name)
		"""
		输入的原始图像引用
		"""

		if mask_file_name != "None":
			self._mask_fits = fits.open(mask_file_name)
		else:
			self._mask_fits = None
		"""
		输入的segmentation map引用
		"""

		if segmap_file_name != "None":
			self._segmap_fits = fits.open(segmap_file_name)
		else:
			self._segmap_fits = None
		"""
		输入的segmentation map引用
		"""

		if weightmap != "None":
			self._weightmap_fits = fits.open(weightmap)
		else:
			self._weightmap_fits = None
		"""
		权重图
		"""

		if image_compare_file_name != "None":
			self._image_compare_fits = fits.open(image_compare_file_name)
		else:
			self._image_compare_fits = None

		cdef cnp.ndarray[double,ndim=2] cutout_stamp, weightmap_stamp_old, image_compare_stamp
		cdef cnp.ndarray[int,ndim=2] segmap_stamp
		cdef cnp.ndarray[cnp.npy_bool,ndim=2] mask_stamp_old

		cutout_stamp = cnp.PyArray_Cast(self._image_fits[fits_hdu_index].data, cnp.NPY_DOUBLE)
		"""
		原始图像在该星系处的切片
		"""

		if self._segmap_fits is None:
			segmap_stamp = np.full_like(cutout_stamp, label, dtype=np.int32)
		else:
			segmap_stamp =  cnp.PyArray_Cast(self._segmap_fits[segmap_hdu_index].data, cnp.NPY_INT32)
		"""
		segmap在该星系处的切片
		"""

		if self._weightmap_fits is not None:
			weightmap_stamp_old = cnp.PyArray_Cast(self._weightmap_fits[weightmap_hdu_index].data, cnp.NPY_DOUBLE)
		else:
			weightmap_stamp_old = None
		"""
		weightmap在该星系处的切片，原始值
		"""

		if self._mask_fits is None:
			mask_stamp_old = np.full_like(cutout_stamp, False, dtype=np.bool_)
		else:
			mask_stamp_old = self._mask_fits[mask_hdu_index].data > 0
		"""
		mask在该星系处的切片，原始值
		"""

		if self._image_compare_fits is not None:
			image_compare_stamp = cnp.PyArray_Cast(self._image_compare_fits[image_compare_hdu_index].data, cnp.NPY_DOUBLE)

		print("开始stampmorphology")
		StampMorphology.__init__(self, label, cutout_stamp, segmap_stamp, mask_stamp_old, weightmap_stamp_old, gain, image_compare_stamp, output_image_dir, set_centroid)


	cpdef void close_all(self):
		# self.dump_stamps()
		self._image_fits.close()
		if self._segmap_fits is not None:
			self._segmap_fits.close()
		if self._mask_fits is not None:
			self._mask_fits.close()
		if self._weightmap_fits is not None:
			self._weightmap_fits.close()
		if self._image_compare_fits is not None:
			self._image_compare_fits.close()

cdef class CASInfo(MorphInfo):
	def __init__(self):
		super().__init__()

	def get_values(self):
		return [self._asymmetry_center[0], self._asymmetry_center[1], self.rpetro_circ, self.concentration, self.asymmetry, self.smoothness, self._sky_asymmetry, self.runtime, self.flags.value()]

	@staticmethod
	def get_value_names():
		return ["asymmetry_center_x", "asymmetry_center_y", "rp_circ", "C", "A", "S", "sky_asymmetry", "cas_time", "cas_flag"]

	@staticmethod
	def get_value_formats():
		return ["%f", "%f", "%f", "%f", "%f", "%f", "%f", "%f", "%d"]

cdef class GiniM20Info(MorphInfo):
	def __init__(self):
		super().__init__()

	def get_values(self):
		return [self.rpetro_ellip, self.gini, self.m20, self.runtime, self.flags.value()]

	@staticmethod
	def get_value_names():
		return ["rp_ellip", "G", "M20", "g_m20_runtime", "g_m20_flag"]

	@staticmethod
	def get_value_formats():
		return ["%f", "%f", "%f", "%f", "%d"]

cdef class MIDInfo(MorphInfo):
	def __init__(self):
		super().__init__()

	def get_values(self):
		return [self.multimode, self.intensity, self.deviation, self.runtime, self.flags.value()]

	@staticmethod
	def get_value_names():
		return ["M", "I", "D", "mid_time", "mid_flag"]

	@staticmethod
	def get_value_formats():
		return ["%f", "%f", "%f", "%f", "%d"]

cdef class CompareInfo(MorphInfo):
	def __init__(self):
		super().__init__()

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_nan_compare(self):
		"""
		Flag any NaN or inf values within the postage stamp.
		"""
		cdef cnp.ndarray locs_invalid = ~np.isfinite(self._image_compare_stamp)
		return locs_invalid

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_badpixels_compare(self):
		"""
		Flag badpixels (outliers).
		"""
		cdef cnp.ndarray badpixels = cnp.PyArray_ZEROS(2, [self.base_info.ny_stamp, self.base_info.nx_stamp],
													   cnp.NPY_BOOL, 0)
		if self.base_info.constants.n_sigma_outlier > 0:
			badpixels = self.base_info._get_badpixels(self._image_compare_stamp)
			self.num_badpixels = np.sum(badpixels)
		return badpixels

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_compare(self):
		"""
		Create a total binary mask for the "postage stamp".
		Pixels belonging to other sources (as well as pixels masked
		using the ``mask`` keyword argument) are set to ``True``,
		but the background (segmap == 0) is left alone.
		"""
		# cdef cnp.ndarray segmap_stamp = self.base_info._segmap[self.base_info._slice_stamp]
		cdef cnp.ndarray mask_stamp = (self.base_info._segmap_stamp != 0) & (
					self.base_info._segmap_stamp != self.base_info.label)
		mask_stamp |= self._mask_stamp_nan_compare
		mask_stamp |= self._mask_stamp_badpixels_compare
		return mask_stamp

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_no_bg_compare(self):
		"""
		Similar to ``_mask_stamp``, but also mask the background.
		"""
		# segmap_stamp = self.base_info._segmap[self.base_info._slice_stamp]
		return self._mask_stamp_compare | (self.base_info._segmap_stamp == 0)

	cdef cnp.ndarray[double, ndim=2] get_cutout_stamp_maskzeroed_compare(self):
		"""
		Return a data cutout with its shape and position determined
		by ``_slice_stamp``. Pixels belonging to other sources
		(as well as pixels where ``mask`` == 1) are set to zero,
		but the background is left alone.

		In addition, NaN or inf values are removed at this point,
		as well as badpixels (outliers).
		"""
		return cnp.PyArray_Where(~self._mask_stamp_compare,
								 self._image_compare_stamp, 0.0)

	cdef cnp.ndarray[double, ndim=2] get_cutout_stamp_maskzeroed_no_bg_compare(self):
		"""
		Like ``_cutout_stamp_maskzeroed``, but also mask the
		background.
		"""
		return cnp.PyArray_Where(~self._mask_stamp_no_bg_compare,
								 self._image_compare_stamp, 0.0)

	def get_values(self):
		return [self.color_dispersion, self.runtime]

	@staticmethod
	def get_value_names():
		return ["color_dispersion", "color_dispersion_time"]

	@staticmethod
	def get_value_formats():
		return ["%f", "%f"]


cdef class G2Info(MorphInfo):
	"""
	G2(segmented_image, g2_modular_tolerance=0.01, g2_phase_tolerance=0.01)

	Extracts entropy metric from the supplied image.

	Parameters
	----------
	segmented_image : 2-d `~numpy.ndarray`
		Segmented image data array.
	g2_modular_tolerance : float, optional
		Modular tolerance. How much differences in vector modules will be acepted. Ranges from 0.0 (vectors should be same to count the same)
		to 1.0 (any vectors will be counted as same). Default is 0.01.
	g2_phase_tolerance : float, optional
		Phase tolerance. How much differences in vector phases will be acepted. Ranges from 0.0 (vectors should be same to count the same)
		to 3.14 (any vectors will be counted as same, even completly opposite). Default is 0.01.
	"""

	def __init__(self, cnp.ndarray[double, ndim=2] segmented_image, ConstantsSetting constants):
		super().__init__()
		if segmented_image.shape[0] != segmented_image.shape[1]:
			raise ValueError("array must be square")
		if segmented_image.size == 0:
			raise ValueError("the size array can not be 0")
		if segmented_image.shape[0] % 2 == 0:
			raise ValueError("the stamp shape should be odd")

		self.segmented_image = segmented_image
		self.g2_modular_tolerance = constants.g2_modular_tolerance
		self.g2_phase_tolerance = constants.g2_phase_tolerance


	cdef double calc_g2(self):
		"""
		Get a g2 metric.

		Returns:
			result_g2 : `double`
		"""
		cdef G2Calculator g2c = G2Calculator(
				# need to pass a copy, if not, it is overwritten inside (strange)
				self.segmented_image.copy(),
				_get_contour_count(self.segmented_image),
				self.g2_modular_tolerance,
				self.g2_phase_tolerance)

		try:
			self.result_g2, self.gradient_x, self.gradient_y, self.gradient_asymmetric_x, self.gradient_asymmetric_y, self.modules_normalized, self.phases = g2c.get_g2()
		except ValueError:
			raise ValueError('Not enough valid pixels in image for g2 extraction')

		return self.result_g2

	cdef get_gradient_plot(self):
		"""(Debugging routine) Gradient plot showing the vector field
		before removing symmetrical vector pairs"""
		cdef int figSize = 7
		fig, ax = plt.subplots(figsize=(figSize, figSize))

		x, y = np.meshgrid(np.arange(0, self.gradient_x.shape[1], 1), np.arange(
			0, self.gradient_y.shape[0], 1))
		ax.quiver(x, y, self.gradient_y, self.gradient_x)
		ax.tick_params(labelsize=16)

		return ax

	cdef get_asymmetry_gradient_plot(self):
		"""(Debugging routine) Asymmetrical gradient plot showing the vector field
		after removing symmetrical vector pairs."""
		cdef int figSize = 7
		fig, ax = plt.subplots(figsize=(figSize, figSize))

		x, y = np.meshgrid(np.arange(0, self.gradient_asymmetric_x.shape[1], 1), np.arange(
			0, self.gradient_asymmetric_y.shape[0], 1))
		ax.quiver(x, y, self.gradient_asymmetric_y, self.gradient_asymmetric_x)
		ax.tick_params(labelsize=16)

		return ax

	def get_values(self):
		return [self.result_g2, self.runtime]

	@staticmethod
	def get_value_names():
		return ["g2", "g2_time"]

	@staticmethod
	def get_value_formats():
		return ["%f", "%f"]

cdef class ShapeAsymmetryInfo(MorphInfo):
	def __init__(self):
		super().__init__()

	def get_values(self):
		return [self.rhalf_circ, self.rhalf_ellip, self.shape_asymmetry, self.runtime, self.flags.value()]

	@staticmethod
	def get_value_names():
		return ["rhalf_circ", "rhalf_ellip", "shape_asymmetry", "shape_asym_time", "shape_asym_flag"]

	@staticmethod
	def get_value_formats():
		return ["%f", "%f", "%f", "%f", "%d"]

cdef class SersicInfo:
	pass
