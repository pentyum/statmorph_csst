# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import warnings

from astropy.utils.exceptions import AstropyUserWarning
import numpy as np
from astropy.convolution import convolve
import skimage.measure
import matplotlib.pyplot as plt

from libc.math cimport sqrt, round
from libc.time cimport clock, CLOCKS_PER_SEC
cimport numpy as cnp

from .petrosian cimport _rpetro_circ_generic
from .flags cimport Flags
cimport statmorph_cython.cas
cimport statmorph_cython.g_m20
cimport statmorph_cython.mid
cimport statmorph_cython.multiply
cimport statmorph_cython.color_dispersion
from .constants_setting cimport ConstantsSetting
from .g2 cimport G2Calculator, _get_contour_count, get_G2

cnp.import_array()

cdef class BaseInfo(MorphInfo):
	def __init__(self, cnp.ndarray[double,ndim=2] image, cnp.ndarray[int,ndim=2] segmap, tuple segmap_slice,
				 int label, cnp.ndarray[cnp.npy_bool,ndim=2] mask=None, cnp.ndarray[double,ndim=2] weightmap=None,
				 double gain=-1, bint calc_cas=True, bint calc_g_m20=True, bint calc_mid=True, bint calc_multiply=False,
				 bint calc_color_dispersion=False, cnp.ndarray[double,ndim=2] image_compare=None, bint calc_g2=False,
				 str output_image_dir=None):
		super().__init__()
		self.calc_cas = calc_cas
		self.calc_g_m20 = calc_g_m20
		self.calc_mid = calc_mid
		self.calc_multiply = calc_multiply
		self.calc_color_dispersion = calc_color_dispersion
		self.calc_g2 = calc_g2

		self.constants = ConstantsSetting()

		self.output_image_dir = output_image_dir
		"""
		图像输出文件夹，None表示不输出
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

		self.label = label
		"""
		该星系在在segmap中的label
		"""

		self._mask = mask
		"""
		需要遮蔽的区域
		"""

		self._weightmap = weightmap
		"""
		权重图
		"""

		self._gain = gain
		"""
		增益
		"""

		# Measure runtime
		cdef long global_start = clock()

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

		self._slice_stamp = self.get_slice_stamp()
		"""
		星系区域切片下标，依次为y和x，相对整幅图像
		"""

		self._cutout_stamp = self._image[self._slice_stamp]
		"""
		原始图像在该星系处的切片
		"""

		self._segmap_stamp = self._segmap[self._slice_stamp]
		"""
		segmap在该星系处的切片
		"""

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

		self.nx_stamp = self.get_nx_stamp()
		"""
		图像切片的宽度
		"""

		self.ny_stamp = self.get_ny_stamp()
		"""
		图像切片的高度
		"""

		if self.nx_stamp * self.ny_stamp > 9000000 :
			warnings.warn('Cutout size too big (%d*%d>9M), skip.' % (self.nx_stamp,self.ny_stamp), AstropyUserWarning)
			self._abort_calculations()
			return

		self._mask_stamp_nan = self.get_mask_stamp_nan()
		"""
		图像切片中有哪些点是nan或者inf
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
		if np.sum(self._cutout_stamp_maskzeroed_no_bg) <= 0:
			warnings.warn('Total flux is nonpositive. Returning empty object.',
						  AstropyUserWarning)
			self._abort_calculations()
			return

		self.size = len(self._cutout_stamp[~self._mask_stamp_no_bg])
		"""
		用于计算的星系本体的全部像素的数量
		"""

		self.surface_brightness = np.mean(self._cutout_stamp[~self._mask_stamp_no_bg])
		"""
		用于计算的星系本体的大小每个像素流量的平均值
		"""

		self._centroid = self.get_centroid()
		"""
		星系本体图像切片的一阶矩，即光度分布的质心，依次为x和y，坐标是相对于切片的
		"""

		self.xc_centroid = self.get_xc_centroid()
		"""
		星系光度质心的x坐标，相对于整个图像的
		"""

		self.yc_centroid = self.get_yc_centroid()
		"""
		星系光度质心的y坐标，相对于整个图像的
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

		cdef (double, double) center_used
		cdef long start

		if calc_cas:
			start = clock()
			self.cas = statmorph_cython.cas.calc_cas(self)
			self.cas.calc_runtime(start)
			center_used = self.cas._asymmetry_center
		else:
			self._use_centroid = True
			center_used = self._centroid


		if calc_g_m20:
			start = clock()
			self.g_m20 = statmorph_cython.g_m20.calc_g_m20(self, center_used)
			self.g_m20.calc_runtime(start)

		if calc_mid:
			start = clock()
			self.mid = statmorph_cython.mid.calc_mid(self)
			self.mid.calc_runtime(start)
			if calc_multiply:
				self.multiply = statmorph_cython.multiply.multiply(self.mid._cutout_mid)

		if calc_color_dispersion:
			if image_compare is not None:
				start = clock()
				self.image_compare = image_compare
				self.compare_info = statmorph_cython.color_dispersion.calc_color_dispersion(self, image_compare)
				self.compare_info.calc_runtime(start)
			else:
				warnings.warn("[Color dispersion] compare image not defined")

		if calc_g2:
			start = clock()
			self.g2 = get_G2(self, center_used)
			self.g2.calc_runtime(start)

		# Save image
		if self.output_image_dir is not None:
			self.save_image()

		self.calc_runtime(global_start)

	cdef tuple get_slice_stamp(self):
		"""
		Attempt to create a square slice that is somewhat larger
		than the minimal bounding box containing the labeled segment.
		Note that the cutout may not be square when the source is
		close to a border of the original image.
		"""
		assert self.constants.cutout_extent >= 1.0
		assert self.constants.min_cutout_size >= 2
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
		cdef int dist = int(max(dx, dy) * self.constants.cutout_extent / 2.0)
		# Make sure that cutout size is at least ``min_cutout_size``
		dist = max(dist, self.constants.min_cutout_size // 2)
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

	cdef cnp.ndarray get_mask_stamp_nan(self):
		"""
		Flag any NaN or inf values within the postage stamp.
		"""
		cdef cnp.ndarray locs_invalid = ~np.isfinite(self._cutout_stamp)
		if self._weightmap is not None:
			locs_invalid |= ~np.isfinite(self._weightmap[self._slice_stamp])
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
		if self._mask is not None:
			mask_stamp |= self._mask[self._slice_stamp]
		mask_stamp |= self._mask_stamp_nan
		mask_stamp |= self._mask_stamp_badpixels
		return mask_stamp

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_no_bg(self):
		"""
		Similar to ``_mask_stamp``, but also mask the background.
		"""
		# cdef cnp.ndarray segmap_stamp = self._segmap[self._slice_stamp]
		return self._mask_stamp | (self._segmap_stamp == 0)

	cdef cnp.ndarray get_cutout_stamp_maskzeroed(self):
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

	cdef cnp.ndarray get_cutout_stamp_maskzeroed_no_bg(self):
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
		可能产生警告1,2
		"""
		cdef int ny, nx
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

			warnings.warn('Centroid is out-of-range. Fixing at center of ' +
						  'postage stamp (bad!).', AstropyUserWarning)
			yc = ny / 2.0
			xc = nx / 2.0
			self.flags.set_flag_true(0) # unusual

		# Print warning if centroid is masked:
		cdef int ic = int(round(self._yc_stamp))
		cdef int jc = int(round(self._xc_stamp))
		if self._cutout_stamp_maskzeroed[ic][jc] == 0.0:
			warnings.warn('Centroid is masked.', AstropyUserWarning)
			self.flags.set_flag_true(1)

		#return np.array([xc, yc])
		return xc, yc

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
		可能产生警告3,4,5,6
		"""
		return _rpetro_circ_generic(self._cutout_stamp_maskzeroed, self._centroid, self._diagonal_distance, self.flags, self.constants)

	cdef void _abort_calculations(self):
		"""
		Some cases are so bad that basically nothing can be measured
		(e.g. a bunch of pixels with a nonpositive total sum). We
		deal with these cases by creating an "empty" object and
		interrupting the constructor.
		"""
		self.nx_stamp = -99
		self.ny_stamp = -99
		self.flag_catastrophic = True
		# 图像切片总和不是正数，或者图片过大，直接终止全部计算
		self.runtime = -99.0

		if self.calc_cas:
			self.cas = CASInfo()
		if self.calc_g_m20:
			self.g_m20 = GiniM20Info()
		if self.calc_mid:
			self.mid = MIDInfo()
			if self.calc_multiply:
				self.multiply = -99
		if self.calc_color_dispersion:
			self.compare_info = CompareInfo()
		if self.calc_g2:
			self.g2 = G2Info()


	cdef void save_image(self):
		"""
		其中绿色轮廓为原始segmentation map(_mask_stamp_no_bg)，黄色轮廓为计算G和M20的segmentation map(_segmap_gini)
		绿点为光度中心，白点为不对称中心
		黑色圆圈为以不对称中心为中心的1.5rp，白色圆圈为用于计算C的r20，灰色圆圈为r80
		黑色方框为背景区域
		"""
		cdef int stamp_x = self._slice_stamp[1].start
		cdef int stamp_y = self._slice_stamp[0].start
		cdef int rec_x, rec_y, rec_x_length, rec_y_length

		plt.figure(figsize=(5, 5))
		extent = (stamp_x, self._slice_stamp[1].stop, stamp_y, self._slice_stamp[0].stop)
		plt.imshow(np.log10(self._cutout_stamp), cmap="gray_r", origin="lower", extent=extent, interpolation="none")

		if self.cas is not None:
			rec_x = stamp_x + self.cas._slice_skybox[1].start
			rec_y = stamp_y + self.cas._slice_skybox[0].start
			rec_x_length = self.cas._slice_skybox[1].stop - self.cas._slice_skybox[1].start
			rec_y_length = self.cas._slice_skybox[0].stop - self.cas._slice_skybox[0].start

			rec_sky = plt.Rectangle((rec_x, rec_y), rec_x_length, rec_y_length, edgecolor="black", linewidth=2,
									fill=False)

			asym_center = (self.cas.xc_asymmetry, self.cas.yc_asymmetry)

			circ_15rp = plt.Circle(asym_center, self.constants.petro_extent_cas * self.cas.rpetro_circ, edgecolor="black",
								   linewidth=2,
								   fill=False)
			circ_r80 = plt.Circle(asym_center, self.cas.r80, edgecolor="gray", linewidth=1, fill=False)
			circ_r20 = plt.Circle(asym_center, self.cas.r20, edgecolor="white", linewidth=1, fill=False)

			plt.scatter(*asym_center, s=10, color="cyan", label="asym c (%.1f,%.1f)" % tuple(self.cas._asymmetry_center))

			plt.gca().add_patch(rec_sky)
			plt.gca().add_patch(circ_15rp)
			plt.gca().add_patch(circ_r80)
			plt.gca().add_patch(circ_r20)

		plt.scatter(self.xc_centroid, self.yc_centroid, s=10, color="olive",
					label="centroid (%.1f,%.1f)" % tuple(self._centroid))
		plt.contour(self._mask_stamp_no_bg, colors="green", extent=extent, levels=1)

		if self.g_m20 is not None:
			plt.contour(self.g_m20._segmap_gini, colors="yellow", extent=extent, levels=1)

		plt.tight_layout()
		plt.legend()

		plt.savefig("%s/%d.png" % (self.output_image_dir, self.label))

cdef class CASInfo(MorphInfo):
	def __init__(self):
		super().__init__()

	def get_values(self):
		return [self.rpetro_circ, self.concentration, self.asymmetry, self.smoothness, self.flags.value()]

	@staticmethod
	def get_value_names():
		return ["rp_circ", "C", "A", "S", "cas_flag"]

	@staticmethod
	def get_value_formats():
		return ["%f", "%f", "%f", "%f", "%d"]

cdef class GiniM20Info(MorphInfo):
	def __init__(self):
		super().__init__()

	def get_values(self):
		return [self.rpetro_ellip, self.gini, self.m20, self.flags.value()]

	@staticmethod
	def get_value_names():
		return ["rp_ellip", "G", "M20", "g_m20_flag"]

	@staticmethod
	def get_value_formats():
		return ["%f", "%f", "%f", "%d"]

cdef class MIDInfo(MorphInfo):
	def __init__(self):
		super().__init__()

	def get_values(self):
		return [self.multimode, self.intensity, self.deviation, self.flags.value()]

	@staticmethod
	def get_value_names():
		return ["M", "I", "D", "mid_flag"]

	@staticmethod
	def get_value_formats():
		return ["%f", "%f", "%f", "%d"]

cdef class CompareInfo(MorphInfo):
	def __init__(self):
		super().__init__()

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_nan_compare(self):
		"""
		Flag any NaN or inf values within the postage stamp.
		"""
		cdef cnp.ndarray locs_invalid = ~np.isfinite(self._image_compare[self.base_info._slice_stamp])
		return locs_invalid

	cdef cnp.ndarray[cnp.npy_bool, ndim=2] get_mask_stamp_badpixels_compare(self):
		"""
		Flag badpixels (outliers).
		"""
		cdef cnp.ndarray badpixels = cnp.PyArray_ZEROS(2, [self.base_info.ny_stamp, self.base_info.nx_stamp],
													   cnp.NPY_BOOL, 0)
		if self.base_info.constants.n_sigma_outlier > 0:
			badpixels = self.base_info._get_badpixels(self._image_compare[self.base_info._slice_stamp])
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
								 self._image_compare[self.base_info._slice_stamp], 0.0)

	cdef cnp.ndarray[double, ndim=2] get_cutout_stamp_maskzeroed_no_bg_compare(self):
		"""
		Like ``_cutout_stamp_maskzeroed``, but also mask the
		background.
		"""
		return cnp.PyArray_Where(~self._mask_stamp_no_bg_compare,
								 self._image_compare[self.base_info._slice_stamp], 0.0)

	def get_values(self):
		return [self.color_dispersion]

	@staticmethod
	def get_value_names():
		return ["color_dispersion"]

	@staticmethod
	def get_value_formats():
		return ["%f"]


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
		return [self.result_g2]

	@staticmethod
	def get_value_names():
		return ["g2"]

	@staticmethod
	def get_value_formats():
		return ["%f"]


cdef class MorphInfo:
	def __init__(self):
		# These flags will be modified during the calculations:
		self.flags = Flags()  # attempts to flag bad measurements
		"""
		bitmask用于标记错误
		"""
		self.runtime = 0

	@staticmethod
	cdef double get_duration_sec(long end, long start):
		return <double> (end - start) / CLOCKS_PER_SEC

	cdef void calc_runtime(self, long start):
		self.runtime = MorphInfo.get_duration_sec(clock(), start)

	def get_values(self):
		pass

	@staticmethod
	def get_value_names():
		pass

	@staticmethod
	def get_value_formats():
		pass
