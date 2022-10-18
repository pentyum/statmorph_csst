# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import warnings
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np
import scipy.ndimage as ndi
cimport numpy as cnp
import skimage.transform
from libc.math cimport fabs, log10, sqrt
from numpy.math cimport isnan, isfinite

from .statmorph cimport BaseInfo, CASInfo
from .constants_setting cimport ConstantsSetting
from .petrosian cimport _rpetro_circ_generic
from .photutils_simplified cimport CircularAperture, CircularAnnulus, _aperture_area, do_photometry, _radius_at_fraction_of_total_circ
from .flags cimport Flags
from .optimize.neldermead cimport fmin

cnp.import_array()

cdef void check_rp_beyond_edge(double xc_centroid, double yc_centroid, double _rpetro_circ_centroid, cnp.npy_intp* image_shape, Flags flags, ConstantsSetting constants):
	"""
	可能产生警告CAS警告0
	"""
	cdef double rp_min_x = xc_centroid - constants.petro_extent_cas * _rpetro_circ_centroid
	cdef double rp_min_y = yc_centroid - constants.petro_extent_cas * _rpetro_circ_centroid
	cdef double rp_max_x = xc_centroid + constants.petro_extent_cas * _rpetro_circ_centroid
	cdef double rp_max_y = yc_centroid + constants.petro_extent_cas * _rpetro_circ_centroid

	if rp_min_x < 0 or rp_min_y < 0 or rp_max_x > image_shape[1] or rp_max_y > image_shape[0]:
		warnings.warn('[CAS] _petro_extent_cas*_rpetro_circ_centroid is out of the image', AstropyUserWarning)
		flags.set_flag_true(0)

cdef tuple get_slice_skybox(cnp.ndarray[int,ndim=2] _segmap, tuple _slice_stamp, int nx_stamp, int ny_stamp, _mask, Flags flags, ConstantsSetting constants):
	"""
	Try to find a region of the sky that only contains background.

	In principle, a more accurate approach is possible
	(e.g. Shi et al. 2009, ApJ, 697, 1764).
	可能产生CAS警告1,2
	"""
	cdef cnp.ndarray segmap = _segmap[_slice_stamp]
	cdef int ny = ny_stamp
	cdef int nx = nx_stamp
	# cdef cnp.ndarray mask = np.zeros((ny, nx), dtype=np.bool8)
	cdef cnp.ndarray mask = cnp.PyArray_ZEROS(2, [ny, nx], cnp.NPY_BOOL, 0)
	if _mask is not None:
		mask = _mask[_slice_stamp]

	assert constants.skybox_size >= 2
	cdef int cur_skybox_size = constants.skybox_size
	cdef tuple boxslice
	cdef int i, j
	while True:
		for i in range(ny - cur_skybox_size):
			for j in range(nx - cur_skybox_size):
				boxslice = (slice(i, i + cur_skybox_size),
							slice(j, j + cur_skybox_size))
				if np.all(segmap[boxslice] == 0) and np.all(~mask[boxslice]):
					return boxslice
		# 循环完成了仍然没找到合适区域就减小一半
		cur_skybox_size //= 2
		if cur_skybox_size <= 2:
			# If we got here, a skybox of the given size was not found.
			warnings.warn('[skybox] Skybox not found.', AstropyUserWarning)
			flags.set_flag_true(1)
			return slice(0, 0), slice(0, 0)
		if constants.verbose:
			warnings.warn('[skybox] Reducing skybox size to %d.' % (
				cur_skybox_size), AstropyUserWarning)
		flags.set_flag_true(2)

cdef double get_sky_asymmetry(cnp.ndarray[double,ndim=2] bkg, Flags flags):
	"""
	Asymmetry of the background. Equal to -99.0 when there is no
	skybox. Note the peculiar normalization (for reference only).

	"""
	# cdef cnp.ndarray bkg = _cutout_stamp_maskzeroed[_slice_skybox]
	cdef cnp.ndarray bkg_180 = bkg[::-1, ::-1]
	if cnp.PyArray_SIZE(bkg) == 0:
		assert flags.get_flag(1)
		return -99.0

	return np.sum(np.abs(bkg_180 - bkg)) / float(cnp.PyArray_SIZE(bkg))

cdef double get_sky_mean(cnp.ndarray[double,ndim=2] bkg, Flags flags):
	"""
	Mean background value. Equal to -99.0 when there is no skybox.
	"""
	# cdef cnp.ndarray bkg = _cutout_stamp_maskzeroed[_slice_skybox]
	if cnp.PyArray_SIZE(bkg) == 0:
		assert flags.get_flag(1)
		return -99.0

	return np.mean(bkg)

cdef double get_sky_median(cnp.ndarray[double,ndim=2] bkg, Flags flags):
	"""
	Median background value. Equal to -99.0 when there is no skybox.
	"""
	# cdef cnp.ndarray bkg = _cutout_stamp_maskzeroed[_slice_skybox]
	if cnp.PyArray_SIZE(bkg) == 0:
		assert flags.get_flag(1)
		return -99.0

	return np.median(bkg)

cdef double get_sky_sigma(cnp.ndarray[double,ndim=2] bkg, Flags flags):
	"""
	Standard deviation of the background. Equal to -99.0 when there
	is no skybox.
	"""
	# cdef cnp.ndarray bkg = _cutout_stamp_maskzeroed[_slice_skybox]
	if cnp.PyArray_SIZE(bkg) == 0:
		assert flags.get_flag(1)
		return -99.0

	return np.std(bkg)

cdef double get_sky_smoothness(cnp.ndarray[double,ndim=2] bkg, double rpetro_circ, Flags flags, ConstantsSetting constants):
	"""
	Smoothness of the background. Equal to -99.0 when there is no
	skybox. Note the peculiar normalization (for reference only).
	"""
	# cdef cnp.ndarray bkg = _cutout_stamp_maskzeroed[_slice_skybox]
	if cnp.PyArray_SIZE(bkg) == 0:
		assert flags.get_flag(1)
		return -99.0

	# If the smoothing "boxcar" is larger than the skybox itself,
	# this just sets all values equal to the mean:
	cdef int boxcar_size = int(constants.petro_fraction_cas * rpetro_circ)
	cdef cnp.ndarray bkg_smooth = ndi.uniform_filter(bkg, size=boxcar_size)

	cdef cnp.ndarray bkg_diff = bkg - bkg_smooth
	bkg_diff[bkg_diff < 0] = 0.0  # set negative pixels to zero

	return np.sum(bkg_diff) / float(cnp.PyArray_SIZE(bkg))

cdef cnp.ndarray simplified_rot180(cnp.ndarray image, (double, double) center):
	cdef cnp.ndarray rotted_image = image[::-1, ::-1]
	# cdef cnp.ndarray new_image = np.zeros_like(rotted_image)
	cdef cnp.ndarray new_image = cnp.PyArray_ZEROS(2, image.shape, image.dtype.num, 0)

	cdef int Y = image.shape[0]
	cdef int X = image.shape[1]
	cdef double x = center[0] + 0.5
	cdef double y = center[1] + 0.5
	cdef int double_x = int(2 * x)
	cdef int double_y = int(2 * y)
	if double_x < X:
		if double_y < Y:
			new_image[0:double_y, 0:double_x] = rotted_image[Y - double_y:Y, X - double_x:X]
		else:
			new_image[double_y - Y:Y, 0:double_x] = rotted_image[0:2 * Y - double_y, X - double_x:X]
	else:
		if double_y < Y:
			new_image[0:double_y, double_x - X:X] = rotted_image[Y - double_y:Y, 0:2 * X - double_x]
		else:
			new_image[double_y - Y:Y, double_x - X:X] = rotted_image[0:2 * Y - double_y, 0:2 * X - double_x]
	return new_image

cpdef double _asymmetry_function((double, double) center, cnp.ndarray[double,ndim=2] image, cnp.ndarray[cnp.npy_bool,ndim=2] _mask_stamp,
								 double _rpetro_circ_centroid, double _sky_asymmetry, Flags flags,
								 ConstantsSetting constants):
	"""
	Helper function to determine the asymmetry and center of asymmetry.
	The idea is to minimize the output of this function.
	可能产生CAS警告3,4

	Parameters
	----------
	center : tuple or array-like
		The (x,y) position of the center.
	image : array-like
		The 2D image.

	Returns
	-------
	asym : The asymmetry statistic for the given center.

	"""
	# image = np.float64(image)  # skimage wants double
	cdef int ny = image.shape[0]
	cdef int nx = image.shape[1]
	cdef double xc = center[0]
	cdef double yc = center[1]
	cdef int image_size = nx * ny

	if xc < 0 or xc >= nx or yc < 0 or yc >= ny:
		warnings.warn('[asym_center] Minimizer tried to exit bounds.',
					  AstropyUserWarning)
		flags.set_flag_true(3)
		# Return high value to keep minimizer within range:
		return 100.0

	# Rotate around given center
	cdef cnp.ndarray image_180
	# cdef cnp.ndarray image_180 = skimage.transform.rotate(image, 180.0, center=center)
	if 0 <= constants.simplified_rot_threshold < image_size:
		image_180 = simplified_rot180(image, center)
	else:
		image_180 = skimage.transform.rotate(image, 180.0, center=center)

	# Apply symmetric mask
	#cdef cnp.ndarray mask = self._mask_stamp.copy()
	cdef cnp.ndarray[cnp.npy_bool,ndim=2] mask = cnp.PyArray_NewCopy(_mask_stamp, cnp.NPY_ANYORDER)
	# cdef cnp.ndarray mask_180 = skimage.transform.rotate(mask, 180.0, center=center)
	# mask_180 = mask_180 >= 0.5  # convert back to bool
	cdef cnp.ndarray[cnp.npy_bool,ndim=2] mask_180
	if 0 <= constants.simplified_rot_threshold < image_size:
		mask_180 = simplified_rot180(mask, center)
	else:
		mask_180 = skimage.transform.rotate(mask, 180.0, center=center)
		mask_180 = mask_180 >= 0.5

	cdef cnp.ndarray[cnp.npy_bool,ndim=2] mask_symmetric = mask | mask_180
	image = cnp.PyArray_Where(~mask_symmetric, image, 0.0)
	image_180 = cnp.PyArray_Where(~mask_symmetric, image_180, 0.0)

	# Create aperture for the chosen kind of asymmetry
	cdef double r = constants.petro_extent_cas * _rpetro_circ_centroid
	cdef CircularAperture ap = CircularAperture(center, r)

	# Apply eq. 10 from Lotz et al. (2004)
	cdef double ap_abs_sum = do_photometry(ap, np.abs(image))
	cdef double ap_abs_diff = do_photometry(ap, np.abs(image_180 - image))

	if ap_abs_sum == 0.0:
		warnings.warn('[asymmetry_function] Zero flux sum.',
					  AstropyUserWarning)
		flags.set_flag_true(4) # unusual
		return -99.0  # invalid

	cdef double asym

	if _sky_asymmetry == -99.0:  # invalid skybox
		asym = ap_abs_diff / ap_abs_sum
	else:
		ap_area = _aperture_area(ap, mask_symmetric)
		asym = (ap_abs_diff - ap_area * _sky_asymmetry) / ap_abs_sum

	return asym

cdef (double,double) get_asymmetry_center(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, (double,double) center_0,
										  cnp.ndarray[cnp.npy_bool,ndim=2] _mask_stamp, double _rpetro_circ_centroid, double _sky_asymmetry,
										  Flags flags, ConstantsSetting constants):
	"""
	Find the position of the central pixel (relative to the
	"postage stamp" cutout) that minimizes the (CAS) asymmetry.
	可能产生CAS警告3,4,5
	"""
	# cdef cnp.ndarray center_0 = np.array([self._xc_stamp, self._yc_stamp])  # initial guess

	cdef (double,double) center_asym = fmin(_asymmetry_function, center_0,
											(_cutout_stamp_maskzeroed, _mask_stamp, _rpetro_circ_centroid, _sky_asymmetry, flags, constants),
											1e-6, 1e-4, constants.fmin_maxiter, constants.fmin_maxiter, False)
	#cdef (double,double) center_asym
	#center_asym = center_asym_array[0],center_asym_array[1]

	# Check if flag was activated by _asymmetry_function:
	if flags.get_flag(5):
		warnings.warn('Using centroid instead of asymmetry center.',
					  AstropyUserWarning)
		center_asym = center_0

	# Print warning if center is masked
	cdef int ic = int(round(center_asym[1]))
	cdef int jc = int(round(center_asym[0]))
	if _cutout_stamp_maskzeroed[ic, jc] == 0:
		warnings.warn('[asym_center] Asymmetry center is masked.',
					  AstropyUserWarning)
		flags.set_flag_true(5)

	return center_asym

cdef double get_asymmetry(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, (double,double) _asymmetry_center,
						  cnp.ndarray[cnp.npy_bool,ndim=2] _mask_stamp, double _rpetro_circ_centroid, double _sky_asymmetry, Flags flags, ConstantsSetting constants):
	"""
	Calculate asymmetry as described in Lotz et al. (2004).
	可能产生CAS警告3,4
	"""
	cdef cnp.ndarray image = _cutout_stamp_maskzeroed
	cdef double asym = _asymmetry_function(_asymmetry_center, image, _mask_stamp, _rpetro_circ_centroid, _sky_asymmetry, flags, constants)

	return asym

cdef double get_flux_circ(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, (double,double) _asymmetry_center, double rpetro_circ, ConstantsSetting constants):
	"""
	Return the sum of the pixel values over a circular aperture
	with radius equal to ``petro_extent_flux`` (usually 2) times
	the circular Petrosian radius.
	"""
	cdef cnp.ndarray image = _cutout_stamp_maskzeroed
	cdef double r = constants.petro_extent_flux * rpetro_circ
	cdef CircularAperture ap = CircularAperture(_asymmetry_center, r)
	# Force flux sum to be positive:
	cdef double ap_sum = fabs(do_photometry(ap, image))
	return ap_sum

cdef double _radius_at_fraction_of_total_cas(double fraction, cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, (double,double) _asymmetry_center, double rpetro_circ, Flags flags, ConstantsSetting constants):
	"""
	Specialization of ``_radius_at_fraction_of_total_circ`` for
	the CAS calculations.
	可能产生CAS警告11,12
	"""
	cdef cnp.ndarray image = _cutout_stamp_maskzeroed
	cdef (double, double) center = _asymmetry_center
	cdef double r_upper = constants.petro_extent_cas * rpetro_circ
	cdef double r
	cdef int flag
	r, flag = _radius_at_fraction_of_total_circ(image, center, r_upper, fraction)
	if flag:
		flags.set_flag_true(11) # unusual
	#self.flag = max(self.flag, flag)

	if isnan(r) or (r <= 0.0):
		warnings.warn('[CAS] Invalid radius_at_fraction_of_total.',
					  AstropyUserWarning)
		flags.set_flag_true(12) # unusual
		r = -99.0  # invalid

	return r

cdef double get_r20(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, (double,double) _asymmetry_center, double rpetro_circ, Flags flags, ConstantsSetting constants):
	"""
	The radius that contains 20% of the light within
	'petro_extent_cas' (usually 1.5) times 'rpetro_circ'.
	可能产生CAS警告11,12
	"""
	return _radius_at_fraction_of_total_cas(0.2, _cutout_stamp_maskzeroed, _asymmetry_center, rpetro_circ, flags, constants)

cdef double get_r50(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, (double,double) _asymmetry_center, double rpetro_circ, Flags flags, ConstantsSetting constants):
	"""
	The radius that contains 50% of the light within
	'petro_extent_cas' (usually 1.5) times 'rpetro_circ'.
	可能产生CAS警告11,12
	"""
	return _radius_at_fraction_of_total_cas(0.5, _cutout_stamp_maskzeroed, _asymmetry_center, rpetro_circ, flags, constants)

cdef double get_r80(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, (double,double) _asymmetry_center, double rpetro_circ, Flags flags, ConstantsSetting constants):
	"""
	The radius that contains 80% of the light within
	'petro_extent_cas' (usually 1.5) times 'rpetro_circ'.
	可能产生CAS警告11,12
	"""
	return _radius_at_fraction_of_total_cas(0.8, _cutout_stamp_maskzeroed, _asymmetry_center, rpetro_circ, flags, constants)

cdef double get_concentration(double r20, double r80):
	"""
	Calculate concentration as described in Lotz et al. (2004).
	"""
	cdef double C
	if (r20 == -99.0) or (r80 == -99.0):
		C = -99.0  # invalid
	else:
		C = 5.0 * log10(r80 / r20)

	return C

cdef double get_smoothness(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, (double,double) _asymmetry_center, double rpetro_circ,
						   double _sky_smoothness, Flags flags, ConstantsSetting constants):
	"""
	Calculate smoothness (a.k.a. clumpiness) as defined in eq. (11)
	from Lotz et al. (2004). Note that the original definition by
	Conselice (2003) includes an additional factor of 10.
	可能产生CAS警告14,15
	"""
	cdef cnp.ndarray image = _cutout_stamp_maskzeroed

	# Exclude central region during smoothness calculation:
	cdef double r_in = constants.petro_fraction_cas * rpetro_circ
	cdef double r_out = constants.petro_extent_cas * rpetro_circ
	cdef CircularAnnulus ap = CircularAnnulus(_asymmetry_center, r_in, r_out)

	cdef int boxcar_size = int(constants.petro_fraction_cas * rpetro_circ)
	cdef cnp.ndarray image_smooth = ndi.uniform_filter(image, size=boxcar_size)

	cdef cnp.ndarray image_diff = image - image_smooth
	image_diff[image_diff < 0] = 0.0  # set negative pixels to zero

	cdef double ap_flux = do_photometry(ap, image)
	cdef double ap_diff = do_photometry(ap, image_diff)

	if ap_flux <= 0:
		warnings.warn('[smoothness] Nonpositive total flux.',
					  AstropyUserWarning)
		flags.set_flag_true(14) # unusual
		return -99.0  # invalid

	cdef double S, area
	if _sky_smoothness == -99.0:  # invalid skybox
		S = ap_diff / ap_flux
	else:
		area = ap.area()
		S = (ap_diff - area * _sky_smoothness) / ap_flux

	if not isfinite(S):
		warnings.warn('Invalid smoothness.', AstropyUserWarning)
		flags.set_flag_true(15) # unusual
		return -99.0  # invalid

	return S

cdef CASInfo calc_cas(BaseInfo base_info, (double, double) set_asym_center):
	cdef CASInfo cas_info = CASInfo()

	check_rp_beyond_edge(base_info.xc_centroid, base_info.yc_centroid, base_info._rpetro_circ_centroid, base_info._image.shape, cas_info.flags, base_info.constants)
	"""
	检查1.5倍rp是否超出图像边缘
	"""

	cas_info._slice_skybox = get_slice_skybox(base_info._segmap, base_info._slice_stamp, base_info.nx_stamp, base_info.ny_stamp, base_info._mask, cas_info.flags, base_info.constants)
	"""
	背景区域切片下标，依次为y和x，相对于图像切片
	"""

	cas_info._bkg = base_info._cutout_stamp_maskzeroed[cas_info._slice_skybox]

	cas_info._sky_asymmetry = get_sky_asymmetry(cas_info._bkg, cas_info.flags)
	"""
	获得背景区域不对称度
	"""

	cdef double dx_c, dy_c

	if set_asym_center == (-1, -1):
		cas_info._asymmetry_center = get_asymmetry_center(base_info._cutout_stamp_maskzeroed, base_info._centroid, base_info._mask_stamp, base_info._rpetro_circ_centroid, cas_info._sky_asymmetry, cas_info.flags, base_info.constants)
		dx_c = cas_info._asymmetry_center[0] - base_info._centroid[0]
		dy_c = cas_info._asymmetry_center[1] - base_info._centroid[1]

		if sqrt(dx_c ** 2 + dy_c ** 2) >= base_info.constants.petro_extent_cas * base_info._rpetro_circ_centroid:
			base_info._use_centroid = True
			cas_info._asymmetry_center = base_info._centroid
			warnings.warn('[CAS] Asymmetry center is too far, using centroid center', AstropyUserWarning)
			cas_info.flags.set_flag_true(6)
	else:
		cas_info._asymmetry_center = set_asym_center
	"""
	获得不对称中心，依次为x和y，坐标是相对于切片的
	"""

	cas_info.xc_asymmetry = base_info.xmin_stamp + cas_info._asymmetry_center[0]
	cas_info.yc_asymmetry = base_info.ymin_stamp + cas_info._asymmetry_center[1]

	cas_info.asymmetry = get_asymmetry(base_info._cutout_stamp_maskzeroed, cas_info._asymmetry_center, base_info._mask_stamp, base_info._rpetro_circ_centroid, cas_info._sky_asymmetry, cas_info.flags, base_info.constants)
	"""
	以不对称中心计算得到的不对称度
	"""

	if base_info._use_centroid:
		cas_info.rpetro_circ = _rpetro_circ_generic(base_info._cutout_stamp_maskzeroed, cas_info._asymmetry_center, base_info._diagonal_distance, cas_info.flags, base_info.constants)
		"""
		以不对称中心为中心的Petrosian圆形孔径半径
		"""
	else:
		cas_info.rpetro_circ = base_info._rpetro_circ_centroid

	cas_info.r20 = get_r20(base_info._cutout_stamp_maskzeroed, cas_info._asymmetry_center, cas_info.rpetro_circ, cas_info.flags, base_info.constants)
	"""
	获得圆形孔径内光度占总光度20%的半径，中心为不对称中心，总光度默认为1.5倍rpetro_circ圆形孔径内测量得到。
	"""

	cas_info.r80 = get_r80(base_info._cutout_stamp_maskzeroed, cas_info._asymmetry_center, cas_info.rpetro_circ, cas_info.flags, base_info.constants)
	"""
	获得圆形孔径内光度占总光度80%的半径，中心为不对称中心，总光度默认为1.5倍rpetro_circ圆形孔径内测量得到。
	"""

	cas_info.concentration = get_concentration(cas_info.r20, cas_info.r80)
	"""
	集中度
	"""

	cas_info.sky_mean = get_sky_mean(cas_info._bkg, cas_info.flags)
	"""
	背景的平均值
	"""

	cas_info.sky_sigma = get_sky_sigma(cas_info._bkg, cas_info.flags)
	"""
	背景的标准差
	"""

	# Check if image is background-subtracted; set flag=1 if not.
	if fabs(cas_info.sky_mean) > cas_info.sky_sigma:
		warnings.warn('Image is not background-subtracted.', AstropyUserWarning)
		cas_info.flags.set_flag_true(13)

	cas_info._sky_smoothness = get_sky_smoothness(cas_info._bkg, cas_info.rpetro_circ, cas_info.flags, base_info.constants)
	"""
	背景的成团性
	"""

	cas_info.smoothness = get_smoothness(base_info._cutout_stamp_maskzeroed, cas_info._asymmetry_center, cas_info.rpetro_circ, cas_info._sky_smoothness, cas_info.flags, base_info.constants)
	"""
	成团性
	"""

	return cas_info