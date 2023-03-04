# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

##############################
# Calculate Color Dispersion #
##############################
"""
The color dispersion parament is quiet complex to calculate,
Here we repeat the same option for the image_compare

https://ui.adsabs.harvard.edu/abs/2003ApJ...598..827P
"""
import numpy as np
import scipy.optimize as opt

cimport numpy as cnp

from .statmorph cimport BaseInfo, CompareInfo

cnp.import_array()

'''
As the defination, 
we choose the elliptical petrosian to determine the aperture
'''

cdef cnp.ndarray[cnp.npy_bool,ndim=2] segmap_color_dispersion(double rpetro, (double,double) center, int nx, int ny):
	#rpetro = self.rpetro_circ

	#center = self._asymmetry_center
	#nx = self.nx_stamp
	#ny = self.ny_stamp
	cdef cnp.ndarray[cnp.npy_int64,ndim=3] indices = np.indices((ny, nx))
	cdef cnp.ndarray[double,ndim=2] distance = np.sqrt((indices[0] - center[1]) ** 2 + (indices[1] - center[0]) ** 2)
	cdef cnp.ndarray[cnp.npy_bool,ndim=2] segmap_dispersion = distance < 1.5 * rpetro
	return segmap_dispersion

'''
In this part
we calculate the assistant parament
'''

cpdef double find_alpha_beta_help(cnp.ndarray[double,ndim=1] array, cnp.ndarray image, cnp.ndarray image_compare):
	cdef cnp.ndarray error = image_compare - image * array[0] - array[1]
	return np.sum(error ** 2)


cdef (double,double) find_alpha_beta(cnp.ndarray _cutout_stamp_maskzeroed, cnp.ndarray _cutout_stamp_maskzeroed_compare):
	cdef cnp.ndarray[double,ndim=1] alpha_beta = opt.fmin(find_alpha_beta_help, [0.0, 0.0], args=(_cutout_stamp_maskzeroed, _cutout_stamp_maskzeroed_compare),
							 xtol=1e-6, disp=False)
	return alpha_beta[0], alpha_beta[1]

'''
In this part 
we calculate the color dispersion for the image
'''

cdef double _sky_dispersion(cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed, cnp.ndarray[double,ndim=2] _cutout_stamp_maskzeroed_compare,
							cnp.ndarray[double,ndim=2] _bkg, cnp.ndarray[double,ndim=2] _bkg_compare):
	cdef (double,double) help_para = find_alpha_beta(_cutout_stamp_maskzeroed, _cutout_stamp_maskzeroed_compare)
	cdef double sky_dispersion = np.sum((_bkg - help_para[1] * _bkg_compare) ** 2)
	return sky_dispersion

cdef double get_color_dispersion(CompareInfo compare_info):
	cdef double alpha, beta, dispersion_1, dispersion_2, sky_dispersion, color_dispersion
	cdef cnp.ndarray[cnp.npy_bool,ndim=2] segmap_dispersion
	cdef cnp.ndarray[double,ndim=1] image, image_compare

	assert cnp.PyArray_SAMESHAPE(compare_info._image_compare, compare_info.base_info._image)
	assert compare_info.base_info.cas is not None

	alpha, beta = find_alpha_beta(compare_info.base_info._cutout_stamp_maskzeroed,
								  compare_info._cutout_stamp_maskzeroed_compare)
	# print(alpha, beta)
	segmap_dispersion = segmap_color_dispersion(compare_info.base_info.cas.rpetro_circ,
												compare_info.base_info.cas._asymmetry_center,
												compare_info.base_info.nx_stamp,
												compare_info.base_info.ny_stamp)

	image = compare_info.base_info._cutout_stamp_maskzeroed_no_bg[segmap_dispersion]
	image_compare = compare_info._cutout_stamp_maskzeroed_no_bg_compare[segmap_dispersion]

	dispersion_1 = np.sum((image - image_compare * alpha - beta) ** 2)
	dispersion_2 = np.sum((image_compare - beta) ** 2)
	sky_dispersion = _sky_dispersion(compare_info.base_info._cutout_stamp_maskzeroed,
									 compare_info._cutout_stamp_maskzeroed_compare,
									 compare_info.base_info.cas._bkg, compare_info._bkg_compare)

	color_dispersion = (dispersion_1 - sky_dispersion) / (dispersion_2 - sky_dispersion)

	return color_dispersion

cdef CompareInfo calc_color_dispersion(BaseInfo base_info, cnp.ndarray[double, ndim=2] image_compare):
	cdef CompareInfo compare_info = CompareInfo()

	compare_info._image_compare = image_compare
	compare_info.base_info = base_info
	compare_info.num_badpixels = -1

	compare_info._mask_stamp_nan_compare = compare_info.get_mask_stamp_nan_compare()
	compare_info._mask_stamp_badpixels_compare = compare_info.get_mask_stamp_badpixels_compare()
	compare_info._mask_stamp_compare = compare_info.get_mask_stamp_compare()
	compare_info._mask_stamp_no_bg_compare = compare_info.get_mask_stamp_no_bg_compare()

	compare_info._cutout_stamp_maskzeroed_compare = compare_info.get_cutout_stamp_maskzeroed_compare()
	compare_info._cutout_stamp_maskzeroed_no_bg_compare = compare_info.get_cutout_stamp_maskzeroed_no_bg_compare()

	'''
	In this part we get the sky area for the following calculation
	'''
	compare_info._bkg_compare = compare_info._cutout_stamp_maskzeroed_compare[compare_info.base_info.cas._slice_skybox]

	compare_info.color_dispersion = get_color_dispersion(compare_info)

	return compare_info
