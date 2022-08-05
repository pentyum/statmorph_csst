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
"""
import numpy as np
import scipy.optimize as opt

cimport numpy as cnp

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