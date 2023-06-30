from abc import ABC

import numpy as np
from scipy.interpolate import interp1d


class EmpiricalEstimator(ABC):
	pass


class Lotz04Estimator(EmpiricalEstimator):
	sn_per_pixel_list = np.array([2, 5])

	c_sys_err = np.array([-0.3, -0.1])
	a_sys_err = np.array([-0.3, -0.1])
	s_sys_err = np.array([-0.4, -0.2])
	g_sys_err = np.array([-0.05, -0.01])
	m20_sys_err = np.array([0.2, 0.02])

	c_sys_error_calc = interp1d(sn_per_pixel_list, c_sys_err)
	a_sys_error_calc = interp1d(sn_per_pixel_list, a_sys_err)
	s_sys_error_calc = interp1d(sn_per_pixel_list, s_sys_err)
	g_sys_error_calc = interp1d(sn_per_pixel_list, g_sys_err)
	m20_sys_error_calc = interp1d(sn_per_pixel_list, m20_sys_err)


class Lotz06Estimator(EmpiricalEstimator):
	sn_per_pixel_list = np.array([2, 3, 4, 5, 6])

	c_sys_err = np.array([-0.29, -0.215, -0.215, -0.185, 0.01])
	c_rand_err = np.array([0.575, 0.445, 0.44, 0.435, 0.26])

	g_sys_err = np.array([-0.0355, -0.0181, -0.0148, -0.0105, 0.0055])
	g_rand_err = np.array([0.0574, 0.0329, 0.0291, 0.0275, 0.0228])

	m20_sys_err = np.array([0.086, 0.061, 0.046, 0.034, 0.026])
	m20_rand_err = np.array([0.172, 0.1, 0.1, 0.067, 0.066])

	c_sys_error_calc = interp1d(sn_per_pixel_list, c_sys_err)
	c_rand_error_calc = interp1d(sn_per_pixel_list, c_rand_err)
	g_sys_error_calc = interp1d(sn_per_pixel_list, g_sys_err)
	g_rand_error_calc = interp1d(sn_per_pixel_list, g_rand_err)
	m20_sys_error_calc = interp1d(sn_per_pixel_list, m20_sys_err)
	m20_rand_error_calc = interp1d(sn_per_pixel_list, m20_rand_err)
