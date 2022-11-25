import logging
import multiprocessing
from typing import Sequence, Tuple, Type, Iterable, List, Union, Optional

import cv2
import numpy as np
from astropy.convolution import convolve
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.wcs import WCS
import astropy.units as u
from photutils import HanningWindow, CosineBellWindow, TukeyWindow, TopHatWindow, create_matching_kernel
import scipy.optimize as opt
from reproject import reproject_exact

from utils.check_psfmatch import measure_kernel

convolve_logger = logging.getLogger("Convolver")
resampler_logger = logging.getLogger("Resampler")


def psf_reshape(psf: np.ndarray, new_shape: Sequence[int]) -> np.ndarray:
	psf = psf / np.sum(psf)
	new_psf: np.ndarray = cv2.resize(psf, new_shape, interpolation=cv2.INTER_LINEAR_EXACT)
	return new_psf / np.sum(new_psf)


def get_size_matched_psf_from_file(origin_fits_file: str, psf_file: str) -> np.ndarray:
	origin_fits = fits.open(origin_fits_file)
	psf_fits = fits.open(psf_file)
	psf_pixel_length: float = psf_fits[0].header["PIXSCL"]
	origin_wcs = WCS(origin_fits[0].header)
	origin_pixel_length: float = np.sqrt(origin_wcs.proj_plane_pixel_area).to(u.arcsec).value

	psf: np.ndarray = psf_fits[0].data
	if abs(psf_pixel_length - origin_pixel_length) > 1e-5:
		resampler_logger.info("%s的每个像素的长度为%g, 而PSF文件%s的像素长度为%g，需要对PSF进行缩放" % (
			origin_fits_file, origin_pixel_length, psf_file, psf_pixel_length))
		new_shape: np.ndarray = (np.array(psf.shape) * psf_pixel_length / origin_pixel_length).astype(np.int32)
		if (new_shape[0] % 2 == 0) and (new_shape[1] % 2 == 0):
			new_shape = new_shape + 1
		psf = psf_reshape(psf, new_shape)

	origin_fits.close()
	psf_fits.close()
	return psf


def test_value(D: float, W_minus: float) -> float:
	return 2000 * D ** 5 + 1 / 3 * W_minus ** 3


def get_d(p: float, window_type: Type, origin_psf: np.ndarray, target_psf: np.ndarray) -> float:
	if window_type != HanningWindow:
		window = window_type(p)
	else:
		window = HanningWindow()

	kernel = create_matching_kernel(origin_psf, target_psf, window=window)

	# D<0.1, W~0.2

	D, W_minus = measure_kernel(target_psf, convolve(origin_psf, kernel), kernel)

	return test_value(D, W_minus)


def get_best_kernel(origin_psf: np.ndarray, target_psf: np.ndarray) -> Tuple[
	np.ndarray, Type, float, float, float, float]:
	global min_x
	opt_result = dict()
	opt_result[CosineBellWindow] = opt.minimize_scalar(get_d, args=(CosineBellWindow, origin_psf, target_psf))
	opt_result[TukeyWindow] = opt.minimize_scalar(get_d, args=(TukeyWindow, origin_psf, target_psf))
	opt_result[TopHatWindow] = opt.minimize_scalar(get_d, args=(TopHatWindow, origin_psf, target_psf))
	opt_result[HanningWindow] = opt.OptimizeResult(x=0, fun=get_d(0, HanningWindow, origin_psf, target_psf))

	min_fun: float = np.inf
	for window_type, r in opt_result.items():
		if r.fun < min_fun:
			min_fun = r.fun
			min_x = r.x
			min_window_type = window_type
			min_window = window_type(r.x)
	if np.isinf(min_fun):
		raise ValueError("无法找到最低值")

	kernel = create_matching_kernel(origin_psf, target_psf, window=min_window)
	D, W_minus = measure_kernel(target_psf, convolve(origin_psf, kernel), kernel)

	return kernel, min_window_type, min_x, min_fun, D, W_minus


def get_best_kernel_from_file(origin_file, origin_psf_file, target_psf_file):
	origin_psf = get_size_matched_psf_from_file(origin_file, origin_psf_file)
	target_psf = get_size_matched_psf_from_file(origin_file, target_psf_file)
	kernel, min_window_type, min_x, min_fun, D, W_minus = get_best_kernel(origin_psf, target_psf)

	convolve_logger.info("寻找到的%s到%s的最佳匹配参数为(基于%s的像素大小):" % (origin_psf_file, target_psf_file, origin_file))
	convolve_logger.info("\twindows_type: %s" % min_window_type.__name__)
	convolve_logger.info("\tx: %.3f" % min_x)
	convolve_logger.info("\tfun: %.3f" % min_fun)
	convolve_logger.info("\tD: %.3f" % D)
	convolve_logger.info("\tW_minus: %.3f" % W_minus)
	return kernel, min_window_type, min_x, min_fun, D, W_minus


def get_multiband_kernel_list(origin_file_list: Iterable[str], origin_psf_file_list: Iterable[str],
							  target_psf_file: str) -> List[np.ndarray]:
	result_list = []
	pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)

	for origin_file, origin_psf_file in zip(origin_file_list, origin_psf_file_list):
		result_list.append(pool.apply_async(get_best_kernel_from_file, (origin_file, origin_psf_file, target_psf_file)))

	pool.close()
	pool.join()

	return [result.get()[0] for result in result_list]


def do_convolve(origin_fits_file: str, save_fits_file: str, kernel) -> str:
	with fits.open(origin_fits_file) as fits_file:
		cov_result = convolve(fits_file[0].data, kernel)
		hdu = fits.PrimaryHDU(cov_result, header=fits_file[0].header)
		hdu.writeto(save_fits_file, overwrite=True)
	return save_fits_file


def do_convolve_wht(origin_fits_file: str, save_fits_file: str, kernel) -> str:
	with fits.open(origin_fits_file) as fits_file:
		wht_inv = 1 / fits_file[0].data
		wht_inv = np.where(np.isfinite(wht_inv), wht_inv, np.nan)
		kernel_square = kernel ** 2
		kernel_square = kernel_square / np.sum(kernel_square)
		cov_result = convolve(wht_inv, kernel_square, normalize_kernel=False)
		hdu = fits.PrimaryHDU(1 / cov_result, header=fits_file[0].header)
		hdu.writeto(save_fits_file, overwrite=True)
	return save_fits_file


def multiband_convolve(file_list: Iterable[str], kernel_list: Iterable[str], output_file_list: Iterable[str],
					   is_wht: Union[bool, Iterable[bool]], threads=0) -> List[str]:
	if threads == 0:
		threads = multiprocessing.cpu_count() - 1
	pool = multiprocessing.Pool(threads)
	result_list = []
	if isinstance(is_wht, bool):
		is_wht_list = np.full_like(file_list, is_wht)
	else:
		is_wht_list = is_wht

	for file_name, kernel, output_file_name, wht in zip(file_list, kernel_list, output_file_list, is_wht_list):
		if wht:
			result_list.append(pool.apply_async(do_convolve_wht, (file_name, output_file_name, kernel)))
		else:
			result_list.append(pool.apply_async(do_convolve, (file_name, output_file_name, kernel)))

	pool.close()
	pool.join()
	return [result.get() for result in result_list]


def reproject(data: np.ndarray, wcs: WCS, target_wcs: WCS, shape: Optional[Sequence[int]] = None) -> np.ndarray:
	if shape is None:
		shape = target_wcs.pixel_shape[::-1]
	array, footprint = reproject_exact((data, wcs), target_wcs, shape)
	area_ratio = target_wcs.proj_plane_pixel_area() / wcs.proj_plane_pixel_area()
	return array * area_ratio


def reproject_file(file_name: str, target_wcs: WCS, target_shape: Optional[Sequence[int]], output_file: str,
				   inv: bool = False) -> str:
	assert target_wcs is not None
	assert target_shape is not None
	ccddata = CCDData.read(file_name)
	if inv:
		data = 1 / ccddata.data
	else:
		data = ccddata.data
	reprojected_image = reproject(data, ccddata.wcs, target_wcs, target_shape)
	if inv:
		reprojected_image = 1 / reprojected_image
	ccddata.data = reprojected_image.value.astype(np.float32)
	ccddata.wcs = target_wcs
	ccddata.write(output_file, overwrite=True)
	return output_file
