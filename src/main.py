#!/bin/python3
import getopt
import logging
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from configparser import ConfigParser
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import photutils
from astropy.io import fits
from astropy.table import Table, Column, join

from morph_provider import MorphProvider, StatmorphVanilla, StatmorphCython
from sextractor import SExtractor

logging.basicConfig(level=logging.INFO,
					format="[%(asctime)s][%(name)s - %(processName)s/%(levelname)s]: %(message)s")
logger = logging.getLogger("Statmorph")


def read_properties(path) -> Dict[str, str]:
	config = ConfigParser()
	s_config = open(path, 'r').read()
	s_config = "[properties]\n%s" % s_config
	config.read_string(s_config)
	items = config.items('properties')
	item_dict = {}
	for key, value in items:
		item_dict[key] = value
	return item_dict


def work_with_shared_memory(shm_img_name: str, shm_segm_name: str, shm_noise_name: Optional[str], segm_slice,
							label: int, shape, shm_img_cmp_name: Optional[str],
							output_image_dir: str, set_centroid: Tuple[float, float],
							set_asym_center: Tuple[float, float], img_dtype, segm_dtype,
							morph_provider: MorphProvider) -> Tuple:
	"""
	Returns:
		label: int
		rpetro_circ_centroid: double
		rpetro_circ: double
		rpetro_ellip: double
		c: double
		a: double
		s: double
		g: double
		m20: double
		size: int
		surface_brightness: double
		runtime: double
		flag: int
	"""
	# logger.info("%s, label=%d" % (current_process().name, label))
	# Locate the shared memory by its name
	shm_img = SharedMemory(shm_img_name)
	shm_segm = SharedMemory(shm_segm_name)
	# Create the np.recarray from the buffer of the shared memory
	image: np.ndarray = np.ndarray(shape, buffer=shm_img.buf, dtype=img_dtype)
	segmap: np.ndarray = np.ndarray(shape, buffer=shm_segm.buf, dtype=segm_dtype)
	noisemap: Optional[np.ndarray] = None
	image_compare: Optional[np.ndarray] = None

	if shm_noise_name is not None:
		shm_noise = SharedMemory(shm_noise_name)
		noisemap = np.ndarray(shape, buffer=shm_noise.buf, dtype=img_dtype)

	if shm_img_cmp_name is not None:
		shm_img_cmp = SharedMemory(shm_img_cmp_name)
		image_compare = np.ndarray(shape, buffer=shm_img_cmp.buf, dtype=img_dtype)

	try:
		return_list = morph_provider.measure_label(image, segmap, noisemap, segm_slice, label,
												   image_compare, output_image_dir, set_centroid, set_asym_center)
	except:
		logger.error(str(label) + ": " + traceback.format_exc())
		return_list = morph_provider.get_empty_result(label)

	del image, segmap, noisemap, image_compare

	return tuple(return_list)


def work_with_individual_file(label: int, flux_file_name: str, flux_hdu_index: int,
							  noise_file_name: Optional[str], noise_hdu_index: Optional[int],
							  mask_file_name: Optional[str], mask_hdu_index: Optional[int],
							  cmp_file_name: Optional[str], cmp_hdu_index: Optional[int],
							  output_image_dir: str, set_centroid: Tuple[float, float],
							  set_asym_center: Tuple[float, float], morph_provider: MorphProvider) -> Tuple:
	"""
	Returns:
		label: int
		rpetro_circ_centroid: double
		rpetro_circ: double
		rpetro_ellip: double
		c: double
		a: double
		s: double
		g: double
		m20: double
		size: int
		surface_brightness: double
		runtime: double
		flag: int
	"""
	# logger.info("%s, label=%d" % (current_process().name, label))
	# Locate the shared memory by its name

	try:
		return_list = morph_provider.measure_individual(label, flux_file_name, flux_hdu_index,
														noise_file_name, noise_hdu_index,
														mask_file_name, mask_hdu_index,
														cmp_file_name, cmp_hdu_index, output_image_dir,
														set_centroid, set_asym_center)
	except:
		logger.error(str(label) + ": " + traceback.format_exc())
		return_list = morph_provider.get_empty_result(label)

	return tuple(return_list)


def run_sextractor(work_dir: str, detect_file: str, wht_file: str, use_existed: bool,
				   measure_file: Optional[str] = None,
				   config_values: Optional[Dict[str, Union[float, int, str]]] = None):
	sextractor = SExtractor(work_dir, SExtractor.merge_sex_dict(SExtractor.CANDELS_UKIDSS_USF_CONFIG,
																SExtractor.CANDELS_UKIDSS_USF_HOT_CONFIG,
																SExtractor.GLASS_JWST_VALUES, config_values),
							SExtractor.OUTPUT_LIST_DEFAULT)
	sextractor.run(detect_file, wht_file, measure_file, use_existed=use_existed)
	return sextractor


def run_statmorph_init_calc_para_str_list(threads: int, calc_cas: bool = True, calc_g_m20: bool = True,
										  calc_shape_asymmetry: bool = False, calc_mid: bool = False, calc_multiplicity: bool = False,
										  calc_color_dispersion: bool = False, calc_g2: bool = False) -> List[str]:
	logger.info("欢迎使用Statmorph, 线程数%d" % threads)

	calc_para_str_list = []
	if calc_cas:
		calc_para_str_list.append("CAS")
	if calc_g_m20:
		calc_para_str_list.append("G_M20")
	if calc_shape_asymmetry:
		calc_para_str_list.append("shape_asymmetry")
	if calc_mid:
		calc_para_str_list.append("MID")
	if calc_multiplicity:
		calc_para_str_list.append("multiplicity")
	if calc_color_dispersion:
		calc_para_str_list.append("color_dispersion(ξ)")
	if calc_g2:
		calc_para_str_list.append("G2")

	logger.info("计算参数: " + ", ".join(calc_para_str_list))
	return calc_para_str_list


def get_center_in_center_table(center_table: Table, label: int, calc_cas: bool) -> Tuple[
	Tuple[float, float], Tuple[float, float]]:
	set_centroid: Tuple[float, float] = (-1, -1)
	set_asym_center: Tuple[float, float] = (-1, -1)
	if center_table is not None:
		center_info = center_table[center_table["label"] == label]
		if len(center_info) > 0:
			center_info = center_info[0]
			if center_info["size"] > 0:
				set_centroid = (center_info["centroid_x"], center_info["centroid_y"])
				if calc_cas:
					set_asym_center = (center_info["asymmetry_center_x"], center_info["asymmetry_center_y"])
			else:
				logger.warning("size of label %d in center_file is zero" % label)
		else:
			logger.warning("label %d not existed in center_file" % label)

	return set_centroid, set_asym_center


def run_statmorph(catalog_file: str, image_file: str, segmap_file: str, noise_file: Optional[str], save_file: str,
				  threads: int, run_percentage: int, run_specified_label: int, ignore_mag_fainter_than: float = 26.0,
				  ignore_class_star_greater_than: float = 0.9, calc_cas: bool = True, calc_g_m20: bool = True,
				  calc_shape_asymmetry: bool = False, calc_mid: bool = False, calc_multiplicity: bool = False,
				  calc_color_dispersion: bool = False, image_compare_file: Optional[str] = None, calc_g2: bool = False,
				  output_image_dir: Optional[str] = None, center_file: Optional[str] = None, use_vanilla: bool = False):
	calc_para_str_list = run_statmorph_init_calc_para_str_list(threads, calc_cas, calc_g_m20, calc_shape_asymmetry, calc_mid,
															   calc_multiplicity, calc_color_dispersion, calc_g2)
	logger.info("进入大图模式")
	sextractor_table: Table = Table.read(catalog_file, format="ascii")
	center_table: Optional[Table] = None
	if center_file is not None:
		logger.info("使用预定义的中心位置文件" + center_file)
		center_table = Table.read(center_file, format="ascii")

	if output_image_dir is not None:
		if not os.path.exists(output_image_dir):
			os.mkdir(output_image_dir)

	# Make a large data frame with date, float and character columns
	image_fits = fits.open(image_file)
	segmap_fits = fits.open(segmap_file)

	image = image_fits[0].data.astype(np.float64)
	segmap = segmap_fits[0].data.astype(np.int32)

	if noise_file is not None:
		noise_fits = fits.open(noise_file)
		noisemap = noise_fits[0].data.astype(np.float64)
	else:
		noisemap = None

	if image_compare_file is not None:
		image_compare_fits = fits.open(image_compare_file)
		image_compare = image_compare_fits[0].data.astype(np.float64)
	else:
		image_compare = None

	# Convert into numpy recarray to preserve the dtypes

	shape, img_dtype = image.shape, image.dtype
	segm_dtype = segmap.dtype

	logger.info(f"image's size={image.nbytes / 1e6}MB")
	logger.info(f"segmap's size={segmap.nbytes / 1e6}MB")
	if image_compare is not None:
		logger.info(f"image_compare's size={image_compare.nbytes / 1e6}MB")

	segm_image = photutils.SegmentationImage(segmap)
	labels: Column = sextractor_table["NUMBER"]
	logger.info("图像数量%d" % len(labels))
	if run_specified_label <= 0:
		labels = labels[(sextractor_table["MAG_AUTO"] <= ignore_mag_fainter_than) & (
				sextractor_table["CLASS_STAR"] <= ignore_class_star_greater_than)]
		logger.info(
			"忽略MAG_AUTO>%.1f和CLASS_STAR>%.1f的label" % (ignore_mag_fainter_than, ignore_class_star_greater_than))
		run_labels = labels[0:len(labels) * run_percentage // 100]
		logger.info("实际运行%d%%，共%d" % (run_percentage, len(run_labels)))
	else:
		run_labels = [run_specified_label]
		logger.info("只运行label=%d" % run_specified_label)

	if use_vanilla:
		morph_provider: MorphProvider = StatmorphVanilla(calc_cas, calc_g_m20, calc_shape_asymmetry, calc_mid, calc_multiplicity,
														 calc_color_dispersion, calc_g2)
	else:
		morph_provider: MorphProvider = StatmorphCython(calc_cas, calc_g_m20, calc_shape_asymmetry, calc_mid, calc_multiplicity,
														calc_color_dispersion, calc_g2)
	logger.info("使用" + morph_provider.__class__.__qualname__)

	result_format = morph_provider.get_result_format()
	result_all = [" ".join(morph_provider.get_result_header()) + "\n"]

	start_time = time.time()
	with SharedMemoryManager() as smm:
		# Create a shared memory of size np_arry.nbytes
		shm_img = smm.SharedMemory(image.nbytes)
		shm_segm = smm.SharedMemory(segmap.nbytes)

		# Create a np.recarray using the buffer of shm
		shm_img_array = np.ndarray(shape, buffer=shm_img.buf, dtype=img_dtype)
		shm_segm_array = np.ndarray(shape, buffer=shm_segm.buf, dtype=segm_dtype)

		if noisemap is not None:
			shm_noise = smm.SharedMemory(noisemap.nbytes)
			shm_noise_array = np.ndarray(shape, buffer=shm_noise.buf, dtype=img_dtype)
			shm_noise_name = shm_noise.name
		else:
			shm_noise_name = None

		if image_compare is not None:
			shm_img_cmp = smm.SharedMemory(image_compare.nbytes)
			shm_img_cmp_array = np.ndarray(shape, buffer=shm_img_cmp.buf, dtype=img_dtype)
			shm_img_cmp_name = shm_img_cmp.name
		else:
			shm_img_cmp_name = None

		# Copy the data into the shared memory
		logger.info("开始向共享内存复制数组")
		np.copyto(shm_img_array, image)
		del image
		logger.info("image复制完成")
		np.copyto(shm_segm_array, segmap)
		del segmap
		logger.info("segmap复制完成")

		if noisemap is not None:
			np.copyto(shm_noise_array, noisemap)
			del noisemap
			logger.info("noisemap复制完成")
		if image_compare is not None:
			np.copyto(shm_img_cmp_array, image_compare)
			del image_compare
			logger.info("image_compare复制完成")

		logger.info("全部复制完成")

		# Spawn some processes to do some work
		with ProcessPoolExecutor(threads) as exe:
			fs = []
			for label in run_labels:
				try:
					label_index = segm_image.get_index(label)
				except ValueError as e:
					logger.warning(e)
					continue
				segm_slice = segm_image.slices[label_index]

				set_centroid, set_asym_center = get_center_in_center_table(center_table, label, calc_cas)

				fs.append(
					exe.submit(work_with_shared_memory, shm_img.name, shm_segm.name, shm_noise_name, segm_slice, label,
							   shape, shm_img_cmp_name, output_image_dir, set_centroid, set_asym_center,
							   img_dtype, segm_dtype, morph_provider))
			for result in as_completed(fs):
				line = result_format % result.result()
				result_all.append(line)

	logger.info(f'用时: {time.time() - start_time:.2f}s')

	logger.info("开始写入文件")

	with open(save_file, "w") as f:
		f.writelines(result_all)

	result_table: Table = Table.read(save_file, format="ascii")
	# result_table.sort("label")
	result_table["NUMBER"] = result_table["label"]
	result_table = join(sextractor_table, result_table)
	result_table.write(save_file, format="ascii", overwrite=True)

	logger.info("文件已保存至" + save_file)


def table_split(table: Table, max_rows: int) -> List[Table]:
	split_num = int(np.ceil(len(table) / max_rows))
	table_list = []
	for i in range(split_num):
		table_list.append(table[i * max_rows:(i + 1) * max_rows])
	return table_list


def run_statmorph_stamp(catalog_file: str, save_file: str, threads: int, run_percentage: int, run_specified_label: int,
						calc_cas: bool = True, calc_g_m20: bool = True, calc_shape_asymmetry: bool = False,
						calc_mid: bool = False, calc_multiplicity: bool = False, calc_color_dispersion: bool = False,
						calc_g2: bool = False, output_image_dir: Optional[str] = None,
						center_file: Optional[str] = None, use_vanilla: bool = False):
	calc_para_str_list = run_statmorph_init_calc_para_str_list(threads, calc_cas, calc_g_m20, calc_shape_asymmetry, calc_mid,
															   calc_multiplicity, calc_color_dispersion, calc_g2)
	logger.info("进入独立stamp模式")
	catalog_table = Table.read(catalog_file, format="ascii")
	center_table: Optional[Table] = None
	if center_file is not None:
		logger.info("使用预定义的中心位置文件" + center_file)
		center_table = Table.read(center_file, format="ascii")

	if output_image_dir is not None:
		if not os.path.exists(output_image_dir):
			os.mkdir(output_image_dir)

	logger.info("图像数量%d" % len(catalog_table))
	if run_specified_label <= 0:
		run_rows: Table = catalog_table[0:len(catalog_table) * run_percentage // 100]
		logger.info("实际运行%d%%，共%d" % (run_percentage, len(catalog_table)))
	else:
		run_rows = catalog_table[catalog_table["label"] == run_specified_label]
		logger.info("只运行label=%d" % run_specified_label)

	if use_vanilla:
		morph_provider: MorphProvider = StatmorphVanilla(calc_cas, calc_g_m20, calc_shape_asymmetry, calc_mid, calc_multiplicity,
														 calc_color_dispersion, calc_g2)
	else:
		morph_provider: MorphProvider = StatmorphCython(calc_cas, calc_g_m20, calc_shape_asymmetry, calc_mid, calc_multiplicity,
														calc_color_dispersion, calc_g2)
	logger.info("使用" + morph_provider.__class__.__qualname__)

	result_format = morph_provider.get_result_format()
	result_all = [" ".join(morph_provider.get_result_header()) + "\n"]

	start_time = time.time()

	if "image_hdu_index" not in run_rows.colnames:
		run_rows["image_hdu_index"] = 0
	if "noise_hdu_index" not in run_rows.colnames:
		run_rows["noise_hdu_index"] = 0
	if "mask_hdu_index" not in run_rows.colnames:
		run_rows["mask_hdu_index"] = 0
	if "cmp_hdu_index" not in run_rows.colnames:
		run_rows["cmp_hdu_index"] = 0

	if "noise_file_name" not in run_rows.colnames:
		run_rows["noise_file_name"] = None
	if "mask_file_name" not in run_rows.colnames:
		run_rows["mask_file_name"] = None
	if "cmp_file_name" not in run_rows.colnames:
		run_rows["cmp_file_name"] = None

	if threads > 1:
		run_rows_list = table_split(run_rows, 20000)
		for block_i in range(len(run_rows_list)):
			run_rows_block = run_rows_list[block_i]
			logger.info("表过长(%d>20000)，分段运行，当前段数%d/%d" % (len(run_rows), block_i + 1, len(run_rows_list)))
			with ProcessPoolExecutor(threads) as executor:
				set_centroid_list = []
				set_asym_center_list = []
				for row in run_rows_block:
					set_centroid, set_asym_center = get_center_in_center_table(center_table, row["label"], calc_cas)
					set_centroid_list.append(set_centroid)
					set_asym_center_list.append(set_asym_center)

				output_image_dir_list = np.repeat(output_image_dir, len(run_rows_block))
				morph_provider_list = np.repeat(morph_provider, len(run_rows_block))

				result_iter = executor.map(work_with_individual_file, run_rows_block["label"],
										   run_rows_block["image_file_name"], run_rows_block["image_hdu_index"],
										   run_rows_block["noise_file_name"], run_rows_block["noise_hdu_index"],
										   run_rows_block["mask_file_name"], run_rows_block["mask_hdu_index"],
										   run_rows_block["cmp_file_name"], run_rows_block["cmp_hdu_index"],
										   output_image_dir_list, set_centroid_list,
										   set_asym_center_list, morph_provider_list
										   )
				result_all_block = [result_format % r for r in result_iter]
			result_all = result_all + result_all_block

	else:
		for row in run_rows:
			label = row["label"]
			set_centroid, set_asym_center = get_center_in_center_table(center_table, label, calc_cas)
			result = work_with_individual_file(label,
											   row["image_file_name"], row["image_hdu_index"],
											   row["noise_file_name"], row["noise_hdu_index"],
											   row["mask_file_name"], row["mask_hdu_index"],
											   row["cmp_file_name"], row["cmp_hdu_index"],
											   output_image_dir, set_centroid,
											   set_asym_center, morph_provider
											   )
			result_all.append(result)

	logger.info(f'用时: {time.time() - start_time:.2f}s')

	logger.info("开始写入文件")

	with open(save_file, "w") as f:
		f.writelines(result_all)

	result_table: Table = Table.read(save_file, format="ascii")
	# result_table.sort("label")
	result_table = join(catalog_table, result_table)
	result_table.write(save_file, format="ascii", overwrite=True)

	logger.info("文件已保存至" + save_file)


def opts_to_dict(opts: List[Tuple[str, str]], arg_short_dict: Dict[str, Tuple[str, bool]],
				 opt_dict: Optional[Dict[str, str]] = None) -> Dict[str, str]:
	if opt_dict is None:
		opt_dict = dict()
	for opt, arg in opts:
		if opt.startswith("--"):
			opt_dict[opt[2:]] = arg
		else:
			opt_short_char = opt[1]
			opt_dict[arg_short_dict[opt_short_char][0]] = arg
	return opt_dict


def gen_opts(arg_short_dict: Dict[str, Tuple[str, bool]]):
	shortopts: str = ""
	longopts: List[str] = []
	for shortopt, v in arg_short_dict.items():
		shortopts = shortopts + shortopt
		longopt = v[0]
		if v[1]:
			shortopts = shortopts + ":"
			longopt = longopt + "="
		longopts.append(longopt)
	return shortopts, longopts


def check_not_false(value) -> bool:
	return value not in ("false", "False", "FALSE", False)


def check_not_null(value) -> bool:
	return value not in ("null", "NULL", "None", "NONE", "", None)


def get_basename_without_end(path) -> str:
	bn = os.path.basename(path)
	if bn.endswith(".fits.gz"):
		bn = bn[0:-8]
	elif bn.endswith(".fits"):
		bn = bn[0:-5]
	return bn


help_str = """SExtractor-Statmorph_csst 简化合并版使用说明

	-j, --threads=并行进程数量，若为0则为CPU核心数量-1(若为单核则为1)
	-i, --image_file=原始图像文件(未扣除背景)，双图像模式中指用来探测源的图像文件(深度越深，PSF越大越好)，若跳过SExtractor可以不需要
	-y, --measure_file=双图像模式中用于测量的图像文件(未扣除背景)，若不指定(为null)则为单图像模式，若跳过SExtractor可以不需要
	-w, --wht_file=权重图像文件，若跳过SExtractor可以不需要
	-o, --save_file=形态学参数输出文件名，若不指定则默认为image_file的文件名(不包括后缀).txt，双图像模式则还包括measure_file
	-p, --run_percentage=运行全部源数量的百分比，100表示全部运行
	-l, --run_specified_label=仅运行指定编号的源，若为0则运行全部源
	-s, --sextractor_work_dir=SExtractor的输出文件存放文件夹，若不指定则默认为image_file的文件名(不包括后缀)，双图像模式下默认文件名还会包括measure_file，如果跳过sextractor，那么必须指定该项
	-k, --skip_sextractor 是否直接利用SExtractor已经生成的结果，SExtractor的输出文件夹必须包含subback.fits,segmap.fits,noise.fits三个图像文件和catalog.txt星表文件
	-D, --sextractor_detect_minarea
	-T, --sextractor_detect_thresh
	-A, --sextractor_analysis_thresh
	-B, --sextractor_deblend_nthresh
	-M, --sextractor_deblend_mincont
	-S, --sextractor_back_size
	-F, --sextractor_back_filtersize
	-P, --sextractor_backphoto_thick
	-r, --stamp_catalog 如果填写则进入stamp模式，每个星系具有独立的stamp的fits文件，而不是从segmap中创建，stamp_catalog文件必须包含id，image_file_name，image_hdu_index，(noise_file_name，noise_hdu_index，cmp_file_name，cmp_hdu_index)列，如果不指定hdu_index，则默认为0。指定该项后，image_file、measure_file、wht_file、sextractor_work_dir、skip_sextractor将全部失效。
	-a, --output_image_dir=输出示意图的文件夹，若为null则不输出示意图
	-f, --ignore_mag_fainter_than=忽略测量视星等比该星等更高的源
	-t, --ignore_class_star_greater_than=忽略测量像恒星指数大于该值的源
	-n, --center_file=预定义的星系中心文件，用于取代星系质心和不对称中心
	-c, --calc_cas 是否测量CAS
	-g, --calc_g_m20 是否测量Gini,M20
	-x, --calc_shape_asymmetry 是否测量shape_asymmetry，依赖CAS和G_M20
	-d, --calc_mid 是否测量MID
	-u, --calc_multiplicity 是否测量multiplicity
	-e, --calc_color_dispersion 是否测量color_dispersion
	-m, --image_compare_file 测量color_dispersion中用于比较的图像(已经扣除了背景)，若不测量则为null
	-b, --calc_g2 是否测量G2
	-v, --use_vanilla 是否使用vanilla版
	-h, --help 显示此帮助
"""


def main(argv) -> int:
	config: Dict = read_properties("./config.properties")
	config["help"] = False

	arg_short_dict = {
		"h": ("help", False),
		"j": ("threads", True),
		"i": ("image_file", True),
		"y": ("measure_file", True),
		"w": ("wht_file", True),
		"o": ("save_file", True),
		"p": ("run_percentage", True),
		"l": ("run_specified_label", True),
		# "r": ("simplified_rot_threshold", True),
		# "m": ("fmin_maxiter", True),
		"s": ("sextractor_work_dir", True),
		"k": ("skip_sextractor", False),
		"D": ("sextractor_detect_minarea", True),
		"T": ("sextractor_detect_thresh", True),
		"A": ("sextractor_analysis_thresh", True),
		"B": ("sextractor_deblend_nthresh", True),
		"M": ("sextractor_deblend_mincont", True),
		"S": ("sextractor_back_size", True),
		"F": ("sextractor_back_filtersize", True),
		"P": ("sextractor_backphoto_thick", True),
		"r": ("stamp_catalog", True),
		"a": ("output_image_dir", True),
		"f": ("ignore_mag_fainter_than", True),
		"t": ("ignore_class_star_greater_than", True),
		"n": ("center_file", True),
		"c": ("calc_cas", False),
		"g": ("calc_g_m20", False),
		"x": ("calc_shape_asymmetry", False),
		"d": ("calc_mid", False),
		"u": ("calc_multiplicity", False),
		"e": ("calc_color_dispersion", False),
		"m": ("image_compare_file", True),
		"b": ("calc_g2", False),
		"v": ("use_vanilla", False)
	}

	try:
		opts, _other_args = getopt.getopt(argv[1:], *gen_opts(arg_short_dict))
	except getopt.GetoptError as e:
		print("-h 查看帮助信息")
		return 1

	opts_to_dict(opts, arg_short_dict, config)

	if check_not_false(config["help"]):
		print(help_str)
		return 0

	threads = int(config["threads"])
	detect_file: str = config["image_file"]
	measure_file: Optional[str] = config["measure_file"]
	wht_file: str = config["wht_file"]
	save_file: str = config["save_file"]
	run_percentage: int = int(config["run_percentage"])
	run_specified_label: int = int(config["run_specified_label"])
	# simplified_rot_threshold = int(config["simplified_rot_threshold"])
	# fmin_maxiter = int(config["fmin_maxiter"])
	sextractor_work_dir: str = config["sextractor_work_dir"]
	skip_sextractor: bool = check_not_false(config["skip_sextractor"])

	my_sextractor_config: Dict[str, Union[float, int, str]] = dict()
	my_sextractor_config["detect_minarea"] = int(config["sextractor_detect_minarea"])
	my_sextractor_config["detect_thresh"] = float(config["sextractor_detect_thresh"])
	my_sextractor_config["analysis_thresh"] = float(config["sextractor_analysis_thresh"])
	my_sextractor_config["deblend_nthresh"] = int(config["sextractor_deblend_nthresh"])
	my_sextractor_config["deblend_mincont"] = float(config["sextractor_deblend_mincont"])
	my_sextractor_config["back_size"] = int(config["sextractor_back_size"])
	my_sextractor_config["back_filtersize"] = int(config["sextractor_back_filtersize"])
	my_sextractor_config["backphoto_thick"] = int(config["sextractor_backphoto_thick"])

	stamp_catalog: Optional[str] = config["stamp_catalog"]

	output_image_dir: Optional[str] = config["output_image_dir"]
	ignore_mag_fainter_than: float = float(config["ignore_mag_fainter_than"])
	ignore_class_star_greater_than: float = float(config["ignore_class_star_greater_than"])
	calc_cas: bool = check_not_false(config["calc_cas"])
	calc_g_m20: bool = check_not_false(config["calc_g_m20"])
	calc_shape_asymmetry: bool = check_not_false(config["calc_shape_asymmetry"])
	calc_mid: bool = check_not_false(config["calc_mid"])
	calc_multiplicity: bool = check_not_false(config["calc_multiplicity"])
	calc_color_dispersion: bool = check_not_false(config["calc_color_dispersion"])
	image_compare_file: Optional[str] = config["image_compare_file"]
	center_file: Optional[str] = config["center_file"]
	calc_g2: bool = check_not_false(config["calc_g2"])
	use_vanilla: bool = check_not_false(config["use_vanilla"])

	if not check_not_null(measure_file):
		measure_file = None
	if not check_not_null(output_image_dir):
		output_image_dir = None
	if not check_not_null(image_compare_file):
		image_compare_file = None
	if not check_not_null(center_file):
		center_file = None
	if not check_not_null(stamp_catalog):
		stamp_catalog = None
	if calc_shape_asymmetry:
		if not calc_cas:
			calc_cas = True
			logger.warning("shape_asymmetry依赖CAS，因此添加CAS的计算!")
		if not calc_g_m20:
			calc_g_m20 = True
			logger.warning("shape_asymmetry依赖G_M20，因此添加G_M20的计算!")

	run_name = get_basename_without_end(detect_file)
	if measure_file is not None:
		run_name = run_name + "_" + get_basename_without_end(measure_file)

	if not check_not_null(sextractor_work_dir):
		sextractor_work_dir = "sextractor_" + run_name
	if not check_not_null(save_file):
		save_file = "statmorph_" + run_name + ".txt"

	if threads <= 0:
		threads = min(multiprocessing.cpu_count() - 1, 1)

	if stamp_catalog is None:
		sextractor = run_sextractor(sextractor_work_dir, detect_file, wht_file, skip_sextractor, measure_file,
									my_sextractor_config)
		run_statmorph(sextractor.output_catalog_file, sextractor.output_subback_file, sextractor.output_segmap_file,
					  sextractor.noise_file,
					  save_file, threads, run_percentage, run_specified_label, ignore_mag_fainter_than,
					  ignore_class_star_greater_than, calc_cas, calc_g_m20, calc_shape_asymmetry, calc_mid,
					  calc_multiplicity, calc_color_dispersion, image_compare_file, calc_g2, output_image_dir,
					  center_file,
					  use_vanilla)
	else:
		run_statmorph_stamp(stamp_catalog, save_file, threads, run_percentage, run_specified_label, calc_cas,
							calc_g_m20, calc_shape_asymmetry, calc_mid, calc_multiplicity, calc_color_dispersion,
							calc_g2, output_image_dir, center_file, use_vanilla)

	return 0


if __name__ == "__main__":
	exit(main(sys.argv))
