import getopt
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from configparser import ConfigParser
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Optional, Tuple, List

import numpy as np
import photutils
import statmorph_cython.statmorph as statmorph
from astropy.io import fits
from astropy.table import Table, Column, join

from sextractor import SExtractor

logging.basicConfig(level=logging.INFO,
					format="[%(asctime)s][%(name)s/%(levelname)s]: %(message)s")
logger = logging.getLogger("Statmorph")


def read_properties(path) -> dict:
	config = ConfigParser()
	s_config = open(path, 'r').read()
	s_config = "[properties]\n%s" % s_config
	config.read_string(s_config)
	items = config.items('properties')
	itemDict = {}
	for key, value in items:
		itemDict[key] = value
	return itemDict


def work_with_shared_memory(shm_img_name, shm_segm_name, segm_slice, label: int, shape,
							calc_cas: bool, calc_g_m20: bool, calc_mid: bool, calc_multiply: bool,
							calc_color_dispersion: bool, shm_img_cmp_name, calc_g2: bool,
							output_image_dir: str, img_dtype, segm_dtype):
	"""
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
	image = np.frombuffer(shm_img.buf, dtype=img_dtype).reshape(shape)
	segmap = np.frombuffer(shm_segm.buf, dtype=segm_dtype).reshape(shape)

	if shm_img_cmp_name is not None:
		shm_img_cmp = SharedMemory(shm_img_cmp_name)
		image_compare = np.frombuffer(shm_img_cmp.buf, dtype=img_dtype).reshape(shape)
	else:
		image_compare = None

	morph = statmorph.BaseInfo(
		image, segmap, segm_slice, label, calc_cas=calc_cas, calc_g_m20=calc_g_m20, calc_mid=calc_mid,
		calc_multiply=calc_multiply, calc_color_dispersion=calc_color_dispersion, image_compare=image_compare,
		calc_g2=calc_g2, output_image_dir=output_image_dir)

	return_list = [label, morph.size, morph.surface_brightness, morph._rpetro_circ_centroid]
	if calc_cas:
		return_list.extend([morph.cas.rpetro_circ, morph.cas.concentration, morph.cas.asymmetry, morph.cas.smoothness])
	if calc_g_m20:
		return_list.extend([morph.g_m20.rpetro_ellip, morph.g_m20.gini, morph.g_m20.m20])
	if calc_mid:
		return_list.extend([morph.mid.multimode, morph.mid.intensity, morph.mid.deviation])
		if calc_multiply:
			return_list.extend([morph.multiply])
	if calc_color_dispersion:
		return_list.extend([morph.compare_info.color_dispersion])
	if calc_g2:
		return_list.extend([morph.g2.result_g2])

	return_list.extend([morph.runtime, morph.flags.value()])
	return tuple(return_list)


def run_sextractor(work_dir: str, image_file: str, wht_file: str, use_existed: bool):
	sextractor = SExtractor(work_dir, SExtractor.BRIGHT_VALUES, SExtractor.OUTPUT_LIST_DEFAULT)
	sextractor.run(image_file, wht_file, use_existed=use_existed)
	return sextractor


def run_statmorph(catalog_file: str, image_file: str, segmap_file: str, save_file: str, run_percentage: int,
				  run_specified_label: int, ignore_mag_fainter_than: float = 26.0,
				  ignore_class_star_greater_than: float = 0.9, calc_cas: bool = True, calc_g_m20: bool = True,
				  calc_mid: bool = False, calc_multiply: bool = False, calc_color_dispersion: bool = False,
				  image_compare_file: Optional[str] = None, calc_g2: bool = False, output_image_dir: Optional[str] = None):
	logger.info("欢迎使用Statmorph, 线程数%d" % threads)
	sextractor_table: Table = Table.read(catalog_file, format="ascii")

	if output_image_dir is not None:
		if not os.path.exists(output_image_dir):
			os.mkdir(output_image_dir)

	# Make a large data frame with date, float and character columns
	image_fits = fits.open(image_file)
	segmap_fits = fits.open(segmap_file)

	image = image_fits[0].data.astype(np.float64)
	segmap = segmap_fits[0].data.astype(np.int32)

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

	result_header = ["label", "size", "surface_brightness", "rp_circ_centroid"]
	result_format = ["%d %d %f %f"]

	if calc_cas:
		result_header.extend(["rp_circ", "C", "A", "S"])
		result_format.extend(["%f", "%f", "%f", "%f"])
		logger.info("计算CAS")
	if calc_g_m20:
		result_header.extend(["rp_ellip", "G", "M20"])
		result_format.extend(["%f", "%f", "%f"])
		logger.info("计算G,M20")
	if calc_mid:
		result_header.extend(["M", "I", "D"])
		result_format.extend(["%f", "%f", "%f"])
		logger.info("计算MID")
		if calc_multiply:
			result_header.extend(["multiply"])
			result_format.extend(["%f"])
	if calc_color_dispersion:
		result_header.extend(["color_dispersion"])
		result_format.extend(["%f"])
	if calc_g2:
		result_header.extend(["g2"])
		result_format.extend(["%f"])

	result_header.extend(["runtime", "flag"])
	result_format.extend(["%f", "%d"])

	result_all = [" ".join(result_header) + "\n"]
	result_format = " ".join(result_format) + "\n"

	# result_all = ["label rp_circ_centroid rp_circ rp_ellip C A S G M20 size surface_brightness runtime flag\n"]

	start_time = time.time()
	with SharedMemoryManager() as smm:
		# Create a shared memory of size np_arry.nbytes
		shm_img = smm.SharedMemory(image.nbytes)
		shm_segm = smm.SharedMemory(segmap.nbytes)

		# Create a np.recarray using the buffer of shm
		shm_img_array = np.frombuffer(
			shm_img.buf, dtype=img_dtype).reshape(shape)
		shm_segm_array = np.frombuffer(
			shm_segm.buf, dtype=segm_dtype).reshape(shape)
		if image_compare is not None:
			shm_img_cmp = smm.SharedMemory(image_compare.nbytes)
			shm_img_cmp_array = np.frombuffer(
				shm_img_cmp.buf, dtype=img_dtype).reshape(shape)
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
		if image_compare is not None:
			np.copyto(shm_img_cmp_array, image_compare)
			del image_compare
			logger.info("image_compare复制完成")
		logger.info("复制完成")

		# Spawn some processes to do some work
		with ProcessPoolExecutor(threads) as exe:
			fs = [exe.submit(work_with_shared_memory, shm_img.name, shm_segm.name,
							 segm_image.slices[segm_image.get_index(label)], label, shape,
							 calc_cas, calc_g_m20, calc_mid, calc_multiply, calc_color_dispersion, shm_img_cmp_name,
							 calc_g2, output_image_dir, img_dtype, segm_dtype)
				  for label in run_labels]
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
	for shortopt, v in enumerate(arg_short_dict):
		shortopts = shortopts + shortopt
		longopt = v[0]
		if v[1]:
			shortopts = shortopts + ":"
			longopt = longopt+"="
		longopts.append(longopt)
	return shortopts, longopts


if __name__ == "__main__":
	config: Dict = read_properties("./config.properties")

	arg_short_dict = {
		"j": ("threads", True),
		"i": ("image_file", True),
		"w": ("wht_file", True),
		"o": ("save_file", True),
		"p": ("run_percentage", True),
		"l": ("run_specified_label", True),
		# "r": ("simplified_rot_threshold", True),
		# "m": ("fmin_maxiter", True),
		"s": ("sextractor_work_dir", True),
		"k": ("skip_sextractor", False),
		"a": ("output_image_dir", True),
		"f": ("ignore_mag_fainter_than", True),
		"t": ("ignore_class_star_greater_than", True),
		"c": ("calc_cas", False),
		"g": ("calc_g_m20", False),
		"d": ("calc_mid", False),
		"u": ("calc_multiply", False),
		"e": ("calc_color_dispersion", False),
		"b": ("calc_g2", False)
	}

	opts, _other_args = getopt.getopt(sys.argv[1:], *gen_opts(arg_short_dict))

	opts_to_dict(opts, arg_short_dict, config)

	threads = int(config["threads"])
	image_file: str = config["image_file"]
	wht_file: str = config["wht_file"]
	save_file: str = config["save_file"]
	run_percentage: int = int(config["run_percentage"])
	run_specified_label: int = int(config["run_specified_label"])
	# simplified_rot_threshold = int(config["simplified_rot_threshold"])
	# fmin_maxiter = int(config["fmin_maxiter"])
	sextractor_work_dir: str = config["sextractor_work_dir"]
	skip_sextractor: bool = config["skip_sextractor"] not in ("false", "False")
	output_image_dir: Optional[str] = config["output_image_dir"]
	ignore_mag_fainter_than: float = float(config["ignore_mag_fainter_than"])
	ignore_class_star_greater_than: float = float(config["ignore_class_star_greater_than"])
	calc_cas: bool = config["calc_cas"] not in ("false", "False")
	calc_g_m20: bool = config["calc_g_m20"] not in ("false", "False")
	calc_mid: bool = config["calc_mid"] not in ("false", "False")
	calc_multiply: bool = config["calc_multiply"] not in ("false", "False")
	calc_color_dispersion: bool = config["calc_color_dispersion"] not in ("false", "False")
	image_compare_file: Optional[str] = config["image_compare_file"]
	calc_g2: bool = config["calc_g2"] not in ("false", "False")

	if output_image_dir in ("null", "NULL", "None", ""):
		output_image_dir = None
	if image_compare_file in ("null", "NULL", "None", ""):
		image_compare_file = None

	sextractor = run_sextractor(sextractor_work_dir, image_file, wht_file, skip_sextractor)
	run_statmorph(sextractor.output_catalog_file, sextractor.output_subback_file, sextractor.output_segmap_file,
				  save_file, run_percentage, run_specified_label, ignore_mag_fainter_than,
				  ignore_class_star_greater_than, calc_cas, calc_g_m20, calc_mid,
				  calc_multiply, calc_color_dispersion, image_compare_file, calc_g2, output_image_dir)
