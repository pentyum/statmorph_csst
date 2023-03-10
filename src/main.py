#!/bin/python3
import abc
import getopt
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from configparser import ConfigParser
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import photutils
import statmorph_cython.statmorph as statmorph
from statmorph_cython.statmorph_vanilla import SourceMorphology
from statmorph_cython.statmorph import CASInfo, GiniM20Info, MIDInfo, CompareInfo, G2Info
from astropy.io import fits
from astropy.table import Table, Column, join

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


class MorphProvider(abc.ABC):
	def __init__(self, calc_cas, calc_g_m20, calc_mid, calc_multiplicity, calc_color_dispersion, calc_g2):
		self.calc_cas: bool = calc_cas
		self.calc_g_m20: bool = calc_g_m20
		self.calc_mid: bool = calc_mid
		self.calc_multiplicity: bool = calc_multiplicity
		self.calc_color_dispersion: bool = calc_color_dispersion
		self.calc_g2: bool = calc_g2
		self.result_format: str = ""
		self.result_header: List[str] = []

	def get_result_format(self) -> str:
		if self.result_format != "":
			return self.result_format

		result_format = ["%d %d %f %f %f %f %f"]
		if self.calc_cas:
			result_format.extend(CASInfo.get_value_formats())
		if self.calc_g_m20:
			result_format.extend(GiniM20Info.get_value_formats())
		if self.calc_mid:
			result_format.extend(MIDInfo.get_value_formats())
			if self.calc_multiplicity:
				result_format.extend(["%f"])
		if self.calc_color_dispersion:
			result_format.extend(CompareInfo.get_value_formats())
		if self.calc_g2:
			result_format.extend(G2Info.get_value_formats())

		result_format.extend(["%f", "%d"])

		result_format_str = " ".join(result_format) + "\n"
		self.result_format = result_format_str
		return result_format_str

	def get_result_header(self) -> List[str]:
		if len(self.result_header) != 0:
			return self.result_header

		result_header = ["label", "size", "surface_brightness", "centroid_x", "centroid_y", "rp_circ_centroid",
						 "sn_per_pixel"]

		if self.calc_cas:
			result_header.extend(CASInfo.get_value_names())
		if self.calc_g_m20:
			result_header.extend(GiniM20Info.get_value_names())
		if self.calc_mid:
			result_header.extend(MIDInfo.get_value_names())
			if self.calc_multiplicity:
				result_header.extend(["multiplicity"])
		if self.calc_color_dispersion:
			result_header.extend(CompareInfo.get_value_names())
		if self.calc_g2:
			result_header.extend(G2Info.get_value_names())

		result_header.extend(["runtime", "base_flag"])
		self.result_header = result_header
		return result_header

	@abc.abstractmethod
	def measure_label(self, image: np.ndarray, segmap: np.ndarray, noisemap: Optional[np.ndarray], segm_slice,
					  label: int, image_compare: Optional[np.ndarray], output_image_dir: str,
					  set_centroid: Tuple[float, float], set_asym_center: Tuple[float, float]) -> List:
		pass

	def get_empty_result(self, label) -> List:
		empty_result: List = np.zeros(len(self.get_result_header())).tolist()
		empty_result[0] = label
		return empty_result


class StatmorphVanilla(MorphProvider):

	def measure_label(self, image: np.ndarray, segmap: np.ndarray, noisemap: Optional[np.ndarray], segm_slice,
					  label: int, image_compare: Optional[np.ndarray], output_image_dir: str,
					  set_centroid: Tuple[float, float], set_asym_center: Tuple[float, float]) -> List:
		morph = SourceMorphology(
			image, segmap, label, weightmap=noisemap)
		if morph.flag != 4:
			morph._calculate_morphology(self.calc_cas, self.calc_g_m20, self.calc_mid)

		return_list = [label, 0, 0, morph._centroid[0], morph._centroid[1],
					   morph._rpetro_circ_centroid, morph.sn_per_pixel]
		if self.calc_cas:
			return_list.extend(
				[morph._asymmetry_center[0], morph._asymmetry_center[1], morph.rpetro_circ, morph.concentration,
				 morph.asymmetry, morph.smoothness, morph._sky_asymmetry, 0, 0])
		if self.calc_g_m20:
			return_list.extend([morph.rpetro_ellip, morph.gini, morph.m20, 0, 0])
		if self.calc_mid:
			return_list.extend([morph.multimode, morph.intensity, morph.deviation, 0, 0])
			if self.calc_multiplicity:
				return_list.extend([0])
		if self.calc_color_dispersion:
			return_list.extend([0, 0])
		if self.calc_g2:
			return_list.extend([0, 0])

		return_list.extend([morph.runtime, morph.flag])
		return return_list


class StatmorphCython(MorphProvider):
	def measure_label(self, image: np.ndarray, segmap: np.ndarray, noisemap: Optional[np.ndarray], segm_slice,
					  label: int, image_compare: Optional[np.ndarray], output_image_dir: str,
					  set_centroid: Tuple[float, float], set_asym_center: Tuple[float, float]) -> List:
		morph = statmorph.BaseInfo(
			image, segmap, segm_slice, label, weightmap=noisemap, image_compare=image_compare,
			output_image_dir=output_image_dir, set_centroid=set_centroid)
		if not morph.flag_catastrophic:
			morph.calculate_morphology(self.calc_cas, self.calc_g_m20, self.calc_mid, self.calc_multiplicity,
									   self.calc_color_dispersion, self.calc_g2,
									   set_asym_center)

		return_list = [label, morph.size, morph.surface_brightness, morph._centroid[0], morph._centroid[1],
					   morph._rpetro_circ_centroid, morph.sn_per_pixel]
		if self.calc_cas:
			return_list.extend(morph.cas.get_values())
		if self.calc_g_m20:
			return_list.extend(morph.g_m20.get_values())
		if self.calc_mid:
			return_list.extend(morph.mid.get_values())
			if self.calc_multiplicity:
				return_list.extend([morph.multiplicity])
		if self.calc_color_dispersion:
			return_list.extend(morph.compare_info.get_values())
		if self.calc_g2:
			return_list.extend(morph.g2.get_values())

		return_list.extend([morph.runtime, morph.flags.value()])
		return return_list


def work_with_shared_memory(shm_img_name: str, shm_segm_name: str, shm_noise_name: Optional[str], segm_slice,
							label: int, shape, shm_img_cmp_name: Optional[str],
							output_image_dir: str, set_centroid: Tuple[float, float],
							set_asym_center: Tuple[float, float], img_dtype, segm_dtype, morph_provider: MorphProvider):
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


def run_sextractor(work_dir: str, detect_file: str, wht_file: str, use_existed: bool,
				   measure_file: Optional[str] = None,
				   config_values: Optional[Dict[str, Union[float, int, str]]] = None):
	sextractor = SExtractor(work_dir, SExtractor.merge_sex_dict(SExtractor.CANDELS_UKIDSS_USF_CONFIG,
																SExtractor.CANDELS_UKIDSS_USF_HOT_CONFIG,
																SExtractor.GLASS_JWST_VALUES, config_values),
							SExtractor.OUTPUT_LIST_DEFAULT)
	sextractor.run(detect_file, wht_file, measure_file, use_existed=use_existed)
	return sextractor


def run_statmorph(catalog_file: str, image_file: str, segmap_file: str, noise_file: Optional[str], save_file: str,
				  threads: int, run_percentage: int, run_specified_label: int, ignore_mag_fainter_than: float = 26.0,
				  ignore_class_star_greater_than: float = 0.9, calc_cas: bool = True, calc_g_m20: bool = True,
				  calc_mid: bool = False, calc_multiplicity: bool = False, calc_color_dispersion: bool = False,
				  image_compare_file: Optional[str] = None, calc_g2: bool = False,
				  output_image_dir: Optional[str] = None, center_file: Optional[str] = None, use_vanilla: bool = False):
	logger.info("????????????Statmorph, ?????????%d" % threads)

	calc_para_str_list = []
	if calc_cas:
		calc_para_str_list.append("CAS")
	if calc_g_m20:
		calc_para_str_list.append("G_M20")
	if calc_mid:
		calc_para_str_list.append("MID")
	if calc_multiplicity:
		calc_para_str_list.append("multiplicity")
	if calc_color_dispersion:
		calc_para_str_list.append("color_dispersion(??)")
	if calc_g2:
		calc_para_str_list.append("G2")

	logger.info("????????????: " + ", ".join(calc_para_str_list))
	sextractor_table: Table = Table.read(catalog_file, format="ascii")
	center_table: Optional[Table] = None
	if center_file is not None:
		logger.info("????????????????????????????????????" + center_file)
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
	logger.info("????????????%d" % len(labels))
	if run_specified_label <= 0:
		labels = labels[(sextractor_table["MAG_AUTO"] <= ignore_mag_fainter_than) & (
				sextractor_table["CLASS_STAR"] <= ignore_class_star_greater_than)]
		logger.info(
			"??????MAG_AUTO>%.1f???CLASS_STAR>%.1f???label" % (ignore_mag_fainter_than, ignore_class_star_greater_than))
		run_labels = labels[0:len(labels) * run_percentage // 100]
		logger.info("????????????%d%%??????%d" % (run_percentage, len(run_labels)))
	else:
		run_labels = [run_specified_label]
		logger.info("?????????label=%d" % run_specified_label)

	if use_vanilla:
		morph_provider: MorphProvider = StatmorphVanilla(calc_cas, calc_g_m20, calc_mid, calc_multiplicity,
														 calc_color_dispersion, calc_g2)
	else:
		morph_provider: MorphProvider = StatmorphCython(calc_cas, calc_g_m20, calc_mid, calc_multiplicity,
														calc_color_dispersion, calc_g2)
	logger.info("??????" + morph_provider.__class__.__qualname__)

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
		logger.info("?????????????????????????????????")
		np.copyto(shm_img_array, image)
		del image
		logger.info("image????????????")
		np.copyto(shm_segm_array, segmap)
		del segmap
		logger.info("segmap????????????")

		if noisemap is not None:
			np.copyto(shm_noise_array, noisemap)
			del noisemap
			logger.info("noisemap????????????")
		if image_compare is not None:
			np.copyto(shm_img_cmp_array, image_compare)
			del image_compare
			logger.info("image_compare????????????")

		logger.info("??????????????????")

		# Spawn some processes to do some work
		with ProcessPoolExecutor(threads) as exe:
			fs = []
			for label in run_labels:
				set_centroid: Tuple[float, float] = (-1, -1)
				set_asym_center: Tuple[float, float] = (-1, -1)
				try:
					label_index = segm_image.get_index(label)
				except ValueError as e:
					logger.warning(e)
					continue
				segm_slice = segm_image.slices[label_index]

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

				fs.append(
					exe.submit(work_with_shared_memory, shm_img.name, shm_segm.name, shm_noise_name, segm_slice, label,
							   shape, shm_img_cmp_name, output_image_dir, set_centroid, set_asym_center,
							   img_dtype, segm_dtype, morph_provider))
			for result in as_completed(fs):
				line = result_format % result.result()
				result_all.append(line)

	logger.info(f'??????: {time.time() - start_time:.2f}s')

	logger.info("??????????????????")

	with open(save_file, "w") as f:
		f.writelines(result_all)

	result_table: Table = Table.read(save_file, format="ascii")
	# result_table.sort("label")
	result_table["NUMBER"] = result_table["label"]
	result_table = join(sextractor_table, result_table)
	result_table.write(save_file, format="ascii", overwrite=True)

	logger.info("??????????????????" + save_file)


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


help_str = """SExtractor-Statmorph ???????????????????????????

	-j, --threads=???????????????????????????0??????CPU????????????-1(??????????????????1)
	-i, --image_file=??????????????????(???????????????)??????????????????????????????????????????????????????(???????????????PSF????????????)????????????SExtractor???????????????
	-y, --measure_file=?????????????????????????????????????????????(???????????????)???????????????(???null)?????????????????????????????????SExtractor???????????????
	-w, --wht_file=??????????????????????????????SExtractor???????????????
	-o, --save_file=?????????????????????????????????????????????????????????image_file????????????(???????????????).txt??????????????????????????????measure_file
	-p, --run_percentage=????????????????????????????????????100??????????????????
	-l, --run_specified_label=????????????????????????????????????0??????????????????
	-s, --sextractor_work_dir=SExtractor?????????????????????????????????????????????????????????image_file????????????(???????????????)??????????????????????????????measure_file
	-k, --skip_sextractor ??????????????????SExtractor????????????????????????SExtractor??????????????????????????????subback.fits,segmap.fits,noise.fits?????????????????????catalog.txt????????????
	-D, --sextractor_detect_minarea
	-T, --sextractor_detect_thresh
	-A, --sextractor_analysis_thresh
	-B, --sextractor_deblend_nthresh
	-M, --sextractor_deblend_mincont
	-S, --sextractor_back_size
	-F, --sextractor_back_filtersize
	-P, --sextractor_backphoto_thick
	-a, --output_image_dir=????????????????????????????????????null?????????????????????
	-f, --ignore_mag_fainter_than=?????????????????????????????????????????????
	-t, --ignore_class_star_greater_than=?????????????????????????????????????????????
	-n, --center_file=???????????????????????????????????????????????????????????????????????????
	-c, --calc_cas ????????????CAS
	-g, --calc_g_m20 ????????????Gini,M20
	-d, --calc_mid ????????????MID
	-u, --calc_multiplicity ????????????multiplicity
	-e, --calc_color_dispersion ????????????color_dispersion
	-m, --image_compare_file ??????color_dispersion????????????????????????(?????????????????????)?????????????????????null
	-b, --calc_g2 ????????????G2
	-v, --use_vanilla ????????????vanilla???
	-h, --help ???????????????
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
		"a": ("output_image_dir", True),
		"f": ("ignore_mag_fainter_than", True),
		"t": ("ignore_class_star_greater_than", True),
		"n": ("center_file", True),
		"c": ("calc_cas", False),
		"g": ("calc_g_m20", False),
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
		print("-h ??????????????????")
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

	output_image_dir: Optional[str] = config["output_image_dir"]
	ignore_mag_fainter_than: float = float(config["ignore_mag_fainter_than"])
	ignore_class_star_greater_than: float = float(config["ignore_class_star_greater_than"])
	calc_cas: bool = check_not_false(config["calc_cas"])
	calc_g_m20: bool = check_not_false(config["calc_g_m20"])
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

	run_name = get_basename_without_end(detect_file)
	if measure_file is not None:
		run_name = run_name + "_" + get_basename_without_end(measure_file)

	if not check_not_null(sextractor_work_dir):
		sextractor_work_dir = "sextractor_" + run_name
	if not check_not_null(save_file):
		save_file = "statmorph_" + run_name + ".txt"

	if threads <= 0:
		threads = min(multiprocessing.cpu_count() - 1, 1)

	sextractor = run_sextractor(sextractor_work_dir, detect_file, wht_file, skip_sextractor, measure_file,
								my_sextractor_config)
	run_statmorph(sextractor.output_catalog_file, sextractor.output_subback_file, sextractor.output_segmap_file,
				  sextractor.noise_file,
				  save_file, threads, run_percentage, run_specified_label, ignore_mag_fainter_than,
				  ignore_class_star_greater_than, calc_cas, calc_g_m20, calc_mid,
				  calc_multiplicity, calc_color_dispersion, image_compare_file, calc_g2, output_image_dir, center_file,
				  use_vanilla)
	return 0


if __name__ == "__main__":
	exit(main(sys.argv))
