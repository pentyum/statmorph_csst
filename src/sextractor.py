import logging
import multiprocessing
import os
import re
import shutil
import time
from multiprocessing.pool import Pool
from typing import List, Dict, Union, Optional

import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.nddata import CCDData
from astropy.wcs import WCS


class SExtractor:
	default_sex_str_model = """# Default configuration file for SExtractor 2.8.6
# EB 2009-04-09
#
#-------------------------------- Catalog ------------------------------------

CATALOG_NAME     %s       # Will be overwritten anyway! name of the output catalog
CATALOG_TYPE     ASCII_HEAD      # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,
							   # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC
PARAMETERS_NAME  %s  # Will be overwritten anyway! name of the file containing catalog contents

#------------------------------- Extraction ----------------------------------

DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)
DETECT_MINAREA   %d             # minimum number of pixels above threshold
THRESH_TYPE      RELATIVE       # threshold type: RELATIVE (in sigmas)
								# or ABSOLUTE (in ADUs)
DETECT_THRESH    %.2f            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
ANALYSIS_THRESH  %.2f           # <sigmas> or <threshold>,<ZP> in mag.arcsec-2

FILTER           Y              # apply filter for detection (Y or N)?
FILTER_NAME      /usr/share/source-extractor/gauss_5.0_9x9.conv   # name of the file containing the filter
FILTER_THRESH                   # Threshold[s] for retina filtering

DEBLEND_NTHRESH  64             # Number of deblending sub-thresholds
DEBLEND_MINCONT  %.3f            # Minimum contrast parameter for deblending

CLEAN            Y              # Clean spurious detections? (Y or N)?
CLEAN_PARAM      1.0            # Cleaning efficiency

MASK_TYPE        CORRECT        # type of detection MASKing: can be one of
								# NONE, BLANK or CORRECT

#-------------------------------- WEIGHTing ----------------------------------

WEIGHT_TYPE      MAP_WEIGHT           # type of WEIGHTing: NONE, BACKGROUND,
								# MAP_RMS, MAP_VAR or MAP_WEIGHT
WEIGHT_IMAGE     %s           # Will be overwritten anyway! weight-map filename
WEIGHT_GAIN      N              # modulate gain (E/ADU) with weights? (Y/N)
WEIGHT_THRESH                   # weight threshold[s] for bad pixels

#-------------------------------- FLAGging -----------------------------------

FLAG_IMAGE       flag.fits      # filename for an input FLAG-image
FLAG_TYPE        OR             # flag pixel combination: OR, AND, MIN, MAX
								# or MOST

#------------------------------ Photometry -----------------------------------

PHOT_APERTURES   13.33,20.00,47.33      # MAG_APER aperture diameter(s) in pixels
PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>
PHOT_PETROPARAMS 2.5, 0.5       # MAG_PETRO parameters: <Petrosian_fact>,
								# <min_radius>
PHOT_AUTOAPERS   2.5,3.5        # <estimation>,<measurement> minimum apertures
								# for MAG_AUTO and MAG_PETRO
PHOT_FLUXFRAC    0.2,0.5,0.8            # flux fraction[s] used for FLUX_RADIUS

SATUR_LEVEL      30000        # level (in ADUs) at which arises saturation
SATUR_KEY        NOEXIT         # keyword for saturation level (in ADUs) - don't trust SATURATE

MAG_ZEROPOINT    25.936           # magnitude zero-point
MAG_GAMMA        4.0              # gamma of emulsion (for photographic scans)
GAIN             1           	 # detector gain in e-/ADU
GAIN_KEY         GAIN           # keyword for detector gain in e-/ADU
PIXEL_SCALE      0.03           # size of pixel in arcsec (0=use FITS WCS info)

#------------------------- Star/Galaxy Separation ----------------------------

SEEING_FWHM      0.11              # stellar FWHM in arcsec
STARNNW_NAME     /usr/share/source-extractor/default.nnw    # Neural-Network_Weight table filename

#------------------------------ Background -----------------------------------

BACK_TYPE        AUTO          # AUTO or MANUAL
BACK_SIZE        %d          # Background mesh: <size> or <width>,<height>
BACK_FILTERSIZE  %d             # Background filter: <size> or <width>,<height>

BACKPHOTO_TYPE   LOCAL         # can be GLOBAL or LOCAL
BACKPHOTO_THICK  200            # thickness of the background LOCAL annulus
BACK_FILTTHRESH  3             # Threshold above which the background-
							   # map filter operates

#------------------------------ Check Image ----------------------------------

CHECKIMAGE_TYPE  -BACKGROUND,SEGMENTATION           # can be NONE, BACKGROUND, BACKGROUND_RMS,
							   # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,
							   # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,
							   # or APERTURES
CHECKIMAGE_NAME  %s,%s     # Filename for the check-image

#--------------------- Memory (change with caution!) -------------------------

MEMORY_OBJSTACK  3000           # 3000 number of objects in stack
MEMORY_PIXSTACK  9000000         # 1200000 number of pixels in stack - x4 from default
MEMORY_BUFSIZE   1024           # 1024 number of lines in buffer

#------------------------------- ASSOCiation ---------------------------------

ASSOC_NAME       sky.list       # name of the ASCII file to ASSOCiate
ASSOC_DATA       2,3,4          # columns of the data to replicate (0=all)
ASSOC_PARAMS     2,3,4          # columns of xpos,ypos[,mag]
ASSOC_RADIUS     2.0            # cross-matching radius (pixels)
ASSOC_TYPE       NEAREST        # ASSOCiation method: FIRST, NEAREST, MEAN,
							   # MAG_MEAN, SUM, MAG_SUM, MIN or MAX
ASSOCSELEC_TYPE  MATCHED        # ASSOC selection type: ALL, MATCHED or -MATCHED

#----------------------------- Miscellaneous ---------------------------------

NTHREADS         4              # 1 single thread

	"""

	DEFAULT_CONFIG: Dict[str, Union[float, int, str]] = {
		# -------------------------------- Catalog ------------------------------------

		"CATALOG_NAME": "test.cat",  # name of the output catalog
		"CATALOG_TYPE": "ASCII_HEAD",  # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,
		# ASCII_VOTABLE, FITS_1.0 or FITS_LDAC
		"PARAMETERS_NAME": "default.param",  # name of the file containing catalog contents

		# ------------------------------- Extraction ----------------------------------

		"DETECT_TYPE": "CCD",  # CCD (linear) or PHOTO (with gamma correction)
		"DETECT_MINAREA": 3,  # min. # of pixels above threshold

		"DETECT_THRESH": 1.5,  # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
		"ANALYSIS_THRESH": 1.5,  # <sigmas> or <threshold>,<ZP> in mag.arcsec-2

		"FILTER": "Y",  # apply filter for detection (Y or N)?
		"FILTER_NAME": "default.conv",  # name of the file containing the filter

		"DEBLEND_NTHRESH": 32,  # Number of deblending sub-thresholds
		"DEBLEND_MINCONT": 0.005,  # Minimum contrast parameter for deblending

		"CLEAN": "Y",  # Clean spurious detections? (Y or N)?
		"CLEAN_PARAM": 1.0,  # Cleaning efficiency

		# -------------------------------- WEIGHTing ----------------------------------

		"WEIGHT_TYPE": "NONE",  # type of WEIGHTing: NONE, BACKGROUND,
		# MAP_RMS, MAP_VAR or MAP_WEIGHT
		"WEIGHT_IMAGE": "weight.fits",  # weight-map filename

		# -------------------------------- FLAGging -----------------------------------

		"FLAG_IMAGE": "flag.fits",  # filename for an input FLAG-image
		"FLAG_TYPE": "OR",  # flag pixel combination: OR, AND, MIN, MAX
		# or MOST

		# ------------------------------ Photometry -----------------------------------

		"PHOT_APERTURES": 5,  # MAG_APER aperture diameter(s) in pixels
		"PHOT_AUTOPARAMS": "2.5, 3.5",  # MAG_AUTO parameters: <Kron_fact>,<min_radius>
		"PHOT_PETROPARAMS": "2.0, 3.5",  # MAG_PETRO parameters: <Petrosian_fact>,
		# <min_radius>

		"PHOT_AUTOAPERS": "0.0, 0.0",  # <estimation>,<measurement> minimum apertures
		# for MAG_AUTO and MAG_PETRO

		"SATUR_LEVEL": 50000.0,  # level (in ADUs) at which arises saturation
		"SATUR_KEY": "SATURATE",  # keyword for saturation level (in ADUs)

		"MAG_ZEROPOINT": 0.0,  # magnitude zero-point
		"MAG_GAMMA": 4.0,  # gamma of emulsion (for photographic scans)
		"GAIN": 0.0,  # detector gain in e-/ADU
		"GAIN_KEY": "GAIN",  # keyword for detector gain in e-/ADU
		"PIXEL_SCALE": 1.0,  # size of pixel in arcsec (0=use FITS WCS info)

		# ------------------------- Star/Galaxy Separation ----------------------------

		"SEEING_FWHM": 1.2,  # stellar FWHM in arcsec
		"STARNNW_NAME": "default.nnw",  # Neural-Network_Weight table filename

		# ------------------------------ Background -----------------------------------

		"BACK_TYPE": "AUTO",  # AUTO or MANUAL
		"BACK_VALUE": 0.0,  # Default background value in MANUAL mode
		"BACK_SIZE": 64,  # Background mesh: <size> or <width>,<height>
		"BACK_FILTERSIZE": 3,  # Background filter: <size> or <width>,<height>

		# ------------------------------ Check Image ----------------------------------

		"CHECKIMAGE_TYPE": "NONE",  # can be NONE, BACKGROUND, BACKGROUND_RMS,
		# MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,
		# FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,
		# or APERTURES
		"CHECKIMAGE_NAME": "check.fits",  # Filename for the check-image

		# --------------------- Memory (change with caution!) -------------------------

		"MEMORY_OBJSTACK": 3000,  # number of objects in stack
		"MEMORY_PIXSTACK": 300000,  # number of pixels in stack
		"MEMORY_BUFSIZE": 1024,  # number of lines in buffer

		# ------------------------------- ASSOCiation ---------------------------------

		"ASSOC_NAME": "sky.list",  # name of the ASCII file to ASSOCiate
		"ASSOC_DATA": "2, 3, 4",  # columns of the data to replicate (0=all)
		"ASSOC_PARAMS": "2, 3, 4",  # columns of xpos,ypos[,mag]
		"ASSOC_RADIUS": 2.0,  # cross-matching radius (pixels)
		"ASSOC_TYPE": "NEAREST",  # ASSOCiation method: FIRST, NEAREST, MEAN,
		# MAG_MEAN, SUM, MAG_SUM, MIN or MAX
		"ASSOCSELEC_TYPE": "MATCHED",  # ASSOC selection type: ALL, MATCHED or -MATCHED

		# ----------------------------- Miscellaneous ---------------------------------

		"VERBOSE_TYPE": "NORMAL",  # can be QUIET, NORMAL or FULL
		"HEADER_SUFFIX": ".head",  # Filename extension for additional headers
		"WRITE_XML": "N",  # Write XML file (Y/N)?
		"XML_NAME": "sex.xml",  # Filename for XML output
		"XSL_URL": "file:///usr/local/share/sextractor/sextractor.xsl"  # Filename for XSL style-sheet
	}

	WEIGHT_CONFIG = {
		"WEIGHT_TYPE": "MAP_WEIGHT",
		"WEIGHT_GAIN": "N",
		"GAIN": 1
	}

	CANDELS_UKIDSS_USF_CONFIG = {
		"MASK_TYPE": "CORRECT",
		"PHOT_FLUXFRAC": "0.2, 0.5, 0.8",
		"PHOT_APERTURES": "1.47, 2.08, 2.94, 4.17, 5.88, 8.34, 11.79, 16.66, 23.57, 33.34, 47.13",
		"PIXEL_SCALE": 0.060,
		"SEEING_FWHM": 0.18,
		"STARNNW_NAME": "/usr/share/source-extractor/default.nnw",
		"BACKPHOTO_TYPE": "LOCAL",
		# "WEIGHT_THRESH": "10000.0, 10000.0",
		"MEMORY_OBJSTACK": 4000,
		"MEMORY_PIXSTACK": 3000000,
		"MEMORY_BUFSIZE": 5000
	}  # Galametz et al. (2013)

	CANDELS_UKIDSS_USF_COLD_CONFIG = {
		"DETECT_MINAREA": 5.0,
		"DETECT_THRESH": 0.75,
		"ANALYSIS_THRESH": 5.0,
		"FILTER_NAME": "/usr/share/source-extractor/tophat_9.0_9x9.conv",
		"DEBLEND_NTHRESH": 16,
		"DEBLEND_MINCONT": 0.0001,
		"SATUR_LEVEL": 120.0,
		"BACK_SIZE": 256,
		"BACK_FILTERSIZE": 9,
		"BACKPHOTO_THICK": 100
	}

	CANDELS_UKIDSS_USF_HOT_CONFIG = {
		"DETECT_MINAREA": 10.0,
		"DETECT_THRESH": 0.7,
		"ANALYSIS_THRESH": 0.7,
		"FILTER_NAME": "/usr/share/source-extractor/gauss_4.0_7x7.conv",
		"DEBLEND_NTHRESH": 64,
		"DEBLEND_MINCONT": 0.001,
		"SATUR_LEVEL": 3900.0,
		"BACK_SIZE": 128,
		"BACK_FILTERSIZE": 5,
		"BACKPHOTO_THICK": 48
	}

	GLASS_JWST_VALUES: Dict[str, Union[float, int, str]] = {
		"DETECT_MINAREA": 8,
		"DETECT_THRESH": 0.7071,
		"ANALYSIS_THRESH": 0.7071,
		"DEBLEND_NTHRESH": 32,
		"DEBLEND_MINCONT": 0.0003,
		"BACK_SIZE": 64,
		"BACK_FILTERSIZE": 3,
		"BACKPHOTO_TYPE": "LOCAL",
		"BACKPHOTO_THICK": 48.0,
		"PIXEL_SCALE": 0
	}  # Merlin et al. (2022)

	BRIGHT_VALUES: Dict[str, Union[float, int]] = {
		"DETECT_MINAREA": 140,
		"DETECT_THRESH": 2.2,
		"ANALYSIS_THRESH": 2.2,
		"DEBLEND_MINCONT": 0.04,
		"BACK_SIZE": 400,
		"BACK_FILTERSIZE": 5
	}

	FAINT_VALUES: Dict[str, Union[float, int]] = {
		"DETECT_MINAREA": 18,
		"DETECT_THRESH": 1.0,
		"ANALYSIS_THRESH": 1.0,
		"DEBLEND_MINCONT": 0.065,
		"BACK_SIZE": 100,
		"BACK_FILTERSIZE": 3
	}

	logging.basicConfig(level=logging.INFO,
						format="[%(asctime)s][%(name)s - %(processName)s/%(levelname)s]: %(message)s")

	OUTPUT_LIST_DEFAULT: List[str] = [
		"NUMBER",
		"ALPHA_J2000",
		"DELTA_J2000",
		"MAG_AUTO",
		"CLASS_STAR"
	]

	default_param_file = "default.param"
	default_param_bak_file = "default.param.bak"

	default_sex_file = "default.sex"

	TEMP_DIR = "/tmp/sextractor/"
	LOGGER = logging.getLogger("SExtractor")
	POOL: Optional[Pool] = None

	@staticmethod
	def get_ab_zero_point(photflam: float, photplam: float) -> float:
		return -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408

	@staticmethod
	def get_ab_zero_point_from_unit(unit: u.Unit) -> float:
		return u.Magnitude((1 * unit).to(u.AB)).value

	@staticmethod
	def unzip(gz_file_name: str, target_file_name: str) -> int:
		return os.system("gunzip -c %s > %s" % (gz_file_name, target_file_name))

	def __init__(self, work_dir: str, config: Dict[str, Union[float, int]], output_list: List[str]):
		self.return_value: int = -1
		self.logger = logging.getLogger(
			"SExtractor(%s)" % os.path.basename(work_dir))
		self.work_dir: str = work_dir
		self.config: Dict[str, Union[float, int, str]] = SExtractor.merge_sex_dict(SExtractor.DEFAULT_CONFIG,
																				   SExtractor.WEIGHT_CONFIG, config)
		self.output_list: List[str] = output_list
		self.output_catalog_file: str = ""
		self.output_subback_file: str = ""
		self.output_segmap_file: str = ""

	def unzip_fits_gz(self, file_name: str) -> str:
		if file_name.endswith(".gz"):
			if not os.path.exists(SExtractor.TEMP_DIR):
				os.mkdir(SExtractor.TEMP_DIR)
			base_file_name = os.path.basename(file_name)[0:-3]
			new_file_name = SExtractor.TEMP_DIR + base_file_name
			self.logger.info("检测到" + base_file_name + "为压缩文件，解压至" + new_file_name)
			if os.path.exists(new_file_name):
				self.logger.info(new_file_name + "已经存在，直接使用")
			else:
				SExtractor.unzip(file_name, new_file_name)
			return new_file_name
		else:
			return file_name

	def make_work_dir(self, clean_existed: bool = True):
		if clean_existed:
			if os.path.exists(self.work_dir):
				self.logger.warning(self.work_dir + "文件夹已存在，自动删除已有文件夹")
				shutil.rmtree(self.work_dir)
			os.mkdir(self.work_dir)
		else:
			if not os.path.exists(self.work_dir):
				os.mkdir(self.work_dir)
				self.logger.info("创建" + self.work_dir + "文件夹")
			else:
				self.logger.warning(self.work_dir + "文件夹已存在")

	def make_default_param(self):
		os.system("source-extractor -dp > " + self.work_dir + "/" + SExtractor.default_param_bak_file)
		self.logger.info("已生成default.param")

		string = "^#" + "(" + "|".join(self.output_list) + ")\s"
		pattern = re.compile(string)

		with open(self.work_dir + "/" + SExtractor.default_param_bak_file, "r") as default_param_bak, open(
				self.work_dir + "/" + SExtractor.default_param_file, "w") as default_param:
			for line in default_param_bak:
				default_param.write(pattern.sub("\\1  ", line))
		os.remove(self.work_dir + "/" + SExtractor.default_param_bak_file)
		self.logger.info("已取消" + ", ".join(self.output_list) + "的注释")

	@staticmethod
	def merge_sex_dict(old_dict: Dict[str, Union[float, int, str]], *mod_dicts: Dict[str, Union[float, int, str]]) -> \
			Dict[str, Union[float, int, str]]:
		new_dict = old_dict.copy()
		for mod_dict in mod_dicts:
			for k, v in mod_dict.items():
				new_dict[k] = v
		return new_dict

	@staticmethod
	def sex_dict_to_str(sex_config_dict: Dict[str, Union[float, int, str]]):
		s = "# Default configuration file for Source Extractor 2.25.0\n"
		for k, v in sex_config_dict.items():
			s = s + k + "\t" + str(v) + "\n"
		return s

	def make_default_sex(self, wht_file_unzipped: str, output_catalog_file: str, output_subback_file: str,
						 output_segmap_file: str, zeropoint: float) -> str:
		self.config["CATALOG_NAME"] = self.work_dir + "/" + output_catalog_file
		self.config["PARAMETERS_NAME"] = self.work_dir + "/" + SExtractor.default_param_file
		self.config["WEIGHT_IMAGE"] = wht_file_unzipped
		subback_save_path = self.work_dir + "/" + output_subback_file
		segmap_save_path = self.work_dir + "/" + output_segmap_file
		self.config["MAG_ZEROPOINT"] = zeropoint
		self.config["CHECKIMAGE_TYPE"] = "-BACKGROUND,SEGMENTATION"
		self.config["CHECKIMAGE_NAME"] = "%s,%s" % (subback_save_path, segmap_save_path)

		default_sex_str = SExtractor.sex_dict_to_str(self.config)

		"""
		default_sex_str = SExtractor.default_sex_str_model % (
			self.work_dir + "/" + output_catalog_file,
			self.work_dir + "/" + SExtractor.default_param_file,
			self.config["DETECT_MINAREA"],
			self.config["DETECT_THRESH"],
			self.config["ANALYSIS_THRESH"],
			self.config["DEBLEND_MINCONT"],
			wht_file_unzipped,
			self.config["BACK_SIZE"],
			self.config["BACK_FILTERSIZE"],
			self.work_dir + "/" + output_subback_file,
			self.work_dir + "/" + output_segmap_file
		)
		"""

		sex_file_path: str = self.work_dir + "/" + SExtractor.default_sex_file
		with open(sex_file_path, "w") as default_sex:
			default_sex.write(default_sex_str)
		self.logger.info("已生成default.sex")
		self.logger.info("配置信息: " + str(self.config))
		return sex_file_path

	def handle_unzipped_file_end(self, temp_file_name: str, clean_temp: bool):
		"""
		处理解压后的文件，如果clean_tamp则删除，否则不删除
		"""
		if clean_temp:
			os.remove(temp_file_name)
			self.logger.info("清理" + temp_file_name)
		else:
			self.logger.warning("未清理解压后的文件" + temp_file_name)

	@staticmethod
	def is_main_process():
		"""
		是否是主进程
		"""
		return not bool(re.match(r'ForkPoolWorker-\d+', multiprocessing.current_process().name))

	def handle_unzip_fits_gz_list(self, unzip_file_list: List[str]) -> List[str]:
		threads = len(unzip_file_list)
		if SExtractor.is_main_process():
			self.logger.info("启动%d线程同时解压" % threads)
			pool = multiprocessing.Pool(threads)
			result = pool.map(self.unzip_fits_gz, unzip_file_list)
		else:
			result = [self.unzip_fits_gz(file) for file in unzip_file_list]

		return result

	@staticmethod
	def read_bunit(fits_unit_string: str) -> u.Unit:
		# Convert the BUNIT header keyword to a unit and if that's not
		# possible raise a meaningful error message.
		try:
			kifus = CCDData.known_invalid_fits_unit_strings
			if fits_unit_string in kifus:
				return kifus[fits_unit_string]
			else:
				return u.Unit(fits_unit_string)
		except ValueError:
			raise ValueError(
				'The Header value for the key BUNIT ({}) cannot be '
				'interpreted as valid unit. To successfully read the '
				'file as CCDData you can pass in a valid `unit` '
				'argument explicitly or change the header of the FITS '
				'file before reading it.'
				.format(fits_unit_string))

	@classmethod
	def read_zeropoint(cls, file: str, index: int = 0) -> float:
		zeropoint: float = 0
		with fits.open(file) as file_fits:
			header: fits.Header = file_fits[index].header
			if "ZP" in header:
				cls.LOGGER.info("zeropoint直接读取header的ZP字段")
				zeropoint = header["ZP"]
			elif "BUNIT" in header:
				unit_str: str = header["BUNIT"]
				unit = SExtractor.read_bunit(unit_str)
				if unit.physical_type == u.AB.physical_type:  # 流量单位Jy或者erg/s/Hz/cm2
					cls.LOGGER.info("zeropoint由单位(%s)计算得到" % unit)
					zeropoint = SExtractor.get_ab_zero_point_from_unit(unit)
				elif unit.physical_type == (u.AB / u.sr).physical_type:  # 流量/立体角单位Jy/sr或者erg/s/Hz/cm2/sr
					cls.LOGGER.info("zeropoint由单位(%s)和像素面积计算得到" % unit)
					w = WCS(header)
					area = w.proj_plane_pixel_area()
					unit_in_pixel = unit * area
					zeropoint = SExtractor.get_ab_zero_point_from_unit(unit_in_pixel)
				elif unit.physical_type == u.K.physical_type:  # 亮温度
					raise NotImplementedError
				elif "PHOTFLAM" in header and "PHOTPLAM" in header and unit == u.electron / u.s:  # 电子数/秒单位
					cls.LOGGER.info("zeropoint由header的PHOTFLAM和PHOTPLAM字段计算得到")
					zeropoint = SExtractor.get_ab_zero_point(header["PHOTFLAM"], header["PHOTPLAM"])
				else:
					cls.LOGGER.error("未知单位类型(%s)" % unit)
			else:  # 未知单位
				cls.LOGGER.error("header没有ZP或者正确的单位或者PHOTFLAM和PHOTPLAM字段，无法获取测光零点")
		cls.LOGGER.info("zeropoint=%.2f" % zeropoint)
		return zeropoint

	def run(self, detect_file: str, wht_file: str, measure_file: Optional[str] = None,
			output_catalog_file: str = "catalog.txt",
			output_subback_file: str = "subback.fits", output_segmap_file: str = "segmap.fits",
			clean_temp: bool = True, use_existed: bool = False):
		self.output_catalog_file = self.work_dir + "/" + output_catalog_file
		self.output_subback_file = self.work_dir + "/" + output_subback_file
		self.output_segmap_file = self.work_dir + "/" + output_segmap_file

		if use_existed and os.path.exists(self.work_dir):
			self.logger.info("使用已经生成的SExtractor文件")
			if os.path.exists(self.output_catalog_file) and os.path.exists(self.output_subback_file) and os.path.exists(
					self.output_segmap_file):
				self.logger.info("catalog使用" + self.output_catalog_file)
				self.logger.info("subback使用" + self.output_subback_file)
				self.logger.info("segmap使用" + self.output_segmap_file)
				return
			else:
				self.logger.warning("文件不全，继续运行SExtractor")

		start_time = time.time()
		self.make_work_dir()
		self.logger.info("SExtractor工作文件夹: " + self.work_dir)

		if not os.path.exists(self.work_dir + "/" + SExtractor.default_param_file):
			self.make_default_param()

		if measure_file is None:
			self.logger.info("sci图像文件: " + detect_file)
		else:
			self.logger.info("detect图像文件: " + detect_file)
			self.logger.info("measure图像文件: " + measure_file)

		self.logger.info("weight图像文件: " + wht_file)

		if measure_file is None:
			detect_file_unzipped, wht_file_unzipped = self.handle_unzip_fits_gz_list([detect_file, wht_file])
			zeropoint = SExtractor.read_zeropoint(detect_file_unzipped)
			measure_file_unzipped = None
		else:
			detect_file_unzipped, wht_file_unzipped, measure_file_unzipped = self.handle_unzip_fits_gz_list(
				[detect_file, wht_file, measure_file])
			zeropoint = SExtractor.read_zeropoint(measure_file_unzipped)

		self.logger.info("zeropoint=%.2f" % zeropoint)

		sex_file_path: str = self.make_default_sex(wht_file_unzipped, output_catalog_file, output_subback_file,
												   output_segmap_file, zeropoint)

		if measure_file_unzipped is None:
			self.logger.info("开始运行source-extractor单图像模式")
			self.return_value = os.system("source-extractor " + detect_file_unzipped + " -c " + sex_file_path)
		else:
			self.logger.info("开始运行source-extractor双图像模式")
			self.return_value = os.system(
				"source-extractor " + detect_file_unzipped + "," + measure_file_unzipped + " -c " + sex_file_path)

		if self.return_value == 0:
			self.logger.info("SExtractor运行成功")
			self.logger.info("catalog保存至" + self.output_catalog_file)
			self.logger.info("subback保存至" + self.output_subback_file)
			self.logger.info("segmap保存至" + self.output_segmap_file)
		else:
			err_msg = "SExtractor(%s)运行错误！返回值%d" % (self.work_dir, self.return_value)
			self.logger.critical(err_msg)
			raise RuntimeError(err_msg)

		if detect_file_unzipped != detect_file:
			self.handle_unzipped_file_end(detect_file_unzipped, clean_temp)
		if wht_file_unzipped != wht_file:
			self.handle_unzipped_file_end(wht_file_unzipped, clean_temp)
		if measure_file_unzipped != measure_file:
			self.handle_unzipped_file_end(measure_file_unzipped, clean_temp)

		self.logger.info(f'用时: {time.time() - start_time:.2f}s')

	@classmethod
	def run_individual(cls, work_dir: str, image_file: str, wht_file: str, measure_file: str = None):
		sextractor = SExtractor(work_dir, cls.BRIGHT_VALUES, cls.OUTPUT_LIST_DEFAULT)
		try:
			sextractor.run(image_file, wht_file, measure_file)
		except:
			pass
		return sextractor.return_value

	@classmethod
	def run_list(cls, work_dir_list: List[str], image_file_list: List[str], wht_file_list: List[str], threads: int = 0):
		cls.LOGGER.info("共%d张图" % len(work_dir_list))
		if threads <= 0:
			threads = multiprocessing.cpu_count() - 1
		cls.POOL = multiprocessing.Pool(threads)
		cls.LOGGER.info("启动%d进程" % threads)
		start_time = time.time()
		result_list = []
		for work_dir, image_file, wht_file in zip(work_dir_list, image_file_list, wht_file_list):
			result_list.append(cls.POOL.apply_async(cls.run_individual, (work_dir, image_file, wht_file)))
		cls.POOL.close()
		cls.POOL.join()
		result_list = [result.get() for result in result_list]
		cls.LOGGER.info(f'总共用时: {time.time() - start_time:.2f}s')
		return result_list
