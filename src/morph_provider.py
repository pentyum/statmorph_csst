import abc
from typing import List, Optional, Tuple

import numpy as np

import statmorph_cython.statmorph as statmorph
from statmorph_cython.statmorph_vanilla import SourceMorphology
from statmorph_cython.statmorph import CASInfo, GiniM20Info, MIDInfo, CompareInfo, G2Info, ShapeAsymmetryInfo


class MorphProvider(abc.ABC):
	def __init__(self, calc_cas: bool, calc_g_m20: bool, calc_shape_asymmetry: bool,
				 calc_mid: bool, calc_multiplicity: bool, calc_color_dispersion: bool, calc_g2: bool):
		self.calc_cas: bool = calc_cas
		self.calc_g_m20: bool = calc_g_m20
		self.calc_shape_asymmetry: bool = calc_shape_asymmetry
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
		if self.calc_shape_asymmetry:
			result_format.extend(ShapeAsymmetryInfo.get_value_formats())
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
		if self.calc_shape_asymmetry:
			result_header.extend(ShapeAsymmetryInfo.get_value_names())
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
					  label: int, image_compare: Optional[np.ndarray], output_image_dir: str, save_stamp_dir: str,
					  set_centroid: Tuple[float, float], set_asym_center: Tuple[float, float]) -> List:
		pass

	@abc.abstractmethod
	def measure_individual(self, label: int, flux_file_name: str, flux_hdu_index: int,
						   noise_file_name: Optional[str], noise_hdu_index: Optional[int],
						   segmap_file_name: Optional[str], segmap_hdu_index: Optional[int],
						   mask_file_name: Optional[str], mask_hdu_index: Optional[int],
						   cmp_file_name: Optional[str], cmp_hdu_index: Optional[int],
						   output_image_dir: str,
						   set_centroid: Tuple[float, float], set_asym_center: Tuple[float, float]) -> List:
		pass

	def get_empty_result(self, label) -> List:
		empty_result: List = np.zeros(len(self.get_result_header())).tolist()
		empty_result[0] = label
		return empty_result


class StatmorphVanilla(MorphProvider):

	def measure_individual(self, label: int, flux_file_name: str, flux_hdu_index: int,
						   noise_file_name: Optional[str], noise_hdu_index: Optional[int],
						   segmap_file_name: Optional[str], segmap_hdu_index: Optional[int],
						   mask_file_name: Optional[str], mask_hdu_index: Optional[int],
						   cmp_file_name: Optional[str], cmp_hdu_index: Optional[int],
						   output_image_dir: str,
						   set_centroid: Tuple[float, float], set_asym_center: Tuple[float, float]) -> List:
		pass

	def measure_label(self, image: np.ndarray, segmap: np.ndarray, noisemap: Optional[np.ndarray], segm_slice,
					  label: int, image_compare: Optional[np.ndarray], output_image_dir: str, save_stamp_dir: str,
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
	def _calc(self, morph: statmorph.BaseInfo, set_asym_center: Tuple[float, float]):
		if not morph.flag_catastrophic:
			morph.calculate_morphology(self.calc_cas, self.calc_g_m20, self.calc_shape_asymmetry, self.calc_mid, self.calc_multiplicity,
									   self.calc_color_dispersion, self.calc_g2,
									   set_asym_center)

		return_list = [morph.label, morph.size, morph.surface_brightness, morph._centroid[0], morph._centroid[1],
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

	def measure_individual(self, label: int, flux_file_name: str, flux_hdu_index: int,
						   noise_file_name: Optional[str], noise_hdu_index: Optional[int],
						   segmap_file_name: Optional[str], segmap_hdu_index: Optional[int],
						   mask_file_name: Optional[str], mask_hdu_index: Optional[int],
						   cmp_file_name: Optional[str], cmp_hdu_index: Optional[int],
						   output_image_dir: str,
						   set_centroid: Tuple[float, float], set_asym_center: Tuple[float, float]) -> List:
		morph = statmorph.IndividualBaseInfo(label, str(flux_file_name), flux_hdu_index,
											 str(segmap_file_name), segmap_hdu_index,
											 str(mask_file_name), mask_hdu_index,
											 str(noise_file_name), noise_hdu_index,
											 image_compare_file_name=str(cmp_file_name),
											 image_compare_hdu_index=cmp_hdu_index,
											 output_image_dir=output_image_dir, set_centroid=set_centroid)
		result = self._calc(morph, set_asym_center)
		morph.close_all()
		return result

	def measure_label(self, image: np.ndarray, segmap: np.ndarray, noisemap: Optional[np.ndarray], segm_slice,
					  label: int, image_compare: Optional[np.ndarray], output_image_dir: str, save_stamp_dir: str,
					  set_centroid: Tuple[float, float], set_asym_center: Tuple[float, float]) -> List:
		morph = statmorph.BaseInfo(
			image, segmap, segm_slice, label, weightmap=noisemap, image_compare=image_compare,
			output_image_dir=output_image_dir, save_stamp_dir=save_stamp_dir, set_centroid=set_centroid)
		return self._calc(morph, set_asym_center)
