import src.statmorph_cython.statmorph as statmorph

import numpy as np

from astropy.io import fits
import photutils


def main():
	image_fits = fits.open("/home/yaoyao/python/statmorph_simplified/gal.fits")
	segm_fits = fits.open("/home/yaoyao/python/statmorph_simplified/segmap.fits")
	image = image_fits[0].data.astype(np.float64)
	segmap = segm_fits[0].data.astype(np.int32)
	segmap_image = photutils.SegmentationImage(segmap)

	label = 3

	slc = segmap_image.slices[segmap_image.get_index(label)]

	morph = statmorph.BaseInfo(
		image, segmap, slc, label, calc_cas=True, calc_color_dispersion=True, image_compare=image)

	print("_rpetro_circ_centroid", morph._rpetro_circ_centroid)
	if morph.calc_cas:
		print("rpetro_circ", morph.cas.rpetro_circ)
		print("c=", morph.cas.concentration)
		print("a=", morph.cas.asymmetry)
		print("s=", morph.cas.smoothness)
		print("cas_time=", morph.cas.runtime)
	if morph.calc_g_m20:
		print("rpetro_ellip", morph.g_m20.rpetro_ellip)
		print("g=", morph.g_m20.gini)
		print("m20=", morph.g_m20.m20)
		print("g_m20_time=", morph.g_m20.runtime)
	if morph.calc_mid:
		print("m=", morph.mid.multimode)
		print("i=", morph.mid.intensity)
		print("d=", morph.mid.deviation)
		print("mid_time=", morph.mid.runtime)
	if morph.calc_color_dispersion:
		print("color_dispersion=", morph.compare_info.color_dispersion)
		print("color_dispersion_time=", morph.compare_info.runtime)

	print("total_time=", morph.runtime)


if __name__ == '__main__':
	main()
