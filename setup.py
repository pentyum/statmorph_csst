import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
	Extension("statmorph_cython.__init__", ["src/statmorph_cython/__init__.pyx"]),
	Extension("statmorph_cython.optimize.__init__", ["src/statmorph_cython/optimize/__init__.pyx"]),
	Extension("statmorph_cython.optimize.neldermead", ["src/statmorph_cython/optimize/neldermead.pyx"]),
	Extension("statmorph_cython.array_utils", ["src/statmorph_cython/array_utils.pyx"]),
	Extension("statmorph_cython.flags", ["src/statmorph_cython/flags.pyx"]),
	Extension("statmorph_cython.photutils_simplified", ["src/statmorph_cython/photutils_simplified.pyx"]),
	Extension("statmorph_cython.petrosian", ["src/statmorph_cython/petrosian.pyx"]),
	Extension("statmorph_cython.cas", ["src/statmorph_cython/cas.pyx"]),
	Extension("statmorph_cython.g_m20", ["src/statmorph_cython/g_m20.pyx"]),
	Extension("statmorph_cython.mid", ["src/statmorph_cython/mid.pyx"]),
	Extension("statmorph_cython.multiply", ["src/statmorph_cython/multiply.pyx"]),
	Extension("statmorph_cython.color_dispersion", ["src/statmorph_cython/color_dispersion.pyx"]),
	Extension("statmorph_cython.statmorph", ["src/statmorph_cython/statmorph.pyx"]),
]

compiler_directives = {"language_level": 3, "embedsignature": True}

extensions = cythonize(extensions, compiler_directives=compiler_directives)

setup(
	include_dirs=[np.get_include()],
	ext_modules=extensions
)