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
	Extension("statmorph_cython.constants_setting", ["src/statmorph_cython/constants_setting.pyx"]),
	Extension("statmorph_cython.petrosian", ["src/statmorph_cython/petrosian.pyx"]),
	Extension("statmorph_cython.cas", ["src/statmorph_cython/cas.pyx"]),
	Extension("statmorph_cython.g_m20", ["src/statmorph_cython/g_m20.pyx"]),
	Extension("statmorph_cython.shape_asymmetry", ["src/statmorph_cython/shape_asymmetry.pyx"]),
	Extension("statmorph_cython.mid", ["src/statmorph_cython/mid.pyx"]),
	Extension("statmorph_cython.multiplicity", ["src/statmorph_cython/multiplicity.pyx"]),
	Extension("statmorph_cython.color_dispersion", ["src/statmorph_cython/color_dispersion.pyx"]),
	Extension("statmorph_cython.g2", ["src/statmorph_cython/g2.pyx"]),
	Extension("statmorph_cython.sersic", ["src/statmorph_cython/sersic.pyx"]),
	Extension("statmorph_cython.statmorph", ["src/statmorph_cython/statmorph.pyx"]),
]

compiler_directives = {"language_level": 3, "embedsignature": True}

extensions = cythonize(extensions, compiler_directives=compiler_directives)

setup(
	name='statmorph_csst',
	version='0.5.0',
	description='Non-parametric morphological diagnostics of galaxy images (Cython version)',
	url='https://gitee.com/pentyum/statmorph_csst',
    author='Vicente Rodriguez-Gomez & Yao Yao',
    author_email='vrodgom.astro@gmail.com & pentyum@189.cn',
    license='BSD',
	classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    keywords='astronomy galaxies galaxy-morphology non-parametric',
	packages=['statmorph_cython'],
    install_requires=['numpy',
                      'scipy',
                      'scikit-image',
                      'astropy',
                      'opencv-python',
					  'photutils',
					  'Cython',
					  'setuptools'],

	include_dirs=[np.get_include()],
	ext_modules=extensions
)