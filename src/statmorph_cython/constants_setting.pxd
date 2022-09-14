# cython: language_level=3

cdef class ConstantsSetting:
	cdef double cutout_extent
	"""
	图像切片往外延伸的倍率
	"""

	cdef int min_cutout_size
	"""
	最小切片大小，默认48
	"""

	cdef int n_sigma_outlier
	"""
	排除几倍sigma，默认10
	"""

	cdef double annulus_width
	"""
	圆环宽度，默认1像素
	"""

	cdef double eta
	"""
	求Petrosian半径时使用的eta，默认0.2
	"""

	cdef double petro_fraction_gini
	"""
	求G-M20的segmap前用几倍的椭圆Petrosian半长轴进行平滑，默认0.2
	"""

	cdef int skybox_size
	"""
	背景区域的大小，默认32*32像素
	"""

	cdef double petro_extent_cas
	"""
	CAS求和范围为几倍rp，默认1.5
	"""

	cdef double petro_fraction_cas
	"""
	S平滑大小为几倍rp，并忽略中心，默认0.25
	"""

	cdef double petro_extent_flux
	"""
	测光用几倍rp，默认2.0
	"""

	cdef int simplified_rot_threshold
	"""
	多大以上的图采用简化旋转
	"""

	cdef int fmin_maxiter
	"""
	求不对称度的最大迭代次数
	"""

	cdef double boxcar_size_mid
	cdef double sigma_mid
	cdef int niter_bh_mid

	cdef double g2_modular_tolerance
	cdef double g2_phase_tolerance

	cdef bint verbose
