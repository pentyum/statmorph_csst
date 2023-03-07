# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

cdef class ConstantsSetting:
	def __init__(self):
		self.cutout_extent = 1.5
		"""
		图像切片往外延伸的倍率
		"""

		self.min_cutout_size = 48
		"""
		最小切片大小，默认48
		"""

		self.n_sigma_outlier = 10
		"""
		排除几倍sigma，默认10
		"""

		self.annulus_width = 1.0
		"""
		圆环宽度，默认1像素
		"""

		self.eta = 0.2
		"""
		求Petrosian半径时使用的eta，默认0.2
		"""

		self.petro_fraction_gini = 0.2
		"""
		求G-M20的segmap前用几倍的椭圆Petrosian半长轴进行平滑，默认0.2
		"""

		self.skybox_size = 32
		"""
		背景区域的大小，默认32*32像素
		"""

		self.petro_extent_cas = 1.5
		"""
		CAS求和范围为几倍rp，默认1.5
		"""

		self.petro_fraction_cas = 0.25
		"""
		S平滑大小为几倍rp，并忽略中心，默认0.25
		"""

		self.petro_extent_flux = 2.0
		"""
		测光用几倍rp，默认2.0
		"""

		self.simplified_rot_threshold = -1
		"""
		多大以上的图采用简化旋转
		"""

		self.fmin_maxiter = 100
		"""
		求不对称度的最大迭代次数
		"""

		self.boxcar_size_mid = 3.0
		self.sigma_mid = 1.0
		self.niter_bh_mid = 5

		self.segmap_overlap_ratio = 0.25

		self.g2_modular_tolerance = 0.01
		self.g2_phase_tolerance = 0.01

		self.verbose = False

		self.label = 0