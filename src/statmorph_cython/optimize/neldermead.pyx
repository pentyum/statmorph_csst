# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from numpy.math cimport INFINITY

cnp.import_array()

cdef (double,double) _fmin(func, (double,double) x0, tuple args=(), double xtol=1e-4, double ftol=1e-4, int maxiter=-1, int maxfun=-1,
			bint disp=True):
	cdef (double,double) res = _minimize_neldermead(func, x0, args, maxiter, maxfun, disp, xtol, ftol)
	return res

cdef (double,double) fmin(func, (double,double) x0, tuple args, double xtol, double ftol, int maxiter, int maxfun, bint disp):
	cdef (double,double) res = _minimize_neldermead(func, x0, args, maxiter, maxfun, disp, xtol, ftol)
	return res

cdef dict _status_message = {'success': 'Optimization terminated successfully.',
				   'maxfev': 'Maximum number of function evaluations has '
							  'been exceeded.',
				   'maxiter': 'Maximum number of iterations has been '
							  'exceeded.',
				   'pr_loss': 'Desired error not necessarily achieved due '
							  'to precision loss.',
				   'nan': 'NaN result encountered.',
				   'out_of_bounds': 'The result is outside of the provided '
									'bounds.'}

cdef (double,double) _minimize_neldermead(func, (double,double) x0_tuple, tuple args=(),
						 int maxiter=-1, int maxfev=-1, bint disp=False,
						 double xatol=1e-4, double fatol=1e-4):

	#cdef cnp.ndarray x0 = np.asfarray(x0_tuple).flatten()
	cdef int maxfun = maxfev

	cdef int rho = 1
	cdef int chi = 2
	cdef double psi = 0.5
	cdef double sigma = 0.5

	cdef double nonzdelt = 0.05
	cdef double zdelt = 0.00025

	cdef int N = 2

	#cdef double[:,:] sim_memv = np.empty((3, 2), dtype=np.float64)
	cdef double[:,:] sim_memv = cnp.PyArray_EMPTY(2,[3, 2],cnp.NPY_DOUBLE,0)

	sim_memv[0,0] = x0_tuple[0]
	sim_memv[0,1] = x0_tuple[1]

	cdef int k, j
	cdef double y

	y = x0_tuple[0]
	if y != 0:
		y = (1 + nonzdelt)*y
	else:
		y = zdelt
	sim_memv[1,0] = y
	sim_memv[1,1] = x0_tuple[1]

	y = x0_tuple[1]
	if y != 0:
		y = (1 + nonzdelt)*y
	else:
		y = zdelt
	sim_memv[2,0] = x0_tuple[0]
	sim_memv[2,1] = y

	# If neither are set, then set both to default
	# -1 default value, 0 infinity
	if maxiter < 0 and maxfun < 0:
		maxiter = N * 200
		maxfun = N * 200
	elif maxiter < 0:
		# Convert remaining Nones, to np.inf, unless the other is np.inf, in
		# which case use the default to avoid unbounded iteration
		if maxfun == 0:
			maxiter = N * 200
			maxfun = 10000000
		else:
			maxiter = 10000000
	elif maxfun < 0:
		if maxiter == 0:
			maxfun = N * 200
			maxiter = 10000000
		else:
			maxfun = 10000000

	cdef double[:] fsim = np.full(3, INFINITY, dtype=np.float64)
	cdef cnp.ndarray ind

	fun_caller = Maxfun_validator(func, args, maxfun)

	fsim[0] = fun_caller.call(sim_memv[0])
	fsim[1] = fun_caller.call(sim_memv[1])
	fsim[2] = fun_caller.call(sim_memv[2])

	cdef cnp.ndarray sim = sim_memv.base

	ind = cnp.PyArray_ArgSort(fsim.base,-1,cnp.NPY_QUICKSORT)
	sim = cnp.PyArray_Take(sim, ind, 0)
	fsim = cnp.PyArray_Take(fsim.base, ind, 0)

	ind = cnp.PyArray_ArgSort(fsim.base,-1,cnp.NPY_QUICKSORT)
	fsim = cnp.PyArray_Take(fsim.base, ind, 0)
	# sort so sim[0,:] has the lowest function value
	sim = cnp.PyArray_Take(sim, ind, 0)

	cdef int iterations = 1
	cdef double fxr,fxe,fxc,fxcc
	cdef cnp.ndarray xr,xe,xc,xcc,xbar
	cdef bint doshrink

	while fun_caller.ncalls+4 < maxfun and iterations < maxiter:
		if (np.max(cnp.PyArray_Ravel(np.abs(sim[1:] - sim[0]),cnp.NPY_CORDER)) <= xatol and
				max(abs(fsim[0]-fsim[1]),abs(fsim[0]-fsim[2])) <= fatol):
			break

		xbar = np.add.reduce(sim[:2], 0) / N
		xr = (1 + rho) * xbar - rho * sim[2]
		fxr = fun_caller.call(xr)
		doshrink = False

		if fxr < fsim[0]:
			xe = (1 + rho * chi) * xbar - rho * chi * sim[2]
			fxe = fun_caller.call(xe)

			if fxe < fxr:
				sim[2] = xe
				fsim[2] = fxe
			else:
				sim[2] = xr
				fsim[2] = fxr
		else:  # fsim[0] <= fxr
			if fxr < fsim[1]:
				sim[2] = xr
				fsim[2] = fxr
			else:  # fxr >= fsim[-2]
				# Perform contraction
				if fxr < fsim[2]:
					xc = (1 + psi * rho) * xbar - psi * rho * sim[2]
					fxc = fun_caller.call(xc)

					if fxc <= fxr:
						sim[2] = xc
						fsim[2] = fxc
					else:
						doshrink = True
				else:
					# Perform an inside contraction
					xcc = (1 - psi) * xbar + psi * sim[2]
					fxcc = fun_caller.call(xcc)

					if fxcc < fsim[2]:
						sim[2] = xcc
						fsim[2] = fxcc
					else:
						doshrink = True

				if doshrink:
					sim[1] = sim[0] + sigma * (sim[1] - sim[0])
					fsim[1] = fun_caller.call(sim[1])

					sim[2] = sim[0] + sigma * (sim[2] - sim[0])
					fsim[2] = fun_caller.call(sim[2])

		iterations += 1

		ind = cnp.PyArray_ArgSort(fsim.base,-1,cnp.NPY_QUICKSORT)
		sim = cnp.PyArray_Take(sim, ind, 0)
		fsim = cnp.PyArray_Take(fsim.base, ind, 0)

	x = sim[0]
	cdef double fval = min(fsim[0],fsim[1],fsim[2])
	cdef int warnflag = 0
	cdef str msg

	if fun_caller.ncalls >= maxfun:
		warnflag = 1
		msg = _status_message['maxfev']
		if disp:
			print('Warning: ' + msg)
	elif iterations >= maxiter:
		warnflag = 2
		msg = _status_message['maxiter']
		if disp:
			print('Warning: ' + msg)
	else:
		msg = _status_message['success']
		if disp:
			print(msg)
			print("		 Current function value: %f" % fval)
			print("		 Iterations: %d" % iterations)
			print("		 Function evaluations: %d" % fun_caller.ncalls)

	return x[0], x[1]


cdef class Maxfun_validator:
	cdef int ncalls
	cdef int maxfun
	cdef function
	cdef tuple args

	def __init__(self, function, tuple args, int maxfun):
		self.ncalls = 0
		self.function = function
		self.args = args
		self.maxfun = maxfun

	cdef double call(self, double[:] x):
		self.ncalls += 1
		# A copy of x is sent to the user function (gh13740)
		cdef double fx = self.function((x[0],x[1]), *self.args)

		# Ideally, we'd like to a have a true scalar returned from f(x). For
		# backwards-compatibility, also allow np.array([1.3]),
		# np.array([[1.3]]) etc.
		return fx

	cdef double call_1d(self, double x):
		self.ncalls += 1
		# A copy of x is sent to the user function (gh13740)
		cdef double fx = self.function(x, *self.args)
		return fx
