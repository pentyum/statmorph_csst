# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

from libc.math cimport exp

import numpy as np
import scipy.optimize
from scipy._lib._util import check_random_state
from numpy.math cimport INFINITY

cdef basinhopping(func, x0, int niter=100, double T=1.0, double stepsize=0.5,
				  dict minimizer_kwargs=None, take_step=None, accept_test=None,
				  callback=None, int interval=50, bint disp=False, int niter_success=-1,
				  seed=None, double target_accept_rate=0.5, double stepwise_factor=0.9):
	if target_accept_rate <= 0. or target_accept_rate >= 1.:
		raise ValueError('target_accept_rate has to be in range (0, 1)')
	if stepwise_factor <= 0. or stepwise_factor >= 1.:
		raise ValueError('stepwise_factor has to be in range (0, 1)')

	x0 = np.array(x0)

	# set up the np.random generator
	rng = check_random_state(seed)

	# set up minimizer
	if minimizer_kwargs is None:
		minimizer_kwargs = dict()
	wrapped_minimizer = MinimizerWrapper(scipy.optimize.minimize, func,
										 **minimizer_kwargs)

	# set up step-taking algorithm
	if take_step is not None:
		if not callable(take_step):
			raise TypeError("take_step must be callable")
		# if take_step.stepsize exists then use AdaptiveStepsize to control
		# take_step.stepsize
		if hasattr(take_step, "stepsize"):
			take_step_wrapped = AdaptiveStepsize(
				take_step, interval=interval,
				accept_rate=target_accept_rate,
				factor=stepwise_factor,
				verbose=disp)
		else:
			take_step_wrapped = take_step
	else:
		# use default
		displace = RandomDisplacement(stepsize=stepsize, random_gen=rng)
		take_step_wrapped = AdaptiveStepsize(displace, interval=interval,
											 accept_rate=target_accept_rate,
											 factor=stepwise_factor,
											 verbose=disp)

	# set up accept tests
	accept_tests = []
	if accept_test is not None:
		if not callable(accept_test):
			raise TypeError("accept_test must be callable")
		accept_tests = [accept_test]

	# use default
	metropolis = Metropolis(T, random_gen=rng)
	accept_tests.append(metropolis)

	if niter_success < 0:
		niter_success = niter + 2

	bh = BasinHoppingRunner(x0, wrapped_minimizer, take_step_wrapped,
							accept_tests, disp=disp)

	# The wrapped minimizer is called once during construction of
	# BasinHoppingRunner, so run the callback
	if callable(callback):
		callback(bh.storage.minres.x, bh.storage.minres.fun, True)

	# start main iteration loop
	cdef int count, i
	count, i = 0, 0
	message = ["requested number of basinhopping iterations completed"
			   " successfully"]
	for i in range(niter):
		new_global_min = bh.one_cycle()

		if callable(callback):
			# should we pass a copy of x?
			val = callback(bh.xtrial, bh.energy_trial, bh.accept)
			if val is not None:
				if val:
					message = ["callback function requested stop early by"
							   "returning True"]
					break

		count += 1
		if new_global_min:
			count = 0
		elif count > niter_success:
			message = ["success condition satisfied"]
			break

	# prepare return object
	res = bh.res
	res.lowest_optimization_result = bh.storage.get_lowest()
	res.x = np.copy(res.lowest_optimization_result.x)
	res.fun = res.lowest_optimization_result.fun
	res.message = message
	res.nit = i + 1
	res.success = res.lowest_optimization_result.success
	return res

cdef class RandomDisplacement:
	def __init__(self, stepsize=0.5, random_gen=None):
		self.stepsize = stepsize
		self.random_gen = check_random_state(random_gen)

	def __call__(self, x):
		x += self.random_gen.uniform(-self.stepsize, self.stepsize,
									 np.shape(x))
		return x
	
cdef class MinimizerWrapper:
	cdef minimizer
	cdef func
	cdef dict kwargs

	def __init__(self, minimizer, func=None, **kwargs):
		self.minimizer = minimizer
		self.func = func
		self.kwargs = kwargs

	def __call__(self, x0):
		if self.func is None:
			return self.minimizer(x0, **self.kwargs)
		else:
			return self.minimizer(self.func, x0, **self.kwargs)
		
cdef class AdaptiveStepsize:
	cdef takestep
	cdef double target_accept_rate
	cdef int interval
	cdef double factor
	cdef bint verbose
	cdef int nstep
	cdef int nstep_tot
	cdef int naccept

	def __init__(self, takestep, accept_rate=0.5, interval=50, factor=0.9,
				 bint verbose=True):
		self.takestep = takestep
		self.target_accept_rate = accept_rate
		self.interval = interval
		self.factor = factor
		self.verbose = verbose

		self.nstep = 0
		self.nstep_tot = 0
		self.naccept = 0

	def __call__(self, x):
		return self.take_step(x)

	cdef _adjust_step_size(self):
		old_stepsize = self.takestep.stepsize
		accept_rate = float(self.naccept) / self.nstep
		if accept_rate > self.target_accept_rate:
			# We're accepting too many steps. This generally means we're
			# trapped in a basin. Take bigger steps.
			self.takestep.stepsize /= self.factor
		else:
			# We're not accepting enough steps. Take smaller steps.
			self.takestep.stepsize *= self.factor
		if self.verbose:
			print("adaptive stepsize: acceptance rate %f target %f new "
				  "stepsize %g old stepsize %g" % (accept_rate,
				  self.target_accept_rate, self.takestep.stepsize,
				  old_stepsize))

	cdef take_step(self, x):
		self.nstep += 1
		self.nstep_tot += 1
		if self.nstep % self.interval == 0:
			self._adjust_step_size()
		return self.takestep(x)

	cdef report(self, bint accept, **kwargs):
		if accept:
			self.naccept += 1
			
cdef class Metropolis:
	cdef double beta
	cdef random_gen

	def __init__(self, double T, random_gen=None):
		# Avoid ZeroDivisionError since "MBH can be regarded as a special case
		# of the BH framework with the Metropolis criterion, where temperature
		# T = 0." (Reject all steps that increase energy.)
		self.beta = 1.0 / T if T != 0 else INFINITY
		self.random_gen = check_random_state(random_gen)

	cdef accept_reject(self, energy_new, energy_old):
		"""
		If new energy is lower than old, it will always be accepted.
		If new is higher than old, there is a chance it will be accepted,
		less likely for larger differences.
		"""
		with np.errstate(invalid='ignore'):
			# The energy values being fed to Metropolis are 1-length arrays, and if
			# they are equal, their difference is 0, which gets multiplied by beta,
			# which is inf, and array([0]) * float('inf') causes
			#
			# RuntimeWarning: invalid value encountered in multiply
			#
			# Ignore this warning so so when the algorithm is on a flat plane, it always
			# accepts the step, to try to move off the plane.
			prod = -(energy_new - energy_old) * self.beta
			w = exp(min(0, prod))

		rand = self.random_gen.uniform()
		return w >= rand

	def __call__(self, **kwargs):
		"""
		f_new and f_old are mandatory in kwargs
		"""
		return bool(self.accept_reject(kwargs["f_new"],
					kwargs["f_old"]))
	
cdef class BasinHoppingRunner:
	cdef x
	cdef MinimizerWrapper minimizer
	cdef AdaptiveStepsize step_taking
	cdef accept_tests
	cdef bint disp
	cdef int nstep
	cdef res
	cdef x
	cdef energy
	cdef Storage storage

	cdef xtrial
	cdef energy_trial
	cdef bint accept

	def __init__(self, x0, minimizer, AdaptiveStepsize step_taking, list accept_tests, bint disp=False):
		self.x = np.copy(x0)
		self.minimizer = minimizer
		self.step_taking = step_taking
		self.accept_tests = accept_tests
		self.disp = disp

		self.nstep = 0

		# initialize return object
		self.res = scipy.optimize.OptimizeResult()
		self.res.minimization_failures = 0

		# do initial minimization
		minres = minimizer(self.x)
		if not minres.success:
			self.res.minimization_failures += 1
			if self.disp:
				print("warning: basinhopping: local minimization failure")
		self.x = np.copy(minres.x)
		self.energy = minres.fun
		if self.disp:
			print("basinhopping step %d: f %g" % (self.nstep, self.energy))

		# initialize storage class
		self.storage = Storage(minres)

		if hasattr(minres, "nfev"):
			self.res.nfev = minres.nfev
		if hasattr(minres, "njev"):
			self.res.njev = minres.njev
		if hasattr(minres, "nhev"):
			self.res.nhev = minres.nhev

	cdef _monte_carlo_step(self):
		# Take a random step.  Make a copy of x because the step_taking
		# algorithm might change x in place
		x_after_step = np.copy(self.x)
		x_after_step = self.step_taking.__call__(x_after_step)

		# do a local minimization
		minres = self.minimizer.__call__(x_after_step)
		x_after_quench = minres.x
		energy_after_quench = minres.fun
		if not minres.success:
			self.res.minimization_failures += 1
			if self.disp:
				print("warning: basinhopping: local minimization failure")

		if hasattr(minres, "nfev"):
			self.res.nfev += minres.nfev
		if hasattr(minres, "njev"):
			self.res.njev += minres.njev
		if hasattr(minres, "nhev"):
			self.res.nhev += minres.nhev

		# accept the move based on self.accept_tests. If any test is False,
		# then reject the step.  If any test returns the special string
		# 'force accept', then accept the step regardless. This can be used
		# to forcefully escape from a local minimum if normal basin hopping
		# steps are not sufficient.
		cdef bint accept = True
		for test in self.accept_tests:
			testres = test(f_new=energy_after_quench, x_new=x_after_quench,
						   f_old=self.energy, x_old=self.x)
			if testres == 'force accept':
				accept = True
				break
			elif testres is None:
				raise ValueError("accept_tests must return True, False, or "
								 "'force accept'")
			elif not testres:
				accept = False

		# Report the result of the acceptance test to the take step class.
		# This is for adaptive step taking
		self.step_taking.report(accept, f_new=energy_after_quench,
								x_new=x_after_quench, f_old=self.energy,
								x_old=self.x)

		return accept, minres

	cdef one_cycle(self):
		cdef bint accept, new_global_min

		self.nstep += 1
		new_global_min = False

		accept, minres = self._monte_carlo_step()

		if accept:
			self.energy = minres.fun
			self.x = np.copy(minres.x)
			new_global_min = self.storage.update(minres)

		# print some information
		if self.disp:
			self.print_report(minres.fun, accept)
			if new_global_min:
				print("found new global minimum on step %d with function"
					  " value %g" % (self.nstep, self.energy))

		# save some variables as BasinHoppingRunner attributes
		self.xtrial = minres.x
		self.energy_trial = minres.fun
		self.accept = accept

		return new_global_min

	cdef print_report(self, energy_trial, accept):
		minres = self.storage.get_lowest()
		print("basinhopping step %d: f %g trial_f %g accepted %d "
			  " lowest_f %g" % (self.nstep, self.energy, energy_trial,
								accept, minres.fun))
		
cdef class Storage:
	cdef minres

	def __init__(self, minres):
		self._add(minres)

	cdef _add(self, minres):
		self.minres = minres
		self.minres.x = np.copy(minres.x)

	cdef update(self, minres):
		if minres.fun < self.minres.fun:
			self._add(minres)
			return True
		else:
			return False

	cdef get_lowest(self):
		return self.minres