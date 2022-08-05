# cython: language_level=3

cdef (double,double) fmin(func, (double,double) x0, tuple args, double xtol, double ftol,
						  int maxiter, int maxfun, bint disp)