# Numpy
import numpy as np
# Scipy
from scipy.stats import linregress
# Matplotlib
import matplotlib.pyplot as plt


def plot_spectrum(spectrum, freqs=None, drop_zero_frequency=True, ax=None,
                  xlog=False, ylog=False, loglog=False,
                  variance_preserving=False, xlim=None,
                  ylim=None, title=None, **kwargs):
	"""Define a nice spectrum with twin x-axis, one with frequencies, the
	other one with periods.

	Parameters
	----------
	spectrum : 1d xarray.DataArray or 1darray
		The array where the spectrum is stored
	freqs: 1d vector, optional
		The frequency vector. If None, the frequency vector is inferred
		from the DataArray
	drop_zero_frequency : bool, optional
	    If True, do not plot the zero frequency
	ax : matplotlib axes, optional
		If None, uses the current axis.
	xlog : bool, optional
		If True, use log scaling for the x axis
	ylog : bool, optional
		If True, use log scaling for the y axis
	loglog : bool, optional
		If True, use log scaling for both axis
	variance_preserving : bool, optional
		If True, scale the spectrum by the log of frequencies to use the
		variance preserving form
	xlim : tuple, optional
		Set x-axis limits
	ylim : tuple, optional
		Set y-axis limits
	title : string, optional
		Set the title
	**kwargs : optional
		Additional arguments to matplotlib.pyplot.plot
	"""
	if ax is None:
		ax = plt.gca()
	if freqs is None:
		freqs = spectrum[spectrum.dims[0]]
	if drop_zero_frequency:
		spectrum = spectrum.where(freqs != 0.)
		freqs = freqs.where(freqs != 0.)
		#import pytest
		#pytest.set_trace()
	if variance_preserving:
		spectrum = freqs * spectrum
		xlog = True
	ax.plot(freqs, spectrum, **kwargs)

	if xlog or loglog:
		ax.set_xscale('log', nonposx='clip')
		try:
			xmin = np.ceil(np.log10(np.abs(xlim[0]))) - 1
			xmax = np.ceil(np.log10(np.abs(xlim[1])))
			ax.set_xlim((10 ** xmin, 10 ** xmax))
		except TypeError:
			try:
				xmin = np.ceil(np.log10(abs(freqs[1]))) - 1
				xmax = np.ceil(np.log10(abs(freqs[-1])))
				ax.set_xlim((10 ** xmin, 10 ** xmax))
			except TypeError:
				pass
	else:
		ax.set_xlim(xlim)

	if ylog or loglog:
		ax.set_yscale('log', nonposy='clip')
		try:
			ymin = np.ceil(np.log10(np.abs(ylim[0]))) - 1
			ymax = np.ceil(np.log10(np.abs(ylim[1])))
			ax.set_ylim((10 ** ymin, 10 ** ymax))
		except TypeError:
			try:
				ymin = np.ceil(np.log10(spectrum.min())) - 1
				ymax = np.ceil(np.log10(spectrum.max()))
				ax.set_ylim((10 ** ymin, 10 ** ymax))
			except TypeError:
				pass
	else:
		ax.set_ylim(ylim)

	twiny = ax.twiny()
	if xlog or loglog:
		twiny.set_xscale('log', nonposx='clip')
		twiny.set_xlim((10 ** xmin, 10 ** xmax))
		new_major_ticks = 10 ** np.arange(xmin + 1, xmax, 1)
		new_major_ticklabels = 1. / new_major_ticks
		new_major_ticklabels = ["%.3g" % i for i in new_major_ticklabels]
		twiny.set_xticks(new_major_ticks)
		twiny.set_xticklabels(new_major_ticklabels, rotation=60, fontsize=12)
		A = np.arange(2, 10, 2)[np.newaxis]
		B = 10 ** (np.arange(-xmax, -xmin, 1)[np.newaxis])
		C = np.dot(B.transpose(), A)
		new_minor_ticklabels = C.flatten()
		new_minor_ticks = 1. / new_minor_ticklabels
		new_minor_ticklabels = ["%.3g" % i for i in new_minor_ticklabels]
		twiny.set_xticks(new_minor_ticks, minor=True)
		twiny.set_xticklabels(new_minor_ticklabels, minor=True, rotation=60,
		                      fontsize=12)
	ax.grid(True, which='both')


def plot_power_law(power, scale_factor=1., ax=None, **kwargs):
	"""Plot a logarithmic power law

	Parameters
	----------
	power : float
		The exponent of the power law
	scale_factor : float, optional
		The factor to scale the power law with
	ax : matplotlib axes, optional
		If None, uses the current axis.
	**kwargs : optional
		Additional arguments to matplotlib.pyplot.plot

	Returns
	-------
	lines : Line2D
		Return a Line2D object created by the matplotlib.axes.Axes.plot method
	"""
	if ax is None:
		ax = plt.gca()
	xlim = np.array(ax.get_xlim())
	power_law = scale_factor * xlim ** power
	return ax.plot(xlim, power_law, **kwargs)


def fit_power_law(freq, spectrum):
	"""Fit a logarithmic spectral law based on the input  one
	dimensional spectrum

	Parameters
	----------
	freq : 1darray
		The frequency coordinates
	spectrum : 1darray
		The one-dimensional spectrum

	Returns
	-------
	power : float
		The power characteristic of a power law spectrul
	scale_factor: float
		The scale factor related to fit the power law with the input spectrum
	"""
	power, intercept, _, _, _ = linregress(np.log(freq), np.log(spectrum))
	scale_factor = np.exp(intercept)
	return power, scale_factor


def _plot_spectrum2d(ax, x, y, z, xlog=False, ylog=False, zlog=False, **kwargs):
	"""
	Define a nice spectrum with twin x-axis and twin y-axis, one with
	frequencies, the other one with periods, on a predefined axis
	object.

	Parameters
	----------
	x,y : array_like
		1D array defining the coordinates
	z : array_like
		2D array
	xlog, ylog, zlog : bool, optional
		Define if the x-axis, y-axis and z-axis are plotted with a
		log	scale
	** kwargs : optional keyword arguments
		See matplotlib.axes.Axes.contourf method in matplotlib
		documentation
	"""
	if not 'xlim' in kwargs:
		xlim = None
	else:
		xlim = kwargs['xlim']
		del kwargs['xlim']
	if not 'ylim' in kwargs:
		ylim = None
	else:
		ylim = kwargs['ylim']
		del kwargs['ylim']
	if not 'zlim' in kwargs:
		zlim = None
	else:
		zlim = kwargs['zlim']
		del kwargs['zlim']

	n_lev = 40
	# if symmetric:
	# lim = max(np.max(z), abs(np.min(z)))
	# lev = np.hstack((np.linspace(- lim, 0, n_lev / 2 + 1),
	# np.linspace(0, lim, n_lev / 2)[1:]))
	#
	# else:
	# lev = np.linspace(np.min(z), np.max(z), n_lev / 2 + 1)
	if zlog:
		plot = ax.pcolormesh(np.log10(z), **kwargs)
	else:
		plot = ax.pcolormesh(z, **kwargs)
	# X limits
	if xlog:
		ax.set_xscale('symlog', nonposx='clip')
		xmin = np.ceil(np.log10(x[1,])) - 1
		xmax = np.ceil(np.log10(x[-1,]))
		ax.set_xlim((10 ** xmin, 10 ** xmax))
	else:
		try:
			ax.set_xlim(xlim)
		except:
			ax.set_xlim(np.min(x), np.max(x))
	# Y limits
	if ylog:
		ax.set_yscale('symlog', nonposx='clip')
		ymin = np.ceil(np.log10(x[1,])) - 1
		ymax = np.ceil(np.log10(x[-1,]))
		ax.set_ylim((-10 ** ymin, 10 ** ymax))
	else:
		try:
			ax.set_ylim(ylim)
		except:
			ax.set_ylim(np.min(y), np.max(y))
	axtwiny = ax.twiny()
	if xlog:
		axtwiny.set_xscale('symlog', nonposx='clip')
		axtwiny.set_xlim((-10 ** xmin, 10 ** xmax))
		A = np.arange(2, 10, 2)[np.newaxis]
		B = 10 ** (np.arange(-xmax, -xmin, 1)[np.newaxis])
		C = np.dot(B.transpose(), A)
		new_major_ticks = 10 ** np.arange(xmin + 1, xmax, 1)
		new_minor_ticklabels = C.flatten()
		new_minor_ticklabels = new_minor_ticklabels.astype(int)
		new_minor_ticks = 1. / new_minor_ticklabels
		axtwiny.set_xticks(new_minor_ticks, minor=True)
		axtwiny.set_xticklabels(new_minor_ticklabels, minor=True,
		                        rotation=30)
		new_major_ticklabels = 1. / new_major_ticks
		new_major_ticklabels = new_major_ticklabels.astype(int)
		axtwiny.set_xticks(new_major_ticks)
		axtwiny.set_xticklabels(new_major_ticklabels, rotation=30)
	axtwinx = ax.twinx()
	if ylog:
		axtwinx.set_yscale('symlog', nonposx='clip')
		axtwinx.set_ylim(y[1], y[-1])
		axtwinx.set_ylim((10 ** ymin, 10 ** ymax))
		new_major_ticks = 10 ** np.arange(ymin + 1, ymax, 1)
		new_major_ticklabels = 1. / new_major_ticks
		new_major_ticklabels = new_major_ticklabels.astype(int)
		axtwinx.set_yticks(new_major_ticks)
		axtwinx.set_yticklabels(new_major_ticklabels)
		A = np.arange(2, 10, 2)[np.newaxis]
		B = 10 ** (np.arange(-ymax, -ymin, 1)[np.newaxis])
		C = np.dot(B.transpose(), A)
		new_minor_ticklabels = C.flatten()
		new_minor_ticklabels = new_minor_ticklabels.astype(int)
		new_minor_ticks = 1. / new_minor_ticklabels
		axtwinx.set_yticks(new_minor_ticks, minor=True)
		axtwinx.set_yticklabels(new_minor_ticklabels, minor=True)
	ax.grid(True, which='both')