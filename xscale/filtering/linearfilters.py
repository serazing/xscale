"""Define functions for linear filtering that works on multi-dimensional
xarray.DataArray and xarray.Dataset objects.
"""
# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
# Internal
import copy
from collections import Iterable
# Numpy and scipy
import numpy as np
import scipy.signal as sig
import scipy.ndimage as im
# Xarray and dask
from dask.diagnostics import ProgressBar
import dask.array as da
import xarray as xr
from xarray.ufuncs import log10
# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import transforms
# Current package
from .. import _utils
from ..spectral.fft import fft, psd

import pdb

@xr.register_dataarray_accessor('window')
class Window(object):
	"""
	Class for all different type of windows
	"""

	_attributes = ['order', 'cutoff', 'dx', 'window']

	def __init__(self, xarray_obj):
		self._obj = xarray_obj
		self.obj = xarray_obj # Associated xarray object
		self.n = None # Size of the window
		self.dims = None # Dimensions of the window
		self.ndim = 0 # Number of dimensions
		self.cutoff = None # Window cutoff
		self.window = None # Window type (scipy-like type)
		self.order = None # Window order
		self.coefficients = 1. # Window coefficients
		self._depth = dict() # Overlap between different blocks
		self.fnyq = dict() # Nyquist frequency

	def __repr__(self):
		"""
		Provide a nice string representation of the window object
		"""
		# Function copied from xarray.core.rolling
		attrs = ["{k}->{v}".format(k=k, v=getattr(self, k))
		         for k in self._attributes if
		         getattr(self, k, None) is not None]
		return "{klass} [{attrs}]".format(klass=self.__class__.__name__,
		                                  attrs=', '.join(attrs))

	def set(self, n=None, dim=None, cutoff=None, dx=None, window='boxcar',
	        chunks=None):
		"""Set the different properties of the current window.

		Parameters
		----------
		n : int, sequence or dict, optional
			The window order over the dimensions specified through a dictionary
			or through the ``dims`` parameters. If ``n`` is ``None``, the window
			order is set to the total size of the corresponding dimensions
			according to the ``dims`` parameters
		dim : str or sequence, optional
			Names of the dimension associated to the window. If ``dims`` is
			None, all the dimensions are taken.
		cutoff : float, sequence or dict, optional
			The window cutoff over the dimensions specified through a
			dictionary, or through the ``dims`` parameters. If `cutoff`` is
			``None``, the cutoff parameters will be not used in the design
			the window.
		dx : float or sequence, optional
			Define the resolution of the dimensions. If not precised,
			the resolution is computed directly from the coordinates
			associated to the dimensions.
		window : string, tuple of string and parameter values, or dict
			Desired window to use. See scipy.signal.get_window for a list of
			windows and required parameters.
        chunks : int, tuple or dict, optional
            Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
            ``{'x': 5, 'y': 5}``
		"""

		# Check and interpret n and dims parameters
		self.n, self.dims = _utils.infer_n_and_dims(self._obj, n, dim)
		self.ndim = len(self.dims)
		self.order = {di: nbw for nbw, di in zip(self.n, self.dims)}
		self.cutoff = _utils.infer_arg(cutoff, self.dims)
		self.dx = _utils.infer_arg(dx, self.dims)
		self.window = _utils.infer_arg(window, self.dims,
		                               default_value='boxcar')
		# Rechunk if needed
		self.obj = self._obj.chunk(chunks=chunks)

		# Reset attributes
		self.fnyq = dict()
		self.coefficients = 1.
		self._depth = dict()

		# Build the multi-dimensional window: the hard part
		for di in self.obj.dims:
			axis_num = self.obj.get_axis_num(di)
			axis_chunk = self.obj.chunks[axis_num][0]
			if di in self.dims:
				self._depth[axis_num] = self.order[di] // 2
				if self.dx[di] is None:
					self.dx[di] = _utils.get_dx(self.obj, di)
				self.fnyq[di] = 1. / (2. * self.dx[di])
				# Compute the coefficients associated to the window using scipy functions
				if self.cutoff[di] is None:
					# Use get_window if the cutoff is undefined
					coefficients1d = sig.get_window(self.window[di],
					                                self.order[di])
				else:
					# Use firwin if the cutoff is defined
					coefficients1d = sig.firwin(self.order[di],
					                            1. / self.cutoff[di],
					                            window=self.window[di],
					                            nyq=self.fnyq[di])
				coefficients1d_dask = da.from_array(coefficients1d,
				                                    chunks=axis_chunk)
				# Normalize the coefficients
				#self.coefficients = np.outer(self.coefficients, coefficients1d)
				self.coefficients = (np.expand_dims(self.coefficients, axis=-1)
				                     * coefficients1d_dask)
				# TODO: Try to add the rotational convention using meshgrid,
				# in complement to the outer product
				# TODO: check the order of dimension of the kernel compared to
				# the DataArray/DataSet objects
				#self.coefficients = self.coefficients.squeeze()
			else:
				self.coefficients = np.expand_dims(self.coefficients,
				                                   axis=axis_num)

	def convolve(self, mode='reflect', weights=1., trim=False, compute=False):
		"""Convolve the current window with the data

		Parameters
		----------
		mode : {'reflect', 'periodic', 'any-constant'}, optional
			The mode parameter determines how the array borders are handled.
			Default is 'reflect'.

		mask : DataArray

		weights :

		trim :

		compute : bool, optional
			If True, the computation is performed after the dask graph has
			been made. If False, only the dask graph is made is the computation
			will be performed later on. The latter allows the integration into
			a larger dask graph, which could include other computational steps.

		Returns
		-------
		res : xarray.DataArray
			Return the filtered  the low-passed filtered
		"""
		# Check if the data has more dimensions than the window and add
		# extra-dimensions to the window if it is the case
		coeffs = self.coefficients / self.coefficients.sum()
		if trim:
			mode = np.nan
			new_data = self.obj.data
		else:
			new_data = self.obj.fillna(0.).data
		boundary = {self._obj.get_axis_num(di): mode for di in self.dims}
		conv = lambda x: im.convolve(x, coeffs, mode=mode)
		data_conv = new_data.map_overlap(conv, depth=self._depth,
		                                       boundary=boundary,
		                                       trim=True)
		res = 1. / weights *  xr.DataArray(data_conv, dims=self.obj.dims,
		                                              coords=self.obj.coords,
		                                              name=self.obj.name)
		if compute:
			with ProgressBar():
				out = res.compute()
		else:
			out = res
		return out

	def boundary_weights(self, mode='reflect', mask=None, drop_dims=[],
	                     compute=False):
		"""
		Compute the boundary weights

		Parameters
		----------
		mode:

		drop_dims:
			Specify dimensions along which the mask is constant

		Returns
		-------
		weights:
		"""
		coeffs = self.coefficients / self.coefficients.sum()
		new_coeffs = da.squeeze(coeffs, axis=[self.obj.get_axis_num(di)
		                                     for di in drop_dims])
		new_obj = self.obj.isel(**{di: 0 for di in drop_dims}).squeeze()
		depth = {new_obj.get_axis_num(di): self.order[di] // 2
		         for di in new_obj.dims}
		if mask is None:
			mask = da.notnull(new_obj.data)
		conv = lambda x: im.convolve(x, new_coeffs, mode=mode)
		weights = mask.astype(float).map_overlap(conv, depth=depth,
		                                               boundary=mode,
		                                               trim=True)

		res = xr.DataArray(mask * weights, dims=new_obj.dims,
		                                   coords=new_obj.coords,
		                                   name='boundary_weights')
		res = res.where(res != 0)
		if compute:
			with ProgressBar():
				out = res.compute()
		else:
			out = res
		return out

	def tapper(self, overlap=0.):
		"""
		Do a tappering of the data using the current window

		Parameters
		----------
		overlap:

		Returns
		-------
		data_tappered : dask array
			The data tappered y the window

		Notes
		-----
		"""
		# TODO: Improve this function to implement multitapper
		res = xr.DataArray(self.coefficients * self.obj.data,
		                   dims=self.obj.dims, coords=self.obj.coords,
		                   name=self.obj.name)
		return res

	def plot(self):
		"""
		Plot the weights distribution of the window and the associated
		spectrum (work only for 1D and 2D windows).
		"""
		win_array = xr.DataArray(self.coefficients.squeeze(),
		                         dims=self.dims).squeeze()
		win_spectrum = psd(fft(win_array, nfft=1024, dim=self.dims,
		                                  dx=self.dx, sym=True))
		win_spectrum_norm = 20 * log10(win_spectrum / abs(win_spectrum).max())
		if self.ndim == 1:
			_plot1d_window(win_array, win_spectrum_norm)
		elif self.ndim == 2:
			_plot2d_window(win_array, win_spectrum_norm)
		else:
			raise ValueError("This number of dimension is not supported by the "
			                 "plot function")


def _plot1d_window(win_array, win_spectrum_norm):

	dim = win_spectrum_norm.dims[0]
	freq = win_spectrum_norm[dim]
	min_freq = np.extract(freq > 0, freq).min()
	cutoff_3db = 1. / abs(freq[np.argmin(np.abs(win_spectrum_norm + 3))])
	cutoff_6db = 1. / abs(freq[np.argmin(np.abs(win_spectrum_norm + 6))])

	# Plot window properties
	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

	# First plot: weight distribution
	win_array.plot(ax=ax1)
	ax1.set_ylabel("Amplitude")
	ax1.set_xlabel("Sample")

	# Second plot: frequency response
	win_spectrum_norm.plot(ax=ax2)
	ax2.set_xscale('symlog', linthreshx=min_freq,
	               subsx=[2, 3, 4, 5, 6, 7, 8, 9])
	box = dict(boxstyle='round', facecolor='white', alpha=1)
	textstr = '$\lambda^{3dB}=%.1f$ \n $\lambda^{6dB}=%.1f$' % (cutoff_3db,
	                                                            cutoff_6db)
	ax2.text(0.5, 0.45, textstr, transform=ax2.transAxes, fontsize=14,
	         verticalalignment='top',
	         horizontalalignment='center', bbox=box)
	ax2.set_ylim((-200, 20))
	ax2.set_ylabel("Normalized magnitude [dB]")
	ax2.set_xlabel("Frequency [cycles per sample]")
	ax2.grid(True)
	plt.tight_layout()


def _plot2d_window(win_array, win_spectrum_norm):

	n_x, n_y = win_array.shape
	n_fx, n_fy = win_spectrum_norm.shape
	dim_fx, dim_fy = win_spectrum_norm.dims
	win_array_x = win_array[:, n_y // 2]
	win_array_y = win_array[n_x // 2, :]
	win_spectrum_x = win_spectrum_norm.isel(**{dim_fy: n_fy // 2})
	win_spectrum_y = win_spectrum_norm.isel(**{dim_fx: n_fx // 2})
	freq_x, freq_y = win_spectrum_norm[dim_fx], win_spectrum_norm[dim_fy]

	min_freq_x = np.extract(freq_x > 0, freq_x).min()
	min_freq_y = np.extract(freq_y > 0, freq_y).min()

	cutoff_x_3db = 1. / abs(freq_x[np.argmin(np.abs(win_spectrum_x + 3))])
	cutoff_x_6db = 1. / abs(freq_x[np.argmin(np.abs(win_spectrum_x + 6))])
	cutoff_y_3db = 1. / abs(freq_y[np.argmin(np.abs(win_spectrum_y + 3))])
	cutoff_y_6db = 1. / abs(freq_y[np.argmin(np.abs(win_spectrum_y + 6))])

	fig = plt.figure(1, figsize=(16, 8))
	# Definitions for the axes
	left, width = 0.05, 0.25
	bottom, height = 0.05, 0.5
	offset = 0.05
	bottom_h = bottom + height + offset

	rect_2D_weights = [left, bottom, width, height]
	rect_x_weights = [left, bottom_h, width, height / 2]
	rect_y_weights = [left + width + offset, bottom, width / 2, height]
	rect_2D_spectrum = [left + 3. / 2 * width + 2 * offset, bottom, width,
	                    height]
	rect_x_spectrum = [left + 3. / 2 * width + 2 * offset, bottom_h, width,
	                   height / 2]
	rect_y_spectrum = [left + 5. / 2 * width + 3 * offset, bottom,
	                   width / 2, height]
	ax_2D_weights = plt.axes(rect_2D_weights)
	ax_x_weights = plt.axes(rect_x_weights)
	ax_y_weights = plt.axes(rect_y_weights)
	ax_x_spectrum = plt.axes(rect_x_spectrum)
	ax_y_spectrum = plt.axes(rect_y_spectrum)
	ax_2D_spectrum = plt.axes(rect_2D_spectrum)

	# Weight disribution along y
	win_array_y.squeeze().plot(ax=ax_x_weights)
	ax_x_weights.set_ylabel('')
	ax_x_weights.set_xlabel('')

	# Weight disribution along x
	base = ax_y_weights.transData
	rot = transforms.Affine2D().rotate_deg(270)
	win_array_x.plot(ax=ax_y_weights, transform=rot + base)
	ax_y_weights.set_ylabel('')
	ax_y_weights.set_xlabel('')

	# Full 2d weight distribution
	win_array.plot(ax=ax_2D_weights, add_colorbar=False)

	# Spectrum along f_y
	win_spectrum_y.plot(ax=ax_x_spectrum)
	ax_x_spectrum.set_xscale('symlog', linthreshx=min_freq_y,
	                         subsx=[2, 3, 4, 5, 6, 7, 8, 9])
	ax_x_spectrum.set_ylim([-200, 20])
	ax_x_spectrum.grid()
	ax_x_spectrum.set_ylabel("Normalized magnitude [dB]")
	ax_x_spectrum.set_xlabel("")
	box = dict(boxstyle='round', facecolor='white', alpha=1)
	# place a text box in upper left in axes coords
	textstr = '$\lambda_y^{3dB}=%.1f$ \n $\lambda_y^{6dB}=%.1f$' % (
		cutoff_y_3db, cutoff_y_6db)
	ax_x_spectrum.text(0.5, 0.45, textstr,
	                   transform=ax_x_spectrum.transAxes,
	                   fontsize=14, verticalalignment='top',
	                   horizontalalignment='center', bbox=box)

	# Spectrum along f_x
	base = ax_y_spectrum.transData
	rot = transforms.Affine2D().rotate_deg(270)
	win_spectrum_x.squeeze().plot(ax=ax_y_spectrum,
	                              transform=rot + base)
	ax_y_spectrum.set_yscale('symlog', linthreshy=min_freq_x,
	                                   subsy=[2, 3, 4, 5, 6, 7, 8, 9])
	ax_y_spectrum.set_xlim([-200, 20])
	ax_y_spectrum.grid()
	ax_y_spectrum.set_ylabel("")
	ax_y_spectrum.set_xlabel("Normalized magnitude [dB]")
	textstr = '$\lambda_x^{3dB}=%.1f$ \n $\lambda_x^{6dB}=%.1f$' % (
		cutoff_x_3db, cutoff_x_6db)
	ax_y_spectrum.text(0.7, 0.5, textstr, transform=ax_y_spectrum.transAxes,
	                                      fontsize=14,
	                                      verticalalignment='center',
	                                      horizontalalignment='right',
	                                      bbox=box)

	# Full 2d spectrum
	win_spectrum_norm.plot(ax=ax_2D_spectrum,
	                       add_colorbar=False,
	                       vmin=-200,
	                       vmax=0,
	                       cmap=matplotlib.cm.Spectral_r)
	ax_2D_spectrum.set_xscale('symlog', linthreshx=min_freq_y)
	ax_2D_spectrum.set_yscale('symlog', linthreshy=min_freq_x)