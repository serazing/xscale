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
import xarray as xr
# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import transforms
# Current package
from .. import _utils
from ..spectral.fft import fft, psd

import pdb

@xr.register_dataarray_accessor('window')
@xr.register_dataset_accessor('window')
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
			Window order over dimensions specified through an integer coupled
			with the ``dim`` parameter. A dictionnary can also be used to specify
			the order.
		dim : str or sequence, optional
			Names of the dimensions associated with the window.
		cutoff : float, sequence or dict, optional
			The window cutoff over the dimensions specified through a
			dictionnary or coupled with the dim parameter. If None,
			the cutoff is not used to desgin the filter.
		dx : float, sequence or dict, optional
			Define the resolution of the dimensions. If None, the resolution
			is directly infered from the coordinates associated to the
			dimensions.
		trim : bool, optional
			If True, choose to only keep the valid data not affected by the
			boundaries.
		window : string, tupple, or string and parameters values, or dict, optional
			Window to use, see :py:func:`scipy.signal.get_window` for a list
			of windows and required parameters
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
		self.coefficients = xr.DataArray(1.)
		#/!\ Modif for Dataset
		#self._depth = dict()

		# Build the multi-dimensional window: the hard part
		for di in self.obj.dims:
			#/!\ Modif for Dataset
			#axis_num = self.obj.get_axis_num(di)
			#dim_chunk = self.obj.chunks[di][0]
			if di in self.dims:
				#/!\ Modif for Dataset
				#self._depth[axis_num] = self.order[di] // 2
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
				try:
					chunks = self.obj.chunks[di][0]
				except TypeError:
					axis_num = self.obj.get_axis_num(di)
					chunks = self.obj.chunks[axis_num][0]
				n = len(coefficients1d)
				coords =  {di: np.arange(-(n - 1) // 2, (n + 1) // 2)}
				coeffs1d = xr.DataArray(coefficients1d, dims=di, 
										coords=coords).chunk(chunks=chunks) 
				self.coefficients = self.coefficients * coeffs1d
				# TODO: Try to add the rotational convention using meshgrid,
				# in complement to the outer product
				#self.coefficients = self.coefficients.squeeze()
			else:
				self.coefficients = self.coefficients.expand_dims(di, axis=-1)
			#	self.coefficients = self.coefficients.expand_dim(di, axis=-1)
			#	np.expand_dims(self.coefficients,
			#	                                   axis=axis_num)

	def convolve(self, mode='reflect', weights=1., trim=False):
		"""Convolve the current window with the data

		Parameters
		----------
		mode : {'reflect', 'periodic', 'any-constant'}, optional
			The mode parameter determines how the array borders are handled.
			Default is 'reflect'.
		weights : DataArray, optional
			Array to weight the result of the convolution close to the
			boundaries.
		trim : bool, optional
			If True, choose to only keep the valid data not affected by the
			boundaries.

		Returns
		-------
		res : xarray.DataArray
			Return a filtered DataArray
		"""
		if isinstance(self.obj, xr.DataArray):
			res = _convolve(self.obj, self.coefficients, self.dims, self.order, 
							mode, weights, trim)
		elif isinstance(self.obj, xr.Dataset):
			res = self.obj.apply(_convolve, keep_attrs=True,
								 args=(self.coefficients, self.dims, self.order, 
									   mode, weights, trim))
		return res

	def boundary_weights(self, mode='reflect', mask=None, drop_dims=[], trim=False):
		"""
		Compute the boundary weights

		Parameters
		----------
		mode : {'reflect', 'periodic', 'any-constant'}, optional
			The mode parameter determines how the array borders are handled.
			Default is 'reflect'.
		mask : array-like, optional
			Specify the mask, if None the mask is inferred from missing values
		drop_dims : list, optional
			Specify dimensions along which the weights do not need to be
			computed

		Returns
		-------
		weights : xarray.DataArray or xarray.Dataset
			Return a DataArray or a Dataset containing the weights
		"""
		# Drop extra dimensions if
		if drop_dims:
			new_coeffs = self.coefficients.squeeze()
		else:
			new_coeffs = self.coefficients
		if mask is None:
			# Select only the first
			new_obj = self.obj.isel(**{di: 0 for di in drop_dims}).squeeze()
			mask = 1. - np.isnan(new_obj)
		if isinstance(mask, xr.DataArray):
			res = _convolve(mask, new_coeffs, self.dims, self.order,
							mode, 1., trim)
		elif isinstance(mask, xr.Dataset):
			res = mask.apply(_convolve, keep_attrs=True,
								        args=(self.coefficients, self.dims, self.order,
									          mode, 1., trim))
		# Mask the output
		res = res.where(mask == 1.)
		return res

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
		win_spectrum_norm = 20 * np.log10(win_spectrum / abs(win_spectrum).max())
		self.win_spectrum_norm = win_spectrum_norm
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
	# next, should eventually be udpated in order to delete call to .values
	# https://github.com/pydata/xarray/issues/1388
	# Changed by using load()
	cutoff_3db = 1. / abs(freq[np.abs(win_spectrum_norm + 3).argmin(dim).data])
	cutoff_6db = 1. / abs(freq[np.abs(win_spectrum_norm + 6).argmin(dim).data])

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

	fig = plt.figure(figsize=(18, 9))
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

	cutoff_x_3db = 1. / abs(freq_x[np.abs(win_spectrum_x + 3).argmin(dim_fx).data])
	cutoff_x_6db = 1. / abs(freq_x[np.abs(win_spectrum_x + 6).argmin(dim_fx).data])
	cutoff_y_3db = 1. / abs(freq_y[np.abs(win_spectrum_y + 3).argmin(dim_fy).data])
	cutoff_y_6db = 1. / abs(freq_y[np.abs(win_spectrum_y + 6).argmin(dim_fy).data])

	#fig = plt.figure(1, figsize=(16, 8))
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

	
def _convolve(dataarray, coeffs, dims, order, mode, weights, trim):
		"""Convolve the current window with the data
		"""
		# Check if the kernel has more dimensions than the input data,
		# if so the extra dimensions of the kernel are squeezed
		squeezed_dims = [di for di in dims if di not in dataarray.dims]
		new_coeffs = coeffs.squeeze(squeezed_dims)
		new_coeffs /= new_coeffs.sum()
		if trim:
			mode = np.nan
			mode_conv = 'constant'
			new_data = dataarray.data
		else:
			new_data = dataarray.fillna(0.).data
			if mode is 'periodic':
				mode_conv = 'wrap'
			else:
				mode_conv = mode
		boundary = {dataarray.get_axis_num(di): mode for di in dims}
		depth = {dataarray.get_axis_num(di): order[di] // 2 for di in dims}
		conv = lambda x: im.convolve(x, new_coeffs.data, mode=mode_conv)
		data_conv = new_data.map_overlap(conv, depth=depth,
		                                       boundary=boundary,
		                                       trim=True)
		res = 1. / weights *  xr.DataArray(data_conv, dims=dataarray.dims,
		                                              coords=dataarray.coords,
		                                              name=dataarray.name)
		return res
