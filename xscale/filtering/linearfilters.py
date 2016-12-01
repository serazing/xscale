"""Define functions for linear filtering that works on multi-dimensional
xarray.DataArray and xarray.Dataset objects.
"""
# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
# Internal
import copy
# Numpy and scipy
import numpy as np
import scipy.signal.windows as win
import scipy.ndimage as im
import scipy.special as spec
# Xarray and dask
from dask.diagnostics import ProgressBar
import xarray as xr
# Matplotlib
import pylab as plt
from matplotlib import gridspec
# Current package
from .. import utils
from ..plot.plot2d import spectrum2d_plot
from ..plot.plot1d import spectrum_plot


# ------------------------------------------------------------------------------
# First part : Definition of custom window functions
# ------------------------------------------------------------------------------
def _lanczos(n, fc=0.02):
	"""
	Compute the coefficients of a Lanczos window

	Parameters
	----------
	n : int
		Number of points in the output window, must be an odd integer.
	fc : float
		Cutoff frequency

	Returns
	-------
	w : ndarray
		The weights associated to the boxcar window
	"""
	if not isinstance(n, int):
		try:
			n = int(n)
		except:
			TypeError, "n must be an integer"
	if not n % 2 == 1:
		raise ValueError, "n must an odd integer"
	k = np.arange(- n / 2 + 1, n / 2 + 1)
	# k = np.arange(0, n + 1)
	# w = (np.sin(2 * pi * fc * k) / (pi * k) *
	#     np.sin(pi * k / n) / (pi * k / n))
	w = (np.sin(2. * np.pi * fc * k) / (np.pi * k) *
	     np.sin(np.pi * k / (n / 2.)) / (np.pi * k / (n / 2.)))
	# Particular case where k=0
	w[n / 2] = 2. * fc
	return w


# Define a list of available windows
_scipy_window_dict = win._win_equiv
# Remove scipy windows that are not supported yet
del(_scipy_window_dict['dpss'])
del(_scipy_window_dict['dss'])
del(_scipy_window_dict['ggs'])
del(_scipy_window_dict['gss'])
del(_scipy_window_dict['optimal'])
del(_scipy_window_dict['slepian'])
del(_scipy_window_dict['slep'])
del(_scipy_window_dict['gaussian'])
del(_scipy_window_dict['gauss'])
del(_scipy_window_dict['general_gaussian'])
del(_scipy_window_dict['general_gauss'])
del(_scipy_window_dict['general gaussian'])
del(_scipy_window_dict['general gauss'])
del(_scipy_window_dict['cheb'])
del(_scipy_window_dict['chebwin'])
del(_scipy_window_dict['kaiser'])
del(_scipy_window_dict['ksr'])
_local_window_dict = {'lanczos': _lanczos, 'lcz': _lanczos}


# ------------------------------------------------------------------------------
# First part : Definition of window classes for filtering
# ------------------------------------------------------------------------------

def _infer_n_and_dims(n, dims):
	"""Logic for setting the window properties"""
	#TODO: Finish this function
	if utils.is_dict_like(n):
		# Dictionary case
		new_n = n.values()
		new_dims = n.keys()
	elif utils.is_sequence(n):
		# Iterable case
		raise NotImplementedError
	elif isinstance(n, int):
		# Scalar case
		if utils.is_sequence(dims):
			new_n = []
			new_dims = []
			for dim in dims:
				new_n.append(n)
				new_dims.append(dim)
		elif utils.is_scalar(dims):
			new_n = [n]
			new_dims = [dims]
		else:
			raise TypeError
	else:
		raise TypeError
	return new_n, new_dims

#@xr.register_dataset_accessor('win')
@xr.register_dataarray_accessor('win')
class Window(object):
	"""
	Class for all different type of windows
	"""

	_attributes = ['name', 'dims', 'order']

	def __init__(self, xarray_obj):
		self._obj = xarray_obj
		self.obj = xarray_obj
		self.name = 'boxcar'
		self.dims = None
		self.order = 2
		self.coefficients = 1.
		self.coords = []
		self._depth = dict()

	def __repr__(self):
		"""
		Provide a nice string representation of the window object
		"""
		# Function copied from xarray.core.rolling
		attrs = ["{k}->{v}".format(k=k, v=getattr(self, k))
		         for k in self._attributes if
		         getattr(self, k, None) is not None]
		return "{klass} [{attrs}]".format(klass=self.__class__.__name__,
		                                  attrs=','.join(attrs))

	def set(self, window_name=None, n=None, dims=None, chunks=None, **kargs):
		"""
		Set the different properties of the current window

        If the variable associated to the window objetc is a non-dask array,
        it will be converted to dask array. If it's a dask array, it will be
        rechunked to the given chunksizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

		Parameters
		----------
		window_name : str
			Name of the window among the available windows (see Notes below)
		n : int, sequence or dict
			Half-size of the window
		dims : str or sequence, optional
			Name of the dimension of the window. If not precised, the first dimension
			of the data is taken by default.
        chunks : int, tuple or dict, optional
            Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or ``{'x': 5, 'y': 5}``.
		keywords
			Additional parameters specific to some windows functions (see the
			Notes below)

		"""
		self.obj = self._obj.chunk(chunks=chunks)
		if window_name is not None:
			self.name = window_name
		if dims is None and self.dims is None:
			dims = self._obj.dims[0]
		if isinstance(dims, str):
			self.dims = [dims, ]
		else:
			self.dims = dims
		# Check if the dimensions of the window correspond to dimensions of the data
		for dim in self.dims:
			if not dim in self._obj.dims:
				raise ValueError, ("Dimension " + dim + " does not exist in the data")
		if window_name in _scipy_window_dict:
			window_function = _scipy_window_dict[window_name]
		elif window_name in _local_window_dict:
			window_function = _local_window_dict[window_name]
		else:
			raise ValueError, "This type of window is not supported"
		self.order = dict()
		self.coefficients = 1.
		self.coords = []
		for weight_numbers, dimension_name in zip(n, dims):
			self.order[dimension_name] = weight_numbers
			# Compute the coefficients associated to the window using the right
			# function
			coefficients1d = window_function(2 * weight_numbers + 1, **kargs)
			# Normalize the coefficients
			coefficients1d /= np.sum(coefficients1d)
			self.coefficients = np.outer(self.coefficients, coefficients1d)
			#TODO: Try to add the rotational convention using meshgrid, in complement to the outer product
			#TODO: check the order of dimension of the kernel compared to the DataArray/DataSet objects
		self.coefficients = self.coefficients.squeeze()
		for dim in self.obj.dims:
			axis_dim = self.obj.get_axis_num(dim)
			if dim not in self.dims:
				self.coefficients = np.expand_dims(self.coefficients, axis=axis_dim)
			else:
				self._depth[self.obj.get_axis_num(dim)] = self.order[dim]
			self.coords.append(np.asarray(self.obj.coords[dim]))

	def apply(self, mode='reflect', weights=None, compute=True):
		"""Convolve the current window with the data

		Parameters
		----------
		mode : {'reflect', 'same', 'valid'}, optional

		weights :

		compute : bool,

		Returns
		-------
		res : xarray.DataArray or DataSet
			Return the filtered  the low-passed filtered
		"""
		# Check if the data has more dimensions than the window and add
		# extra-dimensions to the window if it is the case
		mask = self.obj.notnull()
		if weights is None:
			weights = im.convolve(mask.astype(float), self.coefficients, mode=mode)
		filled_data = self.obj.fillna(0.).data

		def convolve(x):
			xf = im.convolve(x, self.coefficients, mode=mode)
			return xf

		data = filled_data.map_overlap(convolve, depth=self._depth, boundary=mode, trim=True)
		if compute:
			with ProgressBar():
				out = data.compute()
		else:
			out = data
		res = xr.DataArray(out, dims=self.obj.dims, coords=self.coords, name=self.obj.name) / weights

		return res.where(mask == 1)

	def boundary_weights(self, mode='reflect', drop_dims=None):
		"""
		Compute the boundary weights

		Parameters
		----------
			mode:

			drop_dims:
				Specify dimensions along which the mask is constant

		Returns
		-------
		"""
		mask = self.obj.notnull()
		new_dims = copy.copy(self.obj.dims)
		new_coords = copy.copy(self.coords)
		for dim in drop_dims:
			#TODO: Make the function work
			mask = mask.isel({dim:0})
			del(new_dims[dim])
			del(new_coords[dim])
		weights = im.convolve(mask.astype(float), self.coefficients, mode=mode)
		res = xr.DataArray(weights, dims=new_dims, coords=new_coords, name='boundary weights')
		return res.where(mask == 1)

	def plot(self, format = 'landscape'):
		"""
		Plot the weights distribution of the window and the associated
		spectrum (work only for 1D and 2D windows).
		"""
		nod = len(self.dims)
		if nod == 1:
			# Compute 1D spectral response
			spectrum = np.fft.fft(self.coefficients.squeeze(), 2048) / (len(self.coefficients.squeeze()) / 2.0)
			freq = np.linspace(-0.5, 0.5, len(spectrum))
			response = 20 * np.log10(np.abs(np.fft.fftshift(spectrum / abs(spectrum).max())))
			# Plot window properties
			fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
			# First plot: weight distribution
			n = self._depth.values()[0]
			ax1.plot(np.arange(-n, n + 1), self.coefficients.squeeze())
			ax1.set_xlim((-n, n))
			ax1.set_title("Window: " + self.name)
			ax1.set_ylabel("Amplitude")
			ax1.set_xlabel("Sample")
			# Second plot: frequency response
			ax2.plot(freq, response)
			ax2.axis([-0.5, 0.5, -120, 0])
			ax2.set_title("Frequency response of the " + self.name + " window")
			ax2.set_ylabel("Normalized magnitude [dB]")
			ax2.set_xlabel("Normalized frequency [cycles per sample]")
			ax2.grid(True)
			plt.tight_layout()
		elif nod == 2:
			# Compute 2D spectral response
			spectrum = (np.fft.fft2(self.coefficients.squeeze(), [1024, 1024]) /
			            (np.size(self.coefficients.squeeze()) / 2.0))
			response = np.abs(np.fft.fftshift(spectrum / abs(spectrum).max()))
			fx = np.linspace(-0.5, 0.5, 1024)
			fy = np.linspace(-0.5, 0.5, 1024)
			f2d = np.meshgrid(fy, fx)
			if  format == 'landscape':
				gs = gridspec.GridSpec(2, 4, width_ratios=[2, 1, 2, 1], height_ratios=[1, 2])
				plt.figure(figsize=(11.69, 8.27))
			elif format == 'portrait':
				plt.figure(figsize=(8.27, 11.69))
			# Weight disribution along x
			ax_nx = plt.subplot(gs[0])
			ax_nx.plot(np.arange(-nx, nx + 1), self.coefficients.squeeze()[:, ny])
			ax_nx.set_xlim((-nx, nx))
			# Weight disribution along y
			ax_nx = plt.subplot(gs[5])
			ax_nx.plot(self.coefficients.squeeze()[nx, :], np.arange(-ny, ny + 1))
			ax_nx.set_ylim((-ny, ny))
			# Full 2d weight distribution
			ax_n2d = plt.subplot(gs[4])
			nx2d, ny2d = np.meshgrid(np.arange(-nx, nx + 1), np.arange(-ny, ny + 1), indexing='ij')
			ax_n2d.pcolormesh(nx2d, ny2d, self.coefficients.squeeze())
			ax_n2d.set_xlim((-nx, nx))
			ax_n2d.set_ylim((-ny, ny))
			box = dict(facecolor='white', pad=10.0)
			ax_n2d.text(0.97, 0.97, r'$w(n_x,n_y)$', fontsize='x-large', bbox=box, transform=ax_n2d.transAxes,
			            horizontalalignment='right', verticalalignment='top')
			# Frequency response for fy = 0
			ax_fx = plt.subplot(gs[2])
			spectrum_plot(ax_fx, fx, response[:, 512].squeeze(),)
			# ax_fx.set_xlim(xlim)
			ax_fx.grid(True)
			ax_fx.set_ylabel(r'$R(f_x,0)$', fontsize=24)
			# Frequency response for fx = 0
			ax_fy = plt.subplot(gs[7])
			spectrum_plot(ax_fy, response[:, 512].squeeze(), fy)
			#ax_fy.set_ylim(ylim)
			ax_fy.grid(True)
			ax_fy.set_xlabel(r'$,R(0,f_y)$', fontsize=24)
			# Full 2D frequency response
			ax_2d = plt.subplot(gs[6])
			spectrum2d_plot(ax_2d, fx, fy, response, zlog=True)
			ax_2d.set_ylabel(r'$f_y$', fontsize=24)
			ax_2d.set_xlabel(r'$f_x$', fontsize=24)
			ax_2d.grid(True)
			box = dict(facecolor='white', pad=10.0)
			ax_2d.text(0.97, 0.97, r'$R(f_x,f_y)$', fontsize='x-large', bbox=box, transform=ax_2d.transAxes,
			           horizontalalignment='right', verticalalignment='top')
			plt.tight_layout()
		else:
			raise ValueError, "This number of dimension is not supported by the plot function"


def tapper(data, window_name, dims, **kargs):
	"""
	Do a tappering of the data using any window among the available windows (
	see the notes below)

	Parameters
	----------
	data : Datarray
		The data on which apply the window function
	window_name : string
		The name of the window to use
	dims : string or tuple of string
		Dimensions of the window
	keywords arguments:
		Any arguments relative to the additional parameters of the window
		function (see the notes below)

	Returns
	-------
	data_tappered : dask array
		The data tappered y the window

	Notes
	-----
	"""
	# TODO: Write the function using the Window class
	pass
