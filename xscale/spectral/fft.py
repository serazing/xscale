# Xarray
import xarray as xr
# Numpy
import numpy as np
from numpy.compat import integer_types
from numpy.core import integer

# Dask
import dask.array as da
# Internals
import copy
# Xscale
from .. import _utils
# Warnings
import warnings

integer_types = integer_types + (integer,)


def amplitude(spectrum):
	"""
	Return the amplitude spectrum from the Fourier Transform

	Parameters
	----------
	spectrum : DataArray
		A DataArray spectrum computed using xscale.spectral.fft.fft
	deg : bool, optional
		If True, return the phase spectrum in degrees. Default is to return
		the phase spectrum in radians

	Returns
	-------
	res : DataArray
		The phase spectrum
	"""
	return abs(spectrum)


def phase(spectrum, deg=False):
	"""
	Return the phase spectrum from the Fourier Transform

	Parameters
	----------
	spectrum : DataArray
		A DataArray spectrum computed using xscale.spectral.fft.fft
	deg : bool, optional
		If True, return the phase spectrum in degrees. Default is to return
		the phase spectrum in radians

	Returns
	-------
	res : DataArray
		The phase spectrum
	"""
	return xr.DataArray(da.angle(spectrum.data, deg), coords=spectrum.coords,
	                    dims=spectrum.dims, name='Phase Spectrum',
	                    attrs=spectrum.attrs)


def ps(spectrum):
	"""
	Return the Power Spectrum (PS) from the Fourier Transform

	Parameters
	----------
	spectrum : DataArray
		A DataArray spectrum computed using xscale.spectral.fft.fft

	Returns
	-------
	power_spectrum : DataArray
		The PS spectrum
	"""
	power_spectrum = spectrum.attrs['ps_factor'] * amplitude(spectrum) ** 2
	if spectrum.name is None:
		power_spectrum.name = 'PS'
	else:
		power_spectrum.name = 'PS_' + spectrum.name
	power_spectrum.attrs['description'] = 'Power Spectrum (PS)'
	return power_spectrum


def psd(spectrum):
	"""
	Return the Power Spectrum density (PSD) from the Fourier Transform

	Parameters
	----------
	spectrum : DataArray
		A DataArray spectrum computed using xscale.spectral.fft.fft

	Returns
	-------
	power_spectrum_density : DataArray
		The PSD spectrum
	"""
	power_spectrum_density = (amplitude(spectrum) ** 2 *
	                          spectrum.attrs['psd_factor'])
	if spectrum.name is None:
		power_spectrum_density.name = 'PSD'
	else:
		power_spectrum_density.name = 'PSD_' + spectrum.name
	power_spectrum_density.attrs['description'] = ('Power Spectrum Density '
	                                               '(PSD)')
	return power_spectrum_density


def fft(array, dim=None, nfft=None, dx=None, detrend=None, tapering=False,
        shift=True, sym=False, chunks=None):
	"""Compute the spectrum on several dimensions of xarray.DataArray objects
	using the Fast Fourrier Transform parallelized with dask.

	Parameters
	----------
	array : xarray.DataArray
		Array from which compute the spectrum
	dim : str or sequence
		Dimensions along which to compute the spectrum
	nfft : float or sequence, optional
		Number of points used to compute the spectrum
	dx : float or sequence, optional
		Define the resolution of the dimensions. If not precised,
		the resolution is computed directly from the coordinates associated
		to the dimensions.
	detrend : {None, 'mean', 'linear'}, optional
		Remove the mean or a linear trend before the spectrum computation
	tapering : bool, optional
		If True, tapper the data with a Tukey window
	shift : bool, optional
		If True, the frequency axes are shifted to center the 0 frequency,
		otherwise negative frequencies follow positive frequencies as in
		numpy.fft.ftt
	sym : bool, optional
		If True, force the spectrum to be symmetrical even if the input data
		is real
	chunks : int, tuple or dict, optional
		Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
		``{'x': 5, 'y': 5}``

	Returns
	-------
	res : DataArray
		A multi-dimensional complex DataArray with the corresponding
		dimensions transformed in the Fourier space.

	Notes
	-----
	If the input data is real, a real fft is performed over the first
	dimension, which is faster. Then the transform over the remaining
	dimensions are computed with the classic fft.
	"""
	temp_nfft, new_dim = _utils.infer_n_and_dims(array, nfft, dim)
	new_nfft = _utils.infer_arg(temp_nfft, dim)
	new_dx = _utils.infer_arg(dx, dim)
	if detrend is 'mean':
		# Tackling the issue of the dask graph by computing and loading the
		# mean here
		for di in new_dim:
			mean_array = array.mean(dim=di).load()
			preproc_array = array - mean_array
	elif detrend is 'linear':
		preproc_array = _detrend(array, new_dim)
	else:
		preproc_array = array
	if tapering:
		preproc_array = _tapper(array, new_dim)
	# TODO: Check if this part may work with dask using np.iscomplexobj
	# If the array is complex, set the symmetry parameters to True
	if np.any(np.iscomplex(array)):
		sym = True
	spectrum_array, spectrum_coords, spectrum_dims = \
		_fft(preproc_array, new_dim, new_nfft, new_dx, shift=shift,
		     chunks=chunks, sym=sym)
	if not array.name:
		name = 'spectrum'
	else:
		name = 'F_' + array.name
	spec = xr.DataArray(spectrum_array, coords=spectrum_coords,
	                    dims=spectrum_dims, name=name)
	_compute_norm_factor(spec, new_nfft, new_dim, new_dx, tapering, sym=sym)
	return spec


def ifft(spectrum_array, dim=None, n=None, shift=True, real=True, chunks=None):
	"""Perform the inverse Fourier transformCompute the field associated with
	the spectrum on
	several dimensions of
	xarray.DataArray objects
	using the Fast Fourrier Transform parallelized with dask.

	Parameters
	----------
	spectrum_array : xarray.DataArray
		Spectral array with
	dim : str or sequence
		Name of the original dimensions used to compute
	n : float or sequence, optional
	shift : bool, optional
		If True, the input spectrum have the frequency axes center
		the 0 frequency.
	real : bool, optional
		If True, the inverse Fourier transform is forced to return a real
		output by applying np.fft.irfft to the first dimension
	chunks : int, tuple or dict, optional
		Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
		``{'x': 5, 'y': 5}``

	Returns
	-------
	res : DataArray
		A multi-dimensional complex DataArray with the corresponding
		dimensions transformed in the Fourier space.

	Notes
	-----
	If the input data is real, a real fft is performed over the first
	dimension, which is faster. Then the transform over the remaining
	dimensions are computed with the classic fft.
	"""
	_, new_dim = _utils.infer_n_and_dims(spectrum_array, n, dim)
	new_n = _utils.infer_arg(n, new_dim, default_value=None)
	print(new_dim, new_n)
	array, coords, dims = _ifft(spectrum_array, new_dim, new_n, shift=shift,
	                            real=real, chunks=chunks)
	data = xr.DataArray(array, coords=coords, dims=dims)
	return data


def _fft(array, dim, nfft, dx, shift=False, sym=True, chunks=None):
	"""This function is for private use only.
	"""
	spectrum_array = array.chunk(chunks=chunks).data
	spectrum_coords = dict()
	spectrum_dims = tuple()
	for di in array.dims:
		if di not in dim:
			spectrum_dims += (di,)
			spectrum_coords[di] = np.asarray(array[di])
		else:
			spectrum_dims += ('f_' + di,)
	chunks = copy.copy(spectrum_array.chunks)
	first = True
	for di in dim:
		if di in array.dims:
			axis_num = array.get_axis_num(di)
			axis_size = array.sizes[di]
			# Compute the resolution of the different dimension
			if dx[di] is None:
				dx[di] = _utils.get_dx(array, di)
			#FFT part
			if first and not sym:
				# The first FFT is performed on real numbers: the use of rfft
				# is faster
				spectrum_coords['f_' + di] = np.fft.rfftfreq(nfft[di], dx[di])
				spectrum_array = \
					(da.fft.rfft(spectrum_array.rechunk({axis_num: axis_size}),
					             n=nfft[di],
					             axis=axis_num).
					 rechunk({axis_num: chunks[axis_num][0]}))
			else:
				# The successive FFTs are performed on complex numbers: need to
				# use classic fft
				spectrum_coords['f_' + di] = np.fft.fftfreq(nfft[di], dx[di])
				spectrum_array = \
					(da.fft.fft(spectrum_array.rechunk({axis_num: axis_size}),
					            n=nfft[di],
					            axis=axis_num).
				    rechunk({axis_num: chunks[axis_num][0]}))
				if shift:
					spectrum_coords['f_' + di] = \
						np.fft.fftshift(spectrum_coords['f_' + di])
					spectrum_array = _fftshift(spectrum_array, axes=axis_num)
			first = False
		else:
			warnings.warn("Cannot find dimension %s in DataArray" % di)
	return spectrum_array, spectrum_coords, spectrum_dims


def _ifft(spectrum_array, dim, n, shift=False, real=False, chunks=None):
	"""This function is for private use only.
	"""
	array = spectrum_array.chunk(chunks=chunks).data
	array_coords = dict()
	array_dims = tuple()
	spectrum_dims = tuple()
	real_di = None
	for di in spectrum_array.dims:
		if (di[:2] == 'f_') and (di in dim):
			array_dims += (di[2:],)
			spectrum_dims += (di, )
		else:
			array_dims += (di,)
			array_coords[di] = np.asarray(spectrum_array[di])
	chunks = copy.copy(array.chunks)
	for di in spectrum_dims:
		axis_num = spectrum_array.get_axis_num(di)
		axis_size = spectrum_array.sizes[di]
		axis_coord = spectrum_array.coords[di]
		if np.all(axis_coord >= 0):
			# If there are only positive frequencies, the dimension is supposed
			# to have been created by rfft, thus irfft is applied later
			real_di = di
		else:
			# Other dimensions are supposed to have been created by fft,
			# thus ifft is applied
			if shift:
				array = _ifftshift(array, axes=axis_num)
			array = (da.fft.ifft(array.rechunk({axis_num: axis_size}),
			                     n=n[di],
			                     axis=axis_num).
				    rechunk({axis_num: chunks[axis_num][0]}))
	# irfft is applied here if there is a dimension with only positive
	# frequencies
	if real_di:
		axis_num = spectrum_array.get_axis_num(real_di)
		axis_size = spectrum_array.sizes[real_di]
		array = (da.fft.irfft(array.rechunk({axis_num: axis_size}),
	                          n=n[di],
	                          axis=axis_num).
	            rechunk({axis_num: chunks[axis_num][0]}))
	if real:
		array = array.real
	return array, array_coords, array_dims


def _detrend(array, dim):
	# TODO: implement the detrending function
	raise NotImplementedError("The linear detrending option is not implemented "
	                          "yet.")


def _tapper(array, dim, window=('tukey', 0.25)):
	"""Perform a tappering of the data over the specified dimensions with a tukey window
	"""
	# TODO: improve the tapering function by multitapering
	win = array.window
	win.set(dim=dim, window=window)
	return win.tapper()


def _compute_norm_factor(array, nfft, dim, dx, tapering, sym=True):
	"""Compute the normalization factor for Power Spectrum and Power Spectrum Density
	"""
	try:
		ps_factor = array.attrs['ps_factor']
	except KeyError:
		ps_factor = 1.
	try:
		psd_factor = array.attrs['psd_factor']
	except KeyError:
		psd_factor = 1.
	first = True
	for di in dim:
		# Get the sampling frequency
		fs = 1. / dx[di]
		if tapering:
			#TODO: Make a correct normalization by computing the weights of
			# window used for the tapering
			s1 = nfft[di]
			s2 = s1
		else:
			s1 = nfft[di]
			s2 = s1
		ps_factor /= s1 ** 2
		psd_factor /= fs * s2
		if first and not sym:
			ps_factor *= 2.
			psd_factor *= 2.
		first = False
	array.attrs['ps_factor'] = ps_factor
	array.attrs['psd_factor'] = psd_factor


def _fftshift(x, axes=None):
	"""Similar to numpy.fft.fttshift but based on dask.array"""
	if axes is None:
	    axes = list(range(x.ndim))
	elif isinstance(axes, integer_types):
	    axes = (axes,)
	for k in axes:
	    n = x.shape[k]
	    p2 = (n + 1) // 2
	    mylist = np.concatenate((np.arange(p2, n), np.arange(p2)))
	    x = da.take(x, mylist, k)
	return x


def _ifftshift(x, axes=None):
	"""Similar to numpy.fft.ifttshift but based on dask.array"""
	if axes is None:
	    axes = list(range(x.ndim))
	elif isinstance(axes, integer_types):
	    axes = (axes,)
	for k in axes:
	    n = x.shape[k]
	    p2 = n - (n + 1) // 2
	    mylist = np.concatenate((np.arange(p2, n), np.arange(p2)))
	    x = da.take(x, mylist, k)
	return x