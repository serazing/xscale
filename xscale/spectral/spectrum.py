# Xarray
import xarray as xr
# Numpy
import numpy as np
# Dask
import dask.array as da
# Pandas
import pandas as pd
# Internals
import copy


def psd(array, dim=None, detrend=None, tappering=None, shift=False, chunks=None):
	"""
	Compute the power spectrum

	Parameters
	----------
	array : xarray.DataArray
		Array from which compute the spectrum
	dims : str or sequence
		Dimensions along which to compute the spectrum
        chunks : int, tuple or dict, optional
            Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or ``{'x': 5, 'y': 5}``
	Returns
	-------
	spectrum : xarray.DataArray
		Spectral array computed over the different arrays
	"""
	spec = spectrum(array, dim=dim, detrend=detrend, tappering=tappering, shift=shift, chunks=chunks)
	#TODO: Make the correct normalization for the power spectrum and check with the Parseval theorem
	psd = (spec * da.conj(spec)).real
	psd['description'] = 'Power spectrum density performed along dimension(s) %s ' % dim
	return psd


def spectrum(array, dim=None, detrend=None, tappering=None, shift=False, chunks=None):
	"""
	Compute the spectrum on several dimensions of xarray.DataArray objects using the Fast Fourrier Transform
	parrallelized with dask.array.

	Parameters
	----------
	array : xarray.DataArray
		Array from which compute the spectrum
	dim : str or sequence
		Dimensions along which to compute the spectrum
	detrend : {None, 'zeromean', 'linear'}, optional
		Remove a the mean or a linear trend,
	tappering : bool, optional
		If True, tapper the data with a Tukey window
	shift : bool,
		If True,
	chunks : int, tuple or dict, optional
		Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or ``{'x': 5, 'y': 5}``

	Returns
	-------
	res :
		A new

	Notes
	-----
	If the input data is real, a real fft is performed over the first
	dimension, which is faster. Then the transform over the remaining
	dimensions are computed with the classic fft.
	"""
	if dim is None:
		dim = array.dims
	if tappering is not None:
		# TODO: implement the tappering function
		pass
	else:
		if detrend is 'zeromean':
			# TODO: remove the mean here
			preproc_array = array - array.mean(dim=dim)
		elif detrend is 'linear':
			# TODO: implement the detrending function
			# preproc_array = _detrend(array, dims)
			pass
		else:
			preproc_array = array
	if tappering is not None:
		pass
	spectrum, spectrum_coords, spectrum_dims = _fft(preproc_array, dim, shift=shift, chunks=chunks)
	return xr.DataArray(spectrum, coords=spectrum_coords, dims=spectrum_dims, name='spectrum')


def _fft(array, dim, shift=False, chunks=None):
	"""This function is for private use only.
	"""
	spectrum = array.chunk(chunks=chunks).data
	spectrum_coords = dict()
	spectrum_dims = tuple()
	for di in array.dims:
		if di not in dim:
			spectrum_dims += (di,)
			spectrum_coords[di] = np.asarray(array[di])
		else:
			spectrum_dims += ('f_' + di,)
	chunks = copy.copy(spectrum.chunks)
	shape = spectrum.shape
	first = True
	for di in dim:
		if di in array.dims:
			axis_num = array.get_axis_num(di)
			nfft = shape[axis_num]

			# Compute the resolution of the different dimension
			delta = np.diff(array[di])
			if pd.core.common.is_timedelta64_dtype(delta):
				# Convert to seconds so we get hertz
				delta = delta.astype('timedelta64[s]').astype('f8')
			if not np.allclose(delta, delta[0]):
				raise Warning, "Coordinate %s is not evenly spaced" % di
			dx = delta[0]

			#FFT part
			if first and not np.iscomplexobj(spectrum):
				# The first FFT is performed on real numbers: the use of rfft is faster
				spectrum_coords['f_' + di] = np.fft.rfftfreq(nfft, dx)
				spectrum = da.fft.rfft(spectrum.rechunk({axis_num: nfft}), axis=axis_num) \
					.rechunk({axis_num: chunks[axis_num][0]})
			else:
				# The successive FFTs are performed on complex numbers: need to use classic fft
				spectrum_coords['f_' + di] = np.fft.fftfreq(nfft, dx)
				spectrum = da.fft.fft(spectrum.rechunk({axis_num: nfft}), axis=axis_num) \
					.rechunk({axis_num: chunks[axis_num][0]})
				if shift is True:
					spectrum_coords['f_' + di] = np.fft.fftshift(spectrum_coords['f_' + di])
					spectrum = np.fft.fftshift(spectrum, axes=axis_num)
			first = False
		else:
			raise Warning, "Cannot find dimension %s in DataArray" % di
	return spectrum, spectrum_coords, spectrum_dims


def _tapper(array, dim):
	"""Perform a tappering of the data over the specified dimensions with a tukey window
	"""
	pass

def _detrend():
	pass

