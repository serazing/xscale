.. _fft:

Fast Fourier Transform
======================

.. ipython:: python
   :suppress:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt

The package ``xscale.spectral.fft`` provides easy and comprehensive functions
to estimate a multi-dimensional spectrum using the Fast Fourier Transform
(FFT). Out-of-core computation as well as parallelization are made possible
by using FFT functions from :py:mod:`dask.array.fft`.


Using multi-dimensional FFT
---------------------------

The spectrum over one or multiple dimensions of a :py:class:`xarray.DataArray`
can be estimated with the FFT using the main function
:py:func:`xscale.spectral.fft.fft`.

For a one-dimensional Fourier transform:

.. math::

   S(f_x, f_y, f_t) = \left | S(f_x, f_y, t) \right | ^2 = \Re[S]^2 + \Im[S]^2,

For a two-dimensional Fourier transform:

.. math::

   S(f_x, f_y, t) = \left | S(f_x, f_y, t) \right | ^2 = \Re[S]^2 + \Im[S]^2,

This function takes optionally the following parameters:

* *Spectrum parameters*
   - ``nfft``: the number of points used for the FFT
   - ``dim``: the dimensions over which the FFT will be performed
   - ``dx``: the resolution of the dimensions ; if the resolution is not
     precised, it will be inferred from the dimensions associated with the data
* *Data pre-processing*
   - ``detrend``: Precise the manner to detrend the data
   - ``tapering``: Multiply by a ``Tukey`` window with a coefficient a 0.25



.. note::
   - The parameters ``nfft``, ``dx`` are very flexible and may either be a
     number, a list/tuple or a dictionary.
   - Only the `mean` option is available for the moment to detrend the data
     before FFT computation.
   - The ``tapering`` option is still experimental, it should be use with
     cautious

One dimensional FFT
*******************

Let us start by importing a three-dimensional signal as an example:

.. ipython:: python

   import xscale.signal.generator as xgen
   foo = xgen.example_xyt()
   foo
   @savefig fft_example_time.png width=100%
   foo.isel(x=25, y=25).plot();

Here, we only want to estimate the spectrum along the ``time`` dimension, but
for every grid point.

.. ipython:: python

   import xscale.spectral.fft as xfft
   foo_time_spectrum = xfft.fft(foo, dim='time', dx=1., detrend='mean',
   tapering=True)
   foo_time_spectrum


Two dimensional FFT
*******************

.. ipython:: python

   @savefig fft_example_xy.png width=100%
   foo.isel(time=0).plot();


.. ipython:: python

   foo_yx_spectrum = xfft.fft(foo, dim=['y', 'x'], detrend='mean')
   foo_yx_spectrum



Spectrum normalization
----------------------

The function :py:func:`xscale.spectral.fft.fft` returns a complex spectrum
:math:`S(f_x,f_y, t)`, which is not straightforward to interpret in a physical
sense. There exist several quantities and normalization that can be derived
from the complex spectrum, which are useful to give a physical interpretation to
the spectral estimates. We detail here the different quantities that
``xscale.spectral.fft`` is able to compute. All normalization methods involve a
:py:mod:`dask.array` functions so that they can be easily combined with the FFT
computation to increment a dask graph.

.. ipython:: python
   foo_time_spectrum.attrs


Amplitude spectrum
******************

The amplitude spectrum is simply the squared sample modulus of the spectrum
:math:`S`:

.. math::

   A(f_x, f_y, t) = \left | S(f_x, f_y, t) \right | ^2 = \Re[S]^2 + \Im[S]^2,

where :math:`\Re[S]` and :math:`\Im[S]` are the real and the imaginary parts of
the spectrum, respectively. The amplitude spectrum can be computed from the
previous example using the function :py:func:`xscale.spectral.fft.amplitude`.

.. ipython:: python

   from xscale.spectral.tools import plot_spectrum
   foo_time_amplitude = xfft.amplitude(foo_time_spectrum)
   foo_time_amplitude
   @savefig fft_amplitude_spectrum_time.png width=100%
   plot_spectrum(foo_time_amplitude.isel(x=25, y=25));

Phase spectrum
**************

.. math::

   \phi(f_x, f_y, t) = \arg(S) = \arctan(\frac{\Im[S]}{\Re[S]})

.. ipython:: python

   foo_time_phase = xfft.phase(foo_time_spectrum)
   foo_time_phase
   @savefig fft_phase_spectrum_time.png width=100%
   plot_spectrum(foo_time_phase.isel(x=25, y=25), xlog=True, color='r');

Power spectrum (PS)
*******************

.. math::

   PS(f) = \frac{A(f_x, f_y, t)}{N_x^2 N_y^2}

.. ipython:: python

   foo_time_ps = xfft.ps(foo_time_spectrum)
   foo_time_ps
   @savefig fft_power_spectrum_time.png width=100%
   plot_spectrum(foo_time_ps.isel(x=25, y=25), variance_preserving=True);

Power spectrum density (PSD)
****************************

.. math::

   PSD(f) = \frac{A(f_x, f_y, t)}{(fs_x N_x) (fs_y N_y)}

.. ipython:: python

   foo_time_psd = xfft.ps(foo_time_spectrum)
   foo_time_psd
   @savefig fft_power_spectrum_density_time.png width=100%
   plot_spectrum(foo_time_ps.isel(x=25, y=25), loglog=True);


Cross spectrum
--------------

This function is not implemented yet but will be available soon.


