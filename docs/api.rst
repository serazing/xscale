#############
API reference
#############

This page provides an auto-generated summary of xscale's API.



Filtering tools
===============
.. currentmodule:: xscale

Linear filtering
----------------
.. autosummary::
   :toctree: generated/

          Window.set
          Window.convolve
          Window.boundary_weights
          Window.plot


Spectral estimates
==================

Fast Fourier Transform
----------------------
.. autosummary::
   :toctree: generated/

      spectral.fft.fft
      spectral.fft.amplitude
      spectral.fft.phase
      spectral.fft.ps
      spectral.fft.psd


Spectral tools
--------------
.. autosummary::
   :toctree: generated/

       spectral.tools.plot_spectrum
       spectral.tools.fit_power_law
       spectral.tools.plot_power_law


Signal tools
============

Signal generator
----------------
.. autosummary::
   :toctree: generated/

      signal.generator.rednoise
      signal.generator.ar
      signal.generator.window1d


Fitting methods
---------------
.. autosummary::
   :toctree: generated

      signal.fitting.polyfit
      signal.fitting.polyval
      signal.fitting.linreg
      signal.fitting.trend
      signal.fitting.detrend
      signal.fitting.sinfit
      signal.fitting.sinval
