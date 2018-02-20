======
Xscale
======

*Xscale a library of multi-dimensional signal processing tools using parallel
computing*


Principle
---------

Xscale is defined to work with multi-dimensional self-described arrays using
the `xarray`_ objects. Xscale also benefits from parallel computation of
several `numpy`_ and `scipy`_ functions implemented in the `dask`_ library.
Most of the tools found in xscale are developed to analyse and separate time
and spatial scales.





.. note::

   For the moment, the API is unstable and likely to change without notice.


Index
-----

**Getting Started**

* :doc:`install`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   install.rst


**Filtering**

* :doc:`window`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Filtering methods

   window.rst

**Spectral**

* :doc:`fft`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Spectral estimates

   fft.rst

**Fitting**

* :doc:`fitting`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Signal methods

   fitting.rst

**API**

* :doc:`api`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API

   api.rst



Get in touch
------------

- Report bugs, suggest feature ideas or view the source code `on GitHub`_.

.. _on GitHub: http://github.com/serazing/xscale
.. _xarray: http://xarray.pydata.org
.. _dask: http://dask.pydata.org
.. _scipy: https://www.scipy.org
.. _numpy: http://www.numpy.org/