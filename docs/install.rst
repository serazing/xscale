Install Xscale
==============

For the moment, you can only install xscale from source.


Install from Source
-------------------

To install dask from source, clone the repository from `github
<https://github.com/serazing/xscale>`_::

    git clone https://github.com/serazing/xscale.git
    cd xscale
    python setup.py install

or use ``pip`` locally if you want to install all dependencies as well::

    pip install -e .[complete]

You can view the list of all dependencies within the ``extras_require`` field
of ``setup.py``.


Test
----

Test dask with ``py.test``::

    cd xscale
    py.test xscale
