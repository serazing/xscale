#!/usr/bin/env python

from os.path import exists
from setuptools import setup
import versioneer

extras_require = {
  'array': ['numpy', 'toolz >= 0.7.2'],
  'bag': ['cloudpickle >= 0.2.1', 'toolz >= 0.7.2', 'partd >= 0.3.6'],
  'dataframe': ['numpy', 'pandas >= 0.18.0', 'toolz >= 0.7.2',
                'partd >= 0.3.5', 'cloudpickle >= 0.2.1'],
  'distributed': ['distributed >= 1.14', 's3fs >= 0.0.7'],
  'imperative': ['toolz >= 0.7.2'],
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

packages = ['xscale', 'xscale.filtering', 'xscale.pca', 'xscale.signals', 'xscale.plot']

tests = [p + '.tests' for p in packages]


setup(name='xscale',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Signal processing tools to analyze spatio-temporal scales',
      url='http://github.com/serazing/xscale',
      maintainer='Guillaume Serazin',
      maintainer_email='guillaume.serazin@legos.obs-mip.fr',
      license='BSD',
      keywords='signal processing',
      packages=packages + tests,
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      extras_require=extras_require,
      zip_safe=False)
