from setuptools import setup, find_packages

from __version__ import __version__

setup(
    name='AutoPrune',
    version=__version__,

    url='https://github.com/fpavesi11/AutoPrune',
    author='Federico Pavesi',
    author_email='f.pavesi11@campus.unimib.it',

    packages=find_packages(),
)
