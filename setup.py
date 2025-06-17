from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("ernesto/adaptation/cython_loss.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)