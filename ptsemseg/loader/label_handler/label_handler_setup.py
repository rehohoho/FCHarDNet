import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "label_handler",
        ["label_handler.pyx"]
        #extra_compile_args=['/openmp'],
        #extra_link_args=['/openmp'],
    )
]

setup(
  name = 'MyProject',
  ext_modules = cythonize(ext_modules, annotate=True),
  include_dirs=[numpy.get_include()]
)