import numpy

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("minkfncts2d",
              ["minkfncts2d.pyx"],
              libraries=["m"],
              extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=["-fopenmp"],
              include_dirs=[numpy.get_include()]
             )
]

setup(
    version = '1.0',
    name = "minkfncts2d",
    author = 'Moutaz Haq',
    description = 'An implemention of the 2D Minkowski Functionals using Cython and OpenMP.',
    url = 'https://github.com/moutazhaq/minkfncts2d',
    keywords = 'Minkowski Functionals',
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)

