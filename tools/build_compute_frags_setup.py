from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "fragnnet.frag.compute_frags",
    ["src/fragnnet/frag/compute_frags.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3"],
)

setup(
    name="build_compute_frags",
    ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
)
