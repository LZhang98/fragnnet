from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "src/frag_gnn/frag/compute_frags.pyx",
        compiler_directives={'language_level':"3"},
    )
)
