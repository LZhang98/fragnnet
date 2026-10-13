import numpy as np
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(
        [
            "src/fragnnet/models/fragnnet/dag/compute_frags.pyx",
            "src/fragnnet/models/masster/candidates/cut_mask_kernels.pyx",
            "src/fragnnet/models/fragnnet/dag/multi_cut_bfs.pyx",
            "src/fragnnet/baselines/massformer/algos.pyx",
        ],
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[np.get_include()],
)
