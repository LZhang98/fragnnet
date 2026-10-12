"""MASSter: the bond-cut fragment model, its sampler, dataset filter, export tables and hooks.
Candidate generation (CPU, Cython and CUDA) and its SMARTS templates are in ``path_selection``.

Deliberately empty: importing one submodule (for example ``masster_hooks`` during config
validation) must not load the model or its compiled extensions.
"""
