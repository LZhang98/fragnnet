"""MASSter path selection: which bond cuts and fragments the model scores, on CPU, Cython or CUDA.

- candidate_builder: `build_masster_candidates` (CPU or CUDA backend), the CPU path and the
  candidate dataclasses.
- core: compiled Cython kernels for the CPU path (top-k cut masks, component arrays).
- cut_masks: the pure-Python bond-mask generator the Cython kernels reproduce.
- connected_fragment_beam: the `connected_fragment_beam` proposal, a beam search over connected
  atom sets on the model device (the alternative to `cut_masks`).
- smarts_templates: SMARTS pathway templates applied during candidate generation.
- cuda_candidates: the production CUDA candidate builder.
- cuda_selection, cuda_topology: ctypes wrappers of the native mask-selection and topology kernels
  (`.cu` sources alongside).
- cuda_build: nvcc compilation, the /tmp build cache and library loading for those kernels.
- cuda_topology_prototype: benchmark-only GPU topology after the CPU mask selector.

Deliberately empty: importing one module must not load the compiled extensions.
"""
