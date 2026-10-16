"""MASSter candidate generation: the bond cuts and fragments the model scores (CPU, Cython or CUDA).

Status as of 2026-10-16. PRODUCTION means on the path of the current recipe (`i4`, which sets
``masster_candidate_backend: cuda`` like nearly every MASSter experiment config) or of the
template-default CPU backend. OPT-IN means a tested model option behind a flag that is off by
default and off in `i4`. PROTOTYPE means built and tested but not adopted.

PRODUCTION
- builder: `build_masster_candidates`, the single entry point for both backends; the CPU
  backend (template default) and the candidate dataclasses both backends return.
- cut_mask_kernels: compiled Cython kernels of the CPU backend (top-k cut masks, component arrays).
- cuda_candidates: the CUDA backend (``masster_candidate_backend: cuda``).
- cuda_mask_ranking, cuda_topology: ctypes wrappers of the CUDA backend's native mask-ranking
  and topology kernels (`.cu` sources alongside).
- cuda_build: nvcc compilation, the /tmp build cache and library loading for those kernels.

OPT-IN
- smarts_templates: SMARTS template cuts and assemblies (``masster_smarts_templates``) on both
  backends. Its pathway plan is a CLOSED NULL; `MASSTER_RULE_VOCABULARY` is always imported.

PROTOTYPE
- rewire_rules: admitted rearrangement edit rules (``masster_rewire_rules``); MSGYM screen ran
  2026-10-16 (validation within noise of control), NIST20 arms prepared.
- rewire_null: the topology-null control for those rules (``masster_rewire_null``, arm a0null).
- connected_fragment_beam: the ``connected_fragment_beam`` proposal, a beam search over connected
  atom sets on the model device; STOPPED 2026-10-06, kept as an option.
- cuda_topology_hybrid: benchmark-only CPU selection followed by GPU topology; not a backend.

TEST REFERENCE
- reference_cut_masks: pure-Python mask ranker and dataclass wrappers the Cython kernels are
  checked against; no production code imports it.

Deliberately empty: importing one module must not load the compiled extensions.
"""
