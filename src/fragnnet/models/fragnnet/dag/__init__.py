"""FraGNNet's fragment DAGs: construction during preprocessing, and the layout the model reads.

- builder: molecule info, the Cython enumeration (compute_frags, multi_cut_bfs), the SMARTS/NL
  prepass injection, and batch generation into HDF5.
- layout: dtypes, size constants, node and edge feature columns, boundary masks.
- formula: fragment formulas from connected-component bitmasks, and their isotope peaks.
- hashing: Weisfeiler-Lehman subgraph hashes for nb-isomorphism grouping.
- stats: peak statistics of generated DAGs.
- graph: one DAG as a PyG graph, and DAG batching.

The on-disk DAG format is core (``fragnnet.utils.dag_format_utils``), because core reads DAG
files. Deliberately empty: importing one module must not load the compiled extensions.
"""
