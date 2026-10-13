"""The models, one package each: FraGNNet (``models.fragnnet``) and MASSter (``models.masster``),
the two owner models, and NEIMS, precursor and GNN (``models.neims``, ``models.precursor``,
``models.gnn``).

Each imports ``fragmentation`` and core only, never another model or a baseline; core reaches
them only through the registry (docs/plans/repo_reorg_plan.md).
"""
