# Baselines

> **Status:** REFERENCE · **Updated:** 2026-10-12
> GRAFF and MassFormer are retained in `src/fragnnet/baselines/`; ICEBERG runs externally through
> ms-pred. Set up by the [reorganization plan](../plans/repo_reorg_plan.md) (Phases 1 and 6).

## Retained baselines

Both run through the shared training entry point and the shared `config/template.yml`. Their
code imports core and `fragmentation/` only (see the
[architecture map](../architecture.md#package-layout-and-dependency-rule)).

| `model_type` | Code | Lightning module and dataset | Configs | Template keys |
| --- | --- | --- | --- | --- |
| `graff` | [baselines/graff/](../../src/fragnnet/baselines/graff/) | `GrAFFPL`, `SpecMolAnnDataset` | `config/nist20_merged_{inchikey,scaffold}/graff/s0-s4.yml` | `graff_*` and the `ann_params` block ("GrAFF only" section) |
| `massformer` | [baselines/massformer/](../../src/fragnnet/baselines/massformer/) | `MassFormerPL`, `SpecMolMFDataset` | `config/{massspecgym_merged,nist20_merged_inchikey,nist20_merged_scaffold}/massformer/s0-s4.yml` | `mf_*` ("MassFormer only" section) |

- **GRAFF** needs a formula-annotation table: `ann_fp` (for example `data/proc/nist20/ann_df.pkl`,
  written by `preproc_scripts/02_prepare_proc.py`; see the
  [preprocessing README](../../preproc_scripts/README.md)) and `mol_params.graff: True`. Its
  `GrAFFHooks.apply_inference_config` sets `ann_params.formula_str` and, with
  `disable_preproc`, `ann_params.preprocess` for inference.
- **MassFormer** needs `mol_params.mf: True`, which only `SpecMolMFDataset` honors; the shared
  `SpecMolDataset` raises if it is set. Its Graphormer input uses the Cython extension
  `baselines/massformer/algos.pyx`, built by `setup.py` with the other extensions.
- **NEIMS** is a baseline but stays a core model (`model/neims_model.py`, `pl_model/neims_pl.py`),
  because ChemLM shares its head and reads `neims_bottleneck_factor`.

Train with the shared entry point, for example:

```bash
python scripts/run_pl_model_fit.py -t config -c config/nist20_merged_inchikey/graff/s0.yml
```

Checkpoints: none are stored locally. W&B runs with the `graff3_long` prefix keep their
checkpoint files (for example run `61utcpnx`, `nist20v3-1_inchikey_graff3_long_s4`); the
`massformer_long` runs do not.

## ICEBERG (external ms-pred)

The local ICEBERG fork was retired on 2026-10-11. Selecting `iceberg_gen` or `iceberg_inten`
raises an error that names ms-pred and the archive revision.

- **Archive.** The fork, its configs and its integration live in revision
  `13b1599d3cd85ef8f214799894e527adaf3ddd0a` (reachable from `origin/masster`), recorded as
  `registry.ICEBERG_ARCHIVE_SHA`. Git history is the archive; no importable copy is kept.
- **Local checkpoints.** `saved_ckpts/speed_benchmark_1000/*iceberg*.ckpt` were trained with the
  fork. Each has a provenance JSON beside it with its SHA-256 and the archive revision. Load them
  only from a checkout of that revision; ms-pred cannot load them.
- **Running ms-pred.** Use [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred) in its own
  checkout and environment, outside the artifact stores and not in `FRAGNNET-GPU`. For each
  comparison, pin the upstream commit, model version, config and checkpoint identity.

### Evaluation-file contract

ms-pred predictions enter this repository as files, scored by its evaluation tools. Before
scoring, check sample identity, split, adduct, collision energy, m/z units, intensity transform
and binning against the evaluation set.

- **Readers:** `load_glacier_peaks` in `scripts/eval/compare_msgym_predictions.py`,
  `scripts/eval/score_retrieval_candidates.py --pred_format mspred`, and the `mspred_*` scoring in
  `src/fragnnet/utils/retrieval_utils.py`, which follows ms-pred's retrieval benchmark at commit
  `318bb7b`.
- **Format they accept:** a PredSpecDB HDF5 from ms-pred, with exactly one InChIKey and one
  collision energy per entry. Intensities are square roots. Only MassSpecGym retrieval is served.
- **Not built yet:** NIST20 per-seed ICEBERG rows need an adapter to the `run_save_inference` pkl
  schema (`{split}/{model}/s{seed}/{split}.pkl`, read by
  `scripts/stats_scripts/calculate_similarity_stats.py`), proven on a fixture before use.
