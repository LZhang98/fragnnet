# Annotation-aware candidate retrieval

> **Status:** PROPOSED · **Updated:** 2026-09-25
> Score MassSpecGym retrieval candidates with what the predicted spectrum carries beyond its
> thresholded intensities: the full predicted support, the low-intensity tail, and per-peak
> fragment annotations. Nothing is implemented or run. Step 1 needs no annotation columns and can
> run on the MASSter MSGYM checkpoint. Steps 2-3 need fragment annotations, which only FraGNNet
> exports today. Cross-model, so it sits at the top of `plans/`.

## Objective and acceptance criteria

Test whether the true molecule ranks higher among its retrieval candidates when the score uses
support, the tail below the export threshold and fragment identity, rather than cosine alone.

- **Population.** MassSpecGym 1.5 val fold, `[M+H]+` queries with a CE: 9,734 queries, the
  `ms_gym_benchmark_1_5` val split. Formula-bounded candidates first, then mass-bounded ones.
- **Gate 1 (after Step 1).** A Step 1 score, alone or blended with cosine, raises hit@1 over
  cosine. The lower bound of the 95% molecule-bootstrap interval must exceed a minimum useful gain
  that is fixed before the first measurement, and hit@5 and hit@20 must move the same way. The
  gain must survive the spectrum-blind controls below. Select the blend weight on even molecules
  and report it on odd molecules.
- **Stop rule.** If neither score (b) nor score (c) beats cosine on the formula set, cosine is not
  discarding retrieval signal that the support or tail carries. Do not build Steps 2-3.
- The test fold stays unread until the scoring method is frozen.

## Established facts

**Current scoring is cosine only.** The PubChem retrieval evaluator ranks candidates by
`cos_sim_0.01` or `cos_hun`
([eval_pubchem_retrieval_val.py:92-96](../../scripts/ms2c/eval_pubchem_retrieval_val.py#L92-L96)).
MSGYM retrieval has no scorer. The only code that reads `source_identifier` is
[01_prepare_msgym_retrieval.py](../../preproc_scripts/msgym_ms2c/01_prepare_msgym_retrieval.py).
The manifest's `peaks` column is empty on every row, so query spectra must be joined from
`data/raw/msgym_1_5/MassSpecGym1.5.tsv` by `source_identifier`. No retrieval hit rate for any of
our models is recorded in `docs/`.

**Candidate sets.** Measured 2026-09-25 from each val zip's `query_candidates.pkl.gz` in
`data/ms2c/msgym/v1_5/`, deduplicated on `(source_identifier, candidate_smiles)` as `build-proc`
does:

| | formula val | mass val |
| --- | ---: | ---: |
| all queries | 19,429 | 19,429 |
| candidate rows, all queries | 2,433,451 | 4,640,577 |
| `[M+H]+` queries with a CE | 9,734 | 9,734 |
| candidate rows, that subset | 1,609,714 | 2,417,128 |
| unique candidates, that subset | 416,967 | 587,675 |
| candidates per query, median | 222 | 253 |
| single-candidate queries | 47 | 0 |

- **The subset is the benchmark val split.** All 9,734 identifiers match
  `data/df/ms_gym_benchmark_1_5_val_dset_ids.csv`, and none is in the train ids. They cover 2,407
  distinct query SMILES.
- **The rest is out of the MSGYM model's domain.** 4,737 queries (24%) are `[M+Na]+`, and 48.7%
  have no CE (`ace` and `nce` both NaN). The MSGYM model trained only on the `[M+H]+`
  simulation-challenge set, and the 2026-09-24 scope decision in
  [project state](../project-state.md) keeps it there.
- **Score hits by 2D InChIKey.** Raw SMILES equality finds the true molecule in only 69.8% of the
  9,734 queries. The 2D InChIKey first block finds it in all of a 400-query sample, exactly once
  each.

**Annotation export is FraGNNet only.** The FraGNNet export carries a per-peak neutral formula
plus its adduct delta, `P(n|f)` over about four candidate DAG nodes per formula, and each node's
atom bitmask ([run_inference guide](../guides/run_inference.md#fragment-annotation)). The MASSter
export sums candidates into one row per peak and emits no annotation columns
([script_utils.py:1771](../../src/fragnnet/utils/script_utils.py#L1771)). On MSGYM test
(`ll04i00m`), its unfiltered export has a median of 924 peaks per spectrum.

**Anchors.**
- MASSter: `ll04i00m` at epoch 60 (W&B), the checkpoint in the
  [GLACIER comparison](../findings/masster/masster_msgym_vs_glacier_released_ckpt_2026_09_24.md).
- FraGNNet: a depth-3 MSGYM config exists
  ([ms_gym_benchmark_1_5.yml](../../config/massspecgym/ms_gym_benchmark_1_5.yml)). No trained
  checkpoint or val score for it is recorded in `docs/`, and W&B has not been checked.

**Why support and the tail should carry signal.** This is evidence, not proof.
- On the NIST20 FraGNNet test export, base-peak misses are about 7:1 misranking against missing
  coverage ([dominant-peak decomposition](../findings/fragnnet/dominant_peak_outcome_decomposition.md)).
- The MASSter ranking error is a generalization failure: val ranking 0.121 against train 0.011
  ([ranking decomposition](../findings/masster/masster_ranking_decomposition_2026_09_23.md)).
- Independent runs agree less on novel molecules. Agreement falls from 0.94 to 0.75 as
  nearest-train Tanimoto drops from above 0.8 to below 0.3
  ([variance is epistemic](../findings/masster/masster_variance_novelty_2026_09_24.md)).
  Decoy candidates are unseen molecules by construction, so their intensities are the least
  reliable part of their predictions.

**Spectrum-blind baselines are high.** A structure-only model gets 26.9% top-1 under a
conventional structure-disjoint MassSpecGym split
([spurious evaluations](../notes/spurious_evaluations_spectraverse.md)). The PubChem default
ordering gets 49.98% hit@1 on the retrieval challenge
([MassSpecGym audit](../notes/massspecgym_in_the_wild_audit.md)). Every number here needs the
spectrum-blind controls next to it.

## Scores

Notation: `y_k` is the query's L1-normalized intensity at m/z `m_k`. For candidate `c`, `S_c` is
its predicted m/z support at `--min_intensity -1` and `p̂_c(m)` is its predicted intensity summed
within the instrument tolerance. Candidates are predicted at the query's CE, adduct and
instrument, all of which the manifest carries.

- **(a) Cosine baseline.** `cos_sim_0.01`, the linear cosine MassSpecGym requires. `cos_hun` is a
  sensitivity check.
- **(b) Chance-corrected support explainability.** `E_c = Σ_k y_k² · [m_k ∈ S_c]`, the L2 share
  of the query that `c` can explain at all. It uses the same closed form as the DAG oracle in
  [dag_oracle_scores.py](../../scripts/eval/dag_oracle_scores.py).
  - Subtract `E_c` computed on copies of the query shifted by fixed mass offsets. A denser support
    explains random peaks by chance, and the shifted copies measure that.
  - Exclude peaks within tolerance of the precursor. Every candidate in a formula-bounded set
    explains the precursor.
  - For FraGNNet, `S_c` is the DAG's formula support, which is purely structural. For MASSter it
    is the proposal-selected candidates, so part of it comes from the model. Record which one was
    used.
- **(c) Query log-likelihood under the full prediction.** `L_c = Σ_k y_k · log max(p̂_c(m_k), ε)`.
  - Suppose one candidate predicts a query peak at 0.1% of its base and another cannot form that
    peak at all. `L_c` separates them. A thresholded cosine does not.
  - Keep the precursor in this score, because the predicted precursor survival differs between
    candidates.
- **(d) Blends.** `α·z(a) + (1−α)·z(b or c)`, where `z` standardizes within each query.

Break exact score ties at random, or report the average rank. Never break them by candidate list
position, which may carry the source ordering's bias.

**Controls.**
- Spectrum-blind rankings: support size `|S_c|` alone, and heavy-atom count alone.
- Score (b) and score (c) computed against a different query that has the same precursor formula.
  This measures how much of each score is a candidate prior rather than a spectrum match.

**Report** hit@1, hit@5 and hit@20 with 95% molecule-bootstrap intervals. Show the formula and
mass sets separately, and FT and QTOF separately. Also bin by the query molecule's nearest-train
Tanimoto, because the hypothesis predicts the gain concentrates on novel molecules.

## Plan

**Step 1: support and tail scores, no annotations.**
- Build the MSGYM scorer. It joins the query spectra from the TSV, identifies the truth by 2D
  InChIKey, applies the tie rule, and computes scores (a)-(d) plus the controls.
- Predict the formula val subset with `ll04i00m` at `--min_intensity -1`, which disables the
  filter. Only `pred_mzs` and `pred_ints` are needed.
- Size: at about 924 peaks per spectrum, 1.61M candidate spectra is about 12 GB of float32
  m/z-intensity pairs. That fits under `output/` but not in the small local `/tmp`. Scoring
  during inference avoids storing the export at all.
- Time it on 200 queries before the full run. Then run the mass set.

**Step 2: fragment-level scores (only if gate 1 passes).** First decide the anchor: a trained
FraGNNet MSGYM checkpoint plus DAGs for about 417k candidates, or a new MASSter annotation export
that gives each aggregated peak its candidates' atom masks and formulas. Then add two scores:
- **Explanation cost.** For each query peak, the cheapest fragment of `c` that explains it: bonds
  cut, ring bonds cut, depth and the size of the H shift. Include the model's node score where
  the predicted intensity is below the export threshold.
- **Diagnostic-peak weighting.** Weight each query peak by how much the top candidates' predicted
  intensities disagree there. This removes the intensity-error noise that shared losses add
  without adding information.

**Step 3: learned reranker.** A listwise ranker over the scores from Steps 1-2, cross-fitted by
molecule on val. It must not be fitted on the train-fold retrieval sets
(`retrieval_candidates_*_train`). The true train molecules' predictions are fitted (MASSter's
train ranking gap is 0.011 against 0.121 on val), so a ranker fitted there learns to trust cosine
far more than val supports. A reranker that sees only the spectrum-blind features is its control.
Read the test fold once, after freezing.

**Out of scope for now.**
- **Substructure voting.** Query peaks vote, through every candidate's bitmasks, for the
  substructures that explain them.
- **Ambiguity sets and CE planning.** Flag candidates whose annotated spectra are
  indistinguishable, and choose the CE that best separates the top candidates. This and
  substructure voting are applications, worth building only after a score has proven itself.
- **Per-peak uncertainty as a cosine weight.** This was rejected in design. Ensemble spread is
  mostly spectrum-level: inside one spectrum, the higher-spread peak is the worse one only 58-60%
  of the time ([per-peak uncertainty](../findings/masster/masster_per_peak_uncertainty_2026_09_24.md)).
  Across candidates, unusual decoys are the uncertain ones, so down-weighting their mismatches
  would reward them. Uncertainty may enter only as a predictive spread inside score (c).

**Constraints.**
- Step 1 needs inference only, no training.
- Cluster inference follows the [submission guide](../guides/submit_cluster_runs.md).
- Queries without a CE, and `[M+Na]+` queries, need a separate CE and adduct policy, such as a CE
  grid with a max over it. They are excluded here.

**Related.** [Energy-based spectrum reranking](energy_based_spectrum_reranking_plan.md) reranks
alternative intensity allocations for one molecule. This plan ranks molecules. They share the
`--min_intensity -1` export contract.

## Current state

Documentation only. The candidate-set facts above were measured on 2026-09-25 from the val zips.
No scorer, export or run exists.

## Open questions

1. What are the minimum useful hit@1 gain, the ε floor for score (c) and the shift offsets for
   score (b)? All three must be fixed before the first measurement.
2. Is there a trained FraGNNet MSGYM checkpoint, or should Step 2 build a MASSter annotation
   export?
3. Does the goal need the no-CE and `[M+Na]+` queries, which are 50% of the val fold?

## Next concrete check

Write the scorer and run score (a) on a 200-query formula val sample with `ll04i00m`. Check that
the truth is found for every query, that ties are handled, and that hit@1 exceeds the
spectrum-blind controls. Record throughput to size the full run.
