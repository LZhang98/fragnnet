# Conditioning Injection: Audit & Consolidation Plan

> **Status:** ACTIVE &nbsp;·&nbsp; **Updated:** 2026-09-08
>
> MASShatter GNN context input implemented with a matched merged pair; no runs launched.
> The combined FraGNNet context module is already implemented. Earlier refactor stages below
> are historical proposals, not the current implementation status.

## Current controlled screen (2026-09-08)

Execution update (2026-09-11): the 2026-09-11 W&B screen (never written up in `docs/findings/`;
the link that stood here pointed at a file that does not exist)
confirms a completed GNN + heads run and SMARTS attempts. No matching full-metadata
heads-only run appeared in the inspected recent runs; routing benefit remains unresolved.
The no-training statements below describe the earlier preparation handoff.

The [pre-run design audit](../findings/masshatter_context_design_audit.md) identified ACE clipping
and row-order-dependent ACE selection. Both are now fixed in the new merged pair: `ace_max: 300`
and opt-in weighted CE aggregation, supported by both models. Historical configurations and
checkpoint behavior retain legacy aggregation. See the audit follow-up for validation results.
The configured split is exclusively `[M+H]+`/FT/HCD, so it tests CE routing rather than diverse
categorical metadata. GNN concatenation initialization is unchanged. Next: a real-data GPU smoke
check; no training or held-out quality evaluation has run.

### Objective and constraints

Give MASShatter full acquisition metadata and measure how existing context routes affect
held-out predictions before adding a conditioning layer. This is the user-selected scope.
The user reports that existing runs already show benefit from **NCE + ACE + adduct + instrument
+ fragmentation mode**. Treat that full input set as fixed for the next conditioning experiments;
do not make another metadata-availability ablation a prerequisite. Run IDs and effect sizes were
not supplied in this session, so this is recorded user-provided evidence, not a new local result.
The subsequent user request adds optional GNN context input for merged MASShatter below.
Target preprocessing and existing experiment configs remain unchanged.
New arms require retraining: changing metadata vocabularies or injection widths changes parameter
shapes, and disabling trained scalers changes predictions. Do not load an old checkpoint into a
new arm and treat that as its trained result.

### Merged MASShatter: context at GNN input

The user explicitly requested that merged spectra's context also enter the GNN. Implemented
`mass_shatter_context_in_gnn`, default false, in the model and Lightning wrapper. When true,
the existing per-spectrum context is indexed by atom batch and concatenated to atom features
before the precursor GNN. The optional separate proposal GNN receives it as well; its context
input follows the existing proposal-gradient detachment setting. Downstream heads retain context.
Merged CE readings are embedded individually and pooled with schedule/replicate weights in
the new pair; legacy configs retain their original mean pooling.

| Arm | Config | Context destinations |
| --- | --- | --- |
| Control | [Full metadata, heads only](../../config/massshatter/merged/context_ablation/nist20_merged_full_metadata_context_heads_only.yml) | Proposal and intensity heads |
| Requested variant | [Full metadata, GNN + heads](../../config/massshatter/merged/context_ablation/nist20_merged_full_metadata_context_gnn_heads.yml) | Atom inputs plus proposal and intensity heads |

Both derive from the merged probability-CE warmup control: U256/E1024, Gaussian cross-entropy,
consensus targets, shared encoder, and proposal-only epochs 0–15. Both enable NCE, ACE, adduct,
instrument, and fragmentation mode. Their resolved configurations differ only in the new switch
and run name. This is the next MASShatter routing comparison; the unmerged configs below remain
available separately. No claim about the optimal warmup policy or improved quality follows yet.

Compatibility: false preserves GNN parameter shapes and the structural-only path. The repository's
`get_pl_hparams` checkpoint reader explicitly migrates older MASShatter checkpoint configs to
false; fresh configs obtain the required key from the template. True widens input-layer weights
and requires retraining. `mass_shatter_share_proposal_across_ev: True` is rejected with GNN context
input because that path reuses atom embeddings across spectra with potentially different context.

Validation: 182 tests passed, 5 skipped across the MASShatter model, Lightning wrapper, and new
GNN-context tests. Checks cover merged CE routing, spectrum isolation, context gradients through
atom embeddings, separate-proposal gradient isolation, and strict state loading after legacy
checkpoint migration. Both resolved merged configs constructed successfully; their intended-only
diff passed. Ruff passed. CUDA was unavailable, and no training, full repository suite, or held-out
evaluation ran. Before launch, verify acquisition metadata is homogeneous within each merge group:
the current merger retains one metadata row per group while aggregating CE readings. Do not
change merge grouping or target construction silently if this check fails.

### MASShatter: full-metadata baseline

Use the prepared full-metadata, dual-CE mask1024 config as the conditioning baseline. It retains
the existing unmerged mask1024 architecture; it does not settle the newer U256 encoder/warmup comparison in the
[MASShatter improvement plan](masshatter/masshatter_improvement_plan.md).

| Arm | Config | Conditioning |
| --- | --- | --- |
| M0 — historical reference | [Existing baseline](../../config/massshatter/ma_mi_unmerged/baseline/nist20_ma_mi_unmerged_mask1024_gine_gsa384_bondcentric_a2_b3_transformer_context_embed.yml) | NCE + adduct |
| M1 — optional diagnostic, outside the next run queue | [NCE with categorical metadata](../../config/massshatter/ma_mi_unmerged/context_ablation/nist20_ma_mi_unmerged_mask1024_full_meta_nce.yml) | NCE + adduct + instrument + mode |
| M2 — selected conditioning baseline | [Full metadata, dual CE](../../config/massshatter/ma_mi_unmerged/context_ablation/nist20_ma_mi_unmerged_mask1024_full_meta_dual_ce.yml) | NCE + ACE + adduct + instrument + mode |

M0/M1 remain available for diagnosis; the active sequence starts at M2. Instrument
vocabulary is FT/QTOF/IT; mode vocabulary is HCD/CID with the existing unknown token. Keep the
same 128-dimensional context, mask budget, loss, optimizer, split, and seed 397623. Global loss
tolerance stays fixed (`inst_type_loss_tol: null`), so enabling instrument input does not also
change the objective. Keep `mass_shatter_share_proposal_across_ev: False`: proposals and
intensities both receive the spectrum's CE and metadata.

The molecular encoder remains structural; existing nonlinear proposal and intensity heads mix
structural features with context. This already permits fragment-specific context effects.
The dataset computes both CE columns whenever either CE flag is enabled, filling missing NCE
from ACE or vice versa using precursor m/z and charge. Dual CE is therefore not necessarily two
independently measured signals. Keep this conversion unchanged for the screen.
MASShatter context-routing comparisons retain all five inputs and CE-specific proposals.
The unmerged configs establish a separate baseline; the implemented merged routing pair is above.

### FraGNNet: routing and output scaling

Complete a 2×2 comparison around the existing NodeMLP context configs:

| Context location | CE/mode output scalers on | All metadata output scalers off |
| --- | --- | --- |
| `mol` | [Existing mol-only](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_context_embedding_mol_only.yml) | [New mol-only](../../config/exp_fraggnn_ma_mi/context_ablation/nist20_ma_mi_context_mol_no_output_scalers.yml) |
| `mol,mlp` | [Existing mol+MLP](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_context_embedding.yml) | [New mol+MLP](../../config/exp_fraggnn_ma_mi/context_ablation/nist20_ma_mi_context_mol_mlp_no_output_scalers.yml) |

All four use NCE, ACE, adduct, instrument, and mode with a 64-dimensional combined context.
The off arms set `ce_scale`, `frag_mode_scale`, `adduct_scale`, `stratum_scale`, and
`depth_ce_bias` false; the last three were already false in the controls. The new pair differs
only in routing and run labels. Keep seed 42 inherited from the template, batching, tolerance,
optimizer, and schedule matched within this family. MASShatter and FraGNNet families are
within-model experiments, not a matched head-to-head comparison.

The per-fragment MLP already learns nonlinear fragment/context interactions. Shared affine
output scalers preserve fragment ordering within each channel, but per-channel affine parameters
can change channel preferences and aggregated rankings; a depth bias can change ordering across
depths. The older blanket statements below about all these paths being unable to re-rank are
too broad. Removing them is an experiment, not an established improvement.

### Acceptance checks, execution, and next step

Before launch, resolve configs against the same template revision and record code/config identity,
dataset and split fingerprints, actual retained sample IDs, checkpoint destinations, and seed.
Check instrument coverage and unknown-mode frequency on retained training/validation rows.
Use the [session workflow](../workflow/session.md) for provenance and
[cluster guide](../guides/submit_cluster_runs.md) if jobs are subsequently authorized.

Start with the full-metadata MASShatter baseline and the full-metadata FraGNNet routing/scaler
comparison; skip M0/M1 reruns unless a specific diagnostic requires them. Screen one seed per
family, then confirm promising treatments with matched seeds. Compare held-out
cosine and loss at equal optimizer progress; report CE/adduct/instrument/mode slices, train–validation
gap, and formula retrieval. For MASShatter also report candidate weighted recall and emitted peak
count so proposal coverage is separated from intensity quality. Use the existing
[fragment-context routing plan](fragnnet/fragment_context_gate_experiment_plan.md) for switch diagnostics
and the gate experiment's go/stop rules. No new layer is justified solely by a training gain.

Local validation with `FRAGNNET-GPU`: all four configs passed template resolution, sampler
validation, and actual Lightning/model construction. Resolved diffs contain only the intended
metadata, scaler, routing, and run-label differences. Synthetic context checks verified finite
outputs, dependence on every enabled metadata field, MASShatter gradients into every active
metadata encoder, multi-reading CE pooling, and the unknown-mode token. FraGNNet checks verified
the requested insertion widths and absence of all output scalers/depth bias. The configured data,
split, and DAG paths exist locally, but their contents and retained-row coverage were not audited.

Next: audit retained-row metadata coverage and checkpoint persistence, then run the matched
screen when authorized. No training, dataset preprocessing, checkpoint migration, cluster
submission, full test suite, or validation-set evaluation has been performed for this screen.

## Summary

FraGNNet conditions on acquisition metadata (collision energy, adduct/precursor type,
instrument type, fragmentation mode) through **three** parallel mechanisms, and most metadata
is conditioned **2–3× redundantly**. The bulk of the machinery (the output-FiLM "scaler"
chain) can only *rescale* the intensity distribution uniformly across fragments — the exact
lever the sharpening analysis (`project_oracle_decomposition`) proved
buys ~nothing — while the one path that can actually *re-rank* fragments (the "insert"
embeddings) was, until the `mol,mlp` change, used only weakly at the mol-encoder input.

Target: collapse the 8 conditioning knobs into **one combined context (stratum) embedding**
used only in re-ranking-capable places (representation insertion at mol+mlp, and the
competition module), and deprecate the redundant rescale-only paths.

## Current state (audit, 2026-07-12)

### Mechanism ① — "insert" embeddings (condition the representation; CAN re-rank)
A learned embedding of the metadata concatenated into node features, at `mol` (mol-encoder
input) and/or `mlp` (per-fragment nodeMLP input). Feeds a nonlinear downstream, so it can
change which fragment scores highest.
- `ce_insert`, `prec_insert`, `inst_insert`, `frag_mode_insert`
- Params: `*_insert_location`, `*_insert_size` (+ `ce_insert_type`, `ce_insert_merge`).
- `ce_insert_location` is now multi-valued ("mol,mlp"); the other three are still single-value
  exact-match (see `project_ce_perfrag_conditioning`).

### Mechanism ② — "scaler" FiLM chain (condition output logits; RESCALE ONLY)
Applied in sequence to `frag_joint_logits` in `fragnnet_model.py` forward
(~L1343–1374): `frag_mode_scaler → ce_scaler → adduct_scaler → stratum_scaler`. Each is
`logits_out = logits * exp(log_scale) + bias`, with `(log_scale, bias)` **shared across all
fragments** (a scalar for `FragModeScaler`, per-H-shift-dim for the others). Uniform across
fragments ⇒ **cannot re-rank**, only global rescale + H-shift preference. All zero-init identity.
- `FragModeScaler` (base_model.py ~L541): scalar affine per mode.
- `CEScaler` (~L433): MLP(CE-stats) → per-H-shift affine.
- `AdductScaler` (~L599): embedding(adduct) → per-H-shift affine.
- `StratumScaler` (~L652): embed(inst)+embed(mode)+embed(adduct) summed → per-H-shift affine.

### Mechanism ③ — `depth_ce_bias` (special-case CE path)
`depth_ce_bias_layer` (~L441): `Linear(ce_embed) → per-depth bias`, added per node by its
depth. A 3rd CE conditioning path. Zero-init.

(`output_scale` — a learned scalar before sparsemax/entmax projection — is orthogonal: it is
not metadata-conditioned and is out of scope here.)

### Redundancy table

| metadata     | ① insert          | ② own scaler       | ② stratum_scaler | ③ bias         | total |
|--------------|-------------------|--------------------|------------------|----------------|-------|
| CE           | ce_insert         | ce_scaler          | —                | depth_ce_bias  | **3×** |
| frag_mode    | frag_mode_insert  | frag_mode_scaler   | ✓ (incl. mode)   | —              | **3×** |
| adduct       | prec_insert       | adduct_scaler      | ✓ (incl. adduct) | —              | **3×** |
| instrument   | inst_insert       | —                  | ✓ (incl. inst)   | —              | **2×** |

The table lists every mechanism that EXISTS in the code. **What the current nist20 baseline
actually ENABLES** is narrower: all four `*_insert` at `mol`, plus `frag_mode_scale=True` and
`ce_scale=True`. `adduct_scale` / `stratum_scale` / `depth_ce_bias` are wired but default-off.
So the *active* redundancy today is just **CE and frag_mode double-conditioned** (insert +
scaler); adduct and instrument are insert-only. The rest is latent complexity (available knobs
that invite the confusing overlaps below if switched on).

Two specific smells (if the full set is enabled, e.g. via `stratum_scale`):
- **`stratum_scaler` overlaps `frag_mode_scaler` and `adduct_scaler`** — mode and adduct each
  get two independent output-FiLM affines stacked, on top of their insert embedding.
- **The chain composes four affines** — a composition of affines is itself an affine, so most
  of that chain is redundant re-parameterization of a single rescale.

## The core problem

Counting capability against the diagnosis: mechanisms ② and ③ (four scalers + a bias) can
only rescale/shift uniformly across fragments — the temperature-style lever shown to be
near-dead. Mechanism ① (insert) is the only re-ranking-capable path, and it was under-used
(mol-only, diluted through mean-pooling into the `cc` feature). So ~6 of 8 conditioning knobs
are aimed at the wrong axis, redundantly.

## Target design

One combined **context (stratum) embedding**:

    context = f_ctx( embed_CE(E) ⊕ embed_adduct(a) ⊕ embed_inst(i) ⊕ embed_mode(m) )

used in exactly the two re-ranking-capable places:
1. **Insert at `mol` + `mlp`** — one context vector replacing the four separate `*_insert`
   paths (generalizes the `ce_insert_location: mol,mlp` change to all metadata).
2. **Feed the competition module** (`ce_perfrag_competition_plan`) — so
   competition is stratum-dependent, not CE-only.

**Deprecate** the output-FiLM chain (②) and `depth_ce_bias` (③). Optionally retain a single
global output FiLM (`stratum_scaler` alone, which already subsumes inst+mode+adduct) as a
rescale safety net; drop `frag_mode_scaler` / `adduct_scaler` / `ce_scaler` / `depth_ce_bias`.

Net: **8 knobs → 1 context embedding (+ optionally 1 output FiLM)**, every remaining path
re-ranking-capable and non-redundant.

## Staging

1. **Doc (this file).**
2. **Ablation first (before any refactor).** The baseline only turns on two output scalers, so
   the ablation is simply: `frag_mode_scale=False`, `ce_scale=False` (adduct/stratum/depth_ce
   are already off), keeping all four insert embeddings. Retrain and compare `cos_hun` + the
   CE-binned intensity diagnostic vs. the full baseline. Hypothesis (from the sharpening
   result): near-zero loss — the two rescale-only scalers carry little signal. If confirmed,
   the "keep one output FiLM" question is moot (none are needed) and the refactor is pure
   simplification: one context embedding at mol+mlp, no output-FiLM chain at all.
3. **Build the combined context embedding** (gated on a clean ablation): a `ContextEmbedding`
   module; multi-location parsing extended to all four metadata; wire context into the
   competition module; deprecate the removed scalers with config-schema deprecation warnings
   (do NOT hard-remove keys — trained baselines and old configs reference them).

## Migration / risk

- **Trained baselines reference these params.** All scalers are zero-init identity, so a
  checkpoint trained with them off is a strict subset; but removing config keys breaks old
  configs. Deprecate with warnings, don't delete (respect the config-schema constraint).
- **Keep this refactor separate from the CE/competition experiments** so attribution stays
  clean — land the consolidation as its own branch/PR, validated by the ablation, not bundled
  with a new-capability run.
- **Thin strata** ([M+Cl]⁻, IT, [M+Na]⁺) have few spectra; a richer combined context
  conditioner can memorize them. The shared-parameter combined embedding is the safer form vs.
  separate per-stratum modules.

## Fragment-head gate follow-up

With `context_insert_location: mol,mlp`, the nonlinear formula MLP already provides
fragment-specific context responses. The separate
[`fragment_context_gate_experiment_plan.md`](fragnnet/fragment_context_gate_experiment_plan.md) first tests
that existing route with a zero-code `mol` versus `mol,mlp` ablation. A final-logit multiplicative
replacement is considered only after a positive result, as an ordering/optimization experiment—not
as new fragment × context capacity.
