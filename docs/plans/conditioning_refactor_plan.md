# Conditioning Injection: Audit & Consolidation Plan

> **Status:** ACTIVE &nbsp;·&nbsp; **Updated:** 2026-07-16
>
> Collapse 3 overlapping metadata-conditioning mechanisms into one. Plan + ablation config built, not run. Recovered from branch `research/ce_perfrag_features`, where it was the only copy.

## Summary

FraGNNet conditions on acquisition metadata (collision energy, adduct/precursor type,
instrument type, fragmentation mode) through **three** parallel mechanisms, and most metadata
is conditioned **2–3× redundantly**. The bulk of the machinery (the output-FiLM "scaler"
chain) can only *rescale* the intensity distribution uniformly across fragments — the exact
lever the sharpening analysis ([oracle decomposition](project_oracle_decomposition)) proved
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
  exact-match (see [ce-perfrag note](project_ce_perfrag_conditioning)).

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
2. **Feed the competition module** ([competition plan](ce_perfrag_competition_plan)) — so
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
