# MASSter: SMARTS rules as part of the fragmentation pathway

> **Status:** CLOSED NULL · **Updated:** 2026-09-30
> Every Phase 1 gate failed its pre-registered rule, so no Phase 2 arm runs: G0 +0.00125 < 0.003
> (`t1i` on `[M-H]-`), G1 STOP (`p1`), and Track B at 0 promoted (`p1c`)
> ([finding](../../findings/masster/masster_smarts_pathway_gates_2026_09_30.md)). The Phase 0
> code and the `p1` token stay built (off by default), and the `t1i` / `p1` configs were never
> submitted. Open to the user: D2 (a template-subset target; G0's subset value is 0.022) and the
> G2 load/no-caps rules. MASSter only; FraGNNet gets
> no SMARTS work (user decision: VRAM). MASSter will expand to `[M-H]-` (user decision), so both
> polarities are in scope. Revised the same day after review: each arm now has its own gate, G1
> is a cross-fitted read, and graft is scoped to what the registered transfer rules do not
> already cover.

## Objective and acceptance

Make rules part of how a MASSter fragment arises, not only a precursor-level candidate source.
There are three pieces, each with its own gate:

- **Templates on (`t1i`):** switch on the 22 registered rules. The evidence here is coverage,
  and it is strongest on `[M-H]-` (§1). Gated by G0. `t1i` is also the control for both tracks.
- **Track A, identity:** a candidate's rule provenance (which rule, which step) reaches the
  intensity allocator. Ranking, not coverage, is MASSter's measured bottleneck. Gated by G1.
- **Track B, steps on fragments:** gated rule operations apply to pool candidates, so products
  of "cut then rule" and "rule then cut" exist. Gated by G2/G2b.

Accept an arm when it passes the +0.003 whole-split tail gate against its control (§4 table).
Cost is judged best-by-wall-clock-hour against that control first, epoch-matched second. If
the user accepts a subset-only target (decision D2), the alternative is a template-subset gain
larger than that subset's own seed spread, measured from two seeds.

## 1. Established facts

| Fact | Value | Source |
|---|---|---|
| `i1` val cosine-gap decomposition (gap 0.212) | coverage 0.020 (9%), ranking 0.104 (49%), calibration 0.088 (42%); train ranking 0.010 | [base-peak errors](../../findings/masster/masster_i1_base_peak_errors_2026_09_28.md) |
| `i1` base-peak misses already in the candidate set | 94% | base-peak errors |
| Spectra carrying an eligible template (14 active rules, 2026-09-29) | `[M+H]+` `i1` split 4.95% (val 4.99%, 187 molecules); `[M-H]-` NIST20 OB/OB_F 5.6% | [gates finding](../../findings/masster/masster_smarts_pathway_gates_2026_09_30.md). The earlier 8.0% was 22 rules on FT labels |
| Whole-split gain needed per template-spectrum gain | whole ≈ 0.05 × subset, so +0.003 needs ≈ +0.06 on the subset | arithmetic |
| Precursor-template products beyond `c0`'s pool, `[M+H]+` | +0.196% of the intensity of template-bearing spectra (CI 0.089–0.333; 239 molecules) | CUDA audit §3.1 |
| Share of intensity at template-product m/z, `[M+H]+`, the nine 2026-09-23 new-assembly rules | 0.3–3.7% of bearing-spectrum intensity (aminobenzothiazole 3.7%, sulfonate 1.9%, diaryl ketone 1.0%); twins included | [SMARTS usage audit](../../findings/smarts_usage_audit_2026_09_23.md) |
| CO / CO2 / HCN / C2H2 rules | 0 unique m/z (a same-formula connected set always exists) | CUDA audit §7 |
| Generic bridge excision (≤3 atoms) over `c0`'s pool, `[M+H]+`, train | +0.05 pp over random (CI −0.09 to +0.21); 379 sets/spectrum before formula pruning vs 228 ordinary candidates | [excision gate](../../findings/masster/masster_excision_continuation_gate_2026_09_28.md) |
| Restricted aromatic nitro NO loss | confirms on val (+0.038 pp over all 512 val molecules) | excision gate |
| Rearrangement then 1–2 cuts (ChemFrag) | thioether below random; Wagner–Meerwein 0 unique | [ChemFrag triage](../../findings/masster/chemfrag_rearrangement_triage_2026_09_14.md) |
| Hand-built post-cut descriptors / `fc1` | ≤ +0.001 / +0.0008 calibrated | [post-cut gate](../../findings/masster/masster_i1_postcut_information_gate_2026_09_29.md), [`fc1` screen](../../findings/masster/masster_fragment_connectivity_local_screen_2026_09_29.md) |

The per-class gate, in % of each rule's own spectrum intensity, is model-free, run on all of
NIST20 and exploratory (not held out). The strongest signal is negative-mode. Every rule below
is already registered, so this is what `t1i` on `[M-H]-` would switch on:

| Rule | `[M-H]-` unique over random | Molecules |
|---|---|---:|
| `diacylhydrazine_carboxylate_o_transfer` | +49.6 | 3 |
| `n_nitramine_n2o` | +6.98 | 5 |
| `aryl_sulfonate_so2` | +6.40 | 121 |
| `primary_aryl_sulfonamide_so2` | +6.39 | 67 |
| `n_acyl_aryl_sulfonamide_carbonyl_o_transfer` | +5.69 | 10 |
| `aryl_sulfonamide_so2` | +3.75 | 237 |
| `aminobenzothiazole_hcn_loss` | 0 unique, yet its product explains 26.8% of intensity | 4 |

The last row is the clean Track A case: a large explained share and no new m/z. It rests on
four molecules, so it motivates G1 on `[M-H]-` but cannot decide it.

The following have never run on `[M-H]-`: the proposal-reach probe, the excision continuation
gate, and a template census on the new scope. The `[M+H]+` nulls do not transfer, because
negative mode is where Smiles-type transfers occur.

## 2. Code behavior before Phase 0 (verified 2026-09-29)

Phase 0 (§4) since changed four of these: the valence bound, positional rule indices,
`fork_rng`, and rule identity, which is now available as provenance. The line numbers below
are pre-Phase-0.

- **Rules match the precursor only.** The root mask is hard-coded to all-ones
  ([smarts_prepass.py:746](../../../src/fragnnet/frag/smarts_prepass.py#L746)).
- **A template is a forced cut mask plus an alternative partition.** Disconnected sides become
  new unique fragments, and connected sides map to ordinary ones
  ([masster_candidates.py:1100-1126](../../../src/fragnnet/frag/masster_candidates.py#L1100-L1126)).
- **The mechanism head tells the mechanisms apart by the rule embedding alone.**
  - It pools the same cut bonds for the ordinary and the SMARTS mechanism of a mask
    ([masster_model.py:2470-2523](../../../src/fragnnet/model/masster_model.py#L2470-L2523)).
  - It scores them with a SiLU MLP over (boundary embedding, rule embedding, proposal context)
    ([masster_model.py:625-657](../../../src/fragnnet/model/masster_model.py#L625-L657)).
  - So the ordinary-vs-rule split can vary with the bond environment and CE, but the head sees
    nothing from the product side.
- **Rule identity is lost before the intensity head.** Fragments are log-sum-exp'd over
  (mask, mechanism) ([masster_model.py:3131-3140](../../../src/fragnnet/model/masster_model.py#L3131-L3140)).
  - Under `c0`/`i1` (selection-only), the mechanism trains through the proposal loss only.
  - No candidate feature or intensity token carries a rule.
  - The exported "pathway" is (fragment, ion mode, H shift).
- **Nothing applies a rule to a fragment.** The only fragment-of-fragment logic is the offline
  [continuation probe](../../../scripts/debug/debug_masster_excision_continuations.py).
- **The valence bound is optimistic for formed bonds.** It is `slots + cut_count`, with no −2
  for the new bond ([masster_model.py:2116](../../../src/fragnnet/model/masster_model.py#L2116)).
  The gate probes subtract it (`reconnect=True`).
- **Rule indices are positional.** `MASSTER_RULE_NAMES` concatenates four groups (shared FRAG,
  shared NL, MASSter-only FRAG, MASSter-only NL), and the index is the position
  ([masster_templates.py:270-281](../../../src/fragnnet/frag/masster_templates.py#L270-L281)).
  - Inserting any rule shifts every later index.
  - The embedding has 1 + `MASSTER_NUM_RULES` rows, so the checkpoint's tensor shapes change too.
- **`SmartsMechanismHead` is built without `fork_rng`**
  ([masster_model.py:1868](../../../src/fragnnet/model/masster_model.py#L1868)), unlike the
  four other optional modules. Enabling SMARTS therefore changes the intensity head's
  initialization.
  - Wrapping it restores identical initialization only.
  - The arms stay unpaired regardless: templates change the pool every batch, and the head's
    dropout advances the CUDA random stream. Read single-seed contrasts against the seed
    spread (0.0042, whole split).
- **Modes that refuse SMARTS:** `fragment_selection: marginal`, `connected_fragment_beam` and
  the CUDA topology prototype. The full CUDA builder supports it, with exact CPU parity.

## 3. AIMe, and the direction chosen

**Sources.** AIMe (DeepMS2Reasoner; [note](../../notes/aime_deepms2reasoner.md)) was read in
the main text, plus both bioRxiv supplements.

- Neither supplement contains the "Supplementary AI Methods": one holds metrics and synthesis,
  the other is a UI guide. `github.com/gomes-lab/aime` returns 404 (checked 2026-09-29).
- Excision size limits, regularization weights and H-transfer states are therefore
  unpublished.
- The paper does not ablate excision or ring opening.

| AIMe mechanism | MASSter today | Plan |
|---|---|---|
| Excision at any fragment state: remove X, bond its two outer neighbours | Precursor templates; offline continuation probe | Gated operators on pool candidates (G2, G2b). As atom sets, "cut then excise" and "excise then cut" are the same object, so no sequential state is needed |
| Showcase excision (imide C=O, new C–N) | Reached by an isomeric connected set | Evidence that excision's value is identity: Track A |
| Penalty on expected excision / ring-opening use | Twin products split intensity arbitrarily | Optional knob in `p1` |
| Excision keeps the charge on the retained fragment | Charge head normalizes over both template sides | Optional: charge only on the product side for neutral-loss templates |
| Hard checks for connectivity and H balance on new bonds | Valence bound ignores formed bonds | Phase 0.2 |
| Features recomputed on the current fragment graph | Pooled precursor embeddings; `fc1` never sees a formed bond | Optional in `p1`: `fc1` over the product graph with its formed bond |
| Multi-step ring opening | One-shot masks already cut both ring bonds | None |
| H and charge as state, conditioning later steps | Per-fragment ion/H channels | Out of scope (sequential model) |

**Direction.** Curated, gated SMARTS rules are the model mechanism. AIMe-style generic
operators are an offline discovery tool whose confirmed families become rules (the way nitro
NO was found). The one exception: if G2b promotes a generic operator on `[M-H]-`, it becomes a
model operator for that polarity.

**The known transfers are already rules.** `n_acyl_aryl_sulfonamide_carbonyl_o_transfer`,
`diacylhydrazine_carboxylate_o_transfer`, `pfas_sulfonate_f_transfer` and
`n_fluoro_sulfonamide_f_transfer` are registered and gated to `[M-H]-`; they act on the
precursor. A generic operator can add only two things beyond them:

- the same transfers applied to pool fragments;
- transfer families that are not registered yet.

**Two generic operators, not one.** Written as excision, the transfer families remove a
*large* X, so a bounded excision operator cannot reach them:

- **N-acyl sulfonamide O transfer.** Ar–S(=O)2–N–C(=O)–R gives phenoxide Ar–O⁻ (C6H5O⁻,
  m/z 93.03, for N-benzoyl benzenesulfonamide). X = SO2 + N + C + R; the kept O is terminal,
  four bonds from the aryl C.
- **Diacylhydrazine.** R1–C(=O)–N(tBu)–NH–C(=O)–R2 gives R1–CO2⁻. X = N(tBu)–NH–C–R2.
- **PFAS F transfer.** For triflate this *is* a 3-atom excision (X = CF2). For longer chains,
  X is the whole chain minus one F.

Every transfer product has the same shape: a connected piece P plus one terminal heteroatom t
from a few bonds away. So G2b gates two operators:

- **Bridge:** S∖X, with X connected, ≤3 heavy atoms, no ring atoms, and two external bonds to
  distinct endpoints. This is the existing probe's operator.
- **Graft:** P ∪ {t}, where P is a candidate and t is a terminal O or F not adjacent to P,
  within k ≤ 4 bonds of P (4 covers both transfer examples). The new bond joins t to P's
  nearest boundary atom. Cost is about |pool| × (terminal O/F within k), before formula
  deduplication.

## 4. Plan

### Phase 0: prerequisites (code) — BUILT 2026-09-29

The user authorized implementation on 2026-09-29. Every item is off-path bit-identical: with
`masster_smarts_templates: False` no new module, buffer or state_dict key exists. Tests:
[`test_masster_smarts_pathway.py`](../../../tests/test_masster_smarts_pathway.py), plus
export tests in `test_inference_masster_annotation.py`, `test_export_schema.py` and
`test_run_inference_hdf5_ragged_families.py`.

1. **`fork_rng`: done.** `SmartsMechanismHead` is built under `th.random.fork_rng(devices=[])`.
   A SMARTS-on model now starts with every parameter identical to the SMARTS-off model's
   (tested). It changes no measurement (§2).
2. **Formed-bond valence: done, unconditionally for template sides.** The decision was taken
   by implementing it; `t1i`'s config records it.
   - A new batch field, `component_formed_bond_counts`, holds pieces − 1 for a template side
     (`formed_bond_count` in `masster_templates.py`) and 0 for every ordinary component. The CPU
     and CUDA builders fill it; the parity tests compare it.
   - The `valence` bound subtracts 2 per formed bond from a side's H-gain cap. Whether a
     channel disappears depends on which cap binds first (tested on a two-piece side of
     pentane: +2 H without the correction, 0 with it).
   - Continuations will reuse the field.
3. **Stable vocabulary: done, stored in the state dict rather than hparams.**
   - `MASSTER_RULE_VOCABULARY` is an explicit append-only name list, and today's order keeps
     every index unchanged. A SMARTS-on model registers the names' CRC-32 hashes as the
     `smarts_rule_name_hashes` buffer.
   - Why not hparams: `get_config_from_ckpt` overwrites checkpoint hparams with template +
     custom-config values at inference. A training config that leaves the key at its template
     default would therefore reset the stored list.
   - A checkpoint trained on a shorter prefix loads: its rule-indexed weights get zero rows,
     it keeps its own list, and `_eligible_smarts_templates` drops the newer rules.
   - A reordered, renamed or missing list raises.
4. **Provenance: done, without a step index.** Every template acts on the precursor, so a step
   would be constant; it is added with continuations.
   - With `return_annotation`, the model returns (fragment, rule) pairs,
     `fragment_rule_fragment_idxs` / `fragment_rule_idxs`, for either template side.
   - The export writes them as `pred_node_rules`: per node, `;`-joined rule names, empty for
     ordinary nodes. It is opt-in (`--return_rule_provenance` in `run_inference.py` and
     `run_save_inference_eval.py`, which requires `--return_cc_hash`) and refused for
     non-MASSter models, so FraGNNet exports and the frozen default schema are unchanged.
5. **Docstrings: done.** The CUDA-refuses-SMARTS claims are corrected in `masster_candidates.py`
   and `masster_model.py`.

### Phase 1: gates (local; no training)

**Coverage ceiling.** G0, G2 and G2b report the same unit: the rise in the base-peak
decomposition's `oracle` term (`‖T_covered‖ / ‖T‖`) on the whole split when the new products'
channels are added. No prediction on those candidates can gain more cosine than that. The
existing gates report pp of linear intensity, so the conversion is a small addition to
[`debug_masster_smarts_rule_gate.py`](../../../scripts/debug/debug_masster_smarts_rule_gate.py).

**G0: `[M-H]-` census and coverage ceiling.** This decides `t1i` on `[M-H]-`.
- Measure:
  - the share of `[M-H]-` spectra and molecules with an eligible template, per rule, with
    [`debug_smarts_rule_census.py`](../../../scripts/debug/debug_smarts_rule_census.py);
  - the whole-split coverage ceiling of the eligible template products, using the per-class
    gate's unique-coverage rule.
- It needs only the `[M-H]-` rows, so it can run before D1.
- **Run `t1i` on `[M-H]-`** if the coverage ceiling is at least +0.003. That is necessary,
  not sufficient: the arm itself is the test.

**G1: rule provenance as a cross-fitted correction.** This decides Track A.
- Method: the [post-cut gate](../../findings/masster/masster_i1_postcut_information_gate_2026_09_29.md)
  pipeline ([`debug_masster_postcut_information_gate.py`](../../../scripts/debug/debug_masster_postcut_information_gate.py)),
  with a rule-provenance family in place of the post-cut descriptors.
  - `extract`: add columns per candidate peak marking whether its m/z is the channel of an
    eligible template product (within ±4 H), and which rule. A template product's disconnected
    side is never in `i1`'s pool, so on `i1` the marked peaks are covered by twins only.
    Target peaks that only a template product reaches are counted as coverage under G0, not here.
  - Ceiling: set the marked peaks' predicted shares to their target shares and renormalize.
    Repeat for control peaks matched on m/z band and target rank. The headroom is the excess
    of the rule ceiling over the control ceiling.
  - Achievable gain: `fit` the correction `log p + g([log p, context, x])` with `x` the
    rule-provenance family, cross-fitted by `mol_id` parity, read as the gain over the
    context-only baseline with molecule-clustered 95% intervals.
  - Unlike the post-cut descriptors, `x` here is exactly what `p1`'s token carries, so this
    read is a fair estimate of what `p1` can learn from provenance alone.
- **Continue** to `p1` if the headroom is at least +0.003 whole-split (or D2's subset target)
  **and** the cross-fitted gain has CI lower > 0.
- **Stop** otherwise, but only when the interval's half-width is below the target. A wider
  interval is reported as underpowered, not as a stop.
- Runs:
  - `i1` (`i50ig7rf`), `[M+H]+`, now: about 10 min of GPU for `extract`. The headroom is
    expected to be small, since peaks at template-product m/z hold only 0.3–3.7% of
    bearing-spectrum intensity (§1). It runs anyway to close `[M+H]+`.
  - The first `[M-H]-` checkpoint, later. Aminobenzothiazole alone (four molecules) is
    underpowered there.

**G2: rule-restricted continuations.**
- Method:
  - The continuation probe, with X limited to registered rule motifs plus nitro NO.
  - The frozen train→val protocol of the excision gate: train-selected families, at least ten
    molecules, a Bonferroni-adjusted val confirmation, and no capped searches.
  - The baseline, as in the excision gate: the pool, all eligible registered template
    products, and all generic whole-precursor bridge products.
  - Report the candidate load after formula deduplication, and the coverage ceiling.
- `[M+H]+`: over `i1`'s pool.
- `[M-H]-`, model-free, runnable now: take S from all components of 1- and 2-cut masks as a
  stand-in for the pool, following the ChemFrag unique-coverage method. Rerun over the pool
  once an `[M-H]-` checkpoint exists.
- **Promote** a family if it confirms on val and adds at most 10 unique fragments per bearing
  spectrum (4% of the 256 budget; proposed threshold).

**G2b: generic operators on `[M-H]-`.** This is the deciding measurement for AIMe-style
operators.
- Method: bridge and graft (§3) under G2's protocol and baseline. Because the baseline holds the
  registered transfer products, graft gets no credit for them. Each operator is scored against
  a random disconnected null, excluding same-formula connected twins (±4 H), using
  `reachable_h` and `ChannelScorer` from
  [`debug_masster_smarts_rule_gate.py`](../../../scripts/debug/debug_masster_smarts_rule_gate.py).
- **Model operator** if its whole val union beats the null (CI lower > 0) within G2's load cap.
- **Otherwise** it stays a discovery probe, and its confirmed families enter the registry
  through the per-class gate.

**Track B's arm gate.** Run `p1c` only if the promoted G2 families and G2b operators together
reach a coverage ceiling of at least +0.003, since `p1c` must pass the same tail gate.

**G3: rule mining** (only after a positive G1).
- Re-measure the NIST-annotation "heavy-skeleton mismatch" at corpus scale in both polarities.
  It was 8.1% on 400 molecules and 5.5% on 40
  ([inventory](../../findings/unused_information_inventory_2026_09_22.md)), so it is
  sample-sensitive.
- Group rows by neutral-loss formula and cut-site substructure into a ranked candidate list for
  the per-class gate.

### Phase 2: training

| Arm | Gate | Change | Control | Read |
|---|---|---|---|---|
| `t1i` `[M-H]-` | D1 + G0 | The `[M-H]-` baseline + templates, after Phase 0.2–0.3 | That baseline | Candidate recall; tail against the gate; template-subset cosine |
| `t1i` ([config](../../../config/masster/mh_smarts_pathway_2026_09_29/t1i_smarts_templates.yml), built) | G1 continues | `i1` + `masster_smarts_templates: true`, after Phase 0.2–0.3 | `i1` | Same. Its own expected gain is small (+0.196% coverage, §1); it runs as `p1`'s control. Twin products enter with their own embeddings, so it also tests "correct atom set, no label" |
| `p1` ([config](../../../config/masster/mh_smarts_pathway_2026_09_29/p1_rule_pathway_token.yml), built) | G1 continues (per polarity) | + `masster_rule_pathway_token: true`: per unique fragment, a 0/1 indicator per vocabulary rule that produces it (an indicator, not a pathway count), through a zero-initialized, bias-free projection added after `input_projection`, i.e. the sum of its rule embeddings; `fork_rng`, so step 0 equals `t1i` exactly (tested); input only, no m/z-matched hard labels; no step index until continuations exist | `t1i` of the same polarity | Template-subset cosine and base-peak hit on the same spectra; whole-split tail; best-by-hour |
| `p1c` | Track B's arm gate | + continuations promoted in G2/G2b | `p1` | Same, plus builder ms and unique-fragment fill |

- Optional `p1` knobs, used only when a read motivates them: `fc1` over rule products with the
  formed bond; an AIMe-style penalty favoring the ordinary twin when both share an m/z; charge
  only on the product side for neutral-loss templates.
- Seeds:
  - Run a second seed when the whole-split effect is within twice the whole-split seed spread
    (0.0042).
  - For a D2 subset read, measure the subset's own seed spread from those two seeds; 0.0042
    does not apply to a subset.
- `t1` ([config](../../../config/masster/mh_ft_smarts_2026_09_27/t1_smarts_templates.yml)) is
  `c0`-based and superseded by `t1i`.

### Constraints any implementation keeps

- **CPU/CUDA parity**, in fields and gradients.
  - Changes go into `_template_forced_masks`, `_append_smarts_mechanisms` and the kernel's
    extra forced words.
  - CUDA asserts that a connected template side is an ordinary component of its mask.
- **Grouped CE views share identical template cuts.** Continuations derive from the shared
  pool, so they stay group-invariant.
- **Bit-identical off path** at the defaults.
- **Per-mask mechanism normalization.** A per-fragment rule feature needs an explicit
  aggregation over pathways.
- **Shared registry.** Editing `FRAG_RULES`/`NL_RULES` also changes FraGNNet DAG builds, so new
  rules go in [`masster_templates.py`](../../../src/fragnnet/frag/masster_templates.py).

## 5. Not planned

- **SMARTS for FraGNNet:** user decision (VRAM).
- **A general graph-edit / rearrangement engine with bond-graph node identity:** ChemFrag
  closed it, and generic bridge excision is at random on `[M+H]+`.
  - ChemFrag's closure used unique coverage, the criterion Track A rejects: Wagner–Meerwein
    scores 0 because a formula twin always exists.
  - So if G1 continues, the zero-unique ChemFrag transforms become Track A candidates, as
    curated templates read the same way. The engine stays out.
- **Unbounded excision:** every connected subgraph becomes a candidate X. The graft operator
  covers the transfer shape at a fraction of the cost.
- **Broad or charge-relaxed rules:** `crf12`-like breadth, up to 80 matches per molecule.
- **Bond-level SMARTS flags as features:** recoverable from existing inputs (balanced accuracy
  0.93–0.999; [feature headroom](../../findings/feature_headroom_2026_09_22.md)).

## 6. Decisions for the user

- **D1.** `[M-H]-` setup: joint or separate polarity, and instrument labels (OB/OB_F). This
  fixes the baseline for `t1i` `[M-H]-`, the `[M-H]-` G1 run and G2's pool rerun. G0 does not
  wait for it.
- **D2.** Is a gain confined to template spectra worth pursuing, for example for annotation
  quality, when the whole split does not move?
- **D3.** Should bridge excision try X above 3 heavy atoms (AIMe's limit is unpublished), or
  should graft cover the large-X cases?

## Current state and next check

**Built 2026-09-29.**
- Phase 0 (all five items) and the `p1` token (`masster_rule_pathway_token`).
- The [`t1i`](../../../config/masster/mh_smarts_pathway_2026_09_29/t1i_smarts_templates.yml) and
  [`p1`](../../../config/masster/mh_smarts_pathway_2026_09_29/p1_rule_pathway_token.yml)
  configs: `i1` plus their keys and W&B naming, NOT submitted.

**Validation.**
- The MASSter, inference and export test files pass, including the CPU/CUDA candidate-field
  parity and the new tests.
- A 2-epoch real-data GPU smoke of `p1` passed: CUDA backend, local split, 4 overfit batches of
  64 ([log and wrapper](../../../output/smarts_pathway_smoke_2026_09_29/)).
  - 56 of 107,404 fragments carried a rule, and 28 were assembled sides with a formed bond.
  - The `p1` projection trained in exactly one column, rule 19
    (`primary_aryl_sulfonamide_o_transfer`).
- No gate and no training run yet.

**Registry change (2026-09-29).** A parallel session commented 8 rules out of the allowlist,
leaving 14 active; the vocabulary keeps all 22 names. The removed rules include
`aminobenzothiazole_hcn_loss`, §1's Track A example. The gates below used the 14-rule registry.
The `t1i` config header still quotes the old 22-rule / 8.0% figures.

**Gates run 2026-09-29/30** ([finding](../../findings/masster/masster_smarts_pathway_gates_2026_09_30.md)).
The scope was NIST20 only; MONA was excluded by user decision on 2026-09-30.
- **G0 FAIL.** `[M-H]-` whole-split ceiling +0.00125 [0.00091, 0.00163]; the fully pool-free
  upper bound is 0.00269.
- **G1 STOP.** Headroom −0.000012 and gain over context +0.000026, both CIs ≪ 0.003.
- **G2:** 0 families promoted. The real effects (nitro, sulfonamide NH, sulfonate OH) are all
  blocked by the no-caps rule and the load rule.
- **G2b:** bridge and graft do not beat the null.
- **Track B FAIL.**
- The MONA-only rules (diacylhydrazine, PFAS) have no NIST20 evidence.

**Next.** None within this plan's gates.
- The two protocol rules above (load ≤10 on a stand-in pool; zero capped searches) are
  effectively unsatisfiable, but changing them would not reach +0.003: even every motif family
  together reaches only 0.00104 on val.
- A D2 decision is the only open route: the target would be the template subset, not the
  whole split.
