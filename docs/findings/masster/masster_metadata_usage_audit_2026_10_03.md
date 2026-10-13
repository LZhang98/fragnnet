# MASSter metadata usage audit vs published forward models

> **Status:** FINDING · **Updated:** 2026-10-03
> Read-only code + data audit at `c5e38529` plus
> the working tree; literature comparison from papers/READMEs. No code, configs or data changed.

## Summary

- **Inputs that vary.** Collision energy (CE), instrument and fragmentation mode (when enabled)
  are learned inputs. Each passes through one per-spectrum `MassterContextEncoder`, and the
  resulting vector is appended to every candidate token of the intensity head and to the bond
  and charge proposal heads.
- **Where metadata does not reach.** The GNN never sees metadata (`masster_context_in_gnn: False`).
  The proposal heads get a **detached** context in both reference configs.
- **Adduct and charge.** These act **physically**: they set the ion mass and element deltas
  and mask invalid channels. The learned adduct embedding is a **constant bias** in every
  reference config, because training has one adduct.
- **NIST20 references (`i1`/`i4`).** Only two inputs vary: NCE (42 distinct integers) and
  instrument (ORB/ORB_F). The Fourier NCE code is a fixed per-integer table, not a smooth
  function of energy. With the instrument it effectively learns an instrument-specific NCE
  calibration, which the peptide models do explicitly.
- **MSGYM reference.** `frag_mode` is CID for **100% of rows**, but that "CID" is the parser's
  fallback for an unknown mode, not data. The embedding is a constant and the HCD row is
  untrained. The CE units are mixed (29% of Orbitrap `ace` is NCE already converted to eV), and
  the NCE channel is derived from that column. See §7.
- **Against the literature.** MASSter's injection point (spectrum-level context concatenated
  per fragment or candidate) matches FraGNNet and FIORA. It is behind GLACIER, the one verified
  model that injects CE into the encoder (node embeddings plus attention bias). No small-molecule
  model found does instrument-specific NCE calibration. MASSter's instrument × NCE code is the
  closest thing to it, and it is learned implicitly.

## 1. How each field is consumed

Code references: `MM` = [masster_model.py](../../../src/fragnnet/model/masster_model.py),
`BD` = [base_dataset.py](../../../src/fragnnet/dataset/base_dataset.py),
`BH` = [bond_fragment_heads.py](../../../src/fragnnet/model/bond_fragment_heads.py).

| Field | Encoding | Enters | Physical use | `i1`/`i4` (NIST20) | MSGYM |
| --- | --- | --- | --- | --- | --- |
| NCE | `embed`: clamp [-1, `nce_max`], **round to int**, fixed Fourier table (`requires_grad=False`) → `Linear(20/22, 32)`; mean-pooled over readings (MM:870-895) | context only | none | on, `nce_max` 200 | on, `nce_max` 300 |
| ACE | same table type, own channel; missing ACE filled `nce·prec_mz·cf/500` (BD:227-310) | context only | none | off | on |
| Adduct (`prec_type`) | `nn.Embedding(n, 32)` (MM:802) | context | ion mass and element deltas, negative-count channel masking (MM:2085-2222); CMF deltas only with `include_cmf` (off) | vocab 1 → constant | vocab 1 → constant |
| Instrument | `nn.Embedding(n, 32)`, no unknown row (MM:804-808) | context; also selects metric tolerance | none | ORB / ORB_F | ORB / QTOF |
| Frag mode | `nn.Embedding(n+1, 32)`, last row = unknown (MM:809-813) | context; gates SMARTS templates when on (off) | template filter only | off | on, data 100% CID |
| Precursor m/z | not a model input; used for the CE fill and metrics | — | — | — | — |
| Ion mode / charge | derived from `prec_type`; the `ion_mode` column is unread | — | via adduct | constant + | constant + |

How the context is routed:

- **Intensity head.** Gets `context[fragment_batch_indices]` appended to each candidate token
  (BH:1449). This head is the only one that trains the context encoder.
- **Proposal heads.** The bond, atom and mixture heads get `context.detach()` when
  `proposal_selection_only` is True and proposal input gradients are disabled (MM:3265, 3382).
  They see CE and instrument as input, but their own loss cannot shape it. The
  [head-input asymmetry finding](masster_head_input_asymmetry_2026_09_22.md) measured −0.011
  to −0.013 when gradients were connected.
- **GNN.** Encodes the molecule metadata-blind (MM:3203).
- **Off in the references.** Optional CE-FiLM and CE-pair attention bias on the intensity head
  exist (`masster_intensity_ce_film`, `masster_intensity_ce_pair_bias`).

The context encoder is `Linear → SiLU → Linear`, with no normalization or dropout in the
references. Its input width is 96 for `i4` (adduct + instrument + NCE) and 160 for MSGYM.

## 2. What actually varies in the training data

| | NIST20 train (`nist20_inchikey`) | MSGYM 1.5 (all rows) |
| --- | --- | --- |
| Spectra | 157,678 (ORB 114,966, ORB_F 42,712) | 118,991 (ORB 81,313, QTOF 37,678) |
| Adduct | `[M+H]+` only | `[M+H]+` only |
| Frag mode | HCD only (embedding off) | **CID only** |
| NCE | never missing; 2–195, 42 distinct integers (ORB 34, ORB_F 22) | **missing 100%**, derived from ACE; 0.94% of train > 300 clamp |
| ACE | missing 18.2% raw (ORB 0%, **ORB_F 67.3%**); filled | present 100%; 23% non-integer, rounded by `embed` |
| CE type | single only | — |

Only 7 of 12,668 NIST20 train molecules were measured on both Orbitrap instruments
([i1 memorization read](masster_i1_memorization_read_2026_09_28.md)). The instrument
embedding therefore also separates two largely disjoint sets of molecules.

## 3. Measured evidence already in `docs/`

- **Instrument.** `i1` (ORB vs ORB_F) scored **+0.0278** over `n4` (one `FT` class) in the
  epoch 54–63 tail. Lumos val cosine rose from 0.688 to 0.742, and the Lumos precursor-fraction
  bias fell from +0.092 to +0.020. One seed per arm.
  [Instrument CE scales](../nist_orbitrap_instrument_ce_scales_2026_09_28.md) ·
  [i1 read](masster_i1_memorization_read_2026_09_28.md)
- **CE encoding.** The smooth inputs `qet`, `qet_mlp` and `id` lost 0.009–0.012 to the
  per-integer Fourier code. With the instrument present, the penalty shrank from −0.021 to
  −0.0036 (epochs 10–14 only). The reading: the per-NCE code had been carrying the instrument.
  [QET / NCE input gates](masster_qet_projection_candidate_union_2026_09_26.md)
- **Other recorded fields.** Campaign, replicate count, m/z error, solvent, vial and eV add
  ≤ 0 held-out gain. A per-molecule energy shift would be worth +0.010, but it is not
  predictable (R² ≤ 0). [Residual structure](masster_i1_residual_structure_2026_09_28.md)
- **FraGNNet, for contrast.** Per-fragment CE at the node MLP was null (−0.0007). FraGNNet
  conditions through per-field `*_insert_location` embeddings and optional output scalers; see
  [conditioning refactor plan](../../plans/conditioning_refactor_plan.md). That plan's
  "three overlapping mechanisms" describes FraGNNet. MASSter has one.

## 4. Comparison with published forward models

V = verified from paper/README text; U = unverified detail. CFM-ID is from the authors' own
knowledge.

| Model | CE | Adduct | Instrument | Where metadata enters | Adduct physical? |
| --- | --- | --- | --- | --- | --- |
| CFM-ID 2–4 | separate model per energy level (10/20/40) | separate model per adduct | none | model selection | yes (per-model) |
| MassFormer (V) | scalar NCE | categorical | categorical | concat to the graph embedding before the output MLP | no |
| 3DMolMS (V) | NCE = eV·500·cf/mz | one-hot (4) | none; needs fine-tuning off QTOF | concat to the pooled latent | no |
| ICEBERG v2 (V) | per-CE spectra; eV = NCE·mz/500; 64-d sinusoid; CE ablation +1.1 pt top-1 | one-hot + multi-hot chemistry flags | none | Generate and Score (exact site U) | U |
| FraGNNet paper (V) | merged; mean of Fourier embeddings | none (`[M+H]+` filter) | none | concat per fragment node | — |
| FIORA (V) | continuous eV (0–100) | `[M+H]+`/`[M-H]-` (+`[M]±`) | categorical | concat per edge in the final FC layers; GNN on the neutral graph | partly: per-fragment H-shift ion set |
| GLACIER (V, code + released MSG checkpoint) | raw float, no NCE/eV conversion; sinusoid; learned token for missing CE | one-hot | one-hot, fixed (Orbitrap / QTOF / IT-FT / Unknown); `embed_instrument: True` in the checkpoint | concatenated to **every atom's input features** in the Graphormer encoder | U |
| MassSpecGym task (V) | float CE | `[M+H]+` only | QTOF / Orbitrap | benchmark fields; also exposes precursor m/z | — |
| Prosit / AlphaPeptDeep (peptides) | scalar NCE (+ instrument one-hot in AlphaPeptDeep) | charge | AlphaPeptDeep only | meta embedding added to the sequence | — |
| **MASSter `i4`** | per-integer Fourier NCE code | physical + constant embedding | embedding (2 classes) | concat per candidate token; detached into proposal heads; GNN blind | **yes** (ion mass and element deltas) |

Sources: [MassFormer](https://arxiv.org/abs/2111.04824),
[3DMolMS](https://pmc.ncbi.nlm.nih.gov/articles/PMC10281859/),
[ICEBERG v2](https://pmc.ncbi.nlm.nih.gov/articles/PMC12154671/) and
[ms-pred](https://github.com/coleygroup/ms-pred),
[FraGNNet](https://arxiv.org/abs/2404.02360),
[FIORA](https://pmc.ncbi.nlm.nih.gov/articles/PMC11889238/),
[GLACIER](https://arxiv.org/html/2606.29161v1),
[MassSpecGym](https://arxiv.org/html/2410.23326v1),
[NCE calibration in peptide models](https://pubs.acs.org/doi/10.1021/acs.jproteome.3c00857).
Unverified: MARASON, SCARF and GRAFF-MS covariate handling, and MassFormer's encoding details.

Notes:

1. **NCE ↔ eV conventions disagree.** ICEBERG uses no charge factor; 3DMolMS uses
   cf = 0.9 for z = 2. MASSter's fill uses `CHARGE_FACTOR_MAP`, so the choice only matters
   for multiply charged ions. None of these is instrument-calibrated. In peptide work the
   instrument-to-instrument offsets are 1–6 NCE (Lumos vs Q Exactive, for example). Our
   Elite/Lumos result is the small-molecule version of that effect.
2. **Encoder-level injection has one verified precedent: GLACIER.** It concatenates its CE
   sinusoid and its adduct and instrument one-hots to every atom's Graphormer input
   (`ms_pred/glacier/joint_model.py:251-289` at `318bb7b`). MASSter has the switch
   (`masster_context_in_gnn`, plus the zero-init suggestion in the
   [context design audit](masster_context_design_audit.md) §4). No result on the current
   references is recorded.
3. **Physical adduct treatment is rare.** Only CFM-ID (per-model) and FIORA (H-shift set)
   do it. MASSter's mass and element-delta tables already handle 16 adducts, including
   negative mode. What has never been exercised is a learned adduct signal, because no
   training set has more than one adduct.
4. **ICEBERG's multi-hot adduct flags** (polarity, alkali, H±, halide, −H2O, −CO2, +NH3)
   are a ready design if multiple adducts are added. They share strength across rare adducts,
   which a plain one-hot cannot.

## 5. Defects and risks

1. **Constant embeddings in the references.** The adduct embedding (both references) and
   frag mode (MSGYM) are constants. Any claim that "MASSter conditions on adduct or
   fragmentation mode" is untested. On MSGYM, an HCD spectrum would reach an untrained row.
   - Action: keep frag mode off on MSGYM, or note that it is inert.
2. **The NCE code is per-integer and discontinuous.**
   - The table frequencies are 2π·0.5^k, k = 0…9. For integer inputs, the k = 0 columns are
     constant and the k = 1 columns are a ±1 parity bit. Only the low-frequency columns
     vary smoothly.
   - Non-integer CE is rounded. That affects 23% of MSGYM ACE.
   - An NCE value never seen in training gets an untrained combination. This is fine inside
     NIST20's 42-value ladder; transfer to other data (NIST23, Enveda, QE ladders) is
     untested.
   - The smooth encoders lost on NIST20, so do not replace it blindly. Any test should be
     instrument-present (`i`-series), not `n`-series.
3. **Redundant CE channels on MSGYM.** NCE is entirely derived from ACE and precursor m/z.
   The two channels plus the instrument (= data source) are not independent signals.
   - The `fill_missing_ace` / NCE-fill formula is a convention, not a measurement.
4. **No unknown-instrument fallback.** An out-of-vocabulary instrument raises `KeyError`
   (BD:1003). That is loud, which is good. But any new instrument (QE, QTOF in NIST20 terms)
   needs fresh training or a mapping decision.
   - The retrieval `spec_df` files still say `FT`
     ([project state](../../project-state.md)).
5. **`nce_max: 200` in the template.** NIST23 negative-mode NCE reaches 320
   ([instrument CE scales](../nist_orbitrap_instrument_ce_scales_2026_09_28.md) §2). A config
   that inherits 200 would silently clamp those values. MSGYM already pins 300.
6. **Precursor m/z is not an explicit input.** Mass reaches the model only through the graph
   and the per-candidate `c_mass` features. MassFormer-style and FIORA-style models pass
   precursor mass or MW as a covariate. No result says it is missing here.

## 6. Suggested next steps (none run)

- **CE into the encoder.** A zero-init `context_in_gnn` arm on top of `i4`, the GLACIER-style
  route. This is the only literature-backed injection site MASSter has not tested on the
  current references.
- **Before any multi-adduct dataset** (NIST23 `[M+Na]+`, `[M-H]-`; Enveda): add
  ICEBERG-style adduct flags and enable `include_cmf`.
- **Before training on non-NIST20 data:** pin `nce_max` per dataset, and decide the
  unknown-instrument policy.

## 7. MSGYM config check (2026-10-03)

Config: [msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_k1_ema999_fourier.yml](../../../config/masster/massspecgym/msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_k1_ema999_fourier.yml).
Raw source: `data/raw/msgym_1_5/MassSpecGym1.5.tsv`. That file has one `collision_energy`
float column, with no unit and no fragmentation-mode column.

1. **CE units are mixed inside `ace`.**
   - [`parse_mass_gym_ce_str`](../../../src/fragnnet/utils/data_utils.py) marks a value as NCE
     only if the string contains `normalized=True`. The MSGYM 1.5 values are bare floats, so
     **every row is stored as eV**.
   - The data contradict that for part of the Orbitrap set. 23,884 of the 23,924 non-integer
     Orbitrap values (29% of 81,313) satisfy `ace·500/prec_mz` = a multiple of 5 within 0.05.
     They are NCE that an upstream step converted to eV. For those rows the derived NCE
     channel recovers the true setting exactly (q99 180).
   - The other 57,389 Orbitrap rows are integers on the 15/20/30/45/60/75/90 ladder. Their
     unit cannot be determined from the file. Their derived NCE reaches 874, and 4% exceed
     200, which is implausible for Orbitrap NCE. So at least some of these are probably NCE
     stored as eV.
   - 1,018 Orbitrap molecules have rows of both kinds.
   - The ICEBERG team treats this field as NCE: their missing-CE imputation searches a
     0–150% NCE grid (MassSpecGym in the Wild, App. A.5,
     `data/paper/MassSpecGym_in_the_Wild_arxiv_2606.19624.txt:1117`).
   - QTOF rows are 96% integer eV-like settings. Their derived "NCE" is a formula artifact,
     not an instrument setting.
   - Consequence: neither CE channel has one unit across Orbitrap rows. The per-integer
     Fourier code can still tell the two populations apart: converted rows land on
     multiples of 5 in the NCE channel, integer rows do not. That makes it a source cue, not
     an energy.
2. **`frag_mode` CID is a default, not a label.**
   [`data_utils.py:1005-1007`](../../../src/fragnnet/utils/data_utils.py) sets CID whenever the
   mode is missing ("default is CID if we don't know"). The Orbitrap rows are most likely
   beam-type HCD.
   - The config header says HCD is kept in `frag_modes` "so a checkpoint stays usable on HCD
     data". That is not true: the HCD row is never trained, so HCD input reaches a random
     initialization. The embedding adds a constant bias and nothing else.
   - Recommended: turn `frag_mode` off for MSGYM, or relabel it as unknown. Changing the
     embedding changes the context width, so this needs a new run, not a checkpoint patch.
3. **`ORB` means different things in different datasets.** In NIST20 configs `ORB` means the
   Elite specifically, with Lumos as `ORB_F`. In MSGYM, `ORB` is every Orbitrap (`Orbitrap`
   raw label). An embedding row trained on one dataset is not the same instrument in the
   other. Moving checkpoints between datasets, or pooling them, needs a mapping decision.
4. **Clamps.** At `nce_max` 300, 0.79% of derived NCE values clamp onto one row. Almost all
   of these are integer-ladder rows, so the clamp mostly hits the suspect unit population.
   `ace_max` 300 clips 4 rows.
5. **The control carries the same issues.** The base config (`1imawkx1`) uses the same
   metadata keys, so the planned read (this arm against `1imawkx1`) is still like-for-like.
   Only cross-model comparisons that interpret CE are affected. That includes ICEBERG/GLACIER
   CE ablations and any NIST20 → MSGYM transfer.

Not determined here: the unit of the integer Orbitrap ladder, which needs MassSpecGym's
source-library provenance; and whether fixing units changes cosine (no run).

## 8. What we fed GLACIER (2026-10-03)

The released-checkpoint comparisons
([GLACIER vs `ll04i00m`](masster_msgym_vs_glacier_released_ckpt_2026_09_24.md), the retrieval
pilot and the formula-retrieval shards) all build an ms-pred labels TSV. For the test set it is
`output/glacier_msg_eval_2026_09_24/labels_test.tsv`. Columns:
`spec, ionization, formula, smiles, inchikey, instrument, collision_energies, precursor`.

| Field | Value passed | How GLACIER uses it (`predict_smis_joint.py`, `joint_model.py`) |
| --- | --- | --- |
| adduct | `[M+H]+` (all 9,954) | one-hot, concatenated per atom |
| CE | raw MSGYM `collision_energy`, e.g. `['20']`; 1,063 non-integer strings | `float(s.split()[0])`, no unit conversion; sinusoid concatenated per atom |
| instrument | raw `Orbitrap` (8,374) / `QTOF` (1,580) | fixed one-hot, positions 0 / 1, concatenated per atom |
| precursor | precursor m/z in the TSV | the inference dataset sets `precursor: 0.0`; the model forward does not use it (loss only) |

So both models received **the same CE number** for every test spectrum. GLACIER reads it
unit-agnostically. Our pipeline calls it ACE and also derives an NCE channel from it. The
comparison is input-matched, and the mixed-unit problem in §7 applies to both models alike.

## Verification

- Context routing: MM:870-895, 3198-3204, 3263-3265, 3382.
- Fourier table frozen: `form_embedder.py:140-141`.
- MSGYM metadata counts and CE unit test: computed directly from
  `data/proc/ms_gym_benchmark_1_5/spec_df.pkl` (`frag_mode` CID 118,991/118,991; `nce` NaN 100%;
  `ace` NaN 0%; non-integer `ace` against `ace·500/prec_mz` on the 5-grid).
- NIST20 split counts: computed by the audit from `data/proc/nist20/spec_df.pkl` plus the split
  ID files. They match the val counts in the instrument CE finding.
- Literature rows marked V were read from paper or README text; nothing was reimplemented.
