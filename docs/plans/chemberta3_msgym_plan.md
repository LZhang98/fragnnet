# SMILES + ChemBERTa-3 spectrum prediction on MassSpecGym

> **Status:** ACTIVE · **Updated:** 2026-09-27 (basic model built, uncommitted and not yet
> trained; cut to a basic model on 2026-09-26 at the user's request; the earlier design and
> its [Review](#review-2026-09-26) are kept)
> **Current scope: a basic model, to see whether it works.** ChemBERTa-3's first 6 of 12 layers,
> loaded encoder-only, fine-tuned **end to end** with the NEIMS binned 0.01 Da head on
> MassSpecGym's simulation-challenge spectra. Three arms, 2 seeds each: the model (E1), the model
> with random SMILES (T2) and the NEIMS fingerprint baseline (E4). The augmentation ladder, the
> other controls, the frozen-encoder checks, full depth and the exact-mass stages are
> [deferred](#deferred-not-scheduled). Built on 2026-09-27 as `model_type: chemlm`, with
> `transformers` 4.46.3 installed in FRAGNNET-GPU; see [Built](#built-2026-09-27). No run has
> trained. Cross-model, so it sits at the top of `plans/`.

## Objective and acceptance criteria

Does a basic ChemBERTa-3 spectrum model work? Two reads, each against a run with the same head,
data, CE input and schedule:

1. **E1 against E4.** Does the pretrained SMILES encoder, trained end to end, match a fingerprint
   MLP? Within the seed spread, it matches fingerprints; above it, it adds something; below it,
   the basic model does not work yet.
2. **T2 against E1.** Do random SMILES strings in training help? T2 helps if it beats E1 by more
   than the seed spread.

- **Data.** `ms_gym_benchmark_1_5`, unmerged; this is exactly the simulation-challenge rows.
- **Primary metric.** Linear `cos_sim_0.01`, which is MassSpecGym's required `cos_sim` (see the
  [GLACIER comparison](../findings/masster/masster_msgym_vs_glacier_released_ckpt_2026_09_24.md)).
  The basic runs read val only; the one-time test read is deferred. Secondary metrics:
  `cos_hun_inst_tol` and the sqrt variants. There is no 0.1 Da headline.
- **Reference points.** On test, MASSter `ll04i00m` scores 0.5713 and GLACIER's released
  checkpoint 0.5455. The best MASSter val is 0.5754 (`1imawkx1`).
- **Seeds.** This model family's seed spread is unknown, so every arm runs 2 seeds, and the
  spread within each pair sets the noise floor.
- **Expectation.** The head is binned at 0.01 Da. It has to learn each fragment's mass defect bin
  by bin, whereas MASSter and GLACIER place peaks at exact formula masses. So E1 is judged against
  E4, which has the same head, not against MASSter.

## How ChemBERTa-3 handled SMILES

Evidence levels are marked: **stated** (paper or README), **code**, or **inferred**. The primary
text (Singh et al., *Digital Discovery* 2026; ChemRxiv preprint 2025) was not readable: paywalled
and behind Cloudflare. Its Methods section should be checked before the deferred raw SMILES form is adopted.

- **Code: no canonicalization step.** The pretraining scripts in
  [deepforestsci/chemberta3](https://github.com/deepforestsci/chemberta3) load prepared ZINC20 and
  PubChem SMILES CSVs from the authors' S3 bucket and use them as they are. A search for
  `canonic`, `MolToSmiles`, `kekul`, `isomeric`, `SaltRemover` and randomization helpers finds
  nothing. The only cleaning rule documented is in the benchmarking data-preprocessing readme: SMILES over about 200
  characters are dropped, following MoLFormer. Whether that rule also applied to the pretraining
  corpus is not stated.
- **Stated for ChemBERTa-2 only.** The ChemBERTa-2 paper
  ([arXiv 2209.01712](https://arxiv.org/abs/2209.01712)) says "we canonicalize and globally shuffle
  the SMILES", without naming a toolkit. No accessible source for ChemBERTa, ChemBERTa-2,
  ChemBERTa-3 or MoLFormer mentions randomized-SMILES augmentation, at training or at test time.
- **Code: tokenizer and model.** Measured from the Hugging Face files of
  `DeepChem/ChemBERTa-100M-MLM`:
  - `RobertaForMaskedLM`: 12 layers, hidden size 768, 12 heads, FFN 3,072, 512 positions,
    about 92M parameters.
  - `RobertaTokenizer`: byte-level BPE with 7,924 tokens and 7,663 merges.
  - The GPT-2 pre-tokenizer splits runs of letters, digits and punctuation, so `[C@@H]` becomes
    `[`, `C`, `@@`, `H`, `]` before BPE. A byte-level vocabulary cannot emit `<unk>` for an ASCII
    SMILES.
- **No smaller ChemBERTa-3 checkpoint is released** (Hugging Face `DeepChem` listing, 2026-09-26).
  - **The size in the name is molecules.** "100M" is the pretraining set, 10% of ZINC20. The
    ChemBERTa-3 README also lists a ChemBERTa-MLM-10M, pretrained on 1% of ZINC20. It has the same
    architecture and is not published under that name.
  - **The small DeepChem checkpoints are ChemBERTa-2.** `ChemBERTa-5M/10M/77M-MLM` and `-MTR`
    date from 2022. They have 3 layers, hidden size 384 and about 3.4M parameters, and are
    pretrained on PubChem.
  - **Their tokenizer is broken, and the weights show it (code).** The vocabulary holds 591
    atom-level tokens (`Cl`, `[nH]`, `[C@@H]`), but it is stored as a byte-level BPE with zero
    merges and no `unk_token`. The fast tokenizer can therefore emit only the 33 single-character
    tokens and silently drops every other character.
    - In `ChemBERTa-77M-MLM`, every multi-character token has an embedding norm of about 2.4 and
      an MLM output bias of about −12, the same as the never-used `[unused5]`. Single-character
      tokens sit at about 3-4 and about +0.1. So pretraining never produced an atom-level token.
    - The model was pretrained on the broken strings. It reads chlorine as carbon, bromine as
      boron, and drops charges and bracket H.
    - On MSGYM, 28.4% of the 18,110 unique raw SMILES (stereo removed) lose characters: 12.7%
      contain Cl, 2.8% Br and 15.0% a bracket atom.
    - Two open Hugging Face discussions on the checkpoint, "The Tokenizer is Broken" (#3) and
      "Broken Tokenizer" (#5), report the same break.
  - **c3-MoLFormer** is the only other ChemBERTa-3 release with weights: `MoLFormer-c3-1.1B`,
    46.8M parameters, a linear-attention architecture loaded through IBM's remote code.
- **Inferred from the merges.** A merge rank is the corpus frequency order, so the merge list shows
  what the tokenizer's corpus contained:
  - **Stereo-free.** `@@` was never merged: no vocabulary entry contains `@@`, `[C@` or a
    merged `/` or `\`. In a corpus where roughly a third of molecules carry stereo, as in raw
    MSGYM (31%), `@@` would be among the most frequent punctuation runs.
  - **Aromatic.** `c c` is merge #0 and `cc c` merge #3, so lowercase aromatic SMILES dominate.
    Kekulé strings are rare.
  - **Charged forms are common.** `+ ]` is merge #5 and `- ]` merge #19. That fits ZINC's pH-7
    protomer SMILES (`[NH+]`, `[NH3+]`, `C(=O)[O-]`); MSGYM SMILES are neutral. This is an
    inference about ZINC's distribution files and is not tested.
- **Inferred from compression.** A BPE vocabulary compresses text that looks like its training
  corpus best. On 4,000 MSGYM molecules, stereo-free canonical aromatic SMILES compress best:

  | SMILES form (4,000 MSGYM molecules) | tokens p50 / p99 / max | characters per token |
  | --- | --- | ---: |
  | our standardized `mol_df` form (RDKit canonical, no stereo) | 29 / 85 / 112 | **1.439** |
  | raw MSGYM, isomeric | 31 / 120 / 215 | 1.389 |
  | randomized (RDKit `doRandom`), aromatic | 33 / 90 / 114 | 1.356 |
  | RDKit canonical, Kekulé | 37 / 89 / 126 | 1.309 |

  Nothing approaches 512 tokens, so truncation never applies. The compression ratio is a weak
  proxy, and nothing checks it further: Stage 0b was dropped.

**Working conclusion.** The pretraining text was probably canonical, aromatic and stereo-free, with
ZINC's charge states. That toolkit's canonical order need not match RDKit's. Randomized SMILES and
Kekulé forms are out of distribution for the pretrained model. That is a reason to measure them,
not to rule out augmentation.

## Established facts about the data and code

**Data** (`data/df/ms_gym_benchmark_1_5_df.csv`, `data/proc/ms_gym_benchmark_1_5/`; header of
[the MSGYM MASSter config](../../config/masster/massspecgym/msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_ema999.yml)):

- 99,306 train / 9,731 val / 9,954 test spectra after processing, on 12,310 / 2,331 / 2,319
  `mol_id`s (8.07 train spectra per molecule). `mol_df` has 16,967 rows, 16,960 of them used.
  The raw CSV's `fold` column, before processing drops 38 rows, has 12,321 / 2,334 / 2,319
  first-block InChIKeys.
- All `[M+H]+` and CID. FT 81,313, QTOF 37,678. ACE is present on every row, and NCE is derived
  from it.
- **Raw SMILES** are RDKit canonical isomeric (3,000 of 3,000 sampled, RDKit 2022.09.5). 31.0% of
  the 18,110 unique raw SMILES carry stereo, and 941 InChIKeys have more than one raw SMILES.
- **`mol_df` SMILES are standardized.** `rdkit_ml_standardize` strips stereo, runs RDKit `Normalize`
  (which rewrites sulfoxides as `[S+][O-]`) and `Reionize`, and canonicalizes tautomers.
  - **Rate.** Joining each spectrum's raw SMILES to its `mol_id` through `dset_spec_id`, the
    stereo-free raw form differs from the `mol_df` SMILES on **1,491 of 16,960 molecules
    (8.8%)**. The 4.6% recorded earlier matched on InChIKey and so dropped the rewrites that
    change the key.
  - **Tautomers dominate.** Of the 1,500 differing (molecule, form) pairs, 1,418 equal RDKit's
    canonical tautomer of the raw form, and 103 involve a charged atom.
  - **Half change the molecule's key.** 751 of the 1,500 change the first InChIKey block, and
    the other 749 keep the full key.
  - **Some rewrites are chemically doubtful.** Examples:
    - `CS(C)=O` → `C[S+](C)[O-]`;
    - the imidazole H moves (`c1cnc[nH]1` → `c1c[nH]cn1`);
    - guanine becomes `Nc1nc(=O)c2ncn(...)c2[nH]1`;
    - a prenyl double bond migrates (`c(CC=C(C)C)` → `c(C=CC(C)C)`);
    - a 2-amino-4H-chromene becomes its imine.
  - **One molecule, two raw forms.** 14 `mol_id`s have two stereo-free raw forms that tautomer
    canonicalization merged. D1 needs a rule for them, such as each spectrum's own form.

**Code:**

- **The batch carries the standardized SMILES, not the raw one.** `mol_params.smiles` puts
  `mol_entry["smiles"]`, the `mol_df` string, into the batch
  ([spec_mol_dataset.py:214](../../src/fragnnet/dataset/spec_mol_dataset.py#L214)). `spec_df`
  has no SMILES column. The raw SMILES exist only in `data/df/ms_gym_benchmark_1_5_df.csv`, and
  all 118,991 processed spectra join to it by `dset_spec_id`. The fingerprints NEIMS reads are
  also computed from the `mol_df` molecule.
- **`randomize_smiles` is broken.**
  [data_utils.py:86](../../src/fragnnet/utils/data_utils.py#L86) calls `rng.shuffle` on the value
  that `np_temp_seed` yields, which is `None`. Its tests are strict xfails.
- **`split_smiles` fails on 569 of the 18,110 raw isomeric SMILES.** Its regex matches `\\`, not
  `\`. The HF tokenizer does not use it; do not build on it.
- **RDKit's seeded generator works.** `Chem.MolToRandomSmilesVect(mol, n, randomSeed=s)` exists in
  the installed RDKit and returns the same strings for the same seed.
- **The binned head exists.** NEIMS and MassFormer predict 0.01 Da bins up to m/z 1,500 through the
  bidirectional `SpecFFN` head, with a sparse cosine loss on sqrt intensities.
  - The prediction path undoes the transform before scoring
    ([spectrum_pl.py:1285](../../src/fragnnet/pl_model/spectrum_pl.py#L1285)), so `cos_sim_0.01`
    is linear.
  - **The head is larger than the encoder.** With the MSGYM NEIMS settings (`mlp_hidden_size`
    1,024, `ff_output_map_size` 256, 150,000 bins), the three low-rank output maps of the
    bidirectional `SpecFFN` hold about 116M parameters.
  - **No reusable per-group LR.** `MassFormerPL` has two parameter groups, but they differ only in
    weight decay and share one LR. Its scheduler call does not pass `schedule_type` or
    `total_steps`, so it cannot run cosine. That is the silent fallback to exponential decay that
    `SpectrumPL.configure_optimizers` documents.
    - `SpectrumPL` has a cosine path, a norm/bias decay exemption and a per-parameter decay hook
      (`_weight_decay_overrides`).
    - Token and position embeddings are 2-D, so the exemption does not cover them.
    - Nothing in the repository sets a separate LR per group.
  - Their only MSGYM configs are merged, on the old `massspecgym_f` split. No NEIMS or MassFormer
    number is recorded for `ms_gym_benchmark_1_5`.
- **Environment.** `transformers` 4.46.3 and `tokenizers` 0.20.3 were installed in FRAGNNET-GPU
  on 2026-09-27, with the user's approval. The install added only `huggingface_hub` 0.36.2,
  `safetensors` 0.8.0, `regex` and `hf-xet`; torch 2.4.1 and numpy 1.26.4 were unchanged.
  - **Cluster.** The wheelhouse is not checked. Cluster jobs install packages with
    `pip install --no-index` in `scripts/setup_job_cc.py`, and nothing there adds these
    packages yet.
  - **Checkpoint.** It is at `data/pretrained/ChemBERTa-100M-MLM`, whose `REVISION` file holds
    the D2 revision. `data/` is gitignored, so the directory has to be copied to a cluster.
    Fir compute nodes have internet; Narval's do not, and the model never downloads.

## Design decisions

| # | Decision | Recommendation | Why |
| --- | --- | --- | --- |
| D1 | SMILES form fed to the encoder | **Basic model:** the `mol_df` SMILES that `mol_params.smiles` already puts in the batch: RDKit canonical, aromatic, stereo-free and standardized. The raw form with stereo removed is [deferred](#deferred-design-choices). | It needs no data-pipeline change, and E1, T2 and E4 read the same structure, as MASSter does. Like the pretraining text, it is stereo-free and aromatic, and it tokenizes as compactly as the raw form (1.439 and 1.436 characters per token, on different samples). The two differ on 8.8% of molecules, mostly RDKit tautomer rewrites that are less likely in ZINC text. |
| D2 | Checkpoint and depth | `DeepChem/ChemBERTa-100M-MLM` at revision `f5c45f44d3061f0346888f5c09db17ec1146d29d`, **truncated to its first 6 of 12 layers** for now: the pretrained embeddings and layers 0-5, about 49M parameters (42.5M outside the embeddings). The full 12-layer model (D12) is deferred. | The user asked for a smaller base model that is still ChemBERTa-3 (2026-09-26). No smaller ChemBERTa-3 checkpoint is released. The small DeepChem checkpoints are ChemBERTa-2, pretrained through a broken tokenizer. Cutting layers keeps ChemBERTa-3's tokenizer, pretraining corpus and `RobertaModel` code. c3-MoLFormer is a different architecture, and the README does not show scale winning. |
| D3 | Pooling | Mean over non-padding tokens | An MLM-only RoBERTa never trains `<s>` as a summary token. |
| D4 | Output head | The existing binned 0.01 Da bidirectional `SpecFFN`, as in NEIMS | It matches the controls, reuses the tested code, and is scored linearly. The exact-mass head (Stage 3) is deferred. |
| D5 | CE and instrument | `ce_insert_type: qet` plus the instrument embedding, identical in every arm | The categorical `embed` aliases CE values. `qet` works in the shared `CEModel` that NEIMS uses, and it won the held-out-NCE probe ([project state](../project-state.md)). No full training run has used it yet: NIST20 arm `n1` is not submitted. |
| D6 | Intensity space | `ints_transform: sqrt`, as in the tuned NEIMS and MassFormer recipes | The metric stays linear through the untransform. A linear-space arm is an optional follow-up on the winner. |
| D7 | Fine-tuning recipe | AdamW; encoder LR 3e-5 (all encoder parameters, embeddings included), head LR 4e-4 (NEIMS's); no weight decay on embeddings, LayerNorm or biases; 1,000-step warmup, cosine decay; 100 epochs; best-val checkpoint; no EMA in any arm | The recipe is kept matched. Excluding embeddings from decay stops rows of tokens absent from MSGYM from shrinking while they receive no gradient. EMA can be added to all arms later. |
| D8 | Gradient path | **End to end.** The spectrum loss back-propagates through the head, the pooling, every kept layer (6 under D2), and the token and position embedding matrices. No frozen layers, no cached encoder outputs, no stop-gradient. | The user's requirement (2026-09-26). The SMILES encoder has to be reshaped by the spectrum loss, not read as fixed features. Augmentation rules out caching anyway: each step sees new strings. |

### End-to-end training (D8)

- **What receives gradient.** Everything the forward pass uses.
  - The word-embedding rows of the tokens in the batch: row-sparse, so tokens absent from MSGYM
    keep their pretrained values.
  - The position embeddings up to the batch's longest sequence.
  - All attention and FFN weights, and the LayerNorms.
  - The pooling and the head.
- **What must not appear in the path.**
  - `torch.no_grad()`, `.detach()` or `requires_grad = False` on any encoder parameter.
  - The layer-freezing pattern of `MassFormerModel` (`mf_fix_num_pt_layers`), which must not be
    copied.
  - Embeddings precomputed into `mol_df` or a cache.
  - The only no-grad path is evaluation.
- **Weights: the encoder only.** Training loads an embedding model and nothing else:
  `RobertaModel.from_pretrained(path, num_hidden_layers=6, add_pooling_layer=False)`.
  - That is the token and position embeddings plus layers 0-5, 49,009,152 parameters, read from
    the `-MLM` checkpoint. The checkpoint is the only released ChemBERTa-3 encoder; "MLM" names
    how it was pretrained.
  - The MLM head (`lm_head.*`, 600,052 parameters, its decoder tied to the word embeddings) and
    layers 6-11 are never loaded into a training job.
  - The checkpoint has no pooler, and D3 mean-pools, so `add_pooling_layer=False` avoids a
    randomly initialized pooler that nothing trains.
  - This is standard Hugging Face loading. The smoke test confirms that layers 0-5 equal the
    checkpoint's and that only `lm_head.*` and layers 6-11 are reported unused.
  - No step uses the MLM head. Stage 0b and P1 did, and they were
    [dropped](#dropped-outright).
- **Verification, before any run.** A unit test on a tiny random `RobertaConfig` runs one
  backward pass of the spectrum loss. It asserts nonzero gradients on
  `embeddings.word_embeddings.weight`, on exactly the rows of the batch's token ids, and on
  `position_embeddings` and every layer.
- **Monitoring.** The gradient norm per group (embeddings, each layer, head), which shows the
  embeddings receive gradient in the real run. The drift measures and the L2-SP remedy are
  [deferred](#deferred-design-choices).
- **Cost (estimate, not measured).** Dynamic padding pads to the longest string in the batch.
  - **Padded length.** With the checkpoint's BPE merges applied to train spectra, strings average
    28 tokens canonical and 31 randomized. Batches of 256 pad to about 90 and 96, so about 68% of
    the padded tokens are padding.
  - **Unbucketed, 6 layers (D2).** One step through the 42.5M non-embedding parameters is about
    6 TFLOP, or about 2.5 PFLOP per epoch of 388 steps. fp32 activations are about 9 GB. Both
    scale with depth, so the deferred D12 doubles them to about 12 TFLOP and 17-18 GB.
  - **The head.** It adds about 116M parameters and their optimizer state, about 2 GB in fp32.
  - A smoke run replaces these estimates.
- **Local GPU (measured 2026-09-26, stand-in model).** The workstation has an RTX 2070 SUPER:
  7.6 GiB usable, Turing, so fp16 but no bf16 or TF32.
  - **What ran.** A PyTorch `TransformerEncoderLayer` stack of the D2 shape (768 wide, 12 heads,
    FFN 3,072, post-LN, GELU), plus the repository's `SpecFFN` with the NEIMS MSGYM settings.
    Each configuration did forward, backward and an AdamW step in its own process. Peak
    allocated memory and steady-state step time are below.
  - **Parameters.** Encoder 49M at 6 layers; head 121.5M. The head alone takes 4.4 GB at
    batch 256.

  | Model, batch 256, 96 tokens | Peak | s/step | min/epoch | 100 epochs |
  | --- | ---: | ---: | ---: | ---: |
  | 6 layers, fp32 | out of memory | | | |
  | 6 layers, fp32, gradient checkpointing | 5.6 GB | 1.43 | 9.2 | about 15 h |
  | 6 layers, fp16 autocast on the encoder, gradient checkpointing | 5.6 GB | 0.58 | 3.8 | about 6 h |
  | 6 layers, fp16 autocast, no checkpointing, batch 128 (90 tokens) | 6.2 GB | 0.25 | 3.2 | about 5 h |
  | E4 NEIMS head, fp32 | 4.4 GB | 0.12 | 0.8 | about 1.3 h |
  | 12 layers (D12), fp16 autocast, gradient checkpointing (90 tokens) | 6.6 GB | 0.97 | 6.3 | about 10.5 h |

  - **The basic plan fits.** Its six runs take about 1.2 days in sequence with fp16 autocast,
    or about 2.7 days in fp32.
  - **Keep precision matched.** Every ChemBERTa arm must use the same precision, so a run set
    stays on one machine.
  - **Not measured.** Hugging Face's eager attention stores attention probabilities that this
    stand-in does not; loading with `attn_implementation="sdpa"` avoids them. T2's per-step
    RDKit calls in the forward pass are also not measured.
  - **Other jobs share the GPU.** Other sessions' processes held 0.3-0.4 GB during the
    measurement.

## Plan: basic model

### Runs (2 seeds each)

| Arm | Model | Notes |
| --- | --- | --- |
| E1 | ChemBERTa-3, first 6 layers (D2), end to end (D8), canonical `mol_df` SMILES (D1) | D1-D8. |
| T2 | E1 with random SMILES in training | On each train step, every SMILES is replaced by one unrestricted random SMILES of the same molecule (`Chem.MolToRandomSmilesVect(mol, 1, randomSeed=s)`, with `s` from a generator seeded by the run seed). Evaluation uses the canonical string. Everything else equals E1. |
| E4 | NEIMS (Morgan + RDKit + MACCS fingerprint MLP) | A new unmerged `ms_gym_benchmark_1_5` config with `qet`. Its fingerprints come from the same `mol_df` molecule as E1's SMILES. |

- **T2's strings.** On 1,500 MSGYM molecules, 20 of 20 draws per molecule were distinct. They
  tokenize at 1.351 characters per token against 1.436 for canonical SMILES, so they sit further
  from the pretraining text (see the deferred [augmentation design](#augmentation-design)). Over
  100 epochs, each molecule sees about 800 strings.
- **Read.** Val `cos_sim_0.01`: the best epoch and the last-10-epoch tail, with the seed spread
  from each pair. The train loss, already logged, is the train-fit signal.
- **Budget.** 6 training runs, after a smoke run of E1 that measures epoch time and peak memory.
  Epoch time is unmeasured.
- **If E1 works,** the first additions are E3, the same 6-layer model from random init, to show
  what pretraining contributes, and the one-time test read. Both are
  [deferred](#deferred-not-scheduled).

## Implementation: basic model

Built on 2026-09-27 and uncommitted: see [Built](#built-2026-09-27) after this list. The list
is the design it followed.

1. **Dependencies.** Confirm, then install `transformers` and `tokenizers`, pinned, in
   FRAGNNET-GPU and in the cluster wheelhouse. Download the checkpoint at the D2 revision to a
   local directory and load it by path, so Narval works offline.
2. **Model type.** Add `chemlm` to `VALID_MODEL_TYPES`, to `setup_utils`, and to
   `config/template.yml` under a `chemlm_*` prefix.
   - The keys: checkpoint path, number of layers kept (6), encoder LR, and `p_trav` (0 for E1,
     1 for T2).
   - Only the new model's code reads them. Shared code may read any template key without
     checking `model_type` ([runner.py:47](../../src/fragnnet/runner.py#L47)).
3. **Data.** Turn on `mol_params.smiles`. The batch then carries each spectrum's `mol_df` SMILES
   ([spec_mol_dataset.py:214](../../src/fragnnet/dataset/spec_mol_dataset.py#L214)), so the
   dataset does not change.
4. **Model.** One differentiable forward pass:
   - on train steps with `p_trav` above 0, each SMILES is replaced by one random SMILES of the
     same molecule, from a generator the model seeds with the run seed;
   - the fast tokenizer pads to the batch's longest string;
   - then the encoder (`RobertaModel`, loaded as in D8), mean pooling, the `qet` CE and instrument
     embeddings, an MLP, and `SpecFFN`.

   Drawing the random SMILES inside the model keeps shared dataset code unchanged.
5. **Training loop.** Subclass `BinnedPL` with three optimizer groups, each carrying its own `lr`
   and weight decay:
   - encoder weights, with decay;
   - encoder embeddings, LayerNorms and biases, without decay;
   - the head.

   Build the groups on `SpectrumPL`'s `_split_weight_decay_parameters` and keep its
   `build_lr_scheduler` call with `schedule_type` and `total_steps`. `LambdaLR` scales each
   group's own base LR. Do not copy `MassFormerPL.configure_optimizers`: it shares one LR and
   cannot run cosine. Log the gradient norm per group.
6. **Tests.**
   - The D8 gradient test: nonzero gradient on the word-embedding rows of the batch's tokens, on
     the position embeddings and on every layer, from the spectrum loss alone.
   - A forward and backward pass on a tiny random `RobertaConfig`, needing no download.
   - Truncation: a tiny random 4-layer `RobertaForMaskedLM`, saved and reloaded as a 2-layer
     `RobertaModel`, keeps the embeddings and layers 0-1 bit-identical. It has no layer 2 or 3,
     no MLM head and no pooler.
   - Random SMILES: the same seed gives the same strings, and every string parses back to the
     input's canonical SMILES.
7. **Configs.** E1 and T2 under `config/massspecgym/chemlm/`, and E4 as an unmerged
   `ms_gym_benchmark_1_5` NEIMS config with `qet`. Run the smoke run before sizing the jobs.

### Built (2026-09-27)

- **Code.**
  - [chemlm_model.py](../../src/fragnnet/model/chemlm_model.py): the tokenizer, the truncated
    `RobertaModel` with SDPA attention, mean pooling, and the `qet` CE and instrument inputs
    into the NEIMS `SpecFFN`.
  - [chemlm_pl.py](../../src/fragnnet/pl_model/chemlm_pl.py): `ChemLMPL(BinnedPL)`.
  - Registration in `runner.py`, `setup_utils.py` and `pl_model/__init__.py`.
  - A `ChemLM only` section in `config/template.yml`, with the keys `chemlm_ckpt_dp`,
    `chemlm_num_layers`, `chemlm_encoder_lr`, `chemlm_p_trav` and
    `chemlm_gradient_checkpointing`.
  - An optional `chemlm` dependency group in `pyproject.toml`.
- **Loading is checked.** `from_pretrained` runs with `output_loading_info`.
  - The model raises if any encoder weight is missing or resized, which would mean random
    initialization, or if an unused weight is anything but `lm_head.*` or a cut layer.
  - On the real checkpoint: 49,009,152 parameters. Layers 0-5 and the embeddings are
    bit-identical to the full model's, and 101 keys are unused: the MLM head and layers 6-11.
- **`transformers` is imported lazily**, inside `ChemLMModel`. Environments without it,
  including today's cluster venvs, still import every other model.
- **Random SMILES (T2).** Each training SMILES is replaced with probability `chemlm_p_trav`, as
  the plan says.
  - The generator is seeded from `seed` when the model is built. Its state is not checkpointed,
    so a resumed T2 run draws a different stream.
  - Parsed molecules are cached per SMILES.
- **Optimizer.** `ChemLMPL` returns four groups, encoder or head crossed with decayed or exempt,
  through `SpectrumPL`'s own group hook. So `SpectrumPL`'s cosine scheduler runs unchanged and
  scales each group from its own LR.
  - The exempt set is `_split_weight_decay_parameters` plus the encoder's token, position and
    token-type embeddings.
  - The model requires `weight_decay_exclude_norm_bias: True`, and rejects `compile` and
    `activation_checkpointing`. Whole-model recompute would redraw the random SMILES.
- **Monitoring (D8).** The gradient norm of the embeddings, each layer and the head is logged
  every `log_every_n_steps` (50 in the configs).
- **Configs** ([config/massspecgym/chemlm/](../../config/massspecgym/chemlm/)): E1
  `e1_chemberta3_l6.yml`, T2 `t2_chemberta3_l6_random_smiles.yml`, E4 `e4_neims_qet.yml`. The
  second seed comes from `--override seed=132548 wandb_name=..._s1`.
  - **Checkpoints.** All three arms set `upload_checkpoints: false`, at the user's request
    (2026-09-27). Each run keeps best and last under `wandb/run-*/files/ckpt` in the checkout.
    - E1 and T2: 2.05 GB apiece, about 4.1 GB per run.
    - E4: 1.5 GB apiece, about 3 GB per run.
    - About 22 GB over the six runs.
- **Fir jobs (`scripts/setup_job_cc.py`).** Only for `model_type: chemlm`, the generator:
  - stages `chemlm_ckpt_dp` into the job's working directory, and fails if `config.json` did
    not arrive (a checkpoint missing from the checkout already fails at generation);
  - installs `transformers==4.46.3 tokenizers==0.20.0 huggingface_hub==0.36.2
    safetensors==0.8.0 regex` with `--no-index`;
  - exports `TOKENIZERS_PARALLELISM=false`.

  Every other job's script is unchanged, which a test checks.
  - **Fir wheelhouse** (`avail_wheels --all-versions`, run by the user on 2026-09-27): every
    pinned version is there. The exception is `tokenizers` 0.20.3, the FRAGNNET-GPU version:
    Fir's 0.20.x is 0.20.0.
  - **0.20.0 was checked in an isolated venv**, with FRAGNNET-GPU unchanged. It gives
    identical token ids to 0.20.3 on all 16,967 `mol_df` SMILES, on one random SMILES of each,
    and on a padded batch of 256. The 18 `chemlm` tests pass under it.
  - Local stays on 0.20.3; `pyproject.toml` allows `tokenizers>=0.20,<0.21`.
- **Choices the plan left open, now fixed in the configs.**
  - Weight decay is NEIMS's tuned 1e-7 on weight matrices, in every arm.
  - The cosine schedule ends at 1% (`lr_min_ratio` 0.01, as the MASSter MSGYM recipe does)
    over 38,800 steps: 100 epochs of 388 steps.
  - `qet` statistics are spectrum-weighted over the train split: ACE mean 37.8 and std 27.8 eV;
    NCE, derived as ACE × 500 / precursor m/z, mean 68.3 and std 59.1.
  - Both CE channels are fed, as MASSter's MSGYM config does, with no precursor-type input (all
    rows are `[M+H]+`).
  - `eval_test_split: false` is explicit.
  - Val logs `cos_sim_0.01` and its sqrt variant. `cos_hun_inst_tol` is not logged. A 0.01 Da
    bin centre can sit up to 0.005 Da from the true peak, more than the FT tolerance (10 ppm,
    with a 0.002 Da floor) below m/z 500, so it would penalize the binned head for binning.
- **A NEIMS bug fixed on the way.** `NeimsModel.forward` never passed `spec_ace` to `embed_ce`,
  so any NEIMS config with `spec_params.ace` failed its assert. That includes the merged MSGYM
  NEIMS configs, which set `ace: True`, and E4. It now reads ACE from the batch, as
  `FraGNNetModel` does.
- **Tests.**
  - [test_chemlm.py](../../tests/test_chemlm.py), 18 tests on a tiny random 4-layer checkpoint
    with no download:
    - truncation is bit-identical, with no MLM head or pooler;
    - the loading guard;
    - normalized output, and padding does not change a prediction;
    - the D8 gradient test: nonzero rows are exactly the batch's token ids and positions, and
      every layer's weight matrices get gradient;
    - gradient checkpointing gives the same gradients;
    - random SMILES are deterministic, parse back to the input, and are used only in training;
    - the optimizer groups and the per-group cosine schedule;
    - rejected settings.
  - [test_neims_model.py](../../tests/test_neims_model.py): NEIMS with each CE-channel
    combination.
  - With the config-template, runner-config, CE/`qet`, data-utils and metric tests: 251 passed
    and 2 xfailed, the existing `randomize_smiles` xfails.
- **CPU pipeline check.** T2 and E4 each ran 4 training steps of batch 32 through
  `scripts/run_pl_model_fit.py` with `debug_overfit` and exited 0. All eight gradient-norm
  groups were nonzero, the embeddings included.
- **GPU smoke run (2026-09-27, local RTX 2070 SUPER).** One E1 epoch, fp32, with
  `chemlm_gradient_checkpointing`, W&B disabled.
  - **Speed.** 1.30 s/step (50 steps in 65 s); one epoch with validation took 9.4 min.
  - **Memory.** Peak 6.1 GB as nvidia-smi reports it, context and allocator cache included.
  - **Score.** Val `cos_sim_0.01` 0.196 after epoch 1, still inside the warmup (388 of 1,000
    steps).
  - **Checkpoint.** 2.05 GB with optimizer state; it loads, with layers 0-5 in its state dict.
  - **Then the process failed, and lost `last.ckpt`.** With both `upload_checkpoints` and
    `delete_checkpoints` off, the runner copied checkpoints into a `/tmp` temporary directory
    and back.
    - `last.ckpt` copied; the second 2 GB file then filled the workstation's 3.7 GB `/tmp`.
    - The temporary directory's cleanup deleted the copied `last.ckpt` at exit.
    - The log, also on `/tmp`, lost the traceback.
  - **Fixed the same day, in `runner.py`.** `hold_checkpoints_from_wandb` moves checkpoints
    into a `ckpt_hold_*` directory beside `wandb.run.dir`. That is inside the run folder,
    outside what `wandb.finish()` uploads, and on the same filesystem, so every move is a
    rename.
    - `restore_held_checkpoints` moves them back after `wandb.finish()`.
    - `mkdtemp` keeps them if the process dies in between.
    - With W&B disabled nothing is uploaded, so nothing moves.
    - [Tests](../../tests/test_runner_checkpoint_hold.py) cover the Fir layout, where the job's
      `wandb` is a symlink into the checkout.
    - A tiny CPU run with W&B offline and upload off exited 0. It left `last.ckpt` and the
      best checkpoint in `files/ckpt`, no hold directory, and no `.ckpt` in the offline
      `.wandb` log that `wandb sync` reads.
  - **Log noise.** `tokenizers` warns on every DataLoader fork ("parallelism has already been
    used"). It is harmless; `TOKENIZERS_PARALLELISM=false` in the job environment silences it.
  - **Not measured.** Fir epoch time. A Fir slice runs without checkpointing, with TF32.

## Risks and limits

- **The binned head caps absolute scores.** E1 is compared with E4, which has the same head, not
  with MASSter.
- **The pretraining corpus is inferred.** It comes from the tokenizer and code, not the paper,
  and the tokenizer's corpus could differ from the model's. Nothing checks it directly.
- **Domain shift.** ZINC20 is purchasable and drug-like; MSGYM includes natural products and
  metabolites. The MSGYM split is MCES-based, so val and test measure exactly this transfer.
- **Small training set.** 12,321 train molecules against a 49M-parameter encoder: memorization of
  molecule identity is likely. Random SMILES vary the strings, not the molecules, so T2 can stop
  string-level memorization but not molecule-level memorization.
- **End-to-end training can overwrite the pretraining.** With 12,321 molecules, the spectrum
  gradient can pull the encoder away from its pretrained solution. The basic runs do not measure
  this; the drift measures and the L2-SP remedy are deferred.
- **A cut encoder is not a trained small model.** Layer 6's output was never trained to be a final
  layer. The MLM objective only shaped it as input to layers 6-11, so fine-tuning has to adapt it.
  Without E3, the basic runs cannot say how much the pretrained weights contribute.
- **Random SMILES are off the pretraining distribution.** They tokenize less compactly than
  canonical SMILES, so T2 can cost more than it returns; that is what T2 measures.
- **`mol_df` structures.** On 8.8% of molecules, `mol_df` holds RDKit tautomer and normalization
  rewrites of the raw SMILES, some chemically doubtful. The basic runs inherit them, as MASSter
  and NEIMS do.
- **Pretraining overlap is not leakage.** Test molecules' SMILES are probably in ZINC or PubChem,
  but without spectra.

## Deferred (not scheduled)

The user cut the plan to the basic model on 2026-09-26 ("for now all I want is a basic model, just
want to see if it works"). Everything below is kept for later and is not scheduled. E1, T2 and E4
are the basic runs; where a deferred stage lists them, it reuses those runs.

### Dropped outright

The user dropped two steps on 2026-09-26, asking for a simple model. They were the only steps that
needed the MLM head:

- Stage 0b, the MLM likelihood of each SMILES form. It would have checked D1 and ranked the
  augmentation families by distance from the pretraining text.
- P1, augmented continued MLM pretraining on train-fold molecules. It was a ladder step and the
  Stage 1 stop rule's last try.

### Deferred design choices

- **D1, the raw SMILES form.** The raw MSGYM SMILES with stereo removed, as RDKit canonical
  aromatic SMILES, were the earlier recommendation. RDKit's charge-separated sulfoxides and
  tautomer choices are unlikely in ZINC text, and half the tautomer rewrites change the
  molecule's InChIKey.
  - The raw form comes from the raw CSV, joined on `dset_spec_id`, with stereo removed.
  - The 14 `mol_id`s with two raw forms need a rule, such as each spectrum's own form.
  - E4's fingerprints must then come from the same molecule, so E1 and E4 read the same
    structure.
  - Check the ChemBERTa-3 paper's Methods for how its pretraining SMILES were written first.
- **Drift measures.** Both are logged every epoch.
  - The relative distance from the pretrained weights, per group.
  - Linear CKA between the mean-pooled embeddings of a fixed probe batch of val SMILES under the
    current encoder and under the pretrained one, under no-grad. The pretrained embeddings are
    computed once and cached. CKA stays at 1 while the pretrained similarity structure among
    molecules survives, under any rotation or uniform rescaling.
- **L2-SP arm.** If drift costs val late in training, this arm penalizes the squared distance of the
  encoder weights from their pretrained values, with the penalty weight chosen on val with one
  seed. It replaces an auxiliary-MLM arm, which would have put the MLM head and layers 6-11 back
  into training.
- **Full depth (D12).** The chosen configuration with all 12 pretrained layers, 91.5M encoder
  parameters.
- **Length bucketing.** Batching by length cuts encoder compute and activations roughly threefold.
- **Train-subset cosine.** Nothing in the model code logs it yet.

### Implementation the deferred stages add

- Tautomer and Kekulé pools cached at preprocessing, and the sampler in `__getitem__` for train.
- Restricted traversal needs `RenumberAtoms` with a numpy `Generator`, which means fixing
  `randomize_smiles`.
- The V-view consistency loss and TTA averaging.
- Config keys for pooling, the freeze flag, the SMILES form, the traversal type, `p_kek`,
  `p_taut`, views V, λ and TTA K.
- Tests: collate padding; the sampler is deterministic per seed; traversal and Kekulé variants give
  the input's canonical SMILES back, and tautomers give the same RDKit canonical tautomer
  (InChIKey equality fails on keto-enol pairs); the V-view consistency loss and TTA averaging.

### Augmentation design

The aim is many strings per molecule that are all correct. Every variant below writes the same
compound, so the target spectrum does not change.

#### Families (measured 2026-09-26, 1,500 MSGYM molecules)

Variants start from the D1 molecule. "Characters per token" uses ChemBERTa-3's BPE; higher means
closer to the pretraining text, where canonical SMILES score 1.436.

| Family | How | Distinct strings per molecule | Molecules with more than one | Characters per token | Tokens p50 |
| --- | --- | --- | ---: | ---: | ---: |
| canonical | `MolToSmiles` | 1 | 0% | 1.436 | 29 |
| rooted canonical | canonical traversal from each atom (`rootedAtAtom=i`) | p50 23, p90 34 | 100% | 1.383 | 31 |
| restricted random | random atom renumbering, then `canonical=False` | 20 of 20 draws | 100% | 1.347 | 33 |
| unrestricted random | `MolToRandomSmilesVect` (`doRandom`) | 20 of 20 draws | 100% | 1.351 | 33 |
| tautomers | RDKit `TautomerEnumerator`, capped at 64; canonical SMILES of each | p50 3, p90 22 | 75.0% | 1.383 | 33 |
| Kekulé assignments | `ResonanceMolSupplier(KEKULE_ALL)`, formal charges unchanged | p50 2, p90 4 | 64.5% | 1.281 | 41 |
| explicit H | `allHsExplicit=True` | 1 | 0% | 1.658 | 69 |

The families compose. A variant is a tautomer, written aromatic or in one Kekulé assignment, then
traversed. The space multiplies across the three.

- **Rooted canonical** is the grammar closest to the pretraining text, and it is finite: at most
  one string per heavy atom.
- **Tautomers** are the same compound, but RDKit's rules also produce improbable forms such as
  imidic acids and enols. A keto-enol move also changes the first InChIKey block. They enter at low
  probability, and the ladder decides whether they stay.
- **Kekulé assignments** are furthest from the pretraining text (1.281) and use 40% more tokens.
- **Not planned.**
  - Explicit H: it doubles the length, and the pretraining text shows bracket H only on charged or
    aromatic atoms.
  - Charge-separated forms: only 0.3% of molecules have a sulfoxide. Nitro groups and N-oxides have
    a single form that RDKit accepts.
  - Stereo: absent from the pretraining text and merged out of `mol_df`.
  - Protomers: for `[M+H]+` the protonation site is information about the precursor, not an
    augmentation, and the environment has no pKa tool.

#### Sampler

Each training sample draws, with a seed derived from (run seed, epoch, sample index):

1. **Tautomer.** The D1 tautomer with probability 1 − `p_taut`, otherwise a uniform draw from
   the molecule's cached tautomers. Under the raw SMILES form (deferred D1), this is not RDKit's
   canonical tautomer, which differs from it on 8.4% of molecules.
2. **Aromaticity.** Aromatic with probability 1 − `p_kek`, otherwise a uniform draw from its
   Kekulé assignments.
3. **Traversal.** Canonical with probability 1 − `p_trav`, otherwise rooted, restricted or
   unrestricted random, whichever the arm fixes.

- **Caching.** Tautomers and Kekulé assignments are enumerated once per molecule at preprocessing
  and cached. All families together took 23 s per 1,500 molecules, so a few minutes for all 16,967.
  Traversals are drawn on the fly, on the train split only. The seed keeps runs reproducible and
  stops DataLoader workers from repeating one stream.
- **Views per sample.** At V = 1, each sample gets one variant. At V ≥ 2, each sample gets V
  variants, the target loss applies to each, and λ × the mean of (1 − cosine) between each view's
  prediction and their average is added. λ ∈ {0.1, 1} is chosen on val with one seed.
  - **Matching is undecided.** At 256 spectra per step, V4 runs 1,024 strings per step: four
    times the encoder compute. At 6 layers its unbucketed fp32 activations, about 35 GB, leave
    almost no room on the 40 GB slice for the head, its optimizer state and its outputs.
  - **The choice.** Either hold strings per step fixed (64 spectra × 4 views) or hold spectra per
    step fixed with gradient checkpointing, then report compute per arm.
- **Count against variety.** At `p_trav = 1` and V = 1, each molecule already sees about 800
  strings over a 100-epoch run, at about 8 spectra per molecule per epoch. Extra views and families
  add kinds of variation, not count.

#### Test-time augmentation (TTA)

- **Method.** Draw K variants from the training sampler, so test strings come from the training
  distribution, and add the canonical string. Average the untransformed spectra and renormalize.
  K ∈ {4, 16, 64}.
- **Reporting.** It is a K+1-pass ensemble, reported next to a 2-seed ensemble, not folded into the
  single-model number.
- **Uncertainty for free.** The spread of the K predictions is a per-peak uncertainty estimate at
  no training cost. Score it with the
  [per-peak uncertainty](../findings/masster/masster_per_peak_uncertainty_2026_09_24.md) protocol.

#### Diagnostics and exclusions

- **Invariance diagnostic.** On val, the mean cosine between predictions for the same molecule and
  CE from different variants; a pure molecule-to-spectrum model scores 1.
  - It is reported per family, alongside the cross-seed disagreement and the train-subset cosine,
    which every arm logs.
  - The MASSter screen logged train cosine in only 4 of 25 runs
    ([overfit readout §1](../findings/masster/masster_overfit_readout_2026_09_22.md)).
- **Not planned.**
  - Spectrum-side noise, peak dropout and CE jitter: no repository evidence, and target cleaning
    was null on MASSter.
  - Extra spectral data such as NIST: it contains MSGYM test molecules, so it leaks on this
    benchmark.
- **Literature** (cited from memory, not re-read for this plan): SMILES enumeration as
  augmentation (Bjerrum 2017, arXiv:1703.07076); restricted against unrestricted randomized SMILES
  (Arús-Pous et al. 2019, *J. Cheminform.* 11:71); many-fold augmentation with test-time averaging
  in transformers (Tetko et al. 2020, *Nat. Commun.* 11:5575); augmented SMILES in pretraining
  (Chemformer, Irwin et al. 2022). None of them predicts MS/MS.

### Stage 0: model-free and inference-only checks (needs `transformers` and the weights)

- **0a. Tokenization.** Done; see the table above. No `<unk>` is possible and nothing is truncated.
- **0c. Embedding invariance.** At layer 6 (D2) and layer 12, compare the cosine between
  mean-pooled frozen embeddings of the canonical and randomized SMILES of one molecule with the cosine between a molecule and its
  nearest different train molecule. If randomized strings of one molecule land farther apart than
  distinct neighbours, the frozen encoder is bound to the string, and fine-tuning or augmentation
  has to supply the invariance.
- **0d. kNN spectrum baseline.** Predict each val spectrum by the spectrum of the nearest train
  molecule, at the closest ACE on the same instrument. Nearest is by frozen ChemBERTa-3 embedding
  cosine for one baseline and by Morgan Tanimoto for the other. Score `cos_sim_0.01`. This shows
  whether the pretrained neighbourhoods carry spectral meaning before any training.
  - **Per layer.** The ChemBERTa-3 baseline is scored on the pooled output of every layer, 1 to 12.
    One forward pass gives them all. The curve shows what the D2 cut at layer 6 gives up in frozen
    features. It informs D12 and does not by itself change D2: frozen neighbourhoods need not
    predict end-to-end fine-tuning.
- **Gate 0.** No stop rule: a weak 0d result deprioritizes the frozen probe, not fine-tuning.

### Stage 1: matched encoder comparison, no augmentation (2 seeds per arm)

| Arm | Encoder | Notes |
| --- | --- | --- |
| E1 | ChemBERTa-3, first 6 layers (D2), end to end (D8) | D1-D8. This is the candidate model; every later stage builds on it. |
| E2 | ChemBERTa-3, first 6 layers, frozen | **Control only**, never a candidate. Same head. E1 − E2 is what gradients into the encoder add. |
| E3 | The same 6-layer RoBERTa, random init, end to end | Encoder LR chosen from {1e-4, 3e-4} on val with one seed, then 2 seeds. Isolates pretraining. |
| E4 | NEIMS (Morgan + RDKit + MACCS fingerprint MLP) | New unmerged `ms_gym_benchmark_1_5` config with `qet`. Its fingerprints come from the same molecule as E1's SMILES. |
| E5 (optional) | MassFormer (pretrained Graphormer) | A pretrained graph encoder with the same head |

- **Read.** Val `cos_sim_0.01` best and the last-10-epoch tail; the train-subset cosine; seed
  spread from each pair.
- **Gate 1.** Both must hold:
  - E1 − E3 exceeds the seed spread, so pretraining helps;
  - E1 ≥ E4, so a SMILES language model at least matches fingerprints.
- **Stop rule.** If E1 ≤ E4 and E1 ≈ E3, record `CLOSED NULL` and do not run Stages 2-3. A domain
  shift from ZINC to MSGYM would remain a possible explanation, which the plan no longer tests.

### Stage 2: augmentation ladder on the E1 recipe (2 seeds per arm, matched steps)

Each step builds on the previous step's winner, W.

1. **Traversal.**
   - T0 = E1 (no augmentation);
   - T1: unrestricted random, `p_trav` 0.5;
   - T2: unrestricted random, `p_trav` 1;
   - T3: restricted random, `p_trav` 1;
   - T4: rooted canonical, `p_trav` 1.
2. **Chemical forms, on W.**
   - C1: W + `p_kek` 0.25;
   - C2: W + `p_taut` 0.25;
   - C3: W + both, run only if C1 and C2 each pass.
3. **Views, on W.** V2 and V4: 2 and 4 views per sample with the consistency term.
4. **Depth.** D12: the frozen configuration with all 12 pretrained layers, at the same recipe. It
   is the only arm allowed to change depth, and it runs last, so every other comparison stays at
   6 layers. It passes if it beats W by more than the seed spread, and it is then weighed against
   its measured cost per epoch.

- **Evaluation.** Score every arm canonical-only and with TTA at K = 4, 16 and 64, then run the
  invariance diagnostic.
- **Gate 2, at every step.** The new arm beats that step's baseline on canonical-only val by more
  than the seed spread. TTA gains are reported separately, against a 2-seed ensemble of the step
  baseline.
- **Budget.** At most 10 arms × 2 seeds, about 20 training runs. The λ choice adds at least two
  single-seed runs. Stage 1 adds 8 runs, 2 more for E3's LR choice, and 2 for the optional E5.
  Epoch time is unmeasured.

### Test read (after Stage 2 is frozen)

- **Scope.** One export each of the chosen configuration and E4, with
  `scripts/eval/run_save_inference_eval.py --eval_split test`.
- **Comparison.** Use `scripts/eval/compare_msgym_predictions.py` against `ll04i00m` and GLACIER,
  with molecule-bootstrap CIs, split by instrument and CE bin as in the GLACIER finding.
- **Audit.** Follow the
  [MassSpecGym in-the-wild audit](../notes/massspecgym_in_the_wild_audit.md).

### Stage 3 (conditional on Gate 1; its own plan)

Move the representation to an exact-mass head, by one of two routes. Both stay end to end (D8):
the encoder runs inside the model's forward pass on each step's strings, and the exact-mass loss
back-propagates into its embeddings.

- **(B) Subformula-candidate head.** Score the precursor's subformulas, with H shifts. Each formula
  embedding cross-attends to the token states, conditioned on CE, so gradients reach the encoder
  through every candidate.
- **(C) Per-atom injection into MASSter.** ChemBERTa-3 token states become per-atom input features
  of MASSter's molecule GNN.
  - **Atom to token.** RDKit's `_smilesAtomOutputOrder` gives the atom written at each atom
    position in the string. The fast tokenizer's character offsets give the token covering each
    atom's characters. The atom feature is an index gather from the token states, which is
    differentiable. BPE tokens such as `ccc` span several atoms, which then share one state and
    split its gradient.
  - **Atom to graph node.** MASSter's graph comes from `mol_df`; the raw SMILES form (deferred
    D1) differs from it on 8.8% of molecules, mostly as tautomers. Map atoms with `GetSubstructMatch` on the heavy-atom skeleton,
    ignoring bond orders, charges and H. Tautomer and Kekulé variants keep the same heavy atoms,
    so the mapping survives augmentation.
  - **Memory.** MASSter is activation-bound on the 40 GB slice. Unbucketed, at a 512-spectrum
    microbatch and about 93 padded tokens, the 6-layer encoder adds about 18 GB of fp32
    activations, and the 12-layer one about 35 GB (estimates). Plan on length bucketing and HF gradient checkpointing, and measure peak memory in
    a smoke run before sizing.
  - **Optimizer.** The encoder group takes the D7 encoder LR, and MASSter's groups keep their own
    settings.
  - It tests whether a pretrained prior, trained by MASSter's own loss, helps the generalization
    failure that the [memorization route plan](masster/masster_memorization_route_plan.md) is
    working on.

### Later option: more molecules (not requested)

The user's "more SMILES" meant augmentation (2026-09-26). This section is kept as measured
context and is not scheduled. Local sources:

| Source | Molecules | Labels | Leakage handling |
| --- | ---: | --- | --- |
| MassSpecGym 1.5 train fold, non-challenge rows | 10,425 new (12,321 → 22,746) | Real: 36,757 `[M+H]+` spectra on new molecules; CE missing on 98.6% | Same MCES folds |
| Retrieval candidates of train-fold queries only | 5,620,205 (mass) / 4,404,514 (formula) unique SMILES | None | Drop val/test-query sets; about 0.05% are val/test molecules, so filter by InChIKey |
| DeepMet/HMDB table (`data/other/mas_250905_filtered/`) | 318,971 | None | InChIKey filter |
| Enveda-180 | 183,476 | Real timsTOF spectra | The user chose standalone Enveda training on 2026-09-26 |

The routes would be three:

- real rows, which change the training protocol behind the 0.5713/0.5455 numbers;
- MLM on the unlabeled pool, which keeps the protocol;
- MASSter-labeled pool molecules (distillation), which inherit the teacher's errors.

## Review (2026-09-26)

The plan was checked against the code, `spec_df`/`mol_df`, the raw CSV and the checkpoint's
Hugging Face files. The scripts are in the session scratchpad and are not kept.

- **Reproduced.**
  - Spectrum counts, adducts and instruments.
  - The stereo share: 31.0% of 18,110 unique raw SMILES.
  - 941 InChIKeys with more than one raw SMILES.
  - The 569 `split_smiles` failures, all on `\`.
  - The `randomize_smiles` bug: `np_temp_seed` yields `None`, despite the comment in
    `randomize_smiles`.
  - Deterministic `MolToRandomSmilesVect`.
  - The model config: 12 layers, 768 hidden, 12 heads, 3,072 FFN, 512 positions.
  - The tokenizer: 7,924 tokens and 7,663 merges, which is 256 bytes plus 5 specials; no `@`
    merge; `c c` #0, `cc c` #3, `+ ]` #5, `- ]` #19.
  - The NEIMS MSGYM recipe: LR 4.006e-4, sqrt, 1,500 / 0.01, bidirectional.
  - The untransform before scoring.
  - The reference scores 0.5713, 0.5455 and 0.5754.
  - `transformers` and `tokenizers` absent.
  - The tautomer and Kekulé rates, on an independent 1,500-molecule sample: 73.9% (p50 3, p90 24)
    and 62.9% (p50 2, p90 4).
- **Corrected in place.**
  - The batch's SMILES are `mol_df`'s, not raw.
  - The raw form differs on 8.8% of molecules, not 4.6%.
  - Molecule counts after processing.
  - `MassFormerPL` has no per-group LR or cosine.
  - The `qet` justification.
  - The sampler's "canonical tautomer" contradicted D1.
  - E4's fingerprints must come from D1.
  - The padded length is about 90 tokens, not 50.
  - V ≥ 2 memory and matching.
  - The run budget.
- **Outside this plan.** Every MSGYM model built from `mol_df` trains on the tautomer-rewritten
  structure for 8.8% of molecules. That includes MASSter and its DAGs. Half of those rewrites
  change the first InChIKey block, and some, like the prenyl double-bond shift, look chemically
  wrong. Formulas are unchanged, so the precursor mass is too; bond structure, and therefore the
  candidate fragments, can change. This is not measured here.

## Open questions for the user

1. Answered 2026-09-27: the six runs train on Fir. The generated jobs install the Hugging Face
   packages (all in the Fir wheelhouse); the checkpoint directory still has to be copied. The install question was also
   answered that day: installed in FRAGNNET-GPU.

## Next concrete check

Fir setup: install the packages in the job venv, copy `data/pretrained/ChemBERTa-100M-MLM`,
then a one-epoch E1 smoke job for Fir epoch time before the six 100-epoch runs.
