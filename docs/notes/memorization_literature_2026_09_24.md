# Addressing memorization: literature mapped onto MASSter

> **Status:** NOTE · **Updated:** 2026-09-24
>
> A literature search for the
> [memorization-route plan](../plans/masster/masster_memorization_route_plan.md). Each
> method family is mapped onto the mechanism the
> route map (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_route_map_2026_09_24.md`) measured:
>
> - the encoder's global attention writes a per-(molecule, atom) key;
> - the boundary block reads it;
> - each molecule's key recurs across about 12 spectra per epoch.
>
> The note lists what each method would change in MASSter. None of these methods has been
> run here except weight EMA. §1-7 cover graph-model and optimizer work. §8 (second pass) covers
> general memorization research: repetition, goldfish-style loss masking, noise injection, label
> smoothing, resets, grokking, distillation, invariance and activity cliffs. §9 (third pass, run
> through the Consensus paper database) adds attention-specific dropout and bottlenecks, the
> learning-rate schedule, gradient-agreement filtering, shard averaging, per-molecule memorization
> scores, and evidence that demotes periodic resets.

**Follow-up, 2026-09-27:** the [further literature review](masster_generalization_methods_2026_09_27.md)
revises this historical ranking after the route-removal experiments. It develops peak-level
R-Drop and late SAM into bounded tests, adds spectral decoupling and representation-learning
options, and reopens same-data distillation as uncertain rather than categorically ineffective.
The run-status statements above describe 2026-09-24, not the current experiment state.

## 1. Give memorization a dedicated, removable home

This is our `m1` sink.

- **MemSinks** (Ghosal, Maini, Raghunathan, 2025):
  [arXiv 2507.09937](https://arxiv.org/abs/2507.09937).
  - A sequence identifier activates a unique set of memorization neurons on every repetition.
    Memorized content lands there and can be removed without losing general ability.
  - Shown at billion-parameter scale on real data.
  - Parallel for us: the molecule ID is the identifier, and each molecule's ~12 CE spectra are
    the repetitions.
- **Example-tied dropout** (Maini et al., 2023, "Can Neural Network Memorization Be Localized?"):
  - Memorization sits in a few neurons spread across layers, not in the last layers.
  - Tying each example to a preset set of neurons, then dropping them at test time, cut
    memorized-example accuracy from 100% to 3% and narrowed the generalization gap.
- **Random effects for a high-cardinality ID with repeated measures (LMMNN)** (Simchoni &
  Rosset, NeurIPS 2021):
  - Treats the ID as a random effect under the linear-mixed-model likelihood inside a network.
  - Beat entity embeddings on real repeated-measures data.
  - This is the closest statistical match to our setting: one molecule measured under several
    conditions (NCE). H-likelihood variants (Lee et al., 2023) estimate the random-effect
    variance instead of fixing it.

Refinements to `m1` these suggest:

- estimate the sink's variance (its shrinkage) instead of using plain weight decay;
- apply sink dropout on part of the batch, as MemSinks mixes shared and specific units, so the
  shared path is also trained without the sink;
- try a per-(molecule, candidate) sparse residual, in the style of the sparse per-sample term in
  SOP (Liu et al., 2022).

## 2. Limit what global attention can carry

This targets `r2`, and global-message dropout.

- **Over-globalizing** (Xing et al., ICML 2024,
  [arXiv 2405.01102](https://arxiv.org/abs/2405.01102)):
  - Global attention in graph transformers over-weights distant nodes against the local
    neighbourhood, which holds most of the useful information.
  - Their fix, CoBFormer, uses bi-level intra- and inter-cluster attention with collaborative
    training, and improves generalization.
- **Capacity-controlled global attention (SigGate-GT)** (Guo et al., 2026,
  [arXiv 2604.17324](https://arxiv.org/abs/2604.17324)):
  - A learned, per-head, input-conditioned sigmoid gate on the attention output, so a node can
    "attend to nothing".
  - Significant gains over GraphGPS on ZINC, ogbg-molhiv and ogbg-molpcba, at under 3% overhead.

Candidate changes:

- `r2` removes global attention outright.
- A **gated global message** (a sigmoid gate on `gsa_out`, optionally with a sparsity penalty)
  or **global-message dropout** keeps the route but makes it cost something to use.

## 3. Bottleneck the representation

- **Deep VIB** (Alemi et al., 2016) generalizes better than other regularizers of the time.
- **Graph Information Bottleneck** (Wu et al., NeurIPS 2020;
  [pdf](https://www-cs.stanford.edu/~jure/pubs/gib-neurips20.pdf)) and its OOD variants (e.g.
  [IS-GIB, arXiv 2306.15902](https://arxiv.org/abs/2306.15902)) constrain what node features and
  structure pass into the representation.

Candidate: our `v1` token bottleneck. Given the route map, a bottleneck on the **global
message** is the more targeted version.

## 4. Suppress per-example gradient directions

These are route-agnostic.

- **Coherent gradients** (Chatterjee, 2020): the useful signal is the gradient directions many
  examples share, and memorization is in the weak, idiosyncratic ones.
- **M3/RM3** (Zieliński et al., 2020): a coordinate-wise median of mini-batch gradients.
  - Greatly reduced overfitting and memorization at ImageNet scale.
  - RM3 reuses the previous two steps' gradients, so it needs no extra forward or backward pass.
- **AND-mask** (Parascandolo et al., 2020): averaging gradients favours memorized "patchwork"
  solutions; keeping only the components whose signs agree across examples prevents
  memorization. Variants are SAND-mask (Shahtalebi et al., 2021) and CGLearn (Chowdhury et al.,
  2024).

Candidate: `c1` using molecule-disjoint micro-batches as the environments. RM3 is the cheap
version, and matters here because MASSter's per-batch overhead is sync-bound.

## 5. Flatten the minimum and average the weights

- **SWA** (Izmailov et al., 2018) and **SWAD** (Cha et al., 2021) average weights along the
  trajectory; SWAD densely and with awareness of overfitting.
- **SAM** (Foret et al., 2021) minimizes loss sharpness and is robust to label noise. It costs
  about 2x compute; ESAM brings that down to about 1.4x.

Status here: weight EMA already passed on both lineages (+0.0094 unmerged, +0.0097 merged,
ranking-transfer tail read (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_ranking_transfer_plan.md`)).
SAM is an optional later arm.

## 6. Augment inputs, with caution

- **DropEdge** lowers the effective receptive field and can make models overfit to short-range
  artefacts
  ([OpenReview, effects of random edge-dropping](https://openreview.net/forum?id=ZZwP9zljas)).
- **Adversarial node-feature perturbation** (FLAG-style) is used for molecular graphs because it
  leaves the chemistry intact
  ([example](https://www.sciencedirect.com/science/article/pii/S0957417425001344)).

Low priority for us: the route map puts the key in global attention, not in topology or atom
features. `a_charge` precision is null.

## 7. Framing and evaluation

- **Shortcut learning** (Geirhos et al., 2020): decision rules that pass the standard benchmark
  and fail under harder test conditions. The remedy is evaluation on genuinely harder splits.
- **MassSpecGym** ([arXiv 2410.23326](https://arxiv.org/abs/2410.23326)) splits by MCES
  distance (at least 10 edits between splits) to test generalization to novel molecules. Our
  acceptance already bins val by nearest-train Tanimoto; MCES-distance bins would be the
  stricter version.

## 8. General literature (second pass, 2026-09-24)

Sections 1-7 lean on graph-model work. This pass covers memorization research in general. Each
item says what it would mean for MASSter.

### Why memorization happens, and where it sits

- **Networks fit random labels** (Zhang et al., 2017), and explicit regularization does not
  qualitatively change that. This is consistent with our dropout and weight-decay arms all
  landing inside noise: generic capacity limits do not remove a lookup.
- **Shared patterns first, memorization later** (Arpit et al., 2017). Tuned dropout can slow the
  fitting of noise without hurting real-data generalization. Our gap likewise grows steadily
  after about epoch 8.
- **Memorization can help the long tail** (Feldman & Zhang, 2020). Rare examples are best handled
  partly by memorizing them. So the aim is to stop per-molecule lookup from *replacing* transfer,
  not to remove every molecule-specific fit. That is the case for a sink over blocking.
- **Where it sits is contested.**
  - Stephenson et al. (2021) find memorization mostly in the deeper layers. Reverting only the
    final layers to an earlier epoch restored generalization.
  - Maini et al. (2023, §1) find it in a few neurons spread across layers.

  This gives a cheap diagnostic: rewind one module at a time (the intensity head, the cut-bond
  encoder, the global-attention projections) to an early-epoch checkpoint, keep everything else
  at its final weights, and read the train/val gap. It needs a run that saved early checkpoints.

### Data and loss side

- **Goldfish loss** (Hans et al., 2024). A fixed pseudo-random subset of targets per training
  example is excluded from the loss. Those targets are never memorized, which cut extractable
  memorization with little benchmark cost.
  - MASSter version: hash (molecule, target peak) to exclude, say, 25% of each molecule's
    peaks in every exposure. No lookup can then reproduce that molecule's full allocation, and
    the excluded peaks must be predicted by transfer.
  - The loss must drop the masked candidates from **both** the target and the predicted
    distribution, renormalizing over the rest. Otherwise cross-entropy teaches that the masked
    peak is absent.
  - Cheap to build; a new arm candidate.
- **Repetition drives memorization superlinearly.**
  - A sequence seen 10 times is regenerated about 1000 times more often (Kandpal et al., 2022).
  - Deduplication cut memorized output tenfold (Lee et al., 2021).
  - Repeating 0.1% of the data 100 times degraded an 800M-parameter model to the level of one
    half its size (Hernandez et al., 2022).

  MASSter analogue: each molecule recurs about 12 times per epoch across NCE. Our exposure
  check found train cosine flat from 4 to 24+ spectra per molecule, which argues against
  exposure being the lever here. Sampling a few CE spectra per molecule per epoch is still a
  cheap test.
- **Per-example gradient clipping (DP-SGD)** bounds how much any one example can be memorized
  (Yang et al., 2022). It hurts rare, informative examples (Zhang et al., 2026), and
  per-molecule gradients are costly. Not recommended.

### Regularizers with a memorization rationale

- **Noise injection is Tikhonov regularization** (Bishop, 1995). Gaussian noise on activations
  penalizes high-frequency functions (Camuto et al., 2020). A per-molecule lookup is exactly such
  a function of the embedding. This is the principled form of **noise on the global message**,
  the continuous sibling of global-message dropout.
- **Label smoothing** (Müller et al., 2019; Lukasik et al., 2020) prevents overconfidence and acts
  like loss correction under noise. MASSter is measurably over-sharp: entropy 0.12-0.15 nats
  below the target, and `P^0.75` adds +0.007. Mixing each target with a small share of a smooth
  prior (uniform over candidates, or the formula prior) is a cheap arm aimed at both the
  sharpness and the memorization.
- **mixup** (Zhang et al., 2018) reduces memorization of corrupt labels. Mixing across molecules
  has no natural form for candidate-set spectra, so low priority.
- **Consistency regularization (R-Drop)** (Liang et al., 2021) makes two stochastic passes agree.
  The natural MASSter form is agreement across two candidate-subset views, already listed in
  the overfit read-out §7.2. It costs about 2x the forward pass.

### Training procedure

- **Early-learning regularization (ELR)** (Liu et al., 2020) steers training toward the model's
  own early predictions, which fit shared patterns before memorization. The MASSter version is
  an EMA of per-spectrum predictions from early epochs as a soft target. Storage is heavy with
  variable candidate sets. Optional.
- **Layer reinitialization and later-layer forgetting** (Alabdulmohsin et al., 2021; Sarfi et
  al., 2023). Periodically reinitializing some layers favours general rules over memorization and
  finds flatter minima. MASSter version: periodically reset the key reader (the cut-bond
  encoder) or the global-attention projections. Cheap as a callback; the evidence is mostly
  small-data image classification.
- **Self-distillation** (born-again networks; Furlanello et al., 2018): a student of the same size
  beats its teacher.
  - Caveat from our variance-novelty finding: members fit training molecules to 0.96, so
    teacher outputs on train add little.
  - The version with something to transfer uses **out-of-fold teachers**: K models trained on
    molecule-disjoint folds label the held-out fold's training molecules.
- **Weight decay and grokking** (Power et al., 2022; Omnigrok, Liu et al., 2022; Varma et al.,
  2023).
  - Memorizing and generalizing solutions coexist. Weight decay favours the generalizing one,
    which reaches the same logits with a smaller weight norm.
  - Omnigrok induced grokking on molecular tasks by controlling weight norm.
  - Our 630-epoch horizon probe at wd 0.01 showed no delayed generalization. A targeted larger
    weight decay (encoder global attention plus head) over a longer horizon is the untested
    version.
- **Co-teaching** (Han et al., 2018) targets noisy labels. Our targets are not mostly noise
  (replicate noise is about 10% of squared error). Not recommended.

### Invariance

- **IRM** (Arjovsky et al., 2019) and **group DRO** (Sagawa et al., 2019). Group DRO works only
  with stronger-than-usual regularization. Environments could be scaffold clusters or novelty
  bins. The practical evidence is mixed. AND-mask (§4) is the cheaper form of the same idea.

### Molecular ML specifically

- **Activity cliffs** (van Tilborg et al., 2022, MoleculeACE; Tamura et al., 2023). All models
  struggle on near-identical molecules with different properties. Descriptor-based methods beat
  deep models there. Activity-cliff prediction accuracy tracked memorization of shared compounds,
  not model complexity.
- MASSter angle: the variance-novelty table shows val molecules above 0.9 Tanimoto scoring
  *below* the 0.8-0.9 bin (0.823 against 0.836). That is a possible copy-the-neighbour signature.
  An activity-cliff-style metric (near-identical molecule pairs with dissimilar spectra) would
  measure it directly.

## 9. Consensus pass (third pass, 2026-09-24)

This pass ran 18 queries through the Consensus paper database (10 results each). It adds evidence
for items already ranked, fixes one attribution, and adds four candidates and three diagnostics.
Links go to the Consensus paper pages.

**Correction.** SigGate-GT's first author is Guo, not Liu. §2 is fixed.

### Restricting global attention (plan decision B)

- **Dropout designed for attention.** Structured dropout on attention beats unit dropout.
  - DropHead drops whole heads on a scheduled rate
    ([Zhou et al., 2020](https://consensus.app/papers/details/38dd511742b75130affece3e2d37e374/?utm_source=claude_desktop)).
  - DropKey drops keys before the softmax, with a rate that decreases through the layers
    ([Li et al., CVPR 2023](https://consensus.app/papers/details/f4c4629ed57356d095ba2510f4bfd202/?utm_source=claude_desktop)).
  - Token-level masking of attention connections beat both on GLUE
    ([Wu et al., 2023](https://consensus.app/papers/details/0a60925464795e5d85033e154c66b2ad/?utm_source=claude_desktop)).

  MASSter version: DropKey on the two GSA layers, so each atom's global message is built from
  a random subset of the molecule's atoms. This weakens molecule identity in the message but does
  not remove it. Global-message dropout (substituting the batch mean) removes it for the dropped
  molecules. DropKey is a fourth option for decision B.
- **A bottleneck built for attention.**
  - The nonparametric VIB (NVIB) limits both how many vectors attention can reach and how much
    information each carries
    ([Henderson et al., 2022](https://consensus.app/papers/details/29b44fee5840533c967aac99527f900e/?utm_source=claude_desktop)).
  - Reading a pretrained transformer's attention as NVIB and changing only its initialization
    improved out-of-domain generalization with no further training
    ([Fehr et al., 2023](https://consensus.app/papers/details/7caf94a7361957f0b20fa1ea4b0fcfb6/?utm_source=claude_desktop)).
  - VIB during low-resource fine-tuning suppressed dataset-bias features and improved 13 of 15
    out-of-domain benchmarks
    ([Mahabadi et al., 2021](https://consensus.app/papers/details/5a6cde138dc65327bed758bf65244753/?utm_source=claude_desktop)).

  This is the principled form of ranked item 7, a bottleneck on the global message.
- **Noise on the global message.** Camuto et al. (2020) is confirmed. Three additions:
  - several noise samples per example give a tighter bound on the true objective
    ([Noh et al., 2017](https://consensus.app/papers/details/c58760a84d89515380bad8e12d0c136a/?utm_source=claude_desktop));
  - adaptive Gaussian noise (whiteout) beats dropout on small training sets
    ([Li et al., 2016](https://consensus.app/papers/details/af3ee9d897495dbd9d731daf179d269f/?utm_source=claude_desktop));
  - Ghost Noise Injection reproduces BatchNorm's small-batch noise without the train/eval
    mismatch
    ([Kosson et al., 2023](https://consensus.app/papers/details/6481da5f59b35e6b9486e6bfac4d8537/?utm_source=claude_desktop)).
    This one matters here: the MASSter encoder uses BatchNorm, and the regularization audit
    flagged BatchNorm train/eval skew.
- **Counter-evidence to keep in view.**
  - MassFormer was built on the premise that global graph-transformer attention helps spectrum
    prediction generalize
    ([Young et al., 2024](https://consensus.app/papers/details/f2bcae06c7a353f1b11cff704e8ac119/?utm_source=claude_desktop)).
    HDSE-MS makes the same claim
    ([Liu et al., 2025](https://consensus.app/papers/details/1fb72757a256531ab8eb36fdd0fb23ca/?utm_source=claude_desktop)).
  - Theory for graph transformers says attention helps generalization when training makes the
    attention map sparse around the core neighbourhood
    ([Li et al., 2024](https://consensus.app/papers/details/4df8964cf45f52c39f6da58ab964871b/?utm_source=claude_desktop)).

  The route map does not contradict these. MassFormer was pretrained and evaluated on a different
  split, and our finding is that MASSter's global message carries identity, not that global
  context is useless. `r2` is the test. If it loses val, the next step is a gated, noisy or
  DropKey global message, not dropping global attention.

### The sink (plan item 1)

- **Random-effect embeddings keep beating entity embeddings** for high-cardinality IDs:
  - GLMMNet
    ([Avanzi et al., 2023](https://consensus.app/papers/details/be8e72e359c657579568359e882ae728/?utm_source=claude_desktop));
  - MC-GMENN
    ([Tschalzev et al., 2024](https://consensus.app/papers/details/815e05bf2220599eb7e8ea9cff43f851/?utm_source=claude_desktop));
  - a tabular comparison, where models with random effects beat those without
    ([Sigrist, 2023](https://consensus.app/papers/details/d0babe3a6799536898d42fb597cbc9fb/?utm_source=claude_desktop)).

  Richman et al. describe the failure mode we see: entity embeddings of a high-cardinality ID,
  regularized by early stopping, stop very early and give a poor model
  ([2024](https://consensus.app/papers/details/c2eab26151aa549da0dda71ee448d417/?utm_source=claude_desktop)).
  MMbeddings treat the embedding as a latent random effect with a KL term
  ([Simchoni et al., 2025](https://consensus.app/papers/details/c939977d027951daa4a864ad9c00951a/?utm_source=claude_desktop)).
- **MASSter version: a variational sink.** `m1` with a Gaussian posterior per molecule and a
  KL term to N(0, σ²I), with σ learned. The KL replaces fixed weight decay with estimated
  shrinkage (refinement 1 in the plan).
- **Removing memorization after training is entangled; isolating it by design works.**
  - Localization methods find neurons that are not specific to any single memorized sequence
    ([Chang et al., 2023](https://consensus.app/papers/details/a10a04e1d0565e639570fd02cf487ce4/?utm_source=claude_desktop)).
    This is the motivation for MemSinks.
  - Memorization of minority groups sits in a small set of neurons, and removing it during
    training changed those groups' performance
    ([You et al., 2025](https://consensus.app/papers/details/b2c3deffa4385f4396956c460dab6419/?utm_source=claude_desktop)).

### Optimizer and training procedure

- **Gradient Agreement Filtering (GAF)**
  ([Chaubard et al., 2024](https://consensus.app/papers/details/2045b1bb26ad5ab0b634e93c46982c29/?utm_source=claude_desktop)).
  - Micro-batch gradients turn orthogonal or opposed late in training, which the authors tie to
    memorization.
  - Averaging only micro-batches whose gradients agree (by cosine) improved validation and
    reduced noisy-label memorization.
  - MASSter version: split each 512-spectrum batch into molecule-disjoint micro-batches and
    average only the agreeing ones. It joins RM3 and AND-mask in ranked item 6. It costs one
    backward pass per micro-batch.
- **Large initial learning rate, then decay.**
  - A small learning rate memorizes an easy-to-fit patch first and generalizes worse. A large one
    ignores the patch until the rate is annealed
    ([Li, Wei & Ma, 2019](https://consensus.app/papers/details/a3b1341dcb6154c9afda569c5a06993e/?utm_source=claude_desktop)).
  - A large initial rate suppresses memorizing noisy data, and the decay then learns complex
    patterns
    ([You et al., 2019](https://consensus.app/papers/details/1854350148d55eebbad2aa6144b284d9/?utm_source=claude_desktop)).

  A molecule-identity key is exactly such an easy-to-fit patch. Every MASSter run, the Step 1
  arms included, uses a constant `lr: 1e-3` with no warmup or decay. The one LR-decay arm with a
  result, `d3` (cosine decay with a 1000-step warmup; merged lineage; n=1), gained +0.0072
  (overfit read-out (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_overfit_readout_2026_09_22.md`)). See
  decision C in the plan.
- **Ratio of learning rate to batch size.** A higher ratio finds wider minima and generalizes
  better
  ([Jastrzebski et al., 2017](https://consensus.app/papers/details/0b56e7d01e37542d9bd407110220adf3/?utm_source=claude_desktop)).
  Not a lever here: smaller batches are sync-bound and expensive in MASSter.
- **Fisher-trace penalty**
  ([Jastrzebski et al., 2020](https://consensus.app/papers/details/767ce4dd209d5ebca07f65660eabae5b/?utm_source=claude_desktop)).
  An explicit penalty on the Fisher trace improved generalization and slowed learning of
  noisy-label examples more than of clean ones. It costs a second backward pass. Later option.
- **SAM variants.**
  - SAM prevents noise learning early in training
    ([Chen et al., 2023](https://consensus.app/papers/details/9fc7ba4b169f5c80ac814fd2bef64c94/?utm_source=claude_desktop)).
  - Fine-tuning a converged model with SAM alone gives significant gains
    ([Andriushchenko & Flammarion, 2022](https://consensus.app/papers/details/ed10bead0b6c553c8af16ad1d4f28f2c/?utm_source=claude_desktop)).
    That suggests a cheaper arm: SAM for only the last 16-32 epochs, starting from a checkpoint.
  - SAF replaces the sharpness term with a KL divergence to the model's own past outputs, at
    near-zero cost
    ([Du et al., 2022](https://consensus.app/papers/details/451a4ebec1dd51038f63738fc450e57c/?utm_source=claude_desktop)).
    It needs stored per-spectrum outputs, which variable candidate sets make heavy.
- **Resets add little once regularization is tuned.**
  - Across 15,000 models, reinitialization helped when nothing else regularized. It added little
    or nothing once weight decay, augmentation and a learning-rate schedule were tuned, except
    under label noise
    ([Zaidi et al., 2022](https://consensus.app/papers/details/1cfe7695bb2a58aa99cb4107260640ed/?utm_source=claude_desktop)).
    This demotes the periodic reset.
  - A cheap variant exists because EMA already runs: periodically reset the online weights to
    the slow average (Hare & Tortoise,
    [Lee et al., 2024](https://consensus.app/papers/details/ff6cd5a731d15a7796f410dab80ff4f0/?utm_source=claude_desktop)).
- **Excluding low-loss samples.** Dropping the lowest-loss samples from each batch reduced
  memorization in diffusion models
  ([Liu et al., 2024](https://consensus.app/papers/details/28140902b3b25d8ca1cdab3ffafe500f/?utm_source=claude_desktop)).
  In MASSter, low train loss does not separate memorized molecules from easy ones, so this
  needs a per-molecule memorization score (see Diagnostics). Not recommended alone.
- **Shard soup.**
  - Averaging the weights of models trained on disjoint data shards gave anti-memorization
    guarantees and beat a model trained on the union of the shards
    ([Biggs et al., 2024](https://consensus.app/papers/details/04b7b973df2457cc8278ae0b42873331/?utm_source=claude_desktop),
    diffusion models).
  - Weight averaging needs a shared starting point
    ([model soups, Wortsman et al., 2022](https://consensus.app/papers/details/952fef9c9a8a582d9f8d1b28278ff064/?utm_source=claude_desktop);
    [DiWA, Ramé et al., 2022](https://consensus.app/papers/details/a0aa1572ef3550c4af5f8118a9efe303/?utm_source=claude_desktop)).
  - MASSter version: branch K runs from one early checkpoint (before the gap opens, about
    epoch 8), train each on a molecule-disjoint shard, and average the weights. Each molecule's
    lookup lives in one of the K models and is diluted to 1/K. Shared chemistry is in all of them.
  - Measured bounds: a 2-checkpoint weight average gave +0.003 and a 2-run ensemble +0.020. Each
    shard model sees fewer molecules and may lose more than averaging recovers. Later option;
    route-agnostic.
- **A caveat on distillation.**
  - Students trained on a teacher's soft labels recover that teacher's memorized answers on data
    they never saw
    ([Behrens et al., 2025](https://consensus.app/papers/details/806506cb28b65c2db708e98c9cc8547a/?utm_source=claude_desktop)).
    Same-data self-distillation can therefore carry the lookup across.
  - Only out-of-fold teachers avoid this, as in PATE's teachers on disjoint partitions
    ([Papernot et al., 2016](https://consensus.app/papers/details/047273e5980a576f8768a98e8d821965/?utm_source=claude_desktop)).
  - A label-smoothed teacher distills worse (Müller et al., 2019), so do not combine ranked
    item 5 with a teacher.

### Label smoothing (ranked item 5)

- Data-dependent smoothing beats uniform smoothing:
  - structural label smoothing
    ([Li et al., 2020](https://consensus.app/papers/details/28410bbb57095c57b556a7a9131e5310/?utm_source=claude_desktop));
  - per-instance smoothing set by a teacher
    ([Maher et al., 2021](https://consensus.app/papers/details/de0bf4189fe45903ba298c3b59abc1fd/?utm_source=claude_desktop));
  - the model's own predictions as the prior
    ([Lee et al., 2022](https://consensus.app/papers/details/1ca1b8785aab59e7877809a12c784e6f/?utm_source=claude_desktop)).
- There is an optimal smoothing strength
  ([Chen et al., 2020](https://consensus.app/papers/details/a385b41ebbe2577981b0737e9576ed67/?utm_source=claude_desktop)),
  so this is a two-point sweep, not one arm.
- The formula prior over candidates is our data-dependent prior.

### When memorization starts

- Generalization and memorization run on separate timescales, and memorization starts after a
  delay that grows with training-set size
  ([Bonnaire et al., 2025](https://consensus.app/papers/details/aa232833ce275759b4cbe5ba719472a3/?utm_source=claude_desktop),
  diffusion models).
- In LLM fine-tuning, memorization rises in the first few epochs, before the validation optimum,
  and an early memorization score works as a stopping signal
  ([Slack et al., 2025](https://consensus.app/papers/details/b058d18dd06158e48be927580800ba5b/?utm_source=claude_desktop)).
- MASSter's gap grows steadily after about epoch 8. So read each arm's gap trajectory, not
  only its final gap. A run meant for layer rewinding should save checkpoints at epochs 4, 8
  and 16.

### Diagnostics

- **A per-molecule memorization score without retraining.**
  - Loss curvature around each sample, averaged over epochs, matched the Feldman-Zhang
    memorization scores from a single run and exposed duplicated images with different labels
    ([Garg et al., 2023](https://consensus.app/papers/details/d9838f525fc556719064bb2512b8da4f/?utm_source=claude_desktop)).
  - Turn-over dropout estimates influence and memorization efficiently by training each example
    through a fixed subnetwork
    ([Liu et al., 2021](https://consensus.app/papers/details/846f74fe31cd5ccdb5834f085aac089d/?utm_source=claude_desktop)).
    That is the mechanism of example-tied dropout.
  - Proposed: the norm of each molecule's `m1` sink vector is a free version of the score.
- **Near-duplicates hide memorization.** Near-duplicates lower self-influence while the samples
  stay extractable
  ([Meeus et al., 2025](https://consensus.app/papers/details/0d78558771855fd5a9e3228088045e49/?utm_source=claude_desktop)).
  This fits the bin above 0.9 Tanimoto scoring below the 0.8-0.9 bin. Read the activity-cliff pair
  metric per near-duplicate cluster.
- **Gradient coherence over training.** m-coherence is O(m) to compute and interpretable
  ([Chatterjee et al., 2020](https://consensus.app/papers/details/54cab9fcd8195fab9fd4bcfcaad94ffd/?utm_source=claude_desktop)).
  Logged each epoch on a fixed molecule sample, it would show when updates stop being shared.

### Evaluation

- **Overall val can hide a win on novel molecules.**
  - Scaffold splits behave like random splits, and similarity-cluster splits are the hardest.
  - In-distribution performance predicts cluster-split performance poorly: r ≈ 0.4, against
    ≈ 0.9 for scaffold splits
    ([Fooladi et al., 2025](https://consensus.app/papers/details/763bfc0daa3e5d1fb17cdc4150b9a306/?utm_source=claude_desktop)).
  - Plan acceptance criterion 4 already reads the Tanimoto bins at or below 0.6. The implication
    is that an arm that misses criterion 1 but wins those bins clearly deserves a replicate, not
    rejection.
- **MS/MS benchmarks.**
  - Leakage in naive splits inflated de novo results. On MassSpecGym the best models reach 4.1%
    top-10
    ([Schneider et al., 2026](https://consensus.app/papers/details/5d2358fd52c75e21b5c5315afe53df3a/?utm_source=claude_desktop)).
  - 17 of 26 MassSpecGym papers had evaluation problems: leakage, shortcut learning or metric
    bugs
    ([Liu et al., 2026](https://consensus.app/papers/details/758a714449125bb4ba92037a73d49938/?utm_source=claude_desktop)).
  - Collision energy drives errors in MS/MS similarity models
    ([Strobel et al., 2025](https://consensus.app/papers/details/e41c23abe2c45d10bede61f331f7b5dd/?utm_source=claude_desktop)).
- **Activity cliffs.** An activity-cliff-aware triplet loss reshaped GNN latent spaces and
  improved accuracy across 52 datasets
  ([Cui et al., 2026](https://consensus.app/papers/details/0afc5a69e14a552994d62e28b3759a08/?utm_source=claude_desktop)).
  It is the training-side counterpart of the pair metric. Not ranked.

## Ranking for MASSter, given our measurements

Updated after §9.

1. **Weight EMA** as the baseline for every arm. It passed here; the averaging literature
   (SWA, SWAD) explains why. **Decide the learning-rate schedule too**: every run so far is
   constant-rate, and the large-initial-rate literature says the schedule changes what is
   memorized first (plan decision C).
2. **Restrict global attention**: `r2`, then a noisy, dropped-out (batch mean or DropKey) or gated
   version of the global message. Sources: over-globalizing, SigGate-GT, Gaussian noise as
   Tikhonov regularization, DropKey and DropHead. This targets the route the map found. MassFormer's
   pro-global-attention result is the counter-evidence `r2` tests.
3. **Memorization sink** `m1` (MemSinks, example-tied dropout, LMMNN). Refinements: sink dropout,
   and a variational sink with learned shrinkage (GLMMNet, MC-GMENN, MMbeddings).
4. **Goldfish-style masked peaks**, with renormalized loss. Cheap, and it directly denies the
   lookup a full allocation.
5. **Label smoothing toward the formula prior**, as a two-point sweep. Cheap; it targets the
   measured over-sharpness too. Keep it out of any distillation teacher.
6. **RM3, AND-mask or GAF** over molecule-disjoint micro-batches, as the route-agnostic optimizer
   arm.
7. **A bottleneck on the global message** (VIB, or NVIB for attention), or on the token.
8. **Later options:**
   - SAM over the last 16-32 epochs only;
   - a shard soup from an early checkpoint;
   - out-of-fold distillation;
   - targeted larger weight decay over a longer horizon;
   - a Fisher-trace penalty;
   - periodic reset of the key reader, demoted by Zaidi et al. (2022); the EMA-reset variant is
     the cheap form.

Diagnostics to add:

- layer rewinding (which module holds the memorization), with checkpoints saved at epochs 4, 8
  and 16;
- an activity-cliff pair metric, read per near-duplicate cluster;
- a per-molecule memorization score (loss curvature, or the `m1` sink norm);
- gradient coherence per epoch.

Not recommended: generic dropout (`h1` was rejected at −0.0067), DropEdge, more atom-feature
precision work, DP-SGD, co-teaching, mixup across molecules, low-loss sample exclusion without a
memorization score, same-data self-distillation (soft labels carry memorized answers), and
smaller batches.
