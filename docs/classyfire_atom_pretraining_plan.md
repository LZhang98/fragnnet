# ClassyFire atom-level pretraining for FraGNNet

**Status:** research plan

**Goal:** pretrain a transferable molecular encoder on large, labelled molecular data, then reuse
that encoder inside FraGNNet without requiring ClassyFire annotations at inference time.

## Why this can help FraGNNet

FraGNNet's molecular GNN produces atom embeddings that are reused by fragment prediction:

- `cc` pools the embeddings of atoms contained in each fragment;
- `boundary_pair` and cut-bond features use embeddings around fragmentation boundaries;
- fragment-node and intensity heads inherit those molecular representations.

Atom-level functional-group and motif supervision is therefore better aligned with fragmentation
than molecule-level taxonomy alone. It can teach the encoder that a carbonyl carbon, phenolic
oxygen, aromatic nitrogen, ionizable site, or attachment atom has a transferable chemical role
before spectrum training starts.

This addresses representation generalization. It does not replace the spectrum and ranking losses,
which still determine how FraGNNet assigns intensities.

## Pretraining model

Use the same molecular backbone as the target FraGNNet model:

- identical atom and bond input features;
- identical GNN type, hidden size, layer count, and normalization;
- parameter names compatible with FraGNNet's `mol_embedder`;
- downstream-only CE, instrument, adduct, fragment, and intensity modules excluded from pretraining.

The pretraining model adds temporary prediction heads around this shared encoder:

```text
molecular GNN encoder
├── molecule-level ClassyFire hierarchy head
├── atom-level multi-label motif head
├── optional bond/cut-role head
└── optional masked-subgraph or cut-context JEPA predictor
```

Discard all four prediction heads after pretraining. Only the compatible molecular encoder weights
are transferred into FraGNNet.

## Labels and objectives

### Molecule-level hierarchy

Predict every available level of the ClassyFire taxonomy, such as kingdom, superclass, class,
subclass, and direct parents. Preserve the hierarchy instead of flattening it into one class.

### Atom-level labels

Use multi-label targets because functional groups and motifs overlap. The most useful labels are:

- functional-group and motif membership;
- ring and conjugated-system membership;
- ionizable or charge-bearing roles;
- motif attachment and boundary atoms;
- local roles within heteroatom-containing groups.

Do not force one exclusive label per atom.

### Bond/cut labels

When available, predict whether a bond connects motifs, lies inside a motif, is a ring bond, or is a
plausible motif boundary. These targets are more directly aligned with fragmentation than atom
classification alone.

### Optional JEPA objectives

Use chemically connected masks rather than independent random atoms:

- masked-subgraph JEPA: predict the target embedding of a hidden connected motif;
- cut-context JEPA: predict the removed-fragment embedding from the retained context and cut bond.

A candidate joint objective is:

```text
L_pretrain =
    L_molecule_hierarchy
  + lambda_atom * L_atom_multilabel
  + lambda_bond * L_bond_cut
  + lambda_jepa * L_jepa
```

Start with supervised molecule and atom objectives. Add JEPA only after those baselines transfer,
so its contribution is measurable.

## Transfer into FraGNNet

The recommended first integration is weight initialization:

1. Save a checkpoint whose shared encoder parameters use FraGNNet-compatible names and shapes.
2. Initialize FraGNNet's `mol_embedder` from that checkpoint.
3. Initialize fragment, formula, ion-channel, and intensity-ranking modules normally.
4. Fine-tune the transferred encoder at a lower learning rate than the new downstream modules.
5. Keep the normal spectrum objective and the selected sibling/listwise ranking objective.

The existing FraGNNet fine-tuning path supports compatible partial checkpoint loading and
differential learning rates:

```yaml
finetune: true
finetune_pretrained_ckpt: saved_ckpts/classyfire_atom_pretrain.ckpt
finetune_strict_load: false
finetune_freeze_all_shared: false
finetune_shared_all: false
finetune_shared_components:
  - mol_embedder
finetune_weights_lr_ratio: 0.1
```

Only add `mol_pool` to `finetune_shared_components` when the pretraining checkpoint contains a
shape-compatible pooling module that FraGNNet actually uses.

The current implementation supports either freezing loaded weights for the whole run or training
them with a reduced learning rate. It does not automatically freeze for several epochs and then
unfreeze. That schedule requires two training phases or a new unfreezing callback and should not be
part of the first transfer test.

## Data construction

Before scaling to 100 million molecules:

- canonicalize and deduplicate structures;
- preserve atom mapping between structures and atom labels;
- retain overlapping atom labels;
- measure label frequencies and use class-balanced sampling or weighting;
- limit domination by large near-duplicate analogue series;
- fingerprint the molecule set, label generator, taxonomy version, and atom-mapping procedure.

Remove all NIST20 validation and test InChIKeys before pretraining. Also report a stricter
scaffold-overlap audit because exact-molecule removal alone does not rule out analogue leakage.

ClassyFire labels are derived from structure. They provide useful chemical inductive bias, but they
are not independent experimental measurements. Downstream FraGNNet validation is the deciding
metric, not pretraining accuracy.

## Experiment sequence

Run a 1-million-molecule transfer pilot before using all 100 million:

| Arm | Pretraining target | Purpose |
|---|---|---|
| scratch | none | downstream baseline |
| molecule | ClassyFire hierarchy | value of global taxonomy |
| atom | atom-level multi-label motifs | value of local supervision |
| molecule + atom | both supervised targets | primary supervised candidate |
| joint | molecule + atom + cut/JEPA | task-aligned extension |

For each pretrained arm, compare:

- frozen molecular encoder for the full downstream run;
- trainable encoder with `finetune_weights_lr_ratio: 0.1`;
- full-rate fine-tuning only if reduced-rate fine-tuning under-adapts.

If the winning 1-million arm transfers, repeat it at 10 million molecules. Use 100 million only if
the 1M-to-10M scaling curve remains positive.

## Evaluation and gate

Keep the downstream FraGNNet architecture, data split, DAG, loss, sampler, and seed list fixed.
Report:

- validation `cos_hun_inst_tol`;
- sibling and global node-ranking diagnostics;
- matched-peak Spearman and base-peak accuracy;
- train-validation gap;
- formula-retrieval metrics;
- results by ClassyFire class, adduct, instrument, and collision energy.

Promote pretraining only when its three-seed downstream gain exceeds the matching scratch baseline
without retrieval regression. High ClassyFire or atom-label accuracy alone is not evidence of a
better FraGNNet model.

## Optional second-stage distillation

If transferred representations help initially but are destroyed during spectrum fine-tuning, keep
the pretrained encoder frozen as a teacher and add an atom-representation consistency loss:

```text
L =
    L_spectrum
  + lambda_rank * L_ranking
  + lambda_distill * distance(h_atoms, h_teacher_atoms)
```

This is a fallback after the initialization experiment, not part of v1.
