# MASSter candidate features as one declared list

> **Status:** IMPLEMENTED 2026-09-22, untrained · **Updated:** 2026-09-22
>
> Replace the five per-feature boolean config keys that govern MASSter's candidate token with
> a single ordered list of feature names, backed by one spec table. Adding a candidate
> descriptor is now one spec entry plus one config-list entry, not edits in nine places.
> No checkpoints exist for any of these keys, so this carried **no compatibility burden** and
> no migration shim.
>
> **Landed:** new `src/fragnnet/model/candidate_features.py`; the head takes one
> `candidate_feature_dim` and one `candidate_features` tensor; `MassterModel` takes
> `candidate_features: Sequence[str]`; 33 configs rewritten; the five old keys deleted from
> `config/template.yml`. Behaviour-preserving: all 229 MASSter configs resolve to the token
> they had before, 1,285 columns across the production family. Full suite 4,244 passed.

## The problem, counted

Adding `masster_candidate_cut_adjacency_features` on 2026-09-22 — two columns, no
parameters — required edits in nine places:

| # | Site | What it costs |
|---|---|---|
| 1 | `config/template.yml` | a bool key + comment block inside the `MASSter only` section |
| 2 | `MassterModel.__init__` signature | one more tail kwarg on a 60+ argument constructor |
| 3 | `MassterModel.__init__` body | `self.candidate_X = X` |
| 4 | `MassterModel.__init__` head call | `X_feature_dim=(N if candidate_X else 0)` |
| 5 | `MassterModel._X_features` | a builder whose first line is `if not self.flag: return None` |
| 6 | `CandidateIntensityTransformerHead.__init__` | kwarg + nonnegativity check + attribute + `input_dim += X` |
| 7 | `CandidateIntensityTransformerHead.forward` | kwarg + "supplied exactly when declared" check + shape check + append |
| 8 | `MassterModel.forward` | **two** near-identical call sites (unique / non-unique), one builder call and one head kwarg each |
| 9 | `masster_pl.py` | hparam passthrough |

Plus `tests/test_masster_model.py::_build_model` (a tenth site) and the arm config.

What the head carries today, and where each piece is declared:

| Token block | Width at `c4a_base_s1` | Governed by |
|---|---:|---|
| fragment / complement / boundary pools | 3 × 384 | structural |
| context | 128 | `masster_context_dim` |
| `cut_counts` | 1 | unconditional |
| `log1p(component_sizes)` | 1 | unconditional |
| `log1p(mask_component_counts)` | 1 | non-unique mode only |
| charge / mask log-probabilities | 1–2 | derived from `masster_proposal_anchored` |
| mass | 2 | `masster_candidate_mass_features` |
| Gasteiger charge | 1 | `masster_candidate_charge_feature` |
| complement composition | `len(CANONICAL_ELEMENT_ORDER) + 1` | `masster_candidate_composition_features` |
| cut-bond sum + mean | 2 × `edge_feature_dim` | `masster_candidate_cut_bond_features` |
| cut adjacency | 2 | `masster_candidate_cut_adjacency_features` |

Five booleans for five appends, three separate `*_feature_dim` integers on the head, five
forward kwargs with five copies of the same two validation branches — and 32 configs that each
restate `masster_candidate_mass_features: true` / `..._charge_feature: true` to describe a
token that never varies between them.

Nothing here is wrong. It does not scale: [the cut-set plan](masster_cut_set_features_plan.md)
still has item D (free-valence slots) and item B (a candidate-side bond encoder) queued, and
each would repeat all nine edits.

## Design

### 1. One config key holding an ordered list

```yaml
# Descriptor columns appended to each candidate's intensity token, in registry order.
masster_candidate_features:
- c_cut_count
- c_size
- c_mass
- c_charge
```

Precedent is `mol_params.pyg_node_feats` / `pyg_edge_feats`: a list of registered feature names
whose widths resolve at construction. The key stays flat and `masster_`-prefixed inside the
`MASSter only` template section — `tests/test_config_templates.py` asserts prefixed keys live
only in their own section, and the
[shared-reader trap](masster_cut_set_features_plan.md) is the reason not to invent a nested
`masster_params:` mapping for it.

**The name keeps the `masster_` prefix.** A bare `candidate_features` is the shorter read,
but `config/template.yml` is one flat schema shared by every model, and FraGNNet already declares
`candidate_ion_head`, `candidate_ion_formula_features`, `candidate_ion_neutral_loss_features`,
`candidate_ion_chunk_size` and `candidate_ion_checkpoint` unprefixed in its own section. An
unprefixed `candidate_features` beside them reads as "the candidate features" for either model.
The prefix is also what `test_config_templates.py` keys its section rule on: an unprefixed key in
a model section has to be added to `MODEL_SECTIONS`, whose comment says that is "the moment to
ask whether it should be prefixed instead". Dropping the prefix is one line in that tuple if
preferred.

Consequences worth stating:

- `deep_update` replaces a list wholesale rather than merging it, so an arm config declares the
  **whole** token. The campaign diff stays one key, and reading the config tells you the token
  without cross-referencing five booleans against the template defaults.
- A CLI override is the one place the inline form is unavoidable:
  `--override "masster_candidate_features=[c_cut_count, c_size]"` parses through the
  existing `yaml.safe_load` override path, replacing the three-line `--override` recipe in
  `config/debug/debug_tiny_masster_cut_features.yml`. Config files always use the
  block form.
- `find_unknown_config_keys` does not inspect list contents, so a misspelled entry must raise at
  model construction. That is the right place: the error can name the known features.

Before / after, tail of `candidate_token_2026_09_22/e1_cut_bond_feats.yml`:

```yaml
# before
masster_boundary_bond_encoder: true
masster_candidate_mass_features: true
masster_candidate_charge_feature: true
masster_candidate_cut_bond_features: true

# after
masster_boundary_bond_encoder: true
masster_candidate_features:
- c_cut_count
- c_size
- c_mass
- c_charge
- c_cut_bonds
```

### 2. One spec table, in one new module

`src/fragnnet/model/candidate_features.py`:

```python
@dataclass(frozen=True)
class CandidateFeatureInputs:
    """Every tensor a candidate feature builder may read, for one forward branch."""
    mol_pyg: Batch
    atom_indices: th.Tensor
    atom_component_indices: th.Tensor
    boundary_bond_indices: th.Tensor
    boundary_component_indices: th.Tensor
    component_batch_indices: th.Tensor
    cut_counts: th.Tensor
    num_components: int
    batch_size: int

    @cached_property
    def component_sizes(self) -> th.Tensor: ...
    @cached_property
    def component_neutral_masses(self) -> th.Tensor: ...
    @cached_property
    def precursor_neutral_masses(self) -> th.Tensor: ...
    @cached_property
    def component_element_counts(self) -> th.Tensor: ...
    @cached_property
    def precursor_element_counts(self) -> th.Tensor: ...


@dataclass(frozen=True)
class CandidateFeatureSpec:
    name: str
    width: Callable[[int], int]             # given the precursor bond-feature width
    build: Callable[[CandidateFeatureInputs], th.Tensor]
    requires: tuple[str, ...] = ()          # graph fields the builder reads


CANDIDATE_FEATURE_SPECS: tuple[CandidateFeatureSpec, ...] = (...)   # canonical order
```

An explicit module-level tuple, **not** a decorator registry: `feat_utils.py` carries a
`# TODO: remove this registry` over exactly that pattern, and a tuple gives the canonical column
order for free.

Three functions complete it:

- `resolve_candidate_features(names, dims) -> tuple[CandidateFeatureSpec, ...]` — validates and
  returns the specs in **registry order, not config order**.
- `candidate_feature_width(specs, dims) -> int` — what the head projects from.
- `build_candidate_features(specs, inputs) -> th.Tensor | None` — one `(num_components, width)`
  tensor, `None` for an empty list.

`CandidateFeatureDims` carries the two widths a spec may need: `edge_feature_dim` and
`composition_feature_dim`.

The `cached_property` fields matter for more than tidiness. Today `_candidate_scalar_features`
computes `component_neutral_masses` whether or not `c_mass` is on, because
`intensity_mass_pair_bias` needs the same tensor, and it computes precursor element counts over
the full graph for the composition feature. Caching on the inputs object makes each derived
quantity compute at most once per branch and lets the pair-bias path read
`inputs.component_neutral_masses` without a feature flag deciding for it.

### 3. The head takes one integer and one tensor

`CandidateIntensityTransformerHead.__init__`: `candidate_feature_dim: int` replaces
`use_mass_features`, `use_charge_feature`, `composition_feature_dim`, `cut_bond_feature_dim`
and `cut_adjacency_feature_dim`.

```python
input_dim = 3 * fragment_embedding_dim + context_dim + candidate_feature_dim
if not unique_fragment_tokens:
    input_dim += 1
if use_base_probability_feature:
    input_dim += 1 if unique_fragment_tokens else 2
```

`forward`: one `candidate_features: th.Tensor | None` kwarg replaces five, with one shape check
in place of five pairs. The positional `cut_counts` and `component_sizes` arguments **go away** —
they become `c_cut_count` and `c_size` list entries — along with the head's
`component_sizes.dtype == th.long` check, which the builder now guarantees by construction.
`MassterModel.forward` still computes both tensors for `IonModeProposalHead`, and hands them
to the inputs bundle.

### 4. What deliberately stays out of the list

| Stays out | Why |
|---|---|
| fragment / complement / boundary pools, context | the token's backbone, not descriptors; removing one is an architecture change, not an ablation |
| `mask_component_count`, charge/mask log-probabilities | proposal-derived, and their *presence* is derived from `masster_proposal_anchored` and unique mode rather than chosen. Listing them would allow the incoherent config "anchored **and** base-probability column", needing a cross-check that buys nothing. [Cut-set plan item C](masster_cut_set_features_plan.md) is a `proposal_anchored: false` question and stays where it is |
| `masster_boundary_bond_encoder` | a module (`CutBondEncoder`) that changes how a backbone block is *computed*; it owns no column |
| `masster_candidate_atom_dropout` | a regularizer on the pooled embeddings with a continuous value, inert at eval; it owns no column |
| `component_neutral_masses` for `mass_pair_bias` | an attention input, not a column; served from the cached bundle |

The rule: **the list governs appended descriptor columns; anything that parameterizes a module
keeps its own scalar key.** State this in the module docstring, with the two token blocks that
are not list-governed named explicitly, so the next reader does not have to rediscover it.

### 5. The registry

| Name | Width | Replaces | Source |
|---|---:|---|---|
| `c_cut_count` | 1 | unconditional column | `cut_counts` |
| `c_size` | 1 | unconditional column | `log1p(component_sizes)` |
| `c_mass` | 2 | `masster_candidate_mass_features` | `log1p(neutral mass)`, precursor fraction |
| `c_charge` | 1 | `masster_candidate_charge_feature` | summed Gasteiger charges |
| `c_nl_composition` | `len(CANONICAL_ELEMENT_ORDER) + 1` = 18 | `..._composition_features` | `log1p` complement element counts |
| `c_cut_bonds` | `2 * edge_feature_dim` | `..._cut_bond_features` | summed + averaged `chemical_bond_attr` |
| `c_cut_adjacency` | 2 | `..._cut_adjacency_features` | adjacent cut pairs, retained / lost |
| `c_free_valence` | 1 | *(not built)* | [cut-set plan item D](masster_cut_set_features_plan.md), `atom_free_valence_slots` |

Two judgment calls, made rather than deferred:

- **`c_cut_count` and `c_size` join the list** although they are unconditional today. They are
  per-candidate descriptors like the rest, and the token is then fully described by one key.
  It also makes them ablatable — `c_size` is the other half of the size/count law that the
  cut-set plan argues the head leans on, and there has never been a way to switch it off.
- **Names carry a `c_` prefix**, mirroring `a_` / `b_` in `feat_utils.py`. Bare `mass`, `size`,
  `charge` would be unusable to grep in this repository.

### 6. Column order and numerical equivalence

Order is the registry tuple, so two configs listing the same features in different orders build
bit-identical tokens.

Declare the registry in today's append order so the production token is unchanged:
`c_cut_count`, `c_size`, `c_mass`, `c_charge` → 5 columns → 3×384 + 128 + 5 = **1,285** at the
`c4a_base_s1` geometry, the same width and the same column order as the current base. The
in-flight `e1`–`e5` arms keep their widths too.

One order does change. `c_cut_count` and `c_size` used to sit immediately after the context and
*ahead* of the proposal-derived columns (the per-mask component count and the base charge/mask
log-probabilities), while every other descriptor sat after them; the whole descriptor block now
follows those columns. That is visible whenever `masster_unique_fragment_mode` or
`masster_proposal_anchored` is False.

**Corrected 2026-09-22, after the review:** the plan first said "none currently running", which
is true of the `ma_mi` lineage — every arm there pins both keys True — but false of the
repository. Both flags are `config/template.yml` **defaults**, so
`config/debug/debug_tiny_masster.yml` and `config/nist20_merged_inchikey/masster/s0.yml`
inherit them and do reach the reordered token. Concatenating into one `nn.Linear` makes this a
permutation of weight columns: it changes which weight a given seed initializes and nothing
else, and no checkpoint predates it. Stated precisely in the module docstring rather than
preserved.

### 7. Validation

At construction, `resolve_candidate_features` raises on:

- an unknown name — error lists the known names;
- a duplicate name;
- a non-list value.

At the first forward, each resolved spec's `requires` is checked against the batched graph, so a
molecular cache written before `atom_gasteiger_charges` existed fails with *"candidate feature
`c_charge` needs `atom_gasteiger_charges`; regenerate the molecular preprocessing cache"* rather
than a bare `AttributeError` from inside a builder.

Log the resolved layout once at construction, beside the existing
`Resolved MASSter proposal method` line: each `name: width` pair and the total token width.
Every run then records its own token geometry in its log.

### 8. Migration

No checkpoint exists for any of these keys, so the old names are **deleted**, not deprecated, and
nothing is added to `migrate_legacy_hparams`. A stale config then fails
`find_unknown_config_keys` loudly at load — the intended behaviour.

| Site | Count | Change |
|---|---:|---|
| `config/template.yml` | 5 keys out, 1 in | list default `c_cut_count`, `c_size`, `c_mass`, `c_charge` |
| configs setting mass/charge | 32 | two or three lines → one list |
| configs setting composition / cut-bond / cut-adjacency | 3 / 4 / 4 (overlapping) | one extra list entry |
| `masster_pl.py` | 1 | five passthroughs → one |
| `masster_model.py` | ~6 | constructor, attributes, head kwargs, three builders, two call sites |
| `bond_fragment_heads.py` | ~4 | constructor, `input_dim`, forward kwargs, validation |
| `tests/test_masster_model.py` | 1 factory + 10 uses | `candidate_features: Sequence[str] = ()` |
| `tests/test_masster_candidate_features.py` | 22 uses | same |
| `scripts/debug/masster_candidates/debug_candidate_cut_descriptors.py` | 2 uses | same |

`masster_candidate_atom_dropout` (4 configs) and `masster_boundary_bond_encoder`
(17 configs) are untouched.

### 9. Steps and verification

1. **New module + its tests.** `candidate_features.py` with the specs, the three functions and
   the inputs bundle; builders move over from `MassterModel` unchanged in behaviour.
   *Verify:* new tests — registry order independent of config order, width equals the sum of
   spec widths, unknown/duplicate/non-list raise, each builder reproduces the values the current
   per-feature tests assert.
2. **Head.** One `candidate_feature_dim`, one `candidate_features` tensor, drop the `cut_counts`
   and `component_sizes` positionals. *Verify:* `pytest tests/test_masster_candidate_features.py`
   including an explicit `input_dim == 1285` case at the `c4a_base_s1` geometry.
3. **Model.** Constructor takes `candidate_features: Sequence[str]`; the two forward branches each
   build one inputs bundle and make one builder call. *Verify:*
   `pytest tests/test_masster_model.py tests/test_masster_pl.py`.
4. **Config schema + passthrough.** Template key inside the `MASSter only` section; pl
   passthrough. *Verify:* `pytest tests/test_config_templates.py`, then the tiny smoke config
   `config/debug/debug_tiny_masster_cut_features.yml` with its `--override` recipe rewritten
   to the list form.
5. **Rewrite the 32 configs** (scripted, one pass). *Verify:*
   `pytest tests/test_config_templates.py::test_every_tracked_config_still_resolves` — its
   `assert len(failures) == 18` is the tripwire: a config missed in the rewrite raises the count.
6. **Docs.** Update [the cut-set plan](masster_cut_set_features_plan.md) key names, the
   `docs/README.md` rows, and `docs/project-state.md`.

Full suite before hand-off; report anything not run.

### 10. Not in scope

- The `charge_log_probabilities` positional that unique mode passes as `zeros_like` and the head
  then discards. Noticed, adjacent, left alone.
- Any change to which features are *on* in any lineage. The default list reproduces the current
  production token exactly; this refactor is behaviour-preserving by construction and no arm
  result moves because of it.
- `masster_candidate_atom_dropout`, the boundary bond encoder, and cut-set plan items B
  and D. Item D becomes a two-line spec entry afterwards, which is the point.

## Sample configs

### `config/template.yml`, inside the `MASSter only` section

Five keys and ~45 lines of per-feature rationale collapse to one key. The rationale moves to the
spec docstrings in `candidate_features.py`, where the builder that implements it also lives; the
YAML documents the key and the catalogue, per the
[documentation rules](../../rules/documentation.md).

```yaml
# Descriptor columns appended to each candidate's intensity token, concatenated in registry
# order regardless of the order listed here. `src/fragnnet/model/candidate_features.py` holds
# the specs and documents what each one measures and why. An unknown or duplicated name raises
# at model construction; the resolved layout is logged there too.
#   c_cut_count       1                                bonds severed
#   c_size            1                                log1p atom count
#   c_mass            2                                log1p neutral mass, precursor fraction
#   c_charge          1                                summed Gasteiger partial charges
#   c_nl_composition  18                               log1p complement element counts
#   c_cut_bonds       2 x edge_feature_dim (18 here)   severed-bond attributes, sum then mean
#   c_cut_adjacency   2                                adjacent cut pairs, retained then lost
# The token also carries the pooled fragment/complement/boundary embeddings, the context vector,
# and -- only when masster_proposal_anchored is off -- the proposal log-probability
# columns. None of those are governed by this list.
# `c_charge` requires atom_gasteiger_charges, absent from molecular caches written before
# 2026-09-20; `c_nl_composition` and `c_cut_bonds` widen the token, so a checkpoint trained with
# a different list will not load.
masster_candidate_features:
- c_cut_count
- c_size
- c_mass
- c_charge
```

### `nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_base_s1.yml` — the control

```yaml
# before
masster_boundary_bond_encoder: true
masster_candidate_mass_features: true
masster_candidate_charge_feature: true

# after
masster_boundary_bond_encoder: true
masster_candidate_features:
- c_cut_count
- c_size
- c_mass
- c_charge
```

Both resolve to the same 5 descriptor columns and the same 1,285-wide token. The list is
redundant with the template default here and is still written out: this file is the control every
arm diffs against, and a control that spells out its own token cannot be silently moved by a
later default flip.

### `candidate_token_2026_09_22/e1_cut_bond_feats.yml` — a one-entry arm

```yaml
# before
masster_boundary_bond_encoder: true
masster_candidate_mass_features: true
masster_candidate_charge_feature: true

# Arm override, the only key not in c4a_base_s1:
masster_candidate_cut_bond_features: true

# after
masster_boundary_bond_encoder: true

# Arm override: the only list entry not in c4a_base_s1 (+18 columns, token 1,303).
masster_candidate_features:
- c_cut_count
- c_size
- c_mass
- c_charge
- c_cut_bonds
```

### `e5_cut_bond_plus_adjacency.yml` — the stacked arm

```yaml
# Arm override: both relational entries on top of c4a_base_s1 (+20 columns, token 1,305).
masster_candidate_features:
- c_cut_count
- c_size
- c_mass
- c_charge
- c_cut_bonds
- c_cut_adjacency
```

One entry per line throughout, never the inline `[a, b]` form: the git diff then shows one
added line per added feature, and an arm's entire difference from its control is a single
`+ c_cut_bonds` line.

### `c4a_complement_composition.yml`

```yaml
# Arm override: GLACIER's psi(r_form - f_k) term (+19 columns, token 1,304).
masster_candidate_features:
- c_cut_count
- c_size
- c_mass
- c_charge
- c_nl_composition
```

### `config/debug/debug_tiny_masster_cut_features.yml` — the smoke config

Its paired control drops from three `--override` clauses to one:

```yaml
# before
#   --override masster_candidate_cut_bond_features=False \
#              masster_candidate_cut_adjacency_features=False \
#              masster_candidate_atom_dropout=0.0

# after
#   --override "masster_candidate_features=[c_cut_count, c_size, c_mass, c_charge]" \
#              masster_candidate_atom_dropout=0.0
```

and the body pins the whole token in one place instead of five:

```yaml
# Pinned, not inherited: a later default flip must not change what this smoke test covers.
# c_cut_bonds (+18) and c_cut_adjacency (+2) are the new blocks under test; the first four
# entries are what they sit next to.
masster_candidate_features:
- c_cut_count
- c_size
- c_mass
- c_charge
- c_cut_bonds
- c_cut_adjacency
masster_boundary_bond_encoder: true

# No width change; training-only, and higher than any production arm so the restore-an-emptied-
# candidate branch is actually exercised.
masster_candidate_atom_dropout: 0.3
```

The `--override` value passes through the existing `yaml.safe_load` in `load_config`, so the
quoted inline list parses without any new parsing code. It is the only place an inline list
appears; every config file writes the block form.
