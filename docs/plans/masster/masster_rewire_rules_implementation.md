# MASSter rewire rules: implementation spec for v1 (arm a0 and its null)

> **Status:** PREPARED · **Updated:** 2026-10-15
> Implementation spec for [the rewire-rules training plan](masster_rewire_rules_training_plan.md), phase 1,
> after the user's 2026-10-15 decisions. Built on 2026-10-15 (uncommitted); build deviations are recorded with the review log. Written against the working tree at commit
> `2e111b02` (branch `feiw_v2`); every file and line reference below was re-read against that tree, and the
> plan's stale ones are listed in "Corrections to the plan". Six build steps, each with files, signatures,
> tensor shapes and tests. The stage core was prototyped and checked against the G0 reference
> (`output/rewire_impl_2026_10_15/proto_stage.py`, `test_proto.py`, gitignored scratch; not production code).

## Decisions (user, 2026-10-15, final)

1. Extras are ADDED to the candidate set; they never displace ordinary fragments.
2. v1 is arm `a0`: extras get intensity-loss gradients only. The proposal-loss terms, the proposal outputs and the
   `bond_logits` activations/gradients under a proposal-only backward stay bit-equal to the control `i1`. Shared-encoder
   PARAMETER gradients under the full loss do change (the extras' intensity loss flows into the shared GNN embeddings that also
   feed the bond head); that is intended and is not claimed to be equal.
3. Separate vocabulary `REWIRE_RULE_VOCABULARY` (append-only, edit-spec CRC-32 hashes); not appended to
   `MASSTER_RULE_VOCABULARY`.
4. Relative cap per precursor: `extras <= min(masster_rewire_max_extra, floor(masster_rewire_max_extra_fraction * K))`,
   `K` = number of KEPT ordinary fragments of that spectrum (rows after `fragment_keep`). Applied after dedup and
   validity. Sort key: parent's detached normalized proposal log-probability, then depth, then atom-set bitset words
   (exact integer tie-break). User: "I don't mind having a bit more peaks supported, 25% is not a hard margin", so
   the plan's +25% p95 node-growth bar is NOT binding. The fraction is a config key, template default `0.5`; G0b
   sweeps `0.25 / 0.5 / 1.0` and reports reach that survives the cap against node growth and wall-clock, so the arm
   value is chosen from data. `masster_rewire_max_extra` (default 256) is a memory bound, not a modelling choice.
5. Null arm `a0null`: `masster_rewire_null` (default `False`). When true, each real rule match is replaced by null
   matches of the same edit SHAPE at random sites (the gate's topology null), same count per rule, deterministic
   per (canonical SMILES, rule name) via `zlib.crc32`. Same pool sizes and vocabulary columns; only the chemistry is
   wrong.

## Corrections to the plan (found reading the current code; each is a design input)

| # | Plan statement | Fact in the current tree | Consequence in this spec |
| --- | --- | --- | --- |
| C1 | Proposal outputs "computed from ordinary fragments before the append"; the head gets a filler `0` that is never read | `masster_model.py` step 4 (line 4178 `fragment_scores = th.stack(...)`) normalizes `base_fragment_log_probabilities` jointly with the intensity scores over ALL valid rows, and the `proposal_selection_only` block (line 4611) emits `proposal_candidate_mzs = full_candidate_mzs`, `proposal_fragment_logprobs = base_fragment_log_probabilities`, both over all valid rows. An appended extra with base value 0 would change every ordinary row's proposal log-prob normalizer and put extras into the coverage loss | Extras carry base log-probability `-inf` (so the normalizer of the base column is bit-unchanged: `exp(-inf - max) = 0` adds an exact zero) and the proposal block is SLICED to ordinary rows with a boolean mask (build step 3). Decision 2 then holds in its reworded form: proposal outputs, proposal-loss terms and the `bond_logits` gradient under a proposal-only backward are bit-equal to the control for ALL molecules. The proposal view is built from the PRE-merge validity (`ordinary_valid_channels`, see C5), so a merge that adds channels or makes a row newly valid cannot change the base normalizer or the proposal m/z set |
| C2 | Sulfone worked case: `slots(P) = 0` for aromatic carbons, gain 0 | `get_mol_graph` gives every aromatic carbon `atom_free_valence_slots = 1` (O: 1, S: 2 for sulfonyl; checked on diphenyl sulfone: `[1,2,1,1,...]`) | Diphenyl sulfone SO2 extrusion: `slots(P) = 12`, boundary 2, adjustment -2, `valence_gain = 12` (14 without the adjustment). Tests pin 12, not 0 |
| C3 | G0 `one_cut_pieces` is the depth-1 reference | `debug_masster_rewire_reach_ceiling.one_cut_pieces(piece, formed, bonds)` is called with the UNEDITED precursor bond set, so a rule-cut bond with both endpoints inside a product is RESTORED when another bond is removed. Ring-type cut rules (`PS_O_transfer_6ring`, `s_to_o_aromatic_C` on fused rings) therefore under-split: on `COP(=O)(OC)SCCC(=O)OC` the G0 reference gives 52 depth-1 sets, the edited-graph semantics give 160. Depth 0 is unaffected | Production depth-1 excludes rule-cut bonds (consistent with depth 0). The depth-1 parity test uses a corrected wrapper (`bonds - cut` passed to `one_cut_pieces`). Tell the main session: G0's `R_after1` (29.7%) is a slight UNDER-estimate for ring-type cut rules |
| C4 | Dedup identity is (atom set, edit signature); same atom set with a different edit stays a separate row | The G0 extras count and `edited_products` parity are on distinct ATOM SETS; two rows with one atom set make identical channels and would split one product's intensity | Dedup identity is the atom set alone. Indicators OR, `n_formed / n_internal_cut / n_deleted_heavy` take the max, `depth` the min, valence adjustment the max, parent the best-scoring one (build step 2). The "merge with an ordinary twin" rule of the plan is unchanged |
| C5 | Merged ordinary rows need their valence adjustment raised, which means recomputing their channels | Channel validity is monotone in `max_gain` and `ion_mzs` does not depend on it | The stage computes channels for the merged twin's atom set too and returns a sparse `merged_valid_channels`; the model ORs it into the ordinary row's `valid_channels` FOR THE INTENSITY HEAD ONLY. The model keeps the pre-merge tensor as `ordinary_valid_channels` and builds every proposal-side quantity from it (proposal m/z, proposal fragment log-probs and batch ids, the base-column normalizer, the PL coverage metrics), so merge-gained channels and merge-gained validity never touch the proposal outputs (they count as rewire gain). No recomputation of ordinary rows |
| C6 | Hybrid backend guard | The model constructor only has `candidate_backend in {cpu, cuda}`; `cuda_topology_hybrid` is not reachable from it (`grep hybrid` in `builder.py`, `masster_model.py`: none) | No hybrid guard; `connected_fragment_beam` is still rejected |
| C7 | Export: two columns `pred_node_rewire_rules` plus a parent id | A parent column must survive the exporter's peak filtering and `index_rebased` rebasing of node indices | v1 ships `pred_node_rewire_rules` only; the parent id is returned as the explainability tensor `fragment_rewire_parent_idxs` for analysis scripts. Parent column deferred |
| C8 | `max_matches_per_rule` default 4 | Written before the 17-rule catalog existed; the G0 reference has no truncation | Default 8, per-spectrum truncation counter reported; G0b reports reach lost to truncation |

Stale plan line references, all re-verified (plan -> current): `masster_model.py` keep block "~3731-3746" -> 3707-3745
(`old_to_new_fragment` at 3731); post-keep renormalization "~3746-3755" -> 3747-3755; atom dropout call 3756; pooling call
"~3758" -> 3761; head call "~3871" -> 3866 (zeros arg at 3870); `corrected_fragment_logits` "~3917-3920" -> 3917;
first channel call "~3614" -> 3614 (ok); `fragment_formed_bond_counts` "~3536, ~3967" -> 3536, 3967 (ok);
`_transferable_hydrogen_shifts` "~2453-2458" -> 2435-2460 (`valence_gain` at 2453-2457); marginal guard "~1625-1640" ->
1617-1645; `MassterStaticGraph` "line ~73" -> 73 (ok); `_cached_masster_templates` 153 (ok); standalone fallback "~2389" ->
2391; `_build_static_graph` 2454 (ok); template keys "~836" -> 836 (ok), `template.yml:812` ok; `i1` selection-only line 235 ok;
`masster_candidate_transformer.py` argsort "~1149" -> 1148, residual blocks 945-950 and 1385 ok; export node merge
"~215-240" -> `masster_export.py` 202-232; `cuda_candidates.py:421` ok; `fragment_internal_edges` 309 ok;
`tests/test_masster_pl.py:304` ok.

## Build steps

Keys, in order: `rules_module`, `expansion_stage`, `model_wiring`, `pl_config_export`, `null_arm`, `parity_tests`.
Order is binding (each step's tests run before the next starts). Shared-file edits are small hunks gated on the new flag;
re-read the target file immediately before each Edit.

### Step 1 `rules_module`: rules, vocabulary, matches

New file `src/fragnnet/models/masster/candidates/rewire_rules.py`. No edit to any other file.

```python
PRECURSOR_TYPES_ALLOWED = ("[M+H]+", "[M-H]-", "[M+NH4]+", "[M+Na]+")   # the only values a rule may list

@dataclass(frozen=True)
class RewireRule:
    name: str
    smarts: str                                   # mapped SMARTS, as in the gate
    delete: tuple[int, ...]                       # map numbers; every bond of these atoms goes, they leave in the neutral
    cut: tuple[tuple[int, int], ...]              # map-number pairs, unordered
    form: tuple[tuple[int, int], ...]
    precursors: tuple[str, ...]                   # allowed precursor types
    null_smarts: str | None = None                # overrides the automatic topology null (3 rules)

@dataclass(frozen=True)
class RewireMatch:
    rule_index: int                               # 0-based position in REWIRE_RULE_VOCABULARY (= residual column)
    motif_atoms: tuple[int, ...]                  # sorted molecule-local ids of EVERY mapped atom of the match
    motif_numbers: tuple[int, ...]                # atomic numbers aligned with motif_atoms (graph cross-check)
    deleted: tuple[int, ...]                      # sorted local ids
    cut_pairs: tuple[tuple[int, int], ...]        # each (min, max), sorted
    form_pairs: tuple[tuple[int, int], ...]       # each (min, max), sorted
    n_deleted_heavy: int                          # deleted atoms with atomic number > 1

@dataclass(frozen=True)
class RewireMatchSet:
    matches: tuple[RewireMatch, ...]              # deterministic order, truncated per rule
    num_truncated: int                            # matches dropped by max_matches_per_rule (sum over rules)

REWIRE_RULES: tuple[RewireRule, ...]              # the 17 admitted rules, literal, catalog order (below)
REWIRE_RULE_VOCABULARY: tuple[str, ...]           # LITERAL tuple of the 17 names (append-only), checked against REWIRE_RULES
MASSTER_NUM_REWIRE_RULES: int                      # == len(REWIRE_RULE_VOCABULARY) == 17
def rewire_rule_spec(rule: RewireRule) -> str
REWIRE_RULE_SPEC_HASHES: tuple[int, ...]          # zlib.crc32(rewire_rule_spec(r).encode("utf-8")), vocabulary order

def raw_rule_matches(mol: Chem.Mol, pattern: Chem.Mol) -> list[dict[int, int]]
def build_rewire_matches(mol: Chem.Mol, *, max_matches_per_rule: int) -> RewireMatchSet
def filter_rewire_matches(match_set: RewireMatchSet, *, precursor_type: str, num_known_rules: int) -> RewireMatchSet
```

**Rules, copied from `output/rewire_rule_gate_2026_10_15/rule_catalog.json` (status `"admitted"` only; the file is
gitignored, so the specs live in code).** Name | SMARTS | delete | cut | form | precursors. Precursors `P3` =
`("[M+H]+","[M-H]-","[M+NH4]+")`. Catalog order is the vocabulary order:

| # | name | smarts | delete | cut | form | precursors |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `sulfonyl_so2_extrusion` | `[#6:1][SX4:2](=[OX1:3])(=[OX1:4])[!#1:5]` | 2,3,4 | | (1,5) | P3 |
| 1 | `s_to_o_aromatic_C` | `[c:1][SX4:2](=[OX1:3])(=[OX1:4])[!#1:5]` | | (1,2),(2,3) | (1,3) | P3 |
| 2 | `thiono_thiolo_aryl` | `[c:1][OX2:2][PX4:3]=[SX1:4]` | | (1,2),(3,4) | (1,4) | P3 |
| 3 | `PS_O_transfer_6ring` | `[PX4:1][SX2:2][#6:3][*:4][#6,#16:5]=[OX1:6]` | | (1,2) | (1,6) | P3 |
| 4 | `nitro_nitrite_NO_loss` | `[#6:1][N+:2](=[OX1:3])[OX1-:4]` | 2,3 | | (1,4) | P3 |
| 5 | `acyclic_thioether_s_extrusion` | `[#6:1][#16X2;!R:2][#6:3]` | 2 | | (1,3) | P3 |
| 6 | `cyclic_thioether_s_extrusion` | `[#6:1]@[#16X2;R;!a:2]@[#6:3]` | 2 | | (1,3) | P3 |
| 7 | `diaryl_sulfide_s_extrusion` | `[c:1][#16X2;!R:2][c:3]` | 2 | | (1,3) | P3 |
| 8 | `hetaryl_thio_alkyl_s_extrusion` | `[CX4:1][#16X2;!R:2][c:3]:[n]` | 2 | | (1,3) | P3 |
| 9 | `thiadiazole_cs2_extrusion` | `[#6:1][#16X2:2][#6:3]1~[#7:4]~[#7:5]~[#6:6]~[#16X2:7]1` | 2,3,7 | | (1,4) | P3 |
| 10 | `diaryl_ether_o_excision` | `[c:1][OX2:2][c:3]` | 2 | | (1,3) | P3 |
| 11 | `carbamate_co2_extrusion` | `[#7:1][CX3:2](=[OX1:3])[OX2:4][#6:5]` | 2,3,4 | | (1,5) | P3 |
| 12 | `co2_extrusion_aryl_Oside` | `[#6,#7,#8:1][CX3:2](=[OX1:3])[OX2:4][c:5]` | 2,3,4 | | (1,5) | P3 |
| 13 | `azo_N2_extrusion` | `[#6:1][NX2:2]=[NX2:3][!#1;!X1:4]` | 2,3 | | (1,4) | `("[M-H]-",)` |
| 14 | `dinitrophenol_24_2NO` | `[OX2H1]c1[c:1]([N+:2](=[OX1:3])[OX1-:4])c[c:5]([N+:6](=[OX1:7])[OX1-:8])cc1` | 2,3,6,7 | | (1,4),(5,8) | `("[M-H]-",)` |
| 15 | `acid_acyl_transfer_5` | `[OX2H1:1][CX3:2](=[OX1:3])[#6:4][NX3:5][CX3:6]=[OX1:7]` | | (2,1),(5,6) | (1,6) | `("[M+Na]+",)` |
| 16 | `haloalkyl_sulfonyl_X_to_S` | `[SX4:1](=[OX1])(=[OX1])[CX4:2][F,Cl,Br;D1:3]` | | (2,3) | (1,3) | `("[M-H]-",)` |

Null overrides (`null_smarts`, copied): rule 13 `[#6:1][!#1;!D1:2]~[!#1;!D1:3]~[!#1;!X1:4]`; rule 14
`[#6:1][!#1;!D1:2](~[*;D1:3])~[*:4].[#6:5][!#1;!D1:6](~[*;D1:7])~[*:8]`; rule 15 `[*;D1:1]-[#6:2]-[*:4]-[*:5]-[*:6]`.
All others: `None` (automatic, step 5). Mapped atoms are only those with a `:n`; unmapped SMARTS atoms (rules 8, 14, 16) are not
part of `motif_atoms` (this is the G0 containment definition: `set(match.values()) <= fragment`).

**`rewire_rule_spec(rule)`** returns exactly
`f"{name}|{smarts}|{','.join(map(str, sorted(delete)))}|{';'.join(f'{a}-{b}' for a, b in sorted(tuple(sorted(p)) for p in cut))}|{';'.join(f'{a}-{b}' for a, b in sorted(tuple(sorted(p)) for p in form))}|{','.join(sorted(precursors))}"`.
`null_smarts` is NOT in the hash: it only affects the null arm, and a null SMARTS change must not break loading real-arm
checkpoints.

**Import-time validation (module level, `ValueError`; a rule file typo must fail on import, never at batch time):**
names unique; `REWIRE_RULE_VOCABULARY` (a literal tuple, NOT derived from `REWIRE_RULES`) equals
`tuple(r.name for r in REWIRE_RULES)` (so reordering or removing a rule fails; appending requires editing both); each SMARTS
parses; every map number in `delete/cut/form` exists in the SMARTS; every `cut` pair is a bond of the SMARTS pattern
(checked with `pattern.GetBondBetweenAtoms`); `form` is non-empty (the stage's product definition needs a formed bond) and
no form endpoint is in `delete`; no pair has equal endpoints; every `precursors` entry is in `PRECURSOR_TYPES_ALLOWED`;
`null_smarts` parses and its map numbers are a subset of the rule's.

**`raw_rule_matches(mol, pattern)`** reproduces the gate's `_matches(mol, pattern, limit=None)` exactly:
`mol.GetSubstructMatches(pattern, uniquify=False, maxMatches=5000)`, one dict `{map_number: atom_idx}` per hit
(unmapped atoms omitted), deduplicated on `tuple(sorted(d.items()))`, original order kept. It is shared with the null.

**`build_rewire_matches(mol, *, max_matches_per_rule)`**: for each rule in vocabulary order, compile its pattern once at
import (`_PATTERNS`), take `raw_rule_matches`, convert each dict to a `RewireMatch` (delete maps present, cut and form
pairs through the dict, pairs normalized `(min, max)`, tuples sorted), deduplicate on
`(rule_index, motif_atoms, deleted, cut_pairs, form_pairs)` (two matches that differ only by swapping symmetric mapped atoms,
e.g. the sulfonyl O3/O4, make the same edit and containment), sort by that key, keep the first `max_matches_per_rule` per
rule and add the rest to `num_truncated`. The input is the STANDARDIZED molecule that `_cached_masster_templates` parses
(`Chem.MolFromSmiles(mol_smiles)`), never the dataset `mol_df` object, so atom order equals the graph's; the stage
verifies atomic numbers (build step 2). Pure Python, no torch.

**`filter_rewire_matches(match_set, *, precursor_type, num_known_rules)`**: keep matches whose rule lists `precursor_type`
and whose `rule_index < num_known_rules` (rules appended after a checkpoint stay off, as `_eligible_smarts_templates`);
`num_truncated` is carried over unchanged.

Tests, new `tests/test_masster_rewire_rules.py`:

- Registry: 17 rules; `MASSTER_NUM_REWIRE_RULES == 17`; vocabulary equals the literal list in this spec; the four
  adduct-specific rules (13, 14, 15, 16) list one precursor type; the others list three.
- Append-only: a copy of `REWIRE_RULES` with two rules swapped, or one removed, raises (call the validator on the copy);
  `REWIRE_RULE_SPEC_HASHES` has 17 distinct values; changing a rule's SMARTS, `delete`, `cut`, `form` or `precursors` while
  keeping the name changes its hash; changing only `null_smarts` does not.
- Import validation: a rule with a cut pair that is not a pattern bond, a form endpoint that is deleted, an unknown map
  number, an empty `form`, an unknown precursor type each raise `ValueError`.
- Golden instances (one per rule, hand molecules): e.g. rule 0 on `O=S(=O)(c1ccccc1)c1ccccc1` gives ONE `RewireMatch`
  after dedup (the raw list has four: O3/O4 swapped times the swapped :1/:5 ends, all one edit) with
  `deleted == (0, 1, 2)`, `form_pairs == ((3, 9),)`, `n_deleted_heavy == 3`; rule 4 on nitrobenzene; rule 5 vs 6
  (acyclic vs ring thioether) on `CCSCC` (rule 5 only) and thiane `C1CCSCC1` (rule 6 only); rule 1 on dibenzothiophene
  S,S-dioxide `O=S1(=O)c2ccccc2-c2ccccc12` (four distinct edit keys: two O choices times two aryl ends); rule 15 on a hand
  N-acyl amino acid; each golden SMILES is documented with its expected match tuples.
- Determinism and truncation: `max_matches_per_rule=1` on a molecule with two distinct thioethers keeps the lowest key and
  reports `num_truncated == 1`; the order is independent of the SMILES atom ordering of an isomorphic input only through the
  standardization (do not assert it), but IS identical across two calls.
- Filter: `[M-H]-`-only rules vanish for `[M+H]+`; `num_known_rules=5` drops indices >= 5; `num_truncated` carried.
- A molecule with no match returns an empty set and `num_truncated == 0`.

### Step 2 `expansion_stage`: the after-selection stage

New file `src/fragnnet/models/masster/rewire_expansion.py`. No edit to any shared file. Pure torch on the device of the
inputs, integer ops only, no autograd through anything. Reference prototype: `output/rewire_impl_2026_10_15/proto_stage.py`
(depth 0 and depth 1 equal the corrected G0 reference on hand molecules; dedup and merge verified).

```python
BITS_PER_WORD = 62                                # nonnegative int64 words
EDIT_FEATURE_DIM = 4                              # n_formed, n_internal_cut, n_deleted_heavy, depth

ChannelFn = Callable[
    [th.Tensor, th.Tensor, th.Tensor, int, th.Tensor, th.Tensor],
    tuple[th.Tensor, th.Tensor],
]   # (atom_indices, atom_fragment_indices, fragment_batch_indices, num_fragments, cut_counts, valence_adjustment)
    #  -> (ion_mzs (n, M, H), valid_channels (n, M, H) bool)

@dataclass(frozen=True)
class RewireExpansion:
    num_extra: int                                # E
    extra_atom_indices: th.Tensor                 # (Mx,) long, GLOBAL atom ids of every extra's membership
    extra_atom_extra_indices: th.Tensor           # (Mx,) long in [0, E)
    extra_boundary_bond_indices: th.Tensor        # (Bx,) long, columns of chemical_bond_index
    extra_boundary_extra_indices: th.Tensor       # (Bx,) long in [0, E)
    extra_batch_indices: th.Tensor                # (E,) long spectrum
    extra_cut_counts: th.Tensor                   # (E,) long = |Boundary(P)|
    extra_valence_adjustment: th.Tensor           # (E,) long = -2 * (n_formed - n_internal_cut), merged max
    extra_ion_mzs: th.Tensor                      # (E, M, H) the channel_fn dtype
    extra_valid_channels: th.Tensor               # (E, M, H) bool, every row has >= 1 valid channel
    extra_rule_indicators: th.Tensor              # (E, R) float32 0/1
    extra_edit_features: th.Tensor                # (E, 4) float32
    extra_parent: th.Tensor                       # (E,) long, ordinary post-keep row of the best parent
    merged_rows: th.Tensor                        # (K,) long, ordinary post-keep rows that have a product twin
    merged_valid_channels: th.Tensor              # (K, M, H) bool, OR into the ordinary row's valid_channels
    merged_rule_indicators: th.Tensor             # (K, R) float32
    merged_edit_features: th.Tensor               # (K, 4) float32
    diagnostics: dict[str, th.Tensor]             # name -> (batch_size,) float32, names below

REWIRE_DIAGNOSTIC_NAMES: tuple[str, ...] = (
    "rewire_match_count", "rewire_truncated_match_count", "rewire_pair_count", "rewire_contained_pair_count",
    "rewire_precap_extra_count", "rewire_merged_count", "rewire_dropped_no_valid_count",
    "rewire_dropped_cap_count", "rewire_extra_count", "rewire_cap_tie_count",
)

def atom_set_words(member: th.Tensor, num_words: int) -> th.Tensor          # (n, A) bool -> (n, W) int64
def words_to_member(words: th.Tensor, num_atoms: int) -> th.Tensor          # inverse
def lexsort(keys: Sequence[th.Tensor]) -> th.Tensor                         # permutation, keys[0] most significant, ascending
def label_components(member, edge_row, edge_a, edge_b) -> th.Tensor         # (R, A) long: min atom id of the component, -1 outside

def expand_rewire_candidates(
    *,
    match_sets: Sequence[RewireMatchSet],         # one per spectrum, ALREADY filtered by precursor type
    atom_ptr: th.Tensor,                          # (B + 1,) long, device
    atom_numbers: th.Tensor,                      # (A_total,) long, device (mol_pyg.atom_atomic_numbers)
    chemical_bond_index: th.Tensor,               # (2, Nb) long, device, global atom ids, bonds contiguous per spectrum
    host_chemical_bond_index: th.Tensor,          # (2, Nb) long, CPU (static_graph.chemical_bond_index)
    fragment_atom_indices: th.Tensor,             # (M,) ordinary post-keep memberships, global atom ids
    atom_fragment_indices: th.Tensor,             # (M,) ordinary post-keep fragment index
    fragment_batch_indices: th.Tensor,            # (n_ord,)
    parent_log_probabilities: th.Tensor,          # (n_ord,) detached normalized proposal log-probabilities
    parent_blocked: th.Tensor | None,             # (n_ord,) bool or None: rows that may not be parents
    bond_logits: th.Tensor | None,                # (Nb,) detached; read only when continuation_cuts > 0
    channel_fn: ChannelFn,
    max_extra_fraction: float,
    max_extra: int,
    continuation_cuts: int,
    pair_chunk: int,
    num_rules: int,
) -> RewireExpansion
```

**Algorithm.** Let `B = atom_ptr.numel() - 1`, `n_ord = fragment_batch_indices.numel()`, `active = [s : match_sets[s].matches]`.
If `active` is empty (host check, no device work) return `_empty_expansion(...)`: every tensor with a zero leading
dimension; `extra_ion_mzs` and `extra_valid_channels` are `(0, 0, 0)` because `M, H` are unknown without a channel call, so the
model must NOT concatenate them when `num_extra == 0` (`torch.cat` does not skip `(0, 0, 0)` against `(n, M, H)`; see the
append rules in step 3); `merged_rows` empty; all diagnostics zeros except `rewire_match_count` and
`rewire_truncated_match_count` (host values).

1. **Hard-error checks (host, before any tensor work; every violation is a `ValueError` naming spectrum and rule).**
   `len(match_sets) == B`; per match every `motif_atoms`, `deleted`, `cut_pairs`, `form_pairs` id is in `[0, n_atoms[s])`;
   `atom_numbers[atom_ptr[s] + motif_atoms] == motif_numbers` (one `atom_numbers.cpu()` copy per batch, only when
   `active` is non-empty: an RDKit-vs-graph atom order mismatch is a hard error, the same rule as SMARTS-vs-CUDA
   templates); every cut pair is a bond of that spectrum (dict `{(min,max): global_bond_col}` built from
   `host_chemical_bond_index`; unresolved is a `ValueError`); `chemical_bond_index` columns are sorted by spectrum
   (`bond_batch[1:] >= bond_batch[:-1]`); after the dense membership is built (step 3): every ordinary membership atom lies
   inside its spectrum, and no two ordinary rows of one spectrum have the same bitset (lexsort and adjacent compare). BEFORE writing this check the builder
   must read `_unique_component_memberships` and `_merge_forced_masks` and cite in the code comment why duplicates are impossible
   on each path (forced masks, `share_proposal_across_ev`, SMARTS-appended rows after `fragment_keep`, memorization sink); if any path
   can duplicate, the stage must dedup ordinary duplicates (keep the first row, map the others to it) instead of raising, because a
   hard error would abort a long training job. Parametrized tests (SMARTS on, forced masks, shared EV views, sink on) run the stage
   and assert no error.
2. **Pack matches** (host lists to one device transfer). `Mt` matches in (spectrum, rule, key) order:
   `match_spectrum (Mt,)`, `match_rule (Mt,)`, `motif (Mt, Kmax)` local ids padded `-1`, `deleted_dense (Mt, Amax)` bool,
   `cut_u, cut_v (Mt, Cmax)` local ids padded `-1` (Cmax >= 1), `form_u, form_v (Mt, Fmax)` padded `-1`, `n_deleted_heavy (Mt,)`.
   `Amax = max n_atoms over active spectra`, `W = ceil(Amax / 62)`. Per-spectrum bond tables from `chemical_bond_index`:
   `bond_batch = bucketize(chemical_bond_index[0], atom_ptr[1:], right=True)`, `bond_count (B,)`, `bond_ptr (B + 1,)`,
   `bond_u_local = chemical_bond_index[0] - atom_ptr[bond_batch]`, `bond_v_local` likewise.
3. **Dense membership of the ordinary rows of active spectra** (the only dense matrix; `N x Amax` bool, tiny):
   `active_frag = is_active[fragment_batch_indices]`, `frag_row = cumsum(active_frag) - 1`, `N = int(active_frag.sum())`
   (host sync 1), `active_ids = nonzero(active_frag)`, `member[N, Amax]` set from the memberships of active rows with local
   ids `atom - atom_ptr[batch]`, `ordinary_words = atom_set_words(member, W)`.
4. **Pairs.** For every active ordinary row `f` of spectrum `s`, one pair per match of `s`, enumerated in
   (spectrum, row, match) order: `reps = match_count[batch]`, `pair_frag = repeat_interleave(arange(N), reps)`,
   `pair_match = match_start[batch][pair_frag] + (arange(P) - repeat_interleave(reps.cumsum(0) - reps, reps))`.
   `P` is read once (host sync 2). Per spectrum `rewire_pair_count` is `bincount` of `pair_spectrum`.
   Processing is chunked so memory is bounded by the number of DISTINCT atom sets, not the pre-cap product count. Chunks are sized by
   an ELEMENT budget (`pair_chunk` is the key name kept for config stability but is interpreted as a budget of
   `pairs x max(Amax, max_bonds_per_spectrum)` elements divided by that maximum, so a large molecule gets fewer pairs per chunk). Per chunk:
   a. containment: `member[pair_frag[:, None], motif[pair_match].clamp_min(0)] | (motif[pair_match] < 0)` all-true, and
      `~parent_blocked[active_ids[pair_frag]]` when given (rows with a nonzero SMARTS template formed count never parent);
   b. surviving pairs `R_c`; `P_mem = member[pair_frag] & ~deleted_dense[pair_match]` (`R_c x Amax`);
   c. edges: `(row, bond)` pairs from `repeat_interleave` over `bond_count[spectrum]`, keep bonds with both local endpoints in
      `P_mem` and not a rule cut (endpoint compare against `cut_u/cut_v` of the pair's match, both orientations), plus the
      form pairs `(form_u, form_v)` where `>= 0`. A form endpoint outside `P_mem` is a `ValueError` (impossible for a
      validated rule, since form endpoints are mapped and never deleted; it is an assertion, not a skip);
   d. `labels = label_components(P_mem, edge_row, edge_a, edge_b)` (below);
   e. products: component label of every form slot `fl = labels[row, form_u]` (`-2` for padding); the FIRST slot of each
      distinct label per row starts one piece (`same = (fl[:, :, None] == fl[:, None, :]) & (fl[:, :, None] >= 0)`,
      `earlier = (same & (slot[None, None, :] < slot[None, :, None])).any(2)`, parenthesised: `&` binds tighter than `<`; `[0, i, j]` is `j < i`; a hand test uses two form slots that share a label); piece membership `labels[row] == label`;
      `n_formed` = slots whose label equals the piece's; `n_internal_cut` = rule cuts of that match with both endpoints in the
      piece; `depth = 0`; `parent = pair_frag`; `n_deleted_heavy` of the match. A component without a formed bond is an
      ordinary connected set and is NOT a product (the G0 definition);
   f. continuation (only `continuation_cuts > 0`): for each piece, the precursor bonds with both endpoints in the piece,
      MINUS that match's rule cuts (correction C3), ranked by `(-round(detached bond_logit * 1e6), global bond col)` ascending
      within the piece, top `continuation_cuts`; each (piece, removed bond) row gets `label_components` again with the bond
      removed and formed bonds kept; pieces holding a formed bond are emitted with `depth = 1`, `n_formed/n_internal_cut`
      recomputed for the new piece, parent inherited. A removed bond that does not split the piece re-emits the same atom set,
      which the dedup then merges (depth keeps the min).
   REDUCE PER CHUNK: after each chunk, dedup its pieces on `(batch, words)` and merge into a running table that holds only the
   per-group statistics (OR of rule indicators, max `n_formed/n_internal_cut/n_deleted_heavy`, min `depth`, max valence adjustment,
   best parent key); merge by another lexsort over `(running, chunk)`. Peak memory is therefore the running distinct-set table plus
   one chunk. The per-piece records of step 5 are the running table's rows, not the raw pieces.
5. **Exact dedup, atom set only (C4).** Rows = ordinary active rows (flag 0) followed by pieces (flag 1), key
   `(batch, words[0..W-1], flag)` via `lexsort`; a new group starts where `(batch, words)` changes. Per group: `has_ordinary`,
   `has_piece`. Groups with a piece and no ordinary row are EXTRA candidates; groups with both are MERGED. Group statistics
   over its pieces: `rule_indicator = OR of one-hot(rule)`; `n_formed, n_internal_cut, n_deleted_heavy = max`; `depth = min`;
   `valence_adjustment = max(-2 * (n_formed_i - n_internal_cut_i))` over the group's pieces (computed per piece, then max);
   best parent = the piece with the largest `quantized parent key`, ties to the smaller parent row; `rewire_precap_extra_count`
   counts extra groups, `rewire_merged_count` merged groups, per spectrum.
6. **Boundary, cut counts, channels.** For every extra or merged group the atom set is rebuilt from its words
   (`words_to_member`); `Boundary(P)` = precursor bonds of the spectrum with exactly one endpoint in `P`
   (`(group, bond)` expansion, `P[u] ^ P[v]`), `cut_count = |Boundary(P)|`. Call
   `channel_fn(atom_indices, atom_group_indices, group_batch, num_groups, cut_counts, valence_adjustment)` once over all
   groups (global atom ids `local + atom_ptr[batch]`). Merged groups: `merged_valid_channels = valid[merged]`. Extra groups with
   `valid.flatten(1).any(1) == False` are dropped (`rewire_dropped_no_valid_count`).
7. **Cap (last, on deduplicated valid extras).** Quantized parent key: cast `parent_log_prob` to float32, assert finite (round of -inf to long is undefined), `q = round(. * 2**16)` as int64.
   The quantum only removes noise far below one quantum; it does NOT give exact CPU/CUDA parity (values straddling a rounding
   boundary flip by one quantum with probability ~delta/1.5e-5), and under bf16 autocast near-ties are frequent. The integer tie-break
   makes the order deterministic given the keys. Sort extras by `(batch asc, -q asc, depth asc, words asc)`; rank within a spectrum;
   `limit_s = min(max_extra, floor(max_extra_fraction * K_s + 1e-9))` computed once on the host in int64 (the epsilon prevents `0.29 * 100 = 28.999999999999996` flooring one low; tests use fractions 0.29 and 0.7 against exact integers) with `K_s = bincount(fragment_batch_indices)[s]` (all kept
   ordinary rows, valid or not); keep `rank < limit_s`. `rewire_dropped_cap_count = valid_extras - kept`;
   `rewire_cap_tie_count` = spectra where the last kept and the first dropped row have equal `(q, depth)` (the integer words
   decided; reported so a CPU/CUDA parity failure is diagnosable). Surviving extras are then re-ordered by
   `(batch asc, words asc)` so the appended row order is canonical and independent of parent enumeration.
8. **Emit.** `extra_*` tensors in that order, membership and boundary in sparse form, `extra_parent` = the ordinary
   post-keep row index `active_ids[parent_row]`.

**Helpers (verified in the prototype; copy them).**

```python
def atom_set_words(member, num_words):
    n, a = member.shape
    m = F.pad(member, (0, num_words * BITS_PER_WORD - a)).view(n, num_words, BITS_PER_WORD).to(th.long)
    weights = th.bitwise_left_shift(th.ones(BITS_PER_WORD, dtype=th.long, device=member.device),
                                    th.arange(BITS_PER_WORD, device=member.device))
    return (m * weights).sum(-1)

def words_to_member(words, num_atoms):
    n, w = words.shape
    bits = (th.bitwise_right_shift(words[:, :, None], th.arange(BITS_PER_WORD, device=words.device)) & 1).bool()
    return bits.reshape(n, w * BITS_PER_WORD)[:, :num_atoms]

def lexsort(keys):
    order = th.arange(keys[0].shape[0], device=keys[0].device)
    for key in reversed(keys):
        order = order[th.argsort(key[order], stable=True)]
    return order

def label_components(member, edge_row, edge_a, edge_b):
    r, a = member.shape
    ids = th.arange(a, device=member.device).expand(r, a)
    labels = th.where(member, ids, th.full_like(ids, a)).contiguous()          # a = sentinel
    if edge_row.numel() == 0:
        return th.where(member, labels, th.full_like(labels, -1))
    src = th.cat([edge_row * a + edge_a, edge_row * a + edge_b])
    dst = th.cat([edge_row * a + edge_b, edge_row * a + edge_a])
    for sweep in range(a + 1):
        flat = labels.reshape(-1)
        new = flat.clone().scatter_reduce(0, dst, flat[src], reduce="amin", include_self=True).view(r, a)
        jumped = th.minimum(new, new.gather(1, new.clamp(max=a - 1)))             # pointer jumping
        jumped = th.where(new == a, new, jumped)
        if (sweep % 4 == 3 or sweep == a) and th.equal(jumped, labels):            # one sync every 4 sweeps
            break
        labels = jumped.contiguous()
    else:
        raise ValueError("component labelling did not converge within the atom count")
    return th.where(member, labels, th.full_like(labels, -1))
```

(The prototype checks convergence every sweep; checking every fourth sweep, as here, is equivalent because a `break` only
fires when `jumped == labels`, a fixed point. The `sweep == a` clause keeps molecules with `Amax < 4` from raising. A hand
test covers a path graph of length `Amax` and an `Amax` of 1, 2 and 3.)

Host syncs per forward when any spectrum has a match: `active_frag.sum()`, pair count, one per chunk, one per four
labelling sweeps per chunk, group count, kept-extra count. None when no spectrum has a match. Report the count in G0b.

Peak memory is the running distinct-atom-set table plus one chunk of `chunk x Amax` bools and `chunk x bonds` edge candidates, with the
chunk sized by the element budget above; with `continuation_cuts` the per-piece continuation rows are reduced inside the chunk too.
A test builds a synthetic worst case (many matches, a large pool, 120-atom molecules) and asserts peak allocation under a fixed bound
(CPU `tracemalloc`-free check: the element budget times 8 bytes times a small constant); G0b records `max_memory_allocated` at
`train_batch_size` 512 for the largest fraction (1.0).

Tests, new `tests/test_masster_rewire_expansion.py` (CPU; a `_stub_channel_fn` that returns all-valid channels of shape
`(n, 2, 3)` and `ion_mzs = zeros` is used wherever the model is not needed):

- Helpers: `atom_set_words`/`words_to_member` round trip at `Amax` in `{1, 61, 62, 63, 123, 124, 125}`; `lexsort` against
  `sorted` on random tuples; `label_components` on a path of length `Amax`, a ring, two components, an isolated atom,
  against `networkx.connected_components`.
- Reference parity at depth 0: for the 8 hand molecules of the prototype (sulfonamide, nitro sulfide, diaryl ether sulfone,
  thioether amide, thiophosphate, carbamate, thiazole thioether, azo) with a seeded pool of components of random 1-3 bond
  cuts: the DISTINCT atom sets of all appended rows plus merged groups equal `products_for(...)[0]` of the G0 reference,
  and `extra_count + merged_count == len(reference)` before the cap.
- Reference parity at depth 1 against the CORRECTED G0 wrapper (`one_cut_pieces(piece, formed, bonds - rule_cuts)`),
  `continuation_cuts` large, on the same molecules plus `COP(=O)(OC)SCCC(=O)OC` and `O=S1(=O)c2ccccc2-c2ccccc12` (the
  ring-type cases). A second assertion documents C3: with the UNCORRECTED reference the first molecule differs.
- Mixed batch: the same molecules in two different batch orders and as single-spectrum runs give per-spectrum identical
  extras (atom sets, order, parents' atom sets, adjustments, indicators). A no-match spectrum between two matching ones
  yields zero extras and `rewire_match_count == 0`.
- Hand values (`slots` from `get_mol_graph`, C2): diphenyl sulfone `O=S(=O)(c1ccccc1)c1ccccc1`, rule 0, F = precursor:
  one extra, 12 atoms, `cut_count == 2`, `valence_adjustment == -2`, `n_formed == 1`, `n_internal_cut == 0`,
  `n_deleted_heavy == 3`; with `slots(P) = 12` the formula `slots + cut + adjustment` is 12. Dibenzothiophene S,S-dioxide with
  rule 1 (`s_to_o_aromatic_C`), F = precursor: product atom set equals the ordinary precursor, so it is MERGED (no extra),
  `n_formed == 1`, `n_internal_cut == 2`, adjustment `+2`, and the merged row carries the rule-1 indicator. Nitro rule 4 on
  nitrobenzene (loss-only style: delete NO, form C-O): `n_deleted_heavy == 2`, adjustment `-2`. A rule with
  `net_formed < 0` and the ladder clamp are covered in step 3.
- Dedup: the same product from two parents gives one extra with the better parent; two matches of one rule giving one atom
  set merge; an extra equal to an ordinary row adds no row (`merged_rows`) and sets the indicator; symmetric sulfonyl matches
  (`CS(=O)(=O)c1ccccc1`) produce one extra, not two.
- Cap: relative cap `floor(frac * K)` with `K` from the kept rows; the absolute ceiling binds when smaller; ties broken by
  words; a parent-key order that prefers high-probability parents; `rewire_dropped_cap_count` and `rewire_cap_tie_count`;
  a spectrum with `K = 3` and `frac = 0.25` gets 0 extras (floor).
- Blocked parents (`parent_blocked`): no extras from blocked rows.
- Hard errors, one test each: batch/match-set count mismatch, motif atom out of range, motif element differs from the graph
  atom, unresolved cut pair, `chemical_bond_index` not sorted by spectrum, duplicate ordinary atom sets, ordinary membership
  outside its spectrum, non-converging labelling (monkeypatched sweep limit), `continuation_cuts > 0` with
  `bond_logits=None`.
- Chunking: `pair_chunk` in `{1, 7, 65536}` give identical outputs. `W` boundary molecules (61, 62, 63, 124, 125 heavy atoms,
  e.g. long alkyl chains with an aryl sulfide) run through the stage.
- Zero-extra batch: all outputs have zero leading dimension and `diagnostics` are finite.

### Step 3 `model_wiring`: constructor, forward, head, descriptor

Edits to existing files (all hunks small, all gated on `use_rewire_rules`, which defaults `False`; re-read each file before
editing; no reformatting):

**3a. `candidates/builder.py`.** Add ONE field at the end of `MassterStaticGraph` (line 73), keeping defaults last:
`rewire_matches: tuple[RewireMatchSet, ...] | None = None`, with the import. `_build_cpu_candidates`, `_merge_forced_masks`
and the SMARTS append block are untouched.

**3b. `masster_candidate_features.py`.** Add the field `rewire_edit_features: th.Tensor | None = None` as the LAST field of
`CandidateFeatureInputs` (existing positional/keyword constructions keep working) and one registry entry appended LAST in
`CANDIDATE_FEATURE_SPECS` so existing column order is unchanged:

```python
def _build_rewire_edit(inputs: CandidateFeatureInputs) -> th.Tensor:
    """n_formed, n_internal_cut, n_deleted_heavy, depth of the rule edit; zeros for ordinary rows."""
    if inputs.rewire_edit_features is None:
        raise ValueError("c_rewire_edit needs masster_rewire_rules (no rewire edit features were built)")
    return inputs.rewire_edit_features.to(inputs.dtype)
CandidateFeatureSpec(name="c_rewire_edit", width=lambda edge_feature_dim: 4, build=_build_rewire_edit)
```

Resolved only when named in `masster_candidate_features`. It is NOT listed in the v1 arms (a descriptor changes the
`input_projection` shape and RNG draws, which would make a0 differ from `i1` at initialization; a0 changes only the zero-init
residual). Add its row to the comment table in `config/template.yml` (step 4).

**3c. `masster_candidate_transformer.py`.** `CandidateIntensityTransformerHead.__init__`: new keyword `rewire_pathway_dim: int = 0`
after `rule_pathway_dim` (line 724) with the same `< 0` check; after the `rule_pathway_residual` block (lines 945-950):

```python
self.rewire_pathway_dim = rewire_pathway_dim
self.rewire_pathway_residual: nn.Linear | None = None
if rewire_pathway_dim > 0:
    with th.random.fork_rng(devices=[]):
        self.rewire_pathway_residual = nn.Linear(rewire_pathway_dim, model_dim, bias=False)
    nn.init.zeros_(self.rewire_pathway_residual.weight)
```

`forward`: new keyword `rewire_pathway_features: th.Tensor | None = None` after `rule_pathway_features` (line 1269), the same
"exactly when positive" and shape checks as the rule block (lines 1347-1358), and right after line 1386:

```python
if self.rewire_pathway_residual is not None:
    tokens = tokens + self.rewire_pathway_residual(rewire_pathway_features.to(tokens))
```

With the flag off there is no module, no parameter, no key, no RNG draw (`fork_rng` keeps the global stream untouched either
way) and no extra op in the call.

**3d. `masster_model.py`.**

*Imports:* `REWIRE_RULE_SPEC_HASHES, MASSTER_NUM_REWIRE_RULES, filter_rewire_matches` from `candidates/rewire_rules`;
`RewireExpansion, REWIRE_DIAGNOSTIC_NAMES, expand_rewire_candidates` from `masster/rewire_expansion`.

*Constructor* (after `rule_pathway_token: bool = False`, line 1148), all defaulted so every existing call is unchanged:
`use_rewire_rules: bool = False, rewire_max_extra_fraction: float = 0.5, rewire_max_extra: int = 256,
rewire_continuation_cuts: int = 0, rewire_pair_chunk: int = 65536`. Docstring entries for each. Inline guards
(`ValueError`, placed with the other constructor guards near line 1520, all behind `if use_rewire_rules:`), message names the
offending key:

- requires `unique_fragment_mode`, `fragment_selection == "mask_order"`, `proposal_selection_only`,
  `h_transfer_bound == "valence"`, `proposal_method == "cut_masks"`;
- rejects `separate_ion_mode_proposal_head` and `masster_ion_mode_proposal_loss_weight > 0` (v1: extras must not reach a proposal-side head),
  `fragment_connectivity_layers > 0`, `intensity_candidate_graph`, `return_base_spectrum`, `use_base_probability_feature`
  (already excluded by selection-only; keep the explicit message);
- types and ranges: `use_rewire_rules` bool; `rewire_max_extra_fraction` finite and `> 0`; `rewire_max_extra` int `>= 1`
  (not bool); `rewire_continuation_cuts` int `>= 0`; `rewire_pair_chunk` int `>= 1`;
- `"c_rewire_edit" in candidate_features` requires `use_rewire_rules` (checked also when the flag is off).

Store `self.use_rewire_rules` (always, a plain attribute; it is a constructor argument, not a batch-path hparam read) and the
four numeric attributes only when on.

*Module construction* (head call, line ~2016): `rewire_pathway_dim=MASSTER_NUM_REWIRE_RULES if use_rewire_rules else 0`.

*Buffer and pre-hook* (end of `__init__`, beside the SMARTS buffer at line 2032):

```python
if use_rewire_rules:
    self.register_buffer("rewire_rule_name_hashes", th.tensor(REWIRE_RULE_SPEC_HASHES, dtype=th.long))
    self._register_load_state_dict_pre_hook(self._restore_rewire_rule_vocabulary)
```

`_restore_rewire_rule_vocabulary(self, state_dict, prefix, *args)`, a copy of `_restore_smarts_rule_vocabulary` (line 2071):
key `prefix + "rewire_rule_name_hashes"` missing raises `ValueError("this checkpoint has no rewire rule list; it was not
trained with masster_rewire_rules")` (an old checkpoint switched on gives a clear error; the buffer is the only place the
check lives); saved must be a strict prefix of `REWIRE_RULE_SPEC_HASHES` else `ValueError` ("the checkpoint's rewire rules are
not a prefix of REWIRE_RULE_VOCABULARY; a rule was reordered, renamed or its SMARTS/edit/precursors changed"); set the buffer
to the saved hashes (so `num_known_rules = rewire_rule_name_hashes.numel()` ignores rules appended later); zero-pad axis 1 of
`intensity_head.rewire_pathway_residual.weight` for a shorter prefix (offset 0). A model built WITHOUT the flag loading a
checkpoint that has the buffer or the residual gets the standard strict-load "unexpected key" error; this is the intended
"off model, on checkpoint" behavior and is tested.

*Valence adjustment, two signatures* (defaulted so every caller is unchanged):
`_transferable_hydrogen_shifts(..., fragment_formed_bond_counts=None, fragment_valence_adjustment: th.Tensor | None = None)`
and `_fragment_masses_and_valid_channels(..., fragment_cut_counts=None, fragment_formed_bond_counts=None,
fragment_valence_adjustment=None)` which forwards it. In `_transferable_hydrogen_shifts`, immediately after the existing
`fragment_formed_bond_counts` block (line 2457) and before `max_gain = th.minimum(max_gain, valence_gain)`:

```python
            if fragment_valence_adjustment is not None:
                # A rewire product's own edit: -2 per formed bond inside it (both endpoints spend a slot),
                # +2 per rule cut inside it (both endpoints free one). Bonds to deleted atoms are already
                # in fragment_cut_counts; slots are summed over the fragment's own atoms only.
                valence_gain = valence_gain + fragment_valence_adjustment.to(fragment_slots)
```

The existing `max_gain.clamp_min(0)` on the return line already makes a negative gain harmless. Derivation, for atom set `P`,
deleted set `D`, rule cuts `C`, formed bonds `Fm` (all inside `P`): `valence_gain(P) = slots(P) + |Boundary(P)| - 2|Fm| + 2|C inside P|`,
`slots(P)` summed over P only (D's own valence never enters, so a P-D bond contributes once, through the cut count),
`Boundary(P)` = precursor bonds with exactly one endpoint in `P`. Hence the adjustment is `-2 * (n_formed - n_internal_cut)`
and may be positive.

*Stage call and append, in `forward`.* The insertion point is immediately after the post-keep renormalization (current
lines 3747-3755) and before `_candidate_atom_dropout` (line 3756). Surrounding code as it stands:

```python
            fragment_log_probabilities = fragment_log_probabilities[fragment_keep]
            num_fragments = int(fragment_log_probabilities.shape[0])
            retained_proposal_log_normalizers = _segment_logsumexp(
                fragment_log_probabilities, fragment_batch_indices, batch_size,
            )
            base_fragment_log_probabilities = (
                fragment_log_probabilities
                - retained_proposal_log_normalizers[fragment_batch_indices]
            )
            # >>> INSERT HERE <<<
            pooled_atom_indices, pooled_atom_fragment_indices = self._candidate_atom_dropout(
```

Insert (the only forward hunk in this region; `fragment_keep` and `fragment_formed_bond_counts` are still in scope):

```python
            rewire_rows = None
            if self.use_rewire_rules:
                rewire_rows = self._append_rewire_rows(
                    static_graph=static_graph,
                    mol_pyg=mol_pyg,
                    precursor_type_indices=precursor_type_indices,
                    bond_logits=bond_logits,
                    fragment_atom_indices=fragment_atom_indices,
                    atom_fragment_indices=atom_fragment_indices,
                    fragment_boundary_bond_indices=fragment_boundary_bond_indices,
                    boundary_fragment_indices=boundary_fragment_indices,
                    fragment_batch_indices=fragment_batch_indices,
                    fragment_cut_counts=fragment_cut_counts,
                    ion_mzs=ion_mzs,
                    valid_channels=valid_channels,
                    fragment_rule_indicators=fragment_rule_indicators,
                    fragment_log_probabilities=fragment_log_probabilities,
                    base_fragment_log_probabilities=base_fragment_log_probabilities,
                    parent_blocked=(
                        None if fragment_formed_bond_counts is None
                        else fragment_formed_bond_counts[fragment_keep] > 0
                    ),
                    batch_size=batch_size,
                )
                fragment_atom_indices = rewire_rows["fragment_atom_indices"]
                atom_fragment_indices = rewire_rows["atom_fragment_indices"]
                fragment_boundary_bond_indices = rewire_rows["fragment_boundary_bond_indices"]
                boundary_fragment_indices = rewire_rows["boundary_fragment_indices"]
                fragment_batch_indices = rewire_rows["fragment_batch_indices"]
                fragment_cut_counts = rewire_rows["fragment_cut_counts"]
                ion_mzs = rewire_rows["ion_mzs"]
                valid_channels = rewire_rows["valid_channels"]
                fragment_rule_indicators = rewire_rows["fragment_rule_indicators"]
                fragment_log_probabilities = rewire_rows["fragment_log_probabilities"]       # extras: 0 filler, unread
                base_fragment_log_probabilities = rewire_rows["base_fragment_log_probabilities"]  # extras: -inf
                num_fragments = int(fragment_batch_indices.shape[0])
```

`_append_rewire_rows` is a new method of `MassterModel` (place it after `_unique_component_memberships`, line ~3022). It:

1. Builds the per-spectrum filtered sets:
   `if static_graph is None or static_graph.rewire_matches is None: raise ValueError("rewire-enabled models require static_graph.rewire_matches; use MassterPL with mol_smiles or supply matched rewire sets")`;
   `precursor_indices = precursor_type_indices.detach().cpu().reshape(-1).tolist()` (the same host copy
   `_eligible_smarts_templates` makes); `match_sets = tuple(filter_rewire_matches(ms, precursor_type=self.precursor_types[i],
   num_known_rules=self.rewire_rule_name_hashes.numel()) for ms, i in zip(...))`, with a length check against `batch_size`.
2. Defines `channel_fn` as a closure over `mol_pyg` and `precursor_type_indices`:
   `lambda ai, afi, fbi, n, cuts, adj: self._fragment_masses_and_valid_channels(mol_pyg, ai, afi, fbi, n, precursor_type_indices, cuts, None, adj)`.
3. Calls `expand_rewire_candidates(...)` with `parent_log_probabilities=base_fragment_log_probabilities.detach()`,
   `bond_logits=bond_logits.detach()` (None unless `rewire_continuation_cuts > 0`), `host_chemical_bond_index=static_graph.chemical_bond_index`,
   `num_rules=MASSTER_NUM_REWIRE_RULES` and the stored cap attributes.
4. Concatenates, extras AFTER every ordinary row (ordinary indices unchanged), with `n_ord = num_fragments` of the input:
   - `fragment_atom_indices = cat([ordinary, ex.extra_atom_indices])`;
     `atom_fragment_indices = cat([ordinary, n_ord + ex.extra_atom_extra_indices])`;
   - `fragment_boundary_bond_indices = cat([...])`; `boundary_fragment_indices = cat([ordinary, n_ord + ex.extra_boundary_extra_indices])`;
   - `fragment_batch_indices = cat([ordinary, ex.extra_batch_indices])`; `fragment_cut_counts = cat([ordinary, ex.extra_cut_counts])`;
   - keep `ordinary_valid_channels = valid_channels` (the pre-merge tensor, returned in the dict). Then `valid_channels = valid_channels.clone()`
     and `valid_channels[ex.merged_rows] |= ex.merged_valid_channels` when `merged_rows.numel() > 0` (C5; this merged tensor feeds the
     intensity head only); then, ONLY when `ex.num_extra > 0`, `ion_mzs = cat([ordinary, ex.extra_ion_mzs])` and
     `valid_channels = cat([ordinary, ex.extra_valid_channels])` (an empty expansion carries `(0, 0, 0)` channel tensors that
     `torch.cat` cannot join with `(n, M, H)`);
   - `fragment_rule_indicators`: if not None, `cat([ordinary, zeros(E, MASSTER_NUM_RULES, dtype=ordinary.dtype, device=ordinary.device)])`
     (SMARTS indicators of extras are zero); all fillers use `ordinary.new_zeros` / `ordinary.new_full` so dtype and device follow
     the ordinary tensor under autocast/bf16, and the rewire indicator and edit-feature tensors are built in the head's dtype;
   - `fragment_log_probabilities = cat([ordinary, ordinary.new_zeros(E)])` (a FINITE filler: this is the tensor handed to the
     intensity head, and no `-inf` may reach the head); `base_fragment_log_probabilities = cat([ordinary, ordinary.new_full((E,), -inf)])`
     (C1; `-inf` lives only in the separate base tensor). BEFORE building, grep the head's `forward` for every use of its
     `fragment_log_probabilities` argument (sorting, bucketing, base-probability feature) and cite them in the code comment; the
     filler-invariance test below is kept as the guard;
   - also returns `ordinary_valid_channels` extended with all-False rows for extras (so a per-channel ordinary mask aligns with the
     extended `valid_channels`), and `fragment_rewire_indicators (n_ord + E, R)` = zeros, ordinary merged rows set from `ex.merged_rule_indicators`,
     extras from `ex.extra_rule_indicators`; `fragment_rewire_edit_features (n_ord + E, 4)` likewise;
     `fragment_rewire_parent (n_ord + E,)` long, `-1` for ordinary rows, `ex.extra_parent` for extras;
     `rewire_is_extra (n_ord + E,)` bool; `diagnostics = ex.diagnostics`.
5. A batch with `num_extra == 0` leaves every ordinary tensor unchanged bit for bit (the other concatenations join
   zero-row tensors of matching trailing shape, which is a value no-op).

The method returns a plain dict (so the forward hunk stays a list of assignments). After the append the code below runs
unchanged on the extended tensors: `_candidate_atom_dropout`, pooling, boundary embeddings, `CandidateFeatureInputs`,
`sink_rows[fragment_batch_indices]`, `context[fragment_batch_indices]`, `fragment_indices = arange(num_fragments)`, the head
call, the ion-mode head.

*Two keyword additions in the same branch, each in the existing call:* in `CandidateFeatureInputs(...)` (line 3799):
`rewire_edit_features=None if rewire_rows is None else rewire_rows["fragment_rewire_edit_features"],`; in
`self.intensity_head(...)` (line 3866), after `rule_pathway_features=(...)`:
`rewire_pathway_features=None if rewire_rows is None else rewire_rows["fragment_rewire_indicators"],`.
The legacy (non-unique) branch is unreachable with the flag (guard), so `rewire_rows` is only defined in the unique branch;
initialize `rewire_rows = None` before `if self.unique_fragment_mode:` (line ~3529) so the later reads are always defined.

*Step 4 of forward (line 4125 onward).* No edit: `chemically_valid_fragments` drops invalid rows from the extended (merged) set (extras are
all valid by construction; an ordinary row invalid in the control may become valid through a merge), order is preserved so ordinary valid rows stay first and extras follow. Because the merge can add validity, the PROPOSAL quantities are NOT taken from this tensor: the base column of step 4 sets `base_fragment_log_probabilities = -inf` for ordinary rows whose PRE-merge channels were all invalid (so the base normalizer at lines ~4179-4190 equals the control's), and the proposal block below masks with `ordinary_valid_channels`.
`fragment_scores = th.stack((fragment_log_probabilities, base_fragment_log_probabilities))` normalizes the intensity column over
ordinary plus extras (intended) and the base column over ordinary rows only, because `exp(-inf - max) = 0` is an exact zero
(`_segment_logsumexp`). Extras' base value stays `-inf` after the subtraction.

*Output hunks (only when `rewire_rows is not None`, so with the flag off the output dict is unchanged):*

- Before the `proposal_selection_only` block (line 4611) compute, once:
  `is_extra = rewire_rows["rewire_is_extra"][chemically_valid_fragments]` (bool, aligned with the final fragments) and
  `candidate_is_ordinary = ~is_extra[full_candidate_fragment_idxs] & ordinary_valid_channels.flatten()[full_flat_valid]` (per CHANNEL,
  pre-merge), and `row_is_proposal = ~is_extra & ordinary_valid_channels.flatten(1).any(1)[chemically_valid_fragments]` (per ROW,
  pre-merge validity). Because ordinary valid rows precede extras, ordinary fragment indices are unchanged for rows valid in the control;
  a merge-validated row that was invalid in the control is excluded from the proposal view, so the block becomes:
  `"proposal_candidate_mzs": full_candidate_mzs[candidate_is_ordinary]`,
  `"proposal_candidate_fragment_idxs": full_candidate_fragment_idxs[candidate_is_ordinary]` (remapped through the control's row
  numbering by the same cumulative count over `row_is_proposal`),
  `"proposal_fragment_logprobs": base_fragment_log_probabilities[row_is_proposal]`,
  `"proposal_fragment_batch_idxs": fragment_batch_indices[row_is_proposal]`.
  (Use a small helper `_ordinary_proposal_view(...)` so the block is a one-line change; with the flag off the original lines run.)
- `ion_h_fragment_proposal_logprobs` stays the full extended tensor (extras `-inf`). `-inf` is a NaN hazard where multiplied by zero
  (`masster_pl.py` ~486 `coverage_loss + 0.0 * fragment_logprobs.sum()` is safe only because it receives the SLICED view). The builder
  lists every consumer of the extended tensors (`_channel_support_metrics` ~1900, `_ion_mode_support_loss` ~1925, cross-EV, `pred_*`,
  `base_pathway_log_probabilities` at ~4285) and either proves it only `.exp()`s the value or passes the sliced view. The full-loss
  NaN test below covers every one.
- New outputs: `"rewire_is_extra"` (bool per fragment, aligned with `ion_h_fragment_batch_idxs`),
  `"rewire_candidate_is_extra_by_cut"` = `is_extra[full_candidate_fragment_idxs]` (aligned with `candidate_mzs_by_cut`), and the
  `REWIRE_DIAGNOSTIC_NAMES` tensors (per spectrum). Under `return_annotation`: `"fragment_rewire_fragment_idxs"` and
  `"fragment_rewire_rule_idxs"` (0-based vocabulary index) from `th.nonzero(fragment_rewire_indicators[chemically_valid_fragments])`,
  and `"fragment_rewire_parent_idxs"` (per final fragment; `-1` for ordinary rows or a parent that was itself dropped as invalid,
  remapped through `old_to_new_valid`).

Parity statement for the flag ON under a0 (precise): (i) molecules with no matching rule instance are bit-equal in every
output; (ii) `proposal_candidate_*`, `proposal_fragment_*`, `_proposal_coverage_loss`, `bond_logits` and its gradient under a
proposal-only backward are bit-equal for ALL molecules, including molecules whose merged rows gain channels or validity (proposal
view from pre-merge validity); parameter gradients of the shared encoder under the full loss are NOT claimed equal; (iii) the intensity outputs of a matching molecule differ by construction (extras join its
softmax; a merged row's indicator and validity change). Per-spectrum independence of the batched ops (segment sums via
`index_add`, the head's argsort by batch index) is a tested property.

Documented consequences, each with a test: with `masster_candidate_atom_dropout > 0` and extras present the `th.rand` draw is
shaped by the row count, so the random stream differs from the control (unchanged when off); `light_side_token`/`c_heavy_side`
judge the lighter side by the mass of `P` (deleted atoms land in the complement, which IS the neutral loss); `c_cut_bonds`,
`c_cut_adjacency` read `Boundary(P)`, which includes bonds to deleted atoms (two bonds into a deleted S give a lost-side
adjacent pair, the extrusion signature); the memorization sink indexes by each extra's own spectrum.

Tests, new `tests/test_masster_rewire_model.py`. Helpers: reuse `_build_batch`/`_build_model` from `tests/test_masster_model.py`
(`**model_kwargs` flows to the constructor) and the SMARTS-pathway `_static_graph` pattern, extended with
`rewire_matches=tuple(build_rewire_matches(Chem.MolFromSmiles(s), max_matches_per_rule=8) for s in smiles)`. Fixture kwargs
`_REWIRE_KWARGS = dict(unique_fragment_mode=True, proposal_selection_only=True, use_base_probability_feature=False,
h_transfer_bound="valence", max_masks=8, max_enumerated_masks=32, use_rewire_rules=True, rewire_max_extra_fraction=1.0)`.

- OFF identity: `use_rewire_rules` unset builds a model whose `state_dict()` keys contain no `rewire`, whose
  `intensity_head.rewire_pathway_residual is None`, with the same parameter values as a model built before the change (assert
  against a model built with the new arguments explicitly `False`), and whose forward outputs equal the same model's outputs
  with `static_graph` carrying `rewire_matches=None`. Run the full existing `tests/test_masster_model.py`,
  `test_masster_smarts_pathway.py`, `test_masster_pl.py` unchanged (they are the OFF guard).
- ON with no matching molecule (`CCO`, `CO`): every output key of the OFF model equals the ON model's (the ON model adds the `rewire_*` keys) (same weights loaded with
  `strict=False` for the new buffer and zero-init residual), `th.equal` for floats at `attention_type: "none"`; at the default
  attention allclose with `atol=0, rtol=0` first, documenting if bucket composition forces `1e-6`.
- Placement: output unchanged when the extras' filler `fragment_log_probabilities` is changed to another finite value
  (monkeypatch `_append_rewire_rows`); proposal outputs of a matching molecule bit-equal to the OFF model's (C1), and
  `bond_logits.grad` bit-equal after `loss = proposal-only` backward; mixed batch `[sulfone, ethanol]` equals the per-spectrum
  runs; `fragment_indices`, `context[fragment_batch_indices]`, `sink_rows[fragment_batch_indices]` consistent (sink model);
  extras follow every ordinary row and `rewire_is_extra` marks exactly them.
- Gradients: a0 intensity loss reaches `rewire_pathway_residual.weight` only in columns of the matched rules (zero elsewhere),
  and the descriptor weights when `c_rewire_edit` is listed; at initialization the residual is exactly zero so the ON model
  starts as the control.
- Valence hand values (from step 2) fed through `_transferable_hydrogen_shifts` with synthetic `component_counts` so the
  complement hydrogens are known: diphenyl sulfone extrusion `valence_gain == 12` (14 without the adjustment); the ladder bound
  `min(complement_H, gain)`; a merged dibenzothiophene-dioxide row gets `+2` and its valid channels are the OR of the ordinary
  row's and the extra call's; a hand case with `net_formed < 0` and `valence_gain` forced negative gives ladder bound 0
  (shift 0 allowed, positive shifts not): pins the existing `clamp_min(0)`.
- Merge: an extra equal to an ordinary row adds no row, sets that row's `fragment_rewire_indicators` and raises its valid
  channels (assert a channel that is invalid for the ordinary row alone becomes valid FOR THE INTENSITY HEAD). A case where a merged
  row gains a channel AND an ordinary row invalid in the control becomes newly valid: assert `proposal_candidate_*`,
  `proposal_fragment_*`, the proposal coverage loss and `bond_logits.grad` still equal the control bit for bit.
- Full-loss finiteness: full training loss plus every enabled metric on a sulfone batch under `th.autograd.set_detect_anomaly(True)`;
  assert all parameter grads finite, and test each consumer of the extended tensors listed above (channel-support metrics,
  ion-mode loss, cross-EV off, `pred_*`).
- Candidate-set features: `c_heavy_side`/`light_side_token` with `P` above and below half the precursor mass; `c_cut_bonds` and
  `c_cut_adjacency` against hand values for a sulfone (two cut bonds sharing the deleted S: one lost-side adjacent pair) and a
  nitro case; `_candidate_atom_dropout` keeps descriptors and masses on the true set; `c_rewire_edit` hand row
  `[1, 0, 3, 0]` for the sulfone extra, zeros for ordinary rows.
- Guards, one `pytest.raises(ValueError, match=...)` each: `marginal` selection, `connected_fragment_beam`,
  `fragment_connectivity_layers > 0`, `intensity_candidate_graph`, `separate_ion_mode_proposal_head` and
  `masster_ion_mode_proposal_loss_weight > 0` (both guarded in v1: the proposal-side ion-mode head would otherwise feed extras'
  gradients into a proposal head, contradicting decision 2; `separate_ion_h_support_head` is intensity-side and allowed), missing `proposal_selection_only`,
  `h_transfer_bound != "valence"`, missing unique mode, `return_base_spectrum`, bad fraction/ceiling/chunk/continuation values,
  `c_rewire_edit` without the flag, `static_graph.rewire_matches is None` at forward.
- Checkpoint: save and reload with the flag (equal); a state dict whose `rewire_rule_name_hashes` is a strict prefix loads with
  the residual zero-padded and later rules ignored (rules >= prefix length produce no extras); reordered, renamed or
  edit-changed hashes raise; a checkpoint without the buffer raises the clear error when the flag is on; the flag-off model
  rejects a flag-on checkpoint (unexpected keys); an old checkpoint's hparams plus the template defaults (emulated with
  `deep_update(load_config(...), saved)` of a dict lacking the keys) give a flag-off model.
- Coexistence: `use_smarts_templates=True` and rewire together on a sulfonamide: template-side rows (nonzero formed count) get
  no extras (`parent_blocked`), other rows do, outputs finite, `fragment_rule_indicators` has `E` zero rows appended, no index
  collision with the SMARTS append.
- `share_proposal_across_ev`: two views of one molecule share extras (same atom sets, same order); without sharing no
  cross-spectrum leakage.
- Multi-adduct: a batch with `[M+H]+` and `[M-H]-` spectra of one azo molecule: rule 13 fires only for `[M-H]-`.
- CPU/CUDA (marked, `@pytest.mark.skipif(not th.cuda.is_available(), ...)`; the machine has one 8 GB RTX 2070 SUPER shared with
  another session, so check `nvidia-smi` and keep batches tiny): in `tests/test_masster_cuda_full_candidates.py`, for the same
  molecules the CPU and CUDA backends give the identical unique-fragment order, keep mask, extras (atom sets, parents, order
  after the cap) and forward logits/losses/gradients within the existing tolerances; include symmetric molecules (two equivalent
  sulfonyl matches) to exercise the integer tie-break.

### Step 4 `pl_config_export`: config keys, PL threading, metrics, export, arm configs

**4a. `config/template.yml`** (beside `masster_smarts_templates`, line 836; ONE template, so each new masster key gets a
one-line first comment and is read only when `model_type == "masster"`; none is a data path, so `scripts/setup_job_cc.py`
keyword lists are unchanged):

```yaml
masster_rewire_rules: False  # Apply admitted rewire rules (delete/cut/form edits) to every kept fragment; extras added to the pool.
# docs/plans/masster/masster_rewire_rules_implementation.md. Requires masster_unique_fragment_mode, masster_fragment_selection
# mask_order, masster_proposal_selection_only, masster_h_transfer_bound valence, cut_masks proposals; rejects the connectivity
# encoder, intensity_candidate_graph and base spectrum outputs. False builds nothing and adds no state_dict key.
masster_rewire_max_extra_fraction: 0.5  # Extras per precursor <= floor(this * kept ordinary fragments); a modelling knob (G0b sweeps 0.25/0.5/1.0).
masster_rewire_max_extra: 256  # Absolute per-precursor ceiling on extras; a memory bound, not a modelling choice.
masster_rewire_max_matches_per_rule: 8  # Matches kept per rule per precursor after edit-key dedup, deterministic order; the rest are counted.
masster_rewire_continuation_cuts: 0  # Depth-1 top-k precursor-bond cuts per product by bond logit; 0 = products only.
masster_rewire_pair_chunk: 65536  # (fragment, match) pairs processed per stage chunk; bounds memory, not the result.
masster_rewire_null: False  # Replace each real match by random same-shape matches (the gate's topology null); needs masster_rewire_rules.
```

and a `c_rewire_edit  4  n_formed, n_internal_cut, n_deleted_heavy, depth of a rewire product (zeros otherwise)` row in the
`masster_candidate_features` comment table (line ~855). Run `tests/test_config_templates.py`, `tests/test_setup_job_cc.py` and
`tests/test_legacy_hparam_migration.py` unchanged; add a test that `load_config` returns the seven keys with these defaults and
that the first comment line of each is non-empty (the setup parser reads only the first line).

**4b. `masster_pl.py`.**

- Import `build_rewire_matches, RewireMatchSet` and (step 5) the null builder lazily inside the cached function.
- Next to `_cached_masster_templates` (line 153):

```python
@lru_cache(maxsize=100_000)
def _cached_masster_rewire_matches(smiles: str, max_matches_per_rule: int, null: bool) -> RewireMatchSet:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid molecule SMILES for rewire matching: {smiles}")
    if null:
        from fragnnet.models.masster.candidates.rewire_null import build_null_rewire_matches
        return build_null_rewire_matches(mol, max_matches_per_rule=max_matches_per_rule)
    return build_rewire_matches(mol, max_matches_per_rule=max_matches_per_rule)
```

- Constructor threading (model call, lines 1195-1224): `use_rewire_rules=self.hparams.masster_rewire_rules,
rewire_max_extra_fraction=..., rewire_max_extra=..., rewire_continuation_cuts=..., rewire_pair_chunk=...` read straight from
`self.hparams` (no `.get(default)`); cache `self._rewire_enabled = bool(self.hparams.masster_rewire_rules)` and the two cache
arguments at init so the batch path reads attributes, not new hparams.
- Hparam guards (a `_check_rewire_params()` method called from `__init__` near the other `_check_*`, line ~1586), `ValueError`:
  every key's type and range as in 3d; `masster_rewire_null` requires `masster_rewire_rules`; `cross_ev_fragment_loss_weight > 0`
  rejected with rewire (v1; extras have no cross-view identity); `masster_compile` already rejected. Keys with the flag off are
  only type-checked, never used.
- `_build_static_graph` (line 2454): add `rewire_matches=(tuple(_cached_masster_rewire_matches(str(s), self._rewire_max_matches,
self._rewire_null) for s in batch["mol_smiles"]) if self._rewire_enabled else None)`.
- Standalone-inference fallback in `forward` (line 2391): extend the existing condition to
  `(self.hparams.masster_smarts_templates and (static_graph is None or static_graph.smarts_templates is None)) or (self._rewire_enabled and (static_graph is None or static_graph.rewire_matches is None))`
  and rebuild with `_build_static_graph`.
- Loss diagnostics: the tuple `REWIRE_DIAGNOSTIC_NAMES` imported from `rewire_expansion` (do not duplicate the names);
  in `loss_fn` (line ~2017) add `**{name: kwargs[name].detach().to(prediction_reference) for name in REWIRE_DIAGNOSTIC_NAMES}` to
  `loss_values` ONLY when `self._rewire_enabled`, and extend `loss_names` (line ~2126) with the same names under the same
  condition (the comment above `loss_names` says each name must be a key of `loss_values`).
- Coverage split (validation, only with rewire on AND `masster_candidate_coverage_metrics`): in `_setup_metric_fns`
  (line 2208) register `candidate_wrecall_ordinary` and `candidate_wrecall_rewire_gain`, append
  `"rewire_candidate_is_extra_by_cut"` to `extra_metric_args`, and in `metric_fn` compute both with the existing
  `_candidate_weighted_recall` (the call already used for `pool_candidate_wrecall`): ordinary = recall over
  `candidate_mzs_by_cut[~is_extra & ordinary_channel_mask]` (the same per-channel PRE-merge mask as the proposal view, exported as
  `rewire_candidate_is_ordinary_by_cut`); gain = recall over all candidates minus ordinary, so merge-gained channels count as rewire gain. Rearrangement-specific recall by rule
  and NCE bucket is an analysis script, not a training metric.
- Output cleanup and splitting: grep the key lists around `masster_pl.py:2448` (which names `cross_ev_fragment_logprobs` explicitly)
  and the inference batch splitting / DDP gather code, and add the new per-fragment keys (`rewire_is_extra`, `fragment_rewire_*`,
  `rewire_candidate_is_*_by_cut`) and per-spectrum diagnostics wherever per-fragment or per-batch keys are enumerated. An
  inference-path test (`return_annotation` plus `return_rule_provenance` on a multi-spectrum batch where one spectrum has no match)
  checks the exported node rows are aligned.

**4c. Export** (`return_rule_provenance` gains the rewire column; refused for non-MASSter models by the existing check):

- `masster_export.py`: `masster_annotation_tables(..., fragment_rule_idxs=None, fragment_rewire_fragment_idxs=None,
  fragment_rewire_rule_idxs=None)` (given together or both `None`, else `ValueError`), new field
  `MassterAnnotationTables.node_rewire_rules: np.ndarray | None = None`, built with the SAME node mapping as `node_rules`
  (`fragment_to_node`, line ~217): a node is an atom set, so it carries the union of rewire rule names of every fragment merged
  into it, `";"`-joined in `REWIRE_RULE_VOCABULARY` order, `""` for a node no rewire rule produces. An extra and an ordinary
  fragment with the same atom set therefore give ONE node carrying both its SMARTS and its rewire names (two columns).
- `inference/export_columns.py`: `ColumnSpec("pred_node_rewire_rules", "node", flag="return_rule_provenance")` after
  `pred_node_rules`; docstring line.
- `export_adapter.py`: default `export_node_rewire_rules(self, b_idx, node_idxs, annotation)` returning `np.nan`.
- `inference/batch_rows.py` (line ~437): `if return_rule_provenance: row_d["pred_node_rewire_rules"] = export_adapter.export_node_rewire_rules(b_idx, b_node_idxs, annotation)`.
- `masster_pl.py`: `export_node_rewire_rules` mirroring `export_node_rules` (line 2639) on `annotation.node_rewire_rules`;
  `postprocess_export_peaks` passes `fragment_rewire_fragment_idxs=batch_result.get("fragment_rewire_fragment_idxs")` and
  `fragment_rewire_rule_idxs=...` when `return_rule_provenance` (the keys are absent with the flag off, so the column is NaN
  and `--drop_none_cols` removes it). `scripts/run_inference.py` help text for `--return_rule_provenance` mentions the rewire
  column.
- The parent id is NOT exported as a column (C7); `fragment_rewire_parent_idxs` is returned under `return_explainability_tensors`.

**4d. Arm configs**, new directory `config/masster/mh_rewire_after_selection_2026_10_15/` (the plan said `..._2026_10_14`; the
creation date is today). Each is `config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml` (which already sets
`masster_proposal_selection_only: true`, `masster_h_transfer_bound: valence`, `masster_fragment_selection` default `mask_order`,
`masster_unique_fragment_mode: true`, `masster_candidate_backend: cuda`, 64 epochs, seed 397623) plus the keys below and W&B
naming only, with the header block of the i1 file kept (question, control `i50ig7rf`, read, gates) adapted per arm:

| File | Added keys |
| --- | --- |
| `a0_rewire_ob_obf.yml` | `masster_rewire_rules: true`, `masster_rewire_max_extra_fraction: <chosen from G0b; template default 0.5 until then>` |
| `a0d1_rewire_cont4_ob_obf.yml` | a0 + `masster_rewire_continuation_cuts: 4` |
| `a0null_rewire_null_ob_obf.yml` | a0 + `masster_rewire_null: true` |

None lists `c_rewire_edit` (keeps a0 a pure residual addition). The submit-script maps
(`scripts/manual_slurm/fir/submit_masster_mh_ft_memorization_route.sh` or a sibling: arm name, job name, config and stage maps,
usage string) are done by the main session when scheduling; this spec does not edit them.

Tests, new `tests/test_masster_rewire_pl_export.py` and an extension of `tests/test_inference_masster_annotation.py`:

- `MassterPL(**config)` with the keys (pattern of `test_rule_pathway_token_pl_threads_hparam`) threads every value to the model
  (`model.use_rewire_rules`, caps, residual width 17); with the keys at template defaults the PL has no rewire module and
  `_build_static_graph(...).rewire_matches is None`; `masster_rewire_null` without the rules raises; cross-EV loss weight with
  rewire raises.
- `_build_static_graph` fills one `RewireMatchSet` per spectrum; the lru cache returns the same object for repeated SMILES;
  the standalone `forward` fallback builds the sets when the sidecar lacks them.
- Old-checkpoint round trip: save an i1-style PL checkpoint hparams dict WITHOUT the seven keys and load it through the production
  loader (`get_config_from_ckpt` plus `apply_inference_config`, the path the G0 script uses) and through
  `MassterPL.load_from_checkpoint`; assert the flag is off, no rewire module exists, and the state dict is unchanged. A FraGNNet config
  round-trips with the new keys present and unread, and `scripts/setup_job_cc.py` ignores them. The `masster_rewire_*` keys are added
  to whatever test asserts that every template key has a first-line comment.
- A short `MassterPL` training step (tiny config, sulfone batch) with the flag on: loss finite, the diagnostics appear in the
  logged metric names, `rewire_extra_count > 0` for the sulfone, `candidate_wrecall_rewire_gain >= 0`.
- Export: with merged rows (an extra and an ordinary fragment of one atom set) one node carries both `pred_node_rules` and
  `pred_node_rewire_rules` entries; `pred_node_rewire_rules` is present in the column list exactly when
  `return_rule_provenance`; NaN with the flag off; a non-MASSter adapter refuses `return_rule_provenance` (existing test) and
  the default `export_node_rewire_rules` returns NaN; the exporter's `node_cumsizes` offset lookup is reused.

### Step 5 `null_arm`: the topology null as production code

New file `src/fragnnet/models/masster/candidates/rewire_null.py`. The scripts directory is not importable from `src`, so the
gate's logic is reproduced here and pinned to the gate by a parity test.

```python
def topology_null_smarts(rule: RewireRule) -> str                      # verbatim port of the gate's topology_null_smarts
def null_pattern(rule: RewireRule) -> Chem.Mol                          # MolFromSmarts(rule.null_smarts or topology_null_smarts(rule)); compiled once per rule (import-time cache)
def null_seed(canonical_smiles: str, rule_name: str) -> int             # zlib.crc32(f"{canonical_smiles}|{rule_name}".encode())
def build_null_rewire_matches(mol: Chem.Mol, *, max_matches_per_rule: int) -> RewireMatchSet
```

`topology_null_smarts(rule)` is a copy of `debug_masster_rewire_rule_gate.topology_null_smarts` (and its `_is_terminal_query`)
reading `rule.smarts`, `rule.delete`, `rule.cut`, `rule.form` instead of a dict: elements and bond orders wildcarded
(`~`), terminal query atoms (`X1`/`D1`) stay terminal (`[*;D1:m]`), non-terminal stay non-terminal (`[*;!D1:m]`), deleted
terminal atoms attached to a deleted atom are dropped from the pattern. Expected auto nulls (checked in tests against the
gate's output for all 17 rules, equal strings): rule 0 `[*:1]~[!D1:2]~[*:5]`, rule 1 `[*:1]~[!D1:2]~[D1:3]`, rule 4
`[*:1]~[!D1:2]~[D1:4]`, rules 5-7 and 10 `[*:1]~[!D1:2]~[*:3]`.

`build_null_rewire_matches(mol, ...)` per rule, in vocabulary order:

1. `real = raw_rule_matches(mol, real_pattern)`; if empty, no null for that rule (the gate's `if not real: continue`).
2. `maps = [a.GetAtomMapNum() for a in null_pattern.GetAtoms()]`; `real_atoms = {frozenset(m[k] for k in maps) for m in real}`;
   `pool = [m for m in raw_rule_matches(mol, null_pattern) if frozenset(m.values()) not in real_atoms]` (real sites excluded).
   The null count is the number of DEDUPED real matches of the rule (after the canonical edit-key dedup of step 1, before truncation),
   NOT the raw count, so a0null is size-matched to a0. The null pool is deduped on the same canonical edit key BEFORE choosing.
   The gate-parity draw (raw lists, raw counts) lives in a separate function `gate_null_matches` used only by the parity test.
3. If `pool` is empty, no null for that rule. Else `rng = np.random.default_rng(null_seed(Chem.MolToSmiles(mol), rule.name))`,
   `picked = rng.choice(len(pool_deduped), min(n_real_deduped, len(pool_deduped)), replace=False)` (the gate-parity function reproduces the
   FIRST of the gate's three draws on the raw lists with the same generator and key construction when the seed key is passed in).
   The draw is deterministic for a fixed input SMILES string (pool order follows RDKit atom order of the input); it is NOT invariant
   to atom permutation of the input. If order independence is ever wanted, sort the pool by `Chem.CanonicalRankAtoms` first.
4. Each picked null match becomes a `RewireMatch` through the same converter as the real path: `motif_atoms` = the null
   dict's values (the null pattern's mapped atoms only), `deleted` = delete maps PRESENT in the dict (deleted terminals are not
   in the null pattern; their neighbour is, and the terminal atom stays as an isolated one-atom component that is never a
   product), cut and form pairs through the dict, canonical dedup and the same truncation. NOTE: for rules that delete terminal atoms
   (rules 0, 4, 9, 11, 12, ...) the gate's null drops the deleted terminals from the pattern, so a null edit deletes only the
   central atom: its `deleted` set and `n_deleted_heavy` (e.g. 1 versus 3 for rule 0) differ from the real edit's, and the
   product is the same piece. This is faithful to the gate and kept; report it next to the a0null result so a null-vs-real gap is
   not over-read (the null cannot test the neutral-loss composition, only the topology). Cut pairs of a null match are real
   bonds (the null keeps the SMARTS bonds between touched atoms), so the stage's cut-resolution hard error stays valid.
   Rules whose precursor filter excludes the spectrum are removed afterwards by `filter_rewire_matches`, exactly as real rules.

Same per-rule match counts (deduped real count, or the pool size when smaller), same vocabulary columns, only the chemistry differs; G0b
reports extras and nodes per spectrum for a0 and a0null side by side. `masster_rewire_null` is read only in
`masster_pl.py` (step 4); the model never sees the flag, so a0null checkpoints load as a0 checkpoints.

Tests, new `tests/test_masster_rewire_null.py`:

- Auto null strings for all 17 rules equal `debug_masster_rewire_rule_gate.topology_null_smarts(dict)` output (import the gate via
  `scripts.debug.smarts.debug_masster_rewire_rule_gate`, as `tests/test_debug_masster_rewire_rule_gate.py` does); the three
  `null_smarts` overrides are used verbatim.
- Parity with the gate's draw: for 10 hand molecules and 3 rules, `build_null_rewire_matches` with the seed key
  `f"nist20:{mol_id}|{rule}"` forced through a test hook equals `debug_masster_rewire_reach_ceiling.null_matches(...)[0]`
  (same picked atom tuples). The production seed key (`canonical SMILES|rule`) gives a different but reproducible draw: two
  calls with the identical SMILES string give the identical `RewireMatchSet` (no permuted-SMILES assertion: matches hold
  molecule-local atom ids, so a different atom order gives different ids).
- Real sites excluded: no null `motif_atoms` set equals a real one; null count per rule equals `min(deduped real count, deduped pool size)` (per-rule count equal to the real count, or to the pool size when
  smaller).
- Shape: for each null match, terminal pattern atoms are degree-1 in the molecule and non-terminal ones are not; cut pairs are
  molecule bonds; a rule with no real match yields no null; a molecule with a real match but an empty null pool yields none.
- Null edits go through the stage: on `O=S(=O)(c1ccccc1)c1ccccc1` the null for rule 0 produces valid extras (valence
  adjustments computed, no hard error) whose atom sets differ from the real extra; node counts per spectrum are within the
  gate's null range.
- `a0null` end to end: the PL with `masster_rewire_null: true` builds a null `static_graph`; a model state dict from an a0 and
  an a0null run have identical keys and shapes.

### Step 6 `parity_tests`: reference parity, oracle, CPU/CUDA, G0b script

New `tests/test_masster_rewire_parity.py`, an extension to `tests/test_masster_cuda_full_candidates.py` (step 3 list), and the
G0b script.

- **Parity vs the G0 reference.** `scripts/debug/smarts/debug_masster_rewire_reach_ceiling.py` (`edited_products`,
  `products_for`) imported as in the gate test. On a pool of ordinary kept fragments from the real builder
  (`build_masster_candidates`, CPU backend, `MassterModel._unique_component_memberships`) for the 8 prototype molecules and,
  when `output/rearrangement_mining_2026_10_14/molecules.csv` exists, 200 molecules of it (skip the 200-molecule oracle with a
  clear message when the file is absent; never skip the 8-molecule one): the distinct atom sets of the production stage
  (extras plus merged twins) equal the reference's at depth 0 for every molecule, and depth 1 equals the corrected reference
  (C3). Also compare boundary bonds, cut counts, net formed counts and parents against a networkx edit of every kept fragment
  (oracle), including non-splitting cycle bonds, deleted atoms, multi-component `P`, and a form pair whose endpoint is deleted
  (must raise).
- **Coverage equivalence** on small molecules: with an unlimited pool (all connected sets), ordinary pool plus products equals
  the gate's reachable sets restricted to products whose motif bonds are uncut; document the straddling-cut exception.
- **Precursor guard and caps**, **multi-adduct**, **bitset width** (stage and export `mask_bits` agree at 61/62/63/124/125 atoms),
  **mixtures and grouped CE views** as listed in step 3, run at model level here if not already.
- **Order independence**: shuffling the spectra of a batch changes nothing per spectrum (CPU always; CUDA when available).
- **No-match bit identity** on a batch of 8 diverse non-rule molecules through the full model.
- **CPU/CUDA**: as in step 3 (cuda marked).

**G0b script** `scripts/debug/smarts/debug_masster_rewire_stage_g0b.py` (CPU or GPU; check `nvidia-smi` first; scratch output
under `output/rewire_g0b_2026_10_15/`, never `data/proc/nist23/spec_df.pkl`). It loads `i1` epoch 59 the way
`debug_masster_rewire_reach_ceiling.py` does (`get_config_from_ckpt`, `deep_update(load_config(...), saved)`,
`apply_inference_config`), with the ORB-migrated scratch copy
`/tmp/claude-1002/-home-feiw-gitRepos-frag-gnn/3f5debb8-28ae-4a28-9327-4237177ea097/scratchpad/i1_epoch059_orb.ckpt` (not moved
into `saved_ckpts/`), then enables the stage by setting `masster_rewire_rules: true` in the config and loading the state dict
with the two new entries injected (current `REWIRE_RULE_SPEC_HASHES`, zero residual) so the flag-on model starts as `i1`.
It wraps `model._append_rewire_rows` with a recorder that captures its keyword arguments once per batch, then re-invokes
`expand_rewire_candidates` on the recorded inputs for each `--fractions 0.25 0.5 1.0` (and the template ceiling 256), timing
each call with CUDA synchronization and recording `torch.cuda.max_memory_allocated`, host syncs and the diagnostics. Over the
`nist20_inchikey` and MassSpecGym validation splits it reports, per fraction:

- nodes per spectrum (mean, p50, p95, max) over the whole split and over matching spectra, node growth versus the pool;
- extras before and after the cap, merged rows, rows dropped for no valid channel and for the cap, `rewire_cap_tie_count`,
  truncated matches (and the reach lost to `max_matches_per_rule` 4 vs 8 vs unlimited);
- intensity-weighted reach of the mined rearrangement peaks (reuse `peaks.pkl` and the composition test of the G0 script) at
  `R_after0` BEFORE the cap and AFTER it, and the share of the 16.2 R_after0 points held by spectra above the cap that survives;
  the same for the tail (>= 25%) of 669 spectra;
- F survival (R_after1 / R_all) recomputed with the corrected depth-1 (C3) and the old number alongside;
- stage wall-clock per batch and per spectrum, peak memory, host-sync count;
- CPU precompute: `build_rewire_matches` and `build_null_rewire_matches` over every SMILES of `molecules.csv` (29,802): total
  seconds and p95 ms per SMILES, against 5% of one control epoch.

No pass/fail on node growth: the table feeds the choice of `masster_rewire_max_extra_fraction` for the arm configs (user
decision 4). Report it as reach-after-cap versus node growth versus wall-clock. A smoke test of the script
(`--limit 4`, CPU, tiny batch) is added to `tests/test_masster_rewire_parity.py` using a freshly built tiny model instead of
the checkpoint, so it runs without the 8 GB GPU and without data files.

## Cross-cutting rules every step must keep

- **OFF is bit-identical.** The flag off adds no module, buffer, state-dict key, hparam read in the batch path (the PL caches
  its flag at init), RNG draw or output key. `MASSTER_RULE_VOCABULARY` and its hashes are untouched (separate vocabulary).
  Existing checkpoints load unchanged; the existing test files named in step 3 are the OFF guard and must pass unmodified.
- **Hard errors, never silent skips.** A rule/graph mismatch, an unresolved cut bond, an out-of-range atom, a non-converging
  labelling, an unsorted bond table is a `ValueError` naming spectrum and rule. Dropped rows (no valid channel, cap) are
  counted per spectrum in the diagnostics, never silent.
- **Config.** One template; new keys have a one-line first comment; read inline gated on `model_type == "masster"`; no
  `.get(default)`; no backfill of old checkpoints (hparams are the config, the template fills only keys a checkpoint predates).
- **Cost scales with NODE count.** Every report states nodes per spectrum (mean, p50, p95, max) next to wall-clock.
- **Matching is on the standardized molecule** the model graph was built from; atomic numbers are cross-checked at stage entry.
- **Shared files** (`masster_model.py`, `masster_pl.py`, `template.yml`) are edited live by other sessions: re-read before
  each Edit, small hunks, no reformatting, explicit staging by the user. New behaviour lives in `rewire_rules.py`,
  `rewire_null.py` and `rewire_expansion.py`.
- **Machine.** 31 GB RAM, one RTX 2070 SUPER (8 GB) shared with another session; `nvidia-smi` before GPU use; small test
  batches; `FRAGNNET-GPU` python from the repository root; `pytest -p no:cacheprovider`.

## Risks and open items

- **C1 is the load-bearing correction.** If the proposal slice is wrong the control comparison for decision 2 is void; the
  bit-equality tests of proposal outputs and `bond_logits.grad` are mandatory, not optional.
- **Attention bucketing.** A matching molecule's candidate count moves it between power-of-two buckets in
  `_contextualize_by_molecule`; no-match molecules keep their count and bucket membership by count, but a bucket's batch
  dimension changes when a matching molecule joins or leaves it. Bit equality of no-match molecules is asserted at
  `attention_type: "none"`; at induced/full attention it is asserted exactly first and relaxed to `1e-6` only if a
  documented bmm batch-size effect shows up.
- **Merged rows are common** for cut-type rules (S-to-O migration keeps the atom set of `F`): at depth 0 they add no new peak, only
  an indicator and a validity OR. The value of those rules is at depth >= 1 (`a0d1`); do not read an a0 null result as a
  null for cut-type rules.
- **G0 numbers.** C3 means the published `R_after1` and F survival were computed with restored cut bonds; recompute in G0b
  before quoting them.
- **Match truncation default 8** (C8) is a guess until G0b reports truncated spectra.
- **Quantized cap key** (`2**-16` nats) is a noise filter chosen here, not in the plan, not an exact-parity device. The CPU/CUDA
  test compares extras exactly only when `rewire_cap_tie_count == 0` and no kept/dropped boundary row has `|delta q| <= 1`;
  otherwise it compares the sets modulo those rows. A bf16 smoke test is included.
- **Host syncs** (about 6 plus one per chunk and per four labelling sweeps) exist only when a spectrum has a match; G0b reports
  the count and the stage wall-clock.
- **Parent column deferred** (C7); `a1` (proposal gradient for extras) stays out of v1.
- **Documentation steps** stay with the main session: link this spec from the plan and the MASSter generalization hub, replace
  the project-state block, write the mining finding, fill the plan's "Admitted rules" table from the catalog above (17 rules).

## Review log

Applied (each verified against the code or the spec text):

- Merge (C5) broke proposal bit-equality: correct (`chemically_valid_fragments` and `full_candidate_mzs` read the merged tensor). Fixed with
  `ordinary_valid_channels`, a pre-merge proposal view (base column, proposal m/z, proposal slice, PL coverage metrics) and a test.
- `earlier` precedence, float floor of the relative cap (`+ 1e-9`, host int64), `separate_ion_mode_proposal_head` guard (the head and its
  loss at `masster_model.py` ~1965/3850 and `masster_pl.py` ~1928 are proposal-side), permuted-SMILES null claim removed, dtype/device of
  appended fillers, null size-matching on DEDUPED counts, per-chunk reduction and element-budget chunking, `-inf` consumer audit and
  full-loss NaN test (decision 2 reworded), float32 cast and finite assert of the cap key with honest parity wording, duplicate-ordinary-set
  check made conditional on a code reading with a dedup fallback, old-checkpoint PL round trip, finite filler for the head argument,
  output key list audit with an inference alignment test, and the null terminal-deletion caveat (kept as faithful to the gate, documented).

Rejected: none outright. Adjustments to the reviewer's fixes: the proposal view is built by masking rather than by a larger edit that
keeps merged validity out of `valid_channels`, which would touch the head's mask path; the null caveat is documented rather than changing
the null `deleted` set, to stay equal to the gate.
