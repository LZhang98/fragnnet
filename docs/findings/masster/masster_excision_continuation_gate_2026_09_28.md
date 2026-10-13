# MASSter excision continuation coverage gate

> **Status:** FINDING · **Updated:** 2026-09-28
> Diagnostic completed. A nitro NO-loss
> family confirms on validation; unrestricted bridge excision is not supported by this gate.
> No production loss channel or training run was added.

## Decision

The S-minus-X construction is a valid atom-set representation for products spanning the
new bond. It adds measured support beyond whole-precursor templates. Its useful scope is
narrow: aromatic nitro NO loss confirms on held-out molecules; the generic union has no
clear train advantage over random disconnected candidates. Proceed, if implementing a
channel, with one formula-deduplicated nitro NO-loss family. Do not enable every small bridge.
This coverage gate cannot establish intensity prediction or cosine improvement.

## Method

See the [protocol and commands](../../plans/masster/masster_excision_channel_probe.md).
The diagnostic entry point is
[`debug_masster_smarts_rule_gate.py continuations`](../../../scripts/debug/smarts/debug_masster_smarts_rule_gate.py),
implemented in [the continuation probe](../../../scripts/debug/smarts/debug_masster_excision_continuations.py).

- c0 checkpoint `c0_control_vcpho5qm.ckpt`, epoch 42, strict loading, `[M+H]+` only.
  Historical FT embedding reused for current OB and OB_F inputs; dataset files unchanged.
- Seeded random sample of 512 train molecules / 2,039 spectra and 512 validation molecules /
  2,035 spectra, at most four spectra per molecule. Sample molecule IDs have zero overlap;
  existing `nist20_inchikey` splits are used. Test was not inspected.
- X is connected, at most three heavy atoms, contains no ring atoms, and has exactly two
  non-ring external bonds to distinct endpoints a,b. S must contain X,a,b. A connected
  product in the rewired graph spanning a-b lifts to connected S by restoring X. One-ended
  products are ordinary connected subsets and need no new loss channel.
- Baseline is c0's final retained candidate pool, all eligible current disconnected template
  sides, **and all generic whole-precursor bridge products**. The last baseline deliberately
  removes credit for discovering another whole-precursor loss. Baseline template products
  are generously included without forcing them through a fixed candidate budget.
- Query only channels matching an observed peak, then reject any connected heavy-composition
  twin with an own-H count within ±4 of the channel. This is conservative: the twin's own
  valence filter is not required to accept that shift. Searches exceeding 200,000 steps are
  unresolved and excluded, never counted as unique. This checks formula alternatives, not
  every different-formula near-isobar. Existing pool masses are tolerance-matched directly.
- Intensity is the stored linear peak intensity normalized within each spectrum. No new
  peak filtering or square root transform. OB/OB_F tolerance is 10 ppm with a 200 Da floor.
- Nulls are distinct random disconnected atom sets matched to each environment's product
  count and heavy-atom sizes. One seeded null draw per spectrum/environment. Reported strict
  channels consume two freed boundary valence slots for a single new bond. This remains a
  valence-budget approximation, not a rewired electronic structure or radical model.
- Average spectra within each molecule, then average molecules. Paired molecule bootstrap:
  2,000 draws for ordinary 95% intervals. Train selected environments with at least ten
  molecules, strict lower limit >0, no capped real/null searches, and no missing nulls.
  Four of 767 environments selected. This train search is exploratory.
- Before validation, froze the four environments and a Bonferroni-adjusted percentile
  bootstrap check: 20,000 draws, seed 20260928, limits at 0.00625/0.99375. Confirmation also
  requires ten molecules and no unresolved searches or incomplete nulls. These are approximate
  bootstrap intervals and do not include variability across repeated null realizations.

## Measurements

All coverage values below are **percentage points of normalized linear target intensity**.
The generic train union and validation class rows have different populations.

| Population | Molecules | Strict unique added | Random null | Difference | Interval for difference |
| --- | ---: | ---: | ---: | ---: | --- |
| Train, all generic bridges | 512 | 0.5667 | 0.5161 | +0.0506 | 95%: −0.0941 to +0.2091 |
| Val, three-carbon bridge | 14 | 0.0901 | 0.0519 | +0.0383 | Adjusted: 0.0000 to +0.1339 |
| Val, NH bridge | 13 | 1.2679 | 0.6208 | +0.6470 | Adjusted: +0.0377 to +1.3483 |
| Val, nitro encoding A | 30 | 0.6478 | 0.2755 | +0.3723 | Adjusted: +0.0238 to +0.9099 |
| Val, nitro encoding B | 30 | 0.6478 | 0.1509 | +0.4969 | Adjusted: +0.0641 to +1.2048 |

The carbon family fails the lower-limit criterion and has capped queries. The NH family
has three capped null queries, so its positive interval is **inconclusive under the frozen
gate**. Neither is a demonstrated negative chemical result. Both nitro encodings pass with
zero caps and complete nulls. Their real coverage is identical: retaining the other nitro
oxygen changes atom identity, not formula. They are one family, not two independent wins.
Different null values come from separate random draws.

Exact frozen keys (map 1 = deleted atoms, map 2 = retained boundary):

```text
[CH2:1]([CH2:1][C:2])[CH2:1][CH2:2]
[NH:1]([c:2])[C:2]
[O-:1][N+:1]([c:2])=[O:2]
[O:1]=[N+:1]([c:2])[O-:2]
```

These are canonical local SMILES keys, not executable SMARTS. Nitro coverage contributes
0.0380 percentage points when averaged over all 512 validation molecules. It does **not**
imply a 0.00038 cosine gain. The selected-family union adds 0.0726 points versus 0.0343
for its separately sampled union null; capped NH/carbon cases prevent treating that union
as a fully resolved confirmation.

For context, train pool coverage is 94.8163%; current template sides add 0.0077 points,
all generic whole products add a further 0.0380, and raw continuations add 0.7718 before
connected-formula exclusion. Real/null searches cap in 18/26 of 2,039 train spectra.
The aggregate null result is therefore bounded by unresolved cases, not proof of no benefit.

Generic bridges occur in 438/512 train molecules (85.5%) and 443/512 val molecules (86.5%).
This is broader than the earlier curated-template prevalence claim and uses a molecule
rather than spectrum denominator. Before formula pruning the generic train union averages
379 continuation atom sets per spectrum (maximum 5,186), against 228 ordinary candidates.
This does not validate the proposed 0–2 retained loss channels per candidate; a production
channel needs a separate count after formula deduplication and pruning.

## Chemistry and implications

The nitro result has relevant primary precedent. Pandeti et al. (2017) report low-energy
ESI-CID of protonated nitroimidazoles, including side-chain cleavage followed by NO radical
loss for ronidazole and ornidazole. Their discussion invokes nitro-to-nitrite rearrangement;
nimorazole behaves differently. This supports the family and multistep premise, without
proving every matched NIST peak's mechanism or the order of operations inferred from a
formula. [Paper](https://pubs.rsc.org/en/content/articlepdf/2017/ra/c7ra08312b), pp. 45219–45220.

“Only heteroatom bridges can yield unique formulas” is false even at the graph/formula level:
in `NCCCCN`, excising an internal CH2 leaves C3N2, whereas every connected subset containing
both nitrogens must retain all four carbons. The unit test captures this counterexample;
it is not evidence that the corresponding gas-phase reaction occurs.

The positive result supports a restricted loss-channel design using candidate S minus the
nitro N and one O, retaining the aryl C and other O. Merge resonance-equivalent routes and
same-formula products before allocating probability. Explicit radical/H bookkeeping,
precomputed connected-twin masks, feature construction, runtime, and a matched training
comparison remain implementation work. The existing ±H tensor alone does not provide those
features. Negative-mode SO2/N2O/O-transfer claims were not tested by this positive-only c0.
The hypothesis that correct atom identities help intensity prediction despite formula twins
also remains untested because this gate deliberately excludes such twins.

## Artifacts and validation

Artifacts: [`output/excision_continuations_2026_09_28/`](../../../output/excision_continuations_2026_09_28/).
`train` and `val` each have `.pkl`, `.sample.csv`, `.summary.csv`, `.provenance.json`, and
`.log`; `train.families.json` is the frozen selection. `val.confirmation.csv` and
`confirm_families.py` record the adjusted check. Provenance saves resolved config, input
sizes/mtimes, split identities, source hashes, Git revision/diff hash, device, seeds, and
elapsed time. Final source snapshots are `train_final_probe_source.py` and
`train_final_gate_source.py`. Dataset/checkpoint content hashes were not computed.

Both runs completed: train 578.7 s, val 94.4 s on RTX 2070 SUPER with CPU chemistry checks.
An earlier train attempt was stopped to improve connected-H query termination; its partial
artifacts and source copies are retained as `initial_capped_train.*` and `train_*_source.py`.
The final query stops early only after finding a connected twin for every queried H count;
it otherwise enumerates to completion or records the cap.

Seventeen focused scientific unit tests pass, including explicit-rewiring atom-set identity,
new-bond valence correction, a synthetic continuation-only peak, capped-query exclusion,
bootstrap aggregation, environment invariance, and targeted-H/exhaustive agreement. Ruff
passes for the three changed Python files. No full-suite run or model training was performed.
