# MASSter — One-Pass Bond-Set Fragmentation Plan — log

> **Status:** REFERENCE · **Updated:** 2026-10-06
> Verbatim dated log moved from [masster_plan.md](../plans/masster/masster_plan.md) on 2026-10-06; the plan holds the current summary.


#### Measured finding: fixed base-only CE trades final cosine for support (2026-08-20)

Three matched merged-spectrum grid sweeps varied
`masster_base_ce_weight` over `0.1`, `0.2`, and `0.3` for unique-fragment
targets 128, 256, and 512:

- W&B sweep `o6gsdyvv`: U128, ranked-mask budgets 256 and 2048;
- W&B sweep `9pmv90u1`: U256, ranked-mask budgets 256 and 2048;
- W&B sweep `4wdysag2`: U512, ranked-mask budgets 512 and 2048.

At the lower mask budget, mean best validation cosine decreased monotonically:

| Fixed base-CE weight | Mean best validation cosine |
| -------------------: | --------------------------: |
|                  0.1 |                     0.72846 |
|                  0.2 |                     0.72626 |
|                  0.3 |                     0.72231 |

The `0.3` mean contains two completed runs and the U512 run at epoch 121. At the
2048-mask budget, the mature `0.1` and `0.2` arms showed the same ordering:
`0.73240` versus `0.73044`. Those six arms crashed late at epochs 117--176, so the
comparison uses their best checkpoint metric rather than successful completion. The
2048-mask `0.3` arms were still too immature to compare.

This is not a pure support regression. At mask budget 2048, increasing the weight
from `0.1` to `0.2` raised mean candidate recall from `0.7064` to `0.7156` and mean
candidate weighted recall from `0.8927` to `0.9007`, while mean best cosine fell by
`0.00196`. The auxiliary objective therefore improves candidate inclusion but harms
the final probability allocation. The sweeps have no `0.0` arm, and a matched
proposal-anchored zero-weight run was not found; they establish that more weight is
harmful over `0.1`--`0.3`, not that disabling the auxiliary term is optimal.

The most plausible mechanism is gradient interference, because the implemented
base-only loss is not proposer-only. `base_pred_logprobs` combines the base
fragment probabilities with the same `ion_h_log_probabilities` produced by the
candidate intensity transformer for the final spectrum. Its gradient therefore
updates the shared transformer trunk and ion/H head as well as the bond, charge,
and precursor paths. Meanwhile, proposal anchoring removes the base probabilities
from the transformer's input even though the final score remains `l_i + r_i`, so
the residual must correct a moving base offset that it cannot observe directly.
Consistent with this interpretation, the U128/256 validation base-to-final CE gain
shrinks from `0.3666` at weight `0.1` to `0.3226` at `0.2` and `0.2972` at `0.3`.
The raw weighted base-loss magnitude also grows from roughly 8% of the primary-loss
magnitude at `0.1` to 25% at `0.3`; this is not a gradient-norm estimate, but it
shows that the nominally auxiliary term is not negligible.

Revise the Defect 1 experiments in this order:

1. **Isolate the shared-head gradient.** In the base-spectrum branch only, use
   `ion_h_log_probabilities.detach()`. This preserves the base-CE value and its
   gradients to the bond/charge proposal while preventing it from training the
   candidate transformer through the shared ion/H prediction. Compare fixed `0.1`
   with and without the detach at matched seed and budget. Advance the change only
   if the candidate recall gain remains and final cosine recovers.
2. **Measure gradient conflict directly.** On matched batches, log gradient norms
   and cosine similarity between `L_final` and `L_base_only` for the bond head,
   charge head, precursor encoder, candidate-transformer trunk, residual head, and
   ion/H head. The working hypothesis is alignment in the proposer and conflict in
   shared intensity parameters; do not redesign the shared encoder without this
   attribution.
3. **Reduce late-training pressure.** Compare fixed `0.1` with a warm-up schedule
   that starts at `0.1` and decays to a small positive floor. Proposal-anchored setup
   currently rejects an exact zero weight, so an exact-zero endpoint requires an
   explicit compatibility change rather than a silent config edit.
4. **Search below the observed boundary.** Test `0.01`, `0.025`, `0.05`, and `0.1`
   after the shared-head ablation. Include an exact `0.0` control only after defining
   how the anchored proposer remains supervised. The present evidence makes `0.2`
   and `0.3` poor default choices.
5. **Control support through selection.** Compare ranked-mask budget and a calibrated
   minimum mask-probability threshold at matched realized candidate count. Candidate
   support is a selection problem; do not use a stronger full-spectrum auxiliary loss
   as the primary recall knob if budget or thresholding can recover the same support
   without disturbing intensity gradients.
6. **Escalate to a fully proposer-local objective only if needed.** If detaching ion/H
   is insufficient, block base-loss gradients into the shared precursor encoder and
   train only the bond/charge proposal parameters, or replace full-spectrum base CE
   with an explicit candidate-ranking objective. This is a larger change and needs
   its own matched control because it changes which representation learns proposal
   information.


## 8. Implementation stages and verification gates

### Stage 1 — Probability translator, no learned model — IMPLEMENTED 2026-08-14

Implement a standalone top-K mask enumerator and mask-to-ion translator using
provided synthetic bond, atom-charge, and H-transfer logits.

Implemented the model-independent portion in
`src/fragnnet/frag/bond_mask_generator.py`: lazy exact mask ranking, bounded
top-K collection, connected-component translation, and truncated fragment
marginal aggregation. `tests/test_bond_mask_generator.py` verifies the ranking
against brute force through eight bonds and covers unrestricted cut counts,
ring cuts, duplicate marginal paths, extreme logits, and input validation.
Charge retention and H-transfer remain Stage 2/3 work.

Bond endpoints use PyG's `(2, num_bonds)` layout across the learned head and
translator. The initial mask budget may be extended when high-ranked masks only
open rings or duplicate previously generated atom subsets, with a separate hard
enumeration cap.

Verify:

- exact top-K mask equality with brute force for small `B`;
- no fixed cut-count/depth assumption;
- components match a reference graph implementation;
- duplicate fragment and ion probabilities equal direct probability sums;
- stable behavior for probabilities near 0 and 1;
- CPU determinism and bounded memory at `K=500`.

### Stage 2 — MASSter bond and charge heads

Add the one-pass molecular encoder, endpoint bond baseline, optional line-graph
bond GNN, and atom charge-affinity head.

**Initial integration implemented, 2026-08-14:**
`src/fragnnet/model/bond_fragment_heads.py` now contains the symmetric endpoint
bond baseline, atom charge-affinity head, differentiable selected-mask
log-probability evaluation, component charge competition, and component/boundary
pooling. Charge competition is segmented by cut mask, and all component pooling
uses sparse membership pairs. `MassterModel` reuses FraGNNet's molecular GNN
as an unpooled `precursor_encoder`; molecular PyG graphs now carry canonical
chemical bonds and atom chemistry fields. `MassterPL` uses `SpecMolDataset`
and `SpectrumPL` rather than the D3/D4 dataset and training wrapper.

**Bond-focused ablation implemented, 2026-08-16:** setting
`masster_bond_num_layers` above zero replaces only the endpoint cleavage
head with explicit bond states and alternating bond-to-atom/atom-to-bond
updates. Bonds communicate when they share an atom, without materializing a
line graph, and endpoint-order invariance is preserved. A value of zero keeps
the original endpoint baseline. The charge/ion heads and independent-Bernoulli
top-K mask ranking remain unchanged. The controlled K256 GINE-EFA experiment is
`config/masster/merged/ablations/bond_model/nist20_merged_mask256_gine_gsa384_bond2.yml`.

**Architecture direction from the initial W&B runs, 2026-08-16:** increasing
the atom-centered GINE encoder from 256/4 layers to 384/5 layers did not improve
validation cosine, while GINE-EFA reached the best observed score and converged
in fewer epochs. Mixed-mask exploration improved candidate weighted recall but
did not improve cosine. These results favor a more bond-focused encoder over a
larger generic atom transformer. The next controlled architecture experiment
should maintain explicit learned bond states, update adjacent bonds through a
line graph or edge-aware attention, alternate atom-to-bond and bond-to-atom
updates, and condition bond attention on acquisition context. Keep the current
independent-Bernoulli mask ranking and the charge/ion heads unchanged for this
comparison so any gain can be attributed to the bond representation. Only if
that model improves held-out results should correlated multi-bond scoring be
tested; pairwise or autoregressive cut dependencies change the exact top-K
semantics and are a separate, higher-risk experiment.

Verify:

- bond predictions are invariant to endpoint ordering;
- charge probabilities across components sum to one;
- the bond GNN beats the endpoint baseline before retaining it;
- held-out bond ranking and fragment coverage beat random mask selection.

### Stage 3 — H-transfer / ion-mode head

Add component/complement/boundary pooling and the joint categorical head.

**Initial integration implemented, 2026-08-14:** the same module contains the
joint ion-mode/H-transfer head and masked channel normalization. MASSter now
constructs per-adduct chemical-validity masks and monoisotopic CRF/CMF masses,
including explicit H-transfer channels. Isotope envelopes remain out of scope
for the first training slice.

Verify:

- invalid formula channels receive zero probability;
- CRF/CMF masses match `formula_utils.py` exactly;
- known complementary components can receive different H-transfer
  distributions;
- per-adduct and per-polarity calibration is reported.

### Stage 4 — Flat fragment-ion integration

Return a flat fragment-ion batch containing atom masks, cut-mask provenance,
formula/mass, generation log probability, charge probability, ion-mode/H-transfer
probability, and combined log probability. Do not require DAG depth or DAG edges.

**Initial integration implemented, 2026-08-14:** sparse mask/component batching
preserves `(molecule, cut mask, component)` provenance and emits
`pred_mzs`, `pred_logprobs`, `pred_batch_idxs`, and truncated OOS probability for
`SpectrumPL`. Training uses the same sparse cross-entropy implementation as
FraGNNet. Empty and all single-bond masks are forced exploration candidates in
addition to learned top-K masks, preventing the zero-signal random-initialization
failure observed with deterministic top-K alone. Consequently, `max_masks` is
the learned ranking budget; the actual count can be larger by the single-bond
exploration set and is logged explicitly.

**Unique-fragment integration implemented, 2026-08-19:** compiled topology assembly
now records component-to-unique-atom-set mappings and canonical representatives.
Charge affinity is evaluated once per unique atom set, normalized per latent mask,
and marginalized across duplicate pathways before the intensity transformer. This
removes duplicate transformer tokens without discarding their probability mass or
changing OOS semantics.

The first production-shaped configuration is
`config/nist20_merged_inchikey/masster/s0.yml`: K=100, CRF/CMF channels,
H-transfer range -4 through +4, and FraGNNet's sparse Gaussian cross-entropy.
Training logs intensity-weighted candidate recall, retained exact-mask
probability, selected mask count, and component pathway count so support errors
can be separated from probability-allocation errors.

Verify at matched candidate budgets against D3 and D4:

- formula recall and intensity-weighted peak recall;
- downstream cosine/JSS metrics;
- unique fragments per evaluated cut mask;
- runtime and peak memory, including combinatorial outlier molecules;
- recall beyond four broken bonds;
- stability across at least three seeds.

The design advances only if it beats both random-mask selection and the existing
D3 candidate set at a matched fragment budget on held-out scaffolds. D4 is the
higher-recall reference, not an inference dependency.
