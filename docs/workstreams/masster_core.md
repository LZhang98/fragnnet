# MASSter core model on NIST20

> **Status:** HUB · **Updated:** 2026-10-16
> Architecture, training recipe, regularization, losses, candidate/proposal machinery, speed/GPU, the 128-epoch screens and campaign reads, ion-H channel selection, and feature/information audits for MASSter on NIST20.

## Where it stands

- **Current speed audit:** [opportunities and limits](../findings/masster/masster_code_path_audit_2026_10_16.md#speed-audit-of-the-current-path-2026-10-16): `i1`/`i4` already use the main speed switches. First candidates are rewire host round-trips and repeated product-pool reductions; base attention packing and loss padding need GPU profiling. 49 focused CPU tests pass; no current GPU speedup measured.

- **Current code-path audit:** [2026-10-16 findings](../findings/masster/masster_code_path_audit_2026_10_16.md): retired-config test inventories repaired; all 177 MASSter PL tests pass. Package lint has four violations. Rewire output-row sorting regression passes. Added 14 batch/padding cases; focused suite: 168 passed, 3 skipped. GPU and real-checkpoint inference unverified.

- **Shared codebase audit:** all six findings and the three rechecked config runtime defects repaired and committed in `e65339ed`; the unfiltered CPU suite passes (record removed 2026-10-18; `git show 731fe493:docs/findings/codebase_audit_2026_10_13.md`). GPU and real-checkpoint evaluation remain unverified.

- **Config organization:** grouping record (removed 2026-10-13; `git show e65339ed:docs/plans/config_model_grouping.md`); experiment configs now live under their model, with the single root template retained. [Audit findings](../findings/config_audit_2026_10_12.md) 1–3 repaired; 6 resolved for MASSter by [retiring 97 configs](../findings/config_audit_2026_10_12.md#masster-config-retirement-2026-10-13) (2026-10-13, recover from `e180f2c2`); 4, 5 and 7 remain open. Next: malformed/duplicate YAML.

- **Shared notebook audit:** [findings](../findings/notebook_audit_2026_10_12.md); metric relabeling, repeat scaling, reversed parameter counts and execution blockers found. NIST20 duplicate consolidated; large saved table outputs cleared with figures preserved. Reporting repairs remain open.

- **Unit test audit:** export expectations and the three false-positive guards fixed in `e65339ed`; the unfiltered CPU suite passes (record removed 2026-10-18; `git show 731fe493:docs/findings/unit_test_audit_2026_10_12.md`). GPU validation remains open.

- **Shared loss consolidation implemented:** [plan](../plans/loss_consolidation_plan.md#completion-and-validation-2026-10-12); numerical fixes, shared preparation and reductions have regression coverage; committed in `1a63257b`. [File merge and renames](../plans/loss_consolidation_plan.md#file-merge-and-renames-2026-10-12) await review.
- **Reorganization implemented, not merged:** [plan](../plans/repo_reorg_plan.md) Phases 0–9 are complete on `refactor/fragnnet-masster-reorg`. FraGNNet code is in `src/fragnnet/models/fragnnet/`, MASSter in `src/fragnnet/models/masster/`, GRAFF and MassFormer in `src/fragnnet/baselines/` ([architecture](../architecture.md#package-layout-and-dependency-rule)); ICEBERG runs externally via ms-pred and the main environment is [DGL-free](../plans/repo_reorg_plan.md#decided-external-iceberg-dgl-free-main-environment).
- **Simplification cleanup committed:** [audit](../findings/masster/masster_simplification_audit_2026_10_12.md#outcome-2026-10-12) items 1–4 and the annotated-export fix (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_export_merge_untransform_2026_10_12.md`) for sqrt/log10 models with merged rows; annotated exports made before `7e73b384` from such models must be regenerated. Module renames: candidate generation (builder, Cython `cut_mask_kernels.pyx`, SMARTS templates, CUDA) lives in `src/fragnnet/models/masster/candidates/` (`4e36bcae`); the heads are split into `candidate_transformer.py`, `proposal_heads.py`, `component_pooling.py` and `pathway_probabilities.py`.
- **RT/CCS:** [RTHead and CCSHead](../plans/masster/masster_rt_ccs_plan.md) with independent label sets completed a new 10-epoch GPU debug run; full Enveda arm remains prepared, heads default off.
- Matched 128-epoch screen on the merged MA/MI base: control `c4a_base_s1` = **0.7270** `val_cos_sim` (tail mean epochs 118-127, replicate spread 0.0007, gate +-0.005), config `config/masster/ma_mi_merged/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_base_s1.yml`.
- Real levers: `a1_prenorm` (`q4t4udee`) **+0.0056**, the only clean single-key architecture pass; cosine LR decay with 1000-step warmup `d3` (`rcfzf67p`) 0.7357 (+0.0072 at epoch 121, n=1); the 2026-09-20 feature bundle is worth about +0.006/+0.007; EMA gained +0.0102 on `i1` (see generalization hub).
- Nulls/rejections at this screen: cut-set and candidate-token features (best +0.0032), intensity escape (+0.0000), proposal freeze/wd (+0.0010/+0.0020), fragment-marginal selection, encoder simplification `g1` (-0.0160), `b2` (-0.0062), mass pair bias (-0.0002).
- The train/val gap (0.151 cosine) is intensity allocation among peaks the model already proposes, not coverage; capacity is not the bottleneck. Work on that gap lives in the generalization hub.
- Prepared, not submitted: full-length `wd10` arm and matched `c4a_base_s1_trainaux` control (`fir:h100-80gb`). The attention arms `a4`, `a5_ce_pair_bias`, `b4` (`config/masster/ma_mi_merged/attention_2026_09_21/`) lost their plan on 2026-10-18 (see Settled).
- Blocked: ion-H Phase 2 selection sweep needs a recoverable trained checkpoint.
- Working-tree changes are tracked in [project state](../project-state.md#uncommitted-working-tree-2026-10-11).
- Next action: none scheduled here; new core levers go through the `a1_prenorm` + LR-decay stack on `i1` ([generalization hub](masster_generalization.md)), and an `a1_prenorm` seed replicate is the cheapest open confirmation.

## Open threads

| Thread | Status | Next concrete step |
| --- | --- | --- |
| [Current speed audit](../findings/masster/masster_code_path_audit_2026_10_16.md#speed-audit-of-the-current-path-2026-10-16) | FINDING | Profile current base/rewire GPU steps; benchmark behavior-preserving rewire host/cap changes first |
| [Current code-path audit](../findings/masster/masster_code_path_audit_2026_10_16.md) | FINDING | Config tests repaired; package lint and GPU/real-checkpoint validation remain open |
| [Config audit](../findings/config_audit_2026_10_12.md) | FINDING | Findings 1–3 repaired; 6 resolved for MASSter (configs retired 2026-10-13); resolve malformed/duplicate YAML next |
| [Shared loss consolidation](../plans/loss_consolidation_plan.md) | LANDED | Review implementation and tests; GPU validation when available |
| [Training runner cleanup](../plans/runner_cleanup_plan.md) | ACTIVE | Phases 1–2 committed (`3deaf652`); Phases 3–5 reviewed, ready to commit |
| [Code simplification audit](../findings/masster/masster_simplification_audit_2026_10_12.md) | FINDING | Items 1–4, export fix and module renames committed; next item 5 (orchestration extraction) |
| [RT/CCS heads](../plans/masster/masster_rt_ccs_plan.md) | DEBUG COMPLETE; full arm PREPARED | Matched Enveda multitask/control evaluation; coordinated in the [CASMI/Enveda hub](casmi_enveda.md) |
| [Regularization sweep / `wd10`](../plans/masster/masster_regularization_sweep_plan.md) | PREPARED | Generate on Fir with `submit_masster_regularization.sh wd10` if the gap question is reopened |
| [Unused NIST peak annotations](../findings/unused_information_inventory_2026_09_22.md) | FINDING (open) | Decide whether to use per-peak formula assignments in the loss |
| [Cross-adduct paired loss](../findings/cross_adduct_loss_headroom_2026_09_17.md) | FINDING (gate half) | Needs a multi-adduct MASSter checkpoint |

## Settled — read before re-proposing
**Training recipe and architecture**
- Epoch-128 arms (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_architecture_arms_epoch128_2026_09_22.md`) — `a1_prenorm` +0.0056 passes; `b2` -0.0062, `g1` -0.0160 rejected; mass pair bias -0.0002; replicate floor on tail mean 0.00036.
- Campaign read-out (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_campaign_readout_2026_09_21.md`) — regularization sweep no promotions; bundle +0.006; val cosine saturates near epoch 100 (last 529 of 630 epochs buy +0.0017).
- Overfit read-out (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_overfit_readout_2026_09_22.md`) — gap 0.151 is ~98% allocation; three LR-decay arms clear the floor (`d3` +0.0072).
- Regularization audit (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_regularization_audit_2026_09_19.md`) — BatchNorm, not dropout, drives candidate-set train/eval divergence; atom decoder dropped (-0.0018, 1.8x faster).
- Head-input asymmetry (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_head_input_asymmetry_2026_09_22.md`) — proposal reads detached inputs; connecting it costs about -0.01 to -0.013; charge head inert in `c4a`.
- Shared-proposal audit (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_shared_proposal_pathway_audit_2026_09_26.md`) — the CE-context sampler was broken (-0.25 val cosine on `c0`), not the sharing.
- Proposal early stop (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_proposal_early_stop_plan.md`) — `p1` +0.0010, `p3` +0.0020; closed.
**Features and information**
- Cut-set features (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_cut_set_features_plan.md`) — five arms null (best +0.0032); adjacency is the only signed one.
- Feature headroom (removed 2026-10-18; `git show 95341fce:docs/findings/feature_headroom_2026_09_22.md`) — descriptors recoverable at 0.93-1.00; only external quantum knowledge has headroom.
- Feature/information audit (removed 2026-10-18; `git show 95341fce:docs/findings/feature_information_audit_2026_09_22.md`) — map of five parts; two conclusions withdrawn.
- Identity supervision (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_identity_supervision_2026_09_22.md`) — structural-teacher proposal retracted; coverage transfers (0.8975 vs 0.8911 `wrecall`).
- [Unexplained peaks](../findings/masster/unexplained_peak_split_2026_09_22.md) — ~11% unreachable intensity; only 2.91 points have no feasible subformula.
- ChemFrag triage (removed 2026-10-18; `git show 95341fce:docs/findings/masster/chemfrag_rearrangement_triage_2026_09_14.md`) — none of 32 transforms adds unique coverage.
**Losses, proposal, intensity**
- Supervised escape (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_supervised_escape_plan.md`) — exact null (+0.0000 at n=2 vs n=2); not a ranking change.
- [Loss setup review](../findings/loss_setup_review_2026_09_13.md) and proposal loss audit (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_proposal_loss_audit_2026_09_22.md`) — proposal loss scales with covered fraction; CE kept as production.
- [Base-CE weight](../history/masster_plan_log.md#measured-finding-fixed-base-only-ce-trades-final-cosine-for-support-2026-08-20) — fixed 0.1/0.2/0.3 lowers cosine monotonically.
- Fragment-marginal selection is closed null (U128 ties U256; owned by the [MSGYM hub](msgym.md)).
**Speed and candidate machinery**
- [Training speed](../findings/masster/masster_training_speed_2026_09_17.md) — monitoring off 1.40x; CUDA construction about 1.9x per epoch; cut-bond encoder dedup cuts encoder FLOPs 20.1 to 2.1 GFLOP.
- FastGNN / beam (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_fast_gnn_proposal.md`) — beam captures flat 0.8695 mass; stopped.
- Direct-loss batching (removed 2026-10-13; `git show e65339ed:docs/plans/masster/masster_direct_loss_batching_plan.md`) — closest 44x, hungarian 11x.
**Ion/H**
- Ion/H plan summary (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_ion_h_channel_selection_plan.md`) — gate gives no benefit (dense cosine 0.7630/0.7655/0.7631); selection never exercised.
**Neighbouring workstreams**: error analyses, memorization, SMARTS pathway, QET, ORB instruments in [generalization](masster_generalization.md); MSGYM and GLACIER in [msgym](msgym.md); datasets in [spectraverse](spectraverse.md), [nist23_ei](nist23_ei.md), [casmi_enveda](casmi_enveda.md); [fragnnet](fragnnet.md).
- **Retired 2026-10-18:** 30 documents of this workstream removed; summaries and open items in [retired documents](../history/retired_docs.md#masster_core).
- Architecture review todo — retired 2026-10-18 while DEFERRED (`git show 95341fce:docs/plans/masster/masster_architecture_review_todo.md`); if revived: `a1_prenorm` seed replicate; `a5` read against `a3`; RWSE needs the `random_walk_pe` fix ported.
- Ion/H channel selection — retired 2026-10-18 while DEFERRED (`git show 95341fce:docs/plans/masster/masster_ion_h_channel_selection_plan.md`); if revived: disable support gate, verify top-k 95/99 metric, recover a checkpoint, run `masster_ion_h_top_k_sweep.py`.
- Attention/bond budget — retired 2026-10-18 while PREPARED (`git show 95341fce:docs/plans/masster/masster_attention_budget_2026_09_21.md`); if revived: submit `a4`/`a5`/`b4` only if architecture work resumes.
- Bond-cut EBM, EBM plan — retired 2026-10-18 while PROPOSED (`git show 95341fce:docs/plans/masster/masster_bond_cut_ebm_plan.md`, `git show 95341fce:docs/plans/masster/masster_energy_based_model_plan.md`); if revived: oracle excess over matched random pools before any implementation.
- Retained-fragment GNN — retired 2026-10-18 while DEFERRED (`git show 95341fce:docs/plans/masster/masster_hierarchical_fragment_graph_plan.md`); if revived: candidate branch unbuilt; `fc1` was not positive.
- **Unresolved when the September findings were retired (none scheduled):** whether `boundary_bond_encoder` stays on (`c4a_no_bbe` 6% faster, -0.0024 on the 128-epoch screen); `proposal_only_until_epoch` still couples staging with gradient routing (38 configs); the BatchNorm candidate-set gap on a trained checkpoint is unmeasured; `d3` LR decay has n=1; proposal sharing (`sp1`) needs a per-NCE coverage read; Hungarian chunking needs a re-measure with `torch_linear_assignment` on a cluster node; inert new keys inside hashed `spec_params` can change an arm's cache identity.

## All documents

| Doc | Status (updated) | One-line content |
| --- | --- | --- |
| [Current code-path audit](../findings/masster/masster_code_path_audit_2026_10_16.md) | FINDING (2026-10-16) | Selected CPU validation, stale config tests, lint baseline and rewire row-order review |
| [MASSter method](../reference/masster_method.md) | REFERENCE (2026-10-16) | Chemically constrained probabilistic framing, intensity-distribution semantics, ranked candidate selection, and comparison with FraGNNet |
| [Config audit](../findings/config_audit_2026_10_12.md) | FINDING (2026-10-12) | Seven findings; 806 YAML files scanned, 109 focused tests passed |
| [Shared loss consolidation](../plans/loss_consolidation_plan.md) | LANDED (2026-10-12) | Phased numerical fixes and behavior-preserving cleanup |
| [Training runner cleanup](../plans/runner_cleanup_plan.md) | ACTIVE (2026-10-12) | `runner.py` audit: 10 bugs (W&B-off upload, `ckpt_path="best"` without checkpoints, dead `interrupted` guards, silent sweep typos), then a behavior-preserving split into config and loading modules |
| [Shared loss audit](../findings/loss_simplification_audit_2026_10_12.md) | FINDING (2026-10-12) | Reproduced correctness issues and prioritized simplifications |
| [Code simplification audit](../findings/masster/masster_simplification_audit_2026_10_12.md) | FINDING (2026-10-12) | Prioritized cleanup; items 1–4 implemented and committed; candidate generation in `candidates/` |
| [MASSter RT/CCS heads](../plans/masster/masster_rt_ccs_plan.md) | PREPARED (2026-10-10) | Shared heads, label evidence, completed GPU debug training and full experiment |
| [repo_reorg_plan](../plans/repo_reorg_plan.md) | ACTIVE (2026-10-12) | Cross-model reorg: registry and hooks, `fragmentation/`, `frag_utils` split, `models/` and `baselines/` packages, external ICEBERG via ms-pred, DGL-free env; Phases 0–9 complete on `refactor/fragnnet-masster-reorg`, awaiting merge |
| [masster_plan](../plans/masster/masster_plan.md) | LANDED (2026-10-06) | Core design record: one-pass bond-set model, mask enumeration, ion/H channels; snapshot of shipped behavior |
| [masster_candidate_feature_list_plan](../plans/masster/masster_candidate_feature_list_plan.md) | LANDED (2026-10-06) | Ordered candidate-token feature list with one spec table |
| [masster_nonprobabilistic_intensity_plan](../plans/masster/masster_nonprobabilistic_intensity_plan.md) | LANDED (2026-10-06) | Direct peak-list objectives (Hellinger, leak, CE auxiliary); CE stays production |
| [masster_regularization_sweep_plan](../plans/masster/masster_regularization_sweep_plan.md) | PREPARED (2026-10-06) | One-hour regularization sweep (no promotions) and the prepared full-length `wd10` arm |
| [masster_smarts_review_2026_09_13](../findings/masster/masster_smarts_review_2026_09_13.md) | FINDING (2026-09-13) | SMARTS rule set and literature review (26 rules) |
| [masster_training_speed_2026_09_17](../findings/masster/masster_training_speed_2026_09_17.md) | FINDING (2026-10-06) | Training speed: monitoring, CPU/CUDA candidates, batched coverage loss, cut-bond dedup |
| [masster_unmerged_overfitting_2026_09_16](../findings/masster/masster_unmerged_overfitting_2026_09_16.md) | FINDING (2026-09-16) | W&B read of the growing unmerged generalization gap |
| [unexplained_peak_split_2026_09_22](../findings/masster/unexplained_peak_split_2026_09_22.md) | FINDING (2026-09-22) | Unreachable ~11% intensity: 73% coverage, 27% escape; isotopologues 46.5% of escape |
| [notebook_audit_2026_10_12](../findings/notebook_audit_2026_10_12.md) | FINDING (2026-10-12) | Shared notebook reporting correctness and reproducibility audit |
| [loss_setup_review_2026_09_13](../findings/loss_setup_review_2026_09_13.md) | FINDING (2026-09-21) | Loss setup review with counterexamples and fixes |
| [unused_information_inventory_2026_09_22](../findings/unused_information_inventory_2026_09_22.md) | FINDING (2026-09-22) | NIST per-peak formula annotations (99.4% of intensity) never reach the loss |
| [cross_adduct_loss_headroom_2026_09_17](../findings/cross_adduct_loss_headroom_2026_09_17.md) | FINDING (2026-09-17) | Cross-adduct paired-loss data gate passes; model half needs a multi-adduct checkpoint |
| [benchmark_masster_cpu](../guides/benchmark_masster_cpu.md) | GUIDE (2026-09-17) | Benchmark MASSter candidates on L40S (CPU/CUDA backend) |
| [benchmark_masster_merged_control](../guides/benchmark_masster_merged_control.md) | GUIDE (2026-09-17) | Merged MASSter arms on the CPU/CUDA candidate backend |
| [submit_cluster_runs](../guides/submit_cluster_runs.md) | GUIDE (2026-09-14) | Submit training runs on Fir and Vulcan |
| [masster_feature_set_audit](../reference/masster_feature_set_audit.md) | REFERENCE (2026-09-27) | Every feature reaching a MASSter parameter; fixes now default (section 11) |
| [feature_set_audit](../reference/feature_set_audit.md) | REFERENCE (2026-09-22) | FraGNNet feature-set audit (wired features) |
| [masster_smarts_rules](../reference/masster_smarts_rules.md) | REFERENCE (2026-10-01) | SMARTS rule catalog; 15 registered, off by default |
| [dataloader_data](../reference/dataloader_data.md) | REFERENCE (2026-09-08) | Data mapping definitions and weighted CE input contract |
| [masster_generalization_methods_2026_09_27](../notes/masster_generalization_methods_2026_09_27.md) | NOTE (2026-09-28) | Literature on practical generalization methods for MASSter |

## Key artifacts
- Control and arm configs: `config/masster/ma_mi_merged/` (`..._c4a_base_s1.yml`, `_s2`, `_c4a_base_s1_trainaux.yml`, `_c4d_wd10.yml`; subfolders `attention_2026_09_21`, `candidate_token_2026_09_22`, `encoder_simplify_2026_09_23`, `fragment_selection_2026_09_23`, `intensity_escape_2026_09_22`, `loss_objective_2026_09_21`, `lr_decay_2026_09_22`).
- W&B runs cited as current: control tail `0.7270` (`c4a_base_s1`/`s2`), `a1_prenorm` `q4t4udee`, `d3` `rcfzf67p`, horizon probe `tkkesus9` (630 epochs), duplicate `a2` pair `xyessyn1`/`cgomm8v4` (reproducibility floor), `g1` `xb9ic2pp`, `p1` `s5i5qqlz`, `p3` `w1dj2ahw`, `h1_hellinger_closest` `oykkffls`, `wd10` (2026-09-19) `kalc4mzo`.
- Cluster generators: `scripts/manual_slurm/fir/submit_masster_{attention,candidate_token,encoder_simplify,feature_arms,fragment_selection,intensity_escape,loss_objective,proposal_early_stop,regularization}.sh`.
- Diagnostics: `scripts/debug/masster_core/` (cut descriptors, unexplained-peak split, headroom probes), `scripts/debug/masster_core/masster_ion_h_top_k_sweep.py`, `scripts/debug/masster_memorization/probe_masster_regularization.py`.
- Code: `src/fragnnet/models/masster/masster_model.py`, `src/fragnnet/models/masster/masster_pl.py`, the heads and helpers beside them (`candidate_transformer.py`, `proposal_heads.py`, `component_pooling.py`, `pathway_probabilities.py`); schema in `config/template.yml`.
- History logs: [masster_plan_log](../history/masster_plan_log.md), [ion/H plan log](../history/masster_ion_h_channel_selection_plan_log.md).
