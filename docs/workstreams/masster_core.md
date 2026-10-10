# MASSter core model on NIST20

> **Status:** HUB · **Updated:** 2026-10-10
> Architecture, training recipe, regularization, losses, candidate/proposal machinery, speed/GPU, the 128-epoch screens and campaign reads, ion-H channel selection, and feature/information audits for MASSter on NIST20.

## Where it stands

- **RT/CCS:** [RTHead and CCSHead](../plans/masster/masster_rt_ccs_plan.md) with independent label sets completed a new 10-epoch GPU debug run; full Enveda arm remains prepared, heads default off.
- Matched 128-epoch screen on the merged MA/MI base: control `c4a_base_s1` = **0.7270** `val_cos_sim` (tail mean epochs 118-127, replicate spread 0.0007, gate +-0.005), config `config/masster/ma_mi_merged/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_base_s1.yml`.
- Real levers: `a1_prenorm` (`q4t4udee`) **+0.0056**, the only clean single-key architecture pass; cosine LR decay with 1000-step warmup `d3` (`rcfzf67p`) 0.7357 (+0.0072 at epoch 121, n=1); the 2026-09-20 feature bundle is worth about +0.006/+0.007; EMA gained +0.0102 on `i1` (see generalization hub).
- Nulls/rejections at this screen: cut-set and candidate-token features (best +0.0032), intensity escape (+0.0000), proposal freeze/wd (+0.0010/+0.0020), fragment-marginal selection, encoder simplification `g1` (-0.0160), `b2` (-0.0062), mass pair bias (-0.0002).
- The train/val gap (0.151 cosine) is intensity allocation among peaks the model already proposes, not coverage; capacity is not the bottleneck. Work on that gap lives in the generalization hub.
- Prepared, not submitted: full-length `wd10` arm and matched `c4a_base_s1_trainaux` control (`fir:h100-80gb`); `a4`, `a5_ce_pair_bias`, `b4` (`config/masster/ma_mi_merged/attention_2026_09_21/`).
- Blocked: ion-H Phase 2 selection sweep needs a recoverable trained checkpoint.
- Working-tree changes are tracked in [project state](../project-state.md#uncommitted-working-tree-2026-10-08).
- Next action: none scheduled here; new core levers go through the `a1_prenorm` + LR-decay stack on `i1` ([generalization hub](masster_generalization.md)), and an `a1_prenorm` seed replicate is the cheapest open confirmation.

## Open threads

| Thread | Status | Next concrete step |
| --- | --- | --- |
| [RT/CCS heads](../plans/masster/masster_rt_ccs_plan.md) | DEBUG COMPLETE; full arm PREPARED | Matched Enveda multitask/control evaluation; coordinated in the [CASMI/Enveda hub](casmi_enveda.md) |
| [Architecture review todo](../plans/masster/masster_architecture_review_todo.md) | DEFERRED | `a1_prenorm` seed replicate; `a5` read against `a3`; RWSE needs the `random_walk_pe` fix ported |
| [Regularization sweep / `wd10`](../plans/masster/masster_regularization_sweep_plan.md) | PREPARED | Generate on Fir with `submit_masster_regularization.sh wd10` if the gap question is reopened |
| [Ion/H channel selection](../plans/masster/masster_ion_h_channel_selection_plan.md) | DEFERRED | Disable support gate, verify top-k 95/99 metric, recover a checkpoint, run `masster_ion_h_top_k_sweep.py` |
| [Attention/bond budget](../plans/masster/masster_attention_budget_2026_09_21.md) | PREPARED | Submit `a4`/`a5`/`b4` only if architecture work resumes |
| [Candidate backend](../plans/masster/masster_candidate_backend_plan.md) and [topology performance](../plans/masster/masster_topology_assembly_performance_plan.md) | LANDED / DEFERRED | Record L40S acceptance numbers if throughput is revisited |
| [Bond-cut EBM](../plans/masster/masster_bond_cut_ebm_plan.md), [EBM plan](../plans/masster/masster_energy_based_model_plan.md) | PROPOSED | Oracle excess over matched random pools before any implementation |
| [Intensity-head bucketing syncs](../plans/masster/masster_intensity_head_bucketing_sync_plan.md) | PROPOSED | Profile `isin`/`nonzero`/`searchsorted` overhead first |
| [Unused NIST peak annotations](../findings/unused_information_inventory_2026_09_22.md) | FINDING (open) | Decide whether to use per-peak formula assignments in the loss |
| [Cross-adduct paired loss](../findings/cross_adduct_loss_headroom_2026_09_17.md) | FINDING (gate half) | Needs a multi-adduct MASSter checkpoint |
| [Retained-fragment GNN](../plans/masster/masster_hierarchical_fragment_graph_plan.md) | DEFERRED | Candidate branch unbuilt; `fc1` was not positive |

## Settled — read before re-proposing
**Training recipe and architecture**
- [Epoch-128 arms](../findings/masster/masster_architecture_arms_epoch128_2026_09_22.md) — `a1_prenorm` +0.0056 passes; `b2` -0.0062, `g1` -0.0160 rejected; mass pair bias -0.0002; replicate floor on tail mean 0.00036.
- [Campaign read-out](../findings/masster/masster_campaign_readout_2026_09_21.md) — regularization sweep no promotions; bundle +0.006; val cosine saturates near epoch 100 (last 529 of 630 epochs buy +0.0017).
- [Overfit read-out](../findings/masster/masster_overfit_readout_2026_09_22.md) — gap 0.151 is ~98% allocation; three LR-decay arms clear the floor (`d3` +0.0072).
- [Regularization audit](../findings/masster/masster_regularization_audit_2026_09_19.md) — BatchNorm, not dropout, drives candidate-set train/eval divergence; atom decoder dropped (-0.0018, 1.8x faster).
- [Head-input asymmetry](../findings/masster/masster_head_input_asymmetry_2026_09_22.md) — proposal reads detached inputs; connecting it costs about -0.01 to -0.013; charge head inert in `c4a`.
- [Shared-proposal audit](../findings/masster/masster_shared_proposal_pathway_audit_2026_09_26.md) — the CE-context sampler was broken (-0.25 val cosine on `c0`), not the sharing.
- [Proposal early stop](../plans/masster/masster_proposal_early_stop_plan.md) — `p1` +0.0010, `p3` +0.0020; closed.
**Features and information**
- [Cut-set features](../plans/masster/masster_cut_set_features_plan.md) — five arms null (best +0.0032); adjacency is the only signed one.
- [Feature headroom](../findings/feature_headroom_2026_09_22.md) — descriptors recoverable at 0.93-1.00; only external quantum knowledge has headroom.
- [Feature/information audit](../findings/feature_information_audit_2026_09_22.md) — map of five parts; two conclusions withdrawn.
- [Identity supervision](../findings/masster/masster_identity_supervision_2026_09_22.md) — structural-teacher proposal retracted; coverage transfers (0.8975 vs 0.8911 `wrecall`).
- [Unexplained peaks](../findings/masster/unexplained_peak_split_2026_09_22.md) — ~11% unreachable intensity; only 2.91 points have no feasible subformula.
- [ChemFrag triage](../findings/masster/chemfrag_rearrangement_triage_2026_09_14.md) — none of 32 transforms adds unique coverage.
**Losses, proposal, intensity**
- [Supervised escape](../plans/masster/masster_supervised_escape_plan.md) — exact null (+0.0000 at n=2 vs n=2); not a ranking change.
- [Loss setup review](../findings/loss_setup_review_2026_09_13.md) and [proposal loss audit](../findings/masster/masster_proposal_loss_audit_2026_09_22.md) — proposal loss scales with covered fraction; CE kept as production.
- [Base-CE weight](../history/masster_plan_log.md#measured-finding-fixed-base-only-ce-trades-final-cosine-for-support-2026-08-20) — fixed 0.1/0.2/0.3 lowers cosine monotonically.
- Fragment-marginal selection is closed null (U128 ties U256; owned by the [MSGYM hub](msgym.md)).
**Speed and candidate machinery**
- [Training speed](../findings/masster/masster_training_speed_2026_09_17.md) — monitoring off 1.40x; CUDA construction about 1.9x per epoch; cut-bond encoder dedup cuts encoder FLOPs 20.1 to 2.1 GFLOP.
- [FastGNN / beam](../plans/masster/masster_fast_gnn_proposal.md) — beam captures flat 0.8695 mass; stopped.
- [Direct-loss batching](../plans/masster/masster_direct_loss_batching_plan.md) — closest 44x, hungarian 11x.
**Ion/H**
- [Ion/H plan summary](../plans/masster/masster_ion_h_channel_selection_plan.md) — gate gives no benefit (dense cosine 0.7630/0.7655/0.7631); selection never exercised.
**Neighbouring workstreams**: error analyses, memorization, SMARTS pathway, QET, ORB instruments in [generalization](masster_generalization.md); MSGYM and GLACIER in [msgym](msgym.md); datasets in [spectraverse](spectraverse.md), [nist23_ei](nist23_ei.md), [casmi_enveda](casmi_enveda.md); [fragnnet](fragnnet.md).

## All documents

- [MASSter RT/CCS heads](../plans/masster/masster_rt_ccs_plan.md) — PREPARED: implementation, label evidence and experiment protocol.
| Doc | Status (updated) | One-line content |
| --- | --- | --- |
| [MASSter RT/CCS heads](../plans/masster/masster_rt_ccs_plan.md) | PREPARED (2026-10-10) | Shared heads, label evidence, completed GPU debug training and full experiment |
| [masster_plan](../plans/masster/masster_plan.md) | LANDED (2026-10-06) | Core design record: one-pass bond-set model, mask enumeration, ion/H channels; snapshot of shipped behavior |
| [masster_ion_h_channel_selection_plan](../plans/masster/masster_ion_h_channel_selection_plan.md) | DEFERRED (2026-10-06) | Ion-mode / H-support / channel selection hierarchy; no trained arm exercised selection; Phase 2 blocked on a checkpoint |
| [masster_architecture_review_todo](../plans/masster/masster_architecture_review_todo.md) | DEFERRED (2026-10-06) | Open items from the 2026-09-19 code-vs-literature review (a4, a5, b4, RWSE, a1 replicate) |
| [masster_attention_budget_2026_09_21](../plans/masster/masster_attention_budget_2026_09_21.md) | PREPARED (2026-10-06) | Attention and bond-head budget options (default-off); `a4`/`a5`/`b4` never run |
| [masster_bond_cut_ebm_plan](../plans/masster/masster_bond_cut_ebm_plan.md) | PROPOSED (2026-09-13) | Bond-cut energy-based model design, stage 1; no implementation |
| [masster_campaign_4h_2026_09_21](../plans/masster/masster_campaign_4h_2026_09_21.md) | REFERENCE (2026-10-06) | Launch record of the 20-arm 2026-09-21 campaign (3 h boxes on Fir) |
| [masster_candidate_backend_plan](../plans/masster/masster_candidate_backend_plan.md) | LANDED (2026-10-06) | CPU-first, optional CUDA candidate construction; in production use |
| [masster_candidate_feature_list_plan](../plans/masster/masster_candidate_feature_list_plan.md) | LANDED (2026-10-06) | Ordered candidate-token feature list with one spec table |
| [masster_candidate_pathway_refactor](../plans/masster/masster_candidate_pathway_refactor.md) | LANDED (2026-08-29) | Five-layer refactor of candidate/pathway assembly, behavior-identical on 72 configs |
| [masster_cut_set_features_plan](../plans/masster/masster_cut_set_features_plan.md) | CLOSED NULL (2026-10-06) | Cut-set descriptors for the candidate token; five arms all null (adjacency +0.0032) |
| [masster_direct_loss_batching_plan](../plans/masster/masster_direct_loss_batching_plan.md) | LANDED (2026-09-19) | Batched direct peak-list losses: closest 44x, hungarian 11x faster |
| [masster_energy_based_model_plan](../plans/masster/masster_energy_based_model_plan.md) | PROPOSED (2026-09-13) | Bond-cut then spectrum EBM program; proposal only |
| [masster_fast_gnn_proposal](../plans/masster/masster_fast_gnn_proposal.md) | STOPPED (2026-10-06) | Connected-fragment beam proposal; stopped (flat 0.8695 mass vs 1.0000 cut_masks) |
| [masster_gpu_audit](../plans/masster/masster_gpu_audit.md) | REFERENCE (2026-09-17) | Static plus measured GPU profile and priorities |
| [masster_hierarchical_fragment_graph_plan](../plans/masster/masster_hierarchical_fragment_graph_plan.md) | DEFERRED (2026-09-29) | Retained-fragment GNN branch; `fc1` local screen not positive |
| [masster_improvement_plan](../plans/masster/masster_improvement_plan.md) | DEFERRED (2026-10-06) | Intensity/support-gating roadmap and cluster commands from 2026-09-17 |
| [masster_intensity_head_bucketing_sync_plan](../plans/masster/masster_intensity_head_bucketing_sync_plan.md) | PROPOSED (2026-09-14) | Remove bucketing host syncs in the intensity head; unmeasured payoff |
| [masster_nonprobabilistic_intensity_plan](../plans/masster/masster_nonprobabilistic_intensity_plan.md) | LANDED (2026-10-06) | Direct peak-list objectives (Hellinger, leak, CE auxiliary); CE stays production |
| [masster_proposal_early_stop_plan](../plans/masster/masster_proposal_early_stop_plan.md) | CLOSED NULL (2026-09-23) | Proposal freeze/weight-decay knobs `p1`/`p3`: both nulls |
| [masster_regularization_sweep_plan](../plans/masster/masster_regularization_sweep_plan.md) | PREPARED (2026-10-06) | One-hour regularization sweep (no promotions) and the prepared full-length `wd10` arm |
| [masster_rl_exploration_plan](../plans/masster/masster_rl_exploration_plan.md) | SUPERSEDED (2026-08-19) | Historical RL exploration analysis; superseded |
| [masster_supervised_escape_plan](../plans/masster/masster_supervised_escape_plan.md) | CLOSED NULL (2026-09-27) | Intensity escape for unreachable peaks; implemented, exact null |
| [masster_topology_assembly_performance_plan](../plans/masster/masster_topology_assembly_performance_plan.md) | DEFERRED (2026-10-06) | CPU topology assembly performance stages and gates |
| [one_pass_bond_fragment_testing_plan](../plans/masster/one_pass_bond_fragment_testing_plan.md) | DEFERRED (2026-10-06) | Verification tracker for the one-pass model (2026-08-14) |
| [chemfrag_rearrangement_triage_2026_09_14](../findings/masster/chemfrag_rearrangement_triage_2026_09_14.md) | FINDING (2026-09-28) | 32 rearrangement transforms triaged; none adds unique coverage |
| [connected_fragment_beam_reuse_and_equivalence_2026_09_17](../findings/masster/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md) | FINDING (2026-09-17) | Beam vs cut_masks reuse/equivalence algebra and candidate-only measurement |
| [connected_fragment_beam_speed_2026_09_17](../findings/masster/connected_fragment_beam_speed_2026_09_17.md) | FINDING (2026-09-17) | Synchronized GPU timing of the connected-fragment beam |
| [masster_architecture_arms_epoch128_2026_09_22](../findings/masster/masster_architecture_arms_epoch128_2026_09_22.md) | FINDING (2026-09-27) | 128-epoch architecture-arm read; reproducibility floor; `g1` rejection |
| [masster_campaign_readout_2026_09_21](../findings/masster/masster_campaign_readout_2026_09_21.md) | FINDING (2026-09-24) | 41-run W&B review; feature bundle +0.006; sweep no promotions |
| [masster_candidate_cut_descriptors_2026_09_22](../findings/masster/masster_candidate_cut_descriptors_2026_09_22.md) | FINDING (2026-09-22) | Cut descriptors on MASSter candidates; adjacency is the missing relation (48.9% ambiguity) |
| [masster_channel_selection_findings](../findings/masster/masster_channel_selection_findings.md) | FINDING (2026-09-05) | Historical channel-selection results (chemistry revalidation pending) |
| [masster_code_audit](../findings/masster/masster_code_audit.md) | REFERENCE (2026-09-13) | Candidate-attention A/B code audit; seven fixes |
| [masster_context_design_audit](../findings/masster/masster_context_design_audit.md) | REFERENCE (2026-09-08) | CE/context conditioning design audit and fixes |
| [masster_head_input_asymmetry_2026_09_22](../findings/masster/masster_head_input_asymmetry_2026_09_22.md) | FINDING (2026-09-24) | Proposal branch reads detached inputs in the merged lineage; charge head inert |
| [masster_identity_supervision_2026_09_22](../findings/masster/masster_identity_supervision_2026_09_22.md) | FINDING (2026-09-22) | 2.75 fragments share a formula; the structural-teacher recommendation is retracted |
| [masster_loss_audit_2026_09_19](../findings/masster/masster_loss_audit_2026_09_19.md) | FINDING (2026-09-19) | Loss-function audit and Hungarian/closest measurements |
| [masster_overfit_readout_2026_09_22](../findings/masster/masster_overfit_readout_2026_09_22.md) | FINDING (2026-09-22) | 25-run read: 0.151 train/val gap is ~98% intensity allocation, not coverage |
| [masster_proposal_loss_audit_2026_09_22](../findings/masster/masster_proposal_loss_audit_2026_09_22.md) | FINDING (2026-09-22) | Consolidated audit of the proposal coverage loss term |
| [masster_proposal_review_2026_09_13](../findings/masster/masster_proposal_review_2026_09_13.md) | FINDING (2026-09-13) | Proposal branch review: budgets, selection, diversity, mixtures |
| [masster_regularization_audit_2026_09_19](../findings/masster/masster_regularization_audit_2026_09_19.md) | FINDING (2026-09-19) | Regularization never varied; BatchNorm drives candidate-set train/eval divergence |
| [masster_shared_proposal_pathway_audit_2026_09_26](../findings/masster/masster_shared_proposal_pathway_audit_2026_09_26.md) | FINDING (2026-09-27) | CE-context sampler, not proposal sharing, was broken (sp0/sp1 collapse) |
| [masster_smarts_review_2026_09_13](../findings/masster/masster_smarts_review_2026_09_13.md) | FINDING (2026-09-13) | SMARTS rule set and literature review (26 rules) |
| [masster_training_speed_2026_09_17](../findings/masster/masster_training_speed_2026_09_17.md) | FINDING (2026-10-06) | Training speed: monitoring, CPU/CUDA candidates, batched coverage loss, cut-bond dedup |
| [masster_unmerged_overfitting_2026_09_16](../findings/masster/masster_unmerged_overfitting_2026_09_16.md) | FINDING (2026-09-16) | W&B read of the growing unmerged generalization gap |
| [unexplained_peak_split_2026_09_22](../findings/masster/unexplained_peak_split_2026_09_22.md) | FINDING (2026-09-22) | Unreachable ~11% intensity: 73% coverage, 27% escape; isotopologues 46.5% of escape |
| [loss_setup_review_2026_09_13](../findings/loss_setup_review_2026_09_13.md) | FINDING (2026-09-21) | Loss setup review with counterexamples and fixes |
| [feature_headroom_2026_09_22](../findings/feature_headroom_2026_09_22.md) | CLOSED NULL (2026-09-22) | Descriptor features closed null; only external quantum knowledge has headroom |
| [feature_information_audit_2026_09_22](../findings/feature_information_audit_2026_09_22.md) | FINDING (2026-09-27) | Map of the feature/information investigation and withdrawn conclusions |
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
- Diagnostics: `scripts/debug/masster_candidates/` (cut descriptors, unexplained-peak split, headroom probes), `scripts/eval/masster_ion_h_top_k_sweep.py`, `scripts/debug/masster_memorization/probe_masster_regularization.py`.
- Code: `src/fragnnet/model/masster_model.py`, `src/fragnnet/pl_model/masster_pl.py`, `src/fragnnet/model/bond_fragment_heads.py`; schema in `config/template.yml`.
- History logs: [masster_plan_log](../history/masster_plan_log.md), [ion/H plan log](../history/masster_ion_h_channel_selection_plan_log.md).
