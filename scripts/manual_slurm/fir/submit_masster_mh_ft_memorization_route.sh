#!/bin/bash
# Generate (but do not submit) the Fir jobs for the memorization-route Step 1 arms, 2026-09-24.
#
# Question: the memorization probe put MASSter's per-molecule allocation lookup in the boundary
# block, but every candidate input comes from the same globally attended atom embeddings, so a
# blocked input can re-form the key from the others. Step 1 measures rerouting and tries fixes that
# act at the source or do not depend on the route. Each arm is c0 plus its own keys:
#   c0   same-code control                  (f0 + naming + candidate coverage monitoring)
#   r0   boundary block removed             (rerouting test; candidate_boundary_block + bbe off)
#   r1   local cut-site encoder             (masster_boundary_local_layers: 1)
#   r2   encoder without global attention   (mol_gnn_type: GINE)
#   k1   dropout on the key-forming layers  (boundary_encoder_dropout, intensity_input_dropout 0.2)
#   m1   per-molecule memorization sink     (sink_dim 16, 35,025 rows, from epoch 4; training only)
# Added 2026-09-25, after the Step 1 read: training-only noise on the encoder's global-attention
# message, where the route map found the key is written. Read against the c0 that already ran:
#   gm1  global-message substitution        (masster_global_message_substitution: 0.3)
#   gm2  Gaussian global-message noise      (masster_global_message_noise_std: 0.5)
# and swap noise on the same message, a feature-level CutMix that keeps the label: a swapped message
# is another molecule's real message (masster_global_message_swap / _swap_unit):
#   gm3  swap per atom                      (0.3, atom)
#   gm4  swap per molecule                  (0.3, molecule)
#   gm5  swap per feature entry             (0.3, feature)
#   gm6  swap per molecule, higher rate     (0.6, molecule)
# and the stacking and seed-replicate arms for the two levers that passed (k1 and weight EMA):
#   k1_ema  k1 + ema_decay 0.999            (control: u2_ema999, c8hh9j1b, already ran)
#   c0_s2   c0 at seed 811237               (control for the two replicates below)
#   k1_s2   k1 at seed 811237               (control: c0_s2)
#   u2_s2   c0_s2 + ema_decay 0.999         (control: c0_s2)
# The three seed-811237 configs do NOT upload checkpoints: they are read on W&B only.
# Added 2026-09-25, plan decision C: the d3 learning-rate schedule (1000-step warmup, cosine to
# 1% of peak over 19,712 steps), where every run above trains at a constant 1e-3:
#   s1         c0 + schedule                (control: c0, vcpho5qm)
#   s1_k1_ema  k1_ema + schedule            (control: k1_ema, j4u25qfd)
# and weight decay that acts within the run (timescale 1/(lr*wd): ~325 epochs at 0.01, ~32 at 0.1):
#   w1         c0 + weight_decay 0.1        (control: c0, vcpho5qm)
# Added 2026-09-26: the NCE input on the breakdown-curve axis instead of the abs-sines code, which
# aliases NCE x with 256 - x and gives unseen energies arbitrary codes:
#   n1         c0 + ce_insert_type qet      (control: c0, vcpho5qm)
# and, after n1 trailed c0 by 0.0124 at epochs 10-13 (mid-run read):
#   n2         n1 + ce_insert_type qet_mlp  (controls: n1 jxczo2kf, c0 vcpho5qm)
#   n3         c0 + ce_insert_type id       (controls: c0 vcpho5qm, n1 jxczo2kf)
# and, after n1-n3 all trailed c0 with train fit unchanged (keep the per-NCE code, fix its table):
#   n4         c0 + int_embedder fourier    (control: c0 vcpho5qm)
# and m1's sink on a random half of the training spectra (MemSinks mixing; m1 failed at -0.0203):
#   m2         m1 + sink dropout 0.5        (controls: c0 vcpho5qm, m1 liwjgdq6)
# Added 2026-09-26, after the side-of-cut probe: each cut represented by its lighter side (the
# neutral loss of a large fragment), the heavier side by element counts only (c_heavy_side):
#   nl1  c0 + masster_light_side_token      (control: c0, vcpho5qm)
#   nl2  nl1 + mol_gnn_type GINE            (controls: c0, r2 9lt5s2u8)
#   nl3  nl2 + k1's dropout                 (controls: nl2, k1 wg0nxymi)
# Added 2026-09-26, after the shared-proposal gate: one CE-blind proposal per molecule. Sharing needs
# the CE-context batch sampler, which batches whole molecules, so sp0 prices the sampler alone:
#   sp0  c0 + CE-context batches            (control: c0, vcpho5qm)
#   sp1  sp0 + share_proposal_across_ev     (control: sp0)
# Added 2026-09-27, after the memorization-attempts audit: training-only structural noise on the
# shared encoder's inputs (noise with every route left in place, as k1 and gm1):
#   am1  c0 + encoder atom mask 0.1          (control: c0, vcpho5qm)
#   de1  c0 + encoder bond dropout 0.1       (control: c0, vcpho5qm)
# Added 2026-09-27, after the allocator-route probe (the set message is the most train-specific
# route, and where the lookup goes when global attention is removed):
#   sm1  c0 + set-message substitution 0.3   (control: c0, vcpho5qm)
#   sd1  c0 + set dropout on the keys 0.3    (control: c0, vcpho5qm; read against sm1 too)
# Added 2026-09-28, after i1_inst_ob_obf (i50ig7rf, n4 on the OB / OB_F instrument labels) led n4
# by +0.021 mid-run: the replicated levers stacked on it, then the merged screen's pre-norm. These
# need the relabelled data/proc/nist20/spec_df.pkl on the cluster (md5 bdc009bb..., see i1's header):
#   i3   i1 + k1 dropout + ema_decay 0.999  (control: i1, i50ig7rf)
#   i4   i3 + intensity pre-norm             (control: i3, same stage)
# Added 2026-09-29: full candidate attention on i1, which has only run merged (a2_full_attention
# +0.0030), then the m/z positional encoding on it (the user's request of 2026-09-27; the local
# fa_h4_mz run stopped at epoch 18). Same relabelled data as i1:
#   fa1  i1 + full candidate attention       (control: i1, i50ig7rf)
#   mz1  fa1 + m/z positional encoding        (control: fa1, same stage)
# Plan: docs/plans/masster/masster_memorization_route_plan.md
# n1 finding: docs/findings/masster/masster_qet_projection_candidate_union_2026_09_26.md
# nl finding: docs/findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md
# sp gate: the n1 finding, section "What a shared proposal would rank".
#
# STAGE:
#   all   (default) c0, r0, r1, r2, k1 and m1.
#   arms  r0, r1, r2, k1 and m1, when c0 is already running.
#   gm    gm1 and gm2 (c0 vcpho5qm has already run).
#   swap  gm3, gm4, gm5 and gm6 (c0 vcpho5qm has already run).
#   batch2  gm1-gm6, k1_ema, c0_s2, k1_s2 and u2_s2.
#   lr    s1 and s1_k1_ema (c0 and k1_ema have run or are running).
#   nl    nl1, nl2 and nl3 (c0, r2 and k1 have run).
#   nce   n2 and n3 (c0 has run; n1 is running).
#   share sp0 and sp1 (c0 has run).
#   drop  am1 and de1 (c0 has run).
#   alloc sm1 and sd1 (c0 has run).
#   stack i3 and i4 (i1 has run or is running).
#   attn  fa1 and mz1 (i1 has run).
#   c0 | r0 | r1 | r2 | k1 | m1 | gm1 ... gm6 | k1_ema | c0_s2 | k1_s2 | u2_s2 | s1 | s1_k1_ema | w1
#   | m2 | n1 | n2 | n3 | n4 | nl1 | nl2 | nl3 | sp0 | sp1 | am1 | de1 | sm1 | sd1 | i3 | i4
#   | fa1 | mz1  one job.
#
# THE CONTROL IS c0, NOT 3bej6fq7. The code has moved since the f0 run, and c0 also logs candidate
# coverage, so submit c0 with the arms.
#
# READING IT (plan § Objective and acceptance): the tail mean of val_cos_sim_0.01_epoch/mean over
# epochs 54-63 against c0 (+0.003 to pass); val candidate_wrecall within 0.005 of c0 (read it
# FIRST for r2, whose proposal reads the changed encoder); then offline, per best checkpoint, the
# ranking decomposition of val and a 25.6k train sample, the Step 0 route map, and val cosine by
# nearest-train Tanimoto bin. upload_checkpoints is true in every config except the seed
# replicates (c0_s2, k1_s2, u2_s2).
#
# Profile fir:h100-3g.40gb, as for the f0/f1 pair these configs derive from: train-side auxiliary
# metrics are off (peak ~26.8 GiB). Wall time 5:00:00: all six Step 1 runs, coverage metrics on,
# finished 64 epochs in 3.2-3.3 h (batch 2 runs at 3.0 min/epoch), so 5 h is ~1.5x headroom, and
# j=<job_name> makes a requeued job resume from last.ckpt.

set -euo pipefail

STAGE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_mh_ft_memorization_route"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/masster/mh_ft_memorization_route_2026_09_24"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"
TIME="5:00:00"

declare -A JOB_BY_ARM=(
    [c0]=msh_mhft_c0_control
    [r0]=msh_mhft_r0_no_boundary
    [r1]=msh_mhft_r1_local_cut
    [r2]=msh_mhft_r2_no_gsa
    [k1]=msh_mhft_k1_key_dropout
    [m1]=msh_mhft_m1_sink
    [m2]=msh_mhft_m2_sink_dropout
    [gm1]=msh_mhft_gm1_message_substitution
    [gm2]=msh_mhft_gm2_message_noise
    [gm3]=msh_mhft_gm3_swap_atom
    [gm4]=msh_mhft_gm4_swap_molecule
    [gm5]=msh_mhft_gm5_swap_feature
    [gm6]=msh_mhft_gm6_swap_molecule_p06
    [k1_ema]=msh_mhft_k1_ema
    [c0_s2]=msh_mhft_c0_s2
    [k1_s2]=msh_mhft_k1_s2
    [u2_s2]=msh_mhft_u2_s2
    [s1]=msh_mhft_s1_lr_cosine
    [s1_k1_ema]=msh_mhft_s1_k1_ema
    [w1]=msh_mhft_w1_wd10
    [n1]=msh_mhft_n1_nce_qet
    [n2]=msh_mhft_n2_nce_qet_mlp
    [n3]=msh_mhft_n3_nce_id
    [n4]=msh_mhft_n4_nce_fourier
    [nl1]=msh_mhft_nl1_light_side
    [nl2]=msh_mhft_nl2_light_side_gine
    [nl3]=msh_mhft_nl3_light_side_gine_k1
    [sp0]=msh_mhft_sp0_ce_context_batches
    [sp1]=msh_mhft_sp1_shared_proposal
    [am1]=msh_mhft_am1_atom_mask
    [de1]=msh_mhft_de1_bond_drop
    [sm1]=msh_mhft_sm1_set_message_substitution
    [sd1]=msh_mhft_sd1_set_key_dropout
    [i3]=msh_mhft_i3_k1_ema_ob_obf
    [i4]=msh_mhft_i4_k1_ema_prenorm_ob_obf
    [fa1]=msh_mhft_fa1_full_attention_ob_obf
    [mz1]=msh_mhft_mz1_full_attention_mz_ob_obf
)
declare -A CONFIG_BY_ARM=(
    [c0]=c0_control.yml
    [r0]=r0_no_boundary.yml
    [r1]=r1_local_cut.yml
    [r2]=r2_no_gsa.yml
    [k1]=k1_key_dropout.yml
    [m1]=m1_sink.yml
    [m2]=m2_sink_dropout.yml
    [gm1]=gm1_message_substitution.yml
    [gm2]=gm2_message_noise.yml
    [gm3]=gm3_swap_atom.yml
    [gm4]=gm4_swap_molecule.yml
    [gm5]=gm5_swap_feature.yml
    [gm6]=gm6_swap_molecule_p06.yml
    [k1_ema]=k1_ema.yml
    [c0_s2]=c0_s2_control.yml
    [k1_s2]=k1_s2.yml
    [u2_s2]=u2_s2_ema999.yml
    [s1]=s1_lr_cosine.yml
    [s1_k1_ema]=s1_k1_ema_lr_cosine.yml
    [w1]=w1_wd10.yml
    [n1]=n1_nce_qet.yml
    [n2]=n2_nce_qet_mlp.yml
    [n3]=n3_nce_id.yml
    [n4]=n4_nce_fourier.yml
    [nl1]=nl1_light_side.yml
    [nl2]=nl2_light_side_gine.yml
    [nl3]=nl3_light_side_gine_k1.yml
    [sp0]=sp0_ce_context_batches.yml
    [sp1]=sp1_shared_proposal.yml
    [am1]=am1_atom_mask.yml
    [de1]=de1_bond_drop.yml
    [sm1]=sm1_set_message_substitution.yml
    [sd1]=sd1_set_key_dropout.yml
    [i3]=i3_k1_ema_ob_obf.yml
    [i4]=i4_k1_ema_prenorm_ob_obf.yml
    [fa1]=fa1_full_attention_ob_obf.yml
    [mz1]=mz1_full_attention_mz_ob_obf.yml
)
# Seed replicates are read on W&B only, so they must not upload checkpoints.
declare -A NO_CKPT_UPLOAD=([c0_s2]=1 [k1_s2]=1 [u2_s2]=1)

case "${STAGE}" in
    all)  ARMS=(c0 r0 r1 r2 k1 m1) ;;
    arms) ARMS=(r0 r1 r2 k1 m1) ;;
    gm)   ARMS=(gm1 gm2) ;;
    swap) ARMS=(gm3 gm4 gm5 gm6) ;;
    batch2) ARMS=(gm1 gm2 gm3 gm4 gm5 gm6 k1_ema c0_s2 k1_s2 u2_s2) ;;
    lr)   ARMS=(s1 s1_k1_ema) ;;
    nl)   ARMS=(nl1 nl2 nl3) ;;
    nce)  ARMS=(n2 n3) ;;
    share) ARMS=(sp0 sp1) ;;
    drop) ARMS=(am1 de1) ;;
    alloc) ARMS=(sm1 sd1) ;;
    stack) ARMS=(i3 i4) ;;
    attn) ARMS=(fa1 mz1) ;;
    c0|r0|r1|r2|k1|m[12]|gm[1-6]|k1_ema|c0_s2|k1_s2|u2_s2|s1|s1_k1_ema|w1|n[1-4]|nl[1-3]|sp[01]|am1|de1|sm1|sd1|i[34]|fa1|mz1) ARMS=("${STAGE}") ;;
    *)
        echo "Usage: $0 [all|arms|gm|swap|batch2|lr|nl|nce|share|drop|alloc|stack|attn|c0|r0|r1|r2|k1|m1|m2|gm1-gm6|k1_ema|c0_s2|k1_s2|u2_s2|s1|s1_k1_ema|w1|n1-n4|nl1-nl3|sp0|sp1|am1|de1|sm1|sd1|i3|i4|fa1|mz1]" >&2
        exit 1
        ;;
esac

cd "${REPO_ROOT}"

# 1. Validate before creating any job script.
for arm in "${ARMS[@]}"; do
    config_fp="${CONFIG_DIR}/${CONFIG_BY_ARM[${arm}]}"
    if [[ ! -f "${config_fp}" ]]; then
        echo "Missing config: ${config_fp}" >&2
        exit 1
    fi
    # c0 and every arm must share the horizon for the matched tail-mean read.
    if ! grep -q '^max_epochs: 64$' "${config_fp}"; then
        echo "${config_fp} is not at max_epochs: 64 -- it would not match c0." >&2
        exit 1
    fi
    # Train metrics on would add a key against c0 and push peak GPU past a 3g slice.
    if ! grep -q '^compute_train_auxiliary_metrics: false$' "${config_fp}"; then
        echo "${config_fp} must keep compute_train_auxiliary_metrics: false (c0 match, 3g fit)." >&2
        exit 1
    fi
    # The offline reads (ranking decomposition, route map) need the checkpoints off the cluster;
    # the seed replicates are read on W&B only and keep theirs on Fir.
    if [[ -n "${NO_CKPT_UPLOAD[${arm}]:-}" ]]; then
        if ! grep -q '^upload_checkpoints: false$' "${config_fp}"; then
            echo "${config_fp} is a seed replicate -- it must set upload_checkpoints: false." >&2
            exit 1
        fi
    elif ! grep -q '^upload_checkpoints: true$' "${config_fp}"; then
        echo "${config_fp} must set upload_checkpoints: true -- the read needs the checkpoints." >&2
        exit 1
    fi
    # Acceptance criterion 5 reads candidate coverage off W&B, for c0 and every arm alike.
    if ! grep -q '^masster_candidate_coverage_metrics: true$' "${config_fp}"; then
        echo "${config_fp} must log candidate coverage (masster_candidate_coverage_metrics)." >&2
        exit 1
    fi
done
if [[ ! -d data/split/nist20_inchikey ]]; then
    echo "Missing data/split/nist20_inchikey -- the [M+H]+/FT split these configs point at." >&2
    exit 1
fi
# The generated job installs both archives from the checkout; wheels/ is Git-ignored, so a fresh
# cluster clone does not have them and the failure would otherwise appear only after job start.
for wheel in wheels/pyteomics-4.7.5-py3-none-any.whl wheels/torch_linear_assignment-0.0.6.tar.gz; do
    if [[ ! -f "${wheel}" ]]; then
        echo "Missing ${wheel} -- the generated job installs it and wheels/ is Git-ignored." >&2
        exit 1
    fi
done

# 2. Generate the Fir job scripts.
for arm in "${ARMS[@]}"; do
    python scripts/setup_job_cc.py \
        --job_name "${JOB_BY_ARM[${arm}]}" \
        --python_script_fp scripts/run_pl_model_fit.py \
        --frag_gnn_dp "${REPO_ROOT}" \
        --script_dp "${SCRIPT_DIR}" \
        --job_dp "${JOB_DIR}" \
        --gpu_profile "${PROFILE}" \
        --time "${TIME}" \
        --cc_account "${ACCOUNT}" \
        --cc_email "${EMAIL}" \
        --script_kwargs w=online c="${CONFIG_DIR}/${CONFIG_BY_ARM[${arm}]}" j="${JOB_BY_ARM[${arm}]}" \
        --no_submit=True
done

echo
echo "Generated scripts hold a W&B credential -- keep them out of shared records."
echo "Review, then submit from a Fir login node:"
for arm in "${ARMS[@]}"; do
    echo "  bash -n ${SCRIPT_DIR}/${JOB_BY_ARM[${arm}]}.sh"
done
for arm in "${ARMS[@]}"; do
    echo "  sbatch ${SCRIPT_DIR}/${JOB_BY_ARM[${arm}]}.sh"
done
echo
echo "The control is c0 (same code); submit it with the arms unless it is already running."
echo "batch2: c0 (vcpho5qm) and u2 (c8hh9j1b) have run; c0_s2 is in the batch."
echo "lr: s1's control c0 (vcpho5qm) has run; s1_k1_ema's control k1_ema (j4u25qfd) is in batch 2."
echo "w1: its control c0 (vcpho5qm) has run."
echo "n1: its control c0 (vcpho5qm) has run."
echo "nce (n2, n3): c0 (vcpho5qm) has run; n1 (jxczo2kf) is the other control."
echo "n4: its control c0 (vcpho5qm) has run."
echo "m2: its controls c0 (vcpho5qm) and m1 (liwjgdq6) have run."
echo "nl: its controls c0 (vcpho5qm), r2 (9lt5s2u8) and k1 (wg0nxymi) have run."
echo "share: sp0's control c0 (vcpho5qm) has run; sp1's control is sp0, in the same stage."
echo "drop (am1, de1): their control c0 (vcpho5qm) has run."
echo "alloc (sm1, sd1): their control c0 (vcpho5qm) has run; read each against c0 and each other."
echo "stack (i3, i4): i3's control is i1 (i50ig7rf); i4's is i3, in the same stage. Both need the"
echo "  relabelled OB / OB_F spec_df.pkl on the cluster (md5 bdc009bbeaf6a38d126326d127630cad)."
echo "attn (fa1, mz1): fa1's control is i1 (i50ig7rf); mz1's is fa1, in the same stage. Both need"
echo "  the same relabelled spec_df.pkl as i1."
