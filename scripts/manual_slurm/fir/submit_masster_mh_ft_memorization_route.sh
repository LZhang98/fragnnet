#!/bin/bash
# Generate (but do not submit) the Fir jobs for the memorization-route arms on the ORB / ORB_F
# instrument labels (data/proc/nist20/spec_df.pkl relabelled 2026-09-28; see i1's header).
#
# The FT-label arms of 2026-09-24 to 2026-09-27 (c0, r0-r2, k1, m1/m2, gm1-gm6, k1_ema, the seed-811237
# replicates c0_s2/k1_s2/u2_s2, s1, s1_k1_ema, w1, n1-n4, nl1-nl3, sp0/sp1, am1/de1, sm1/sd1) were
# removed with their configs on 2026-09-29: on FT labels the identity routes stood in for the missing
# instrument, and the relabelled spec_df.pkl no longer reproduces them. Their configs and stages are
# in Git at 8c52c6d3; their results are in docs/plans/masster/masster_memorization_route_plan.md and
# docs/findings/masster/masster_memorization_attempts_audit_2026_09_27.md. The route arms among them
# are rerun below (stage rerun).
#
# Added 2026-09-28, after i1_inst_ob_obf (i50ig7rf, n4 on the ORB / ORB_F instrument labels) led n4
# by +0.021 mid-run: the replicated levers stacked on it, then the merged screen's pre-norm. These
# need the relabelled data/proc/nist20/spec_df.pkl on the cluster (md5 ce029870..., see i1's header):
#   i3   i1 + k1 dropout + ema_decay 0.999  (control: i1, i50ig7rf)
#   i4   i3 + intensity pre-norm             (control: i3, same stage)
# Added 2026-09-29: full candidate attention on i1, which has only run merged (a2_full_attention
# +0.0030), then the m/z positional encoding on it (the user's request of 2026-09-27; the local
# fa_h4_mz run stopped at epoch 18). Same relabelled data as i1:
#   fa1  i1 + full candidate attention       (control: i1, i50ig7rf)
#   mz1  fa1 + m/z positional encoding        (control: fa1, same stage)
# Added 2026-09-29: the recorded eV as a second CE input on i1, a missing eV left unfilled
# (spec_params.fill_missing_ace: false, new that day). Same relabelled data as i1:
#   i5   i1 + dual CE, no eV fill             (control: i1, i50ig7rf)
# Added 2026-09-29, after mz1 lost to fa1 (-0.0017, train fit up, val down): the neutral loss from
# the precursor, M - m, as a second position (masster_intensity_nl_positional_encoding, new that
# day). Same relabelled data as i1:
#   mznl1  fa1 + m/z and neutral-loss encodings, half the width each  (control: fa1, 0vn8m2gr)
#   nlpe1  fa1 + neutral-loss encoding alone                           (control: fa1, 0vn8m2gr)
# Added 2026-09-29: a mass-window ladder on fa1, restricting who each candidate attends to (the
# final softmax stays global), with no m/z encoding. The control for chemistry-defined groupings:
#   mw30  fa1 + 30 Da mass window                                      (control: fa1, 0vn8m2gr)
#   mw10  fa1 + 10 Da mass window                                      (control: fa1, 0vn8m2gr)
#   mlp1  fa1 with no candidate attention (window-0 end)               (control: fa1, 0vn8m2gr)
# Added 2026-09-29: attention masked to the candidate graph (direct containment and complements;
# the precursor receives but never sends), the chemistry-defined counterpart of the window ladder:
#   cg1   fa1 + candidate-graph attention                              (control: fa1, 0vn8m2gr)
# Added 2026-09-29: the FT-label route arms rerun on ORB/ORB_F, where the identity routes no longer
# stand in for the instrument (i1 memorization read). Each is i1 (or fa1) plus the FT arm's keys:
#   r2_ob  i1 without global attention (mol_gnn_type: GINE)            (control: i1, i50ig7rf)
#   r1_ob  i1 + boundary block from a local encoder                    (control: i1, i50ig7rf)
#   r0_ob  i1 without the boundary block                               (control: i1, i50ig7rf)
#   nl1_ob i1 + light-side token (+ c_heavy_side)                      (control: i1, i50ig7rf)
#   nl2_ob nl1_ob without global attention                             (control: i1, i50ig7rf)
#   sd1_ob i1 + allocator set key dropout 0.3                          (control: i1, i50ig7rf)
#   fa1_s2 fa1 at seed 811237 (control replicate)                      (control: fa1, 0vn8m2gr)
# Added 2026-09-30: NCE into the molecule encoder, which is CE-blind in i1 (FraGNNet's `mol`
# insertion beats `mlp` on unmerged data: +0.0022 to +0.0136). The user asked for it attended or
# as a condition:
#   ce1  i1 + context concatenated to the encoder input (condition)   (control: i1, i50ig7rf)
#   ce2  i1 + context key/value in each global-attention layer         (control: i1, i50ig7rf)
# Added 2026-09-30: SMARTS templates on i1, after i1's worst-prediction read found sulfonamide tail
# spectra missing rearranged candidates and substituted_aryl_sulfonamide_o_transfer was registered:
#   so1  i1 + masster_smarts_templates                               (control: i1, i50ig7rf)
# Added 2026-10-15: rewire rules after selection on i1 (config/masster/mh_rewire_after_selection_2026_10_15/;
# plan docs/plans/masster/masster_rewire_rules_training_plan.md, G0b in output/rewire_g0b_2026_10_15/):
#   a0      i1 + 17 admitted rewire rules on the kept pool, depth 0, extras capped at 0.5 x kept fragments  (control: i1, i50ig7rf)
#   a0d1    a0 + depth-1 continuation, top-4 bond cuts per product (masster_rewire_continuation_cuts: 4)    (control: i1; read vs a0)
#   a0null  a0 with random same-shape edits (masster_rewire_null: true)                                     (control: i1; read vs a0)
# Plan: docs/plans/masster/masster_memorization_route_plan.md
# nl finding: 731fe493:docs/findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md, removed 2026-10-18
#
# STAGE (required):
#   stack i3 and i4 (i1 has run or is running).
#   attn  fa1 and mz1 (i1 has run).
#   pos   mznl1 and nlpe1 (fa1 and mz1 have run).
#   window  mw30, mw10 and mlp1 (fa1 has run).
#   graph   cg1 (fa1 has run).
#   rerun   r2_ob, r1_ob, r0_ob, nl1_ob, nl2_ob, sd1_ob and fa1_s2 (i1 and fa1 have run).
#   ce      ce1 and ce2 (i1 has run).
#   smarts  so1 (i1 has run).
#   rewire  a0, a0d1 and a0null (i1 has run; needs a checkout with masster_rewire_rules, 2026-10-15).
#   i3 | i4 | fa1 | mz1 | i5 | mznl1 | nlpe1 | mw30 | mw10 | mlp1 | cg1 | r2_ob | r1_ob | r0_ob
#   | nl1_ob | nl2_ob | sd1_ob | fa1_s2 | ce1 | ce2 | so1 | a0 | a0d1 | a0null  one job.
#
# CONTROLS: i1_inst_ob_obf (i50ig7rf, tail 0.7607 over epochs 54-63) and fa1_full_attention_ob_obf
# (0vn8m2gr, tail 0.7651) have run; each arm's header names its control. READ: the tail mean of
# val_cos_sim_0.01_epoch/mean over epochs 54-63 against the control, +0.005 to count on one seed
# (seed spread 0.0042); val candidate_wrecall within 0.005 of the control; then the offline reads
# its header names. upload_checkpoints is false for cg1, the window ladder, the pos arms and the
# reruns (user, 2026-09-29: their checkpoints stay on Fir) and true for i3, i4, fa1, mz1 and i5.
#
# Profile fir:h100-3g.40gb, as for the f0/f1 pair this lineage derives from: train-side auxiliary
# metrics are off (peak ~26.8 GiB). Wall time 5:00:00: all six Step 1 runs, coverage metrics on,
# finished 64 epochs in 3.2-3.3 h (batch 2 runs at 3.0 min/epoch), so 5 h is ~1.5x headroom, and
# j=<job_name> makes a requeued job resume from last.ckpt.

set -euo pipefail

STAGE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_mh_ft_memorization_route"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/masster/mh_ft_memorization_route_2026_09_24"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"
TIME="5:00:00"

declare -A JOB_BY_ARM=(
    [i3]=msh_mhft_i3_k1_ema_ob_obf
    [i4]=msh_mhft_i4_k1_ema_prenorm_ob_obf
    [fa1]=msh_mhft_fa1_full_attention_ob_obf
    [mz1]=msh_mhft_mz1_full_attention_mz_ob_obf
    [i5]=msh_mhft_i5_dual_ce_ob_obf
    [mznl1]=msh_mhft_mznl1_full_attention_mz_nl_ob_obf
    [nlpe1]=msh_mhft_nlpe1_full_attention_nl_ob_obf
    [mw30]=msh_mhft_mw30_mass_window30_ob_obf
    [mw10]=msh_mhft_mw10_mass_window10_ob_obf
    [mlp1]=msh_mhft_mlp1_no_attention_ob_obf
    [cg1]=msh_mhft_cg1_candidate_graph_ob_obf
    [r2_ob]=msh_mhft_r2_no_gsa_ob_obf
    [r1_ob]=msh_mhft_r1_local_cut_ob_obf
    [r0_ob]=msh_mhft_r0_no_boundary_ob_obf
    [nl1_ob]=msh_mhft_nl1_light_side_ob_obf
    [nl2_ob]=msh_mhft_nl2_light_side_gine_ob_obf
    [sd1_ob]=msh_mhft_sd1_set_key_dropout_ob_obf
    [fa1_s2]=msh_mhft_fa1_s2_full_attention_ob_obf
    [ce1]=msh_mhft_ce1_context_gnn_ob_obf
    [ce2]=msh_mhft_ce2_context_token_ob_obf
    [so1]=msh_mhft_so1_smarts_o_transfer_ob_obf
    [a0]=msh_mhft_a0_rewire_after_selection_ob_obf
    [a0d1]=msh_mhft_a0d1_rewire_after_selection_d1_ob_obf
    [a0null]=msh_mhft_a0null_rewire_after_selection_null_ob_obf
)
declare -A CONFIG_BY_ARM=(
    [i3]=i3_k1_ema_ob_obf.yml
    [i4]=i4_k1_ema_prenorm_ob_obf.yml
    [fa1]=fa1_full_attention_ob_obf.yml
    [mz1]=mz1_full_attention_mz_ob_obf.yml
    [i5]=i5_dual_ce_ob_obf.yml
    [mznl1]=mznl1_full_attention_mz_nl_ob_obf.yml
    [nlpe1]=nlpe1_full_attention_nl_ob_obf.yml
    [mw30]=mw30_mass_window30_ob_obf.yml
    [mw10]=mw10_mass_window10_ob_obf.yml
    [mlp1]=mlp1_no_attention_ob_obf.yml
    [cg1]=cg1_candidate_graph_ob_obf.yml
    [r2_ob]=r2_no_gsa_ob_obf.yml
    [r1_ob]=r1_local_cut_ob_obf.yml
    [r0_ob]=r0_no_boundary_ob_obf.yml
    [nl1_ob]=nl1_light_side_ob_obf.yml
    [nl2_ob]=nl2_light_side_gine_ob_obf.yml
    [sd1_ob]=sd1_set_key_dropout_ob_obf.yml
    [fa1_s2]=fa1_s2_full_attention_ob_obf.yml
    [ce1]=ce1_context_gnn_ob_obf.yml
    [ce2]=ce2_context_token_ob_obf.yml
    [so1]=so1_smarts_o_transfer_ob_obf.yml
    [a0]=a0_rewire_after_selection_ob_obf.yml
    [a0d1]=a0d1_rewire_after_selection_d1_ob_obf.yml
    [a0null]=a0null_rewire_after_selection_null_ob_obf.yml
)
# The rewire arms live in their own config directory; every other arm uses CONFIG_DIR.
REWIRE_CONFIG_DIR="config/masster/mh_rewire_after_selection_2026_10_15"
declare -A DIR_BY_ARM=([a0]="${REWIRE_CONFIG_DIR}" [a0d1]="${REWIRE_CONFIG_DIR}" [a0null]="${REWIRE_CONFIG_DIR}")
# cg1, the window ladder, the pos arms, the ORB/ORB_F reruns and the ce arms keep their checkpoints
# on Fir for now (user, 2026-09-29); the seed replicate fa1_s2 is read on W&B only.
declare -A NO_CKPT_UPLOAD=([cg1]=1 [mw30]=1 [mw10]=1 [mlp1]=1
    [mznl1]=1 [nlpe1]=1 [r2_ob]=1 [r1_ob]=1 [r0_ob]=1 [nl1_ob]=1 [nl2_ob]=1 [sd1_ob]=1 [fa1_s2]=1
    [ce1]=1 [ce2]=1)

case "${STAGE}" in
    stack) ARMS=(i3 i4) ;;
    attn) ARMS=(fa1 mz1) ;;
    pos)  ARMS=(mznl1 nlpe1) ;;
    window) ARMS=(mw30 mw10 mlp1) ;;
    graph) ARMS=(cg1) ;;
    rerun) ARMS=(r2_ob r1_ob r0_ob nl1_ob nl2_ob sd1_ob fa1_s2) ;;
    ce)   ARMS=(ce1 ce2) ;;
    smarts) ARMS=(so1) ;;
    rewire) ARMS=(a0 a0d1 a0null) ;;
    i[345]|fa1|mz1|mznl1|nlpe1|mw30|mw10|mlp1|cg1|r2_ob|r1_ob|r0_ob|nl1_ob|nl2_ob|sd1_ob|fa1_s2|ce1|ce2|so1|a0|a0d1|a0null) ARMS=("${STAGE}") ;;
    *)
        echo "Usage: $0 [stack|attn|pos|window|graph|rerun|ce|i3|i4|i5|fa1|mz1|mznl1|nlpe1|mw30|mw10|mlp1|cg1|r2_ob|r1_ob|r0_ob|nl1_ob|nl2_ob|sd1_ob|fa1_s2|ce1|ce2|smarts|so1|rewire|a0|a0d1|a0null]" >&2
        exit 1
        ;;
esac

cd "${REPO_ROOT}"

# 1. Validate before creating any job script.
for arm in "${ARMS[@]}"; do
    config_fp="${DIR_BY_ARM[${arm}]:-${CONFIG_DIR}}/${CONFIG_BY_ARM[${arm}]}"
    if [[ ! -f "${config_fp}" ]]; then
        echo "Missing config: ${config_fp}" >&2
        exit 1
    fi
    # Every arm must share its control's horizon for the matched tail-mean read.
    if ! grep -q '^max_epochs: 64$' "${config_fp}"; then
        echo "${config_fp} is not at max_epochs: 64 -- it would not match its control." >&2
        exit 1
    fi
    # Train metrics on would add a key against the control and push peak GPU past a 3g slice.
    if ! grep -q '^compute_train_auxiliary_metrics: false$' "${config_fp}"; then
        echo "${config_fp} must keep compute_train_auxiliary_metrics: false (control match, 3g fit)." >&2
        exit 1
    fi
    # Arms on the no-upload list keep their checkpoints on Fir (the seed replicate is read on W&B
    # only); the others upload them for the offline reads.
    if [[ -n "${NO_CKPT_UPLOAD[${arm}]:-}" ]]; then
        if ! grep -q '^upload_checkpoints: false$' "${config_fp}"; then
            echo "${config_fp} is on the no-upload list -- it must set upload_checkpoints: false." >&2
            exit 1
        fi
    elif ! grep -q '^upload_checkpoints: true$' "${config_fp}"; then
        echo "${config_fp} must set upload_checkpoints: true -- the read needs the checkpoints." >&2
        exit 1
    fi
    # The read takes candidate coverage off W&B, for every arm and its control alike.
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

# The rewire arms run <= 1.3x i1's 3.2-3.3 h (G0b: forward 1.13x, forward+backward 1.11x at fraction 0.5); a0d1
# is unmeasured at depth 1. Ask 6 h for them; j=<job_name> resumes a requeue.
for arm in "${ARMS[@]}"; do
    if [[ -n "${DIR_BY_ARM[${arm}]:-}" ]]; then
        TIME="6:00:00"
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
        --script_kwargs w=online c="${DIR_BY_ARM[${arm}]:-${CONFIG_DIR}}/${CONFIG_BY_ARM[${arm}]}" j="${JOB_BY_ARM[${arm}]}" \
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
echo "stack (i3, i4): i3's control is i1 (i50ig7rf); i4's is i3, in the same stage. Both need the"
echo "  relabelled ORB / ORB_F spec_df.pkl on the cluster (md5 ce029870badc9e9e05935a4424f44c8d)."
echo "attn (fa1, mz1): fa1's control is i1 (i50ig7rf); mz1's is fa1, in the same stage. Both need"
echo "  the same relabelled spec_df.pkl as i1."
echo "i5: its control i1 (i50ig7rf) has run. It needs the relabelled spec_df.pkl and a checkout with"
echo "  spec_params.fill_missing_ace (base_dataset.py, 2026-09-29)."
echo "pos (mznl1, nlpe1): their control fa1 (0vn8m2gr) has run; read mznl1 against mz1 (tck857p7)"
echo "  too. They need the relabelled spec_df.pkl and a checkout with"
echo "  masster_intensity_nl_positional_encoding (bond_fragment_heads.py, 2026-09-29)."
echo "window (mw30, mw10, mlp1): their control fa1 (0vn8m2gr) has run; read against i1 (i50ig7rf)"
echo "  too. They need the relabelled spec_df.pkl; masster_intensity_mass_window is in HEAD."
echo "graph (cg1): its control fa1 (0vn8m2gr) has run; read against mw10/mw30 and i1 too. It needs"
echo "  the relabelled spec_df.pkl and a checkout with masster_intensity_candidate_graph"
echo "  (bond_fragment_heads.py candidate_graph_edges, 2026-09-29)."
echo "rerun (r2_ob, r1_ob, r0_ob, nl1_ob, nl2_ob, sd1_ob, fa1_s2): controls i1 (i50ig7rf) and, for"
echo "  fa1_s2, fa1 (0vn8m2gr) have run; read nl2_ob against r2_ob and nl1_ob too. They need the"
echo "  relabelled spec_df.pkl. Structural arms (r2_ob, r0_ob, nl1_ob, nl2_ob) shift the init RNG:"
echo "  read them against the 0.0042 seed spread."
echo "ce (ce1, ce2): their control i1 (i50ig7rf) has run; read each against the other too. They need"
echo "  the relabelled spec_df.pkl and a checkout with masster_encoder_context_token (nn_blocks.py"
echo "  GINE_GSA context_token_dim, 2026-09-30). ce1 shifts the init RNG; ce2 is init-matched to i1."
echo "rewire (a0, a0d1, a0null): their control i1 (i50ig7rf) has run; read a0d1 and a0null against a0 too."
echo "  They need the relabelled spec_df.pkl and a checkout with masster_rewire_rules (the 17-rule"
echo "  REWIRE_RULES list, src/fragnnet/models/masster/rewire_expansion.py). Time 6:00:00. Watch peak GPU"
echo "  on the 3g.40gb slice: i1 peaks near 26.8 GiB and these add up to +26% nodes per matching spectrum."
