"""G0b of the rewire-rules training plan: the production after-selection stage on i1's validation pool.

Runs the migrated i1 epoch-59 checkpoint on the nist20_inchikey validation split with
``masster_rewire_rules`` off (control), on at several ``masster_rewire_max_extra_fraction`` values
(continuation 0), on uncapped (the production reproduction of G0's R_after0), and with
``masster_rewire_null`` on, and measures per variant:

- nodes per spectrum (the ``ion_h`` rows the intensity head scores) and growth against the control,
  on the matching spectra (exact) and on a random whole-split sample (growth over the split);
- extras before the cap, dropped by the cap, kept, and the merge rate (product already in the pool);
- reach: the share of ALL validation rearrangement-peak intensity whose ion has a product (a row with
  a rewire indicator: a merged ordinary row or a kept extra) of the G0 composition within +-4 H, and the
  part that only a kept EXTRA supplies;
- forward wall-clock and peak GPU memory with and without the stage (warm match cache), and one
  forward+backward time ratio.

Run from the repository root:
    python scripts/debug/smarts/debug_masster_rewire_g0b.py --out output/rewire_g0b_2026_10_15
"""

from __future__ import annotations

import argparse
import copy
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import RDLogger
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import debug_masster_rewire_reach_ceiling as R  # noqa: E402

from fragnnet.inference.checkpoint import (  # noqa: E402
    apply_inference_config,
    get_config_from_ckpt,
    init_model_from_ckpt,
)
from fragnnet.models.masster.candidates.rewire_rules import (  # noqa: E402
    MASSTER_NUM_REWIRE_RULES,
    build_rewire_matches,
    filter_rewire_matches,
)
from fragnnet.runner import load_config  # noqa: E402
from fragnnet.utils.misc_utils import deep_update  # noqa: E402
from fragnnet.utils.setup_utils import get_dataset_cls  # noqa: E402

CHECKPOINT = (
    "/tmp/claude-1002/-home-feiw-gitRepos-frag-gnn/3f5debb8-28ae-4a28-9327-4237177ea097/scratchpad/"
    "i1_epoch059_orb.ckpt"
)
I1_CONFIG = "config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml"
MINING = Path("output/rearrangement_mining_2026_10_14")
FRACTIONS = (0.25, 0.5, 1.0)
UNCAPPED = (1.0e6, 10**7)  # fraction, ceiling
DIAG = (
    "rewire_match_count", "rewire_truncated_match_count", "rewire_pair_count", "rewire_contained_pair_count",
    "rewire_precap_extra_count", "rewire_merged_count", "rewire_dropped_no_valid_count",
    "rewire_dropped_cap_count", "rewire_extra_count", "rewire_cap_tie_count",
)


def build_pl(config: dict, device: th.device, *, rules: bool, null: bool):
    cfg = dict(config)
    cfg["masster_rewire_rules"] = rules
    cfg["masster_rewire_null"] = null
    cfg["masster_rewire_continuation_cuts"] = 0
    model = init_model_from_ckpt(CHECKPOINT, cfg, device)
    model.eval()
    return model


def set_cap(pl, fraction: float, ceiling: int) -> None:
    pl.model.rewire_max_extra_fraction = float(fraction)
    pl.model.rewire_max_extra = int(ceiling)


def forward(pl, batch_cpu, device, annotate: bool = True):
    batch = pl.transfer_batch_to_device(copy.deepcopy(batch_cpu), device, dataloader_idx=0)
    with th.inference_mode():
        return pl(**batch, return_explainability_tensors=annotate), batch


def read_out(out: dict, batch_size: int, with_sets: bool, atom_ptr: np.ndarray | None):
    """Per-view nodes, extras, diagnostics and (optionally) product atom sets."""
    views = out["ion_h_fragment_batch_idxs"].cpu().numpy()
    nodes = np.bincount(views, minlength=batch_size)
    info = {"nodes": nodes}
    if "rewire_is_extra" not in out:
        return info
    is_extra = out["rewire_is_extra"].cpu().numpy().astype(bool)
    info["extras"] = np.bincount(views[is_extra], minlength=batch_size)
    for name in DIAG:
        info[name] = out[name].cpu().numpy()
    if not with_sets:
        return info
    rows = np.unique(out["fragment_rewire_fragment_idxs"].cpu().numpy())
    atoms = out["fragment_atom_indices"].cpu().numpy()
    members = out["fragment_atom_fragment_idxs"].cpu().numpy()
    sel = np.isin(members, rows)
    atoms, members = atoms[sel], members[sel]
    order = np.argsort(members, kind="stable")
    ids, starts = np.unique(members[order], return_index=True)
    merged: dict[int, set] = defaultdict(set)
    extra: dict[int, set] = defaultdict(set)
    for fragment, group in zip(ids, np.split(atoms[order], starts[1:]), strict=True):
        view = int(views[fragment])
        target = extra if is_extra[fragment] else merged
        target[view].add(frozenset((group - atom_ptr[view]).tolist()))
    info["merged_sets"], info["extra_sets"] = merged, extra
    return info


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="matching molecules, for a smoke run")
    parser.add_argument("--sample", type=int, default=1024, help="random whole-split spectra for timing/growth")
    parser.add_argument("--timing-batch", type=int, default=32)
    parser.add_argument("--skip-reach", action="store_true")
    args = parser.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = th.device(args.device if th.cuda.is_available() or args.device == "cpu" else "cpu")

    saved = get_config_from_ckpt(CHECKPOINT)
    config = deep_update(load_config("config", I1_CONFIG, []), saved)
    config = apply_inference_config(
        config, disable_preproc=True, num_workers=0,
        train_batch_size=args.batch_size, eval_batch_size=args.batch_size,
    )
    if device.type == "cpu":
        config["masster_candidate_backend"] = "cpu"
    pls = {
        "off": build_pl(config, device, rules=False, null=False),
        "on": build_pl(config, device, rules=True, null=False),
        "null": build_pl(config, device, rules=True, null=True),
    }
    dataset = get_dataset_cls(config["model_type"])(split="val", **config)
    specs = dataset.spec_df.reset_index(drop=True)
    molecules = pd.read_pickle(config["mol_fp"]).set_index("mol_id")["mol"]
    collate = dataset.get_collate_fn()

    # 1. Spectra whose molecule has a production rule match for [M+H]+.
    matching_mols = set()
    for mol_id in specs.mol_id.unique():
        found = filter_rewire_matches(
            build_rewire_matches(molecules.loc[mol_id], max_matches_per_rule=config["masster_rewire_max_matches_per_rule"]),
            precursor_type="[M+H]+", num_known_rules=MASSTER_NUM_REWIRE_RULES,
        )
        if found.matches:
            matching_mols.add(mol_id)
    matching = sorted(matching_mols)
    if args.limit:
        matching = matching[: args.limit]
    keep = np.flatnonzero(specs.mol_id.isin(set(matching)).to_numpy())
    print(
        f"val {len(specs)} spectra, {specs.mol_id.nunique()} molecules; {len(matching_mols)} match; "
        f"scoring {len(keep)} spectra of {len(matching)} molecules", flush=True,
    )

    variants: list[tuple[str, str, float, int]] = []  # name, model, fraction, ceiling
    variants += [(f"on_f{f}", "on", f, 256) for f in FRACTIONS]
    variants += [("on_uncapped", "on", *UNCAPPED)]
    variants += [(f"null_f{f}", "null", f, 256) for f in FRACTIONS]

    # 2. Reach and nodes on the matching spectra.
    rearr = pd.read_pickle(MINING / "rearrangement.pkl")
    rearr = rearr[(rearr.dataset == "nist20") & (rearr.split == "val")]
    peaks_by_spec = {sid: g for sid, g in rearr.groupby("spec_id")}
    total_weight = float(rearr.weight.sum())
    spec_rows, peak_rows = [], []
    if not args.skip_reach:
        loader = DataLoader(Subset(dataset, keep), batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
        cursor, bond_cache = 0, {}
        for n_batch, batch in enumerate(loader):
            rows = specs.iloc[keep[cursor: cursor + int(batch["batch_size"])]]
            bsz = len(rows)
            atom_ptr = batch["mol_pyg"].ptr.cpu().numpy()
            out, _ = forward(pls["off"], batch, device)
            infos = {"off": read_out(out, bsz, False, atom_ptr)}
            for name, model_key, fraction, ceiling in variants:
                set_cap(pls[model_key], fraction, ceiling)
                out, _ = forward(pls[model_key], batch, device)
                infos[name] = read_out(out, bsz, True, atom_ptr)
            for view, row in enumerate(rows.itertuples()):
                mol = molecules.loc[row.mol_id]
                if int(atom_ptr[view + 1] - atom_ptr[view]) != mol.GetNumAtoms():
                    raise ValueError(f"graph/mol atom count mismatch for mol {row.mol_id}")
                for name, info in infos.items():
                    rec = {"spec_id": row.spec_id, "mol_id": row.mol_id, "variant": name, "nodes": int(info["nodes"][view])}
                    if name != "off":
                        rec["extras"] = int(info["extras"][view])
                        rec.update({k: float(info[k][view]) for k in DIAG})
                    spec_rows.append(rec)
                idx = {}
                for name, info in infos.items():
                    if name == "off":
                        continue
                    merged = info["merged_sets"].get(view, set())
                    extra = info["extra_sets"].get(view, set())
                    idx[name] = (R._composition_index(mol, merged), R._composition_index(mol, merged | extra))
                for peak in peaks_by_spec.get(row.spec_id, pd.DataFrame()).itertuples():
                    rec = {"spec_id": row.spec_id, "mz": peak.mz, "weight": peak.weight, "ion": peak.ion}
                    for name, (merged_idx, all_idx) in idx.items():
                        rec[f"{name}_all"] = R._hit(all_idx, peak.ion)
                        rec[f"{name}_merged"] = R._hit(merged_idx, peak.ion)
                    peak_rows.append(rec)
            cursor += bsz
            if n_batch % 50 == 0:
                print(f"{cursor}/{len(keep)} spectra, {time.time() - started:.0f}s", flush=True)
        pd.DataFrame(spec_rows).to_pickle(out_dir / "spectra.pkl")
        pd.DataFrame(peak_rows).to_pickle(out_dir / "peaks.pkl")

    # 3. Timing, memory and whole-split node growth on a random sample of the split.
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(specs), min(args.sample, len(specs)), replace=False))
    t_loader = DataLoader(Subset(dataset, sample), batch_size=args.timing_batch, shuffle=False, collate_fn=collate, num_workers=0)
    t_batches = list(t_loader)
    timing: dict[str, dict] = {}
    t_variants = [("off", "off", 0.0, 0)] + [v for v in variants if v[0] != "on_uncapped"]

    def timed_pass(pl, annotate: bool):
        """Forward every sample batch; returns (seconds, node counts)."""
        nodes, seconds = [], 0.0
        for batch in t_batches:
            moved = pl.transfer_batch_to_device(copy.deepcopy(batch), device, dataloader_idx=0)
            if device.type == "cuda":
                th.cuda.synchronize()
            t0 = time.perf_counter()
            with th.inference_mode():
                out = pl(**moved, return_explainability_tensors=annotate)
            if device.type == "cuda":
                th.cuda.synchronize()
            seconds += time.perf_counter() - t0
            nodes.append(np.bincount(out["ion_h_fragment_batch_idxs"].cpu().numpy(), minlength=int(batch["batch_size"])))
        return seconds, np.concatenate(nodes)

    for name, model_key, fraction, ceiling in t_variants:
        pl = pls[model_key]
        if name != "off":
            set_cap(pl, fraction, ceiling)
        cold, _ = timed_pass(pl, False)  # fills the match cache (and CUDA kernels)
        if device.type == "cuda":
            th.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(3):
            seconds, nodes = timed_pass(pl, False)
            times.append(seconds)
        timing[name] = {
            "forward_s_per_pass_min": min(times), "forward_s_per_pass_all": times, "first_pass_s": cold,
            "peak_gpu_gib": th.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
            "nodes_mean": float(nodes.mean()), "nodes_p50": float(np.percentile(nodes, 50)),
            "nodes_p95": float(np.percentile(nodes, 95)), "nodes_max": int(nodes.max()),
            "nodes_total": int(nodes.sum()), "n_spectra": int(len(nodes)),
            "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        }
        print(name, json.dumps(timing[name]), flush=True)

    # forward+backward ratio (train mode proxy: gradient of the summed dense channel log weights)
    grad_s = {}
    sub = t_batches[: max(1, min(8, len(t_batches)))]
    for name, model_key, fraction, ceiling in (("off", "off", 0, 0), ("on_f0.5", "on", 0.5, 256), ("on_f1.0", "on", 1.0, 256)):
        pl = pls[model_key]
        if name != "off":
            set_cap(pl, fraction, ceiling)
        pl.train()
        total = 0.0
        try:
            for rep in range(2):
                total = 0.0
                for batch in sub:
                    moved = pl.transfer_batch_to_device(copy.deepcopy(batch), device, dataloader_idx=0)
                    pl.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        th.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = pl(**moved)
                    weights = out["ion_h_dense_log_weights"]
                    loss = weights[th.isfinite(weights)].sum()
                    loss.backward()
                    if device.type == "cuda":
                        th.cuda.synchronize()
                    total += time.perf_counter() - t0
            grad_s[name] = total
        except Exception as exc:  # report, do not hide
            grad_s[name] = f"failed: {type(exc).__name__}: {exc}"
        pl.eval()
    timing["fwd_bwd_s_per_8_batches"] = grad_s

    # 4. Summary.
    summary: dict = {"val_spectra": int(len(specs)), "matching_spectra": int(len(keep)), "total_rearrangement_weight": total_weight}
    if spec_rows:
        sp = pd.DataFrame(spec_rows)
        pk = pd.DataFrame(peak_rows)
        off_nodes = sp[sp.variant == "off"].set_index("spec_id").nodes
        per_variant = {}
        n_total = len(specs)
        sample_off_mean = timing["off"]["nodes_mean"]
        for name in [v[0] for v in variants]:
            v = sp[sp.variant == name].set_index("spec_id")
            nodes = v.nodes
            growth_matching = float(nodes.sum() / off_nodes.sum() - 1)
            whole_growth = float((nodes.sum() - off_nodes.sum()) / (n_total * sample_off_mean))
            reach = {}
            if not pk.empty:
                reach["reach_pct_all_val"] = 100 * float(pk.weight[pk[f"{name}_all"]].sum()) / total_weight
                reach["reach_pct_merged_only"] = 100 * float(pk.weight[pk[f"{name}_merged"]].sum()) / total_weight
                reach["reach_pct_from_extras"] = 100 * float(pk.weight[pk[f"{name}_all"] & ~pk[f"{name}_merged"]].sum()) / total_weight
            extras_precap = v.rewire_precap_extra_count
            per_variant[name] = {
                "nodes_mean": float(nodes.mean()), "nodes_p50": float(nodes.quantile(0.5)),
                "nodes_p95": float(nodes.quantile(0.95)), "nodes_max": int(nodes.max()),
                "off_nodes_mean": float(off_nodes.mean()), "off_nodes_p95": float(off_nodes.quantile(0.95)),
                "growth_matching_pct": 100 * growth_matching,
                "growth_p95_pct": 100 * float(nodes.quantile(0.95) / off_nodes.quantile(0.95) - 1),
                "growth_whole_split_pct_est": 100 * whole_growth,
                "extras_precap_mean": float(extras_precap.mean()), "extras_precap_p95": float(extras_precap.quantile(0.95)),
                "extras_precap_max": float(extras_precap.max()),
                "extras_kept_mean": float(v.extras.mean()), "extras_kept_p95": float(v.extras.quantile(0.95)),
                "extras_kept_max": int(v.extras.max()),
                "dropped_no_valid_mean": float(v.rewire_dropped_no_valid_count.mean()),
                "dropped_cap_mean": float(v.rewire_dropped_cap_count.mean()),
                "cap_binds_pct_of_matching_spectra": 100 * float((v.rewire_dropped_cap_count > 0).mean()),
                "merge_rate": float(v.rewire_merged_count.sum() / max(v.rewire_merged_count.sum() + extras_precap.sum(), 1)),
                "merged_mean": float(v.rewire_merged_count.mean()),
                **reach,
            }
        summary["variants"] = per_variant
    summary["timing"] = timing
    summary["elapsed_s"] = time.time() - started
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
