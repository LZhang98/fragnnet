"""Cross-fit bounded reference-evidence corrections to frozen validation predictions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
import yaml

from scripts.debug.debug_masster_candidate_correspondence import match_groups, sha256

LOGGER = logging.getLogger(__name__)
ARMS = ("structure", "aligned", "shuffled", "random")
ELEMENTS = ("H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I")


def bin_indices(mzs: np.ndarray, maximum: float, resolution: float = 0.01) -> np.ndarray:
    """Use production bin boundaries, preserving each mass array's native dtype."""
    tensor = th.from_numpy(np.asarray(mzs))
    bins = th.arange(resolution, maximum + resolution, resolution, dtype=tensor.dtype)
    return th.searchsorted(bins, tensor, right=True).numpy()


def metric_cache(frame: pd.DataFrame, maximum: float) -> dict:
    """Cache sparse production bins, retaining the full target norm and uncovered peaks."""
    masses = frame.native_mz.to_numpy()
    keep = masses < maximum
    unique, inverse = np.unique(bin_indices(masses[keep], maximum), return_inverse=True)
    tm, ti = frame.attrs["true_mzs"], frame.attrs["true_ints"]
    if (tm >= maximum).any():
        raise ValueError("Target was not eval-preprocessed")
    tb = bin_indices(tm, maximum)
    target = np.bincount(tb, weights=ti, minlength=int(round(maximum / 0.01)))
    return {
        "keep": keep,
        "bins": unique,
        "inverse": inverse,
        "true_at_pred": target[unique],
        "norm": np.linalg.norm(target),
        "top": int(target.argmax()),
        "target": frame.target.to_numpy(),
    }


def scores(logits: np.ndarray, metric: dict) -> tuple[float, float, float]:
    """Return full-spectrum cosine, binned base-peak accuracy and strong-outcome inversion."""
    probs = np.exp(logits - logits.max())
    binned = np.bincount(
        metric["inverse"], weights=probs[metric["keep"]], minlength=len(metric["bins"])
    )
    denom = np.linalg.norm(binned) * metric["norm"]
    cosine = float(binned @ metric["true_at_pred"] / denom) if denom else 0.0
    top = float(metric["bins"][binned.argmax()] == metric["top"]) if len(binned) else 0.0
    strong = np.flatnonzero(metric["target"] >= 0.01)
    a, b = np.triu_indices(len(strong), 1)
    truth = metric["target"][strong[a]] - metric["target"][strong[b]]
    predicted = logits[strong[a]] - logits[strong[b]]
    non_ties = truth != 0
    error = np.where(truth * predicted < 0, 1.0, np.where(predicted == 0, 0.5, 0.0))
    rank = float(error[non_ties].mean()) if non_ties.any() else float("nan")
    return cosine, top, rank


def composition(text: str) -> list[float]:
    """Fixed elemental features; formula equality is not a mechanism assignment."""
    counts = {element: int(n) for element, n in re.findall(r"([A-Z][a-z]?)(\d+)", text)}
    return [float(counts.get(e, 0)) for e in ELEMENTS] + [float(sum(counts.values()))]


def query_features(frame: pd.DataFrame, row: pd.Series, floor: float) -> np.ndarray:
    """Build target-independent query chemistry and acquisition features (C,F)."""
    return np.column_stack(
        [
            np.log(frame.pred.to_numpy().clip(floor)),
            frame.mz / row.query_prec_mz,
            frame.mz / 100,
            np.full(len(frame), row.query_prec_mz / 100),
            np.full(len(frame), row.nce / 100),
            np.full(len(frame), row.inst_type == "OB_F"),
            np.log1p(frame.structural_alternatives),
            frame.formula_alternatives,
            frame.eligible,
            [composition(f) for f in frame.ion],
            [composition(f) for f in frame.loss],
        ]
    ).astype(np.float32)


def reference_features(
    left: pd.DataFrame, right: pd.DataFrame, row: pd.Series, view: str, floor: float, seed: int
) -> tuple:
    """Return structure, real/shuffled labels and a mask; mapping never reads labels."""
    edges = match_groups(left, right, view)
    structure = np.zeros((len(left), 6), dtype=np.float32)
    labels = np.zeros((len(left), 2), dtype=np.float32)
    shuffled = labels.copy()
    mask = np.zeros(len(left), dtype=np.float32)
    if edges:
        a, b = np.array(edges).T
        mask[a] = 1
        structure[a] = np.column_stack(
            [
                np.full(len(a), row.tanimoto),
                right.mz.to_numpy()[b] / 100,
                np.full(len(a), row.donor_prec_mz / 100),
                np.full(len(a), (row.donor_prec_mz - row.query_prec_mz) / 100),
                np.log1p(right.structural_alternatives.to_numpy()[b]),
                right.formula_alternatives.to_numpy()[b],
            ]
        )
        values = right.target.to_numpy()[b]
        labels[a] = np.column_stack([values, np.log(values + floor)])
        # A cyclic shift of a random ordering has no fixed endpoints for n > 1.
        order = np.random.default_rng(seed).permutation(len(a))
        perm = np.arange(len(a))
        perm[order] = np.roll(order, 1)
        shuffled[a] = labels[a[perm]]
    return structure, labels, shuffled, mask


class Correction(th.nn.Module):
    """Zero-initialized residual on frozen candidate-group logits."""

    def __init__(self, size: int, width: int) -> None:
        super().__init__()
        self.net = th.nn.Sequential(th.nn.Linear(size, width), th.nn.SiLU(), th.nn.Linear(width, 1))
        th.nn.init.zeros_(self.net[-1].weight)
        th.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Map (C,F) features to (C,) logit corrections."""
        return self.net(x).squeeze(-1)


def fit_correction(
    data: list[dict],
    fit: list[int],
    stop: list[int],
    features: str,
    baselines: list[np.ndarray],
    cfg: dict,
    seed: int,
) -> tuple:
    """Fit normalization and residual on fit identities; choose epoch on inner identities."""
    th.manual_seed(seed)
    rng = np.random.default_rng(seed)
    matrix = np.concatenate([data[i][features] for i in fit])
    mean, std = matrix.mean(0), matrix.std(0).clip(1e-4)
    xs = [th.from_numpy(((d[features] - mean) / std).astype(np.float32)) for d in data]
    masks = [th.from_numpy(d[features + "_mask"]) for d in data]
    base = [th.tensor(b, dtype=th.float32) for b in baselines]
    targets = [th.tensor(d["target"], dtype=th.float32) for d in data]
    model = Correction(matrix.shape[1], cfg["hidden_dim"])
    optimizer = th.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    best_score = np.mean([scores(baselines[i], data[i]["metric"])[0] for i in stop])
    best, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    history = [{"epoch": 0, "inner_cosine": float(best_score), "fit_ce": None}]
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = rng.permutation(fit)
        fit_losses = []
        for start in range(0, len(order), cfg["batch_spectra"]):
            losses = []
            for i in order[start : start + cfg["batch_spectra"]]:
                if targets[i].sum() == 0:
                    continue
                logp = th.log_softmax(base[i] + masks[i] * model(xs[i]), dim=0)
                losses.append(-(targets[i] / targets[i].sum() * logp).sum())
            if losses:
                optimizer.zero_grad()
                fit_losses.extend(float(loss.detach()) for loss in losses)
                th.stack(losses).mean().backward()
                optimizer.step()
        model.eval()
        with th.no_grad():
            value = np.mean(
                [
                    scores((base[i] + masks[i] * model(xs[i])).numpy(), data[i]["metric"])[0]
                    for i in stop
                ]
            )
        history.append(
            {"epoch": epoch, "inner_cosine": float(value), "fit_ce": float(np.mean(fit_losses))}
        )
        if value > best_score + 1e-8:
            best_score, best_epoch, stale = value, epoch, 0
            best = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(best)
    model.eval()
    with th.no_grad():
        predictions = [(base[i] + masks[i] * model(xs[i])).numpy() for i in range(len(data))]
    state = {
        "history": history,
        "weights": best,
        "mean": mean,
        "std": std,
        "fit": fit,
        "stop": stop,
        "epoch": best_epoch,
        "inner_cosine": best_score,
        "seed": seed,
    }
    return predictions, state


def paired_summary(results: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Bootstrap paired query-identity deltas, conditional on the fixed reference library."""
    records = []
    for view, group in results.groupby("view"):
        for name, subset in [
            ("all", group),
            ("novel", group[group.novel]),
            ("other", group[~group.novel]),
            ("nce40_60", group[group.nce < 60]),
            ("nce60_100", group[group.nce >= 60]),
        ]:
            if subset.empty:
                continue
            for arm in ("raw", "context") + ARMS:
                records.append(
                    {
                        "view": view,
                        "slice": name,
                        "comparison": arm,
                        "n": len(subset),
                        "mean": subset[arm + "_cosine"].mean(),
                        "base_peak": subset[arm + "_top"].mean(),
                        "ranking_error": subset[arm + "_rank"].mean(),
                    }
                )
            for control in ("raw", "context", "structure", "shuffled", "random"):
                delta = (subset.aligned_cosine - subset[control + "_cosine"]).to_numpy()
                rng = np.random.default_rng(cfg["seed"])
                means = delta[
                    rng.integers(len(delta), size=(cfg["bootstrap_repeats"], len(delta)))
                ].mean(1)
                lo, hi = np.quantile(means, [0.025, 0.975])
                record = {
                    "view": view,
                    "slice": name,
                    "comparison": "aligned-" + control,
                    "n": len(delta),
                    "mean": delta.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                }
                for fold in (0, 1):
                    part = subset[subset.fold == fold]
                    record[f"fold{fold}"] = (part.aligned_cosine - part[control + "_cosine"]).mean()
                records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    """Run predeclared ion and loss views with identical outer/inner molecular splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])  # noqa: NPY002
    th.manual_seed(cfg["seed"])
    th.set_num_threads(cfg["threads"])
    started = time.time()
    out = args.input / "correction"
    out.mkdir(exist_ok=True)
    selection = json.loads((args.input / "selection_provenance.json").read_text())
    extraction = json.loads((args.input / "extraction/provenance.json").read_text())
    if extraction["query_split"] != "val" or extraction["status"] != "complete":
        raise ValueError("Requires completed validation-query extraction")
    if selection["pairs_sha256"] != sha256(args.input / "gate_pairs.csv"):
        raise ValueError("Changed selection")
    if extraction["external_pairs_sha256"] != selection["pairs_sha256"]:
        raise ValueError("Extraction used different pairs")
    if extraction["resolved_model_config"]["ints_transform"] != "none":
        raise ValueError("This gate requires untransformed model intensities")
    run = {
        "status": "running",
        "config": cfg,
        "command": sys.argv,
        "device": "cpu",
        "torch": th.__version__,
        "numpy": np.__version__,
        "started_unix": started,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
        "source_sha256": sha256(__file__),
        "config_sha256": sha256(args.config),
        "cache_sha256": sha256(args.input / "extraction/candidate_groups.pkl"),
        "selection_sha256": sha256(args.input / "selection_provenance.json"),
        "extraction_sha256": sha256(args.input / "extraction/provenance.json"),
    }
    (out / "provenance.json").write_text(json.dumps(run, indent=2))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    # 1. Build target-independent features and metric caches on the fixed cohort.
    cache = pd.read_pickle(args.input / "extraction/candidate_groups.pkl")
    pairs = pd.read_csv(args.input / "gate_pairs.csv")
    queries = pairs[pairs.kind == "neighbor"].sort_values("query_spec").reset_index(drop=True)
    controls = pairs[pairs.kind == "control"].set_index("query_spec")
    if queries.identity.duplicated().any() or queries.query_spec.duplicated().any():
        raise ValueError("Outer folds must contain unique query identities")
    maximum = extraction["resolved_model_config"]["mz_max"]
    data, raw = [], []
    for _, row in queries.iterrows():
        frame = cache[row.query_spec]
        data.append(
            {
                "context": query_features(frame, row, cfg["log_floor"]),
                "context_mask": np.ones(len(frame), dtype=np.float32),
                "target": frame.target.to_numpy(),
                "metric": metric_cache(frame, maximum),
            }
        )
        raw.append(np.log(frame.pred.to_numpy().clip(np.finfo(np.float64).tiny)))
    # 2. Freeze common context corrections separately within each outer fold.
    folds, context, states = {}, {}, {}
    for fold in (0, 1):
        train = queries.index[queries.fold != fold].to_numpy()
        rng = np.random.default_rng(cfg["seed"] + fold)
        train = rng.permutation(train)
        stop_count = max(1, int(round(len(train) * cfg["inner_fraction"])))
        fit, stop = train[stop_count:].tolist(), train[:stop_count].tolist()
        held = queries.index[queries.fold == fold].tolist()
        folds[fold] = (fit, stop, held)
        context[fold], states[f"context_{fold}"] = fit_correction(
            data, fit, stop, "context", raw, cfg, cfg["seed"] + fold
        )
        LOGGER.info("Context fold=%d epoch=%d", fold, states[f"context_{fold}"]["epoch"])
    # 3. Fit controlled reference residuals; outer predictions are never used to select epochs.
    records = []
    for view in ("ion", "loss"):
        for i, row in queries.iterrows():
            left = cache[row.query_spec]
            neighbor = reference_features(
                left,
                cache[row.donor_spec],
                row,
                view,
                cfg["log_floor"],
                cfg["seed"] + int(row.query_spec),
            )
            control = controls.loc[row.query_spec]
            random_ref = reference_features(
                left,
                cache[control.donor_spec],
                control,
                view,
                cfg["log_floor"],
                cfg["seed"] + int(row.query_spec),
            )
            st, labels, shuffled, mask = neighbor
            for arm, structure, intensity, present in [
                ("structure", st, np.zeros_like(labels), mask),
                ("aligned", st, labels, mask),
                ("shuffled", st, shuffled, mask),
                ("random", random_ref[0], random_ref[1], random_ref[3]),
            ]:
                data[i][arm] = np.column_stack([data[i]["context"], structure, intensity])
                data[i][arm + "_mask"] = present
            data[i]["shuffle_changed"] = int(np.any(labels != shuffled, axis=1).sum())
        for fold, (fit, stop, held) in folds.items():
            predictions = {"raw": raw, "context": context[fold]}
            for arm in ARMS:
                key = f"{view}_{arm}_{fold}"
                predictions[arm], states[key] = fit_correction(
                    data, fit, stop, arm, context[fold], cfg, cfg["seed"] + 10 + fold
                )
                LOGGER.info("%s epoch=%d", key, states[key]["epoch"])
            for i in held:
                record = queries.loc[i].to_dict()
                record.update(
                    view=view,
                    neighbor_matches=int(data[i]["aligned_mask"].sum()),
                    random_matches=int(data[i]["random_mask"].sum()),
                    shuffle_changed=data[i]["shuffle_changed"],
                )
                for arm, pred in predictions.items():
                    for name, value in zip(
                        ("cosine", "top", "rank"), scores(pred[i], data[i]["metric"])
                    ):
                        record[arm + "_" + name] = value
                records.append(record)
    # 4. Save paired outer results, uncertainty, reproducible correction states and donor reuse.
    results = pd.DataFrame(records).sort_values(["view", "query_spec"])
    results.to_csv(out / "per_query.csv", index=False)
    summary = paired_summary(results, cfg)
    summary.to_csv(out / "summary.csv", index=False)
    th.save(states, out / "corrections.pt")
    pd.DataFrame(
        [{**row, "arm": key} for key, state in states.items() for row in state["history"]]
    ).to_csv(out / "fit_history.csv", index=False)
    pd.DataFrame(
        [
            {"arm": key, "epoch": state["epoch"], "inner_cosine": state["inner_cosine"]}
            for key, state in states.items()
        ]
    ).to_csv(out / "epochs.csv", index=False)
    reuse = pairs.groupby(["kind", "donor_mol"]).size().rename("queries").reset_index()
    reuse.to_csv(out / "donor_reuse.csv", index=False)
    run.update(
        status="complete",
        elapsed_seconds=time.time() - started,
        query_identities=len(queries),
        per_query_sha256=sha256(out / "per_query.csv"),
        summary_sha256=sha256(out / "summary.csv"),
    )
    (out / "provenance.json").write_text(json.dumps(run, indent=2))
    LOGGER.info("Completed %d queries, two formula views", len(queries))


if __name__ == "__main__":
    main()
