"""Select fixed molecular triples across measured NCE bands and summarize their coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def select_banded_pairs(spectra: pd.DataFrame, anchor: pd.DataFrame, bands: list) -> tuple:
    """Keep query, neighbor, control and acquisition fixed; choose a common NCE per band.

    Each band is [lower, upper). Among NCEs present in all three molecules, choose closest
    to the band center, breaking ties by lower NCE. Replicates resolve by lowest spectrum ID.
    Neither peak intensities nor model predictions enter selection. Missing bands are retained
    as missing availability rows, not interpreted as unsuccessful chemical correspondences.
    """
    by_id = spectra.set_index("spec_id")
    output, availability = [], []
    for query_spec, rows in anchor.groupby("query_spec", sort=True):
        if len(rows) != 2 or set(rows.kind) != {"neighbor", "control"}:
            raise ValueError("Expected exactly one neighbor/control per query")
        q = by_id.loc[query_spec]
        refs = rows.set_index("kind").donor_spec.to_dict()
        ids = [query_spec, int(refs["neighbor"]), int(refs["control"])]
        originals = by_id.loc[ids]
        keys = ["inst_type", "prec_type", "frag_mode"]
        if originals.mol_id.nunique() != 3 or any(originals[k].nunique() != 1 for k in keys):
            raise ValueError("Triple identity/acquisition mismatch")
        base = spectra[spectra.mol_id.isin(originals.mol_id)]
        for key in keys:
            base = base[base[key] == q[key]]
        base = base[base.ce_type.eq("single") & base.nce.notna() & base.nce.ge(0)]
        frames = [
            base[base.mol_id == mid].sort_values("spec_id").drop_duplicates("nce").set_index("nce")
            for mid in originals.mol_id
        ]
        common = set(frames[0].index) & set(frames[1].index) & set(frames[2].index)
        for lo, hi in bands:
            label = f"{lo:g}–{hi:g}"
            supported = sorted(n for n in common if lo <= n < hi)
            record = {
                "query_mol": int(q.mol_id),
                "anchor_query_spec": int(query_spec),
                "novel": bool(rows.novel.iloc[0]),
                "band": label,
                "band_lower": lo,
                "band_upper": hi,
                "inst_type": q.inst_type,
                "available": bool(supported),
                "nce": np.nan,
            }
            if supported:
                nce = min(supported, key=lambda n: (abs(n - (lo + hi) / 2), n))
                record["nce"] = float(nce)
                sids = [int(f.loc[nce].spec_id) for f in frames]
                for kind, donor in zip(("neighbor", "control"), sids[1:]):
                    output.append(
                        {
                            **record,
                            "query_spec": sids[0],
                            "donor_spec": donor,
                            "kind": kind,
                            "query_max_tanimoto": float(rows.query_max_tanimoto.iloc[0]),
                        }
                    )
            availability.append(record)
    return pd.DataFrame(output), pd.DataFrame(availability)


def summarize_bands(
    stats: pd.DataFrame, pairs: pd.DataFrame, availability: pd.DataFrame
) -> pd.DataFrame:
    """Produce molecule-balanced summaries for available and common-across-bands cohorts."""
    metadata = pairs[["query_spec", "donor_spec", "kind", "band", "band_lower", "inst_type", "nce"]]
    data = stats.merge(metadata, on=["query_spec", "donor_spec", "kind"], validate="many_to_one")
    complete = availability.groupby("query_mol").available.all()
    complete = set(complete[complete].index)
    tables = []
    for cohort in ("available", "all_bands_common"):
        rows = data if cohort == "available" else data[data.query_mol.isin(complete)]
        for instrument in ["all", *sorted(data.inst_type.unique())]:
            selected = rows if instrument == "all" else rows[rows.inst_type == instrument]
            if selected.empty:
                continue
            summary = (
                selected.groupby(["novel", "kind", "view", "band_lower", "band"])
                .agg(
                    molecules=("query_mol", "nunique"),
                    two_reliable=("two_reliable", "sum"),
                    fraction=("two_reliable", "mean"),
                    mean_matches=("matches", "mean"),
                    query_intensity=("query_covered_intensity", "mean"),
                    median_nce=("nce", "median"),
                )
                .reset_index()
            )
            summary["cohort"] = cohort
            summary["instrument"] = instrument
            tables.append(summary)
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    """Prepare a label-blind band manifest or summarize a completed annotated audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "summarize"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        from scripts.debug.masster_memorization.debug_masster_candidate_correspondence import sha256

        anchor = pd.read_csv(cfg["anchor_pairs"])
        provenance = json.loads((Path(cfg["census_dir"]) / "provenance.json").read_text())
        for record in provenance["inputs"].values():
            if sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"Input changed: {record['path']}")
        ids = pd.read_csv(provenance["inputs"]["train_ids_fp"]["path"]).spec_id
        specs = pd.read_pickle(provenance["inputs"]["spec_fp"]["path"])
        specs = specs[specs.spec_id.isin(ids)].drop(columns="peaks")
        pairs, availability = select_banded_pairs(specs, anchor, cfg["nce_bands"])
        pairs.to_csv(args.output / "band_pairs.csv", index=False)
        availability.to_csv(args.output / "availability.csv", index=False)
        (args.output / "selection_provenance.json").write_text(
            json.dumps(
                {
                    "config": cfg,
                    "anchor_sha256": sha256(cfg["anchor_pairs"]),
                    "source_sha256": sha256(__file__),
                    "inputs": provenance["inputs"],
                    "pair_sha256": sha256(args.output / "band_pairs.csv"),
                },
                indent=2,
            )
        )
        print(
            availability.groupby(["novel", "band_lower"]).available.agg(["sum", "size"]).to_string()
        )
    else:
        stats = pd.read_csv(args.output / "pair_coverage.csv")
        pairs = pd.read_csv(args.output / "band_pairs.csv")
        availability = pd.read_csv(args.output / "availability.csv")
        table = summarize_bands(stats, pairs, availability)
        table.to_csv(args.output / "nce_summary.csv", index=False)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, view, title in zip(
            axes, ["ion", "loss"], ["Product-ion formula", "Neutral-loss formula"]
        ):
            data = table[
                (table.cohort == "all_bands_common")
                & table.novel
                & (table.instrument == "all")
                & (table.view == view)
            ]
            for kind, color in [("neighbor", "#1765ab"), ("control", "#d47b27")]:
                rows = data[data.kind == kind].sort_values("band_lower")
                ax.plot(rows.band, rows.fraction * 100, marker="o", color=color, label=kind)
            ax.set_title(title)
            ax.set_xlabel("Measured NCE band [lower, upper)")
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Queries with two strong matches (%)")
        axes[1].legend()
        common = table[
            (table.cohort == "all_bands_common") & table.novel & (table.instrument == "all")
        ]
        common_n = int(common.molecules.iloc[0])
        fig.suptitle(
            f"Unfamiliar queries (n={common_n}): fixed molecules and references across NCE bands"
        )
        fig.tight_layout()
        fig.savefig(args.output / "nce_coverage.png", dpi=180)
        fig.savefig(args.output / "nce_coverage.svg")
        print(
            table[
                (table.cohort == "all_bands_common") & table.novel & (table.instrument == "all")
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
