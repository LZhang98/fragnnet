"""Training-only class-rarity census and frozen held-out cohorts for the A1/A3 sampling arms.

Resolves an experiment config, rebuilds its filtered train/val/test spectrum tables with the
dataset's own ``_setup_dfs``, and reports:

1. Molecule, spectrum and acquisition-key counts; identity overlap between splits.
2. Per-class training molecule counts, rarity bins, A3 weights and held-out support.
3. A frozen per-molecule held-out cohort table keyed by each molecule's rarest class.
4. How far A1 and A3 move molecule exposure away from A0 (per-spectrum shuffling).
5. One epoch drawn from the real A1/A3 sampler as a smoke check.

The model dataset can drop a few more spectra (missing DAGs), so counts here may slightly
exceed what the training run sees; the sampler logs its own counts at training start.

Usage:
    python scripts/debug/rare_chemistry/rare_chemistry_census.py \\
        --config config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml \\
        --output_dp output/rare_chemistry_census/nist20_i1
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch as th

from fragnnet.dataset.base_dataset import BaseDataset
from fragnnet.dataset.rarity_sampling import (
    RAW_CE_COLS,
    acquisition_keys,
    build_molecule_acquisition_sampler,
    class_rarity_weights,
    match_class_panel,
    mixture_probabilities,
    molecule_keys,
)
from fragnnet.runner import load_config

logger = logging.getLogger(__name__)

# Frozen before reading any validation outcome (plan section 1).
RARITY_BINS = [(0, 0, "unseen"), (1, 5, "very_rare"), (6, 20, "rare"), (21, 100, "intermediate")]
MIN_HELD_OUT_SUPPORT = 20


class TrainTable:
    """Minimal stand-in for the training dataset that the sampler builder reads."""

    def __init__(self, spec_df: pd.DataFrame, mol_df: pd.DataFrame, spec_params: dict) -> None:
        self.spec_df = spec_df
        self.mol_df = mol_df
        self.spec_params = spec_params

    def __len__(self) -> int:
        return len(self.spec_df)


def rarity_bin(count: int) -> str:
    for low, high, name in RARITY_BINS:
        if low <= count <= high:
            return name
    return "common"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--output_dp", required=True, help="Directory for census tables")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.makedirs(args.output_dp, exist_ok=True)

    # 1. Rebuild filtered split tables exactly as the datasets do.
    config_d = load_config("config", args.config, None)
    panel = config_d["molecule_acquisition_sampler_class_panel"]
    raw_spec_df = pd.read_pickle(config_d["spec_fp"])
    raw_ce_df = raw_spec_df.set_index("spec_id")[list(RAW_CE_COLS)]
    tables = {}
    for split in ("train", "val", "test"):
        spec_df, mol_df, *_ = BaseDataset._setup_dfs(
            spec_fp_or_df=raw_spec_df.copy(),
            mol_fp_or_df=config_d["mol_fp"],
            split_dp=config_d["split_dp"],
            splits=[split],
            subsample_params=config_d["subsample_params"],
            spec_params=config_d["spec_params"],
        )
        spec_df = spec_df.assign(
            mol_key=molecule_keys(spec_df, mol_df),
            acq_key=acquisition_keys(spec_df, raw_ce_df),
            smiles=mol_df.loc[spec_df["mol_id"].to_numpy(), "smiles"].to_numpy(),
        )
        tables[split] = (spec_df, mol_df)
    summary = {"config": args.config, "class_panel": panel, "splits": {}}
    for split, (spec_df, _) in tables.items():
        summary["splits"][split] = {
            "spectra": len(spec_df),
            "molecules": int(spec_df["mol_key"].nunique()),
            "molecule_acquisition_keys": int((spec_df["mol_key"] + spec_df["acq_key"]).nunique()),
        }
    train_keys = set(tables["train"][0]["mol_key"])
    for split in ("val", "test"):
        overlap = train_keys & set(tables[split][0]["mol_key"])
        summary["splits"][split]["molecules_shared_with_train"] = len(overlap)
        if overlap:
            logger.warning("%d %s molecules share a 2D InChIKey with train", len(overlap), split)

    # 2. Class panel over every molecule; counts from training molecules only.
    smiles_by_split = {
        split: spec_df.groupby("mol_key", sort=True)["smiles"].first()
        for split, (spec_df, _) in tables.items()
    }
    classes = {split: match_class_panel(smiles, panel) for split, smiles in smiles_by_split.items()}
    train_counts = classes["train"].sum(axis=0)
    class_weight = np.clip(
        np.sqrt(config_d["molecule_acquisition_sampler_ref_count"] / train_counts.clip(lower=1)),
        1.0,
        config_d["molecule_acquisition_sampler_weight_cap"],
    )
    class_table = pd.DataFrame(
        {
            "train_molecules": train_counts,
            "val_molecules": classes["val"].sum(axis=0),
            "test_molecules": classes["test"].sum(axis=0),
            "bin": [rarity_bin(int(n)) for n in train_counts],
            "a3_weight": class_weight.where(train_counts > 0, np.nan),
        }
    ).sort_values("train_molecules")
    class_table.to_csv(os.path.join(args.output_dp, "class_census.csv"), index_label="class")

    # 3. Frozen held-out cohorts: each molecule's rarest matched class by training count.
    cohort_rows = []
    for split in ("val", "test"):
        matrix = classes[split]
        counts = np.where(matrix.to_numpy(), train_counts.to_numpy()[None, :], np.inf)
        rarest = counts.argmin(axis=1)
        rarest_count = counts.min(axis=1)
        spec_counts = tables[split][0].groupby("mol_key").size()
        for row, mol_key in enumerate(matrix.index):
            matched = np.isfinite(rarest_count[row])
            cohort_rows.append(
                {
                    "split": split,
                    "mol_key": mol_key,
                    "spectra": int(spec_counts[mol_key]),
                    "rarest_class": matrix.columns[rarest[row]] if matched else "",
                    "rarest_class_train_molecules": int(rarest_count[row]) if matched else -1,
                    "cohort": rarity_bin(int(rarest_count[row])) if matched else "unmatched",
                    "classes": ";".join(matrix.columns[matrix.iloc[row].to_numpy()]),
                }
            )
    cohort_df = pd.DataFrame(cohort_rows)
    cohort_df.to_csv(os.path.join(args.output_dp, "held_out_cohorts.csv"), index=False)
    support = cohort_df.groupby(["split", "cohort"]).size().unstack(0).fillna(0).astype(int)
    summary["held_out_cohort_molecules"] = support.to_dict()
    summary["cohorts_below_min_support"] = {
        split: sorted(support.index[support[split] < MIN_HELD_OUT_SUPPORT])
        for split in support.columns
    }

    # 4. Molecule exposure under A0, A1 and A3 on the training table.
    train_spec_df = tables["train"][0]
    spectra_per_mol = train_spec_df.groupby("mol_key", sort=True).size()
    a0 = (spectra_per_mol / spectra_per_mol.sum()).to_numpy()
    a1 = np.full(len(a0), 1.0 / len(a0))
    weights = class_rarity_weights(
        classes["train"],
        ref_count=config_d["molecule_acquisition_sampler_ref_count"],
        cap=config_d["molecule_acquisition_sampler_weight_cap"],
    ).loc[spectra_per_mol.index]
    a3 = mixture_probabilities(
        weights.to_numpy(), config_d["molecule_acquisition_sampler_uniform_mix"]
    )
    summary["exposure"] = {
        "spectra_per_molecule_quantiles": spectra_per_mol.quantile([0, 0.1, 0.5, 0.9, 1]).to_dict(),
        "a1_vs_a0_total_variation": float(0.5 * np.abs(a1 - a0).sum()),
        "a3_vs_a1_total_variation": float(0.5 * np.abs(a3 - a1).sum()),
        "a3_molecules_weight_above_1": int((weights > 1).sum()),
        "a3_weight_counts": weights.round(3).value_counts().sort_index().to_dict(),
        "a3_effective_molecules": float(1.0 / np.square(a3).sum()),
        "molecules_matching_no_class": int((~classes["train"].any(axis=1)).sum()),
    }

    # 5. Smoke-check the real sampler on the training table for both arms.
    train_ds = TrainTable(train_spec_df, tables["train"][1], config_d["spec_params"])
    summary["sampler_epoch"] = {}
    for weighting in ("uniform", "class"):
        arm_config = {**config_d, "molecule_acquisition_sampler_weighting": weighting}
        sampler = build_molecule_acquisition_sampler(
            train_ds, arm_config, th.Generator().manual_seed(config_d["seed"])
        )
        draws = np.fromiter(iter(sampler), dtype=np.int64)
        drawn_mols = train_spec_df["mol_key"].to_numpy()[draws]
        summary["sampler_epoch"][weighting] = {
            "draws": len(draws),
            "distinct_molecules": int(pd.unique(drawn_mols).size),
            "distinct_spectra": int(np.unique(draws).size),
        }

    with open(os.path.join(args.output_dp, "summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=2, default=str)
    logger.info("census written to %s", args.output_dp)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
