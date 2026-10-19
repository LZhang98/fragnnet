"""Gate for mechanism SMARTS rules applied with continuation, on mined rearrangement peaks.

A rule is a SMARTS with atom maps plus a graph edit: atoms to delete (they leave in the
neutral), bonds to cut and bonds to form. Applying the edit to the precursor graph and
searching connected atom sets of the *rewired* graph reaches every product the rule can make
followed by any ordinary cuts, which is the same set as applying the rule to every connected
fragment that contains the matched motif (after-selection application with an unlimited pool).

Input peaks come from ``debug_masster_rearrangement_mining.py``: Orbitrap [M+H]+ peaks
(>= 5% of base) whose compositions no connected atom set of the precursor reaches within
+-4 H. For each rule, per peak, the gate records whether some match's rewired graph reaches
the peak's lowest-error composition within +-4 H (``real``), whether the same edit applied to
random matches of a topology-only version of the SMARTS reaches it (``null``), and whether a
registered MASSter template side already reaches it as a whole (``registered``).

Rule file (JSON list):
    [{"name": "sulfonyl_so2_extrusion",
      "smarts": "[#6:1][SX4:2](=[OX1:3])(=[OX1:4])[!#1:5]",
      "delete": [2, 3, 4], "cut": [], "form": [[1, 5]]}, ...]
``delete`` drops every bond of those atoms; ``cut`` drops listed bonds; ``form`` adds bonds.
Optional ``null_smarts`` overrides the automatic topology-only null pattern.

Run from the repository root:
    python scripts/debug/smarts/debug_masster_rewire_rule_gate.py --rules rules.json --out DIR
"""

import argparse
import json
import re
import sys
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from debug_masster_smarts_rule_gate import reachable_h  # noqa: E402

from fragnnet.models.masster.candidates.smarts_templates import (  # noqa: E402
    build_masster_templates,
    filter_masster_templates,
)

MINING = Path("output/rearrangement_mining_2026_10_14")
PRECURSOR = "[M+H]+"
REACH_CAP = 20_000
NULL_DRAWS = 3
_RULES: list[dict] = []


def parse_formula(formula: str) -> dict[str, int]:
    return {el: int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}


def _is_terminal_query(atom) -> bool:
    return bool(re.search(r"(X1|D1)(?!\d)", atom.GetSmarts()))


def topology_null_smarts(rule: dict) -> str:
    """The edit's shape with any elements and bond orders.

    Keeps the atoms the edit touches (formed and cut bonds, deleted atoms), drops deleted
    terminal atoms (deleting their neighbour isolates them anyway), marks terminal atoms D1 and
    other kept atoms non-terminal. E.g. SO2 extrusion becomes "delete any non-terminal atom and
    join two of its neighbours"; S->O migration becomes "move a terminal atom one bond over".
    """
    pattern = Chem.MolFromSmarts(rule["smarts"])
    by_map = {a.GetAtomMapNum(): a for a in pattern.GetAtoms() if a.GetAtomMapNum()}
    touched = {m for pair in rule.get("form", []) + rule.get("cut", []) for m in pair}
    deleted = set(rule.get("delete", []))
    keep = set(touched)
    for m in deleted:
        atom = by_map[m]
        if not (_is_terminal_query(atom) and any(n.GetAtomMapNum() in deleted for n in atom.GetNeighbors())):
            keep.add(m)
    keep_idx = {by_map[m].GetIdx(): m for m in keep}
    pieces = {}
    for idx, m in keep_idx.items():
        atom = pattern.GetAtomWithIdx(idx)
        if _is_terminal_query(atom):
            pieces[idx] = f"[*;D1:{m}]"
        elif atom.GetDegree() > 1 or m in deleted:
            pieces[idx] = f"[*;!D1:{m}]"
        else:
            pieces[idx] = f"[*:{m}]"
    order = sorted(pieces)
    rw = Chem.RWMol(Chem.MolFromSmarts(".".join(pieces[i] for i in order)))
    local = {idx: k for k, idx in enumerate(order)}
    for bond in pattern.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in local and b in local:
            rw.AddBond(local[a], local[b], Chem.BondType.SINGLE)
    text = Chem.MolToSmarts(rw)
    return text.replace("-", "~")


def _edit_edges(mol, match: dict[int, int], rule: dict):
    deleted = {match[m] for m in rule.get("delete", []) if m in match}
    cut = {frozenset((match[a], match[b])) for a, b in rule.get("cut", [])}
    pairs = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in deleted or b in deleted or frozenset((a, b)) in cut:
            continue
        pairs.append((a, b))
    for a, b in rule.get("form", []):
        pairs.append((match[a], match[b]))
    return tuple(zip(*pairs)) if pairs else ((), ())


def _matches(mol, pattern, limit: int | None = None) -> list[dict[int, int]]:
    maps = [a.GetAtomMapNum() for a in pattern.GetAtoms()]
    out = []
    for hit in mol.GetSubstructMatches(pattern, uniquify=False, maxMatches=5000):
        out.append({m: idx for m, idx in zip(maps, hit) if m})
    # Deduplicate on the edit's own atoms.
    seen, unique = set(), []
    for m in out:
        key = tuple(sorted(m.items()))
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique[:limit] if limit else unique


def _reaches(mol, edge_sets, comp: Counter, h: int, size: int) -> str:
    capped = False
    for edges in edge_sets:
        found = reachable_h(mol, edges, comp, size, cap=REACH_CAP, required_h=frozenset([h]))
        if found is None:
            capped = True
        elif any(abs(h - v) <= 4 for v in found):
            return "hit"
    return "capped" if capped else "miss"


def _work(task):
    RDLogger.DisableLog("rdApp.*")
    key, smiles, ions = task
    mol = Chem.MolFromSmiles(smiles)
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    hyd = [a.GetTotalNumHs() for a in mol.GetAtoms()]
    templates = filter_masster_templates(
        build_masster_templates(mol), precursor_type=PRECURSOR, fragmentation_mode=None
    )
    sides = [
        (Counter(sym[i] for i in side), sum(hyd[i] for i in side))
        for t in templates for side in t.components
    ]
    rows = []
    for rule in _RULES:
        real = _matches(mol, rule["_pattern"])
        if not real:
            continue
        real_sets = [_edit_edges(mol, m, rule) for m in real]
        null_maps = [a.GetAtomMapNum() for a in rule["_null_pattern"].GetAtoms()]
        real_atoms = {frozenset(m[k] for k in null_maps) for m in real}
        pool = [m for m in _matches(mol, rule["_null_pattern"]) if frozenset(m.values()) not in real_atoms]
        draws = []
        if pool:
            for _ in range(NULL_DRAWS):
                pick = rng.choice(len(pool), min(len(real), len(pool)), replace=False)
                draws.append([_edit_edges(mol, pool[i], rule) for i in pick])
        for ion in ions:
            comp = parse_formula(ion)
            h = comp.pop("H", 0)
            comp = Counter(comp)
            size = sum(comp.values())
            if size == 0 or size > mol.GetNumAtoms():
                continue
            null = [_reaches(mol, d, comp, h, size) for d in draws]
            rows.append(
                {
                    "key": key,
                    "rule": rule["name"],
                    "ion": ion,
                    "n_matches": len(real),
                    "real": _reaches(mol, real_sets, comp, h, size),
                    "null_hit_rate": (np.mean([x == "hit" for x in null]) if null else np.nan),
                    "null_capped": any(x == "capped" for x in null),
                    "registered": any(c == comp and abs(hh - h) <= 4 for c, hh in sides),
                }
            )
    return key, rows


def _init(rules, precursor="[M+H]+"):
    global _RULES
    RDLogger.DisableLog("rdApp.*")
    global PRECURSOR
    _RULES = rules
    PRECURSOR = precursor
    for rule in _RULES:
        rule["_pattern"] = Chem.MolFromSmarts(rule["smarts"])
        rule["_null_pattern"] = Chem.MolFromSmarts(rule.get("null_smarts") or topology_null_smarts(rule))


def bootstrap_diff(per_mol: pd.DataFrame, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(per_mol) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    diff = (per_mol.real - per_mol.null).to_numpy()
    w = per_mol.w.to_numpy()
    stats = []
    for _ in range(reps):
        i = rng.integers(0, len(diff), len(diff))
        stats.append((diff[i] * w[i]).sum() / w[i].sum())
    return tuple(np.quantile(stats, [0.025, 0.975]))


def summarize(rows: pd.DataFrame, rearr: pd.DataFrame, molecules: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    peaks = rearr.merge(rows, on=["key", "ion"], how="inner")
    peaks["hit"] = peaks.real == "hit"
    # rows were concatenated with peak-less match rows, so these columns arrive as object dtype.
    peaks["registered"] = peaks.registered.fillna(False).astype(bool)
    peaks["null_ok"] = peaks.null_hit_rate.notna()
    peaks["null_w"] = peaks.null_hit_rate.fillna(0)
    out = []
    groups = [(r["name"], peaks[peaks.rule == r["name"]]) for r in rules]
    union = peaks.groupby(["key", "spec_id", "mz"], as_index=False).agg(
        weight=("weight", "first"), share=("share", "first"), split=("split", "first"),
        dataset=("dataset", "first"), hit=("hit", "any"), null_w=("null_w", "max"),
        registered=("registered", "any"), null_ok=("null_ok", "all"),
    )
    groups.append(("UNION", union))
    for name, d in groups:
        if d.empty:
            out.append({"rule": name})
            continue
        if name != "UNION":
            d = d.drop_duplicates(["key", "spec_id", "mz"])
        matched = (
            molecules.key.isin(rows.loc[rows.rule == name, "key"]).sum() if name != "UNION" else np.nan
        )
        row = {"rule": name, "matched_mols_with_rearr": d.key.nunique(), "matched_mols_all_scope": matched}
        for label, sel in (("all", rearr), ("ge10", rearr[rearr.share >= 0.10]), ("ge25", rearr[rearr.share >= 0.25])):
            total = sel.weight.sum()
            dd = d[d.spec_id.isin(sel.spec_id) & d.key.isin(sel.key)]
            row[f"real_{label}_pct"] = 100 * dd.weight[dd.hit].sum() / total
            row[f"null_{label}_pct"] = 100 * (dd.weight * dd.null_w).sum() / total
            ok = dd[dd.null_ok]
            row[f"real_on_null_ok_{label}_pct"] = 100 * ok.weight[ok.hit].sum() / total
            row[f"registered_{label}_pct"] = 100 * dd.weight[dd.registered].sum() / total
            row[f"new_vs_registered_{label}_pct"] = 100 * dd.weight[dd.hit & ~dd.registered].sum() / total
        dn = d[d.null_ok]
        per_mol = dn.assign(r=dn.weight * dn.hit, n=dn.weight * dn.null_w).groupby("key").agg(
            real=("r", "sum"), null=("n", "sum"), w=("weight", "sum"))
        per_mol["real"] /= per_mol.w
        per_mol["null"] /= per_mol.w
        lo, hi = bootstrap_diff(per_mol)
        row.update(
            {
                "null_available_mols": f"{dn.key.nunique()}/{d.key.nunique()}",
                "mol_real_frac": per_mol.real.mean(), "mol_null_frac": per_mol.null.mean(),
                "diff_lo": lo, "diff_hi": hi,
                "mols_explained": int((per_mol.real > 0).sum()),
                "val_mols_explained": d[d.hit & (d.split == "val")].key.nunique(),
                "train_mols_explained": d[d.hit & (d.split == "train")].key.nunique(),
                "capped_frac": (d.real == "capped").mean() if "real" in d else np.nan,
            }
        )
        if name != "UNION" and matched:
            row["precision_mols"] = row["mols_explained"] / matched
        out.append(row)
    return pd.DataFrame(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="molecules, for a smoke run")
    parser.add_argument("--mining", type=Path, default=MINING, help="mining output directory")
    parser.add_argument("--precursor", default="[M+H]+", help="precursor type of the mined peaks")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rules = json.loads(Path(args.rules).read_text())

    rearr = pd.read_pickle(args.mining / "rearrangement.pkl")
    rearr["key"] = rearr.dataset + ":" + rearr.mol_id.astype(str)
    if "ion_mol" in rearr:
        # Adduct mining: test the molecular part of the ion (the adduct is not in the graph).
        rearr["ion_full"] = rearr.ion
        rearr["ion"] = rearr.ion_mol
    share = rearr.groupby(["dataset", "spec_id"]).weight.sum().rename("share")
    rearr = rearr.join(share, on=["dataset", "spec_id"])
    molecules = pd.read_csv(args.mining / "molecules.csv")

    # Molecules matching any rule (all scope, for precision) and their rearrangement ions.
    _init(rules, args.precursor)
    patterns = [r["_pattern"] for r in rules]
    keep = []
    for key, smiles in zip(molecules.key, molecules.smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and any(mol.HasSubstructMatch(p) for p in patterns):
            keep.append(key)
    ions = rearr[rearr.key.isin(keep)].groupby("key").ion.unique()
    smiles_of = molecules.set_index("key").smiles
    tasks = [(k, smiles_of[k], sorted(ions.get(k, []))) for k in keep]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(keep)} molecules match a rule; {len(ions)} have rearrangement peaks", flush=True)

    plain = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rules]
    rows = []
    with Pool(args.workers, initializer=_init, initargs=(plain, args.precursor)) as pool:
        for i, (_, result) in enumerate(pool.imap_unordered(_work, tasks, chunksize=4)):
            rows.extend(result)
            if i % 500 == 0:
                print(f"{i}/{len(tasks)}", flush=True)
    rows = pd.DataFrame(rows)
    # Molecules matching a rule but without rearrangement peaks still count for precision.
    matched_rows = []
    for k in keep:
        mol = Chem.MolFromSmiles(smiles_of[k])
        for r in rules:
            if mol.HasSubstructMatch(r["_pattern"]):
                matched_rows.append({"key": k, "rule": r["name"], "ion": None})
    rows = pd.concat([rows, pd.DataFrame(matched_rows)], ignore_index=True) if len(rows) else pd.DataFrame(matched_rows)
    rows.to_pickle(out / "rows.pkl")
    summary = summarize(rows.dropna(subset=["ion"]) if "real" in rows else rows, rearr, molecules, rules)
    # matched_mols_all_scope needs the full rows (including peak-less matches).
    for i, r in summary.iterrows():
        if r["rule"] != "UNION":
            matched = rows.loc[rows.rule == r["rule"], "key"].nunique()
            summary.loc[i, "matched_mols_all_scope"] = matched
            if matched:
                summary.loc[i, "precision_mols"] = summary.loc[i, "mols_explained"] / matched
    summary.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(summary.round(3).T.to_string())


if __name__ == "__main__":
    main()
