"""Does MASShatter get the bond-type response to CE right but the molecule's own bond wrong?

The ICEBERG-2 check (Extended Data Fig. 3a), run on MASShatter and extended to the question
the intensity-displacement plan cares about. One checkpoint is evaluated in eval mode on a
train slice and a validation slice. For every retained fragment the probe records the bonds
severed at its boundary, the model's fragment mass (summed over ion/H channels) and the
target mass matched to its channels. Each fragment's mass is split equally over its cut
bonds; the intact precursor goes to a pseudo-bond ``precursor``. Bonds are typed by element
pair and order (``C-O``, ``C=O``, ``C:C`` aromatic, ...).

Per spectrum, with ``p_b`` and ``t_b`` the predicted and target bond-level distributions:

    TV_bond   = 1/2 sum_b |p_b - t_b|
    TV_type   = 1/2 sum_k |sum_{b in k} p_b - sum_{b in k} t_b|     (types incl. precursor)
    TV_within = TV_bond - TV_type                                    (>= 0)

``TV_type`` is the error a correct bond-type-by-CE response would remove; ``TV_within`` is
the error left in choosing WHICH bond of a type breaks in this molecule. If the aggregate
type trends match on train and val while ``TV_within`` grows on val, the transfer failure
sits inside bond types, which is what the plan's local lability prior (2A) targets.

Target mass at an m/z shared by several channels is split equally among them (model-free);
``--responsibility_split`` splits it by the model's own channel mass instead, as a check.

Usage (repository root, local GPU):

    python scripts/eval/bond_type_ce_allocation.py \
        --custom_fp config/massshatter/mh_ft_clean_2026_09_22/f1_mh_ft_all_filters.yml \
        --model_ckpt saved_ckpts/f1_mh_ft_all_filters_6h67zps9.ckpt \
        --batch_cutoff 40 --eval_batch_size 256 --out_fp bond_type_f1.pkl
"""

from __future__ import annotations

import argparse
import inspect
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch as th
from lightning.pytorch import seed_everything
from rdkit import Chem
from tqdm import tqdm

from fragnnet.runner import init_dataloader, load_config
from fragnnet.utils.script_utils import apply_inference_config, init_model_from_ckpt, setup_device
from fragnnet.utils.setup_utils import get_dataset_cls

LOG = logging.getLogger("bond_type_ce_allocation")
TOL_PPM, TOL_FLOOR = 10e-6, 200.0
ORDER_CHAR = {1.0: "-", 2.0: "=", 3.0: "#", 1.5: ":"}
PRECURSOR = "precursor"


class FragmentCapture:
    """Record per-fragment validity and boundary bonds from one MASShatter forward pass.

    Wraps two methods of the model instance. In the unique-fragment path,
    ``_fragment_masses_and_valid_channels`` runs on the enumerated fragments and
    ``_boundary_embeddings`` on the fragments that survive ``fragment_keep``. The probe
    requires those two sets to be equal (``min_fragment_relative_probability: 0`` with
    ``mask_order`` selection) and raises otherwise, so fragment indices align exactly.
    """

    def __init__(self, model: th.nn.Module) -> None:
        self.model = model
        self.data: dict[str, Any] = {}
        self._orig_valid = model._fragment_masses_and_valid_channels
        self._orig_boundary = model._boundary_embeddings
        self._boundary_sig = inspect.signature(self._orig_boundary)

    def __enter__(self) -> FragmentCapture:
        def valid(*args: Any, **kwargs: Any) -> Any:
            out = self._orig_valid(*args, **kwargs)
            self.data["valid_channels"] = out[1].detach()
            return out

        def boundary(*args: Any, **kwargs: Any) -> Any:
            bound = self._boundary_sig.bind(*args, **kwargs)
            self.data["boundary_bond_indices"] = bound.arguments["boundary_bond_indices"].detach()
            self.data["boundary_component_indices"] = (
                bound.arguments["boundary_component_indices"].detach()
            )
            self.data["num_components"] = int(bound.arguments["num_components"])
            return self._orig_boundary(*args, **kwargs)

        self.model._fragment_masses_and_valid_channels = valid
        self.model._boundary_embeddings = boundary
        return self

    def __exit__(self, *exc: object) -> None:
        self.model._fragment_masses_and_valid_channels = self._orig_valid
        self.model._boundary_embeddings = self._orig_boundary

    def fragment_bonds(self, num_output_fragments: int) -> list[np.ndarray]:
        """Cut-bond indices (global chemical-bond ids) of each output fragment."""
        valid = self.data["valid_channels"]
        if self.data["num_components"] != valid.shape[0]:
            raise RuntimeError(
                "fragment_keep dropped fragments; this probe needs keep == all "
                f"({self.data['num_components']} kept vs {valid.shape[0]} enumerated)"
            )
        chem_valid = valid.flatten(1).any(dim=1).cpu().numpy()
        if int(chem_valid.sum()) != num_output_fragments:
            raise RuntimeError("validity mask does not match the output fragment count")
        new_index = np.cumsum(chem_valid) - 1
        bonds = self.data["boundary_bond_indices"].cpu().numpy()
        comps = self.data["boundary_component_indices"].cpu().numpy()
        out: list[list[int]] = [[] for _ in range(num_output_fragments)]
        for b, c in zip(bonds, comps):
            if chem_valid[c]:
                out[new_index[c]].append(int(b))
        return [np.asarray(x, dtype=np.int64) for x in out]


def bond_labels(smiles: str, n_atoms: int, pairs: np.ndarray) -> list[str]:
    """Type label per (local atom a, local atom b) bond, checked against RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != n_atoms:
        raise ValueError(f"atom count mismatch for {smiles}: graph {n_atoms}")
    labels = []
    for a, b in pairs:
        bond = mol.GetBondBetweenAtoms(int(a), int(b))
        if bond is None:
            raise ValueError(f"no RDKit bond {a}-{b} in {smiles}")
        ea, eb = sorted((mol.GetAtomWithIdx(int(a)).GetSymbol(), mol.GetAtomWithIdx(int(b)).GetSymbol()))
        labels.append(f"{ea}{ORDER_CHAR.get(bond.GetBondTypeAsDouble(), '?')}{eb}")
    return labels


def tv_decomposition(p: dict[Any, float], t: dict[Any, float], types: dict[Any, str]) -> dict[str, float]:
    """Bond-level, type-level and within-type total variation between two distributions.

    Args:
        p: Predicted mass per bond key (sums to 1).
        t: Target mass per bond key (sums to 1).
        types: Type label of every key in ``p`` or ``t``.

    Returns:
        ``tv_bond``, ``tv_type`` and ``tv_within = tv_bond - tv_type``.
    """
    keys = set(p) | set(t)
    tv_bond = 0.5 * sum(abs(p.get(k, 0.0) - t.get(k, 0.0)) for k in keys)
    pk: dict[str, float] = defaultdict(float)
    tk: dict[str, float] = defaultdict(float)
    for k in keys:
        pk[types[k]] += p.get(k, 0.0)
        tk[types[k]] += t.get(k, 0.0)
    tv_type = 0.5 * sum(abs(pk[k] - tk[k]) for k in set(pk) | set(tk))
    return {"tv_bond": tv_bond, "tv_type": tv_type, "tv_within": tv_bond - tv_type}


def match_groups(chan_mz: list[np.ndarray], tm: np.ndarray, ti: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge channels by m/z (1e-5 rounding) and match each target peak to its nearest group.

    Returns:
        Group index per channel (concatenated over fragments), target intensity per group, and
        the per-peak covered mask (within 10 ppm, 200 m/z floor).
    """
    mz = np.round(np.concatenate(chan_mz), 5)
    umz, inv = np.unique(mz, return_inverse=True)
    j = np.clip(np.searchsorted(umz, tm), 1, len(umz) - 1)
    near = np.where(np.abs(tm - umz[j - 1]) <= np.abs(tm - umz[j]), j - 1, j)
    cov = np.abs(umz[near] - tm) <= TOL_PPM * np.maximum(tm, TOL_FLOOR)
    group_t = np.bincount(near[cov], weights=ti[cov], minlength=len(umz))
    return inv, group_t, cov


def spectrum_records(
    frag_mass: np.ndarray, chan_mz: list[np.ndarray], chan_w: list[np.ndarray],
    frag_bonds: list[np.ndarray], tm: np.ndarray, ti: np.ndarray, responsibility: bool,
) -> tuple[dict[Any, float], dict[Any, float], float]:
    """Bond-level predicted and target distributions for one spectrum.

    Args:
        frag_mass: Model mass per fragment, shape ``(F,)``.
        chan_mz: Per fragment, m/z of its valid channels.
        chan_w: Per fragment, model mass of those channels.
        frag_bonds: Per fragment, its cut-bond ids (empty for the precursor).
        tm, ti: Target peaks.
        responsibility: Split shared-m/z target mass by model channel mass, not equally.

    Returns:
        Predicted and target bond distributions (normalized) and the covered target fraction.
    """
    fid = np.concatenate([np.full(len(m), f) for f, m in enumerate(chan_mz)])
    w = np.concatenate(chan_w)
    inv, group_t, _ = match_groups(chan_mz, tm, ti)
    if responsibility:
        group_w = np.bincount(inv, weights=w, minlength=len(group_t))
        share = np.where(group_w[inv] > 0, w / np.maximum(group_w[inv], 1e-30), 0.0)
    else:
        share = 1.0 / np.bincount(inv, minlength=len(group_t))[inv]
    frag_t = np.bincount(fid, weights=group_t[inv] * share, minlength=len(chan_mz))
    covered = float(group_t.sum() / ti.sum())
    p: dict[Any, float] = defaultdict(float)
    t: dict[Any, float] = defaultdict(float)
    for f, bonds in enumerate(frag_bonds):
        keys = [PRECURSOR] if len(bonds) == 0 else list(bonds)
        for k in keys:
            p[k] += frag_mass[f] / len(keys)
            t[k] += frag_t[f] / len(keys)
    ps, ts = sum(p.values()), sum(t.values())
    p = {k: v / ps for k, v in p.items() if v > 0}
    t = {k: v / ts for k, v in t.items() if v > 0} if ts > 0 else {}
    return p, t, covered


def bond_index_map(smiles: str, pairs: np.ndarray) -> list[int]:
    """RDKit bond index of every (local atom a, local atom b) pair."""
    mol = Chem.MolFromSmiles(smiles)
    return [mol.GetBondBetweenAtoms(int(a), int(b)).GetIdx() for a, b in pairs]


def run_split(
    model, dataloader, device, split: str, batch_cutoff: int, responsibility: bool,
    bond_rows: list[dict] | None = None, fragment_level: bool = False,
) -> list[dict]:
    """Evaluate one split and return per-spectrum records plus per-(spectrum, type) masses.

    With ``bond_rows`` given, also appends one record per spectrum holding every candidate
    bond's RDKit index (``-1`` for the precursor), model mass ``p`` and target mass under the
    equal (``t_eq``) and responsibility (``t_resp``) splits. ``fragment_level`` adds the
    fragment and channel arrays needed to re-score the spectrum at peak level.
    """
    rows = []
    label_cache: dict[str, dict[tuple[int, int], str]] = {}
    index_cache: dict[str, dict[tuple[int, int], int]] = {}
    mm = model.model
    with th.inference_mode(), FragmentCapture(mm) as cap:
        for bi, batch in enumerate(tqdm(dataloader, desc=f"bond-type {split}", unit="batch")):
            if 0 <= batch_cutoff <= bi:
                break
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            out = model.forward(**batch)
            logw = out["ion_h_dense_log_weights"].float()
            valid = out["ion_h_chemical_valid_channels"]
            mzs = out["ion_h_dense_mzs"].float()
            fb = out["ion_h_fragment_batch_idxs"].cpu().numpy()
            frag_bonds = cap.fragment_bonds(len(fb))
            w = th.where(valid, logw.exp(), th.zeros_like(logw))
            frag_mass = w.flatten(1).sum(1).cpu().numpy()
            valid_np = valid.flatten(1).cpu().numpy()
            w_np = w.flatten(1).cpu().numpy()
            mz_np = mzs.flatten(1).cpu().numpy()
            mol_pyg = batch["mol_pyg"]
            ptr = mol_pyg.ptr.cpu().numpy()
            cbi = mol_pyg.chemical_bond_index.cpu().numpy()
            bond_mol = np.searchsorted(ptr, cbi[0], side="right") - 1
            smiles = batch["mol_smiles"]
            tmz = batch["spec_mzs"].float().cpu().numpy()
            tint = batch["spec_ints"].float().cpu().numpy()
            tbi = batch["spec_batch_idxs"].cpu().numpy()
            nce = np.zeros(int(batch["batch_size"]))
            cnt = np.zeros(int(batch["batch_size"]))
            if batch.get("spec_nce") is not None:
                np.add.at(nce, batch["spec_nce_batch_idxs"].cpu().numpy(), batch["spec_nce"].float().cpu().numpy())
                np.add.at(cnt, batch["spec_nce_batch_idxs"].cpu().numpy(), 1.0)
            nce = nce / np.maximum(cnt, 1)
            uid = batch["spec_unique_id"]
            for s in range(int(batch["batch_size"])):
                fs = np.flatnonzero(fb == s)
                sel = tbi == s
                if len(fs) == 0 or not sel.any():
                    continue
                # Bond types for this molecule (local atom indices checked against RDKit).
                mol_bonds = np.flatnonzero(bond_mol == s)
                key = smiles[s]
                if key not in label_cache:
                    local = cbi[:, mol_bonds].T - ptr[s]
                    label_cache[key] = dict(zip(map(tuple, local), bond_labels(key, int(ptr[s + 1] - ptr[s]), local)))
                labels = label_cache[key]
                gid_to_label = {int(g): labels[tuple(cbi[:, g] - ptr[s])] for g in mol_bonds}
                cm = [mz_np[f][valid_np[f]] for f in fs]
                cw = [w_np[f][valid_np[f]] for f in fs]
                fbs = [frag_bonds[f] for f in fs]
                p, t, covered = spectrum_records(frag_mass[fs], cm, cw, fbs, tmz[sel], tint[sel], responsibility)
                if not t:
                    continue
                if bond_rows is not None:
                    if key not in index_cache:
                        local = cbi[:, mol_bonds].T - ptr[s]
                        index_cache[key] = dict(zip(map(tuple, local), bond_index_map(key, local)))
                    t_alt = spectrum_records(frag_mass[fs], cm, cw, fbs, tmz[sel], tint[sel], not responsibility)[1]
                    t_eq, t_resp = (t_alt, t) if responsibility else (t, t_alt)
                    keys = sorted(set(p) | set(t_eq) | set(t_resp), key=str)
                    ridx = [-1 if k == PRECURSOR else index_cache[key][tuple(cbi[:, int(k)] - ptr[s])] for k in keys]
                    rec_b = {
                        "split": split, "spec_unique_id": int(uid[s]), "smiles": key, "nce": nce[s],
                        "covered": covered, "bond_idx": np.asarray(ridx, dtype=np.int32),
                        "p": np.array([p.get(k, 0.0) for k in keys]),
                        "t_eq": np.array([t_eq.get(k, 0.0) for k in keys]),
                        "t_resp": np.array([t_resp.get(k, 0.0) for k in keys]),
                    }
                    if fragment_level:
                        # Peak-level inputs. The retained fragment set depends on NCE (the proposal
                        # stage keeps up to U fragments per spectrum), so it is stored per spectrum:
                        # each fragment's cut bonds (RDKit index), each valid channel's fragment,
                        # m/z group and model mass, and the target mass per group.
                        tsum = tint[sel].sum()
                        inv, group_t, cov = match_groups(cm, tmz[sel], tint[sel])
                        fb_r = [[index_cache[key][tuple(cbi[:, int(g)] - ptr[s])] for g in b] for b in fbs]
                        rec_b.update({
                            "frag_bond_flat": np.asarray([x for b in fb_r for x in b], dtype=np.int16),
                            "frag_bond_ptr": np.cumsum([0] + [len(b) for b in fb_r]).astype(np.int32),
                            "chan_frag": np.concatenate([np.full(len(m), f, dtype=np.int16) for f, m in enumerate(cm)]),
                            "chan_group": inv.astype(np.int16),
                            "chan_w": np.concatenate(cw).astype(np.float32),
                            "group_t": (group_t / tsum).astype(np.float32),
                            "unc_t2": float(((tint[sel][~cov] / tsum) ** 2).sum()),
                        })
                    bond_rows.append(rec_b)
                types = {k: (PRECURSOR if k == PRECURSOR else gid_to_label[int(k)]) for k in set(p) | set(t)}
                dec = tv_decomposition(p, t, types)
                # Within the target's dominant non-precursor type: does the model pick the same bond?
                type_t: dict[str, float] = defaultdict(float)
                type_p: dict[str, float] = defaultdict(float)
                for k, v in t.items():
                    type_t[types[k]] += v
                for k, v in p.items():
                    type_p[types[k]] += v
                top_type_t = max(type_t, key=type_t.get)
                top_type_p = max(type_p, key=type_p.get)
                rec = {"split": split, "spec_unique_id": uid[s], "smiles": key, "nce": nce[s],
                       "covered": covered, **dec, "top_type_hit": top_type_t == top_type_p}
                bond_types = [k for k in type_t if k != PRECURSOR]
                if bond_types:
                    kt = max(bond_types, key=type_t.get)
                    members = [g for g, lab in gid_to_label.items() if lab == kt]
                    cut_members = [g for g in members if g in p or g in t]
                    tb = max((g for g in cut_members if g in t), key=lambda g: t[g])
                    pb = max(cut_members, key=lambda g: p.get(g, 0.0))
                    rec.update({"within_type": kt, "within_hit": tb == pb,
                                "within_n": len(cut_members)})
                rows.append(rec | {f"p_{k}": v for k, v in type_p.items()} | {f"t_{k}": v for k, v in type_t.items()})
    return rows


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--template_fp", default="config")
    p.add_argument("--custom_fp", required=True)
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--batch_cutoff", type=int, default=40)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--responsibility_split", action="store_true")
    p.add_argument("--out_fp", required=True)
    p.add_argument("--bond_records_fp", default=None,
                   help="Also write per-spectrum, per-bond masses here (input to bond_local_learner.py).")
    p.add_argument("--fragment_level", action="store_true",
                   help="With --bond_records_fp, also store fragment/channel arrays for peak-level scoring.")
    return p.parse_args()


def main() -> None:
    args = get_args()
    config_d = load_config(args.template_fp, args.custom_fp, [])
    if config_d["model_type"] != "mass_shatter" or not config_d["mass_shatter_unique_fragment_mode"]:
        raise ValueError("needs a unique-fragment MASShatter config")
    config_d = apply_inference_config(
        config_d, disable_preproc=True, eval_batch_size=args.eval_batch_size, num_workers=args.num_workers
    )
    device = setup_device(config_d)
    seed_everything(config_d["seed"], workers=True)
    model = init_model_from_ckpt(args.model_ckpt, config_d, device)
    model.eval()
    rows = []
    bond_rows: list[dict] | None = [] if args.bond_records_fp else None
    for split in args.splits:
        seed_everything(config_d["seed"], workers=True)
        dataset = get_dataset_cls(config_d["model_type"])(split=split, **config_d)
        rows += run_split(model, init_dataloader(dataset, config_d), device, split, args.batch_cutoff,
                          args.responsibility_split, bond_rows, args.fragment_level)
        pd.DataFrame(rows).to_pickle(args.out_fp)
        if bond_rows is not None:
            pd.DataFrame(bond_rows).to_pickle(args.bond_records_fp)
        LOG.info("%s done, %d rows so far", split, len(rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
