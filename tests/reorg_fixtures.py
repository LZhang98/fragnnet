"""Shared synthetic fixture for the reorganization safety net (plan Phase 0).

Everything is built in code from SMILES and synthetic peak lists and written under a caller's
``tmp_path``; nothing reads the untracked ``data/`` directory. The pieces:

- ``write_synth_data``: spec_df/mol_df pickles, split CSVs and (optionally) FraGNNet DAGs.
- ``make_config``: ``load_config`` for a model type plus small-model overrides, typo-checked
  against the template and validated with the runner's own sampler check.
- ``init_split``: the runner's ``init_dataset`` + ``init_dataloader`` for one split.
- ``deterministic``: the plan's CPU determinism settings, restored on exit.
- ``record_one_step``: exactly one optimizer step through a real Lightning ``Trainer``, returning
  every loss term, the raw gradients and the parameters before and after.
- ``digest_record``: a JSON-serializable, bit-exact digest of tensors for the golden files.

The recipes behind these choices (file formats, required columns, config constraints) were
verified on 2026-10-11; see docs/plans/repo_reorg_plan.md, Phase 0.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# DAG generation through joblib reads this when fragnnet.utils.data_utils is first imported; the
# fixture generates DAGs serially anyway, so this only avoids spawning workers elsewhere.
os.environ.setdefault("FRAGNNET_JOBLIB_N_JOBS", "1")

import h5py  # noqa: E402
import lightning as L  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch as th  # noqa: E402
from lightning import seed_everything  # noqa: E402
from lightning.pytorch.callbacks import Callback  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.Descriptors import ExactMolWt  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402
from torch.utils.data import BatchSampler, DataLoader, Dataset  # noqa: E402

from fragnnet import runner  # noqa: E402
from fragnnet.utils import frag_utils  # noqa: E402
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF  # noqa: E402

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO_ROOT / "config")
NONDETERMINISTIC = re.compile(r"deterministic implementation|nondeterministic", re.IGNORECASE)

# DAG generation settings; the FraGNNet config must use num_depth = max_depth + 1 and
# num_hs <= max_h_transfer (fragnnet_model.py:982, :1829).
DAG_MAX_DEPTH = 3
DAG_MAX_H_TRANSFER = 2

SMILES = ("CCO", "CC(=O)O", "CNC(C)=O", "Oc1ccccc1", "NCC(=O)O", "CC(C)O")

# Ethanol's predicted m/z set does not depend on the weights for either model (MASSter enumerates
# every cut mask under the fixture budgets; FraGNNet predicts every DAG formula), so these two
# literal peaks always fall inside a predicted tolerance window: C2H5+ and CH3O+.
CE_PAIR_A_MZ = 29.0386
CE_PAIR_B_MZ = 31.0178


@dataclass(frozen=True)
class Spectrum:
    """One synthetic spectrum. ``peaks=None`` derives peaks from the molecule's DAG."""

    spec_id: int
    mol_id: int
    group_id: int
    nce: float
    split: str
    peaks: tuple[tuple[float, float], ...] | None = None
    prec_type: str = "[M+H]+"
    inst_type: str = "ORB"


# Default table. Group 0 (ethanol, two CEs) is a CE-pair-eligible group with literal peaks: A is
# the base peak at the low CE and B at the high CE. Group 1 is a single-CE group. Groups 2-4 have
# two CEs with DAG-derived peaks; group 5 is single-CE.
DEFAULT_SPECTRA = (
    Spectrum(0, 0, 0, 20.0, "train", ((CE_PAIR_A_MZ, 1000.0), (CE_PAIR_B_MZ, 400.0))),
    Spectrum(1, 0, 0, 40.0, "train", ((CE_PAIR_A_MZ, 300.0), (CE_PAIR_B_MZ, 1000.0))),
    Spectrum(2, 1, 1, 30.0, "train"),
    Spectrum(3, 2, 2, 20.0, "train"),
    Spectrum(4, 2, 2, 40.0, "train"),
    Spectrum(5, 3, 3, 20.0, "train"),
    Spectrum(6, 3, 3, 40.0, "train"),
    Spectrum(7, 4, 4, 20.0, "val"),
    Spectrum(8, 4, 4, 40.0, "val"),
    Spectrum(9, 5, 5, 30.0, "test"),
)


@dataclass
class SynthData:
    """Paths and frames of one synthetic dataset written by ``write_synth_data``."""

    root: Path
    spec_fp: str
    mol_fp: str
    split_dp: str
    frag_dp: str | None
    spec_df: pd.DataFrame
    mol_df: pd.DataFrame
    dags: dict[int, dict] = field(default_factory=dict)


def build_mol_df(smiles: tuple[str, ...] = SMILES) -> pd.DataFrame:
    """mol_df with mol_id = position in ``smiles`` (dense, as the packed DAG index requires)."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return pd.DataFrame(
        {
            "mol_id": pd.Series(range(len(mols)), dtype="int64"),
            "smiles": [Chem.MolToSmiles(m) for m in mols],
            "mol": mols,
            "formula": [CalcMolFormula(m) for m in mols],
            "exact_mw": [float(ExactMolWt(m)) for m in mols],
            "num_bonds": pd.Series([m.GetNumBonds() for m in mols], dtype="int64"),
        }
    )


def generate_dags(mol_df: pd.DataFrame) -> dict[int, dict]:
    """{mol_id: dag_d}, generated serially in-process (no joblib)."""
    dags = {}
    for mol_id, smiles in zip(mol_df["mol_id"], mol_df["smiles"]):
        dags[int(mol_id)] = frag_utils.compute_dags(
            frag_utils.extract_mol_info(smiles),
            DAG_MAX_DEPTH,
            True,
            DAG_MAX_H_TRANSFER,
            int(1e6),
            False,
            False,
            -1,
            multi_cut_bfs=False,
        )
    return dags


def dag_candidate_mzs(dag_d: dict, prec_type: str = "[M+H]+") -> np.ndarray:
    """Distinct fragment m/z above 20 that the DAG predicts for ``prec_type`` (precursor excluded)."""
    diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
    mzs = dag_d["formula_peak_mzs"][1:, 0].numpy().astype(np.float64) + diff
    mzs = np.unique(np.round(mzs, 4))
    return mzs[mzs > 20.0]


def dag_peaks(
    dag_d: dict, nce: float, prec_type: str = "[M+H]+"
) -> tuple[tuple[float, float], ...]:
    """Four DAG candidate peaks whose base peak moves from one m/z at low CE to another at high."""
    mzs = dag_candidate_mzs(dag_d, prec_type)
    pick = mzs[np.linspace(0, len(mzs) - 1, num=min(4, len(mzs))).round().astype(int)]
    w = 1.0 if nce > 30.0 else 0.0
    ints = [100.0 * (1 - w) + 30.0 * w, 60.0, 40.0, 25.0 * (1 - w) + 100.0 * w][: len(pick)]
    return tuple((float(m), float(i)) for m, i in zip(pick, ints))


def write_dags(dags: dict[int, dict], frag_root: Path, backend: str, meta_info: bool) -> str:
    """Write DAGs as packed ``dags.h5`` or a folder of uncompressed pickles; return ``frag_dp``.

    ``meta_info.json`` ({mol_id: [nodes, edges]}, raw DAG counts) sits beside ``frag_dp``. Its
    presence changes which batches the dynamic samplers form, so callers choose it explicitly.
    """
    frag_root.mkdir(parents=True, exist_ok=True)
    if backend == "h5":
        frag_dp = str(frag_root / "dags.h5")
        with h5py.File(frag_dp, "w") as h5_file:
            frag_utils.initialize_packed_dag_hdf5(h5_file)
            for mol_id, dag_d in dags.items():
                frag_utils.dump_dag_hdf5_packed(h5_file, mol_id, dag_d)
    elif backend == "folder":
        frag_dp = str(frag_root / "dags")
        os.makedirs(frag_dp, exist_ok=True)
        for mol_id, dag_d in dags.items():
            frag_utils.save_frag_d(dag_d, mol_id, frag_dp, is_compressed=False)
    else:
        raise ValueError(f"unknown DAG backend {backend!r}")
    if meta_info:
        meta = {str(k): [d["dag"].num_nodes, d["dag"].num_edges] for k, d in dags.items()}
        (frag_root / "meta_info.json").write_text(json.dumps(meta))
    return frag_dp


def write_synth_data(
    root: Path,
    spectra: tuple[Spectrum, ...] = DEFAULT_SPECTRA,
    smiles: tuple[str, ...] = SMILES,
    dag_backend: str | None = "h5",
    meta_info: bool = True,
) -> SynthData:
    """Write a synthetic dataset under ``root`` in the formats the dataset classes read.

    Args:
        root: Directory to write into (a pytest ``tmp_path``).
        spectra: Spectra to write; ``peaks=None`` derives four DAG candidate peaks.
        smiles: Molecules; ``Spectrum.mol_id`` indexes this tuple.
        dag_backend: ``"h5"`` (packed dags.h5), ``"folder"`` (uncompressed pickles) or ``None``
            (no DAG files; DAGs are still generated to derive peaks).
        meta_info: Write ``meta_info.json`` beside the DAGs.

    Returns:
        The written paths and frames.
    """
    root = Path(root)
    mol_df = build_mol_df(smiles)
    dags = generate_dags(mol_df)
    rows = []
    for spec in spectra:
        peaks = spec.peaks or dag_peaks(dags[spec.mol_id], spec.nce, spec.prec_type)
        exact_mw = float(mol_df.loc[spec.mol_id, "exact_mw"])
        rows.append(
            {
                "spec_id": spec.spec_id,
                "mol_id": spec.mol_id,
                "group_id": spec.group_id,
                "prec_type": spec.prec_type,
                "inst_type": spec.inst_type,
                "prec_mz": exact_mw + PREC_TYPE_TO_MASS_DIFF[spec.prec_type],
                "peaks": [(float(m), float(i)) for m, i in peaks],
                "nce": float(spec.nce),
                "ace": np.nan,
                "nce_extra_1": np.nan,
                "ace_extra_1": np.nan,
                "nce_extra_2": np.nan,
                "ace_extra_2": np.nan,
            }
        )
    spec_df = pd.DataFrame(rows).astype(
        {"spec_id": "int64", "mol_id": "int64", "group_id": "int64"}
    )
    (root / "split").mkdir(parents=True, exist_ok=True)
    spec_fp, mol_fp = str(root / "spec_df.pkl"), str(root / "mol_df.pkl")
    spec_df.to_pickle(spec_fp)
    mol_df.to_pickle(mol_fp)
    split_of = {spec.spec_id: spec.split for spec in spectra}
    for split in ("train", "val", "test"):
        ids = [sid for sid, s in split_of.items() if s == split]
        spec_df[spec_df["spec_id"].isin(ids)][["spec_id", "mol_id", "group_id"]].to_csv(
            root / "split" / f"{split}_ids.csv", index=False
        )
    frag_dp = None
    if dag_backend is not None:
        frag_dp = write_dags(dags, root / "frag", dag_backend, meta_info)
    return SynthData(root, spec_fp, mol_fp, str(root / "split"), frag_dp, spec_df, mol_df, dags)


# Overrides every fixture config gets. Template values that break a CPU fixture are changed here:
# devices (CE-context samplers need 1), num_workers, and sparse_cosine_similarity (both fragment
# models raise without it). spec_params keep same-group spectra apart and turn NCE on.
COMMON_OVERRIDES: dict[str, Any] = {
    "accelerator": "cpu",
    "precision": 32,
    "devices": 1,
    "compile": False,
    "num_workers": 0,
    "pin_memory": False,
    "seed": 42,
    "deterministic": True,
    "sparse_cosine_similarity": True,
    "spec_params": {"merge": False, "merge_train_only": False, "nce": True, "ace": False},
}

MODEL_OVERRIDES: dict[str, dict[str, Any]] = {
    "frag_gnn": {
        "num_depth": DAG_MAX_DEPTH + 1,
        "num_hs": DAG_MAX_H_TRANSFER,
        "mol_hidden_size": 32,
        "mol_num_layers": 2,
        "mol_dropout": 0.0,
        "frag_hidden_size": 32,
        "frag_num_layers": 2,
        "mlp_hidden_size": 32,
    },
    "masster": {
        "masster_candidate_backend": "cpu",
        "masster_proposal_method": "cut_masks",
        "masster_selected_mask_limit": 64,
        "masster_unique_fragment_target": 64,
        "masster_ranked_mask_budget": 64,
        "masster_max_h_transfer": 1,
        "masster_mask_selection_threads": 1,
        "masster_head_hidden_dim": 32,
        "masster_intensity_transformer_dim": 32,
        "masster_intensity_transformer_num_heads": 4,
        "masster_intensity_transformer_feedforward_dim": 64,
        "masster_intensity_transformer_num_inducing_points": 8,
        "mol_hidden_size": 32,
        "mol_num_layers": 2,
        "mol_dropout": 0.0,
        # MASSter's float64 predictions break the non-vectorized Gaussian CE path.
        "loss_vectorized": True,
    },
}
MODEL_OVERRIDES["fragnnet"] = MODEL_OVERRIDES["frag_gnn"]

# Everything Phase 0 item 4 needs to activate CE-pair, spectral leak and spectral CE auxiliary.
ENABLED_LOSS_OVERRIDES: dict[str, dict[str, Any]] = {
    "shared": {
        "loss_type": "hellinger_distance",
        "loss_vectorized": True,
        "spectral_leak_loss_weight": 0.1,
        "spectral_ce_auxiliary_weight": 0.1,
        "ce_pair_loss_weight": 0.1,
        "ce_pair_min_endpoint_confidence": 0.2,
        "target_rel_ints_thresh": 0.05,
        "ce_pair_eligible_inst_types": ["ORB"],
        "ce_pair_eligible_prec_types": ["[M+H]+"],
        "inst_type_loss_tol": {"ORB": {"rel": 1.0e-5, "min_mz": 200.0}},
        "train_sample_weight": "none",
    },
    "frag_gnn": {
        "ce_insert_location": "mol",
        "ce_insert_type": "embed_clamped",
        "batch_sampler_type": "ce_context",
        "dynamic_batch_sampler": True,
        "dynamic_batch_sampler_mode": "frag_edge",
        "dynamic_batch_sampler_max": 10000,
        "automatic_optimization": False,
    },
    "masster": {
        "masster_batch_sampler": True,
        "masster_batch_sampler_max": 1000,
    },
}
ENABLED_LOSS_OVERRIDES["fragnnet"] = ENABLED_LOSS_OVERRIDES["frag_gnn"]


def _merge_overrides(config: dict, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if key not in config:
            raise KeyError(f"override {key!r} is not a template key")
        if isinstance(value, dict) and isinstance(config[key], dict):
            unknown = set(value) - set(config[key])
            if unknown and key not in runner.OPEN_CONFIG_MAPPINGS:
                raise KeyError(f"override {key}.{sorted(unknown)} is not a template key")
            config[key] = {**config[key], **value}
        else:
            config[key] = value


def make_config(
    model_type: str,
    data: SynthData | None = None,
    *overrides: dict[str, Any],
    enabled_losses: bool = False,
) -> dict:
    """Resolved config for ``model_type`` on ``data``.

    Args:
        model_type: Any registered model type.
        data: The synthetic dataset; its paths are filled in when given.
        *overrides: Extra override mappings, applied last and in order.
        enabled_losses: Add ``ENABLED_LOSS_OVERRIDES`` (CE-pair, leak, auxiliary).

    Returns:
        The config, after ``runner.validate_sampler_config``.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        config = runner.load_config(CONFIG_DIR, None, None, model_type=model_type)
    layers = [COMMON_OVERRIDES, MODEL_OVERRIDES.get(model_type, {})]
    if enabled_losses:
        layers += [ENABLED_LOSS_OVERRIDES["shared"], ENABLED_LOSS_OVERRIDES[model_type]]
    if data is not None:
        paths = {"spec_fp": data.spec_fp, "mol_fp": data.mol_fp, "split_dp": data.split_dp}
        if data.frag_dp is not None:
            paths["frag_dp"] = data.frag_dp
            if not data.frag_dp.endswith(".h5"):
                paths["frag_params"] = {"compressed": False}
        layers.append(paths)
    for layer in (*layers, *overrides):
        _merge_overrides(config, layer)
    runner.validate_sampler_config(config)
    return config


def init_split(config: dict, split: str) -> tuple[Any, DataLoader]:
    """The runner's dataset and dataloader for one split (stdout silenced)."""
    with contextlib.redirect_stdout(io.StringIO()):
        (dataset,) = runner.init_dataset(config, splits=(split,))
        loader = runner.init_dataloader(dataset, config)
    return dataset, loader


def close_dataset(dataset: Any) -> None:
    """Close a FraGNNet dataset's open HDF5 handle before ``tmp_path`` teardown."""
    if getattr(dataset, "frag_h5", None) is not None:
        dataset.frag_h5.close()
        dataset.frag_h5 = None


@contextlib.contextmanager
def deterministic(seed: int = 0) -> Iterator[list[warnings.WarningMessage]]:
    """CPU determinism settings of the plan, restored on exit.

    Seeds everything, enables deterministic algorithms (warn-only, like the runner) and pins one
    intra-op thread, which bit-exact gradients require. Yields the list of warnings raised inside
    the block, and raises if any of them reports a non-deterministic op.
    """
    previous = (
        th.are_deterministic_algorithms_enabled(),
        th.is_deterministic_algorithms_warn_only_enabled(),
        th.get_num_threads(),
    )
    seed_everything(seed, workers=True, verbose=False)
    th.use_deterministic_algorithms(True, warn_only=True)
    th.set_num_threads(1)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield caught
    finally:
        th.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        th.set_num_threads(previous[2])
    bad = [str(w.message) for w in caught if NONDETERMINISTIC.search(str(w.message))]
    if bad:
        raise AssertionError(f"non-deterministic op under deterministic mode: {bad}")


def build_module(config: dict, seed: int = 0) -> L.LightningModule:
    """A fresh Lightning module for ``config``, seeded before construction."""
    from fragnnet.utils.setup_utils import get_model_cls

    seed_everything(seed, workers=True, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return get_model_cls(config["model_type"])(**config)


class _OneBatchDataset(Dataset):
    def __init__(self, batch: dict) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        return self.batch


class _OneBatchSampler(BatchSampler):
    """One fixed batch; exposes ``set_epoch`` because CE-context configs require it."""

    def __init__(self) -> None:
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        yield [0]

    def __len__(self) -> int:
        return 1


def one_batch_loader(batch: dict) -> DataLoader:
    """A train loader that yields a deep copy of ``batch`` once per epoch."""
    return DataLoader(
        _OneBatchDataset(copy.deepcopy(batch)),
        batch_sampler=_OneBatchSampler(),
        collate_fn=lambda items: items[0],
    )


class _StepCapture(Callback):
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.grads: dict[str, th.Tensor] = {}
        self.n_backward = 0
        self.n_step = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        seed_everything(self.seed, workers=False, verbose=False)

    def on_after_backward(self, trainer, pl_module) -> None:
        # Raw gradient, before any rescaling or clipping, in both optimization modes.
        self.n_backward += 1
        self.grads = {
            n: p.grad.detach().clone()
            for n, p in pl_module.named_parameters()
            if p.grad is not None
        }

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        self.n_step += 1


_LOSS_INPUT_KEYS = (
    "true_mzs",
    "true_logprobs",
    "true_batch_idxs",
    "pred_mzs",
    "pred_logprobs",
    "pred_intensities",
    "pred_batch_idxs",
    "ce_pair_ce_values",
    "tol_per_sample",
    "min_mz_per_sample",
)


def record_one_step(pl_module: L.LightningModule, batch: dict, seed: int = 1234) -> dict:
    """Run exactly one optimizer step on CPU and return a flat {key: tensor} record.

    Keys: ``loss/<term>`` (every tensor ``loss_fn`` returns), ``reduce/<name>`` (mean_loss,
    total_loss, total_weight), ``in/<kwarg>`` (loss inputs), ``grad/<param>`` (raw gradient) and
    ``param_after/<state key>``. Build a fresh module per record: the step mutates it.

    Must run inside ``deterministic()``. Works for automatic and manual optimization: in manual
    mode the batch must hold at least ``train_batch_size * accumulate_grad_batches`` spectra, or
    the step would run at epoch end.
    """
    hparams = pl_module.hparams
    capture = _StepCapture(seed)
    terms: list[dict] = []
    inputs: list[dict] = []
    reduces: list[dict] = []
    orig_loss_fn = pl_module.loss_fn
    orig_common_step = pl_module._common_step

    def loss_fn(**kwargs):
        out = orig_loss_fn(**kwargs)
        terms.append({k: v.detach().clone() for k, v in out.items() if th.is_tensor(v)})
        inputs.append(
            {k: kwargs[k].detach().clone() for k in _LOSS_INPUT_KEYS if th.is_tensor(kwargs.get(k))}
        )
        return out

    def common_step(step_batch, split="train", log=True):
        result = orig_common_step(step_batch, split=split, log=log)
        reduces.append(
            {k: result[k].detach().clone() for k in ("mean_loss", "total_loss", "total_weight")}
        )
        return result

    trainer_kwargs = {}
    if hparams.automatic_optimization:
        trainer_kwargs = {
            "accumulate_grad_batches": hparams.accumulate_grad_batches,
            "gradient_clip_val": hparams.gradient_clip_val,
            "gradient_clip_algorithm": hparams.gradient_clip_algorithm,
        }
    trainer = L.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        precision=32,
        max_epochs=1,
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[capture],
        **trainer_kwargs,
    )
    pl_module.loss_fn = loss_fn
    pl_module._common_step = common_step
    try:
        pl_module.train()
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.fit(pl_module, one_batch_loader(batch))
    finally:
        pl_module.loss_fn = orig_loss_fn
        del pl_module._common_step
    assert (capture.n_backward, capture.n_step, len(terms)) == (1, 1, 1), (
        capture.n_backward,
        capture.n_step,
        len(terms),
    )
    assert trainer.global_step == 1, trainer.global_step
    missing = set(pl_module.loss_names) - set(terms[0])
    assert not missing, f"loss_fn did not return named terms {sorted(missing)}"
    record = {f"loss/{k}": v for k, v in terms[0].items()}
    record.update({f"reduce/{k}": v for k, v in reduces[0].items()})
    record.update({f"in/{k}": v for k, v in inputs[0].items()})
    record.update({f"grad/{k}": v for k, v in capture.grads.items()})
    record.update(
        {f"param_after/{k}": v.detach().clone() for k, v in pl_module.state_dict().items()}
    )
    return record


def tensor_digest(value: th.Tensor | np.ndarray) -> str:
    """``dtype|shape|sha256[:16]`` of a tensor's bytes; bit-exact."""
    array = value.detach().cpu().contiguous().numpy() if th.is_tensor(value) else value
    array = np.ascontiguousarray(array)
    sha = hashlib.sha256(array.tobytes()).hexdigest()[:16]
    return f"{array.dtype}|{tuple(array.shape)}|{sha}"


def digest_record(record: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable digest of a record: tensors become ``tensor_digest`` strings.

    Small tensors (at most 8 elements) also keep their exact values (``float.hex``), so a failing
    comparison shows what changed. Lists, tuples, dicts and scalars are digested recursively.
    """

    def one(value: Any) -> Any:
        if th.is_tensor(value) or isinstance(value, np.ndarray):
            digest = tensor_digest(value)
            flat = value.detach().cpu().reshape(-1) if th.is_tensor(value) else value.reshape(-1)
            if flat.shape[0] <= 8:
                values = [float(x).hex() if isinstance(x, float) else x for x in flat.tolist()]
                return {"digest": digest, "values": values}
            return digest
        if isinstance(value, dict):
            return {str(k): one(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, (list, tuple)):
            return [one(v) for v in value]
        if isinstance(value, float):
            return float(value).hex()
        if isinstance(value, (np.integer, np.floating)):
            return one(value.item())
        if hasattr(value, "to_dict") and hasattr(value, "keys"):  # torch_geometric Data/Batch
            return {k: one(value[k]) for k in sorted(value.keys())}
        return value

    return {str(k): one(v) for k, v in sorted(record.items(), key=lambda kv: str(kv[0]))}
