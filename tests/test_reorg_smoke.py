"""Smoke test per retained model type and the dataset construction path (plan Phase 0, items 2, 9).

Item 2: for every ``model_type`` the runner accepts except the retired ICEBERG pair, build the
template plus a tiny config on synthetic data, run one forward pass and one optimizer step on
CPU, and check that the loss is finite and that a gradient flows. A bit-exact digest of the
forward outputs, loss terms and gradient summary is recorded in ``tests/golden/reorg/smoke.json``.
The digest holds behavior only (keys, dtypes, shapes, values), never class names or module paths,
because Phase 1 changes some of those (``massformer`` moves to ``SpecMolMFDataset``).

Two types need deliberate handling, and neither is faked:

- ``chemlm`` reads a Hugging Face checkpoint. The test writes a tiny random RoBERTa checkpoint and
  a byte-level tokenizer to ``tmp_path`` (as ``test_chemlm.py`` does), so it neither downloads nor
  reads the untracked ``data/pretrained`` checkpoint. It skips without ``transformers``.
- ``mces_pretraining`` cannot be built at the recorded commit: ``MCESPL._setup_model``
  (``pl_model/mces_pl.py:50``) omits the ``use_nce``, ``use_ace`` and ``ace_*`` arguments
  ``SiameseGNNModel`` requires. The shipped configuration is a strict ``xfail`` that flips when
  the wiring is fixed; a second case patches the missing arguments in the test so the dataset,
  loss and optimizer step still have coverage. That patched record uses this test's own
  constants (the CE switches from ``spec_params`` and the ACE scaling from the config, as the
  sibling wrappers pass them), so it guards the MCES loss and data path, not the shipped wiring.
  The plan's "random-init ``state_dict`` record for mces" (Phase 6, step 5) is unachievable until
  ``MCESPL`` is fixed; delete the patched case and the xfail then (see ``MCES_DEFECT``).

Each case also records an eval-mode forward pass (after the training forward, so BatchNorm
running statistics are those of one training batch); it must return the same output keys.

Item 9: a dataset built through ``get_dataset_cls("masster")(**config)``, bypassing
``runner.init_dataset``, still applies the MASSter spectrum filter. The record holds the dataset
frames and counts after filtering (not constants of the test), and a hydrogen-free molecule shows
that ``masster_max_h_transfer``, ``masster_include_cmf`` and ``masster_h_transfer_bound`` reach
the filter, including through the legacy ``mass_shatter`` spelling.

Phase 1 and later change paths, and the edits that follow are made in the same PR as the move:
``get_model_cls`` and ``get_dataset_cls`` stay importable from ``fragnnet.utils.setup_utils`` as
thin wrappers (plan, registry table), ``fragnnet.utils.data_utils`` and
``fragnnet.utils.formula_utils`` stay in core, and ``masster_has_valid_ion`` (in MASSter's
``frag/masster_dataset_filter`` since Phase 4) is reached only through the dataset constructor,
never imported here. The
Graphormer loader that the MassFormer case patches is found through the model classes, not a
literal module path, so moving the baseline out of core does not break the patch.

Regenerate the golden records deliberately (never to make a failing refactor pass):

    REORG_UPDATE_GOLDEN=1 pytest tests/test_reorg_smoke.py
"""

from __future__ import annotations

import contextlib
import copy
import functools
import importlib
import io
import json
import math
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import lightning as L
import pandas as pd
import pytest
import reorg_fixtures as F
import torch as th
import yaml
from lightning.pytorch.callbacks import Callback
from pyteomics.mass import Composition
from reorg_golden import ratchet_json, ratchet_records
from reorg_layers import MODEL_TYPES

from fragnnet import runner
from fragnnet.utils.data_utils import composition_to_string, formula_to_mass
from fragnnet.utils.formula_utils import NEUTRON_MASS
from fragnnet.utils.setup_utils import get_dataset_cls, get_model_cls

GOLDEN = "smoke"
RETIRED_TYPES = ("iceberg_gen", "iceberg_inten")
SMOKE_TYPES = tuple(t for t in MODEL_TYPES if t not in RETIRED_TYPES)
ELECTRON_MASS = 0.00054858
STEP_SEED = 1234
TIMING_SUFFIX = "_ms"

# Overrides shared by the binned-output baselines (NEIMS, GNN, ChemLM, MassFormer). The output
# is 2000 bins instead of the template's 150000, and the MLP and GNN shrink to 16 units.
BINNED_OVERRIDES: dict[str, Any] = {
    "loss_type": "cosine_distance",
    "mz_max": 200.0,
    "mz_bin_res": 0.1,
    "ff_output_map_size": 8,
    "mlp_hidden_size": 16,
    "mlp_num_layers": 1,
    "mlp_dropout": 0.0,
    "mol_hidden_size": 16,
    "mol_num_layers": 1,
    "mol_dropout": 0.0,
    "ce_insert_type": "embed",
    "ce_insert_location": "mlp",
    "ce_insert_merge": True,
}
NEIMS_OVERRIDES: dict[str, Any] = {
    **BINNED_OVERRIDES,
    "mol_params": {"fingerprint": True, "morgan_nbits": 64, "rdkit_nbits": 64},
}
MASSFORMER_OVERRIDES: dict[str, Any] = {
    **BINNED_OVERRIDES,
    "mol_params": {"pyg": False, "mf": True, "preprocess": False},
    "mf_lr_schedule": False,
    "ff_prec_mz_offset": 50,
}
CHEMLM_OVERRIDES: dict[str, Any] = {
    **BINNED_OVERRIDES,
    "chemlm_num_layers": 2,
    "chemlm_p_trav": 0.0,
    "weight_decay_exclude_norm_bias": True,
}
GRAFF_OVERRIDES: dict[str, Any] = {
    "mol_params": {"pyg": False, "graff": True, "preprocess": False},
    "ann_params": {"library_size": 8, "num_eigs": 4, "max_isotope": 2},
    "graff_encoder_dim": 16,
    "graff_decoder_dim": 16,
    "graff_encoder_depth": 1,
    "graff_decoder_depth": 1,
    "graff_eig_dim": 8,
    "graff_eig_depth": 1,
    "graff_dropout": 0.0,
    "output_distribution": "peak_marginal",
    "gaussian_renormalize": True,
    "pm_tolerance_multiple": 1,
    "oos_tolerance_multiple": 1,
}
CLIP_OVERRIDES: dict[str, Any] = {
    "mol_hidden_size": 16,
    "mol_num_layers": 1,
    "mol_dropout": 0.0,
    "mlp_hidden_size": 16,
    "mlp_num_layers": 1,
    "mlp_dropout": 0.0,
    "ce_insert_type": "embed",
    "ce_insert_location": "mlp",
    "ce_insert_merge": True,
    "clip_embed_dim": 16,
    # input_dim 4 = (mz, intensity, neutral loss, precursor scale); the template's 2 is stale.
    "spectrum_encoder_params": {
        "input_dim": 4,
        "hidden_dim": 16,
        "embed_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "num_inds": 4,
        "dropout": 0.0,
    },
}
MCES_OVERRIDES: dict[str, Any] = {
    "mlp_hidden_size": 16,
    "mlp_num_layers": 1,
    "mlp_dropout": 0.0,
    "mol_hidden_size": 16,
    "mol_num_layers": 1,
    "mol_dropout": 0.0,
    "loss_type": "mse",
    "auxiliary_scores": ["mae", "mse"],
}


# --------------------------------------------------------------------------------------------
# Step machinery
# --------------------------------------------------------------------------------------------


@dataclass
class StepResult:
    """One optimizer step: the loss, its named terms, raw gradients and the parameters after."""

    loss: th.Tensor
    terms: dict[str, th.Tensor]
    grads: dict[str, th.Tensor]
    params_after: dict[str, th.Tensor]


class _StepProbe(Callback):
    """Capture the loss and raw gradients of the single optimizer step of a ``Trainer.fit``."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.loss: th.Tensor | None = None
        self.grads: dict[str, th.Tensor] = {}
        self.n_backward = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        L.seed_everything(self.seed, workers=False, verbose=False)

    def on_after_backward(self, trainer, pl_module) -> None:
        self.n_backward += 1
        self.grads = {
            name: param.grad.detach().clone()
            for name, param in pl_module.named_parameters()
            if param.grad is not None
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.loss = loss.detach().clone()


def _generic_step(module: L.LightningModule, batch: dict) -> StepResult:
    """One automatic-optimization step through a real ``Trainer`` for non-``SpectrumPL`` modules."""
    probe = _StepProbe(STEP_SEED)
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
        callbacks=[probe],
    )
    module.train()
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.fit(module, F.one_batch_loader(batch))
    assert (probe.n_backward, trainer.global_step) == (1, 1)
    assert probe.loss is not None
    params_after = {k: v.detach().clone() for k, v in module.state_dict().items()}
    return StepResult(probe.loss, {"loss": probe.loss}, probe.grads, params_after)


def _spectrum_step(module: L.LightningModule, batch: dict) -> StepResult:
    """One step through the shared ``record_one_step`` helper for ``SpectrumPL`` modules."""
    if not hasattr(module, "loss_names"):
        # The binned loss declares its names in a local variable (binned_pl.py:91), not on self.
        module.loss_names = ["loss", "spec_cd"]
    record = F.record_one_step(module, batch)
    terms = {k.removeprefix("loss/"): v for k, v in record.items() if k.startswith("loss/")}
    grads = {k.removeprefix("grad/"): v for k, v in record.items() if k.startswith("grad/")}
    params = {
        k.removeprefix("param_after/"): v for k, v in record.items() if k.startswith("param_after/")
    }
    return StepResult(record["reduce/mean_loss"], terms, grads, params)


# --------------------------------------------------------------------------------------------
# Cases: how each model type is configured, fed and called
# --------------------------------------------------------------------------------------------


@dataclass
class Case:
    """Everything one smoke run needs for a model type.

    Attributes:
        model_type: The registered type whose class is built.
        config: Resolved tiny config.
        batch: One collated training batch.
        forward: Calls the module's forward pass on a batch.
        spectrum_pl: Whether the module is a ``SpectrumPL`` (per-term loss dictionary).
        trains: False for a model with no trainable signal (``precursor``).
        close: Releases dataset handles before ``tmp_path`` teardown.
    """

    model_type: str
    config: dict
    batch: dict
    forward: Callable[[L.LightningModule, dict], Any]
    spectrum_pl: bool = True
    trains: bool = True
    close: Callable[[], None] = lambda: None


CaseBuilder = Callable[[Path, pytest.MonkeyPatch], Case]


def _first_batch(loader: Any) -> dict:
    """First batch of a loader with its sampler generator pinned (the default is global state)."""
    generator = getattr(getattr(loader, "sampler", None), "generator", None)
    if generator is not None:
        generator.manual_seed(0)
    return next(iter(loader))


def _forward_kwargs(module: L.LightningModule, batch: dict) -> Any:
    return module.forward(**batch)


def _spectrum_case(
    model_type: str,
    root: Path,
    *overrides: dict[str, Any],
    dag_backend: str | None = None,
    **case_kwargs: Any,
) -> Case:
    """A ``SpectrumPL`` case on the default synthetic data."""
    data = F.write_synth_data(root, dag_backend=dag_backend)
    config = F.make_config(model_type, data, *overrides)
    dataset, loader = F.init_split(config, "train")
    return Case(
        model_type,
        config,
        _first_batch(loader),
        _forward_kwargs,
        close=lambda: F.close_dataset(dataset),
        **case_kwargs,
    )


def _build_frag_gnn(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    return _spectrum_case("frag_gnn", root, dag_backend="h5")


def _build_fragnnet(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    return _spectrum_case("fragnnet", root, dag_backend="h5")


def _build_masster(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    return _spectrum_case("masster", root)


def _build_neims(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    return _spectrum_case("neims", root, NEIMS_OVERRIDES)


def _build_gnn(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    return _spectrum_case("gnn", root, BINNED_OVERRIDES)


def _build_precursor(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    # PrecursorModel multiplies its only parameter by 0.0 (precursor_model.py:15): by design
    # nothing trains, so its gradient is exactly zero and the step leaves every weight alone.
    return _spectrum_case("precursor", root, trains=False)


def _write_tiny_roberta(ckpt: Path, num_layers: int = 4) -> str:
    """Write a random RoBERTa-MLM checkpoint with a hand-built byte-level tokenizer.

    The vocabulary is the special tokens plus the 256 byte-level symbols and there are no merges,
    so the tokenizer needs no training and is identical on every run.

    Args:
        ckpt: Directory to create.
        num_layers: Encoder layers of the checkpoint.

    Returns:
        The checkpoint directory.
    """
    transformers = pytest.importorskip("transformers", reason="chemlm needs transformers")
    tokenizers = pytest.importorskip("tokenizers", reason="chemlm needs tokenizers")
    ckpt.mkdir()
    special = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: idx for idx, token in enumerate(special + alphabet)}
    (ckpt / "vocab.json").write_text(json.dumps(vocab))
    (ckpt / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = transformers.RobertaTokenizerFast(
        vocab_file=str(ckpt / "vocab.json"), merges_file=str(ckpt / "merges.txt")
    )
    tokenizer.save_pretrained(str(ckpt))
    config = transformers.RobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=66,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        type_vocab_size=1,
    )
    th.manual_seed(0)
    transformers.RobertaForMaskedLM(config).save_pretrained(str(ckpt))
    return str(ckpt)


def _build_chemlm(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    ckpt = _write_tiny_roberta(root / "tiny_chemlm")
    return _spectrum_case("chemlm", root / "data", CHEMLM_OVERRIDES, {"chemlm_ckpt_dp": ckpt})


def _massformer_nn_utils() -> ModuleType:
    """The module that defines the Graphormer embedder and its pretrained-weight loaders.

    Found through the classes rather than a literal path, so the Phase 1 move of the baseline
    out of core needs no edit here: ``MassFormerPL`` -> its module -> ``MassFormerModel`` -> its
    module -> ``GFv2Embedder`` -> the module that defines it.
    """
    pl_module = importlib.import_module(get_model_cls("massformer").__module__)
    model_module = importlib.import_module(pl_module.MassFormerModel.__module__)
    return importlib.import_module(model_module.GFv2Embedder.__module__)


def _shrink_graphormer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep MassFormer offline and small.

    ``MassFormerModel`` always builds Graphormer-base and downloads its pretrained weights from
    Zenodo (``massformer/model.py:50``). The weights are replaced by a random state dict of the
    same architecture, and the encoder shrinks to one layer of width 32 (251k parameters instead
    of 48M). Any real download attempt fails loudly.
    """
    mfnn = _massformer_nn_utils()

    original_args = mfnn.set_graphormer_base_architecture_args

    def shrunk(args: Any) -> None:
        original_args(args)
        args.encoder_layers, args.encoder_embed_dim = 1, 32
        args.encoder_ffn_embed_dim, args.encoder_attention_heads = 32, 2

    def random_pretrained(name: str) -> dict:
        return mfnn.GFv2Embedder(gf_pretrain_name="none").state_dict()

    def no_download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the smoke test must not download Graphormer weights")

    monkeypatch.setattr(mfnn, "set_graphormer_base_architecture_args", shrunk)
    monkeypatch.setattr(mfnn, "load_pretrained_model", random_pretrained)
    monkeypatch.setattr(mfnn, "load_state_dict_from_url", no_download)


def _build_massformer(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    _shrink_graphormer(monkeypatch)
    return _spectrum_case("massformer", root, MASSFORMER_OVERRIDES)


# Fragment products of [M+H]+: the protonated precursor plus water and methane losses.
def _products(formula: str) -> list[str]:
    precursor = Composition(formula=formula)
    precursor["H"] += 1
    out = [composition_to_string(precursor)]
    for loss in ("H2O", "CH4"):
        product = precursor - Composition(formula=loss)
        if all(count >= 0 for count in product.values()):
            out.append(composition_to_string(product))
    return out


# GRAFF's eigenvectors are only unique up to sign when the Laplacian spectrum has no repeated
# eigenvalue in the kept slots; acetic acid has one, so these three molecules are used instead.
GRAFF_SMILES = ("CCO", "C=CC(=O)O", "Oc1ccccc1")


def _write_graff_data(root: Path) -> tuple[F.SynthData, str]:
    """Write annotated spectra and the ``ann_df`` that GRAFF's library is fitted from.

    Every peak is an annotated fragment of ``[M+H]+`` (precursor, minus H2O, minus CH4), spectrum
    0 also carries an M+1 isotope peak, and every intensity is distinct so the library ranking
    (an unstable sort on summed intensity, graff/data_utils.py:195) has no ties.

    Args:
        root: Directory to write into.

    Returns:
        The synthetic data and the path of ``ann_df.pkl``.
    """
    mol_df = F.build_mol_df(GRAFF_SMILES)
    layout = [(0, 20.0, "train"), (0, 40.0, "train"), (1, 20.0, "train"), (1, 40.0, "train")]
    layout += [(2, 20.0, "val"), (2, 40.0, "val")]
    spectra, ann_rows = [], []
    for spec_id, (mol_id, nce, split) in enumerate(layout):
        products = _products(mol_df.loc[mol_id, "formula"])
        intensities = [1.0 - 0.1 * spec_id, 0.5 - 0.01 * spec_id, 0.25 - 0.003 * spec_id]
        entries = [(product, 0, intensities[k]) for k, product in enumerate(products)]
        if spec_id == 0:
            entries.append((products[0], 1, 0.05))
        peaks, ann = [], {"spec_id": spec_id, "ann_peak_mzs": [], "ann_products": []}
        ann.update({"ann_losses": [], "ann_isotopes": [], "ann_exact_mzs": []})
        for product, isotope, intensity in entries:
            exact = formula_to_mass(product) + isotope * NEUTRON_MASS
            mz = round(exact - ELECTRON_MASS, 4)
            peaks.append((mz, round(100.0 * intensity, 6)))
            ann["ann_peak_mzs"].append(f"{mz:.4f}")
            ann["ann_products"].append(product)
            loss = Composition(formula=products[0]) - Composition(formula=product)
            ann["ann_losses"].append(composition_to_string(loss))
            ann["ann_isotopes"].append(isotope)
            ann["ann_exact_mzs"].append(exact)
        spectra.append(F.Spectrum(spec_id, mol_id, mol_id, nce, split, tuple(sorted(peaks))))
        ann_rows.append(ann)
    data = F.write_synth_data(root, tuple(spectra), GRAFF_SMILES, dag_backend=None)
    spec_df = pd.read_pickle(data.spec_fp)
    spec_df["dset"] = "synthetic"
    spec_df["dset_spec_id"] = [f"synthetic_{spec_id}" for spec_id in spec_df["spec_id"]]
    spec_df.to_pickle(data.spec_fp)
    ann_fp = root / "ann_df.pkl"
    pd.DataFrame(ann_rows).to_pickle(ann_fp)
    return data, str(ann_fp)


def _build_graff(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    data, ann_fp = _write_graff_data(root)
    config = F.make_config("graff", data, GRAFF_OVERRIDES, {"ann_fp": ann_fp})
    dataset, loader = F.init_split(config, "train")
    ranks = dataset.library_df["ints"]
    assert not ranks.duplicated().any(), "tied library intensities make the library order unstable"
    return Case("graff", config, _first_batch(loader), _forward_kwargs)


def _clip_forward(module: L.LightningModule, batch: dict) -> tuple[th.Tensor, th.Tensor]:
    return module.forward(batch)


def _build_clip_family(model_type: str, root: Path) -> Case:
    data = F.write_synth_data(root, dag_backend=None)
    config = F.make_config(model_type, data, CLIP_OVERRIDES)
    _, loader = F.init_split(config, "train")
    return Case(model_type, config, _first_batch(loader), _clip_forward, spectrum_pl=False)


def _build_clip(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    return _build_clip_family("clip", root)


def _build_spec_mol_clip(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    return _build_clip_family("spec_mol_clip_pretraining", root)


def _write_mces_data(root: Path, data: F.SynthData) -> str:
    """Write every molecule pair with a synthetic MCES distance and a train split of all pairs.

    Args:
        root: Directory to write into.
        data: The synthetic molecules.

    Returns:
        Path of ``mces_pairs.pkl``; the split directory is ``root / "mces_split"``.
    """
    n_mols = len(data.mol_df)
    pairs = [(i, j) for i in range(n_mols) for j in range(i + 1, n_mols)]
    mces_df = pd.DataFrame(
        {
            "mol_id_1": [i for i, _ in pairs],
            "mol_id_2": [j for _, j in pairs],
            "mces": [float(2 + (3 * i + 5 * j) % 9) for i, j in pairs],
        }
    )
    mces_fp = root / "mces_pairs.pkl"
    mces_df.to_pickle(mces_fp)
    split_dp = root / "mces_split"
    split_dp.mkdir()
    ids = mces_df.reset_index(names="mces_id")[["mol_id_1", "mol_id_2", "mces_id"]]
    for split in ("train", "val", "test"):
        ids.to_csv(split_dp / f"{split}_ids.csv", index=False)
    return str(mces_fp)


def _mces_forward(module: L.LightningModule, batch: dict) -> th.Tensor:
    return module.forward(batch["mol_1_mol_pyg"], batch["mol_2_mol_pyg"])


def _build_mces(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    data = F.write_synth_data(root, dag_backend=None)
    overrides = {
        **MCES_OVERRIDES,
        "mces_fp": _write_mces_data(root, data),
        "split_dp": str(root / "mces_split"),
    }
    config = F.make_config("mces_pretraining", data, overrides)
    _, loader = F.init_split(config, "train")
    return Case("mces_pretraining", config, _first_batch(loader), _mces_forward, spectrum_pl=False)


def _patch_siamese_wiring(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    """Supply the ``SiameseGNNModel`` arguments ``MCESPL._setup_model`` leaves out.

    The values are the ones every sibling wrapper passes: the CE switches from ``spec_params``
    and the ACE scaling constants from the config.
    """
    from fragnnet.model import SiameseGNNModel

    module = importlib.import_module(get_model_cls("mces_pretraining").__module__)
    patched = functools.partial(
        SiameseGNNModel,
        use_nce=config["spec_params"]["nce"],
        use_ace=config["spec_params"]["ace"],
        ace_max=config["ace_max"],
        ace_mean=config["ace_mean"],
        ace_std=config["ace_std"],
    )
    monkeypatch.setattr(module, "SiameseGNNModel", patched)


MCES_PATCHED = "mces_pretraining_wiring_patched"


def _build_mces_patched(root: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    case = _build_mces(root, monkeypatch)
    _patch_siamese_wiring(monkeypatch, case.config)
    return case


CASES: dict[str, CaseBuilder] = {
    "frag_gnn": _build_frag_gnn,
    "fragnnet": _build_fragnnet,
    "masster": _build_masster,
    "neims": _build_neims,
    "chemlm": _build_chemlm,
    "massformer": _build_massformer,
    "graff": _build_graff,
    "precursor": _build_precursor,
    "gnn": _build_gnn,
    "mces_pretraining": _build_mces,
    "clip": _build_clip,
    "spec_mol_clip_pretraining": _build_spec_mol_clip,
    MCES_PATCHED: _build_mces_patched,
}
MCES_DEFECT = (
    "MCESPL._setup_model omits use_nce, use_ace, ace_max, ace_mean and ace_std, which "
    "SiameseGNNModel requires, so model_type mces_pretraining cannot be built at this commit. "
    "Once the wiring is fixed this case stops raising: drop the xfail mark, delete the "
    "wiring-patched case and its blocked-only-by test, then record its golden with "
    "REORG_UPDATE_GOLDEN=1."
)
# GRAFF's dataset assigns into a slice of a DataFrame (graff/dataset.py:112); harmless, and only
# this case may ignore it.
CASE_MARKS: dict[str, list[Any]] = {
    "mces_pretraining": [pytest.mark.xfail(raises=TypeError, strict=True, reason=MCES_DEFECT)],
    "graff": [pytest.mark.filterwarnings("ignore::pandas.errors.SettingWithCopyWarning")],
}
SMOKE_PARAMS = [pytest.param(name, marks=CASE_MARKS.get(name, [])) for name in CASES]


# --------------------------------------------------------------------------------------------
# Running a case
# --------------------------------------------------------------------------------------------


@dataclass
class Run:
    """Outcome of one forward pass plus one training step on freshly built modules."""

    forward: dict[str, Any]
    eval_forward: dict[str, Any]
    step: StepResult
    state_schema: dict[str, str]
    params_before: dict[str, th.Tensor]


def _flatten_output(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return dict(output)
    if isinstance(output, (tuple, list)):
        return {str(i): value for i, value in enumerate(output)}
    return {"output": output}


def _run_case(case: Case) -> Run:
    """Build a module, run its forward passes, build a second module and take one step.

    The training-mode forward pass comes first, then an eval-mode one on the same module (so
    BatchNorm statistics are those of one training batch). The step runs on a second module
    built from the same seed: it starts from the weights the forward pass saw, and neither pass
    disturbs the other (BatchNorm statistics, Adam state).
    """
    with F.deterministic():
        module = F.build_module(case.config)
        state_schema = {
            name: f"{tensor.dtype}|{tuple(tensor.shape)}"
            for name, tensor in module.state_dict().items()
        }
        module.train()
        th.manual_seed(1)
        forward = _flatten_output(case.forward(module, copy.deepcopy(case.batch)))
        module.eval()
        with th.no_grad():
            th.manual_seed(1)
            eval_forward = _flatten_output(case.forward(module, copy.deepcopy(case.batch)))
        stepper = F.build_module(case.config)
        params_before = {k: v.detach().clone() for k, v in stepper.named_parameters()}
        step = (_spectrum_step if case.spectrum_pl else _generic_step)(
            stepper, copy.deepcopy(case.batch)
        )
    return Run(forward, eval_forward, step, state_schema, params_before)


def _grad_norm(grads: dict[str, th.Tensor]) -> th.Tensor:
    return th.sqrt(sum((g.double() ** 2).sum() for g in grads.values()))


def _n_params_changed(run: Run) -> int:
    return sum(
        not th.equal(before, run.step.params_after[name])
        for name, before in run.params_before.items()
    )


def _describe(value: Any) -> str:
    return str(value.dtype) if th.is_tensor(value) else type(value).__name__


def _structure_record(run: Run) -> dict[str, Any]:
    """Environment-independent record: output keys and dtypes, the state-dict schema."""
    return {
        "forward_outputs": {key: _describe(value) for key, value in sorted(run.forward.items())},
        "eval_outputs": {key: _describe(value) for key, value in sorted(run.eval_forward.items())},
        "loss_terms": sorted(run.step.terms),
        "state_dict": run.state_schema,
    }


def _value_record(run: Run) -> dict[str, Any]:
    """Bit-exact record: forward outputs, loss terms, and a gradient and update summary."""
    record: dict[str, Any] = {}
    for prefix, outputs in (("forward", run.forward), ("eval", run.eval_forward)):
        for key, value in outputs.items():
            if key.endswith(TIMING_SUFFIX):
                # MASSter reports wall-clock candidate timings in its output; only their
                # presence, dtype and shape are reproducible.
                record[f"{prefix}/{key}"] = f"{value.dtype}|{tuple(value.shape)}"
            else:
                record[f"{prefix}/{key}"] = value if th.is_tensor(value) else _describe(value)
    record.update({f"loss/{k}": v for k, v in run.step.terms.items()})
    record["loss"] = run.step.loss
    record["grad_norm"] = _grad_norm(run.step.grads)
    record["n_grad_tensors"] = len(run.step.grads)
    record["n_params_changed"] = _n_params_changed(run)
    record["params_after"] = F.tensor_digest(
        th.cat([run.step.params_after[k].double().flatten() for k in sorted(run.params_before)])
    )
    return F.digest_record(record)


@contextlib.contextmanager
def _built(case_name: str, root: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Case]:
    case = CASES[case_name](root, monkeypatch)
    try:
        yield case
    finally:
        case.close()


def _assert_sane(case: Case, run: Run) -> None:
    """Finite loss, a flowing gradient, and a forward output that is a well-formed spectrum."""
    assert th.isfinite(run.step.loss).all(), run.step.loss
    assert run.step.loss.ndim == 0
    assert th.isfinite(run.step.terms["loss"]).all()
    assert set(run.eval_forward) == set(run.forward), "eval mode changes the output keys"
    for mode, outputs in (("train", run.forward), ("eval", run.eval_forward)):
        for key, value in outputs.items():
            if th.is_tensor(value) and value.is_floating_point():
                assert not th.isnan(value).any(), f"NaN in {mode} forward output {key!r}"
    grad_norm = float(_grad_norm(run.step.grads))
    if case.trains:
        assert grad_norm > 0.0 and math.isfinite(grad_norm)
        assert any(g.abs().max() > 0 for g in run.step.grads.values())
        assert _n_params_changed(run) > 0
    else:
        assert grad_norm == 0.0
        assert _n_params_changed(run) == 0


def _assert_forward_shapes(case: Case, outputs: dict[str, Any]) -> None:
    """Per-family forward contract: predicted peaks per spectrum, or unit-norm embeddings."""
    batch_size = int(case.batch["batch_size"])
    if case.model_type in ("clip", "spec_mol_clip_pretraining"):
        embed_dim = case.config["clip_embed_dim"]
        for key in ("0", "1"):
            assert outputs[key].shape == (batch_size, embed_dim)
            norms = outputs[key].norm(dim=-1)
            assert th.allclose(norms, th.ones_like(norms), atol=1e-5)
    elif case.model_type == "mces_pretraining" or case.model_type == MCES_PATCHED:
        assert outputs["output"].shape[0] == int(case.batch["batch_size"])
    else:
        mzs, logprobs = outputs["pred_mzs"], outputs["pred_logprobs"]
        idxs = outputs["pred_batch_idxs"]
        assert mzs.shape == logprobs.shape == idxs.shape and mzs.ndim == 1
        assert idxs.dtype == th.int64
        assert set(idxs.tolist()) == set(range(batch_size))
        mass = th.zeros(batch_size, dtype=th.float64).index_add_(0, idxs, logprobs.double().exp())
        assert (mass <= 1.0 + 1e-3).all(), mass
        if "pred_specs" in outputs:
            n_bins = round(case.config["mz_max"] / case.config["mz_bin_res"])
            assert outputs["pred_specs"].shape == (batch_size, n_bins)


@pytest.mark.parametrize("case_name", SMOKE_PARAMS)
def test_one_forward_pass_and_training_step(
    case_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every retained type builds, runs a forward pass and takes one finite, non-trivial step."""
    with _built(case_name, tmp_path, monkeypatch) as case:
        first = _run_case(case)
        second = _run_case(case)
        _assert_sane(case, first)
        _assert_forward_shapes(case, first.forward)
        _assert_forward_shapes(case, first.eval_forward)
        # Two runs in one process must agree to the bit, before any golden comparison.
        assert _value_record(first) == _value_record(second)
    ratchet_json(GOLDEN, f"structure:{case_name}", _structure_record(first))
    ratchet_records(GOLDEN, f"step:{case_name}", _value_record(first))


@pytest.mark.parametrize(
    "pair", (("frag_gnn", "fragnnet"), ("clip", "spec_mol_clip_pretraining")), ids="/".join
)
def test_alias_pairs_give_identical_records(
    pair: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alias model types must stay bit-identical through any registry change (Phase 1)."""
    records = []
    for index, name in enumerate(pair):
        with _built(name, tmp_path / str(index), monkeypatch) as case:
            records.append(_value_record(_run_case(case)))
    assert records[0] == records[1]


def test_mces_pretraining_is_blocked_only_by_the_missing_siamese_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the exact cause of the ``mces_pretraining`` xfail, so no other TypeError hides in it.

    The dataset, dataloader and batch of the shipped configuration build fine; only the module
    construction fails. When ``MCESPL`` is fixed this test fails with "DID NOT RAISE": delete it
    and the xfail mark, and record ``mces_pretraining`` like the other types.
    """
    with _built("mces_pretraining", tmp_path, monkeypatch) as case:
        assert set(case.batch) == {"batch_size", "mces", "mol_1_mol_pyg", "mol_2_mol_pyg"}
        assert case.batch["mces"].shape == (int(case.batch["batch_size"]),)
        missing = r"'use_nce', 'use_ace', 'ace_max', 'ace_mean', and 'ace_std'"
        with pytest.raises(TypeError, match=f"missing 5 required positional arguments: {missing}"):
            F.build_module(case.config)


def test_smoke_cases_cover_every_retained_model_type() -> None:
    """A model type added to the registry must get a smoke case (or be retired explicitly)."""
    assert set(SMOKE_TYPES) | {MCES_PATCHED} == set(CASES)
    assert not set(RETIRED_TYPES) & set(MODEL_TYPES)
    assert not set(RETIRED_TYPES) & set(CASES)


# --------------------------------------------------------------------------------------------
# Item 9: the MASSter spectrum filter through the direct dataset constructor
# --------------------------------------------------------------------------------------------

# Butane has no oxygen to lose, so no [M-H2O+H]+ ion channel exists for it
# (masster_dataset_filter.masster_has_valid_ion); its protonated spectrum is chemically valid.
FILTER_SMILES = ("CCO", "CC(=O)O", "CCCC", "Oc1ccccc1")
LOSS_ADDUCT = "[M-H2O+H]+"
FILTER_SPECTRA = (
    F.Spectrum(0, 0, 0, 20.0, "train", prec_type=LOSS_ADDUCT),
    F.Spectrum(1, 0, 0, 40.0, "train", prec_type=LOSS_ADDUCT),
    F.Spectrum(2, 1, 1, 20.0, "train", prec_type=LOSS_ADDUCT),
    F.Spectrum(3, 1, 1, 40.0, "train", prec_type=LOSS_ADDUCT),
    F.Spectrum(4, 2, 2, 20.0, "train", prec_type=LOSS_ADDUCT),
    F.Spectrum(5, 2, 2, 40.0, "train", prec_type=LOSS_ADDUCT),
    F.Spectrum(6, 2, 3, 30.0, "train", prec_type="[M+H]+"),
    F.Spectrum(7, 3, 4, 30.0, "train", prec_type=LOSS_ADDUCT),
)
ALL_SPEC_IDS = list(range(len(FILTER_SPECTRA)))
FILTERED_SPEC_IDS = [4, 5]  # butane under [M-H2O+H]+
KEPT_SPEC_IDS = [i for i in ALL_SPEC_IDS if i not in FILTERED_SPEC_IDS]
# What the filter must leave behind, written out (the dataset's own counts are checked against it).
KEPT_SPEC_PER_MOL = {0: 2, 1: 2, 2: 1, 3: 1}
KEPT_SPEC_PER_GROUP = {0: 2, 1: 2, 3: 1, 4: 1}

# CO2 has no hydrogen, so it can form [M-H]- only by drawing hydrogen from somewhere: a hydrogen
# transfer from a complement it does not have (the "none" bound allows it; "complement" and
# "valence" do not), or a charge-migration channel (``masster_include_cmf``). Ethanol is valid
# under every setting. Each row is (masster_max_h_transfer, masster_include_cmf,
# masster_h_transfer_bound, whether CO2 keeps its spectra).
HFREE_SMILES = ("CCO", "O=C=O")
NEG_ADDUCT = "[M-H]-"
HFREE_SPECTRA = (
    F.Spectrum(0, 0, 0, 20.0, "train", ((29.0, 100.0), (45.0, 30.0)), prec_type=NEG_ADDUCT),
    F.Spectrum(1, 0, 0, 40.0, "train", ((29.0, 40.0), (45.0, 100.0)), prec_type=NEG_ADDUCT),
    F.Spectrum(2, 1, 1, 20.0, "train", ((28.0, 100.0), (42.0, 30.0)), prec_type=NEG_ADDUCT),
    F.Spectrum(3, 1, 1, 40.0, "train", ((28.0, 40.0), (42.0, 100.0)), prec_type=NEG_ADDUCT),
)
HFREE_VERDICTS = (
    (0, False, "none", False),
    (0, True, "none", True),  # a charge-migration channel supplies the missing hydrogen
    (0, True, "valence", True),
    (1, False, "none", True),  # one hydrogen transferred in from nowhere
    (1, False, "complement", False),  # ... which the complement bound forbids
    (1, False, "valence", False),
    (1, True, "valence", False),
)
HFREE_KEPT, HFREE_DROPPED = [0, 1, 2, 3], [0, 1]


def _spec_ids(dataset: Any) -> list[int]:
    return sorted(int(i) for i in dataset.spec_df["spec_id"])


def _int_map(mapping: dict) -> dict[int, int]:
    return {int(key): int(value) for key, value in sorted(mapping.items())}


def _dataset_state(dataset: Any) -> dict[str, Any]:
    """Everything the filter touches on a dataset: items, frames and the per-id counts."""
    return {
        "n_items": len(dataset),
        "spec_ids": _spec_ids(dataset),
        "mol_smiles": sorted(dataset.mol_df["smiles"]),
        "um_spec_ids": sorted(int(i) for i in dataset.um_spec_df["spec_id"]),
        "split_spec_ids": sorted(int(i) for i in dataset.split_df["spec_id"]),
        "spec_per_mol": _int_map(dataset.spec_per_mol),
        "spec_per_group": _int_map(dataset.spec_per_group),
        "group_per_mol": _int_map(dataset.group_per_mol),
    }


def _assert_frames_consistent(dataset: Any) -> None:
    """After filtering, no frame or count keeps a trace of a dropped spectrum or molecule."""
    spec_ids = set(_spec_ids(dataset))
    assert len(dataset) == len(dataset.spec_df)
    assert {int(i) for i in dataset.um_spec_df["spec_id"]} == spec_ids
    assert {int(i) for i in dataset.split_df["spec_id"]} == spec_ids
    assert set(dataset.mol_df["mol_id"]) == set(dataset.spec_df["mol_id"])
    rows = dataset.spec_df[["spec_id", "mol_id", "group_id"]].itertuples(index=False)
    spec_per_mol: Counter[int] = Counter()
    spec_per_group: Counter[int] = Counter()
    groups_of_mol: dict[int, set[int]] = {}
    for _, mol_id, group_id in rows:
        spec_per_mol[int(mol_id)] += 1
        spec_per_group[int(group_id)] += 1
        groups_of_mol.setdefault(int(mol_id), set()).add(int(group_id))
    assert _int_map(dataset.spec_per_mol) == _int_map(spec_per_mol)
    assert _int_map(dataset.spec_per_group) == _int_map(spec_per_group)
    assert _int_map(dataset.group_per_mol) == _int_map(
        {m: len(g) for m, g in groups_of_mol.items()}
    )


def _direct_dataset(config: dict) -> Any:
    """The dataset class ``get_dataset_cls`` resolves, constructed without ``init_dataset``."""
    return get_dataset_cls("masster")(split="train", spec_pp_sd={}, mol_pp_sd={}, **config)


def _both_paths(config: dict) -> tuple[Any, Any]:
    """The same config through ``runner.init_dataset`` and through the direct constructor."""
    with contextlib.redirect_stdout(io.StringIO()):
        (via_runner,) = runner.init_dataset(config, splits=("train",))
    return via_runner, _direct_dataset(config)


def test_masster_filter_applies_through_every_construction_path(tmp_path: Path) -> None:
    """The filter runs from the constructor, so bypassing ``runner.init_dataset`` keeps it."""
    data = F.write_synth_data(
        tmp_path, FILTER_SPECTRA, FILTER_SMILES, dag_backend=None, meta_info=False
    )
    spec_params = {"prec_types": ["[M+H]+", LOSS_ADDUCT]}
    config = F.make_config("masster", data, {"spec_params": spec_params})
    assert config["model_type"] == "masster"

    via_runner, via_constructor = _both_paths(config)
    states = [_dataset_state(d) for d in (via_runner, via_constructor)]
    assert states[0] == states[1]
    state = states[0]
    assert state["n_items"] == len(KEPT_SPEC_IDS) == 6
    assert state["spec_ids"] == state["um_spec_ids"] == state["split_spec_ids"] == KEPT_SPEC_IDS
    # Butane keeps its protonated spectrum, so the molecule stays; only its pair is dropped.
    assert state["mol_smiles"] == sorted(F.build_mol_df(FILTER_SMILES)["smiles"])
    assert state["spec_per_mol"] == KEPT_SPEC_PER_MOL
    assert state["spec_per_group"] == KEPT_SPEC_PER_GROUP
    for dataset in (via_runner, via_constructor):
        _assert_frames_consistent(dataset)

    # The filtered dataset is what the runner's dataloader yields.
    batch = _first_batch(runner.init_dataloader(via_constructor, config))
    assert sorted(batch["spec_unique_id"].tolist()) == KEPT_SPEC_IDS

    # Without model_type masster nothing is filtered, whichever way it is absent.
    without_type = {k: v for k, v in config.items() if k != "model_type"}
    other_type = {**config, "model_type": "neims"}
    unfiltered = [_dataset_state(_direct_dataset(c)) for c in (without_type, other_type)]
    assert unfiltered[0] == unfiltered[1]
    assert unfiltered[0]["spec_ids"] == unfiltered[0]["um_spec_ids"] == ALL_SPEC_IDS
    assert unfiltered[0]["split_spec_ids"] == ALL_SPEC_IDS
    assert unfiltered[0]["n_items"] == 8
    _assert_frames_consistent(_direct_dataset(without_type))

    ratchet_json(GOLDEN, "masster_filter", {"filtered": state, "unfiltered": unfiltered[0]})


def _hfree_config(tmp_path: Path, max_h: int, include_cmf: bool, bound: str) -> dict:
    data = F.write_synth_data(
        tmp_path, HFREE_SPECTRA, HFREE_SMILES, dag_backend=None, meta_info=False
    )
    overrides = {
        "spec_params": {"prec_types": [NEG_ADDUCT]},
        "masster_max_h_transfer": max_h,
        "masster_include_cmf": include_cmf,
        "masster_h_transfer_bound": bound,
    }
    return F.make_config("masster", data, overrides)


def test_masster_filter_honors_its_chemistry_settings(tmp_path: Path) -> None:
    """``masster_max_h_transfer``, ``masster_include_cmf`` and the bound reach the filter."""
    co2 = F.build_mol_df(HFREE_SMILES).loc[1, "smiles"]
    record: dict[str, Any] = {}
    for index, (max_h, include_cmf, bound, kept) in enumerate(HFREE_VERDICTS):
        config = _hfree_config(tmp_path / str(index), max_h, include_cmf, bound)
        via_runner, via_constructor = _both_paths(config)
        states = [_dataset_state(d) for d in (via_runner, via_constructor)]
        assert states[0] == states[1], (max_h, include_cmf, bound)
        spec_ids = states[0]["spec_ids"]
        assert spec_ids == (HFREE_KEPT if kept else HFREE_DROPPED), (max_h, include_cmf, bound)
        # A molecule whose every spectrum is dropped leaves ``mol_df`` too.
        assert (co2 in states[0]["mol_smiles"]) == kept
        for dataset in (via_runner, via_constructor):
            _assert_frames_consistent(dataset)
        record[f"max_h={max_h},cmf={include_cmf},bound={bound}"] = states[0]
    ratchet_json(GOLDEN, "masster_filter_settings", record)


def test_masster_filter_follows_the_legacy_mass_shatter_spelling(tmp_path: Path) -> None:
    """A pre-rename config still reaches the filter, with its ``mass_shatter_*`` settings.

    The constructor filters only for ``model_type == "masster"``, so the legacy name must be
    migrated before a dataset is built, and the registry must not resolve it unmigrated. The
    legacy settings below keep the hydrogen-free molecule that the template's own settings
    (bound ``valence``) would drop, which shows they arrive at the filter.
    """
    data = F.write_synth_data(
        tmp_path, HFREE_SPECTRA, HFREE_SMILES, dag_backend=None, meta_info=False
    )
    legacy = {
        "model_type": "mass_shatter",
        "mass_shatter_max_h_transfer": 1,
        "mass_shatter_include_cmf": False,
        "mass_shatter_h_transfer_bound": "none",
        "spec_fp": data.spec_fp,
        "mol_fp": data.mol_fp,
        "split_dp": data.split_dp,
        "spec_params": {
            "merge": False,
            "merge_train_only": False,
            "nce": True,
            "prec_types": [NEG_ADDUCT],
        },
    }
    custom = tmp_path / "legacy.yml"
    custom.write_text(yaml.safe_dump(legacy))
    with contextlib.redirect_stdout(io.StringIO()):
        config = runner.load_config(F.CONFIG_DIR, str(custom), None)
    assert config["model_type"] == "masster"
    assert (config["masster_max_h_transfer"], config["masster_include_cmf"]) == (1, False)
    assert config["masster_h_transfer_bound"] == "none"

    assert _spec_ids(_direct_dataset(config)) == HFREE_KEPT
    template_settings = {**config, "masster_h_transfer_bound": "valence"}
    assert _spec_ids(_direct_dataset(template_settings)) == HFREE_DROPPED
    with pytest.raises(ValueError):
        get_dataset_cls("mass_shatter")


def test_masster_filter_keeps_every_valid_default_spectrum(tmp_path: Path) -> None:
    """Control: the default protonated spectra are all valid, so the filter must drop none."""
    data = F.write_synth_data(tmp_path, dag_backend=None)
    config = F.make_config("masster", data)
    via_runner, via_constructor = _both_paths(config)
    assert len(via_runner) == len(via_constructor) == 7
    assert _spec_ids(via_runner) == [0, 1, 2, 3, 4, 5, 6]
    assert _dataset_state(via_runner) == _dataset_state(via_constructor)
    _assert_frames_consistent(via_runner)
