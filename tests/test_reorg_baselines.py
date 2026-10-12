"""Baseline records for MassFormer and GRAFF (plan Phase 0 item 3).

Captured before any Phase 1 change, so the registry, dataset and loss moves can be shown to
leave both retained baselines unchanged. For each model the tests record, on a small synthetic
dataset built in code (no ``data/``, no network):

- ``items``: the per-molecule ``mol_mf`` / ``mol_graff`` preprocessing output of the dataset
  (GRAFF also records the per-spectrum ``ann_*`` output and the fitted annotation library);
- ``edge_items`` and ``edge_collation``: preprocessing, and MassFormer collation, of molecules
  the fixture dataset cannot hold (a bond-free atom, a disconnected pair, a chain deeper than the
  collator's 5 hops), called the way the dataset calls it;
- ``batches``: every collated batch of one epoch of the runner's ``init_dataloader`` (val split
  through the ``SequentialSampler``; train split with its sampler generator reseeded explicitly),
  including the ragged final batch and batches that mix molecule sizes, with every key, shape,
  dtype and value digested;
- ``forward_step``: forward outputs in eval and (seeded) train mode, every loss term, the
  gradients and the parameters after exactly one optimizer step.

Only behavior is recorded (keys, shapes, dtypes, values). Dataset and model class names are not,
because Phase 1 maps ``massformer`` to ``SpecMolMFDataset`` by design.

MassFormer builds Graphormer-base and downloads pretrained weights; here the download is replaced
by a random state dict of the same architecture and the encoder is shrunk (1 layer, 32 dim,
2 heads). GRAFF eigenvectors have an arbitrary sign, so its records keep eigenvalues and the
sign-invariant Gram matrix ``V @ V.T`` rather than raw eigenvectors; the fixture uses only
molecules with distinct Laplacian eigenvalues, so each eigenvector is unique up to sign.

Regenerate deliberately (never to make a failing refactor pass):

    REORG_UPDATE_GOLDEN=1 pytest tests/test_reorg_baselines.py
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import reorg_fixtures as F
import torch as th
from pyteomics.mass import Composition
from rdkit import Chem
from reorg_golden import ratchet_records
from torch.utils.data import RandomSampler, SequentialSampler

import fragnnet.baselines.massformer.data_utils as mf_data_utils
from fragnnet.baselines.graff.data_utils import fit_ann_library, graff_preprocess
from fragnnet.utils.data_utils import composition_to_string, formula_to_mass
from fragnnet.utils.formula_utils import NEUTRON_MASS

GOLDEN = "baselines"
MODEL_TYPES = ("massformer", "graff")
MOL_KEY = {"massformer": "mol_mf", "graff": "mol_graff"}
ELECTRON_MASS = 0.00054858
# Keys of a ``mol_mf`` item; the collate function flattens all but the edge lists into
# ``mol_mf_<key>`` batch tensors.
MF_ITEM_KEYS = {
    "x",
    "edge_index",
    "edge_attr",
    "y",
    "idx",
    "attn_bias",
    "attn_edge_type",
    "spatial_pos",
    "in_degree",
    "out_degree",
    "edge_input",
}

# Heavy-atom graphs with no repeated Laplacian eigenvalue in the first NUM_EIGS + 1 nonzero slots
# (acetic acid CC(=O)O, 2-propanol and alanine have a doubly degenerate eigenvalue 1.0, which
# makes GRAFF's eigenvectors an arbitrary basis). Methyl lactate carries a chiral tag.
SMILES = ("CCO", "C=CC(=O)O", "Oc1ccccc1", "C[C@@H](O)C(=O)OC")
# One row per spectrum, spec_id = position: (mol_id, inst_type, nce, split). group_id = mol_id.
# The val split alternates phenol (7 heavy atoms) and ethanol (3), so a val batch pads.
LAYOUT = (
    (0, "ORB", 20.0, "train"),
    (0, "ORB", 40.0, "train"),
    (1, "ORB_F", 20.0, "train"),
    (1, "ORB_F", 40.0, "train"),
    (3, "ORB", 25.0, "train"),
    (3, "ORB", 45.0, "train"),
    (2, "ORB", 20.0, "val"),
    (0, "ORB", 30.0, "val"),
    (2, "ORB", 40.0, "val"),
)
TRAIN_UIDS, VAL_UIDS = (0, 1, 2, 3, 4, 5), (6, 7, 8)
EDGE_SMILES = {"massformer": ("C", "C.C"), "graff": ("C",)}
NO_ANN_SPEC_ID = 3  # a spectrum with no ann_df row (exercises GRAFF's right-merge and fillna)
ISOTOPE_SPEC_ID = 0  # carries an M+1 isotope peak, so ann_isotope_mask is non-trivial
LIBRARY_SIZE = 8
NUM_EIGS = 4
MAX_ISOTOPE = 2
TRAIN_ORDER_SEED = 7  # reseeds the train RandomSampler generator
FORWARD_SEED = 1  # seeds the stochastic train-mode forward (Graphormer dropout is hard-coded)

BATCH_OVERRIDES: dict[str, Any] = {
    "train_batch_size": 4,  # 6 train spectra: batches of 4 and 2
    "eval_batch_size": 2,  # 3 val spectra: batches of 2 (mixed sizes) and 1
    "lr": 1e-3,
    "lr_schedule": False,
}
MASSFORMER_OVERRIDES: dict[str, Any] = {
    **BATCH_OVERRIDES,
    "mol_params": {"pyg": False, "mf": True, "preprocess": False},
    "loss_type": "cosine_distance",  # BinnedPL asserts this
    "mf_lr_schedule": False,  # the template schedule makes the step-0 learning rate 5e-6
    "mlp_hidden_size": 16,
    "mlp_num_layers": 1,
    "mlp_dropout": 0.0,
    "ff_output_map_size": 8,
    "ff_prec_mz_offset": 50,
    "mz_max": 200.0,
    "mz_bin_res": 0.1,
    "ce_insert_type": "embed",
    "ce_insert_location": "mlp",
    "ce_insert_merge": True,
}


@dataclass(frozen=True)
class Synth:
    """The synthetic dataset: shared files, the GRAFF annotation table and its path."""

    data: F.SynthData
    ann_df: pd.DataFrame
    ann_fp: str


@dataclass
class Built:
    """Resolved config plus the runner's train and val datasets and dataloaders."""

    model_type: str
    config: dict
    train_ds: Any
    train_dl: Any
    val_ds: Any
    val_dl: Any


# ------------------------------------------------------------------------------ fixture data
def _precursor_products(formula: str) -> list[str]:
    """[M+H]+ formula followed by its H2O and CH4 loss products (when chemically valid)."""
    precursor = Composition(formula=formula)
    precursor["H"] += 1
    products = [composition_to_string(precursor)]
    for loss in ("H2O", "CH4"):
        product = precursor - Composition(formula=loss)
        if all(count >= 0 for count in product.values()):
            products.append(composition_to_string(product))
    return products


def _build_spectra(mol_df: pd.DataFrame) -> tuple[tuple[F.Spectrum, ...], pd.DataFrame]:
    """Spectra whose peaks are formula-annotated, and the matching ``ann_df``.

    Intensities differ per spectrum and per peak: ``fit_ann_library`` ranks entries by summed
    intensity with an unstable sort, so ties would make the library order build-dependent.
    """
    spectra, ann_rows = [], []
    for spec_id, (mol_id, inst_type, nce, split) in enumerate(LAYOUT):
        products = _precursor_products(mol_df.loc[mol_id, "formula"])
        ints = [100.0 - 10.0 * spec_id, 50.0 - spec_id, 25.0 - 0.3 * spec_id]
        entries = [(product, 0, ints[k]) for k, product in enumerate(products)]
        if spec_id == ISOTOPE_SPEC_ID:
            entries.append((products[0], 1, 5.0))
        rows = []
        for product, isotope, intensity in entries:
            exact = formula_to_mass(product) + isotope * NEUTRON_MASS
            loss = Composition(formula=products[0]) - Composition(formula=product)
            mz = round(exact - ELECTRON_MASS, 4)
            rows.append((mz, intensity, product, composition_to_string(loss), isotope, exact))
        rows.sort(key=lambda row: row[0])
        spectra.append(
            F.Spectrum(
                spec_id,
                mol_id,
                mol_id,
                nce,
                split,
                tuple((mz, intensity) for mz, intensity, *_ in rows),
                inst_type=inst_type,
            )
        )
        if spec_id != NO_ANN_SPEC_ID:
            ann_rows.append(
                {
                    "spec_id": spec_id,
                    "ann_peak_mzs": [f"{row[0]:.4f}" for row in rows],
                    "ann_products": [row[2] for row in rows],
                    "ann_losses": [row[3] for row in rows],  # '' for the precursor
                    "ann_isotopes": [row[4] for row in rows],
                    "ann_exact_mzs": [row[5] for row in rows],
                }
            )
    return tuple(spectra), pd.DataFrame(ann_rows)


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> Synth:
    """Write the shared synthetic dataset (spec_df, mol_df, splits and the GRAFF ann_df) once."""
    root = Path(tmp_path_factory.mktemp("baselines"))
    spectra, ann_df = _build_spectra(F.build_mol_df(SMILES))
    data = F.write_synth_data(root, spectra, SMILES, dag_backend=None)
    spec_df = data.spec_df.assign(
        dset="synth", dset_spec_id=[f"synth_{i}" for i in data.spec_df["spec_id"]]
    )
    spec_df.to_pickle(data.spec_fp)  # GRAFF's fit_ann_library reads dset and dset_spec_id
    ann_fp = str(root / "ann_df.pkl")
    ann_df.to_pickle(ann_fp)
    return Synth(data, ann_df, ann_fp)


@pytest.fixture(autouse=True)
def offline_tiny_graphormer(monkeypatch: pytest.MonkeyPatch) -> None:
    """MassFormer without network: random "pretrained" weights and a 1-layer, 32-dim encoder.

    ``MassFormerModel`` always builds Graphormer-base and loads a pretrained state dict from a
    zenodo URL (nn_utils.py ``load_pretrained_model``). The stub returns the state dict of an
    equally shaped random embedder, and the shrink keeps the model at about 250k parameters.
    """
    import fragnnet.baselines.massformer.nn_utils as mf_nn

    base_args = mf_nn.set_graphormer_base_architecture_args

    def tiny_args(args: Any) -> None:
        base_args(args)
        args.encoder_layers, args.encoder_embed_dim = 1, 32
        args.encoder_ffn_embed_dim, args.encoder_attention_heads = 32, 2

    def random_pretrained(name: str) -> dict[str, th.Tensor]:
        return mf_nn.GFv2Embedder(gf_pretrain_name="none").state_dict()

    monkeypatch.setattr(mf_nn, "set_graphormer_base_architecture_args", tiny_args)
    monkeypatch.setattr(mf_nn, "load_pretrained_model", random_pretrained)


def _overrides(model_type: str, synth: Synth) -> dict[str, Any]:
    if model_type == "massformer":
        return MASSFORMER_OVERRIDES
    return {
        **BATCH_OVERRIDES,
        "ann_fp": synth.ann_fp,
        "mol_params": {"pyg": False, "graff": True, "preprocess": False},
        "ann_params": {
            "library_size": LIBRARY_SIZE,
            "num_eigs": NUM_EIGS,
            "max_isotope": MAX_ISOTOPE,
        },
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


def _build(model_type: str, synth: Synth) -> Built:
    """Config, datasets and dataloaders through the runner's own entry points."""
    config = F.make_config(model_type, synth.data, _overrides(model_type, synth))
    train_ds, train_dl = F.init_split(config, "train")
    val_ds, val_dl = F.init_split(config, "val")
    return Built(model_type, config, train_ds, train_dl, val_ds, val_dl)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to ``a/b/c`` keys, so a golden diff names the exact tensor."""
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, item in value.items():
        flat.update(_flatten(item, f"{prefix}/{key}" if prefix else str(key)))
    return flat


def _ratchet(key: str, record: dict) -> None:
    """Compare ``record`` with the stored one (exactly, in the recording environment)."""
    ratchet_records(GOLDEN, key, _flatten(record))


def _stable(make: Callable[[], dict]) -> dict:
    """Run ``make`` twice from scratch and require identical records."""
    first, second = make(), make()
    assert first == second, "record differs between two fresh builds in one process"
    return first


# ------------------------------------------------------------------------------ record helpers
def _clean_zeros(array: np.ndarray) -> np.ndarray:
    """Float64 copy with ``-0.0`` turned into ``0.0`` (the sign of a zero is not a value)."""
    return np.asarray(array, dtype=np.float64) + 0.0


def _eigen_invariants(graph: Any) -> dict[str, Any]:
    """Sign-invariant GRAFF eigen-features of a graph or batch: eigenvalues and ``V @ V.T``.

    Args:
        graph: A PyG ``Data`` or ``Batch`` with ``eigvals`` (graphs, k), ``eigvecs`` (nodes, k)
            and, for a batch, ``ptr`` (graphs + 1). Eigenvectors are unique up to sign only, so
            the raw ``eigvecs`` are not recorded.

    Returns:
        ``eigvals`` and, per graph, the Gram matrix ``eigvec_gram`` of its eigenvector columns
        (graph nodes including the virtual node, squared), both at full precision. Flipping a
        column's sign leaves every product in the Gram sum bit-identical, so full precision
        costs no sign stability. ``eigvec_gram_r4`` is the same Gram rounded to 4 decimals: it
        ignores the 2e-7 noise of ``scipy.linalg.eigh`` and tells a real change (it moves too)
        from a build-level one (only the exact record moves) when a comparison fails.
    """
    eigvecs = graph.eigvecs.detach().cpu().double().numpy()
    ptr = graph.ptr.tolist() if "ptr" in graph else [0, eigvecs.shape[0]]
    grams = [_clean_zeros(eigvecs[lo:hi] @ eigvecs[lo:hi].T) for lo, hi in zip(ptr[:-1], ptr[1:])]
    return {
        "eigvals": _clean_zeros(graph.eigvals.detach().cpu().numpy()),
        "eigvec_gram": grams,
        "eigvec_gram_r4": [_clean_zeros(np.round(gram, 4)) for gram in grams],
    }


def _graph_content(graph: Any, mol_key: str) -> dict[str, Any]:
    """The contents of a ``mol_mf`` / ``mol_graff`` object, as a dict ready for digesting."""
    content = {key: graph[key] for key in sorted(graph.keys())}
    if mol_key == "mol_graff":
        del content["eigvecs"], content["eigvals"]
        content.update(_eigen_invariants(graph))
    return content


def _batch_content(batch: dict, mol_key: str) -> dict[str, Any]:
    content = dict(batch)
    if mol_key == "mol_graff":
        content["mol_graff"] = _graph_content(batch["mol_graff"], "mol_graff")
    return content


def _all_items(built: Built) -> list[dict]:
    """Every dataset item, train split then val split, through ``Dataset.__getitem__``."""
    return [ds[i] for ds in (built.train_ds, built.val_ds) for i in range(len(ds))]


def _epochs(built: Built) -> tuple[list[dict], list[dict]]:
    """One full epoch of the val loader and of the train loader.

    The val loader uses a ``SequentialSampler``. The train loader's ``RandomSampler`` generator
    is reseeded explicitly, as ``SpectrumPL._seed_dataloader`` does each epoch; unseeded, every
    fresh loader would replay torch's default generator.

    Returns:
        The val batches and the train batches, in loader order.
    """
    assert isinstance(built.val_dl.sampler, SequentialSampler)
    assert isinstance(built.train_dl.sampler, RandomSampler)
    val_epoch = list(built.val_dl)
    built.train_dl.sampler.generator.manual_seed(TRAIN_ORDER_SEED)
    return val_epoch, list(built.train_dl)


def _tensors(output: dict) -> dict[str, th.Tensor]:
    return {key: value for key, value in output.items() if th.is_tensor(value)}


# ------------------------------------------------------------------------------ item checks
def _molecule_items(built: Built) -> dict[int, list[dict]]:
    by_mol: dict[int, list[dict]] = {}
    for item in _all_items(built):
        by_mol.setdefault(int(item["mol_id"]), []).append(item)
    return by_mol


def _check_massformer_graph(graph: Any, mol: Chem.Mol) -> None:
    n_atoms, n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()
    dist = th.as_tensor(Chem.GetDistanceMatrix(mol), dtype=th.long)
    degrees = th.tensor([atom.GetDegree() for atom in mol.GetAtoms()])
    assert set(graph.keys()) == MF_ITEM_KEYS
    assert graph.x.shape == (n_atoms, 9) and graph.x.dtype == th.long
    assert graph.edge_index.shape == (2, 2 * n_bonds) and graph.edge_index.dtype == th.long
    assert graph.edge_attr.shape == (2 * n_bonds, 3)
    assert th.equal(graph.spatial_pos, dist)
    assert th.equal(graph.in_degree, degrees) and th.equal(graph.out_degree, degrees)
    assert graph.attn_bias.shape == (n_atoms + 1, n_atoms + 1)
    assert graph.attn_bias.dtype == th.float32 and not graph.attn_bias.any()
    assert graph.attn_edge_type.shape == (n_atoms, n_atoms, 3)
    assert graph.edge_input.shape == (n_atoms, n_atoms, int(dist.max()), 3)  # not truncated here
    assert graph.y.shape == (1,) and graph.y.dtype == th.float32 and graph.idx == -1


def _check_graff_graph(graph: Any, mol: Chem.Mol) -> None:
    n_atoms, n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()
    n_nodes, n_edges = n_atoms + 1, 2 * n_bonds + 2 * n_atoms  # plus the virtual node
    assert set(graph.keys()) == {"x", "edge_index", "edge_attr", "smiles", "eigvecs", "eigvals"}
    assert graph.smiles == Chem.MolToSmiles(mol)
    assert graph.x.shape == (n_nodes, 10) and graph.x.dtype == th.long
    assert graph.x[:, -1].tolist() == [0] * n_atoms + [1]  # virtual-node flag
    assert graph.edge_index.shape == (2, n_edges)
    assert graph.edge_attr.shape == (n_edges, 5) and graph.edge_attr.dtype == th.long
    virtual_in = graph.edge_attr[:, 3].nonzero().flatten()
    virtual_out = graph.edge_attr[:, 4].nonzero().flatten()
    assert virtual_in.tolist() == list(range(2 * n_bonds, 2 * n_bonds + n_atoms))
    assert virtual_out.tolist() == list(range(2 * n_bonds + n_atoms, n_edges))
    assert (graph.edge_index[1, virtual_in] == n_atoms).all()
    assert (graph.edge_index[0, virtual_out] == n_atoms).all()
    assert graph.eigvals.shape == (1, NUM_EIGS) and graph.eigvals.dtype == th.float32
    assert graph.eigvecs.shape == (n_nodes, NUM_EIGS)
    assert not graph.eigvecs[n_atoms].any()  # the virtual node has no eigenvector entries


def _check_ann_item(item: dict, spec_id: int) -> None:
    """GRAFF annotation output of one spectrum."""
    n_ann = LIBRARY_SIZE * (MAX_ISOTOPE + 1)
    assert item["ann_formula_idxs"].shape == (n_ann,)
    assert item["ann_library_mzs"].shape == item["ann_isotope_idxs"].shape
    assert 0 < item["ann_library_mzs"].numel() <= n_ann
    assert item["ann_formula_idxs"].max() < item["ann_library_mzs"].numel()
    assert item["ann_isotope_mask"].tolist() == [float(spec_id == ISOTOPE_SPEC_ID)]


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_item_records(model_type: str, synth: Synth) -> None:
    """Per-molecule ``mol_mf`` / ``mol_graff`` output (and GRAFF's per-spectrum ann output)."""
    mol_key = MOL_KEY[model_type]

    def make() -> dict:
        with F.deterministic():
            built = _build(model_type, synth)
            record: dict[str, Any] = {}
            for mol_id, items in sorted(_molecule_items(built).items()):
                mol = Chem.MolFromSmiles(SMILES[mol_id])
                graphs = [_graph_content(item[mol_key], mol_key) for item in items]
                assert len(items) == sum(row[0] == mol_id for row in LAYOUT) >= 2
                digests = [F.digest_record(graph) for graph in graphs]
                assert all(digest == digests[0] for digest in digests)  # one molecule, one output
                check = (
                    _check_massformer_graph if model_type == "massformer" else _check_graff_graph
                )
                check(items[0][mol_key], mol)
                record[f"mol{mol_id}"] = digests[0]
            if model_type == "graff":
                for item in _all_items(built):
                    spec_id = int(item["spec_unique_id"])
                    _check_ann_item(item, spec_id)
                    ann = {k: v for k, v in item.items() if k.startswith("ann_")}
                    record[f"ann_spec{spec_id}"] = F.digest_record(ann)
            return record

    record = _stable(make)
    assert set(record) >= {f"mol{i}" for i in range(len(SMILES))}
    _ratchet(f"{model_type}:items", record)


def _edge_graph(model_type: str, smiles: str) -> Any:
    """Preprocess one molecule exactly as ``SpecMolDataset._process_mol`` / GRAFF's dataset do."""
    mol = Chem.MolFromSmiles(smiles)
    if model_type == "massformer":
        return mf_data_utils.gf_preprocess(mol, -1)
    return graff_preprocess(mol, NUM_EIGS)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_edge_item_records(model_type: str) -> None:
    """Preprocessing of molecules the dataset fixture cannot hold, recorded as behavior.

    A bond-free atom (``C``) has empty edge lists, and for MassFormer an empty ``edge_input``
    path axis. A disconnected pair (``C.C``) gets Floyd-Warshall's "no path" distance 510 in
    ``spatial_pos`` and a 510-deep ``edge_input``. GRAFF records only ``C``: a disconnected graph
    has a degenerate zero Laplacian eigenvalue, so its eigenvectors are an arbitrary basis.
    """
    mol_key = MOL_KEY[model_type]
    smiles_list = EDGE_SMILES[model_type]

    def make() -> dict:
        with F.deterministic():
            return {
                smiles: F.digest_record(_graph_content(_edge_graph(model_type, smiles), mol_key))
                for smiles in smiles_list
            }

    record = _stable(make)
    assert set(record) == set(smiles_list)
    with F.deterministic():
        atom = _edge_graph(model_type, "C")
        if model_type == "massformer":
            assert atom.edge_index.shape == (2, 0) and atom.edge_attr.shape == (0, 3)
            assert atom.spatial_pos.tolist() == [[0]] and atom.in_degree.tolist() == [0]
            assert atom.edge_input.shape == (1, 1, 0, 3)
            pair = _edge_graph(model_type, "C.C")
            assert pair.spatial_pos.tolist() == [[0, 510], [510, 0]]
            assert pair.edge_input.shape == (2, 2, 510, 3) and pair.edge_index.shape == (2, 0)
        else:
            assert atom.x.shape == (2, 10) and atom.edge_index.shape == (2, 2)
            assert not atom.eigvals.any() and not atom.eigvecs.any()  # nothing beyond the k=0 mode
    _ratchet(f"{model_type}:edge_items", record)


def test_massformer_edge_collation_record() -> None:
    """Collating a long chain, a small molecule and a disconnected pair into one batch.

    Heptane has graph diameter 6; the collator keeps only ``multi_hop_max_dist`` = 5 hops of
    ``edge_input``, a cut no fixture molecule reaches. The disconnected pair's "no path"
    distance 510 stays below ``spatial_pos_max`` = 1024, so it is not masked with ``-inf``.
    """
    chain, small, pair = "CCCCCCC", "CCO", "C.C"

    def collate() -> tuple[list[Any], dict[str, th.Tensor]]:
        with F.deterministic():
            graphs = [_edge_graph("massformer", smiles) for smiles in (chain, small, pair)]
            return graphs, mf_data_utils.collator(graphs)

    def make() -> dict:
        return F.digest_record(collate()[1])

    record = _stable(make)
    graphs, out = collate()
    assert graphs[0].edge_input.shape == (7, 7, 6, 3)  # the item keeps all 6 hops
    assert out["edge_input"].shape == (3, 7, 7, 5, 3)  # the collator cuts to 5
    assert out["x"].shape == (3, 7, 9) and out["attn_bias"].shape == (3, 8, 8)
    assert out["spatial_pos"][2, :2, :2].tolist() == [[1, 511], [511, 1]]  # 510 + pad shift
    assert out["spatial_pos"][0, 0, 6].item() == 7  # chain end to end: 6 hops + pad shift
    assert th.isneginf(out["attn_bias"][2, :, 3:]).all() and not th.isinf(out["attn_bias"][0]).any()
    _ratchet("massformer:edge_collation", record)


# ------------------------------------------------------------------------------ batch checks
def _check_batch_molecules(batch: dict, mol_key: str, items: list[dict]) -> None:
    """The collated molecule tensors agree with the dataset items they were built from.

    The MassFormer collator shifts ``x``, ``spatial_pos``, ``in_degree`` and ``edge_input`` by +1
    (pad id 0), pads to the largest molecule of the batch and fills padded ``attn_bias`` columns
    with ``-inf``.
    """
    mols = [Chem.MolFromSmiles(SMILES[int(item["mol_id"])]) for item in items]
    sizes = [mol.GetNumAtoms() for mol in mols]
    size, widest = len(items), max(sizes)
    if mol_key == "mol_mf":
        assert batch["mol_mf_x"].shape == (size, widest, 9)
        assert batch["mol_mf_attn_bias"].shape == (size, widest + 1, widest + 1)
        assert batch["mol_mf_attn_edge_type"].shape == (size, widest, widest, 3)
        assert batch["mol_mf_spatial_pos"].shape == (size, widest, widest)
        assert batch["mol_mf_in_degree"].shape == (size, widest)
        assert batch["mol_mf_edge_input"].shape[:3] == (size, widest, widest)
        assert batch["mol_mf_edge_input"].shape[-1] == 3
        assert batch["mol_mf_y"].shape == (size,) and batch["mol_mf_idx"].shape == (size,)
        assert th.equal(batch["mol_mf_out_degree"], batch["mol_mf_in_degree"])
        for b, (item, mol) in enumerate(zip(items, mols)):
            graph, n = item["mol_mf"], mol.GetNumAtoms()
            dist = th.as_tensor(Chem.GetDistanceMatrix(mol), dtype=th.long)
            depth = graph.edge_input.shape[2]
            assert th.equal(batch["mol_mf_x"][b, :n], graph.x + 1)
            assert not batch["mol_mf_x"][b, n:].any()
            assert th.equal(batch["mol_mf_spatial_pos"][b, :n, :n], dist + 1)
            assert not batch["mol_mf_spatial_pos"][b, n:].any()
            assert th.equal(batch["mol_mf_in_degree"][b, :n], graph.in_degree + 1)
            assert th.equal(batch["mol_mf_edge_input"][b, :n, :n, :depth], graph.edge_input + 1)
            assert th.equal(batch["mol_mf_attn_edge_type"][b, :n, :n], graph.attn_edge_type)
            assert th.equal(batch["mol_mf_attn_bias"][b, : n + 1, : n + 1], graph.attn_bias)
            for key, keep in (
                ("mol_mf_edge_input", (slice(0, n), slice(0, n), slice(0, depth))),
                ("mol_mf_attn_edge_type", (slice(0, n), slice(0, n))),
            ):
                padding = batch[key][b].clone()
                padding[keep] = 0
                assert not padding.any(), key  # everything outside the molecule is pad id 0
            assert th.isneginf(batch["mol_mf_attn_bias"][b, :, n + 1 :]).all()
            assert not batch["mol_mf_attn_bias"][b, n + 1 :, : n + 1].any()
        return
    graph = batch["mol_graff"]
    assert graph.ptr.tolist() == np.cumsum([0] + [n + 1 for n in sizes]).tolist()
    assert graph.smiles == [Chem.MolToSmiles(mol) for mol in mols]
    assert graph.x.shape == (sum(sizes) + size, 10)
    assert graph.eigvals.shape == (size, NUM_EIGS)
    assert graph.eigvecs.shape == (sum(sizes) + size, NUM_EIGS)
    assert graph.batch.tolist() == np.repeat(range(size), [n + 1 for n in sizes]).tolist()
    for b, item in enumerate(items):
        lo, hi = graph.ptr[b], graph.ptr[b + 1]
        assert th.equal(graph.x[lo:hi], item["mol_graff"].x)
        assert th.equal(graph.eigvecs[lo:hi], item["mol_graff"].eigvecs)
        assert th.equal(graph.eigvals[b : b + 1], item["mol_graff"].eigvals)


def _atom_counts(batch: dict) -> list[int]:
    """Heavy-atom count of each spectrum's molecule in a batch."""
    uids = batch["spec_unique_id"].tolist()
    return [Chem.MolFromSmiles(SMILES[LAYOUT[uid][0]]).GetNumAtoms() for uid in uids]


def _check_batch_spectra(batch: dict, mol_key: str, items: dict[int, dict]) -> None:
    """Spectrum fields, annotation fields and molecule tensors of one collated batch.

    Args:
        batch: One batch from the runner's dataloader.
        mol_key: ``mol_mf`` or ``mol_graff``.
        items: Every dataset item by ``spec_unique_id``.
    """
    uids = batch["spec_unique_id"].tolist()
    size = len(uids)
    assert int(batch["batch_size"]) == size
    mol_ids = [LAYOUT[uid][0] for uid in uids]
    assert [int(m) for m in batch["mol_id"]] == mol_ids
    assert batch["mol_smiles"] == [Chem.MolToSmiles(Chem.MolFromSmiles(SMILES[m])) for m in mol_ids]
    assert batch["spec_nce"].tolist() == [LAYOUT[uid][2] for uid in uids]
    assert batch["group_id"].tolist() == mol_ids
    assert batch["spec_mzs"].shape == batch["spec_ints"].shape == batch["spec_batch_idxs"].shape
    assert batch["spec_batch_idxs"].tolist() == sorted(batch["spec_batch_idxs"].tolist())
    _check_batch_molecules(batch, mol_key, [items[uid] for uid in uids])
    if mol_key == "mol_mf":
        collated = {f"mol_mf_{key}" for key in MF_ITEM_KEYS - {"edge_index", "edge_attr"}}
        assert {key for key in batch if key.startswith("mol_mf_")} == collated
        assert "mol_mf" not in batch  # flattened into mol_mf_* tensors by the collate function
        return
    n_ann = size * LIBRARY_SIZE * (MAX_ISOTOPE + 1)
    assert batch["ann_formula_idxs"].numel() == n_ann
    assert batch["ann_formula_idxs"].max() < batch["ann_library_mzs"].numel()
    assert batch["ann_library_mzs"].shape == batch["ann_isotope_idxs"].shape
    assert batch["ann_library_mzs"].shape == batch["ann_batch_idxs"].shape
    assert batch["ann_isotope_mask"].tolist() == [float(u == ISOTOPE_SPEC_ID) for u in uids]


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_batch_records(model_type: str, synth: Synth) -> None:
    """Every collated batch of one val and one train epoch through ``init_dataloader``.

    The val epoch is a mixed-size batch (phenol, ethanol) and a single-spectrum final batch; the
    train epoch is a batch of four (sizes 5, 3, 3, 7) and a ragged final batch of two (5, 7). So
    the collators pad, and handle a short last batch, in both splits.
    """
    mol_key = MOL_KEY[model_type]

    def make() -> dict:
        with F.deterministic():
            built = _build(model_type, synth)
            items = {int(item["spec_unique_id"]): item for item in _all_items(built)}
            val_epoch, train_epoch = _epochs(built)
            assert len(built.val_ds) == len(VAL_UIDS) and len(built.train_ds) == len(TRAIN_UIDS)
            val_order = [batch["spec_unique_id"].tolist() for batch in val_epoch]
            train_order = [batch["spec_unique_id"].tolist() for batch in train_epoch]
            assert val_order == [[6, 7], [8]]  # sequential: dataset order, short final batch
            assert sorted(sum(train_order, [])) == list(TRAIN_UIDS)
            assert [len(order) for order in train_order] == [4, 2]
            record: dict[str, Any] = {
                "keys": sorted(val_epoch[0]),
                "val_epoch_order": val_order,
                "train_epoch_order": train_order,
            }
            for split, epoch in (("val", val_epoch), ("train", train_epoch)):
                for i, batch in enumerate(epoch):
                    assert sorted(batch) == record["keys"]
                    _check_batch_spectra(batch, mol_key, items)
                    record[f"{split}_batch{i}"] = F.digest_record(_batch_content(batch, mol_key))
            for batch in (val_epoch[0], *train_epoch):
                assert len(set(_atom_counts(batch))) > 1  # the batch really pads
            assert len(_atom_counts(val_epoch[1])) == 1
            return record

    record = _stable(make)
    _ratchet(f"{model_type}:batches", record)


# ------------------------------------------------------------------------------ forward / step
def _forward_step_record(built: Built) -> dict[str, Any]:
    """Forward outputs, loss terms, gradients and parameters after one optimizer step.

    Eval-mode forwards cover every batch of both epochs; the seeded train-mode forward and the
    optimizer step use the first train batch (four spectra of mixed molecule sizes).
    """
    val_epoch, train_epoch = _epochs(built)
    train_batch = train_epoch[0]
    record: dict[str, Any] = {}

    module = F.build_module(built.config)
    module.eval()
    with th.no_grad():  # every batch of the epochs: mixed sizes, short final batches
        for split, epoch in (("val", val_epoch), ("train", train_epoch)):
            for i, batch in enumerate(epoch):
                output = _tensors(module(**copy.deepcopy(batch)))
                record[f"forward_eval_{split}{i}"] = F.digest_record(output)

    module = F.build_module(built.config)
    module.train()
    th.manual_seed(FORWARD_SEED)  # Graphormer dropout is hard-coded, so train mode is stochastic
    with th.no_grad():
        record["forward_train"] = F.digest_record(_tensors(module(**copy.deepcopy(train_batch))))

    module = F.build_module(built.config)
    before = {k: v.detach().clone() for k, v in module.state_dict().items()}
    record["n_params"] = sum(p.numel() for p in module.parameters())
    record["param_init"] = F.digest_record({f"param_init/{k}": v for k, v in before.items()})
    if built.model_type == "massformer":
        # BinnedPL keeps its loss names in a local variable; record_one_step needs the attribute.
        module.loss_names = ["loss", "spec_cd"]
    step = F.record_one_step(module, train_batch)
    names = {name for name, _ in module.named_parameters()}
    record["params"] = _check_step(built.model_type, step, before, names)
    record["step"] = F.digest_record(step)
    return record


def _check_step(
    model_type: str,
    step: dict[str, th.Tensor],
    before: dict[str, th.Tensor],
    param_names: set[str],
) -> dict[str, list[str]]:
    """Check the loss terms and the step, and name which parameters the step touched.

    Args:
        model_type: ``massformer`` or ``graff``.
        step: The ``record_one_step`` record.
        before: The state dict (parameters and buffers) before the step.
        param_names: Names of the module's parameters (the rest of the state dict is buffers).

    Returns:
        ``moved`` (parameters that changed), ``grad_not_moved`` (parameters with a nonzero
        gradient that the optimizer did not change), ``no_grad`` (parameters without a gradient)
        and ``buffers_moved`` (batch-norm statistics updated by the train-mode forward).
        ``MassFormerModel.get_split_params`` hands the optimizer only the Graphormer and ``ffn``
        parameters, so the CE, precursor and instrument embedders get gradients but never move;
        the lists keep that behavior on record.
    """
    assert {"loss/loss", "reduce/mean_loss", "reduce/total_loss", "reduce/total_weight"} <= set(
        step
    )
    named = "loss/spec_cd" if model_type == "massformer" else "loss/primary_loss"
    assert named in step and th.equal(step[named], step["loss/loss"])
    loss = step["loss/loss"]
    assert loss.shape == (4,) and th.isfinite(loss).all()  # one value per spectrum in the batch
    assert (loss >= 0).all()
    assert th.isclose(step["reduce/mean_loss"], loss.mean())
    grads = {k.removeprefix("grad/"): v for k, v in step.items() if k.startswith("grad/")}
    after = {
        k.removeprefix("param_after/"): v for k, v in step.items() if k.startswith("param_after/")
    }
    assert set(after) == set(before) and set(grads) <= set(before)
    assert grads and all(th.isfinite(g).all() for g in grads.values())
    assert all(th.isfinite(v).all() for v in after.values() if v.is_floating_point())
    changed = {k for k in before if not th.equal(after[k], before[k])}
    moved = sorted(changed & param_names)
    assert moved and set(moved) <= set(grads)  # nothing moves without a gradient
    nonzero = {k for k, g in grads.items() if g.abs().max() > 0}
    return {
        "moved": moved,
        "grad_not_moved": sorted(nonzero - set(moved)),
        "no_grad": sorted(param_names - set(grads)),
        "buffers_moved": sorted(changed - param_names),
    }


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_forward_and_step_records(model_type: str, synth: Synth) -> None:
    """Forward outputs, every loss term and the post-step parameters with fixed seeds."""

    def make() -> dict:
        with F.deterministic():
            return _forward_step_record(_build(model_type, synth))

    record = _stable(make)
    _ratchet(f"{model_type}:forward_step", record)


# ------------------------------------------------------------------------------ GRAFF library
def _train_spec_df(synth: Synth) -> pd.DataFrame:
    train_ids = pd.read_csv(Path(synth.data.split_dp) / "train_ids.csv")["spec_id"]
    spec_df = pd.read_pickle(synth.data.spec_fp)
    return spec_df[spec_df["spec_id"].isin(train_ids)]


def test_graff_library_is_tie_free_and_recorded(synth: Synth) -> None:
    """The fitted annotation library ranks distinct intensities; its content is recorded."""
    ann_df = synth.ann_df
    train_df = _train_spec_df(synth)
    train_ann = ann_df[ann_df["spec_id"].isin(train_df["spec_id"])]
    products = {p for row in train_ann["ann_products"] for p in row}
    losses = {x for row in train_ann["ann_losses"] for x in row}
    n_candidates = len(products) + len(losses)
    assert n_candidates > LIBRARY_SIZE  # the top-k cut is active, so ties would matter

    # Every candidate of the library, ranked: summed intensities must all differ.
    full = fit_ann_library(train_df.copy(), ann_df.copy(), n_candidates, MAX_ISOTOPE)
    assert len(full) == n_candidates and full["ints"].is_unique
    assert full["ints"].is_monotonic_decreasing

    with F.deterministic():
        built = _build("graff", synth)
    library = built.train_ds.library_df
    assert built.val_ds.library_df[["formula", "kind"]].equals(library[["formula", "kind"]])
    assert library.shape[0] == LIBRARY_SIZE and library["ints"].is_unique
    assert library["ann_id"].tolist() == list(range(LIBRARY_SIZE))
    # The library is the top of the full ranking, and the precursor-loss entry ('') leads.
    assert library[["formula", "kind"]].equals(full.head(LIBRARY_SIZE)[["formula", "kind"]])
    assert library.iloc[0]["formula"] == "" and library.iloc[0]["kind"] == "loss"
    record = {
        "formula": library["formula"].tolist(),
        "kind": library["kind"].tolist(),
        "ann_id": library["ann_id"].tolist(),
        "ints_rounded": [round(float(x), 6) for x in library["ints"]],
        "mzs": F.digest_record({"mzs": library["mzs"].to_numpy()})["mzs"],
    }
    _ratchet("graff:library", record)


# ------------------------------------------------------------------------------ GRAFF eigen
def _flip_signs(graph: Any, seed: int = 0) -> Any:
    """Copy of ``graph`` with a random sign per eigenvector column (and per graph of a batch)."""
    rng = np.random.default_rng(seed)
    flipped = copy.deepcopy(graph)
    ptr = flipped.ptr.tolist() if "ptr" in flipped else [0, flipped.eigvecs.shape[0]]
    for lo, hi in zip(ptr[:-1], ptr[1:]):
        signs = rng.choice([-1.0, 1.0], size=NUM_EIGS)
        signs[0] = -1.0  # column 0 is nonzero for every molecule, so something really flips
        flipped.eigvecs[lo:hi] *= th.as_tensor(signs, dtype=flipped.eigvecs.dtype)
    return flipped


def test_graff_eigen_record_is_sign_invariant(synth: Synth) -> None:
    """The recorded eigen-features ignore eigenvector signs and match an independent eigh."""
    with F.deterministic():
        built = _build("graff", synth)
        graphs = {mol_id: items[0]["mol_graff"] for mol_id, items in _molecule_items(built).items()}
        batch_graph = _epochs(built)[1][0]["mol_graff"]
    for mol_id, graph in graphs.items():
        mol = Chem.MolFromSmiles(SMILES[mol_id])
        n_atoms = mol.GetNumAtoms()
        k = min(NUM_EIGS, n_atoms - 1)  # slots with a nonzero eigenvalue; the rest are padding
        adj = Chem.GetAdjacencyMatrix(mol).astype(np.float64)
        inv_sqrt = 1.0 / np.sqrt(adj.sum(axis=1))
        lap = np.eye(n_atoms) - inv_sqrt[:, None] * adj * inv_sqrt[None, :]
        eigvals, eigvecs = np.linalg.eigh(lap)
        # fixture precondition: the nonzero eigenvalues in slots 1..k+1 (when they exist) are
        # separated from each other, so every recorded eigenvector is unique up to sign
        assert np.diff(eigvals[: min(k + 2, n_atoms)]).min() > 1e-3, SMILES[mol_id]

        expected_vals = np.zeros(NUM_EIGS)
        expected_vals[:k] = eigvals[1 : k + 1]
        assert np.allclose(graph.eigvals[0].numpy(), expected_vals, atol=1e-5)
        V = graph.eigvecs.double().numpy()
        assert not V[n_atoms:].any() and not V[:, k:].any()  # virtual node and padding are zero
        assert np.allclose(V[:n_atoms, :k].T @ V[:n_atoms, :k], np.eye(k), atol=1e-5)
        projector = eigvecs[:, 1 : k + 1] @ eigvecs[:, 1 : k + 1].T  # sign- and basis-free
        assert np.allclose(V[:n_atoms] @ V[:n_atoms].T, projector, atol=1e-5)

        flipped = _flip_signs(graph)
        assert F.tensor_digest(flipped.eigvecs) != F.tensor_digest(graph.eigvecs)  # raw: fragile
        before, after = _eigen_invariants(graph), _eigen_invariants(flipped)
        assert np.array_equal(after["eigvals"], before["eigvals"])
        for gram_before, gram_after in zip(before["eigvec_gram"], after["eigvec_gram"]):
            assert np.array_equal(gram_after, gram_before)
        assert F.digest_record(before) == F.digest_record(after)

    flipped_batch = _flip_signs(batch_graph)  # the collated batch (one flip pattern per graph)
    assert not th.equal(flipped_batch.eigvecs, batch_graph.eigvecs)
    assert F.digest_record(_eigen_invariants(flipped_batch)) == F.digest_record(
        _eigen_invariants(batch_graph)
    )


def test_graff_forward_is_sign_invariant(synth: Synth) -> None:
    """SignNet absorbs eigenvector signs, so the forward record is stable across eigh builds."""
    with F.deterministic():
        built = _build("graff", synth)
        train_batch = _epochs(built)[1][0]
        flipped = copy.deepcopy(train_batch)
        flipped["mol_graff"] = _flip_signs(train_batch["mol_graff"])
        assert not th.equal(flipped["mol_graff"].eigvecs, train_batch["mol_graff"].eigvecs)
        module = F.build_module(built.config)
        module.eval()
        with th.no_grad():
            out = _tensors(module(**copy.deepcopy(train_batch)))
            out_flipped = _tensors(module(**flipped))
    assert out.keys() == out_flipped.keys() and out
    for key, value in out.items():  # exact: phi(v) + phi(-v) only swaps its two summands
        assert th.equal(value, out_flipped[key]), key


# ------------------------------------------------------------------------------ extension
def test_algos_extension_is_built() -> None:
    """The compiled Graphormer ``algos`` extension is importable as a binary module.

    MassFormer preprocessing needs ``floyd_warshall`` and ``gen_edge_input`` from it. A missing
    or stale build fails here by name rather than inside every item record. Tests never build
    it: that would write into ``src/``.
    """
    import fragnnet.baselines.massformer.algos as algos

    assert Path(algos.__file__).suffix == ".so"
    dist, path = algos.floyd_warshall(np.array([[0, 1], [1, 0]], dtype=bool))
    assert dist.tolist() == [[0, 1], [1, 0]] and path.shape == (2, 2)
