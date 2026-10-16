"""MassterPL threading, metrics and export of the rewire-rule candidates (build step 4, arm a0)."""

import re
from pathlib import Path

import numpy as np
import pytest
import torch as th
import yaml
from rdkit import Chem
from torch_geometric.data import Batch

from fragnnet import runner
from fragnnet.export_adapter import ExportAdapter
from fragnnet.inference import export_columns
from fragnnet.inference.checkpoint import apply_inference_config, get_config_from_ckpt
from fragnnet.models.masster import masster_pl as masster_pl_module
from fragnnet.models.masster.candidates.rewire_rules import (
    MASSTER_NUM_REWIRE_RULES,
    REWIRE_RULE_VOCABULARY,
    RewireMatchSet,
    build_rewire_matches,
)
from fragnnet.models.masster.candidates.smarts_templates import build_masster_templates
from fragnnet.models.masster.masster_pl import MassterPL
from fragnnet.models.masster.rewire_expansion import REWIRE_DIAGNOSTIC_NAMES
from fragnnet.utils.feat_utils import get_mol_graph
from fragnnet.utils.misc_utils import deep_update

CONFIG_DIR = str(Path(__file__).parents[1] / "config")
TEMPLATE = Path(CONFIG_DIR) / "template.yml"
REWIRE_KEYS = {
    "masster_rewire_rules": False,
    "masster_rewire_max_extra_fraction": 0.5,
    "masster_rewire_max_extra": 256,
    "masster_rewire_max_matches_per_rule": 8,
    "masster_rewire_continuation_cuts": 0,
    "masster_rewire_pair_chunk": 65536,
    "masster_rewire_null": False,
}
_SULFONE = "O=S(=O)(c1ccccc1)c1ccccc1"
_TINY = dict(
    model_type="masster",
    compile=False,
    sparse_cosine_similarity=True,
    oos_tolerance_multiple=1,
    gaussian_renormalize=True,
    mol_hidden_size=16,
    mol_num_layers=1,
    masster_context_dim=8,
    masster_metadata_embedding_dim=4,
    masster_head_hidden_dim=16,
    masster_intensity_transformer_dim=16,
    masster_intensity_transformer_num_layers=2,
    masster_intensity_transformer_feedforward_dim=32,
    masster_intensity_transformer_num_inducing_points=4,
    masster_unique_fragment_mode=True,
    masster_unique_fragment_target=8,
    masster_ranked_mask_budget=16,
    masster_max_h_transfer=1,
    masster_proposal_selection_only=True,
    masster_proposal_anchored=True,
    masster_proposal_loss_weight=0.1,
    masster_intensity_transformer_attention="none",
    masster_candidate_features=["c_cut_count", "c_size", "c_mass"],
    masster_candidate_coverage_metrics=True,
    masster_candidate_cut_diagnostics=False,
    auxiliary_scores=[],
    loss_vectorized=True,  # the production sparse CE path
)
_REWIRE_ON = dict(masster_rewire_rules=True, masster_rewire_max_extra_fraction=1.0)


def _config(**overrides) -> dict:
    config = runner.load_config(CONFIG_DIR, None, None, model_type="masster")
    config.update(_TINY)
    config.update(overrides)
    return config


def _pl(**overrides) -> MassterPL:
    return MassterPL(**_config(**overrides))


def _batch(pl: MassterPL, smiles: tuple[str, ...]) -> dict:
    params = pl.hparams.mol_params
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    graphs = [
        get_mol_graph(
            mol,
            params["pyg_node_feats"],
            params["pyg_edge_feats"],
            params["pyg_pe_embed_k"],
            True,
        )
        for mol in mols
    ]
    n = len(smiles)
    return {
        "mol_pyg": Batch.from_data_list(graphs),
        "mol_smiles": list(smiles),
        "mol_graph_binary": [mol.ToBinary() for mol in mols],
        "spec_prec_type": th.zeros(n, dtype=th.long),
        "spec_inst_type": th.zeros(n, dtype=th.long),
        "spec_nce": th.full((n,), 30.0),
        "batch_size": n,
    }


# --- template ----------------------------------------------------------------------------------


def test_template_declares_the_seven_keys_with_first_line_comments() -> None:
    config = runner.load_config(CONFIG_DIR, None, None, model_type="masster")
    for key, default in REWIRE_KEYS.items():
        assert config[key] == default and type(config[key]) is type(default), key
    lines = TEMPLATE.read_text().splitlines()
    for key in REWIRE_KEYS:
        (line,) = [text for text in lines if text.startswith(f"{key}:")]
        assert re.search(r"#\s*\S", line), f"{key} needs a one-line first comment"
    assert "c_rewire_edit" in TEMPLATE.read_text()


def test_template_keys_are_not_data_paths() -> None:
    text = (Path(__file__).parents[1] / "scripts" / "setup_job_cc.py").read_text()
    assert "masster_rewire" not in text


# --- threading, guards, static graph -----------------------------------------------------------


def test_defaults_build_no_rewire_module_and_no_matches() -> None:
    pl = _pl()
    assert not pl._rewire_enabled and not pl.model.use_rewire_rules
    assert pl.model.intensity_head.rewire_pathway_residual is None
    assert not [key for key in pl.state_dict() if "rewire" in key]
    batch = _batch(pl, (_SULFONE,))
    assert pl._build_static_graph(batch).rewire_matches is None
    assert not [name for name in pl.loss_names if name.startswith("rewire_")]


def test_pl_threads_every_rewire_value_to_the_model() -> None:
    pl = _pl(
        **_REWIRE_ON,
        masster_rewire_max_extra=17,
        masster_rewire_continuation_cuts=2,
        masster_rewire_pair_chunk=999,
        masster_rewire_max_matches_per_rule=3,
    )
    model = pl.model
    assert model.use_rewire_rules is True
    assert model.rewire_max_extra_fraction == 1.0
    assert model.rewire_max_extra == 17
    assert model.rewire_continuation_cuts == 2
    assert model.rewire_pair_chunk == 999
    assert model.intensity_head.rewire_pathway_residual.in_features == MASSTER_NUM_REWIRE_RULES
    assert pl._rewire_max_matches == 3
    assert set(REWIRE_DIAGNOSTIC_NAMES) <= set(pl.loss_names)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(masster_rewire_null=True), "requires masster_rewire_rules"),
        (dict(masster_rewire_max_extra_fraction=0.0), "fraction"),
        (dict(masster_rewire_max_extra_fraction="0.5"), "fraction"),
        (dict(masster_rewire_max_extra=0), "masster_rewire_max_extra"),
        (dict(masster_rewire_max_matches_per_rule=0), "max_matches_per_rule"),
        (dict(masster_rewire_continuation_cuts=-1), "continuation_cuts"),
        (dict(masster_rewire_pair_chunk=True), "pair_chunk"),
        (dict(masster_rewire_rules=1), "boolean"),
        (dict(**_REWIRE_ON, cross_ev_fragment_loss_weight=0.1), "cross_ev_fragment_loss_weight"),
        (
            dict(
                **_REWIRE_ON,
                masster_separate_ion_mode_proposal_head=True,
                masster_ion_mode_proposal_binary_logits=True,
                masster_ion_mode_proposal_loss_weight=0.1,
            ),
            "masster_ion_mode_proposal_loss_weight",
        ),
    ],
)
def test_bad_rewire_configs_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _pl(**overrides)


def test_flag_off_keys_are_only_type_checked() -> None:
    pl = _pl(masster_rewire_max_extra=5, masster_rewire_pair_chunk=7)
    assert not pl._rewire_enabled
    with pytest.raises(ValueError, match="pair_chunk"):
        _pl(masster_rewire_pair_chunk=0.5)


def test_static_graph_has_one_cached_match_set_per_spectrum() -> None:
    pl = _pl(**_REWIRE_ON)
    smiles = (_SULFONE, "CCO", _SULFONE)
    graph = pl._build_static_graph(_batch(pl, smiles))
    assert len(graph.rewire_matches) == 3
    assert all(isinstance(m, RewireMatchSet) for m in graph.rewire_matches)
    assert graph.rewire_matches[0] is graph.rewire_matches[2]  # lru_cache: same object
    assert len(graph.rewire_matches[0].matches) > 0
    assert len(graph.rewire_matches[1].matches) == 0
    assert graph.smarts_templates is None


def test_cache_key_includes_the_cap_and_the_null_flag() -> None:
    f = masster_pl_module._cached_masster_rewire_matches
    sulfone = Chem.MolFromSmiles(_SULFONE).ToBinary()
    assert f(sulfone, 8, False) is f(sulfone, 8, False)
    assert f(sulfone, 1, False) is not f(sulfone, 8, False)
    assert len(f(sulfone, 1, False).matches) <= len(f(sulfone, 8, False).matches)


def test_graph_molecule_without_ring_info_still_matches() -> None:
    # Multi-component retrieval candidates are stored without ring info; SMARTS ring queries on
    # them raised "RingInfo not initialized" until the matcher recomputed it.
    mol = Chem.MolFromSmiles("CC.C1CC1." + _SULFONE)
    edited = Chem.RWMol(mol)
    edited.RemoveAtom(edited.AddAtom(Chem.Atom(6)))  # an edit resets the ring info
    binary = edited.GetMol().ToBinary()
    bare = Chem.Mol(binary)  # keep a reference: RingInfo of a temporary Mol reads freed memory
    with pytest.raises(RuntimeError):
        bare.GetRingInfo().NumRings()
    restored = masster_pl_module._graph_mol(binary)
    assert restored.GetRingInfo().AtomRings() == mol.GetRingInfo().AtomRings()
    f = masster_pl_module._cached_masster_rewire_matches
    expected = build_rewire_matches(mol, max_matches_per_rule=8)
    assert f(binary, 8, False) == expected and expected.matches
    assert masster_pl_module._cached_masster_templates(binary) == build_masster_templates(mol)


def test_static_graph_requires_the_graph_molecule() -> None:
    pl = _pl(**_REWIRE_ON)
    batch = _batch(pl, (_SULFONE,))
    del batch["mol_graph_binary"]
    with pytest.raises(KeyError, match="mol_graph_binary"):
        pl._build_static_graph(batch)


def test_matches_index_the_graph_when_smiles_atom_order_differs() -> None:
    # Retrieval candidates store the raw SMILES next to a standardized mol whose atoms are in
    # another order; matching on the re-parsed SMILES stopped rewire expansion ("RDKit and graph
    # atom order disagree"). The graph's own molecule must drive the match.
    smiles = "CCO." + _SULFONE
    parsed = Chem.MolFromSmiles(smiles)
    mol = Chem.RenumberAtoms(parsed, list(reversed(range(parsed.GetNumAtoms()))))
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] != [a.GetAtomicNum() for a in parsed.GetAtoms()]
    pl = _pl(**_REWIRE_ON).eval()
    params = pl.hparams.mol_params
    graph = get_mol_graph(
        mol, params["pyg_node_feats"], params["pyg_edge_feats"], params["pyg_pe_embed_k"], True
    )
    batch = {
        **_batch(pl, (smiles,)),
        "mol_pyg": Batch.from_data_list([graph]),
        "mol_graph_binary": [mol.ToBinary()],
    }
    static_graph = pl._build_static_graph(batch)
    expected = build_rewire_matches(mol, max_matches_per_rule=pl._rewire_max_matches)
    assert static_graph.rewire_matches[0] == expected and expected.matches
    with th.no_grad():
        output = pl.forward(**batch)
    assert output["rewire_extra_count"][0] > 0


def test_null_flag_routes_to_the_null_builder(monkeypatch) -> None:
    # The null module is build step 5; the PL only has to route to it.
    import sys
    import types

    seen = {}

    def build_null(mol, *, max_matches_per_rule):
        seen["call"] = (Chem.MolToSmiles(mol), max_matches_per_rule)
        return RewireMatchSet(matches=(), num_truncated=0)

    module = types.ModuleType("fragnnet.models.masster.candidates.rewire_null")
    module.build_null_rewire_matches = build_null
    monkeypatch.setitem(sys.modules, module.__name__, module)
    pl = _pl(**_REWIRE_ON, masster_rewire_null=True, masster_rewire_max_matches_per_rule=5)
    masster_pl_module._cached_masster_rewire_matches.cache_clear()
    try:
        pl._build_static_graph(_batch(pl, ("CCOC",)))
    finally:
        masster_pl_module._cached_masster_rewire_matches.cache_clear()  # drop the stub's result
    assert seen["call"] == ("CCOC", 5)


def test_standalone_forward_builds_the_matches_the_sidecar_lacks() -> None:
    pl = _pl(**_REWIRE_ON).eval()
    batch = _batch(pl, (_SULFONE, "CCO"))
    with th.no_grad():
        output = pl.forward(**batch)  # no masster_static_graph: the fallback builds it
        sidecar = pl.forward(**batch, masster_static_graph=pl._build_static_graph(batch))
    assert output["rewire_extra_count"][0] > 0 and output["rewire_extra_count"][1] == 0
    assert th.equal(output["pred_logprobs"], sidecar["pred_logprobs"])
    # A sidecar without matches is rebuilt as well (the SMARTS-only sidecar of an old caller).
    bare = pl._build_static_graph(batch)
    bare = type(bare)(
        chemical_bond_index=bare.chemical_bond_index,
        atom_batch_indices=bare.atom_batch_indices,
        atom_ptr=bare.atom_ptr,
    )
    with th.no_grad():
        rebuilt = pl.forward(**batch, masster_static_graph=bare)
    assert th.equal(rebuilt["rewire_extra_count"], output["rewire_extra_count"])


# --- old checkpoints ---------------------------------------------------------------------------


def test_old_checkpoint_without_the_keys_loads_with_rewire_off(tmp_path) -> None:
    pl = _pl()
    hparams = dict(pl.hparams)
    for key in REWIRE_KEYS:
        del hparams[key]
    ckpt = tmp_path / "old.ckpt"
    th.save({"hyper_parameters": hparams, "state_dict": pl.state_dict()}, ckpt)

    # The standard inference path (get_config_from_ckpt alone, no template merge) loads it.
    direct = apply_inference_config(
        get_config_from_ckpt(str(ckpt)), disable_preproc=True, num_workers=0
    )
    for key, default in REWIRE_KEYS.items():
        assert direct[key] == default
    assert not MassterPL(**direct)._rewire_enabled

    saved = dict(hparams)
    # The G0 path: checkpoint hparams win, the template fills only the keys they predate.
    template = runner.load_config(CONFIG_DIR, None, [], model_type="masster")
    config = apply_inference_config(
        deep_update(template, saved), disable_preproc=True, num_workers=0
    )
    for key, default in REWIRE_KEYS.items():
        assert config[key] == default
    loaded = MassterPL(**config)
    loaded.load_state_dict(th.load(ckpt, weights_only=False)["state_dict"], strict=True)
    assert not loaded._rewire_enabled and not loaded.model.use_rewire_rules
    assert not [key for key in loaded.state_dict() if "rewire" in key]
    assert loaded.state_dict().keys() == pl.state_dict().keys()
    assert loaded._build_static_graph(_batch(loaded, (_SULFONE,))).rewire_matches is None


def test_rewire_checkpoint_round_trips_its_vocabulary_buffer() -> None:
    pl = _pl(**_REWIRE_ON)
    again = _pl(**_REWIRE_ON)
    again.load_state_dict(pl.state_dict(), strict=True)
    assert [key for key in pl.state_dict() if "rewire" in key]
    # A control (OFF) checkpoint loads into an ON model as the control: zero-init residual.
    again.load_state_dict(_pl().state_dict(), strict=True)
    assert not again.model.intensity_head.rewire_pathway_residual.weight.any()


def test_fragnnet_config_carries_the_keys_unread() -> None:
    config = runner.load_config(CONFIG_DIR, None, None, model_type="frag_gnn")
    for key, default in REWIRE_KEYS.items():
        assert config[key] == default


# --- training step and metrics -----------------------------------------------------------------


def _targets(output: dict, batch_size: int) -> dict:
    # Observed m/z arrive as float32; the peak is a predicted peak of each spectrum.
    pred_mzs, pred_batch = output["pred_mzs"], output["pred_batch_idxs"]
    true_mzs = th.stack([pred_mzs[pred_batch == i][0] for i in range(batch_size)]).float()
    return {
        "true_mzs": true_mzs,
        "true_intensities": th.ones(batch_size, dtype=th.float32),
        "true_logprobs": th.zeros(batch_size),
        "true_batch_idxs": th.arange(batch_size),
    }


def test_training_step_logs_the_rewire_diagnostics_and_stays_finite() -> None:
    pl = _pl(**_REWIRE_ON).train()
    batch = _batch(pl, (_SULFONE, "CCO"))
    output = pl.forward(**batch)
    assert output["rewire_extra_count"][0] > 0
    loss = pl.loss_fn(**output, **_targets(output, 2), batch_size=2)
    for name in REWIRE_DIAGNOSTIC_NAMES:
        assert name in loss and name in pl.loss_names
        assert loss[name].shape == (2,) and not loss[name].requires_grad
    assert loss["rewire_extra_count"][0] > 0
    assert loss["rewire_extra_count"][1] == 0
    assert th.isfinite(loss["loss"]).all()
    loss["loss"].sum().backward()
    residual = pl.model.intensity_head.rewire_pathway_residual.weight
    assert residual.grad is not None and th.isfinite(residual.grad).all()


def test_control_model_logs_no_rewire_names() -> None:
    pl = _pl().train()
    batch = _batch(pl, (_SULFONE,))
    output = pl.forward(**batch)
    assert not [key for key in output if "rewire" in key]
    loss = pl.loss_fn(**output, **_targets(output, 1), batch_size=1)
    assert not [key for key in loss if key.startswith("rewire_")]


def test_coverage_split_attributes_extras_to_the_gain() -> None:
    pl = _pl(**_REWIRE_ON, auxiliary_scores=[])
    assert {"candidate_wrecall_ordinary", "candidate_wrecall_rewire_gain"} <= pl.metric_names
    assert "rewire_candidate_is_ordinary_by_cut" in pl.extra_metric_args
    mzs = th.tensor([100.0, 200.0])
    ints = th.tensor([0.25, 0.75])
    metrics = pl.metric_fn(
        true_mzs=mzs,
        true_ints=ints,
        true_logprobs=ints.log(),
        true_batch_idxs=th.zeros(2, dtype=th.long),
        pred_mzs=mzs,
        pred_ints=ints,
        pred_logprobs=ints.log(),
        pred_batch_idxs=th.zeros(2, dtype=th.long),
        true_prec_mzs=th.tensor([300.0]),
        split="val",
        candidate_mzs_by_cut=th.tensor([100.0, 200.0, 300.0]),
        candidate_batch_idxs_by_cut=th.zeros(3, dtype=th.long),
        rewire_candidate_is_ordinary_by_cut=th.tensor([True, False, True]),
    )
    assert metrics["candidate_wrecall_ordinary"].item() == pytest.approx(0.25)
    assert metrics["candidate_wrecall_rewire_gain"].item() == pytest.approx(0.75)
    assert "candidate_wrecall" not in metrics


def test_validation_outputs_feed_the_coverage_split() -> None:
    pl = _pl(**_REWIRE_ON).eval()
    batch = _batch(pl, (_SULFONE,))
    with th.no_grad():
        output = pl.forward(**batch)
    ordinary = output["rewire_candidate_is_ordinary_by_cut"]
    extra = output["rewire_candidate_is_extra_by_cut"]
    assert ordinary.shape == output["candidate_mzs_by_cut"].shape == extra.shape
    assert ordinary.dtype == th.bool and not (ordinary & extra).any()
    assert extra.any()  # the sulfone's extras
    mzs = output["candidate_mzs_by_cut"]
    # A peak only the extras reach is recovered by the gain, not by the ordinary pool.
    only_extra = mzs[extra]
    only_extra = only_extra[~th.isin(only_extra, mzs[ordinary])][:1]
    assert only_extra.numel() == 1, "the sulfone must have an m/z only its extras reach"
    ints = th.ones(1)
    metrics = pl.metric_fn(
        true_mzs=only_extra,
        true_ints=ints,
        true_logprobs=ints.log(),
        true_batch_idxs=th.zeros(1, dtype=th.long),
        pred_mzs=only_extra,
        pred_ints=ints,
        pred_logprobs=ints.log(),
        pred_batch_idxs=th.zeros(1, dtype=th.long),
        true_prec_mzs=th.tensor([float(mzs.max()) + 1.0]),
        split="val",
        candidate_mzs_by_cut=mzs,
        candidate_batch_idxs_by_cut=output["candidate_batch_idxs_by_cut"],
        rewire_candidate_is_ordinary_by_cut=ordinary,
    )
    assert metrics["candidate_wrecall_ordinary"].item() == 0.0
    assert metrics["candidate_wrecall_rewire_gain"].item() == pytest.approx(1.0)


# --- export ------------------------------------------------------------------------------------


def test_default_adapter_returns_nan_for_rewire_rules() -> None:
    assert np.isnan(ExportAdapter().export_node_rewire_rules(0, None, None))


def test_column_is_present_exactly_with_rule_provenance() -> None:
    hparams = dict(
        auxiliary_scores=[], eval_mz_bin_res=[], model_type="masster", masster_rewire_rules=True
    )
    off_hparams = dict(hparams, masster_rewire_rules=False)
    on = export_columns.get_tabular_columns(
        hparams, return_cc_hash=True, output_subset=None, return_rule_provenance=True
    )
    off = export_columns.get_tabular_columns(
        hparams, return_cc_hash=True, output_subset=None, return_rule_provenance=False
    )
    smarts_only = export_columns.get_tabular_columns(
        off_hparams, return_cc_hash=True, output_subset=None, return_rule_provenance=True
    )
    assert on.index("pred_node_rewire_rules") == on.index("pred_node_rules") + 1
    assert "pred_node_rewire_rules" not in off and "pred_node_rules" not in off
    # Rewire off: the provenance column set is exactly the pre-rewire one.
    assert "pred_node_rules" in smarts_only and "pred_node_rewire_rules" not in smarts_only
    assert smarts_only == [c for c in on if c != "pred_node_rewire_rules"]


def test_pl_export_node_rewire_rules_uses_the_batch_offsets() -> None:
    from types import SimpleNamespace

    pl = _pl()
    annotation = SimpleNamespace(
        node_rewire_rules=np.array(["", "a", "a;b", "c"], dtype=object),
        node_cumsizes=th.tensor([0, 2, 4]),
    )
    assert pl.export_node_rewire_rules(1, th.tensor([0, 1]), annotation) == ["a;b", "c"]
    assert np.isnan(pl.export_node_rewire_rules(0, None, annotation))
    assert np.isnan(
        pl.export_node_rewire_rules(0, th.tensor([0]), SimpleNamespace(node_rewire_rules=None))
    )


def test_export_of_a_real_batch_keeps_node_rows_aligned_with_an_unmatched_spectrum() -> None:
    pl = _pl(**_REWIRE_ON).eval()
    batch = _batch(pl, (_SULFONE, "CCO", "CS(=O)(=O)c1ccccc1"))
    with th.no_grad():
        output = pl.forward(**batch, return_explainability_tensors=True)
    assert "fragment_rewire_parent_idxs" in output
    fragment_idxs = output["fragment_rewire_fragment_idxs"]
    batch_idxs = output["ion_h_fragment_batch_idxs"]
    # Every rewire pair belongs to a fragment of a spectrum that has matches; ethanol has none.
    assert set(batch_idxs[fragment_idxs].tolist()) <= {0, 2}
    assert int(output["fragment_rewire_rule_idxs"].max()) < MASSTER_NUM_REWIRE_RULES
    names = {REWIRE_RULE_VOCABULARY[i] for i in output["fragment_rewire_rule_idxs"].tolist()}
    assert names

    probs = output["pred_logprobs"].exp()
    totals = probs.new_zeros(3).index_add(0, output["pred_batch_idxs"], probs)
    ints = probs / totals[output["pred_batch_idxs"]]
    peaks = (output["pred_mzs"], ints, output["pred_batch_idxs"])
    batch_input = {
        "mol_pyg": batch["mol_pyg"],
        "spec_prec_type": batch["spec_prec_type"],
    }
    _, annotation = pl.postprocess_export_peaks(
        batch_input,
        dict(output),
        peaks,
        need_annotation=True,
        return_cc_hash=True,
        return_rule_provenance=True,
    )
    node_names = annotation.node_rewire_rules
    assert node_names is not None and len(node_names) == len(annotation.node_batch_idxs)
    by_spectrum = {
        b: {n for n in node_names[annotation.node_batch_idxs.numpy() == b] if n}
        for b in range(3)
    }
    assert by_spectrum[1] == set()
    assert by_spectrum[0] and by_spectrum[2]
    for text in set().union(*by_spectrum.values()):
        assert set(text.split(";")) <= set(REWIRE_RULE_VOCABULARY)
    # Flag off: no rewire keys in the result, so the column stays NaN.
    off = _pl().eval()
    with th.no_grad():
        off_output = off.forward(**batch, return_explainability_tensors=True)
    probs = off_output["pred_logprobs"].exp()
    totals = probs.new_zeros(3).index_add(0, off_output["pred_batch_idxs"], probs)
    _, off_annotation = off.postprocess_export_peaks(
        batch_input,
        dict(off_output),
        (
            off_output["pred_mzs"],
            probs / totals[off_output["pred_batch_idxs"]],
            off_output["pred_batch_idxs"],
        ),
        need_annotation=True,
        return_cc_hash=True,
        return_rule_provenance=True,
    )
    assert off_annotation.node_rewire_rules is None
    assert np.isnan(off.export_node_rewire_rules(0, th.tensor([0]), off_annotation))


def test_arm_configs_if_present_enable_rewire() -> None:
    directory = Path(CONFIG_DIR) / "masster" / "mh_rewire_after_selection_2026_10_15"
    if not directory.exists():
        pytest.skip("arm configs are written in the final step")
    for file in directory.glob("*.yml"):
        assert "masster_rewire_rules" in yaml.safe_load(file.read_text())
