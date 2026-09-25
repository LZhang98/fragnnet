"""Tests for the intensity-head memorization probe's corruption scopes."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch as th

from fragnnet.model.bond_fragment_heads import CandidateIntensityTransformerHead

SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "debug" / "debug_pooled_embedding_memorization.py"
)
SPEC = importlib.util.spec_from_file_location("debug_pooled_embedding_memorization", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
# The probe defines a dataclass, which resolves its own module through sys.modules.
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)

EMBEDDING_DIM = 6
CONTEXT_DIM = 3
FEATURE_DIM = 2
# Two molecules with five and three candidates, interleaved as the model may deliver them.
BATCH_INDICES = th.tensor([0, 1, 0, 0, 1, 0, 1, 0])


def _head(attention_type: str) -> CandidateIntensityTransformerHead:
    th.manual_seed(0)
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=EMBEDDING_DIM,
        context_dim=CONTEXT_DIM,
        model_dim=12,
        num_layers=2,
        num_heads=3,
        feedforward_dim=24,
        num_inducing_points=4,
        num_ion_modes=2,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=False,
        attention_type=attention_type,
        candidate_feature_dim=FEATURE_DIM,
        unique_fragment_tokens=True,
    )
    # The residual head is zero-initialized, which would make every output identical.
    th.nn.init.normal_(head.component_residual_head.weight)
    return head.eval()


def _inputs() -> tuple[tuple[th.Tensor, ...], dict]:
    generator = th.Generator().manual_seed(1)
    num_components = BATCH_INDICES.shape[0]
    blocks = tuple(th.randn(num_components, EMBEDDING_DIM, generator=generator) for _ in range(3))
    args = (
        *blocks,
        th.zeros(num_components),
        th.zeros(num_components),
        BATCH_INDICES,
        th.arange(num_components),
    )
    kwargs = {
        "batch_size": 2,
        "num_masks": num_components,
        "context": th.randn(num_components, CONTEXT_DIM, generator=generator),
        "candidate_features": th.randn(num_components, FEATURE_DIM, generator=generator),
    }
    return args, kwargs


def _residuals(
    head: CandidateIntensityTransformerHead,
    condition_label: str,
    set_drop_fraction: float = 0.5,
) -> th.Tensor:
    conditions = {
        condition.label: condition
        for condition in PROBE.build_conditions(["cc", "features", "set"], ["zero"])
    }
    args, kwargs = _inputs()
    with th.no_grad(), PROBE.PooledEmbeddingCorruptor(head, 7, set_drop_fraction) as corruptor:
        corruptor.condition = conditions[condition_label]
        return corruptor.head.forward(*args, **kwargs)[0]


def test_build_conditions_gives_set_scope_one_condition() -> None:
    labels = [
        condition.label
        for condition in PROBE.build_conditions(["features", "set"], ["molmean", "zero"])
    ]

    assert labels == ["none", "features_molmean", "features_zero", "set_drop"]
    with pytest.raises(ValueError, match="Unknown corruption scopes"):
        PROBE.build_conditions(["context"], ["zero"])


def test_drop_from_keys_keeps_padding_and_one_real_candidate_per_molecule() -> None:
    padding_mask = th.tensor(
        [
            [False, False, False, True],
            [True, False, False, True],
        ]
    )

    masked = PROBE.drop_from_keys(padding_mask, 0.999, th.Generator().manual_seed(0))

    assert masked.shape == padding_mask.shape
    assert masked[padding_mask].all()
    # Near-certain hiding empties both rows, so each keeps exactly its first real slot.
    assert th.equal(masked, th.tensor([[False, True, True, True], [True, False, True, True]]))


@pytest.mark.parametrize("attention_type", ["induced", "full"])
def test_set_drop_at_zero_fraction_is_the_identity(attention_type: str) -> None:
    head = _head(attention_type)

    reference = _residuals(head, "none")
    unmasked = _residuals(head, "set_drop", set_drop_fraction=0.0)

    assert th.equal(unmasked, reference)


@pytest.mark.parametrize("attention_type", ["induced", "full"])
def test_set_drop_changes_context_but_still_scores_every_candidate(attention_type: str) -> None:
    head = _head(attention_type)

    reference = _residuals(head, "none")
    dropped = _residuals(head, "set_drop", set_drop_fraction=0.5)

    assert dropped.shape == reference.shape == (BATCH_INDICES.shape[0],)
    assert th.isfinite(dropped).all()
    assert not th.allclose(dropped, reference)


def test_features_scope_corrupts_only_the_descriptor_block() -> None:
    head = _head("induced")
    args, kwargs = _inputs()
    with th.no_grad():
        expected = head(*args, **{**kwargs, "candidate_features": th.zeros(8, FEATURE_DIM)})[0]
        cc_zeroed = head(*(th.zeros_like(args[0]), th.zeros_like(args[1]), *args[2:]), **kwargs)[0]

    corrupted = _residuals(head, "features_zero")

    assert th.allclose(corrupted, expected)
    assert th.allclose(_residuals(head, "cc_zero"), cc_zeroed)


def test_restore_returns_the_uncorrupted_head() -> None:
    head = _head("induced")
    args, kwargs = _inputs()
    with th.no_grad():
        reference = head(*args, **kwargs)[0]
        with PROBE.PooledEmbeddingCorruptor(head, 7, 0.5) as corruptor:
            corruptor.condition = PROBE.build_conditions(["set"], [])[1]
            corruptor.head.forward(*args, **kwargs)
        restored = head(*args, **kwargs)[0]

    assert th.equal(restored, reference)


def test_corruptor_rejects_an_out_of_range_drop_fraction() -> None:
    with pytest.raises(ValueError, match="set_drop_fraction"):
        PROBE.PooledEmbeddingCorruptor(_head("induced"), 7, 1.0)


# Routes outside the token (memorization-route plan, Step 0).


def test_build_conditions_gives_the_new_scopes_their_modes() -> None:
    labels = [
        condition.label
        for condition in PROBE.build_conditions(
            ["gsa", "acharge", "ionh", "feat_mass"], ["molmean", "shuffle"]
        )
    ]

    assert labels == [
        "none",
        "gsa_popmean",
        "gsa_molmean",
        "gsa_xmol",
        "acharge_round",
        "ionh_molmean",
        "ionh_popmean",
        "feat_mass_molmean",
        "feat_mass_shuffle",
    ]


def test_corrupt_rows_by_molecule_modes() -> None:
    rows = th.tensor([[1.0, 2.0], [10.0, 20.0], [3.0, 4.0], [30.0, 40.0], [50.0, 60.0]])
    batch_indices = th.tensor([0, 1, 0, 1, 1])
    means = th.tensor([[2.0, 3.0], [30.0, 40.0]])
    generator = th.Generator().manual_seed(0)

    popmean = PROBE.corrupt_rows_by_molecule(rows, batch_indices, 2, "popmean", generator)
    molmean = PROBE.corrupt_rows_by_molecule(rows, batch_indices, 2, "molmean", generator)
    xmol = PROBE.corrupt_rows_by_molecule(rows, batch_indices, 2, "xmol", generator)

    assert th.allclose(popmean, rows.mean(dim=0).expand_as(rows))
    assert th.allclose(molmean, means[batch_indices])
    # With two molecules the only other molecule is the donor, so identity is swapped.
    assert th.allclose(xmol, means[1 - batch_indices])
    ion_h = th.randn(5, 2, 3)
    assert PROBE.corrupt_rows_by_molecule(
        ion_h, batch_indices, 2, "molmean", generator
    ).shape == ion_h.shape
    with pytest.raises(ValueError, match="at least two molecules"):
        PROBE.corrupt_rows_by_molecule(rows, th.zeros(5, dtype=th.long), 1, "xmol", generator)


def test_feature_column_scope_rewrites_only_its_columns() -> None:
    head = _head("induced")
    args, kwargs = _inputs()
    seen: list[th.Tensor] = []
    real_forward = head.forward

    def _spy(*spy_args, **spy_kwargs):
        seen.append(spy_kwargs["candidate_features"])
        return real_forward(*spy_args, **spy_kwargs)

    head.forward = _spy
    condition = PROBE.build_conditions(["feat_mass"], ["molmean"])[1]
    columns = {"c_cut_count": (0, 1), "c_mass": (1, 2)}
    with th.no_grad(), PROBE.PooledEmbeddingCorruptor(
        head, 7, 0.5, feature_columns=columns
    ) as corruptor:
        corruptor.condition = condition
        corruptor.head.forward(*args, **kwargs)

    features = kwargs["candidate_features"]
    rewritten = seen[-1]
    assert th.equal(rewritten[:, 0], features[:, 0])
    expected_mass = PROBE.corrupt_block(
        features[:, 1:2], BATCH_INDICES, 2, "molmean", th.Generator()
    )
    assert th.allclose(rewritten[:, 1:2], expected_mass)


def test_ionh_scope_rewrites_only_the_ion_h_logits() -> None:
    head = _head("induced")
    args, kwargs = _inputs()
    condition = PROBE.build_conditions(["ionh"], ["molmean"])[1]
    with th.no_grad():
        clean = head(*args, **kwargs)
        with PROBE.PooledEmbeddingCorruptor(head, 7, 0.5) as corruptor:
            corruptor.condition = condition
            rewritten = corruptor.head.forward(*args, **kwargs)

    assert th.equal(rewritten[0], clean[0]) and th.equal(rewritten[1], clean[1])
    expected = PROBE.corrupt_rows_by_molecule(
        clean[2], BATCH_INDICES, 2, "molmean", th.Generator()
    )
    assert th.allclose(rewritten[2], expected)


def test_atom_feature_column_locates_the_gasteiger_charge() -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from fragnnet.utils.feat_utils import get_mol_graph

    atom_feats = ["a_onehot", "a_degree", "a_charge"]
    molecule = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")
    graph = get_mol_graph(molecule, atom_feats, ["b_degree"], 0, True)
    column = PROBE.atom_feature_column(atom_feats, ["b_degree"], "a_charge")
    AllChem.ComputeGasteigerCharges(molecule)
    charges = th.tensor([atom.GetDoubleProp("_GasteigerCharge") for atom in molecule.GetAtoms()])

    assert column == graph.x.shape[1] - 1
    assert th.allclose(graph.x[:, column].float(), charges.float(), atol=1e-5)
    assert PROBE.atom_feature_column(["a_onehot"], ["b_degree"], "a_charge") is None


def test_encoder_corruptor_rewrites_only_the_intensity_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proposal and the candidate set stay exact; only the intensity allocation moves."""
    from tests.test_masster_model import _build_batch, _build_model

    batch = _build_batch(("CCOC", "CCO"))
    model = _build_model(
        batch,
        precursor_gnn_type="GINE_GSA",
        boundary_bond_encoder=True,
        max_masks=4,
        max_enumerated_masks=8,
        unique_fragment_mode=True,
    ).eval()
    conditions = {c.label: c for c in PROBE.build_conditions(["gsa"], ["molmean"])}
    original_pool = PROBE.masster_model_module.pool_component_and_complement_embeddings

    def _forward() -> dict[str, th.Tensor]:
        return model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0, 0]),
            spec_nce=th.tensor([10.0, 40.0]),
        )

    with th.no_grad(), PROBE.EncoderCorruptor(model, 7, None) as encoder:
        encoder.condition = conditions["none"]
        clean = _forward()
        encoder.verify(conditions["none"])
        encoder.condition = conditions["gsa_popmean"]
        rewritten = _forward()
        encoder.verify(conditions["gsa_popmean"])
        assert encoder._swaps == 2

    assert th.equal(rewritten["bond_break_logits"], clean["bond_break_logits"])
    assert th.equal(rewritten["pred_mzs"], clean["pred_mzs"])
    assert not th.allclose(rewritten["pred_logprobs"], clean["pred_logprobs"])
    assert PROBE.masster_model_module.pool_component_and_complement_embeddings is (
        original_pool
    )
    assert not encoder._hooks


def test_encoder_corruptor_refuses_a_model_it_cannot_rewrite_exactly() -> None:
    from tests.test_masster_model import _build_batch, _build_model

    batch = _build_batch(("CCO",))
    model = _build_model(batch, precursor_gnn_type="GINE_GSA", boundary_bond_encoder=False)

    with pytest.raises(ValueError, match="boundary block built by CutBondEncoder"):
        PROBE.EncoderCorruptor(model, 7, None).install()
