"""Spectrum context at MASSter atom inputs and legacy checkpoint compatibility."""

import pytest
import torch as th

from fragnnet.pl_model import MassterPL
from fragnnet.utils.pl_utils import get_pl_hparams
from tests.test_masster_model import _build_batch, _build_model
from tests.test_masster_pl import _build_pl


@pytest.mark.parametrize("separate", [False, True])
def test_merged_context_changes_only_its_spectrums_atom_embeddings(separate: bool) -> None:
    batch = _build_batch(("CCO", "CCO"))
    model = _build_model(
        batch,
        context_in_gnn=True,
        ce_insert_type="embed",
        proposal_selection_only=separate,
        separate_proposal_encoder=separate,
        unique_fragment_mode=separate,
        use_base_probability_feature=False,
    ).eval()
    captured = []
    handle = model.precursor_encoder.register_forward_hook(
        lambda module, args, output: captured.append((args[0].detach().clone(), output))
    )
    args = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([20.0, 40.0, 20.0, 40.0]),
        "spec_nce_batch_idxs": th.tensor([0, 0, 1, 1]),
    }
    try:
        output = model(**args)
        args["spec_nce"] = th.tensor([60.0, 80.0, 20.0, 40.0])
        model(**args)
    finally:
        handle.remove()
    first_inputs, first_atoms = captured[0]
    second_inputs, second_atoms = captured[1]
    th.testing.assert_close(first_inputs[:, : batch.x.shape[1]], batch.x)
    th.testing.assert_close(first_atoms[:3], first_atoms[3:])
    assert not th.allclose(first_atoms[:3], second_atoms[:3])
    th.testing.assert_close(first_atoms[3:], second_atoms[3:])
    th.testing.assert_close(first_inputs[3:], second_inputs[3:])
    # Isolate the new route: an atom-embedding loss must reach the CE encoder.
    first_atoms.square().sum().backward()
    projection = model.context_encoder.nce_embedding[1].weight
    assert projection.grad is not None
    assert th.isfinite(projection.grad).all() and projection.grad.abs().sum() > 0
    assert th.isfinite(output["pred_logprobs"]).all()


def test_context_input_rejects_cross_energy_embedding_reuse() -> None:
    with pytest.raises(ValueError, match="incompatible with share_proposal_across_ev"):
        _build_model(_build_batch(), context_in_gnn=True, share_proposal_across_ev=True)


def test_legacy_checkpoint_migration_preserves_strict_state_loading() -> None:
    legacy, config = _build_pl()
    del config["masster_context_in_gnn"]
    checkpoint = {"hyper_parameters": config}
    migrated = get_pl_hparams(checkpoint)
    assert migrated["masster_context_in_gnn"] is False
    assert "masster_context_in_gnn" not in checkpoint["hyper_parameters"]
    restored = MassterPL(**migrated)
    restored.load_state_dict(legacy.state_dict(), strict=True)
    assert not restored.model.context_in_gnn
    migrated["masster_context_in_gnn"] = True
    assert get_pl_hparams({"hyper_parameters": migrated})["masster_context_in_gnn"] is True


def test_fresh_config_requires_explicit_context_option() -> None:
    _, config = _build_pl()
    del config["masster_context_in_gnn"]
    with pytest.raises(AttributeError, match="masster_context_in_gnn"):
        MassterPL(**config)
