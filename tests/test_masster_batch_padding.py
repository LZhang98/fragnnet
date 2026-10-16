"""Batch isolation and padding invariance for MASSter candidates and spectra."""

import pytest
import torch as th

from fragnnet.models.masster.masster_candidate_transformer import CandidateIntensityTransformerHead
from tests.test_masster_model import _build_batch, _build_model


@pytest.fixture(params=["induced", "full", "none"])
def head(request: pytest.FixtureRequest) -> CandidateIntensityTransformerHead:
    """Use multiple layers so padded query outputs can become keys in later layers."""
    th.manual_seed(719)
    return CandidateIntensityTransformerHead(
        fragment_embedding_dim=4,
        context_dim=0,
        model_dim=8,
        num_layers=3,
        num_heads=2,
        feedforward_dim=16,
        num_inducing_points=3,
        num_ion_modes=2,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=False,
        attention_type=request.param,
    ).eval()


@pytest.mark.parametrize("norm_placement", ["pre", "post"])
def test_padding_values_do_not_change_real_outputs_or_gradients(
    head: CandidateIntensityTransformerHead, norm_placement: str,
) -> None:
    """Finite poison in masked slots must not reach real tokens, even after three layers."""
    for layer in head.layers:
        layer.norm_placement = norm_placement
    tokens = th.randn(1, 3, 8)
    probe = th.randn_like(tokens)

    def run(padding_value: float, num_padding: int) -> tuple[th.Tensor, th.Tensor]:
        padded = th.cat([tokens, th.full((1, num_padding, 8), padding_value)], dim=1)
        padded.requires_grad_()
        mask = th.arange(padded.shape[1]).unsqueeze(0) >= 3
        latent = None if head.inducing_tokens is None else head.inducing_tokens.expand(1, -1, -1)
        output = padded
        for layer in head.layers:
            output, latent = layer(output, latent, mask)
        gradient = th.autograd.grad((output[:, :3] * probe).sum(), padded)[0]
        assert th.isfinite(output).all() and th.isfinite(gradient).all()
        assert th.count_nonzero(gradient[:, 3:]) == 0
        return output[:, :3], gradient[:, :3]

    reference = run(0.0, 0)
    for value, width in [(0.0, 1), (1e4, 5), (-1e4, 5)]:
        actual = run(value, width)
        for expected, observed in zip(reference, actual, strict=True):
            th.testing.assert_close(observed, expected, atol=2e-6, rtol=1e-5)


def test_shuffled_ragged_buckets_match_singletons_and_isolate_gradients(
    head: CandidateIntensityTransformerHead,
) -> None:
    """Exercise shared buckets, power-of-two boundaries and noncontiguous spectrum rows."""
    counts = th.tensor([1, 3, 4, 5, 8, 9])
    batch = th.repeat_interleave(th.arange(len(counts)), counts)
    permutation = th.randperm(len(batch))
    batch = batch[permutation]
    tokens = th.randn(len(batch), 8, requires_grad=True)
    output = head._contextualize_by_molecule(tokens, batch, batch_size=len(counts))
    assert output.shape == tokens.shape and th.isfinite(output).all()
    for spectrum in range(len(counts)):
        keep = batch == spectrum
        singleton = head._contextualize_by_molecule(
            tokens[keep], th.zeros(int(counts[spectrum]), dtype=th.long), batch_size=1,
        )
        th.testing.assert_close(output[keep], singleton, atol=2e-6, rtol=1e-5)

    # Spectrum 1 shares its bucket with spectrum 2. Neither it nor other buckets may leak.
    keep = batch == 1
    probe = th.randn_like(output[keep])
    gradient = th.autograd.grad((output[keep] * probe).sum(), tokens)[0]
    assert th.isfinite(gradient).all()
    assert th.count_nonzero(gradient[~keep]) == 0
    assert gradient[keep].abs().sum() > 0
    isolated = tokens[keep].detach().requires_grad_()
    singleton = head._contextualize_by_molecule(
        isolated, th.zeros(3, dtype=th.long), batch_size=1,
    )
    isolated_gradient = th.autograd.grad((singleton * probe).sum(), isolated)[0]
    th.testing.assert_close(gradient[keep], isolated_gradient, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("unique_fragments", "attention_type"),
    [(False, "induced"), (False, "none"), (True, "induced"), (True, "full"), (True, "none")],
)
def test_mixed_molecules_match_singleton_spectra_after_batch_permutation(
    attention_type: str, unique_fragments: bool,
) -> None:
    """Graph offsets, metadata and candidate padding must preserve each spectrum."""
    smiles = ("[Na+]", "CCO", "c1ccccc1CC(=O)O")
    batch = _build_batch(smiles)
    model = _build_model(
        batch, max_masks=8, max_enumerated_masks=32,
        unique_fragment_mode=unique_fragments,
        intensity_transformer_attention=attention_type,
    ).eval()
    precursors = th.tensor([0, 1, 0])
    energies = th.tensor([5.0, 20.0, 60.0])
    with th.no_grad():
        singles = [
            model(
                mol_pyg=_build_batch((molecule,)),
                spec_prec_type=precursors[index:index + 1],
                spec_nce=energies[index:index + 1],
            )
            for index, molecule in enumerate(smiles)
        ]
        for order in [(0, 1, 2), (2, 0, 1)]:
            output = model(
                mol_pyg=_build_batch(tuple(smiles[index] for index in order)),
                spec_prec_type=precursors[list(order)],
                spec_nce=energies[list(order)],
            )
            batch_idxs = output["pred_batch_idxs"]
            assert (batch_idxs[1:] >= batch_idxs[:-1]).all()
            for position, original in enumerate(order):
                keep = batch_idxs == position
                expected = singles[original]
                assert keep.any()
                th.testing.assert_close(output["pred_mzs"][keep], expected["pred_mzs"])
                th.testing.assert_close(
                    output["pred_logprobs"][keep], expected["pred_logprobs"],
                    atol=2e-5, rtol=1e-5,
                )
