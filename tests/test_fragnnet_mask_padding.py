"""Batch independence of the fragment atom-membership (cc) mask.

The cc mask is ``MASK_SIZE`` bits wide; bits at or beyond a molecule's atom count are padding.
After per-molecule offsets are added, a padding position of one molecule indexes atoms of the
next molecule in the batch. Two paths reached padding bits:

- ``cc_interstage_use_rest`` pools ``~frag_node_mask`` (the fragment's complement);
- the ``cc_corruption_frac`` diagnostic flips every bit, including padding.

Both must give a molecule the same prediction alone and inside a batch.
"""

from __future__ import annotations

import pytest
import torch as th
import torch_geometric as pyg

from tests.test_cc_atom_dropout import _forward_kwargs
from tests.test_ce_multi_location import _build

SMALL, LARGE = "CCCO", "CCCCCC"


def _batch(smiles: list[str]) -> dict:
    """Collate single-molecule forward kwargs with cumulative offsets."""
    singles = [_forward_kwargs(s) for s in smiles]
    mol_sizes = th.tensor([int(k["mol_num_nodes"][-1]) for k in singles])
    frag_sizes = th.tensor([int(k["frag_num_nodes"][-1]) for k in singles])
    formula_sizes = th.cat([k["frag_formula_sizes"] for k in singles])
    zero = th.zeros(1, dtype=th.long)
    return {
        "mol_pyg": pyg.data.Batch.from_data_list([k["mol_pyg"].get_example(0) for k in singles]),
        "frag_pyg": pyg.data.Batch.from_data_list([k["frag_pyg"].get_example(0) for k in singles]),
        "mol_num_nodes": th.cat([zero, mol_sizes.cumsum(0)]),
        "frag_num_nodes": th.cat([zero, frag_sizes.cumsum(0)]),
        "frag_formula_sizes": formula_sizes,
        "frag_formula_cumsizes": th.cat([zero, formula_sizes.cumsum(0)]),
        "spec_ce_batch_idxs": th.arange(len(singles)),
        **{
            key: th.cat([k[key] for k in singles])
            for key in (
                "frag_formula_peak_idxs",
                "frag_formula_peak_mzs",
                "frag_formula_peak_probs",
                "frag_formula_peak_sizes",
                "spec_ce",
                "spec_prec_type",
                "spec_inst_type",
                "spec_frag_mode",
            )
        },
    }


def _first_spectrum(out: dict) -> th.Tensor:
    return out["pred_logprobs"][out["pred_batch_idxs"] == 0]


@pytest.mark.parametrize("order", [[SMALL, LARGE], [LARGE, SMALL]])
@pytest.mark.parametrize("mol_pool_type", ["mean", "max"])
def test_rest_pooling_is_batch_independent(order: list[str], mol_pool_type: str) -> None:
    th.manual_seed(0)
    model = _build(cc_interstage_use_rest=True, mol_pool_type=mol_pool_type).eval()
    with th.no_grad():
        alone = model(**_forward_kwargs(order[0]))["pred_logprobs"]
        batched = _first_spectrum(model(**_batch(order)))
    assert batched.shape == alone.shape
    assert th.allclose(batched, alone, atol=1e-5)


@pytest.mark.parametrize("mol_pool_type", ["mean", "max"])
def test_rest_pooling_keeps_trailing_empty_complement(mol_pool_type: str) -> None:
    """A one-atom molecule's only fragment is its root: empty complement in the last row."""
    th.manual_seed(0)
    model = _build(cc_interstage_use_rest=True, mol_pool_type=mol_pool_type).eval()
    with th.no_grad():
        alone = model(**_forward_kwargs(SMALL))["pred_logprobs"]
        model(**_forward_kwargs("C"))
        batched = _first_spectrum(model(**_batch([SMALL, "C"])))
    assert th.allclose(batched, alone, atol=1e-5)


def test_full_corruption_is_batch_independent() -> None:
    """``frac=1.0`` flips every bit deterministically, so batching must not change the result."""
    th.manual_seed(0)
    model = _build().eval()
    model.cc_corruption_frac = 1.0
    with th.no_grad():
        alone = model(**_forward_kwargs(SMALL))["pred_logprobs"]
        batched = _first_spectrum(model(**_batch([SMALL, LARGE])))
    assert th.allclose(batched, alone, atol=1e-5)
