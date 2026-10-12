"""MASSter exports per-peak fragment annotation in FraGNNet's formula / joint / node layout.

MASSter has no DAG, so `masster_annotation_tables` groups its output pathways into the tables
FraGNNet's dataset supplies, and the exporter then filters, rebases and writes them unchanged.
A peak must resolve to a formula whose neutral formula plus adduct is the ion at that m/z, and
to fragments whose atom bitmasks decode to real atom sets of the molecule.
"""

import numpy as np
import pytest
import torch as th
from pyteomics.mass import Composition
from rdkit import Chem
from test_masster_model import _build_batch, _build_model

from fragnnet.frag import masster_export
from fragnnet.frag.masster_candidates import MassterStaticGraph
from fragnnet.frag.masster_templates import build_masster_templates
from fragnnet.models.fragnnet.frag_utils import th_long_to_mask
from fragnnet.pl_model import MassterPL
from fragnnet.utils import script_utils
from fragnnet.utils.formula_utils import ELECTRON_MASS, PERIODIC_TABLE
from fragnnet.utils.spec_utils import aggregate_peaks_exact_mz


def _ion_mz(neutral: str, delta: str, precursor_type: str) -> float:
    """Monoisotopic m/z of neutral + delta carrying the precursor's single charge."""
    counts = Composition(formula=neutral or "") + Composition(formula=delta or "")
    mass = sum(n * PERIODIC_TABLE.GetMostCommonIsotopeMass(el) for el, n in counts.items())
    return mass - ELECTRON_MASS if precursor_type.endswith("+") else mass + ELECTRON_MASS


def _atom_sets(ccs: th.Tensor) -> list[frozenset[int]]:
    return [frozenset(th.nonzero(row).flatten().tolist()) for row in th_long_to_mask(ccs)]


def _tables(pathway_peaks, peak_batch, ints, fragments, h_shifts, **overrides):
    """Tables for "CCO" (atoms 0-2, H 3/2/1) and "CO" (atoms 3-4, H 3/1), both [M+H]+.

    Fragments: 0 = {C, C, O}, 1 and 2 = {C, C} (one atom set reached twice), all in
    spectrum 0; 3 = {C, O} and 4 = {O} in spectrum 1.
    """
    kwargs = {
        "pathway_peak_idxs": th.tensor(pathway_peaks),
        "peak_batch_idxs": th.tensor(peak_batch),
        "pathway_ints": th.tensor(ints),
        "pathway_fragment_idxs": th.tensor(fragments),
        "pathway_ion_mode_idxs": th.zeros(len(fragments), dtype=th.long),
        "pathway_h_shifts": th.tensor(h_shifts),
        "fragment_batch_idxs": th.tensor([0, 0, 0, 1, 1]),
        "fragment_atom_indices": th.tensor([0, 1, 2, 0, 1, 0, 1, 3, 4, 4]),
        "fragment_atom_fragment_idxs": th.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 4]),
        "atom_atomic_numbers": th.tensor([6, 6, 8, 6, 8]),
        "atom_hydrogen_counts": th.tensor([3, 2, 1, 3, 1]),
        "atom_ptr": th.tensor([0, 3, 5]),
        "precursor_types": ["[M+H]+", "[M+H]+"],
        "return_node_ccs": True,
    }
    return masster_export.masster_annotation_tables(**(kwargs | overrides))


def test_tables_group_pathways_into_formulas_and_atom_set_nodes():
    # Spectrum 0: peak 0 holds fragments 1 and 2 (the same {C, C} set), peak 1 fragment 0.
    # Spectrum 1: peak 2 is fragment 4 at H shift -2, peak 3 fragment 3.
    tables = _tables(
        pathway_peaks=[0, 0, 1, 3, 2],
        peak_batch=[0, 0, 1, 1],
        ints=[0.2, 0.1, 0.7, 0.6, 0.4],
        fragments=[1, 2, 0, 3, 4],
        h_shifts=[0, 0, 0, 0, -2],
    )

    # {O} with one H at shift -2 would be the neutral "OH-1": the row carries the ion, "O".
    assert tables.formula_strs.tolist() == ["C2H5", "C2H6O", "O", "CH4O"]
    assert tables.formula_deltas.tolist() == ["H1", "H1", "", "H1"]
    th.testing.assert_close(tables.formula_logprobs.exp(), th.tensor([0.3, 0.7, 0.4, 0.6]))
    assert tables.formula_batch_idxs.tolist() == [0, 0, 1, 1]
    assert tables.formula_cumsizes.tolist() == [0, 2, 4]
    assert tables.frag_formula_peak_idxs.tolist() == [0, 1, 0, 1]
    assert tables.num_multi_formula_peaks == 0

    # Five fragments, four atom sets: the duplicate {C, C} pair shares one node.
    assert tables.node_cumsizes.tolist() == [0, 2, 4]
    node_sets = _atom_sets(tables.node_ccs)
    assert sorted(node_sets[:2], key=len) == [frozenset({0, 1}), frozenset({0, 1, 2})]
    assert sorted(node_sets[2:], key=len) == [frozenset({1}), frozenset({0, 1})]
    cells = {
        (tables.formula_strs[f], node_sets[n])
        for f, n in zip(tables.joint_formula_idxs.tolist(), tables.joint_node_idxs.tolist())
    }
    assert cells == {
        ("C2H5", frozenset({0, 1})),
        ("C2H6O", frozenset({0, 1, 2})),
        ("O", frozenset({1})),
        ("CH4O", frozenset({0, 1})),
    }
    th.testing.assert_close(tables.formula_node_logprobs.exp(), th.ones(4))
    node_probs = dict(zip(node_sets, tables.node_logprobs.exp().tolist()))
    assert node_probs[frozenset({0, 1, 2})] == pytest.approx(0.7)


def test_a_peak_with_two_formulas_links_its_most_intense_one():
    # Two formulas at one m/z do not occur in real molecules this small, so peak 0 is given
    # {C, C} (0.4) and {C, C, O} (0.6) by hand.
    tables = _tables(
        pathway_peaks=[0, 0, 1],
        peak_batch=[0, 1],
        ints=[0.4, 0.6, 1.0],
        fragments=[1, 0, 3],
        h_shifts=[0, 0, 0],
    )
    assert tables.num_multi_formula_peaks == 1
    assert tables.formula_strs.tolist() == ["C2H5", "C2H6O", "CH4O"]
    assert tables.frag_formula_peak_idxs.tolist() == [1, 0]


@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (500, 4), (2000, 19)])
def test_unique_rows_matches_torch_unique(shape):
    rows = th.randint(-3, 4, shape, generator=th.Generator().manual_seed(7))
    rows[:, 1] = 5  # a constant column
    expected_rows, expected_inverse = th.unique(rows, dim=0, return_inverse=True)
    unique_rows, inverse = masster_export._unique_rows(rows)
    assert th.equal(unique_rows, expected_rows)
    assert th.equal(inverse, expected_inverse)


def test_node_ccs_refuse_molecules_beyond_the_mask_size():
    with pytest.raises(ValueError, match="node cc bitmasks hold"):
        _tables(
            pathway_peaks=[0],
            peak_batch=[0],
            ints=[1.0],
            fragments=[0],
            h_shifts=[0],
            atom_ptr=th.tensor([0, 3, 300]),
        )


def _real_output(unique_fragment_mode: bool, precursor_type: str):
    smiles = ("CC(=O)Oc1ccccc1C(=O)O", "OCC(N)C(=O)O")
    batch = _build_batch(smiles)
    model = _build_model(
        batch,
        max_masks=6,
        unique_fragment_mode=unique_fragment_mode,
        max_enumerated_masks=12 if unique_fragment_mode else None,
        max_h_transfer=2,
        precursor_types=(precursor_type,),
    ).eval()
    with th.no_grad():
        output = model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0, 0]),
            spec_nce=th.tensor([20.0, 40.0]),
            return_annotation=True,
        )
    probs = output["pred_logprobs"].exp()
    totals = probs.new_zeros(2).index_add(0, output["pred_batch_idxs"], probs)
    output["pred_ints"] = probs / totals[output["pred_batch_idxs"]]
    return batch, model, output


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
@pytest.mark.parametrize("precursor_type", ["[M+H]+", "[M+Na]+", "[M-H]-"])
def test_real_pathways_reproduce_their_mz_from_formula_and_adduct(
    unique_fragment_mode: bool, precursor_type: str
):
    batch, model, output = _real_output(unique_fragment_mode, precursor_type)
    peak_mzs, _, peak_batch, pathway_peaks = aggregate_peaks_exact_mz(
        output["pred_mzs"], output["pred_ints"], output["pred_batch_idxs"], return_inverse=True
    )
    tables = masster_export.masster_annotation_tables(
        pathway_peak_idxs=pathway_peaks,
        peak_batch_idxs=peak_batch,
        pathway_ints=output["pred_ints"],
        pathway_fragment_idxs=output["pred_pathway_fragment_idxs"],
        pathway_ion_mode_idxs=output["pred_pathway_ion_mode_idxs"],
        pathway_h_shifts=output["pred_pathway_h_shifts"],
        fragment_batch_idxs=output["ion_h_fragment_batch_idxs"],
        fragment_atom_indices=output["fragment_atom_indices"],
        fragment_atom_fragment_idxs=output["fragment_atom_fragment_idxs"],
        atom_atomic_numbers=batch.atom_atomic_numbers,
        atom_hydrogen_counts=batch.atom_hydrogen_counts,
        atom_ptr=batch.ptr,
        precursor_types=[precursor_type, precursor_type],
        return_node_ccs=True,
    )
    if precursor_type == "[M+Na]+":
        # The CMF channel is live for this adduct, so both deltas appear.
        assert set(tables.formula_deltas.tolist()) >= {"Na1", "H1"}

    # Every peak's formula row is the ion at the peak's m/z.
    assert tables.num_multi_formula_peaks == 0
    for peak, local_row in enumerate(tables.frag_formula_peak_idxs.tolist()):
        row = local_row + int(tables.formula_cumsizes[peak_batch[peak]])
        expected = _ion_mz(tables.formula_strs[row], tables.formula_deltas[row], precursor_type)
        assert peak_mzs[peak].item() == pytest.approx(expected, abs=1e-5)

    # Every node is a real atom set of its molecule, and its heavy atoms are the formula's.
    atom_counts = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
    node_sets = _atom_sets(tables.node_ccs)
    for f, n in zip(tables.joint_formula_idxs.tolist(), tables.joint_node_idxs.tolist()):
        spectrum = int(tables.node_batch_idxs[n])
        atoms = node_sets[n]
        assert atoms and max(atoms) < atom_counts[spectrum]
        numbers = batch.atom_atomic_numbers[batch.ptr[spectrum] + th.tensor(sorted(atoms))]
        heavy = {
            PERIODIC_TABLE.GetElementSymbol(z): int((numbers == z).sum())
            for z in numbers.unique().tolist()
        }
        formula = Composition(formula=tables.formula_strs[f])
        assert {el: n for el, n in formula.items() if el != "H" and n} == heavy

    # P(f) sums to one per spectrum, and P(n|f) to one per formula.
    th.testing.assert_close(
        th.zeros(2).index_add(0, tables.formula_batch_idxs, tables.formula_logprobs.exp()),
        th.ones(2),
    )
    per_formula = th.zeros(len(tables.formula_strs)).index_add(
        0, tables.joint_formula_idxs, tables.formula_node_logprobs.exp()
    )
    th.testing.assert_close(per_formula, th.ones_like(per_formula))


class _ReplayMassterStub(MassterPL):
    """MassterPL-typed stub replaying a real model's inference output."""

    def __init__(self, batch_output: dict, precursor_types: tuple[str, ...]):
        self.training = False
        self._batch_output = batch_output
        self._precursor_types = precursor_types

    @property
    def hparams(self):
        return _AttrDict(
            auxiliary_scores=[],
            eval_mz_bin_res=[],
            eval_hun_sqrt=False,
            eval_bin_sqrt=False,
            sum_ints=True,
        )

    @property
    def model(self):
        return _AttrDict(precursor_types=self._precursor_types)

    def eval(self):
        return self

    def inference_step(self, batch_input, *, metrics=None):  # noqa: ARG002
        assert batch_input["return_explainability_tensors"] is True
        return dict(self._batch_output)


class _AttrDict(dict):
    def __getattr__(self, key):
        return self[key]


@pytest.mark.parametrize("min_peak_int", [0.0, 50.0])
def test_exporter_writes_resolvable_masster_annotation(min_peak_int: float):
    batch, _, output = _real_output(True, "[M+H]+")
    model = _ReplayMassterStub(output, ("[M+H]+",))
    batch_input = {
        "spec_unique_id": th.tensor([10, 11]),
        "mol_id": ["a", "b"],
        "batch_size": th.tensor(2),
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
    }
    output_set = {"pred_mzs", "pred_ints", "pred_formula_probs", "pred_formula_node_probs"}
    batch_idx_vars, node_idx_vars, formula_idx_vars, _ = script_utils.get_idx_vars(
        output_set, model, nb_iso=False, return_formula_str=True, return_cc_hash=True
    )
    columns = script_utils.get_tabular_columns(
        model.hparams, return_formula_str=True, return_cc_hash=True, output_subset=output_set
    )
    rows = script_utils.process_batch_to_tabular_row(
        batch_input=batch_input,
        model=model,
        device=th.device("cpu"),
        evaluate=False,
        eval_score_types=[],
        BATCH_IDX_VARS=batch_idx_vars,
        NODE_IDX_VARS=node_idx_vars,
        FORMULA_IDX_VARS=formula_idx_vars,
        min_peak_int=min_peak_int,
        rebase_indices=True,
        return_formula_str=True,
        return_cc_hash=True,
        output_set=output_set,
    )
    table = dict(zip(columns, rows))
    atom_counts = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
    for spectrum in range(2):
        record = {column: values[spectrum] for column, values in table.items()}
        mzs = np.asarray(record["pred_mzs"])
        ints = np.asarray(record["pred_ints"])
        assert len(mzs) > 0 and (ints > min_peak_int).all()
        peak_rows = np.asarray(record["frag_formula_peak_idxs"])
        strs, adducts = record["pred_formula_strs"], record["pred_formula_adducts"]
        assert len(strs) == len(adducts) == len(record["pred_formula_probs"])
        for mz, f in zip(mzs, peak_rows):
            delta = {"+H": "H1", "": ""}[adducts[f]]
            assert mz == pytest.approx(_ion_mz(strs[f], delta, "[M+H]+"), abs=1e-5)
        # Every node behind a surviving formula is written, and decodes to real atoms.
        ccs = th.as_tensor(np.asarray(record["pred_node_ccs"]))
        assert ccs.shape == (len(record["pred_node_node_idxs"]), 2)
        assert set(np.asarray(record["pred_joint_node_idxs"]).tolist()) == set(range(ccs.shape[0]))
        assert set(np.asarray(record["pred_joint_formula_idxs"]).tolist()) == set(range(len(strs)))
        assert all(atoms and max(atoms) < atom_counts[spectrum] for atoms in _atom_sets(ccs))


def test_tables_name_the_smarts_rules_behind_each_node():
    # Fragments 1 and 2 are one {C, C} node, so it carries both of their rules, in rule
    # order. Fragment 3 has no rule; no pathway reaches a rule pair's missing fragment.
    tables = _tables(
        pathway_peaks=[0, 0, 1, 3, 2],
        peak_batch=[0, 0, 1, 1],
        ints=[0.2, 0.1, 0.7, 0.6, 0.4],
        fragments=[1, 2, 0, 3, 4],
        h_shifts=[0, 0, 0, 0, -2],
        fragment_rule_fragment_idxs=th.tensor([2, 1, 1, 4]),
        fragment_rule_idxs=th.tensor([3, 1, 3, 2]),
    )

    node_keys = zip(tables.node_batch_idxs.tolist(), _atom_sets(tables.node_ccs))
    assert dict(zip(node_keys, tables.node_rules.tolist())) == {
        (0, frozenset({0, 1})): "aryl_sulfonamide_so2;n_nitramine_n2o",
        (0, frozenset({0, 1, 2})): "",
        (1, frozenset({1})): "aryl_sulfonate_so2",
        (1, frozenset({0, 1})): "",
    }


def test_tables_need_both_halves_of_the_rule_pairs():
    with pytest.raises(ValueError, match="go together"):
        _tables(
            pathway_peaks=[0],
            peak_batch=[0],
            ints=[1.0],
            fragments=[0],
            h_shifts=[0],
            fragment_rule_fragment_idxs=th.tensor([0]),
        )


def test_exporter_writes_the_smarts_rules_behind_each_node():
    # N-phenyl benzenesulfonamide matches aryl_sulfonamide_so2: its lost SO2 (atoms 0-2) and
    # its two-piece aryl product (atoms 3-15) both come from the rule. It also matches the
    # substituted O-transfer. Ethanol matches none.
    smiles = ("O=S(=O)(Nc1ccccc1)c1ccccc1", "CCO")
    batch = _build_batch(smiles)
    model = _build_model(
        batch,
        max_masks=6,
        unique_fragment_mode=True,
        max_enumerated_masks=16,
        use_smarts_templates=True,
        precursor_types=("[M+H]+",),
    ).eval()
    static_graph = MassterStaticGraph(
        chemical_bond_index=batch.chemical_bond_index,
        atom_batch_indices=batch.batch,
        atom_ptr=batch.ptr,
        smarts_templates=tuple(build_masster_templates(Chem.MolFromSmiles(s)) for s in smiles),
    )
    with th.no_grad():
        output = model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0, 0]),
            spec_nce=th.tensor([20.0, 40.0]),
            static_graph=static_graph,
            return_annotation=True,
        )
    probs = output["pred_logprobs"].exp()
    totals = probs.new_zeros(2).index_add(0, output["pred_batch_idxs"], probs)
    output["pred_ints"] = probs / totals[output["pred_batch_idxs"]]
    stub = _ReplayMassterStub(output, ("[M+H]+",))
    batch_input = {
        "spec_unique_id": th.tensor([10, 11]),
        "mol_id": ["a", "b"],
        "batch_size": th.tensor(2),
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
    }
    output_set = {"pred_mzs", "pred_ints", "pred_node_probs"}
    batch_idx_vars, node_idx_vars, formula_idx_vars, _ = script_utils.get_idx_vars(
        output_set, stub, nb_iso=False, return_formula_str=False, return_cc_hash=True
    )
    columns = script_utils.get_tabular_columns(
        stub.hparams,
        return_cc_hash=True,
        output_subset=output_set,
        return_rule_provenance=True,
    )
    rows = script_utils.process_batch_to_tabular_row(
        batch_input=batch_input,
        model=stub,
        device=th.device("cpu"),
        evaluate=False,
        eval_score_types=[],
        BATCH_IDX_VARS=batch_idx_vars,
        NODE_IDX_VARS=node_idx_vars,
        FORMULA_IDX_VARS=formula_idx_vars,
        min_peak_int=0.0,
        rebase_indices=True,
        return_formula_str=False,
        return_cc_hash=True,
        output_set=output_set,
        return_rule_provenance=True,
    )

    table = dict(zip(columns, rows))
    records = [
        {column: values[spectrum] for column, values in table.items()} for spectrum in (0, 1)
    ]
    node_rules = [
        dict(zip(_atom_sets(th.as_tensor(np.asarray(r["pred_node_ccs"]))), r["pred_node_rules"]))
        for r in records
    ]
    so2, product = frozenset(range(3)), frozenset(range(3, 16))
    assert node_rules[0][so2] == node_rules[0][product] == "aryl_sulfonamide_so2"
    # The secondary sulfonamide also matches substituted_aryl_sulfonamide_o_transfer ([M+H]+).
    assert {rules for atoms, rules in node_rules[0].items() if atoms not in (so2, product)} <= {
        "",
        "substituted_aryl_sulfonamide_o_transfer",
    }
    assert set(node_rules[1].values()) == {""}
