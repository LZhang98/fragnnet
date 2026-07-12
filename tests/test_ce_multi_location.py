"""Tests for multi-location CE injection (ce_insert_location = "mol,mlp").

The model dispatches CE insertion on membership in a parsed set of locations. The key
regression guarded here is the silent-drop bug: before multi-location support, the code used
exact string matches (== "mol" / == "mlp"), so a comma-separated "mol,mlp" matched NEITHER
branch and dropped CE entirely (ce_mol_input_dim == ce_mlp_input_dim == 0). These tests assert
that each location string produces the correct nonzero input dims at the mol encoder and/or the
per-fragment scoring MLP, so a future regression that stops parsing the combo fails loudly.
"""

from __future__ import annotations

import pytest

from fragnnet.model.fragnnet_model import FraGNNetModel


def _build(ce_insert_location: str) -> FraGNNetModel:
    """Smallest valid FraGNNetModel with the given ce_insert_location."""
    return FraGNNetModel(
        num_depth=4,
        num_hs=2,
        num_elements=17,
        int_embedder="fourier",
        int_embedder_tight=True,
        mol_node_feats=["a_onehot", "a_degree", "a_ring", "a_mass"],
        mol_edge_feats=["b_degree"],
        mol_pe_embed_k=0,
        mol_hidden_size=32,
        mol_num_layers=2,
        mol_gnn_type="MPNN",
        mol_normalization="layer",
        mol_dropout=0.0,
        mol_num_heads=1,
        mol_pool_type="mean",
        frag_node_feats=["cc", "base_formula", "depth", "h_formulae_idx"],
        frag_edge_feats=[],
        frag_hidden_size=32,
        frag_num_layers=2,
        frag_gnn_type="NodeMLP",
        frag_normalization="layer",
        frag_dropout=0.0,
        frag_pool_type="max",
        frag_embed_combine="cat",
        frag_pool_combine="none",
        mlp_output_format="formula",
        mlp_hidden_size=32,
        mlp_normalization="layer",
        mlp_dropout=0.0,
        mlp_num_layers=2,
        mlp_use_residuals=True,
        cc_interstage_type="add",
        cc_interstage_use_rest=False,
        nb_iso=False,
        skip_edge_loss=True,
        mask_null_formula=False,
        predict_oos=False,
        bin_output=False,
        mz_bin_res=0.01,
        mz_max=1500.0,
        ce_insert_location=ce_insert_location,
        ce_insert_type="embed_clamped",
        ce_insert_merge=True,
        ce_insert_size=8,
        nce_max=200.0,
        nce_mean=30.0,
        nce_std=20.0,
        prec_insert_location="mol",
        prec_insert_size=8,
        prec_types=["[M+H]+"],
        inst_insert_location="mol",
        inst_insert_size=8,
        inst_types=["FT"],
        frag_mode_insert_location="mol",
        frag_mode_insert_size=8,
        frag_modes=["HCD"],
        frag_mode_scale=False,
        ce_scale=False,
        ce_scaler_hidden_dim=16,
        adduct_scale=False,
        stratum_scale=False,
        stratum_scaler_embed_dim=8,
        output_formula_str=False,
        use_nce=True,
        use_ace=False,
        ace_max=50.0,
        ace_mean=20.0,
        ace_std=10.0,
        cc_feature_dropout=0.0,
    )


class TestCeInsertLocationParsing:
    def test_none_disables_ce(self):
        m = _build("none")
        assert m.ce_insert_locations == set()
        assert m.ce_mol_input_dim == 0
        assert m.ce_mlp_input_dim == 0

    def test_mol_only(self):
        m = _build("mol")
        assert m.ce_insert_locations == {"mol"}
        assert m.ce_mol_input_dim == m.ce_embed_dim > 0
        assert m.ce_mlp_input_dim == 0

    def test_mlp_only(self):
        m = _build("mlp")
        assert m.ce_insert_locations == {"mlp"}
        assert m.ce_mol_input_dim == 0
        assert m.ce_mlp_input_dim == m.ce_embed_dim > 0

    def test_mol_and_mlp_both_active(self):
        """The silent-drop guard: 'mol,mlp' must widen BOTH inputs, not drop CE."""
        m = _build("mol,mlp")
        assert m.ce_insert_locations == {"mol", "mlp"}
        assert m.ce_mol_input_dim == m.ce_embed_dim > 0
        assert m.ce_mlp_input_dim == m.ce_embed_dim > 0

    def test_whitespace_tolerated(self):
        m = _build("mol, mlp")
        assert m.ce_insert_locations == {"mol", "mlp"}
        assert m.ce_mol_input_dim > 0 and m.ce_mlp_input_dim > 0

    def test_mlp_widens_nodemlp_input_vs_mol_only(self):
        """Adding mlp injection must enlarge the actual per-fragment scoring MLP input layer by
        the CE dim -- an end-to-end check that CE reaches the scorer, not just bookkeeping."""
        mol_only = _build("mol")
        both = _build("mol,mlp")
        assert (
            both.formula_module.input_layer.in_features
            - mol_only.formula_module.input_layer.in_features
            == both.ce_embed_dim
        )

    def test_invalid_location_raises(self):
        with pytest.raises(ValueError, match="Invalid ce_insert_location"):
            _build("mol,bogus")
