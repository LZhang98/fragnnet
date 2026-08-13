"""Tests for runner config validation helpers."""

import pytest

from fragnnet.runner import validate_sampler_config


def _base_config(**overrides) -> dict:
    """Return a minimal config dict for validate_sampler_config tests."""
    cfg = {
        "dynamic_batch_sampler": True,
        "dual_group_dynamic_batch_sampler": False,
        "dual_group_dynamic_batch_sampler_mode": "frag_node",
        "dual_group_dynamic_batch_sampler_max": 1000,
        "model_type": "frag_gnn",
        "automatic_optimization": False,
        "dynamic_batch_sampler_mode": "frag_node",
        "dynamic_batch_sampler_max": 1000,
        "batch_sampler_type": "spec_mol_frag",
        "ce_context_spectra_per_context": 5,
        "ce_context_ce_key_by_inst": {"FT": "nce", "QTOF": "ace", "IT": "nce"},
        "devices": 1,
        "dag_size_curriculum": False,
        "dag_size_curriculum_start_fraction": 0.25,
        "dag_size_curriculum_epochs": 10,
        "max_epochs": 100,
        "simple_group_sampler": False,
        "simple_group_sampler_type": "adduct_inst_type",
        "ce_pair_loss_weight": 0.0,
        "ce_pair_min_endpoint_confidence": None,
        "ce_pair_min_endpoint_reliability": None,
        "ce_pair_subset": "switches",
        "ce_pair_eligible_inst_types": ["FT"],
        "ce_pair_eligible_prec_types": ["[M+H]+"],
        "precursor_pair_loss_weight": 0.0,
        "precursor_pair_nce_min": 0.0,
        "precursor_pair_nce_max": 20.0,
        "precursor_pair_absent_intensity_floor": 0.01,
        "precursor_pair_eligible_inst_types": ["FT"],
        "precursor_pair_eligible_prec_types": ["[M+H]+"],
        "target_rel_ints_thresh": 0.01,
        "train_sample_weight": "none",
        "inst_type_loss_tol": {"FT": {"rel": 1.0e-5, "min_mz": 200.0}},
        "spec_params": {
            "merge": False,
            "nce": True,
            "ace": False,
            "inst_types": ["FT", "QTOF", "IT"],
            "prec_types": ["[M+H]+", "[M-H]-"],
        },
    }
    cfg.update(overrides)
    return cfg


class TestValidateSamplerConfig:
    def test_missing_dual_group_key_raises(self):
        """Old configs without dual_group_dynamic_batch_sampler should fail fast."""
        cfg = _base_config()
        del cfg["dual_group_dynamic_batch_sampler"]
        with pytest.raises(KeyError, match="dual_group_dynamic_batch_sampler"):
            validate_sampler_config(cfg)

    def test_disabled_sampler_always_passes(self):
        # dynamic_batch_sampler=False should skip all checks regardless of other values
        cfg = _base_config(
            dynamic_batch_sampler=False,
            model_type="spectrum",  # would fail if checks ran
            dynamic_batch_sampler_mode="invalid",
            dynamic_batch_sampler_max=None,
        )
        validate_sampler_config(cfg)  # must not raise

    def test_frag_node_mode_accepted(self):
        validate_sampler_config(_base_config(dynamic_batch_sampler_mode="frag_node"))

    def test_frag_edge_mode_accepted(self):
        # NodeMLP (or any frag_gnn) must accept frag_edge budget mode
        validate_sampler_config(_base_config(dynamic_batch_sampler_mode="frag_edge"))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="frag_node or frag_edge"):
            validate_sampler_config(_base_config(dynamic_batch_sampler_mode="invalid_mode"))

    def test_non_frag_gnn_model_raises(self):
        with pytest.raises(ValueError, match="frag_gnn"):
            validate_sampler_config(_base_config(model_type="spectrum"))

    def test_automatic_optimization_raises(self):
        with pytest.raises(ValueError, match="automatic optimization"):
            validate_sampler_config(_base_config(automatic_optimization=True))

    def test_missing_max_raises(self):
        with pytest.raises(ValueError, match="dynamic_batch_sampler_max"):
            validate_sampler_config(_base_config(dynamic_batch_sampler_max=None))

    def test_mutually_exclusive_dynamic_and_dual_group_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_sampler_config(
                _base_config(
                    dynamic_batch_sampler=True,
                    dual_group_dynamic_batch_sampler=True,
                )
            )

    def test_ce_context_batch_sampler_valid_config_passes(self):
        validate_sampler_config(
            _base_config(
                batch_sampler_type="ce_context",
                dynamic_batch_sampler_mode="frag_edge",
            )
        )

    def test_ce_context_batch_sampler_requires_frag_edge_budget(self):
        with pytest.raises(ValueError, match="dynamic_batch_sampler_mode='frag_edge'"):
            validate_sampler_config(
                _base_config(
                    batch_sampler_type="ce_context",
                    dynamic_batch_sampler_mode="frag_node",
                )
            )

    def test_ce_context_batch_sampler_requires_dynamic_sampler(self):
        with pytest.raises(ValueError, match="requires dynamic_batch_sampler"):
            validate_sampler_config(
                _base_config(
                    dynamic_batch_sampler=False,
                    batch_sampler_type="ce_context",
                )
            )

    def test_ce_context_batch_sampler_rejects_small_window(self):
        with pytest.raises(ValueError, match="ce_context_spectra_per_context"):
            validate_sampler_config(
                _base_config(
                    batch_sampler_type="ce_context",
                    dynamic_batch_sampler_mode="frag_edge",
                    ce_context_spectra_per_context=1,
                )
            )

    @pytest.mark.parametrize(
        "ce_key_by_inst",
        [None, {}, {"FT": "invalid"}, {1: "nce"}],
    )
    def test_ce_context_batch_sampler_rejects_invalid_ce_mapping(self, ce_key_by_inst):
        with pytest.raises(ValueError, match="ce_context_ce_key_by_inst"):
            validate_sampler_config(
                _base_config(
                    batch_sampler_type="ce_context",
                    dynamic_batch_sampler_mode="frag_edge",
                    ce_context_ce_key_by_inst=ce_key_by_inst,
                )
            )

    def test_ce_context_batch_sampler_rejects_multiple_devices(self):
        with pytest.raises(ValueError, match="devices=1"):
            validate_sampler_config(
                _base_config(
                    batch_sampler_type="ce_context",
                    dynamic_batch_sampler_mode="frag_edge",
                    devices=2,
                )
            )

    def test_ce_context_batch_sampler_requires_all_configured_instruments(self):
        with pytest.raises(ValueError, match="missing configured instrument types"):
            validate_sampler_config(
                _base_config(
                    batch_sampler_type="ce_context",
                    dynamic_batch_sampler_mode="frag_edge",
                    ce_context_ce_key_by_inst={"FT": "nce", "QTOF": "ace"},
                )
            )

    def test_invalid_batch_sampler_type_raises(self):
        with pytest.raises(ValueError, match="batch_sampler_type"):
            validate_sampler_config(_base_config(batch_sampler_type="invalid"))

    def test_ce_pair_loss_valid_config_passes(self):
        validate_sampler_config(
            _base_config(
                batch_sampler_type="ce_context",
                dynamic_batch_sampler_mode="frag_edge",
                ce_pair_loss_weight=0.1,
                ce_pair_min_endpoint_confidence=0.2,
            )
        )

    def test_ce_pair_loss_requires_context_sampler(self):
        with pytest.raises(ValueError, match="batch_sampler_type='ce_context'"):
            validate_sampler_config(
                _base_config(
                    ce_pair_loss_weight=0.1,
                    ce_pair_min_endpoint_confidence=0.2,
                )
            )

    def test_ce_pair_loss_requires_confidence_threshold(self):
        with pytest.raises(ValueError, match="ce_pair_min_endpoint_confidence"):
            validate_sampler_config(
                _base_config(
                    batch_sampler_type="ce_context",
                    dynamic_batch_sampler_mode="frag_edge",
                    ce_pair_loss_weight=0.1,
                )
            )

    @pytest.mark.parametrize("contender_floor", [0.0, -0.1, 1.0, None])
    def test_ce_pair_loss_requires_positive_contender_floor(self, contender_floor):
        with pytest.raises(ValueError, match="target_rel_ints_thresh"):
            validate_sampler_config(
                _base_config(
                    batch_sampler_type="ce_context",
                    dynamic_batch_sampler_mode="frag_edge",
                    ce_pair_loss_weight=0.1,
                    ce_pair_min_endpoint_confidence=0.2,
                    target_rel_ints_thresh=contender_floor,
                )
            )

    @pytest.mark.parametrize("confidence", [0.0, -0.1, 1.1])
    def test_ce_pair_loss_rejects_invalid_confidence(self, confidence):
        with pytest.raises(ValueError, match="ce_pair_min_endpoint_confidence"):
            validate_sampler_config(
                _base_config(
                    batch_sampler_type="ce_context",
                    dynamic_batch_sampler_mode="frag_edge",
                    ce_pair_loss_weight=0.1,
                    ce_pair_min_endpoint_confidence=confidence,
                )
            )

    def test_ce_pair_loss_rejects_unimplemented_reliability_threshold(self):
        with pytest.raises(ValueError, match="reliability table is implemented"):
            validate_sampler_config(
                _base_config(
                    batch_sampler_type="ce_context",
                    dynamic_batch_sampler_mode="frag_edge",
                    ce_pair_loss_weight=0.1,
                    ce_pair_min_endpoint_confidence=0.2,
                    ce_pair_min_endpoint_reliability=0.7,
                )
            )

    @pytest.mark.parametrize("tolerance_by_inst", [None, {}, {"QTOF": {}}])
    def test_ce_pair_loss_requires_eligible_instrument_tolerance(self, tolerance_by_inst):
        with pytest.raises(ValueError, match="define every ce_pair_eligible_inst_type"):
            validate_sampler_config(
                _base_config(
                    batch_sampler_type="ce_context",
                    dynamic_batch_sampler_mode="frag_edge",
                    ce_pair_loss_weight=0.1,
                    ce_pair_min_endpoint_confidence=0.2,
                    inst_type_loss_tol=tolerance_by_inst,
                )
            )

    @pytest.mark.parametrize(
        "tolerance",
        [{}, {"rel": 1.0e-5}, {"rel": 0.0, "min_mz": 200.0}],
    )
    def test_ce_pair_loss_rejects_invalid_eligible_tolerance(self, tolerance):
        with pytest.raises(ValueError, match="inst_type_loss_tol"):
            validate_sampler_config(
                _base_config(
                    batch_sampler_type="ce_context",
                    dynamic_batch_sampler_mode="frag_edge",
                    ce_pair_loss_weight=0.1,
                    ce_pair_min_endpoint_confidence=0.2,
                    inst_type_loss_tol={"FT": tolerance},
                )
            )

    def test_precursor_pair_loss_valid_config_passes(self):
        validate_sampler_config(_base_config(precursor_pair_loss_weight=0.1))

    @pytest.mark.parametrize(
        "spec_params_override, error",
        [
            ({"nce": False, "ace": False}, "requires spec_params.nce or spec_params.ace"),
            ({"merge": True}, "requires unmerged spectra"),
        ],
    )
    def test_precursor_pair_loss_rejects_missing_per_spectrum_nce(
        self, spec_params_override, error
    ):
        config = _base_config(precursor_pair_loss_weight=0.1)
        config["spec_params"].update(spec_params_override)

        with pytest.raises(ValueError, match=error):
            validate_sampler_config(config)

    @pytest.mark.parametrize("nce_interval", [(-1.0, 20.0), (20.0, 20.0), (30.0, 20.0)])
    def test_precursor_pair_loss_rejects_invalid_nce_interval(self, nce_interval):
        with pytest.raises(ValueError, match="precursor_pair_nce_min/max"):
            validate_sampler_config(
                _base_config(
                    precursor_pair_loss_weight=0.1,
                    precursor_pair_nce_min=nce_interval[0],
                    precursor_pair_nce_max=nce_interval[1],
                )
            )

    @pytest.mark.parametrize("absent_floor", [-0.1, 1.0, None])
    def test_precursor_pair_loss_rejects_invalid_absent_floor(self, absent_floor):
        with pytest.raises(ValueError, match="precursor_pair_absent_intensity_floor"):
            validate_sampler_config(
                _base_config(
                    precursor_pair_loss_weight=0.1,
                    precursor_pair_absent_intensity_floor=absent_floor,
                )
            )

    @pytest.mark.parametrize(
        "tolerance",
        [{}, {"rel": 1.0e-5}, {"rel": 0.0, "min_mz": 200.0}],
    )
    def test_precursor_pair_loss_rejects_invalid_tolerance(self, tolerance):
        with pytest.raises(ValueError, match="tolerance"):
            validate_sampler_config(
                _base_config(
                    precursor_pair_loss_weight=0.1,
                    inst_type_loss_tol={"FT": tolerance},
                )
            )

    def test_dag_size_curriculum_valid_config_passes(self):
        validate_sampler_config(_base_config(dag_size_curriculum=True))

    def test_dag_size_curriculum_requires_dynamic_sampler(self):
        with pytest.raises(ValueError, match="requires dynamic_batch_sampler"):
            validate_sampler_config(
                _base_config(dynamic_batch_sampler=False, dag_size_curriculum=True)
            )

    def test_dag_size_curriculum_rejects_automatic_optimization(self):
        with pytest.raises(ValueError, match="automatic optimization"):
            validate_sampler_config(
                _base_config(dag_size_curriculum=True, automatic_optimization=True)
            )

    @pytest.mark.parametrize("max_epochs", [1, 10])
    def test_dag_size_curriculum_must_reach_full_dataset(self, max_epochs):
        with pytest.raises(ValueError, match="less than max_epochs"):
            validate_sampler_config(_base_config(dag_size_curriculum=True, max_epochs=max_epochs))

    @pytest.mark.parametrize("start_fraction", [0, -0.1, 1.1, None])
    def test_dag_size_curriculum_rejects_invalid_start_fraction(self, start_fraction):
        with pytest.raises(ValueError, match="start_fraction"):
            validate_sampler_config(
                _base_config(
                    dag_size_curriculum=True,
                    dag_size_curriculum_start_fraction=start_fraction,
                )
            )

    @pytest.mark.parametrize("curriculum_epochs", [0, -1, 1.5, None])
    def test_dag_size_curriculum_rejects_invalid_epochs(self, curriculum_epochs):
        with pytest.raises(ValueError, match="curriculum_epochs"):
            validate_sampler_config(
                _base_config(
                    dag_size_curriculum=True,
                    dag_size_curriculum_epochs=curriculum_epochs,
                )
            )

    def test_dual_group_sampler_valid_config_passes(self):
        validate_sampler_config(
            _base_config(
                dynamic_batch_sampler=False,
                dual_group_dynamic_batch_sampler=True,
                dual_group_dynamic_batch_sampler_mode="frag_node",
                dual_group_dynamic_batch_sampler_max=1000,
            )
        )

    def test_dual_group_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="frag_node or frag_edge"):
            validate_sampler_config(
                _base_config(
                    dynamic_batch_sampler=False,
                    dual_group_dynamic_batch_sampler=True,
                    dual_group_dynamic_batch_sampler_mode="invalid_mode",
                )
            )

    def test_dual_group_missing_max_raises(self):
        with pytest.raises(ValueError, match="dual_group_dynamic_batch_sampler_max"):
            validate_sampler_config(
                _base_config(
                    dynamic_batch_sampler=False,
                    dual_group_dynamic_batch_sampler=True,
                    dual_group_dynamic_batch_sampler_max=None,
                )
            )

    def test_dual_group_non_frag_gnn_model_raises(self):
        with pytest.raises(ValueError, match="frag_gnn"):
            validate_sampler_config(
                _base_config(
                    dynamic_batch_sampler=False,
                    dual_group_dynamic_batch_sampler=True,
                    model_type="spectrum",
                )
            )

    def test_dual_group_automatic_optimization_raises(self):
        with pytest.raises(ValueError, match="automatic optimization"):
            validate_sampler_config(
                _base_config(
                    dynamic_batch_sampler=False,
                    dual_group_dynamic_batch_sampler=True,
                    automatic_optimization=True,
                )
            )


class TestSimpleGroupSamplerTypeValidation:
    @pytest.mark.parametrize(
        "sampler_type",
        [
            "group",
            "mol",
            "group_mol",
            "adduct_inst_type",
            "adduct_inst_type_group",
            "adduct_inst_type_frag_mode",
            "adduct_inst_type_frag_mode_group",
        ],
    )
    def test_valid_types_pass(self, sampler_type):
        validate_sampler_config(
            _base_config(simple_group_sampler=True, simple_group_sampler_type=sampler_type)
        )

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="simple_group_sampler_type"):
            validate_sampler_config(
                _base_config(simple_group_sampler=True, simple_group_sampler_type="bad_type")
            )

    def test_disabled_sampler_skips_type_check(self):
        """simple_group_sampler=False should not validate simple_group_sampler_type."""
        validate_sampler_config(
            _base_config(simple_group_sampler=False, simple_group_sampler_type="bad_type")
        )
