"""Config validation record (reorganization plan Phase 4 step 2).

``runner.validate_sampler_config`` runs the shared checks and each model's hooks
(``validate_config`` for the model's own keys, ``reject_foreign_config`` from every other model).
This records the outcome of every single-key change to each model type's template config, and to
CE-context configs of MASSter and FraGNNet: a mutated value or a deleted key, giving "ok" or the
exception type and message. It was taken right after the checks moved into the hooks. An
exhaustive comparison against the pre-hook function gave identical outcomes for all 1080
variants but two. In a MASSter config that already uses its own sampler, turning on FraGNNet's
dynamic or dual-group sampler used to report "mutually exclusive" and now reports that the sampler
"can only be used with frag_gnn model". Every config is accepted or rejected as before. A config
with several problems may report a different first error.

Regenerate deliberately: REORG_UPDATE_GOLDEN=1 pytest tests/test_reorg_validation.py
"""

from __future__ import annotations

import copy

from reorg_golden import ratchet_json
from reorg_layers import MODEL_TYPES

from fragnnet.runner import load_config, validate_sampler_config

MUTATIONS = {
    "dynamic_batch_sampler": [False, True],
    "dual_group_dynamic_batch_sampler": [False, True],
    "masster_batch_sampler": [False, True, "yes"],
    "masster_share_proposal_across_ev": [False, True, 1],
    "masster_unique_fragment_mode": [False, True],
    "automatic_optimization": [True, False],
    "batch_sampler_type": ["spec_mol_frag", "ce_context", "bogus"],
    "dag_size_curriculum": [False, True],
    "bond_size_curriculum": [False, True],
    "cross_ev_fragment_loss_weight": [0.0, 0.5, -1, True],
    "ce_pair_loss_weight": [0.0, 0.5],
    "devices": [1, "auto"],
    "dynamic_batch_sampler_mode": ["frag_edge", "frag_node", "bad"],
    "dynamic_batch_sampler_max": [None, 1000],
    "masster_batch_sampler_max": [None, 8, 0, True],
    "dual_group_dynamic_batch_sampler_mode": ["frag_edge", "bad", None],
    "dual_group_dynamic_batch_sampler_max": [None, 500],
    "group_sampler": [False, True],
    "simple_group_sampler": [False, True],
    "simple_group_sampler_type": ["bogus", "mol"],
    "formula_prec_type_group_sampler": [False, True],
    "max_epochs": [0, 3, 100],
    "dag_size_curriculum_epochs": [1, 5, 200],
    "bond_size_curriculum_epochs": [1, 200],
    "ce_context_spectra_per_context": [1, 2, 4],
    "precursor_pair_loss_weight": [0.0, 0.3],
}
DELETABLE = (
    "masster_batch_sampler",
    "masster_unique_fragment_mode",
    "dual_group_dynamic_batch_sampler",
    "dynamic_batch_sampler",
    "cross_ev_fragment_loss_weight",
    "masster_share_proposal_across_ev",
    "dynamic_batch_sampler_mode",
    "batch_sampler_type",
)


def _outcome(config: dict) -> str:
    try:
        validate_sampler_config(copy.deepcopy(config))
    except Exception as error:  # noqa: BLE001 - the type and message are the record
        return f"{type(error).__name__}: {error}"
    return "ok"


def test_single_change_outcomes_match_golden() -> None:
    bases = {t: load_config("config", None, None, model_type=t) for t in MODEL_TYPES}
    # An unregistered type gets the default hooks and every model's foreign-key checks.
    bases["spectrum"] = {**copy.deepcopy(bases["neims"]), "model_type": "spectrum"}
    # CE-context configs reach the CE-window and CE-pair checks.
    bases["masster+ce_context"] = {
        **bases["masster"],
        "masster_batch_sampler": True,
        "masster_batch_sampler_max": 8,
        "devices": 1,
    }
    bases["frag_gnn+ce_context"] = {
        **bases["frag_gnn"],
        "dynamic_batch_sampler": True,
        "automatic_optimization": False,
        "dynamic_batch_sampler_mode": "frag_edge",
        "dynamic_batch_sampler_max": 1000,
        "batch_sampler_type": "ce_context",
        "devices": 1,
    }
    record = {}
    for model_type, base in bases.items():
        assert _outcome(base) == "ok", model_type
        outcomes = {}
        for key, values in MUTATIONS.items():
            for value in values:
                outcomes[f"{key}={value!r}"] = _outcome({**base, key: value})
        for key in DELETABLE:
            outcomes[f"del {key}"] = _outcome({k: v for k, v in base.items() if k != key})
        record[model_type] = outcomes
    ratchet_json("validation", "single_change", record)
