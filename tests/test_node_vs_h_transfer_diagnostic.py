import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "eval" / "node_vs_h_transfer_diagnostic.py"
SPEC = importlib.util.spec_from_file_location(
    "node_vs_h_transfer_diagnostic",
    SCRIPT_PATH,
)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DIAGNOSTIC)


@pytest.fixture
def sample_ranking() -> dict:
    return {
        "formula_rank": {10: 1, 11: 2, 12: 3},
        "formula_prob": {10: 0.7, 11: 0.2, 12: 0.1},
        "node_rank": {100: 2, 101: 1, 102: 3},
        "node_prob": {100: 0.1, 101: 0.8, 102: 0.1},
        "joint_positions_by_formula": {10: [0, 3], 11: [1], 12: [2]},
        "joint_node_idxs": torch.tensor([100, 101, 102, 103]),
        "joint_ranks": torch.tensor([1, 2, 3, 4]),
        "node_formula_probs": torch.tensor([0.8, 0.7, 0.4, 0.2]),
    }


@pytest.mark.parametrize(
    ("matched_formula_idxs", "node_rank_k", "formula_rank_k", "expected"),
    [
        (set(), 1, 1, "coverage"),
        ({10}, 1, 1, "node"),
        ({11}, 1, 1, "h_formula"),
        ({12}, 1, 1, "both_low"),
        ({11}, 2, 2, "both_ok"),
    ],
)
def test_classify_peak_bottleneck_categories(
    sample_ranking: dict,
    matched_formula_idxs: set[int],
    node_rank_k: int,
    formula_rank_k: int,
    expected: str,
) -> None:
    result = DIAGNOSTIC._classify_peak(
        matched_formula_idxs=matched_formula_idxs,
        node_rank_k=node_rank_k,
        formula_rank_k=formula_rank_k,
        **sample_ranking,
    )

    assert result["bottleneck"] == expected


def test_classify_peak_uses_precomputed_joint_support(sample_ranking: dict) -> None:
    result = DIAGNOSTIC._classify_peak(
        matched_formula_idxs={10},
        node_rank_k=2,
        formula_rank_k=1,
        **sample_ranking,
    )

    assert result["support_node_count"] == 2
    assert result["best_joint_rank"] == 1.0
    assert result["best_p_f_given_n"] == pytest.approx(0.8)
    assert result["top_support_node_p_f_given_n"] == pytest.approx(0.8)
