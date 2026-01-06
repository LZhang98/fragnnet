import sys
import numpy as np
from unittest.mock import MagicMock
import torch as th

# Mock modules unconditionally to avoid import errors from deep dependencies
sys.modules["fragnnet.frag"] = MagicMock()
compute_frags_mock = MagicMock()
sys.modules["fragnnet.frag.compute_frags"] = compute_frags_mock
# Mock attributes
compute_frags_mock.MASK_DTYPE = np.int64
compute_frags_mock.MAX_NUM_EDGES = 100
compute_frags_mock.MAX_NUM_NODES = 100

from fragnnet.model.loss import get_pairwise_jss_sim
from fragnnet.utils.misc_utils import EPS

def test_pairwise_jss_hun_basic():
    """
    Test basic properties of pairwise JSS Hungarian similarity:
    - Shape
    - Self-similarity = 1.0
    - Symmetry
    - Known value checks
    """
    batch_size = 3
    
    # Spec 0: Mass 10 (1.0)
    # Spec 1: Mass 10 (1.0) -> Identical to 0
    # Spec 2: Mass 20 (1.0) -> No overlap with 0 or 1
    
    mzs = th.tensor([10.0, 10.0, 20.0])
    logprobs = th.tensor([0.0, 0.0, 0.0]) # exp(0)=1
    batch_idxs = th.tensor([0, 1, 2])
    
    sim_matrix = get_pairwise_jss_sim(
        mzs, logprobs, batch_idxs,
        batch_size=batch_size,
        mz_max=100.0, mz_bin_res=1.0, chunk_size=None
    )
    
    print("Sim Matrix:\n", sim_matrix)
    
    # Check shape
    assert sim_matrix.shape == (batch_size, batch_size)
    
    # Check self-similarity
    assert th.allclose(sim_matrix.diag(), th.ones(batch_size), atol=1e-5)
    
    # Check symmetry
    assert th.allclose(sim_matrix, sim_matrix.t(), atol=1e-5)
    
    # Check 0 vs 1 (Identical)
    assert th.isclose(sim_matrix[0, 1], th.tensor(1.0), atol=1e-5)
    
    # Check 0 vs 2 (No overlap)
    # JSD should be max?
    # JSD of completely disjoint distributions P and Q:
    # M = 0.5(P+Q)
    # H(P)=0 (since 1 peak with p=1), H(Q)=0
    # M has 2 peaks with prop 0.5 each.
    # H(M) = - (0.5 log 0.5 + 0.5 log 0.5) = - log 0.5 = log 2
    # JSD = H(M) - 0.5 H(P) - 0.5 H(Q) = log 2
    # Normalized JSD = JSD / log 2 = 1.0
    # Similarity = 1 - NormJSD = 0.0
    # 
    # BUT, Hungarian matching logic:
    # If no matches, score calculation might vary.
    # If no matches, total_gain = 0.
    # JSD = 1 - 0 = 1.
    # Sim = 0.
    assert th.isclose(sim_matrix[0, 2], th.tensor(0.0), atol=1e-5)

def test_pairwise_jss_hun_partial_overlap():
    """
    Test partial overlap logic.
    Spec 0: 10 (0.5), 20 (0.5)
    Spec 1: 10 (0.5), 30 (0.5)
    Match on 10.
    """
    batch_size = 2
    
    mzs = th.tensor([10.0, 20.0, 10.0, 30.0])
    # logprobs corresponding to probs 0.5, 0.5 -> log(0.5) ~= -0.693
    logprobs = th.full((4,), th.log(th.tensor(0.5)))
    batch_idxs = th.tensor([0, 0, 1, 1])
    
    sim_matrix = get_pairwise_jss_sim(
        mzs, logprobs, batch_idxs,
        batch_size=2,
        mz_max=100.0, mz_bin_res=1.0, chunk_size=None
    )
    
    print("Partial Sim Matrix:\n", sim_matrix)
    
    # Manual Calc:
    # P: 10 (0.5), 20 (0.5)
    # Q: 10 (0.5), 30 (0.5)
    # Match: 10 with 10.
    # Gain for (10, 10):
    # p=0.5, q=0.5 -> sum=1.0
    # term1 = 1.0 * log(1.0) = 0
    # term2 = 0.5 * log(0.5)
    # term3 = 0.5 * log(0.5)
    # score = 0.5 * (0 - 2 * 0.5 * log(0.5)) = - 0.5 * log(0.5) = 0.5 * log(2)
    # 20 and 30 unmatched -> score -1e9 (invalid)
    # Total Gain = 0.5 * log(2)
    # Norm JSD = 1 - (0.5 * log 2) / log 2 = 1 - 0.5 = 0.5
    # Sim = 1 - 0.5 = 0.5
    
    assert th.isclose(sim_matrix[0, 1], th.tensor(0.5), atol=1e-5)

# Tests are collected by pytest; no direct execution block required.
