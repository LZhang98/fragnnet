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

from fragnnet.model.loss import get_pairwise_cross_entropy_hun
from fragnnet.utils.misc_utils import EPS

def test_pairwise_ce_hun_basic():
    """
    Test basic properties of pairwise CE Hungarian:
    - Shape
    - Self-Consistency: CE(P, P) = Entropy(P)
    """
    print("Running test_pairwise_ce_hun_basic...")
    batch_size = 3
    
    # Spec 0: Mass 10 (1.0). Entropy = 0.
    # Spec 1: Mass 10 (0.5), Mass 20 (0.5). Entropy = log(2) ~= 0.693
    # Spec 2: Mass 30 (1.0). Disjoint from 0.
    
    # Mzs
    mzs = th.tensor([10.0, 10.0, 20.0, 30.0], dtype=th.float32)
    # Logprobs
    # 0: log(1)=0
    # 1: log(0.5)
    # 2: log(1)=0
    logprobs = th.tensor([0.0, np.log(0.5), np.log(0.5), 0.0], dtype=th.float32)
    
    # Batch Idxs
    batch_idxs = th.tensor([0, 1, 1, 2])
    
    ce_matrix = get_pairwise_cross_entropy_hun(
        mzs, logprobs, batch_idxs,
        batch_size=batch_size,
        tolerance=0.1,
        relative=False
    )
    
    print("CE Matrix:\n", ce_matrix)
    
    # Check shape
    assert ce_matrix.shape == (batch_size, batch_size)
    
    # Check CE(0,0) (Entropy of Spec 0)
    # P=1.0. H(P) = - 1*log(1) = 0.
    assert th.isclose(ce_matrix[0, 0], th.tensor(0.0, dtype=th.float32), atol=1e-5)
    
    # Check CE(1,1) (Entropy of Spec 1)
    # P=0.5, 0.5. H(P) = - (0.5*log(0.5)*2) = -log(0.5) = log(2) ~= 0.6931
    assert th.isclose(ce_matrix[1, 1], th.tensor(np.log(2), dtype=th.float32), atol=1e-5)
    
    # Check CE(0, 2) (Disjoint)
    # P: 10 (1.0). Q: 30 (1.0).
    # Match: None.
    # Contribution: P_0 * log(eps).
    # CE = - log(eps).
    print(f"CE(0,2) = {ce_matrix[0,2]}, Expected approx {-np.log(EPS)}")
    assert ce_matrix[0, 2] > 10.0
    
    # Check CE(0, 1) (Overlap)
    # P: 10 (1.0). Q: 10 (0.5), 20 (0.5).
    # Match 10 with 10.
    # P_0=1.0 matches Q_0=0.5.
    # Contribution: 1.0 * log(0.5).
    # CE = - log(0.5) = log(2).
    assert th.isclose(ce_matrix[0, 1], th.tensor(np.log(2), dtype=th.float32), atol=1e-5)
    
    # Check CE(1, 0) (Overlap)
    # P: 10 (0.5), 20 (0.5). Q: 10 (1.0).
    # Match 10 with 10.
    # P_0=0.5 matches Q_0=1.0. Contribution: 0.5 * log(1.0) = 0.
    # P_1=0.5 (mass 20) matches nothing. Contribution: 0.5 * log(eps).
    # CE = - (0 + 0.5 * log(eps)) = -0.5 * log(eps).
    assert ce_matrix[1, 0] > 5.0
    
def test_chunk_size():
    """
    Verify that chunk_size argument is accepted and doesn't break things.
    Even if implementation is iterative, API should support it.
    """
    print("Running test_chunk_size...")
    batch_size = 4
    mzs = th.tensor([10.0, 20.0, 30.0, 40.0], dtype=th.float32)
    logprobs = th.zeros(4, dtype=th.float32)
    batch_idxs = th.arange(4)
    
    ce_matrix = get_pairwise_cross_entropy_hun(
        mzs, logprobs, batch_idxs,
        batch_size=batch_size,
        chunk_size=2 # Distinctive chunk size
    )
    assert ce_matrix.shape == (batch_size, batch_size)
    print("Chunk size test passed.")

if __name__ == "__main__":
    test_pairwise_ce_hun_basic()
    test_chunk_size()
