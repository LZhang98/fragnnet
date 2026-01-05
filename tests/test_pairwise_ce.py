import sys
import numpy as np
from unittest.mock import MagicMock

# Mock modules
sys.modules["fragnnet.frag"] = MagicMock()
compute_frags_mock = MagicMock()
sys.modules["fragnnet.frag.compute_frags"] = compute_frags_mock

# Mock attributes
compute_frags_mock.MASK_DTYPE = np.int64
compute_frags_mock.MAX_NUM_EDGES = 100
compute_frags_mock.MAX_NUM_NODES = 100

import torch as th
from fragnnet.model.loss import get_pairwise_binned_cross_entropy
from fragnnet.utils.misc_utils import EPS

def test_pairwise_binned_cross_entropy():
    # Setup synthetic data
    batch_size = 5
    mz_max = 100.0
    mz_bin_res = 1.0
    
    # Create simple spectra: each spectrum has 2 peaks
    # Spec 0: mass 10, 20
    # Spec 1: mass 10, 30
    # Spec 2: mass 40, 50
    # ...
    
    # Mzs
    mzs_list = [
        th.tensor([10.0, 20.0]),
        th.tensor([10.0, 30.0]),
        th.tensor([40.0, 50.0]),
        th.tensor([10.0, 20.0]), # Same as 0
        th.tensor([10.5, 20.5]), # Slightly shifted
    ]
    
    # Logprobs (Unnormalized intensities)
    logprobs_list = [
        th.log(th.tensor([0.5, 0.5])),
        th.log(th.tensor([0.8, 0.2])),
        th.log(th.tensor([0.1, 0.9])),
        th.log(th.tensor([0.5, 0.5])),
        th.log(th.tensor([0.5, 0.5])),
    ]
    
    # Batch Indices
    batch_idxs_list = []
    for i, m in enumerate(mzs_list):
        batch_idxs_list.append(th.full_like(m, i, dtype=th.long))
        
    mzs = th.cat(mzs_list)
    logprobs = th.cat(logprobs_list)
    batch_idxs = th.cat(batch_idxs_list)
    
    # Run the function
    ce_matrix = get_pairwise_binned_cross_entropy(
        mzs, logprobs, batch_idxs,
        batch_size=batch_size,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        chunk_size=None # Full matrix
    )
    
    print("CE Matrix:\n", ce_matrix)
    
    # Verify basics
    assert ce_matrix.shape == (batch_size, batch_size)
    
    # Verify Item 0 and Item 3 are identical -> CE(0,3) should equal CE(0,0) and CE(3,3)
    assert th.isclose(ce_matrix[0, 3], ce_matrix[0, 0])
    assert th.isclose(ce_matrix[3, 0], ce_matrix[3, 3])
    
    # Verify manual calculation for (0, 1)
    # Spec 0: bins 10, 20. Probs: 0.5, 0.5
    # Spec 1: bins 10, 30. Probs: 0.8, 0.2
    # Spec 1 as Q for Spec 0 as P:
    # P has mass at 10 (0.5), 20 (0.5).
    # Q has mass at 10 (0.8), 30 (0.2).
    # Q at 20 is missing -> 0 -> log(eps).
    # CE(P=0, Q=1) = - (0.5 * log(0.8) + 0.5 * log(eps))
    # Note: Using bin center logic, 10.0 falls into bin 10 (if 1-based) or 9 (if 0-based)?
    # Implementation uses: bin_idx = floor(mz / res). 
    # 10.0 / 1.0 = 10. 
    # So both fall in bin 10.
    
    expected_ce_0_1 = - (0.5 * th.log(th.tensor(0.8) + EPS) + 0.5 * th.log(th.tensor(0.0) + EPS))
    loss_0_1 = ce_matrix[0, 1]
    
    print(f"Computed CE(0,1): {loss_0_1}")
    print(f"Expected CE(0,1): {expected_ce_0_1}")
    
    assert th.isclose(loss_0_1, expected_ce_0_1, atol=1e-5)
    
    # Verify chunked computation
    ce_matrix_chunked = get_pairwise_binned_cross_entropy(
        mzs, logprobs, batch_idxs,
        batch_size=batch_size,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        chunk_size=2
    )
    
    assert th.allclose(ce_matrix, ce_matrix_chunked, atol=1e-5)
    print("Chunked computation matches.")

if __name__ == "__main__":
    test_pairwise_binned_cross_entropy()
