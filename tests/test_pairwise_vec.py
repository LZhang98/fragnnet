import sys
import numpy as np
from unittest.mock import MagicMock
import torch as th

# Mock modules
sys.modules["fragnnet.frag"] = MagicMock()
compute_frags_mock = MagicMock()
sys.modules["fragnnet.frag.compute_frags"] = compute_frags_mock
compute_frags_mock.MASK_DTYPE = np.int64
compute_frags_mock.MAX_NUM_EDGES = 100
compute_frags_mock.MAX_NUM_NODES = 100

from fragnnet.model.loss import (
    get_pairwise_cross_entropy,
    get_pairwise_jss_sim,
    get_pairwise_cossim
)
from fragnnet.utils.misc_utils import EPS

def test_pairwise_ce_vec_basic():
    """Test basic shape and gradients for CE Vec."""
    print("Running test_pairwise_ce_vec_basic...")
    batch_size = 3
    # Mzs: 10, 20, 30
    mzs = th.tensor([10.0, 20.0, 30.0], dtype=th.float32)
    # Logprobs (will be normalized inside)
    logprobs = th.tensor([0.0, 0.0, 0.0], dtype=th.float32, requires_grad=True)
    batch_idxs = th.tensor([0, 1, 2])
    
    # Use binned implementation
    ce_mat_g = get_pairwise_cross_entropy(
        mzs, logprobs, batch_idxs, batch_size,
        mz_max=100.0, mz_bin_res=1.0, chunk_size=None
    )
    assert ce_mat_g.shape == (batch_size, batch_size)
    assert not th.isnan(ce_mat_g).any()
    
    # Check gradient
    loss = ce_mat_g.sum()
    loss.backward()
    assert logprobs.grad is not None
    assert not th.isnan(logprobs.grad).any()
    
    # Peak Marginal
    logprobs.grad.zero_()
    # Re-run CE to sanity check alternative path (same signature here)
    ce_mat_pm = get_pairwise_cross_entropy(
        mzs, logprobs, batch_idxs, batch_size,
        mz_max=100.0, mz_bin_res=1.0, chunk_size=None
    )
    assert ce_mat_pm.shape == (batch_size, batch_size)
    
    print("CE Vec Basic passed.")

def test_pairwise_jss_vec_symmetry():
    """Test symmetry and self-similarity for JSS Vec."""
    print("Running test_pairwise_jss_vec_symmetry...")
    batch_size = 2
    mzs = th.tensor([10.0, 10.1, 20.0], dtype=th.float32)
    logprobs = th.tensor([0.0, 0.0, 0.0], dtype=th.float32)
    batch_idxs = th.tensor([0, 0, 1])
    
    # Use gaussian
    jss_mat = get_pairwise_jss_sim(
        mzs, logprobs, batch_idxs, batch_size,
        mz_max=100.0, mz_bin_res=1.0, chunk_size=None
    )
    
    # Check shape
    assert jss_mat.shape == (batch_size, batch_size)
    
    # Self similarity should be close to 1.0
    # Note: For continuous Gaussian approx mixtures, KL(P||P) is 0, so JSS is 1.
    print(f"JSS Diag: {jss_mat.diag()}")
    assert th.allclose(jss_mat.diag(), th.ones(batch_size, dtype=jss_mat.dtype), atol=1e-3)
    
    # Symmetry
    assert th.allclose(jss_mat, jss_mat.t(), atol=1e-5)
    
    print("JSS Vec Symmetry passed.")

def test_pairwise_cossim_vec_symmetry():
    """Test symmetry and self-similarity for Cossim Vec."""
    print("Running test_pairwise_cossim_vec_symmetry...")
    batch_size = 2
    mzs = th.tensor([10.0, 10.1, 20.0], dtype=th.float32)
    logprobs = th.tensor([0.0, 0.0, 0.0], dtype=th.float32)
    batch_idxs = th.tensor([0, 0, 1])
    
    cossim_mat = get_pairwise_cossim(
        mzs, logprobs, batch_idxs, batch_size,
        mz_max=100.0, mz_bin_res=1.0
    )
    
    assert cossim_mat.shape == (batch_size, batch_size)
    
    # Self similarity should be 1.0 (L2 normalized inputs)
    print(f"Cossim Diag: {cossim_mat.diag()}")
    assert th.allclose(cossim_mat.diag(), th.ones(batch_size, dtype=cossim_mat.dtype), atol=1e-3)
    
    # Symmetry
    assert th.allclose(cossim_mat, cossim_mat.t(), atol=1e-5)
    
    print("Cossim Vec Symmetry passed.")

# Tests are collected by pytest; no direct execution block required.
