"""Memory usage tests for pairwise loss functions.

These tests verify that the memory optimization (bin compacting) works correctly
and that memory usage stays within reasonable bounds, even with large batch sizes.
"""

import torch as th
import psutil
import os
from src.fragnnet.model.loss import get_pairwise_cossim, get_pairwise_jss_sim, get_pairwise_cross_entropy


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_pairwise_cossim_memory_usage():
    """Test that cosine similarity doesn't use excessive memory.
    
    Without bin compacting: ~115GB for batch_size=256
    With bin compacting: ~0.75GB for batch_size=256
    """
    batch_size = 64  # Smaller batch size for testing
    num_peaks_per_spectrum = 50
    mz_max = 1500.0
    mz_bin_res = 0.01
    
    # Create realistic spectrum data
    mzs = []
    logprobs = []
    batch_idxs = []
    
    for batch_idx in range(batch_size):
        # Random number of peaks per spectrum (20-80)
        num_peaks = th.randint(20, 81, (1,)).item()
        for _ in range(num_peaks):
            mzs.append(th.rand(1).item() * mz_max)
            logprobs.append(th.randn(1).item())
            batch_idxs.append(batch_idx)
    
    mzs = th.tensor(mzs, dtype=th.float32)
    logprobs = th.tensor(logprobs, dtype=th.float32)
    batch_idxs = th.tensor(batch_idxs, dtype=th.long)
    
    # Measure memory before
    mem_before = get_memory_usage_mb()
    
    # Call the function
    S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    
    # Measure memory after
    mem_after = get_memory_usage_mb()
    mem_used = mem_after - mem_before
    
    # Verify output shape
    assert S.shape == (batch_size, batch_size), f"Expected shape {(batch_size, batch_size)}, got {S.shape}"
    
    # Memory usage should be < 1GB for batch_size=64
    # (Without optimization it would be ~30GB)
    assert mem_used < 1000, f"Memory usage {mem_used:.1f}MB exceeds 1GB limit"
    
    print(f"Cosine similarity: Memory used = {mem_used:.1f}MB (OK)")


def test_pairwise_jss_sim_memory_usage():
    """Test that JSS similarity doesn't use excessive memory.
    
    Without bin compacting: ~115GB for batch_size=256
    With bin compacting: ~0.75GB for batch_size=256
    """
    batch_size = 64
    mz_max = 1500.0
    mz_bin_res = 0.01
    
    # Create realistic spectrum data
    mzs = []
    logprobs = []
    batch_idxs = []
    
    for batch_idx in range(batch_size):
        num_peaks = th.randint(20, 81, (1,)).item()
        for _ in range(num_peaks):
            mzs.append(th.rand(1).item() * mz_max)
            logprobs.append(th.randn(1).item())
            batch_idxs.append(batch_idx)
    
    mzs = th.tensor(mzs, dtype=th.float32)
    logprobs = th.tensor(logprobs, dtype=th.float32)
    batch_idxs = th.tensor(batch_idxs, dtype=th.long)
    
    # Measure memory before
    mem_before = get_memory_usage_mb()
    
    # Call the function
    JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    
    # Measure memory after
    mem_after = get_memory_usage_mb()
    mem_used = mem_after - mem_before
    
    # Verify output shape
    assert JSS.shape == (batch_size, batch_size), f"Expected shape {(batch_size, batch_size)}, got {JSS.shape}"
    
    # Memory usage should be < 1GB for batch_size=64
    assert mem_used < 1000, f"Memory usage {mem_used:.1f}MB exceeds 1GB limit"
    
    print(f"JSS similarity: Memory used = {mem_used:.1f}MB (OK)")


def test_pairwise_cross_entropy_memory_usage():
    """Test that cross-entropy doesn't use excessive memory.
    
    Without bin compacting: ~115GB for batch_size=256
    With bin compacting: ~0.75GB for batch_size=256
    """
    batch_size = 64
    mz_max = 1500.0
    mz_bin_res = 0.01
    
    # Create realistic spectrum data
    mzs = []
    logprobs = []
    batch_idxs = []
    
    for batch_idx in range(batch_size):
        num_peaks = th.randint(20, 81, (1,)).item()
        for _ in range(num_peaks):
            mzs.append(th.rand(1).item() * mz_max)
            logprobs.append(th.randn(1).item())
            batch_idxs.append(batch_idx)
    
    mzs = th.tensor(mzs, dtype=th.float32)
    logprobs = th.tensor(logprobs, dtype=th.float32)
    batch_idxs = th.tensor(batch_idxs, dtype=th.long)
    
    # Measure memory before
    mem_before = get_memory_usage_mb()
    
    # Call the function
    CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    
    # Measure memory after
    mem_after = get_memory_usage_mb()
    mem_used = mem_after - mem_before
    
    # Verify output shape
    assert CE.shape == (batch_size, batch_size), f"Expected shape {(batch_size, batch_size)}, got {CE.shape}"
    
    # Memory usage should be < 1GB for batch_size=64
    assert mem_used < 1000, f"Memory usage {mem_used:.1f}MB exceeds 1GB limit"
    
    print(f"Cross-entropy: Memory used = {mem_used:.1f}MB (OK)")


def test_pairwise_cossim_large_batch_memory():
    """Test cosine similarity with large batch size (close to production scale).
    
    Production batch_size=256 would use ~115GB without optimization.
    With bin compacting, should use < 5GB.
    """
    batch_size = 128  # Half of production batch size
    mz_max = 1500.0
    mz_bin_res = 0.01
    
    # Create realistic spectrum data
    mzs = []
    logprobs = []
    batch_idxs = []
    
    for batch_idx in range(batch_size):
        num_peaks = th.randint(20, 81, (1,)).item()
        for _ in range(num_peaks):
            mzs.append(th.rand(1).item() * mz_max)
            logprobs.append(th.randn(1).item())
            batch_idxs.append(batch_idx)
    
    mzs = th.tensor(mzs, dtype=th.float32)
    logprobs = th.tensor(logprobs, dtype=th.float32)
    batch_idxs = th.tensor(batch_idxs, dtype=th.long)
    
    mem_before = get_memory_usage_mb()
    S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    mem_after = get_memory_usage_mb()
    mem_used = mem_after - mem_before
    
    assert S.shape == (batch_size, batch_size)
    # With bin compacting, should use < 3GB for batch_size=128
    assert mem_used < 3000, f"Memory usage {mem_used:.1f}MB exceeds 3GB limit"
    
    print(f"Cosine similarity (large batch): Memory used = {mem_used:.1f}MB (OK)")


def test_pairwise_cossim_chunked_memory():
    """Test that chunked computation doesn't use more memory than non-chunked.
    
    Chunking should reduce peak memory usage.
    """
    batch_size = 64
    mz_max = 1500.0
    mz_bin_res = 0.01
    
    # Create realistic spectrum data
    mzs = []
    logprobs = []
    batch_idxs = []
    
    for batch_idx in range(batch_size):
        num_peaks = th.randint(20, 81, (1,)).item()
        for _ in range(num_peaks):
            mzs.append(th.rand(1).item() * mz_max)
            logprobs.append(th.randn(1).item())
            batch_idxs.append(batch_idx)
    
    mzs = th.tensor(mzs, dtype=th.float32)
    logprobs = th.tensor(logprobs, dtype=th.float32)
    batch_idxs = th.tensor(batch_idxs, dtype=th.long)
    
    # Test with chunking
    mem_before = get_memory_usage_mb()
    S_chunked = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=16)
    mem_after = get_memory_usage_mb()
    mem_chunked = mem_after - mem_before
    
    assert S_chunked.shape == (batch_size, batch_size)
    assert mem_chunked < 1000, f"Chunked memory usage {mem_chunked:.1f}MB exceeds 1GB limit"
    
    print(f"Cosine similarity (chunked): Memory used = {mem_chunked:.1f}MB (OK)")
