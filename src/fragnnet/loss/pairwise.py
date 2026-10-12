"""Pairwise (B x B) similarity matrices between the spectra in a batch, for the pairwise
loss (docs/reference/pairwise_loss.md).
"""

import math

import torch as th
import torch.nn.functional as F

from fragnnet.utils.misc_utils import EPS, validate_bin_geometry
from fragnnet.utils.scatter_utils import (
    scatter_logl2normalize,
    scatter_logsoftmax,
    scatter_logsumexp,
)
from fragnnet.utils.spec_utils import batched_bin_func


def _build_dense_chunk(
    sorted_batch_idxs: th.Tensor,
    sorted_bin_idxs: th.Tensor,
    sorted_vals: th.Tensor,
    row_ptr_cpu: th.Tensor,
    row_start: int,
    row_end: int,
    num_unique_bins: int,
    device: th.device,
) -> th.Tensor:
    """Build a dense chunk [row_end-row_start, num_unique_bins] from sorted COO data."""
    chunk = th.zeros((row_end - row_start, num_unique_bins), device=device, dtype=sorted_vals.dtype)
    start_idx = int(row_ptr_cpu[row_start])
    end_idx = int(row_ptr_cpu[row_end])
    if end_idx <= start_idx:
        return chunk
    rows = sorted_batch_idxs[start_idx:end_idx] - row_start
    cols = sorted_bin_idxs[start_idx:end_idx]
    chunk[rows, cols] = sorted_vals[start_idx:end_idx]
    return chunk


def get_pairwise_cossim(
    mzs: th.Tensor,
    logprobs: th.Tensor,
    batch_idxs: th.Tensor,
    batch_size: int,
    mz_max: float,
    mz_bin_res: float,
    chunk_size: int = 32,
) -> th.Tensor:
    """Compute pairwise cosine similarity matrix for a batch of spectra.

    Avoids materializing the full [batch_size, num_unique_bins] dense matrix by
    building chunk-sized row slices on the fly. Peak GPU memory for the spectrum
    vectors is O(chunk_size * num_unique_bins) instead of O(batch_size * num_unique_bins),
    which prevents OOM when batch_size is large.

    Args:
        mzs: m/z values.
        logprobs: log-probabilities (intensities).
        batch_idxs: Batch indices.
        batch_size: Number of spectra in the batch.
        mz_max: Maximum m/z value.
        mz_bin_res: m/z bin resolution.
        chunk_size: Number of rows to process at once. Controls the memory/speed
            trade-off: smaller values use less memory but more iterations.

    Note:
        The nominal full spectrum grid size is
        ``num_bins = mz_max / mz_bin_res`` and must be an integer.
        For ``mz_max=1500.0`` and ``mz_bin_res=0.01``, this is 150,000 bins.
        This function compacts to occupied bins to reduce memory.

    Returns:
        Pairwise similarity matrix of shape [batch_size, batch_size].
    """
    batch_size = int(batch_size)
    if mzs.numel() == 0:
        return th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)

    # Binning
    bin_idxs, bin_logprobs, bin_batch_idxs = batched_bin_func(
        mzs, logprobs, batch_idxs, mz_max=mz_max, mz_bin_res=mz_bin_res, agg="lse", sparse=True
    )

    # Normalize (L2 on probabilities)
    norm_logprobs = scatter_logl2normalize(bin_logprobs, bin_batch_idxs)
    norm_probs = norm_logprobs.exp()

    # batched_bin_func returns batch-offset bin indices (batch element k uses range [k*num_bins,
    # (k+1)*num_bins)). Strip the offset to get per-spectrum bin positions so that the same m/z
    # bin in different spectra maps to the same column in the V matrix, enabling valid dot-product
    # cosine similarity. Without this, every spectrum occupies exclusively different columns and
    # all off-diagonal similarities are zero by construction.
    num_bins = validate_bin_geometry(mz_max, mz_bin_res)
    intra_bin_idxs = bin_idxs % num_bins

    # Compact bin space: avoids 150k-wide matrices when only a few bins are occupied
    unique_bins = th.unique(intra_bin_idxs)
    num_unique_bins = unique_bins.numel()
    bin_idx_map = th.full(
        (int(intra_bin_idxs.max().item()) + 1,), -1, device=mzs.device, dtype=th.long
    )
    bin_idx_map[unique_bins] = th.arange(num_unique_bins, device=mzs.device, dtype=th.long)
    compact_bin_idxs = bin_idx_map[intra_bin_idxs]

    effective_chunk = batch_size if (chunk_size is None or chunk_size >= batch_size) else chunk_size

    sort_idx = th.argsort(bin_batch_idxs)
    sorted_batch_idxs = bin_batch_idxs[sort_idx]
    sorted_bin_idxs = compact_bin_idxs[sort_idx]
    sorted_probs = norm_probs[sort_idx]
    counts = th.bincount(sorted_batch_idxs, minlength=batch_size)
    row_ptr = th.zeros(batch_size + 1, device=mzs.device, dtype=th.long)
    row_ptr[1:] = th.cumsum(counts, dim=0)
    row_ptr_cpu = row_ptr.detach().cpu()

    S = th.zeros((batch_size, batch_size), device=mzs.device, dtype=norm_probs.dtype)

    # Exploit symmetry: only compute upper triangle (j >= i) and mirror.
    # V_i is built once per i-chunk; V_j is built once per (i, j) pair with j >= i.
    for i in range(0, batch_size, effective_chunk):
        i_end = min(i + effective_chunk, batch_size)
        V_i = _build_dense_chunk(
            sorted_batch_idxs,
            sorted_bin_idxs,
            sorted_probs,
            row_ptr_cpu,
            i,
            i_end,
            num_unique_bins,
            mzs.device,
        )
        for j in range(i, batch_size, effective_chunk):
            j_end = min(j + effective_chunk, batch_size)
            if i == j:
                block = th.mm(V_i, V_i.t())
                S[i:i_end, i:i_end] = block
            else:
                V_j = _build_dense_chunk(
                    sorted_batch_idxs,
                    sorted_bin_idxs,
                    sorted_probs,
                    row_ptr_cpu,
                    j,
                    j_end,
                    num_unique_bins,
                    mzs.device,
                )
                block = th.mm(V_i, V_j.t())
                S[i:i_end, j:j_end] = block
                S[j:j_end, i:i_end] = block.t()

    return S


def get_pairwise_jss_sim(
    mzs: th.Tensor,
    logprobs: th.Tensor,
    batch_idxs: th.Tensor,
    batch_size: int,
    mz_max: float,
    mz_bin_res: float,
    chunk_size: int = 32,
) -> th.Tensor:
    """Compute pairwise Jensen-Shannon similarity matrix for a batch of spectra.

    Avoids materializing the full [batch_size, num_unique_bins] dense matrix by
    building chunk-sized row slices on the fly.

    Args:
        mzs: m/z values.
        logprobs: log-probabilities (intensities).
        batch_idxs: Batch indices.
        batch_size: Number of spectra in the batch.
        mz_max: Maximum m/z value.
        mz_bin_res: m/z bin resolution.
        chunk_size: Number of rows to process at once. Controls the memory/speed
            trade-off: smaller values use less memory but more iterations.

    Note:
        The nominal full spectrum grid size is
        ``num_bins = mz_max / mz_bin_res`` and must be an integer.
        For ``mz_max=1500.0`` and ``mz_bin_res=0.01``, this is 150,000 bins.
        This function compacts to occupied bins to reduce memory.

    Returns:
        Pairwise similarity matrix of shape [batch_size, batch_size].
    """
    batch_size = int(batch_size)
    if mzs.numel() == 0:
        return th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)

    # Binning
    bin_idxs, bin_logprobs, bin_batch_idxs = batched_bin_func(
        mzs, logprobs, batch_idxs, mz_max=mz_max, mz_bin_res=mz_bin_res, agg="lse", sparse=True
    )

    # Normalize (L1 on probabilities)
    log_probs = scatter_logsoftmax(bin_logprobs, bin_batch_idxs)
    probs = log_probs.exp()

    # Strip batch offsets (same fix as get_pairwise_cossim): batched_bin_func returns
    # batch-offset bin indices; without stripping, each spectrum occupies exclusively
    # different columns, making all pairwise mixtures trivially equal to 0.5*(P+0)=0.5*P
    # and H(M) collapses to a function of individual entropies only.
    num_bins = validate_bin_geometry(mz_max, mz_bin_res)
    intra_bin_idxs = bin_idxs % num_bins

    # Compact bin space
    unique_bins = th.unique(intra_bin_idxs)
    num_unique_bins = unique_bins.numel()
    bin_idx_map = th.full(
        (int(intra_bin_idxs.max().item()) + 1,), -1, device=mzs.device, dtype=th.long
    )
    bin_idx_map[unique_bins] = th.arange(num_unique_bins, device=mzs.device, dtype=th.long)
    compact_bin_idxs = bin_idx_map[intra_bin_idxs]

    effective_chunk = batch_size if (chunk_size is None or chunk_size >= batch_size) else chunk_size

    # Build full [batch_size, num_unique_bins] probability matrix.
    # batched_bin_func + scatter_logsoftmax guarantee unique (batch, bin) pairs,
    # so plain index assignment is safe (no duplicate scatter).
    P = th.zeros((batch_size, num_unique_bins), device=mzs.device, dtype=probs.dtype)
    P[bin_batch_idxs, compact_bin_idxs] = probs

    H_P = -(P * th.log(P + EPS)).sum(dim=1)  # [batch_size]

    # H(M) where M = 0.5*(P_i + P_j): exploit upper-triangle symmetry.
    # Each block is materialized as [ci, cj, K] and reduced along K in one
    # CUDA kernel — no inner Python loop, no row_ptr scaffolding.
    H_M = th.zeros((batch_size, batch_size), device=mzs.device, dtype=probs.dtype)
    for i in range(0, batch_size, effective_chunk):
        i_end = min(i + effective_chunk, batch_size)
        P_i = P[i:i_end]  # [ci, K]
        for j in range(i, batch_size, effective_chunk):
            j_end = min(j + effective_chunk, batch_size)
            P_j = P[j:j_end]  # [cj, K]
            M = 0.5 * (P_i.unsqueeze(1) + P_j.unsqueeze(0))  # [ci, cj, K]
            block = -(M * th.log(M + EPS)).sum(dim=2)  # [ci, cj]
            H_M[i:i_end, j:j_end] = block
            if i != j:
                H_M[j:j_end, i:i_end] = block.t()

    JSD = H_M - 0.5 * H_P.unsqueeze(1) - 0.5 * H_P.unsqueeze(0)
    # Clamp guards against tiny negative JSD (EPS perturbation in log) giving JSS > 1.
    JSS = th.clamp(1.0 - JSD / math.log(2.0), min=0.0, max=1.0)

    return JSS


def get_pairwise_cross_entropy(
    mzs: th.Tensor,
    logprobs: th.Tensor,
    batch_idxs: th.Tensor,
    batch_size: int,
    mz_max: float,
    mz_bin_res: float,
    chunk_size: int = 32,
) -> th.Tensor:
    """
    Compute pairwise cross-entropy matrix for a batch of spectra.
    Cross-Entropy(P, Q) = - sum(P * log(Q))

    Args:
        mzs: m/z values.
        logprobs: log-probabilities (intensities).
        batch_idxs: Batch indices.
        batch_size: Number of spectra in the batch.
        mz_max: Maximum m/z value.
        mz_bin_res: m/z bin resolution.
        chunk_size: Batch size for chunked computation to save memory.

    Note:
        The nominal full spectrum grid size is
        ``num_bins = mz_max / mz_bin_res`` and must be an integer.
        For ``mz_max=1500.0`` and ``mz_bin_res=0.01``, this is 150,000 bins.
        Bins are compacted to occupied indices before sparse/dense assembly.

    Returns:
        Pairwise cross-entropy matrix of shape [batch_size, batch_size].
    """

    batch_size = int(batch_size)
    if mzs.numel() == 0:
        return th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)

    # Binning
    bin_idxs, bin_logprobs, bin_batch_idxs = batched_bin_func(
        mzs, logprobs, batch_idxs, mz_max=mz_max, mz_bin_res=mz_bin_res, agg="lse", sparse=True
    )

    # Normalize to get probabilities (L1 normalization)
    log_probs = scatter_logsoftmax(bin_logprobs, bin_batch_idxs)
    probs = log_probs.exp()

    # Strip batch offsets: batched_bin_func returns batch-offset bin indices; without
    # stripping, each spectrum occupies exclusively different columns so P and Q never
    # share any bin and cross-entropy reduces to -sum(P*log(eps)) for all off-diagonal
    # pairs — masking any real signal from shared peaks.
    num_bins = validate_bin_geometry(mz_max, mz_bin_res)
    intra_bin_idxs = bin_idxs % num_bins

    unique_bins = th.unique(intra_bin_idxs)
    num_unique_bins = unique_bins.numel()
    bin_idx_map = th.full(
        (int(intra_bin_idxs.max().item()) + 1,), -1, device=mzs.device, dtype=th.long
    )
    bin_idx_map[unique_bins] = th.arange(num_unique_bins, device=mzs.device, dtype=th.long)
    compact_bin_idxs = bin_idx_map[intra_bin_idxs]

    effective_chunk = batch_size if (chunk_size is None or chunk_size >= batch_size) else chunk_size

    sort_idx = th.argsort(bin_batch_idxs)
    sorted_batch_idxs = bin_batch_idxs[sort_idx]
    sorted_bin_idxs = compact_bin_idxs[sort_idx]
    sorted_probs = probs[sort_idx]
    counts = th.bincount(sorted_batch_idxs, minlength=batch_size)
    row_ptr = th.zeros(batch_size + 1, device=mzs.device, dtype=th.long)
    row_ptr[1:] = th.cumsum(counts, dim=0)
    row_ptr_cpu = row_ptr.detach().cpu()

    CE = th.zeros((batch_size, batch_size), device=mzs.device, dtype=probs.dtype)

    for i in range(0, batch_size, effective_chunk):
        i_end = min(i + effective_chunk, batch_size)
        P_i = _build_dense_chunk(
            sorted_batch_idxs,
            sorted_bin_idxs,
            sorted_probs,
            row_ptr_cpu,
            i,
            i_end,
            num_unique_bins,
            mzs.device,
        )
        log_P_i = th.log(P_i + EPS)
        for j in range(0, batch_size, effective_chunk):
            j_end = min(j + effective_chunk, batch_size)
            # Reuse log_P_i for diagonal blocks (j == i) to avoid rebuilding the same chunk.
            log_Q_j = (
                log_P_i
                if i == j
                else th.log(
                    _build_dense_chunk(
                        sorted_batch_idxs,
                        sorted_bin_idxs,
                        sorted_probs,
                        row_ptr_cpu,
                        j,
                        j_end,
                        num_unique_bins,
                        mzs.device,
                    )
                    + EPS
                )
            )
            CE[i:i_end, j:j_end] = -th.mm(P_i, log_Q_j.t())

    return CE


def get_cross_jss_contrastive(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    batch_size: int,
    mz_max: float,
    mz_bin_res: float,
    temperature: float = 0.07,
    sqrt_transform: bool = True,
    chunk_size: int = 32,
) -> th.Tensor:
    """Cross-matrix JSS contrastive loss for retrieval-aligned training.

    Computes a [batch_size, batch_size] similarity matrix where row i uses the
    real spectrum of molecule i and column j uses the predicted spectrum of
    molecule j:

        S_ij = JSS-sqrt(real_i, pred_j)

    A contrastive (InfoNCE) loss is then applied row-wise so that the correct
    candidate (diagonal) is ranked above all wrong candidates (off-diagonal).
    This directly optimises the retrieval gap that standard per-sample losses
    never see.

    Args:
        true_mzs: Real spectrum m/z values.
        true_logprobs: Real spectrum log-intensities (L1-normalised, no sqrt).
        true_batch_idxs: Batch indices for real peaks.
        pred_mzs: Predicted m/z values.
        pred_logprobs: Predicted log-probabilities from the model.
        pred_batch_idxs: Batch indices for predicted peaks.
        batch_size: Number of spectra in the batch.
        mz_max: Maximum m/z for binning.
        mz_bin_res: Bin resolution for binning.
        temperature: InfoNCE temperature. Lower values make the loss sharper.
        sqrt_transform: If True, apply sqrt-then-renormalise to both spectra
            before computing JSS so the metric matches JSS-sqrt retrieval.
            Set to False only if inputs are already sqrt-transformed.
        chunk_size: Row chunk size for the [B, B] H_M computation.

    Returns:
        Per-sample contrastive loss of shape [batch_size].
    """
    batch_size = int(batch_size)
    device = true_logprobs.device
    dtype = true_logprobs.dtype

    if true_mzs.numel() == 0 or pred_mzs.numel() == 0:
        return th.zeros(batch_size, device=device, dtype=dtype)

    # Bin both spectra into sparse COO representation
    true_bin_idxs, true_bin_lp, true_bin_bidxs = batched_bin_func(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",
        sparse=True,
    )
    pred_bin_idxs, pred_bin_lp, pred_bin_bidxs = batched_bin_func(
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",
        sparse=True,
    )

    # L1-normalise
    true_lp = scatter_logsoftmax(true_bin_lp, true_bin_bidxs)
    pred_lp = scatter_logsoftmax(pred_bin_lp, pred_bin_bidxs)

    # Optional sqrt-then-renormalise: log(sqrt(p)) = 0.5*log(p), then re-normalise
    if sqrt_transform:
        true_lp_half = 0.5 * true_lp
        true_lp = true_lp_half - scatter_logsumexp(true_lp_half, true_bin_bidxs)[true_bin_bidxs]
        pred_lp_half = 0.5 * pred_lp
        pred_lp = pred_lp_half - scatter_logsumexp(pred_lp_half, pred_bin_bidxs)[pred_bin_bidxs]

    true_probs = true_lp.exp()
    pred_probs = pred_lp.exp()

    # Strip batch offsets and build a shared compact bin space across both sets
    num_bins = validate_bin_geometry(mz_max, mz_bin_res)
    true_intra = true_bin_idxs % num_bins
    pred_intra = pred_bin_idxs % num_bins

    all_intra = th.cat([true_intra, pred_intra])
    unique_bins = th.unique(all_intra)
    K = unique_bins.numel()
    bin_idx_map = th.full((int(all_intra.max().item()) + 1,), -1, device=device, dtype=th.long)
    bin_idx_map[unique_bins] = th.arange(K, device=device, dtype=th.long)
    true_compact = bin_idx_map[true_intra]
    pred_compact = bin_idx_map[pred_intra]

    # Dense [B, K] probability matrices
    P = true_probs.new_zeros((batch_size, K))
    P[true_bin_bidxs, true_compact] = true_probs
    Q = pred_probs.new_zeros((batch_size, K))
    Q[pred_bin_bidxs, pred_compact] = pred_probs

    # Per-spectrum entropies [B]
    H_P = -(P * th.log(P + EPS)).sum(dim=1)
    H_Q = -(Q * th.log(Q + EPS)).sum(dim=1)

    # Cross H_M[i,j] = H(0.5*(P[i] + Q[j])) — no symmetry since P ≠ Q, full [B,B]
    effective_chunk = batch_size if (chunk_size is None or chunk_size >= batch_size) else chunk_size
    H_M = th.zeros((batch_size, batch_size), device=device, dtype=dtype)
    for i in range(0, batch_size, effective_chunk):
        i_end = min(i + effective_chunk, batch_size)
        P_i = P[i:i_end]  # [ci, K]
        for j in range(0, batch_size, effective_chunk):
            j_end = min(j + effective_chunk, batch_size)
            Q_j = Q[j:j_end]  # [cj, K]
            M = 0.5 * (P_i.unsqueeze(1) + Q_j.unsqueeze(0))  # [ci, cj, K]
            H_M[i:i_end, j:j_end] = -(M * th.log(M + EPS)).sum(dim=2)

    # JSS [B, B]: rows = real spectra, cols = predicted spectra
    JSD = H_M - 0.5 * H_P.unsqueeze(1) - 0.5 * H_Q.unsqueeze(0)
    JSS = th.clamp(1.0 - JSD / math.log(2.0), min=0.0, max=1.0)

    # InfoNCE: for each real spectrum (row), rank its own correct prediction highest
    labels = th.arange(batch_size, device=device, dtype=th.long)
    return F.cross_entropy(JSS / temperature, labels, reduction="none")  # [B]
