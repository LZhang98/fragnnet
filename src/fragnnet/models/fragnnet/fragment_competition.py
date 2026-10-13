"""Transformer residuals for competition among predicted fragments."""

import torch as th
import torch.nn as nn


class TopKFragmentTransformer(nn.Module):
    """Add bounded residuals to the highest-scoring fragments per spectrum."""

    def __init__(
        self,
        fragment_dim: int,
        context_dim: int,
        top_k: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_delta: float,
        detach_inputs: bool,
    ) -> None:
        super().__init__()
        if min(fragment_dim, top_k, hidden_dim, num_layers, num_heads) <= 0:
            raise ValueError("fragment and transformer dimensions and top_k must be positive")
        if context_dim < 0:
            raise ValueError("context_dim must be nonnegative")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if max_delta <= 0.0:
            raise ValueError(f"max_delta must be positive, got {max_delta}")

        self.context_dim = context_dim
        self.top_k = top_k
        self.max_delta = float(max_delta)
        self.detach_inputs = detach_inputs
        self.fragment_projection = nn.Sequential(
            nn.Linear(fragment_dim, hidden_dim),
            nn.SiLU(),
        )
        self.token_projection = nn.Linear(hidden_dim + 1, hidden_dim)
        self.context_projection = (
            nn.Linear(context_dim, hidden_dim) if context_dim > 0 else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def select_fragment_indices(
        self,
        scores: th.Tensor,
        batch_idxs: th.Tensor,
        real_mask: th.Tensor,
        *,
        batch_size: int,
    ) -> th.Tensor:
        """Return flat top-k fragment indices grouped by spectrum."""
        if scores.ndim != 1:
            raise ValueError("scores must be one-dimensional")
        if batch_idxs.shape != scores.shape or real_mask.shape != scores.shape:
            raise ValueError("batch_idxs and real_mask must match scores")
        if batch_idxs.dtype != th.long:
            raise ValueError("batch_idxs must have dtype torch.long")
        if real_mask.dtype != th.bool:
            raise ValueError("real_mask must have dtype torch.bool")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        ranking_scores = th.nan_to_num(
            scores.detach(),
            nan=float("-inf"),
            posinf=1e4,
            neginf=float("-inf"),
        )
        selected_parts = []
        for batch_idx in range(batch_size):
            fragment_idxs = th.nonzero(
                real_mask & (batch_idxs == batch_idx), as_tuple=False
            ).flatten()
            if fragment_idxs.numel() == 0:
                raise ValueError(f"spectrum {batch_idx} has no real fragment candidates")
            k = min(self.top_k, fragment_idxs.numel())
            local_top = th.topk(ranking_scores[fragment_idxs], k=k).indices
            selected_parts.append(fragment_idxs[local_top])
        return th.cat(selected_parts)

    def forward(
        self,
        channel_logits: th.Tensor,
        fragment_embeddings: th.Tensor,
        fragment_batch_idxs: th.Tensor,
        real_channel_mask: th.Tensor,
        context: th.Tensor | None,
        *,
        batch_size: int,
    ) -> th.Tensor:
        """Apply one shared residual to all real channels of each selected fragment."""
        if channel_logits.ndim != 2:
            raise ValueError("channel_logits must have shape (num_fragments, num_channels)")
        if (
            fragment_embeddings.ndim != 2
            or fragment_embeddings.shape[0] != channel_logits.shape[0]
        ):
            raise ValueError("fragment_embeddings must contain one row per fragment")
        if fragment_batch_idxs.shape != (channel_logits.shape[0],):
            raise ValueError("fragment_batch_idxs must contain one index per fragment")
        if fragment_batch_idxs.dtype != th.long:
            raise ValueError("fragment_batch_idxs must have dtype torch.long")
        if real_channel_mask.shape != channel_logits.shape:
            raise ValueError("real_channel_mask must match channel_logits")
        if real_channel_mask.dtype != th.bool:
            raise ValueError("real_channel_mask must have dtype torch.bool")
        if self.context_dim == 0:
            if context is not None:
                raise ValueError("context was provided but context_dim is zero")
        elif context is None or context.shape != (batch_size, self.context_dim):
            raise ValueError(f"context must have shape ({batch_size}, {self.context_dim})")

        # 1. Rank fragments by their total valid-channel logit mass.
        real_fragments = real_channel_mask.any(dim=1)
        masked_logits = channel_logits.masked_fill(~real_channel_mask, float("-inf"))
        fragment_scores = th.logsumexp(masked_logits, dim=1)
        selected_idxs = self.select_fragment_indices(
            fragment_scores,
            fragment_batch_idxs,
            real_fragments,
            batch_size=batch_size,
        )
        selected_batch_idxs = fragment_batch_idxs[selected_idxs]

        # 2. Build fragment tokens with base score and acquisition context.
        fragment_inputs = (
            fragment_embeddings.detach() if self.detach_inputs else fragment_embeddings
        )
        score_inputs = fragment_scores.detach() if self.detach_inputs else fragment_scores
        context_inputs = (
            context.detach() if self.detach_inputs and context is not None else context
        )
        selected_features = self.fragment_projection(fragment_inputs[selected_idxs])
        selected_scores = th.nan_to_num(
            score_inputs[selected_idxs],
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        ).unsqueeze(1)
        tokens = self.token_projection(th.cat([selected_features, selected_scores], dim=1))
        if self.context_projection is not None:
            tokens = tokens + self.context_projection(context_inputs)[selected_batch_idxs]

        # 3. Contextualize the selected fragment set independently per spectrum.
        counts = th.bincount(selected_batch_idxs, minlength=batch_size)
        starts = counts.cumsum(0) - counts
        positions = (
            th.arange(selected_idxs.shape[0], device=selected_idxs.device)
            - starts[selected_batch_idxs]
        )
        padded_tokens = tokens.new_zeros(batch_size, self.top_k, tokens.shape[1])
        padded_tokens[selected_batch_idxs, positions] = tokens
        padding_mask = (
            th.arange(self.top_k, device=tokens.device).unsqueeze(0)
            >= counts.unsqueeze(1)
        )

        encoded = self.transformer(padded_tokens, src_key_padding_mask=padding_mask)
        encoded = th.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        # 4. Scatter one bounded logit residual across each fragment's valid channels.
        selected_delta = self.residual_head(
            encoded[selected_batch_idxs, positions]
        ).squeeze(1)
        selected_delta = self.max_delta * th.tanh(selected_delta / self.max_delta)
        fragment_delta = channel_logits.new_zeros(channel_logits.shape[0]).index_copy(
            0,
            selected_idxs,
            selected_delta,
        )
        return channel_logits + fragment_delta.unsqueeze(1) * real_channel_mask.to(
            channel_logits.dtype
        )
