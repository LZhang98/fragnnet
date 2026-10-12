"""Cut-bond encoder shared by the fragment models (FraGNNet and MASSter)."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CutBondEncoder(nn.Module):
    """Encode a fragment's cut bonds from the (inside, outside) atom embeddings of each bond.

    Replaces the ``boundary_pair`` feature, which encodes each cut bond as
    ``Linear(cat(in, out)) == W_in @ in + W_out @ out + b`` and then mean-pools over the
    fragment's bonds. That construction is degenerate in two ways:

    1. The projection is additive, so a bond is only ever the *sum of two independent atom
       terms* -- "C-N bond" cannot differ from "C contribution + N contribution". Bond identity
       is inherently an interaction, so no linear map on a concatenation can express it. This
       holds even for a fragment with a single cut bond.
    2. Linear commutes through mean/sum pooling, so for >= 2 cut bonds the pooled result equals
       ``Linear(mean(inside_atoms), mean(outside_atoms))`` -- which atom was bonded to which is
       discarded exactly. Cutting a1-b1 + a2-b2 gives a bit-identical embedding to cutting
       a1-b2 + a2-b1. That is precisely the multi-cut ring-cleavage regime.

    This encoder fixes both: an explicit elementwise product ``in * out`` gives each bond a
    multiplicative signature, and the hidden nonlinearity stops the projection commuting through
    the pool, so the pairing survives. The (in, out) order is kept asymmetric because it is
    chemically meaningful -- inside atoms are retained by the fragment, outside atoms are lost.

    Pooling is left to the caller (the model reuses ``mol_pool``, as ``boundary_pair`` does) so
    swapping the two features is a clean one-variable comparison at identical width.

    Attributes:
        atom_dim: Width of a single atom embedding from the mol encoder.
        out_dim: Width of the per-bond encoding (matched to boundary_pair for comparability).
        net: MLP mapping [in, out, in * out] to the per-bond encoding.

    Example:
        >>> enc = CutBondEncoder(atom_dim=384, hidden_dim=256, out_dim=384)
        >>> bonds = enc(in_embed, out_embed)  # (num_cut_bonds, 384)
        >>> frag = mol_pool(bonds, bond_frag_idxs, dim_size=num_frags)
    """

    def __init__(self, atom_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        """Initialize CutBondEncoder.

        Args:
            atom_dim: Width of one atom embedding from the mol encoder.
            hidden_dim: Hidden width of the bond MLP.
            out_dim: Width of the per-bond output encoding.
            dropout: Training-only dropout on the hidden layer. It is applied in ``forward``
                rather than added to ``net``, so the state-dict keys, and every checkpoint
                written before it existed, are unchanged. 0.0 draws no random numbers.

        Raises:
            ValueError: If any dimension is not positive or ``dropout`` is outside [0, 1).
        """
        super().__init__()
        for name, dim in (("atom_dim", atom_dim), ("hidden_dim", hidden_dim), ("out_dim", out_dim)):
            if dim <= 0:
                raise ValueError(f"CutBondEncoder {name} must be positive, got {dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"CutBondEncoder dropout must be in [0, 1), got {dropout}")
        self.atom_dim = atom_dim
        self.out_dim = out_dim
        self.dropout = float(dropout)
        # 3 * atom_dim: [in, out, in * out] -- the product is what makes the bond more than the
        # sum of its atoms; the hidden nonlinearity is what stops pooling collapsing the pairing.
        self.net = nn.Sequential(
            nn.Linear(3 * atom_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, in_embed: th.Tensor, out_embed: th.Tensor) -> th.Tensor:
        """Encode each cut bond from its inside/outside atom embeddings.

        Args:
            in_embed: Inside-atom (retained) embeddings of shape (num_cut_bonds, atom_dim).
            out_embed: Outside-atom (lost) embeddings of shape (num_cut_bonds, atom_dim).

        Returns:
            Per-bond encoding of shape (num_cut_bonds, out_dim). Pool over the fragment's bonds
            to obtain a per-fragment feature.

        Raises:
            ValueError: If the two inputs disagree in shape or are not atom_dim wide.
        """
        if in_embed.shape != out_embed.shape:
            raise ValueError(
                f"in_embed {tuple(in_embed.shape)} and out_embed {tuple(out_embed.shape)} "
                "must have the same shape"
            )
        if in_embed.shape[-1] != self.atom_dim:
            raise ValueError(
                f"CutBondEncoder got atom width {in_embed.shape[-1]}, expected {self.atom_dim}"
            )
        features = th.cat([in_embed, out_embed, in_embed * out_embed], dim=-1)
        hidden = self.net[1](self.net[0](features))
        if self.dropout > 0.0:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.net[2](hidden)


def encode_cut_bond_pairs(
    encoder: nn.Module,
    atom_embeddings: th.Tensor,
    in_indices: th.Tensor,
    out_indices: th.Tensor,
) -> th.Tensor:
    """Encode (inside, outside) cut-bond pairs once per distinct pair, then gather per row.

    A :class:`CutBondEncoder` output depends only on the two endpoint embeddings, and the
    same directed bond recurs across every fragment that severs it: about 10x per row in
    MASSter merged batches, about 29x on the FraGNNet test DAG. Encoding each distinct
    pair once and gathering the rows back gives the same values as
    ``encoder(atom_embeddings[in_indices], atom_embeddings[out_indices])`` up to float
    rounding, with the same parameters, at a fraction of the matmul work and activation
    memory. Gradients from repeated rows accumulate through the gather.

    Args:
        encoder: The cut-bond encoder (a :class:`CutBondEncoder`, possibly compiled).
        atom_embeddings: Atom embeddings of shape ``(num_atoms, atom_dim)``.
        in_indices: Inside (retained) atom index per row, shape ``(num_rows,)``, ``long``.
        out_indices: Outside (lost) atom index per row, shape ``(num_rows,)``, ``long``.

    Returns:
        Per-row encoding of shape ``(num_rows, out_dim)``, in the input row order.

    Raises:
        ValueError: If the index tensors are not matching one-dimensional ``long`` vectors.
    """
    if in_indices.ndim != 1 or in_indices.shape != out_indices.shape:
        raise ValueError(
            f"in_indices {tuple(in_indices.shape)} and out_indices "
            f"{tuple(out_indices.shape)} must be matching 1-D vectors"
        )
    if in_indices.dtype != th.long or out_indices.dtype != th.long:
        raise ValueError("cut-bond pair indices must have dtype torch.long")
    unique_pairs, row_to_unique = th.unique(
        th.stack((in_indices, out_indices)), dim=1, return_inverse=True
    )
    encoded = encoder(atom_embeddings[unique_pairs[0]], atom_embeddings[unique_pairs[1]])
    return encoded[row_to_unique]
