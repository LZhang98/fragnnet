"""Is the per-fragment mean pool MASSter's memorization surface?

The candidate token is 1,285 columns and 1,152 of them (89.6%) are mean-pooled atom
embeddings: the fragment's own atoms, its complement, and its severed bonds. They are the
only per-candidate input that scales with molecular identity, and the cc atom mask alone
identifies the fragment uniquely (1.00 distinct fragments per descriptor bucket, measured in
``config/masster/ma_mi_merged/candidate_token_2026_09_22/e2_cc_atom_drop01.yml``). The
2026-09-22 read-out puts ~98% of the 0.151 train/validation cosine gap in intensity
allocation -- the head this token feeds. So the hypothesis is that the head memorizes a
fragment-identity lookup that cannot transfer to unseen scaffolds.

This probe tests it WITHOUT training. One fixed checkpoint is evaluated on a train slice and
a validation slice, and the pooled blocks are corrupted on the way into the intensity head.
The signature of memorization is an ASYMMETRIC collapse: if the pool carries memorized
signal, destroying it costs the train split much more than the validation split, and the
train/validation gap shrinks. If both splits degrade alike, the pool is being used as a
generalizing feature and the gap lives somewhere else.

Why the scopes are separated. ``candidate_atom_dropout`` -- the only implemented
countermeasure -- perturbs the fragment and complement pools and deliberately leaves the
384-column boundary embedding exact (see ``_candidate_atom_dropout``'s docstring and its call
site, which passes raw ``fragment_atom_indices``). With ``boundary_bond_encoder: True``
(221/223 configs) that block encodes oriented (retained, lost) endpoint pairs for the severed
bonds, and the severed-bond set largely determines the fragment given the precursor. So a
null from the prepared ``e2``/``e3`` arms would be ambiguous. Running ``cc`` and ``boundary``
as separate scopes here sizes that leak before any GPU is spent on training.

Corruption modes. All three destroy which-candidate-is-which; they differ in what they put
in its place, and they are NOT ordered by severity -- read them together:

  molmean   replace each candidate's block by the mean over its molecule's candidates. Keeps
            molecule identity and candidate count, destroys WHICH candidate is which. The
            most conservative substitute: still an average of real pooled embeddings.
  shuffle   permute the block among candidates OF THE SAME MOLECULE. Preserves the marginal
            distribution exactly, so it cannot be dismissed as an off-manifold input -- but
            it substitutes a confidently WRONG candidate's embedding, and in the validation
            run below it damages more than zeroing does.
  zero      replace by zeros. Removes molecule identity too, but a zero vector is off the
            input manifold and the head appears to treat it closer to "no information" than
            to "wrong information". Do not read it as an upper bound on the block's value.

Every condition is evaluated on the SAME batches in the same order -- the conditions loop
runs inside the batch loop -- so the deltas are paired and the dataloader is walked once.
Shuffle draws are seeded per batch and shared across conditions, so ``cc_shuffle`` and
``all_shuffle`` apply the same permutation to the cc blocks.

Note that ``candidate_atom_dropout`` is training-only and the model is in eval mode here, so
it is inactive and cannot interact with these corruptions.

Two further scopes cover the head's other memorization routes, so one pass over a checkpoint
ranks every in-head route side by side:

  features  the candidate descriptor block (``masster_candidate_features``, e.g. cut
            count, size, mass, charge), corrupted with the same three modes. These columns
            are few but exact, and mass plus size alone can index a fragment within its
            molecule.
  set       the candidate-set fingerprint. ``set_drop`` hides a random ``--set_drop_fraction``
            of each molecule's candidates from the attention KEYS in every layer while still
            scoring them, so each candidate's own token is untouched and only the set context
            it is contextualized against changes. At least one candidate per molecule stays
            visible. If allocation is looked up from the exact candidate configuration, train
            loses more than validation here.

Validation status. The mechanics are verified end to end on a 128-molecule toy model
(``config/debug/debug_tiny_masster_cut_features.yml``, hidden 64, 60 epochs, the same
``inchikey_s`` split key production uses). That run is far too small to say anything about
the production model, but it does show the readout discriminates: corrupting the cc pools
left the train/validation gap unchanged (gap_closed -0.015) while corrupting the boundary
block closed 76-99% of it. A production checkpoint is required for the real answer.
The ``features`` and ``set`` scopes (added 2026-09-23) are verified only mechanically: unit
tests in ``tests/test_debug_pooled_embedding_memorization.py`` (``set_drop`` at fraction 0 is
exactly the identity under both attention types) and one end-to-end pass on a 6-epoch toy
checkpoint of the same config, which had not overfit and so says nothing about memorization.

Results and their interpretation:
``docs/findings/masster/masster_head_input_asymmetry_2026_09_22.md`` (section 7.1b).

Usage:
    python scripts/debug/masster_memorization/debug_pooled_embedding_memorization.py \
        --model_ckpt ckpts/RUN/last.ckpt \
        --custom_fp config/masster/ma_mi_merged/.../c4a_base_s1.yml \
        --batch_cutoff 20 \
        --output_fp pooled_memorization.csv
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch as th
from lightning import seed_everything
from tqdm import tqdm

import fragnnet.models.masster.masster_model as masster_model_module
from fragnnet.model.nn_blocks import GINE_GSA
from fragnnet.runner import init_dataloader, load_config
from fragnnet.utils.feat_utils import get_mol_feats_sizes
from fragnnet.utils.script_utils import (
    apply_inference_config,
    get_ckpt_path,
    init_model_from_ckpt,
    setup_device,
)
from fragnnet.utils.setup_utils import get_dataset_cls

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Positional layout of CandidateIntensityTransformerHead.forward, which MassterModel
# calls with seven positional arguments. Asserted against the live signature at install time.
FRAGMENT_ARG = 0
COMPLEMENT_ARG = 1
BOUNDARY_ARG = 2
BATCH_INDEX_ARG = 5
NUM_POSITIONAL_ARGS = 7

SCOPE_POSITIONS: dict[str, tuple[int, ...]] = {
    "cc": (FRAGMENT_ARG, COMPLEMENT_ARG),
    "boundary": (BOUNDARY_ARG,),
    "all": (FRAGMENT_ARG, COMPLEMENT_ARG, BOUNDARY_ARG),
}
# Blocks the head receives by keyword rather than position.
SCOPE_KWARGS: dict[str, tuple[str, ...]] = {"features": ("candidate_features",)}
# Not a block: masks candidates out of the attention keys (see the module docstring).
SET_SCOPE = "set"
SET_MODE = "drop"
# Scopes that rewrite only some descriptor columns, named by candidate feature.
FEATURE_COLUMN_SCOPES: dict[str, tuple[str, ...]] = {
    "feat_mass": ("c_mass",),
    "feat_size": ("c_cut_count", "c_size"),
    "feat_charge": ("c_charge",),
}
# Routes outside the token (memorization-route plan, Step 0), each with its own fixed modes:
#   gsa      the global-attention message in every GINE_GSA encoder layer, i.e. the
#            molecule-level part of every atom embedding. Only the intensity path sees the
#            rewritten embeddings; the proposal keeps the exact ones, so candidates are fixed.
#            popmean: the batch mean for every atom (identity removed); molmean: the
#            molecule's own mean (identity kept, per-atom detail removed); xmol: another
#            molecule's mean (identity swapped).
#   acharge  the continuous Gasteiger atom input, rounded to 0.1, through the same second pass.
#   ionh     the per-fragment ion/H logits: molmean gives every fragment its molecule's mean
#            H-shift pattern; popmean the batch mean.
#   ctx      the context vector the intensity head reads (collision energy alone on a
#            one-adduct, one-instrument task). popmean: the batch mean over spectra, so CE is
#            removed; shift_*: the context re-encoded at NCE + delta (ContextShifter). The
#            proposal keeps the exact context, so the candidate set is fixed and only the
#            allocation's CE input moves.
ENCODER_SCOPE = "gsa"
INPUT_SCOPE = "acharge"
OUTPUT_SCOPE = "ionh"
CONTEXT_SCOPE = "ctx"
CONTEXT_SHIFTS: dict[str, int] = {"shift_m5": -5, "shift_m1": -1, "shift_p1": 1, "shift_p5": 5}
#   side     one side of each cut: the heavier side's pool (the fragment when it keeps at least
#            half the precursor mass, else the complement) or the lighter side's, the neutral
#            loss of a large fragment. popmean: the batch mean of that side's rows (identity
#            removed); molmean: the molecule's mean (identity kept, which-candidate removed).
#            The precursor has no lighter side and is left alone by the small_* modes.
SIDE_SCOPE = "side"
SIDE_MASS_FEATURE = "c_mass"
#   setmsg   the message each candidate receives from its candidate set in every allocator
#            attention layer (the candidate_attention output, before the residual add).
#            popmean: the batch-mean message of that layer, recorded on the same batch's clean
#            pass (the head runs size buckets one after another, so no single call sees the
#            batch); molmean: the mean message of the candidate's own spectrum, which keeps the
#            set summary and removes candidate-specific retrieval from the set.
MESSAGE_SCOPE = "setmsg"
UPSTREAM_SCOPES = (ENCODER_SCOPE, INPUT_SCOPE)
FIXED_SCOPE_MODES: dict[str, tuple[str, ...]] = {
    SET_SCOPE: (SET_MODE,),
    ENCODER_SCOPE: ("popmean", "molmean", "xmol"),
    INPUT_SCOPE: ("round",),
    OUTPUT_SCOPE: ("molmean", "popmean"),
    CONTEXT_SCOPE: ("popmean", *CONTEXT_SHIFTS),
    SIDE_SCOPE: ("large_popmean", "large_molmean", "small_popmean", "small_molmean"),
    MESSAGE_SCOPE: ("popmean", "molmean"),
}
ACHARGE_FEATURE = "a_charge"
ACHARGE_ROUNDING = 0.1
# Positional layout of both candidate attention layers' forward.
LAYER_PADDING_MASK_ARG = 2
MODES = ("molmean", "shuffle", "zero")
BASE_METRICS = ("loss", "primary_loss")


@dataclass(frozen=True)
class Condition:
    """One corruption applied to the pooled blocks of the candidate token."""

    label: str
    scope: str | None
    mode: str | None

    @property
    def is_identity(self) -> bool:
        return self.scope is None

    @property
    def positions(self) -> tuple[int, ...]:
        if self.scope is None:
            return ()
        return SCOPE_POSITIONS.get(self.scope, ())

    @property
    def kwargs(self) -> tuple[str, ...]:
        if self.scope is None:
            return ()
        return SCOPE_KWARGS.get(self.scope, ())


def build_conditions(scopes: list[str], modes: list[str]) -> list[Condition]:
    """Return the uncorrupted reference followed by every requested scope/mode pair.

    The ``set``, ``gsa``, ``acharge``, ``ionh`` and ``ctx`` scopes have fixed modes of their
    own (``FIXED_SCOPE_MODES``) and ignore ``modes``.
    """
    known_scopes = {*SCOPE_POSITIONS, *SCOPE_KWARGS, *FEATURE_COLUMN_SCOPES, *FIXED_SCOPE_MODES}
    unknown_scopes = [scope for scope in scopes if scope not in known_scopes]
    if unknown_scopes:
        raise ValueError(f"Unknown corruption scopes: {unknown_scopes}")
    unknown_modes = [mode for mode in modes if mode not in MODES]
    if unknown_modes:
        raise ValueError(f"Unknown corruption modes: {unknown_modes}")
    conditions = [Condition("none", None, None)]
    for scope in scopes:
        scope_modes = FIXED_SCOPE_MODES.get(scope, tuple(modes))
        conditions.extend(Condition(f"{scope}_{mode}", scope, mode) for mode in scope_modes)
    return conditions


def corrupt_rows_by_molecule(
    rows: th.Tensor,
    batch_indices: th.Tensor,
    num_molecules: int,
    mode: str,
    generator: th.Generator,
) -> th.Tensor:
    """Replace every row by a per-molecule or batch-level mean.

    Args:
        rows: ``(num_rows, ...)`` values, e.g. atom embeddings or per-fragment ion/H logits.
        batch_indices: Molecule index of each row, shape ``(num_rows,)``.
        num_molecules: Number of molecules in the batch.
        mode: ``popmean`` (the batch mean for every row, so no molecule identity is left),
            ``molmean`` (the row's own molecule mean) or ``xmol`` (the mean of another
            molecule, fixed per batch by a cyclic shift, so identity is swapped, never kept).
        generator: CPU generator for the ``xmol`` shift.

    Returns:
        Tensor shaped like ``rows``.
    """
    flat = rows.reshape(rows.shape[0], -1)
    if mode == "popmean":
        return flat.mean(dim=0, keepdim=True).expand_as(flat).reshape(rows.shape).clone()
    sums = flat.new_zeros((num_molecules, flat.shape[1])).index_add(0, batch_indices, flat)
    counts = th.bincount(batch_indices, minlength=num_molecules).to(flat).clamp_min(1.0)
    means = sums / counts.unsqueeze(1)
    if mode == "molmean":
        return means[batch_indices].reshape(rows.shape)
    if mode == "xmol":
        if num_molecules < 2:
            raise ValueError("xmol needs at least two molecules in the batch")
        shift = int(th.randint(1, num_molecules, (1,), generator=generator))
        donor = (th.arange(num_molecules, device=batch_indices.device) + shift) % num_molecules
        return means[donor[batch_indices]].reshape(rows.shape)
    raise ValueError(f"Unknown per-molecule corruption mode: {mode}")


def corrupt_set_messages(
    messages: th.Tensor,
    padding_mask: th.Tensor,
    mode: str,
    population_mean: th.Tensor | None = None,
) -> th.Tensor:
    """Replace every real candidate's set message by a batch-level or per-spectrum mean.

    Args:
        messages: ``(num_spectra, num_slots, dim)`` attention output each candidate receives.
        padding_mask: ``(num_spectra, num_slots)`` bool, True on padded slots.
        mode: ``popmean`` (``population_mean`` for every real slot, so no identity is left)
            or ``molmean`` (the mean over the spectrum's real slots).
        population_mean: ``(dim,)`` batch-mean message; required for ``popmean``.

    Returns:
        Tensor shaped like ``messages``; padded slots are unchanged.
    """
    real = (~padding_mask).unsqueeze(-1)
    if mode == "popmean":
        if population_mean is None:
            raise ValueError("popmean needs the batch-mean message")
        replacement = population_mean.to(messages).expand_as(messages)
    elif mode == "molmean":
        weights = real.to(messages)
        replacement = (
            (messages * weights).sum(dim=1, keepdim=True)
            / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        ).expand_as(messages)
    else:
        raise ValueError(f"Unknown set-message corruption mode: {mode}")
    return th.where(real, replacement, messages)


def drop_from_keys(
    padding_mask: th.Tensor,
    fraction: float,
    generator: th.Generator,
) -> th.Tensor:
    """Mark a random fraction of each molecule's real candidates as padding.

    Args:
        padding_mask: ``(num_molecules, num_slots)`` bool, True on padded slots.
        fraction: Probability that a real candidate is hidden from the keys.
        generator: CPU generator for the draw.

    Returns:
        A mask of the same shape that is True on padded AND hidden slots. Each row keeps its
        first real candidate visible if the draw would have hidden all of them.
    """
    real = ~padding_mask
    draws = th.rand(padding_mask.shape, generator=generator).to(padding_mask.device)
    hidden = real & (draws < fraction)
    emptied = (real & ~hidden).sum(dim=1) == 0
    first_real = real.to(th.uint8).argmax(dim=1)
    rows = th.nonzero(emptied, as_tuple=False).flatten()
    hidden[rows, first_real[rows]] = False
    return padding_mask | hidden


def _within_molecule_permutation(
    batch_indices: th.Tensor,
    generator: th.Generator,
) -> tuple[th.Tensor, th.Tensor]:
    """Return (destination, source) index pairs permuting rows inside each molecule.

    ``destination`` lists every row grouped by molecule in ascending molecule order;
    ``source`` lists the same rows with the same grouping but a random order inside each
    molecule. Assigning ``out[destination] = x[source]`` therefore never moves a row across
    molecule boundaries, whatever order ``batch_indices`` arrives in.
    """
    num_rows = batch_indices.shape[0]
    random_order = th.argsort(th.rand(num_rows, generator=generator).to(batch_indices.device))
    source = random_order[th.argsort(batch_indices[random_order], stable=True)]
    destination = th.argsort(batch_indices, stable=True)
    return destination, source


def corrupt_block(
    block: th.Tensor,
    batch_indices: th.Tensor,
    batch_size: int,
    mode: str,
    generator: th.Generator,
) -> th.Tensor:
    """Destroy the per-candidate content of one pooled block."""
    if mode == "zero":
        return th.zeros_like(block)
    if mode == "molmean":
        sums = block.new_zeros((batch_size, block.shape[1])).index_add(0, batch_indices, block)
        counts = th.bincount(batch_indices, minlength=batch_size).to(block).clamp_min(1.0)
        return (sums / counts.unsqueeze(1))[batch_indices]
    if mode == "shuffle":
        destination, source = _within_molecule_permutation(batch_indices, generator)
        shuffled = th.empty_like(block)
        shuffled[destination] = block[source]
        return shuffled
    raise ValueError(f"Unknown corruption mode: {mode}")


class PooledEmbeddingCorruptor:
    """Rewrite the pooled blocks on their way into the candidate intensity head.

    Patching the head's ``forward`` keeps the probe entirely outside model code: the
    corruption sits exactly at the point of use, after pooling and before the token
    projection, so nothing upstream (candidate generation, proposal scoring, the scalar
    features) is disturbed.
    """

    def __init__(
        self,
        head: th.nn.Module,
        seed: int,
        set_drop_fraction: float,
        feature_columns: dict[str, tuple[int, int]] | None = None,
        context_shifter: ContextShifter | None = None,
        set_messages: bool = False,
    ) -> None:
        """Wrap ``head``.

        ``feature_columns`` maps each descriptor name to its column span; ``context_shifter``
        supplies the shifted context the ``ctx`` shift modes substitute. ``set_messages``
        hooks every attention layer's set message, for the ``setmsg`` scope.
        """
        if not 0.0 <= set_drop_fraction < 1.0:
            raise ValueError(f"set_drop_fraction must be in [0, 1), got {set_drop_fraction}")
        self.head = head
        self.seed = seed
        self.set_drop_fraction = set_drop_fraction
        self.feature_columns = feature_columns or {}
        self.context_shifter = context_shifter
        self._original_forward = head.forward
        self._original_layer_forwards = [layer.forward for layer in head.layers]
        self.condition = Condition("none", None, None)
        self.batch_index = 0
        self._installed = False
        self._generator: th.Generator | None = None
        # Every layer of one bucket receives the same padding-mask tensor; reuse one draw
        # for all of them so a hidden candidate stays hidden through the whole stack.
        self._mask_source: th.Tensor | None = None
        self._mask_value: th.Tensor | None = None
        # setmsg: the padding mask of the layer call in progress, and each layer's summed
        # real-slot messages from the current batch's clean pass.
        self.set_messages = set_messages
        self._message_hooks: list[Any] = []
        self._message_padding: th.Tensor | None = None
        self._message_batch: int | None = None
        self._message_sums: dict[int, th.Tensor] = {}
        self._message_counts: dict[int, th.Tensor] = {}

    def install(self) -> None:
        self._assert_argument_layout()
        self.head.forward = self._forward  # type: ignore[method-assign]
        for layer, original in zip(self.head.layers, self._original_layer_forwards):
            layer.forward = self._layer_forward(original)  # type: ignore[method-assign]
        if self.set_messages:
            self._message_hooks = [
                layer.candidate_attention.register_forward_hook(self._message_hook(index))
                for index, layer in enumerate(self.head.layers)
            ]
        self._installed = True

    def _assert_argument_layout(self) -> None:
        """Fail loudly if the head's signature no longer matches the patched positions."""
        parameters = list(inspect.signature(type(self.head).forward).parameters)[1:]
        expected = {
            FRAGMENT_ARG: "fragment_embeddings",
            COMPLEMENT_ARG: "complement_embeddings",
            BOUNDARY_ARG: "boundary_embeddings",
            BATCH_INDEX_ARG: "component_batch_indices",
        }
        for position, name in expected.items():
            if position >= len(parameters) or parameters[position] != name:
                raise ValueError(
                    f"Intensity head argument {position} is "
                    f"{parameters[position] if position < len(parameters) else '<missing>'}, "
                    f"expected {name}; update this probe's argument layout."
                )
        for names in SCOPE_KWARGS.values():
            missing = [name for name in names if name not in parameters]
            if missing:
                raise ValueError(f"Intensity head has no keyword arguments {missing}")
        for layer in self.head.layers:
            layer_parameters = list(inspect.signature(type(layer).forward).parameters)[1:]
            if layer_parameters[LAYER_PADDING_MASK_ARG] != "candidate_padding_mask":
                raise ValueError(
                    f"{type(layer).__name__} argument {LAYER_PADDING_MASK_ARG} is "
                    f"{layer_parameters[LAYER_PADDING_MASK_ARG]}, expected "
                    "candidate_padding_mask; update this probe's layer layout."
                )

    def restore(self) -> None:
        if self._installed:
            self.head.forward = self._original_forward  # type: ignore[method-assign]
            for layer, original in zip(self.head.layers, self._original_layer_forwards):
                layer.forward = original  # type: ignore[method-assign]
            for hook in self._message_hooks:
                hook.remove()
            self._message_hooks = []
            self._installed = False

    def __enter__(self) -> PooledEmbeddingCorruptor:
        self.install()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.restore()

    def _forward(self, *args: Any, **kwargs: Any) -> Any:
        # Upstream scopes rewrite the head's inputs before they arrive; see EncoderCorruptor.
        if self.condition.is_identity or self.condition.scope in UPSTREAM_SCOPES:
            return self._original_forward(*args, **kwargs)
        # setmsg rewrites inside the attention layers, through the hooks set up at install.
        if self.condition.scope == MESSAGE_SCOPE:
            if not self._message_hooks:
                raise RuntimeError("setmsg needs the corruptor built with set_messages=True")
            return self._original_forward(*args, **kwargs)
        if len(args) < NUM_POSITIONAL_ARGS:
            raise ValueError(
                f"Expected at least {NUM_POSITIONAL_ARGS} positional arguments into the "
                f"intensity head, got {len(args)}; the probe's argument layout is stale."
            )
        batch_indices = args[BATCH_INDEX_ARG]
        batch_size = kwargs["batch_size"]
        if batch_indices.ndim != 1 or batch_indices.dtype != th.long:
            raise ValueError("component_batch_indices is not the expected long vector")
        # One draw per batch, shared by every condition, so nested scopes stay comparable.
        generator = th.Generator().manual_seed(self.seed * 1_000_003 + self.batch_index)
        if self.condition.scope == SET_SCOPE:
            self._generator = generator
            self._mask_source = None
            try:
                return self._original_forward(*args, **kwargs)
            finally:
                self._generator = None
                self._mask_source = None
                self._mask_value = None
        if self.condition.scope == OUTPUT_SCOPE:
            outputs = list(self._original_forward(*args, **kwargs))
            ion_h_logits = corrupt_rows_by_molecule(
                outputs[2], batch_indices, batch_size, self.condition.mode, generator
            )
            outputs[2] = ion_h_logits
            # Without a separate support head the support logits ARE the ion/H logits.
            if kwargs.get("return_support_logits") and self.head.ion_h_support_head is None:
                outputs[3] = ion_h_logits
            return tuple(outputs)
        if self.condition.scope == CONTEXT_SCOPE:
            context = self._context(kwargs["context"], batch_indices, batch_size)
            return self._original_forward(*args, **{**kwargs, "context": context})
        if self.condition.scope == SIDE_SCOPE:
            patched = self._side(list(args), kwargs, batch_indices, batch_size, generator)
            return self._original_forward(*patched, **kwargs)
        if self.condition.scope in FEATURE_COLUMN_SCOPES:
            features = kwargs.get("candidate_features")
            names = FEATURE_COLUMN_SCOPES[self.condition.scope]
            missing = [name for name in names if name not in self.feature_columns]
            if features is None or missing:
                raise ValueError(
                    f"Scope {self.condition.scope} needs descriptor columns {list(names)}; this "
                    f"model's token has {sorted(self.feature_columns)}."
                )
            columns = th.tensor(
                [
                    column
                    for name in names
                    for column in range(*self.feature_columns[name])
                ],
                device=features.device,
            )
            patched_features = features.clone()
            patched_features[:, columns] = corrupt_block(
                features[:, columns], batch_indices, batch_size, self.condition.mode, generator
            )
            return self._original_forward(
                *args, **{**kwargs, "candidate_features": patched_features}
            )
        patched = list(args)
        for position in self.condition.positions:
            patched[position] = corrupt_block(
                patched[position],
                batch_indices,
                batch_size,
                self.condition.mode,
                generator,
            )
        patched_kwargs = dict(kwargs)
        for name in self.condition.kwargs:
            if patched_kwargs.get(name) is None:
                raise ValueError(
                    f"Scope {self.condition.scope} corrupts {name}, which this model does not "
                    "pass to the intensity head; drop the scope for this config."
                )
            patched_kwargs[name] = corrupt_block(
                patched_kwargs[name],
                batch_indices,
                batch_size,
                self.condition.mode,
                generator,
            )
        return self._original_forward(*patched, **patched_kwargs)

    def _side(
        self,
        args: list[Any],
        kwargs: dict[str, Any],
        batch_indices: th.Tensor,
        batch_size: int,
        generator: th.Generator,
    ) -> list[Any]:
        """Return the head's positional arguments with one side of every cut rewritten."""
        features = kwargs.get("candidate_features")
        if features is None or SIDE_MASS_FEATURE not in self.feature_columns:
            raise ValueError(f"side needs the {SIDE_MASS_FEATURE} descriptor to find each side")
        # c_mass is (log1p mass, mass / precursor mass); the second column picks the sides.
        fraction = features[:, self.feature_columns[SIDE_MASS_FEATURE][0] + 1]
        side, mode = self.condition.mode.split("_")
        fragment, complement = args[FRAGMENT_ARG], args[COMPLEMENT_ARG]
        if side == "large":
            use_fragment = fraction >= 0.5
            target = th.ones_like(use_fragment)
        else:
            use_fragment = fraction < 0.5
            target = fraction < 1.0 - 1e-4
        rows = th.where(use_fragment.unsqueeze(1), fragment, complement)[target]
        replaced = rows.new_zeros((fragment.shape[0], fragment.shape[1]))
        replaced[target] = corrupt_rows_by_molecule(
            rows, batch_indices[target], batch_size, mode, generator
        )
        on_fragment = (target & use_fragment).unsqueeze(1)
        on_complement = (target & ~use_fragment).unsqueeze(1)
        args[FRAGMENT_ARG] = th.where(on_fragment, replaced, fragment)
        args[COMPLEMENT_ARG] = th.where(on_complement, replaced, complement)
        return args

    def _context(
        self, context: th.Tensor | None, batch_indices: th.Tensor, batch_size: int
    ) -> th.Tensor:
        """Return the ``ctx`` substitute for the per-candidate context rows."""
        if context is None:
            raise ValueError("ctx needs an intensity head that reads the context vector")
        if self.condition.mode == "popmean":
            # Rows repeat their spectrum's context, so average per spectrum first: the batch
            # mean then weights spectra equally rather than by candidate count.
            sums = context.new_zeros((batch_size, context.shape[1])).index_add(
                0, batch_indices, context
            )
            counts = th.bincount(batch_indices, minlength=batch_size)
            present = counts > 0
            per_spectrum = sums[present] / counts[present].unsqueeze(1).to(context)
            return per_spectrum.mean(dim=0, keepdim=True).expand_as(context).clone()
        shifted = None if self.context_shifter is None else self.context_shifter.shifted
        if shifted is None:
            raise RuntimeError(
                f"{self.condition.label} found no shifted context; the ContextShifter is not "
                "installed or the context encoder was not called this step"
            )
        return shifted[batch_indices]

    def _message_hook(self, index: int) -> Any:
        """Return the forward hook on layer ``index``'s set-message module."""

        def hook(module: th.nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            messages = output[0] if isinstance(output, tuple) else output
            padding_mask = self._message_padding
            if self.condition.is_identity:
                # The clean pass records each layer's batch-mean message, bucket by bucket.
                if self._message_batch != self.batch_index:
                    self._message_batch = self.batch_index
                    self._message_sums, self._message_counts = {}, {}
                real = (~padding_mask).unsqueeze(-1).to(messages)
                self._message_sums[index] = self._message_sums.get(index, 0.0) + (
                    messages * real
                ).sum(dim=(0, 1))
                self._message_counts[index] = self._message_counts.get(index, 0.0) + real.sum()
                return None
            if self.condition.scope != MESSAGE_SCOPE:
                return None
            population_mean = None
            if self.condition.mode == "popmean":
                if self._message_batch != self.batch_index or index not in self._message_sums:
                    raise RuntimeError(
                        "setmsg_popmean needs the clean 'none' pass on the same batch first"
                    )
                population_mean = self._message_sums[index] / self._message_counts[index]
            rewritten = corrupt_set_messages(
                messages, padding_mask, self.condition.mode, population_mean
            )
            return (rewritten, *output[1:]) if isinstance(output, tuple) else rewritten

        return hook

    def _layer_forward(self, original: Any) -> Any:
        def forward(*args: Any, **kwargs: Any) -> Any:
            self._message_padding = args[LAYER_PADDING_MASK_ARG]
            if self._generator is None:
                return original(*args, **kwargs)
            padding_mask = args[LAYER_PADDING_MASK_ARG]
            if padding_mask is not self._mask_source:
                self._mask_source = padding_mask
                self._mask_value = drop_from_keys(
                    padding_mask, self.set_drop_fraction, self._generator
                )
            patched = list(args)
            patched[LAYER_PADDING_MASK_ARG] = self._mask_value
            return original(*patched, **kwargs)

        return forward


class EncoderCorruptor:
    """Rewrite the atom embeddings the intensity path reads; the proposal keeps the exact ones.

    MASSter encodes each precursor once. The proposal scores bonds on a detached copy of
    those embeddings, and the intensity path pools them into the fragment, complement and
    boundary blocks. Under ``gsa`` or ``acharge`` the encoder runs twice: the exact output goes
    everywhere as usual, and a rewritten one replaces it in exactly two places, the component
    pool and the boundary block. Candidate generation therefore does not move, and the
    measured change is the intensity allocation alone. ``gsa`` rewrites the global-attention
    message of every GINE_GSA layer through a forward hook on that layer's ``gsa_norms``
    module, whose output is the message added to the local one.
    """

    def __init__(self, model: th.nn.Module, seed: int, charge_column: int | None) -> None:
        """Wrap the ``MassterModel``; ``charge_column`` is the ``a_charge`` input column."""
        self.model = model
        self.seed = seed
        self.charge_column = charge_column
        self.condition = Condition("none", None, None)
        self.batch_index = 0
        self._installed = False
        self._hooks: list[Any] = []
        self._clean: th.Tensor | None = None
        self._rewritten: th.Tensor | None = None
        self._atom_batch: th.Tensor | None = None
        self._num_molecules = 0
        self._rewrite_global_message = False
        self._swaps = 0
        self._original_encoder_forward: Any = None
        self._original_pool: Any = None
        self._original_boundary: Any = None

    def install(self) -> None:
        if self.model.boundary_bond_encoder is None or getattr(
            self.model, "boundary_local_encoder", None
        ) is not None:
            raise ValueError(
                "gsa/acharge scopes need the boundary block built by CutBondEncoder from the "
                "shared encoder; this model builds it another way"
            )
        if self.model.proposal_encoder is not None or self.model.share_proposal_across_ev:
            raise ValueError("gsa/acharge scopes assume one shared encoder pass per batch")
        encoder = self.model.precursor_encoder
        self._original_encoder_forward = encoder.forward
        encoder.forward = self._encoder_forward  # type: ignore[method-assign]
        self._original_pool = masster_model_module.pool_component_and_complement_embeddings
        masster_model_module.pool_component_and_complement_embeddings = self._pool
        self._original_boundary = self.model._boundary_embeddings
        self.model._boundary_embeddings = self._boundary  # type: ignore[method-assign]
        if isinstance(encoder.gnn, GINE_GSA):
            self._hooks = [
                norm.register_forward_hook(self._global_message_hook)
                for norm in encoder.gnn.gsa_norms
            ]
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        self.model.precursor_encoder.forward = self._original_encoder_forward
        masster_model_module.pool_component_and_complement_embeddings = self._original_pool
        self.model._boundary_embeddings = self._original_boundary
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._installed = False

    def __enter__(self) -> EncoderCorruptor:
        self.install()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.restore()

    def verify(self, condition: Condition) -> None:
        """Fail if an upstream condition did not reach both intensity-path consumers."""
        if condition.scope in UPSTREAM_SCOPES and self._swaps < 2:
            raise RuntimeError(
                f"{condition.label} replaced the atom embeddings in {self._swaps} of the 2 "
                "intensity-path consumers; the probe no longer matches the model's forward"
            )

    def _generator(self) -> th.Generator:
        # Rebuilt per call so both encoder layers draw the same xmol donor.
        return th.Generator().manual_seed(self.seed * 1_000_003 + self.batch_index)

    def _encoder_forward(self, x: th.Tensor, batch_index: th.Tensor, *args: Any) -> th.Tensor:
        self._clean = None
        self._rewritten = None
        self._swaps = 0
        clean = self._original_encoder_forward(x, batch_index, *args)
        if self.condition.scope not in UPSTREAM_SCOPES:
            return clean
        inputs = x
        if self.condition.scope == INPUT_SCOPE:
            if self.charge_column is None:
                raise RuntimeError(f"acharge needs {ACHARGE_FEATURE} in pyg_node_feats")
            inputs = x.clone()
            inputs[:, self.charge_column] = (
                th.round(x[:, self.charge_column] / ACHARGE_ROUNDING) * ACHARGE_ROUNDING
            )
        elif not self._hooks:
            raise RuntimeError("gsa needs a GINE_GSA encoder")
        self._atom_batch = batch_index
        self._num_molecules = int(batch_index.max().item()) + 1
        self._rewrite_global_message = self.condition.scope == ENCODER_SCOPE
        try:
            rewritten = self._original_encoder_forward(inputs, batch_index, *args)
        finally:
            self._rewrite_global_message = False
        self._clean, self._rewritten = clean, rewritten
        return clean

    def _global_message_hook(
        self, module: th.nn.Module, inputs: tuple[Any, ...], output: th.Tensor
    ) -> th.Tensor | None:
        if not self._rewrite_global_message:
            return None
        return corrupt_rows_by_molecule(
            output, self._atom_batch, self._num_molecules, self.condition.mode, self._generator()
        )

    def _swap(self, atom_embeddings: th.Tensor) -> th.Tensor:
        if self._clean is not None and atom_embeddings is self._clean:
            self._swaps += 1
            return self._rewritten
        return atom_embeddings

    def _pool(self, atom_embeddings: th.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self._original_pool(self._swap(atom_embeddings), *args, **kwargs)

    def _boundary(self, mol_pyg: Any, atom_embeddings: th.Tensor, *args: Any) -> th.Tensor:
        return self._original_boundary(mol_pyg, self._swap(atom_embeddings), *args)


class ContextShifter:
    """Re-encode each spectrum's context at a shifted NCE, for the intensity head alone.

    MASSter encodes the acquisition context once per batch and passes the same tensor to the
    proposal and to the intensity head. The wrapper returns the exact context, so the proposal
    and the candidate set do not move, and keeps a second encoding at ``NCE + delta`` (clamped
    to ``[1, nce_max]``) that ``PooledEmbeddingCorruptor`` substitutes into the head.
    """

    def __init__(self, context_encoder: th.nn.Module) -> None:
        self.context_encoder = context_encoder
        self.condition = Condition("none", None, None)
        self.batch_index = 0
        self.shifted: th.Tensor | None = None
        self._original_forward = context_encoder.forward
        self._installed = False

    def install(self) -> None:
        if not self.context_encoder.use_nce:
            raise ValueError("ctx shifts need NCE in the context (use_nce: true)")
        self.context_encoder.forward = self._forward  # type: ignore[method-assign]
        self._installed = True

    def restore(self) -> None:
        if self._installed:
            self.context_encoder.forward = self._original_forward  # type: ignore[method-assign]
            self._installed = False

    def __enter__(self) -> ContextShifter:
        self.install()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.restore()

    def _forward(self, *args: Any, **kwargs: Any) -> Any:
        context = self._original_forward(*args, **kwargs)
        self.shifted = None
        if self.condition.scope != CONTEXT_SCOPE or self.condition.mode not in CONTEXT_SHIFTS:
            return context
        if args or not isinstance(context, th.Tensor) or kwargs["nce"] is None:
            raise ValueError(
                "ctx shifts need a keyword-only context call with NCE that returns one tensor "
                "(no CE-conditioned attention features)"
            )
        nce = (kwargs["nce"] + CONTEXT_SHIFTS[self.condition.mode]).clamp(
            min=1.0, max=float(self.context_encoder.nce_max)
        )
        self.shifted = self._original_forward(**{**kwargs, "nce": nce})
        return context


def candidate_feature_columns(specs: Any, edge_feature_dim: int) -> dict[str, tuple[int, int]]:
    """Return each descriptor's ``[start, stop)`` column span in the candidate-feature block."""
    columns: dict[str, tuple[int, int]] = {}
    start = 0
    for spec in specs:
        width = spec.width(edge_feature_dim)
        columns[spec.name] = (start, start + width)
        start += width
    return columns


def atom_feature_column(atom_feats: list[str], bond_feats: list[str], name: str) -> int | None:
    """Return the first column of a width-1 atom feature in the node matrix, or ``None``."""
    if name not in atom_feats:
        return None
    offset, _ = get_mol_feats_sizes(atom_feats[: atom_feats.index(name)], bond_feats, 0)
    return offset


class WeightedMetricAccumulator:
    """Accumulate sample-weighted metric means without retaining predictions."""

    def __init__(self, metric_names: tuple[str, ...]) -> None:
        self.requested = metric_names
        self.metric_names: tuple[str, ...] | None = None
        self.weighted_sums: dict[str, float] = {}
        self.total_weight = 0.0
        self.sample_count = 0

    def update(self, batch_results: dict[str, Any]) -> None:
        # Unweighted means keep the probe's train and validation slices comparable; the
        # per-group reweighting the training loop applies is a different question.
        # `_common_step` returns one entry per sample in `spec_per_group`, not a batch size.
        batch_size = int(batch_results["spec_per_group"].reshape(-1).numel())
        if self.metric_names is None:
            # A config may not produce every requested score; fix the set on the first batch
            # so a missing optional metric cannot silently change the denominator later.
            self.metric_names = tuple(
                name
                for name in self.requested
                if isinstance(batch_results.get(name), th.Tensor)
            )
            missing = set(self.requested) - set(self.metric_names)
            if missing:
                logger.warning("Metrics absent from this run's output: %s", sorted(missing))
        for metric_name in self.metric_names:
            values = batch_results[metric_name].detach().to(dtype=th.float64).reshape(-1)
            if values.numel() != batch_size:
                raise ValueError(
                    f"Metric {metric_name} has {values.numel()} values for a "
                    f"{batch_size}-sample batch."
                )
            self.weighted_sums[metric_name] = self.weighted_sums.get(
                metric_name, 0.0
            ) + float(values.sum().cpu())
        self.total_weight += batch_size
        self.sample_count += batch_size

    def means(self) -> dict[str, float]:
        if self.total_weight <= 0:
            raise ValueError("No evaluation samples were accumulated.")
        return {name: total / self.total_weight for name, total in self.weighted_sums.items()}


def metric_names(config_d: dict[str, Any]) -> tuple[str, ...]:
    """Return the binned cosine metrics plus whatever match scores are configured."""
    auxiliary_scores = set(config_d["auxiliary_scores"])
    match_names = tuple(
        name for name in ("precision", "recall", "wprecision", "wrecall")
        if name in auxiliary_scores
    )
    cosine_names = (
        tuple(f"cos_sim_{resolution}" for resolution in config_d["eval_mz_bin_res"])
        if "cos_sim" in auxiliary_scores
        else ()
    )
    if not cosine_names:
        raise ValueError("This probe reports cosine; add cos_sim to auxiliary_scores")
    return BASE_METRICS + match_names + cosine_names


def spectrum_frame(
    batch_results: dict[str, Any], metrics: tuple[str, ...], split: str, label: str
) -> pd.DataFrame:
    """Return one row per spectrum of a batch: its ``unique_id`` and every metric value."""
    unique_ids = batch_results["unique_id"].reshape(-1).cpu().numpy()
    frame = pd.DataFrame({"split": split, "condition": label, "spec_id": unique_ids})
    for name in metrics:
        values = batch_results[name].detach().to(dtype=th.float64).reshape(-1).cpu().numpy()
        if values.shape != unique_ids.shape:
            raise ValueError(
                f"Metric {name} has {values.shape[0]} values for {unique_ids.shape[0]} spectra"
            )
        frame[name] = values
    return frame


def evaluate_split(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    corruptors: list[PooledEmbeddingCorruptor | EncoderCorruptor | ContextShifter],
    conditions: list[Condition],
    device: th.device,
    split: str,
    metrics: tuple[str, ...],
    batch_cutoff: int,
    per_spectrum: list[pd.DataFrame] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate every corruption condition on one split, paired batch by batch.

    When ``per_spectrum`` is a list, one ``spectrum_frame`` per (batch, condition) is appended.
    """
    accumulators = {condition.label: WeightedMetricAccumulator(metrics) for condition in conditions}
    skipped = 0

    with th.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(dataloader, desc=f"pooled-corruption {split}", unit="batch")
        ):
            if 0 <= batch_cutoff <= batch_index:
                break
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            for corruptor in corruptors:
                corruptor.batch_index = batch_index
            for condition in conditions:
                for corruptor in corruptors:
                    corruptor.condition = condition
                try:
                    batch_results = model._common_step(batch, split=split, log=False)
                except ValueError as error:
                    skipped += 1
                    logger.warning(
                        "Skipping batch %d under %s: %s", batch_index, condition.label, error
                    )
                    continue
                for corruptor in corruptors:
                    if isinstance(corruptor, EncoderCorruptor):
                        corruptor.verify(condition)
                accumulators[condition.label].update(batch_results)
                if per_spectrum is not None:
                    per_spectrum.append(
                        spectrum_frame(
                            batch_results,
                            accumulators[condition.label].metric_names,
                            split,
                            condition.label,
                        )
                    )
            for corruptor in corruptors:
                corruptor.condition = Condition("none", None, None)

    if skipped:
        logger.warning("%d (batch, condition) evaluations were skipped", skipped)
    return {label: accumulator.means() for label, accumulator in accumulators.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template_fp", default="config")
    parser.add_argument("--custom_fp", required=True, help="run config; mandatory, not optional")
    parser.add_argument("--model_ckpt", default=None)
    parser.add_argument("--wandb_run_id", default=None)
    parser.add_argument("--model_save_dp", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["cc", "boundary", "all", "features", SET_SCOPE],
        help=(
            "block scopes (cc, boundary, all, features, feat_mass, feat_size, feat_charge) take "
            "--modes; set, gsa, acharge and ionh use their own fixed modes"
        ),
    )
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument(
        "--set_drop_fraction",
        type=float,
        default=0.5,
        help="fraction of each molecule's candidates hidden from attention keys under set_drop",
    )
    parser.add_argument(
        "--batch_cutoff",
        type=int,
        default=25,
        help="batches per split; -1 walks the whole split",
    )
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_fp", default="pooled_embedding_memorization.csv")
    parser.add_argument(
        "--per_spectrum_fp",
        default=None,
        help="optional pickle of every spectrum's metrics under every condition",
    )
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = build_conditions(list(args.scopes), list(args.modes))

    checkpoint_fp = get_ckpt_path(
        wandb_run_id=args.wandb_run_id,
        model_ckpt=args.model_ckpt,
        model_save_dp=args.model_save_dp,
        use_cached=True,
    )
    config_d = load_config(args.template_fp, args.custom_fp, args.override)
    if config_d["model_type"] != "masster":
        raise ValueError(f"Expected model_type=masster, got {config_d['model_type']}")
    config_d["auxiliary_scores"] = list(
        dict.fromkeys([*config_d["auxiliary_scores"], "cos_sim", "wrecall"])
    )
    # Without this, `_get_active_metric_names` drops every auxiliary metric on the train
    # split and the probe's train column is silently empty -- the same gate that left train
    # cosine unlogged in 232 of 232 runs. Turning it on costs memory, not correctness:
    # `calculate_match_mzs` builds a dense |true| x |pred| product, which on production-scale
    # batches took peak GPU from ~26 GiB to 36-53 GiB. Lower --eval_batch_size if it OOMs.
    config_d["compute_train_auxiliary_metrics"] = True
    config_d = apply_inference_config(
        config_d,
        disable_preproc=True,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    device = setup_device(config_d, device=args.device)
    dataset_cls = get_dataset_cls(config_d["model_type"])
    metrics = metric_names(config_d)

    seed_everything(config_d["seed"], workers=True)
    model = init_model_from_ckpt(checkpoint_fp, config_d, device)
    model.eval()

    # Descriptor column spans and the a_charge input column, for the column and input scopes.
    atom_feats = list(config_d["mol_params"]["pyg_node_feats"])
    bond_feats = list(config_d["mol_params"]["pyg_edge_feats"])
    _, edge_feature_dim = get_mol_feats_sizes(atom_feats, bond_feats, 0)
    feature_columns = candidate_feature_columns(
        model.model.candidate_feature_specs, edge_feature_dim
    )
    for scope in set(args.scopes) & set(FEATURE_COLUMN_SCOPES):
        missing = [name for name in FEATURE_COLUMN_SCOPES[scope] if name not in feature_columns]
        if missing:
            raise ValueError(f"Scope {scope} needs descriptors {missing}; the token has none")
    if INPUT_SCOPE in args.scopes and config_d["mol_params"]["pyg_pe_embed_k"] != 0:
        raise ValueError("acharge assumes no positional-encoding columns (pyg_pe_embed_k: 0)")
    charge_column = atom_feature_column(atom_feats, bond_feats, ACHARGE_FEATURE)

    rows: list[dict[str, Any]] = []
    per_spectrum: list[pd.DataFrame] | None = [] if args.per_spectrum_fp else None
    with ExitStack() as stack:
        context_shifter = (
            stack.enter_context(ContextShifter(model.model.context_encoder))
            if CONTEXT_SCOPE in args.scopes
            else None
        )
        corruptors: list[PooledEmbeddingCorruptor | EncoderCorruptor | ContextShifter] = [
            stack.enter_context(
                PooledEmbeddingCorruptor(
                    model.model.intensity_head,
                    config_d["seed"],
                    args.set_drop_fraction,
                    feature_columns=feature_columns,
                    context_shifter=context_shifter,
                    set_messages=MESSAGE_SCOPE in args.scopes,
                )
            )
        ]
        if context_shifter is not None:
            corruptors.append(context_shifter)
        # The encoder patch is installed only when asked for; it refuses unsupported models.
        if set(args.scopes) & set(UPSTREAM_SCOPES):
            corruptors.append(
                stack.enter_context(EncoderCorruptor(model.model, config_d["seed"], charge_column))
            )
        for split in args.splits:
            # Reseed per split so the batch sampler and any stochastic dataset step repeat.
            seed_everything(config_d["seed"], workers=True)
            dataset = dataset_cls(split=split, **config_d)
            dataloader = init_dataloader(dataset, config_d)
            results = evaluate_split(
                model=model,
                dataloader=dataloader,
                corruptors=corruptors,
                conditions=conditions,
                device=device,
                split=split,
                metrics=metrics,
                batch_cutoff=args.batch_cutoff,
                per_spectrum=per_spectrum,
            )
            if per_spectrum:
                pd.concat(per_spectrum, ignore_index=True).to_pickle(args.per_spectrum_fp)
            for condition in conditions:
                rows.append(
                    {
                        "split": split,
                        "condition": condition.label,
                        "scope": condition.scope or "none",
                        "mode": condition.mode or "none",
                        "checkpoint": os.path.abspath(checkpoint_fp),
                        **results[condition.label],
                    }
                )
            pd.DataFrame(rows).to_csv(args.output_fp, index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_fp, index=False)
    report(frame, metrics)
    logger.info("Saved %d rows to %s", len(frame), args.output_fp)


def report(frame: pd.DataFrame, metrics: tuple[str, ...]) -> None:
    """Print the asymmetry the probe exists to measure."""
    cosine = next(name for name in metrics if name.startswith("cos_sim_"))
    table = frame.pivot(index="condition", columns="split", values=cosine)
    if not {"train", "val"}.issubset(table.columns):
        logger.info("\n%s", table.to_string())
        return
    table["gap"] = table["train"] - table["val"]
    reference = table.loc["none"]
    table["train_drop"] = reference["train"] - table["train"]
    table["val_drop"] = reference["val"] - table["val"]
    # The memorization signature: corruption costs train more than val, so the gap closes.
    table["gap_closed"] = reference["gap"] - table["gap"]
    logger.info("\n%s\n%s", cosine, table.to_string(float_format=lambda v: f"{v:.4f}"))
    logger.info(
        "Read: gap_closed > 0 and train_drop > val_drop means the corrupted route carried "
        "memorized signal. Equal drops mean it is a generalizing feature."
    )


if __name__ == "__main__":
    main()
