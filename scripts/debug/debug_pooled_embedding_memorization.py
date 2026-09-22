"""Is the per-fragment mean pool MASShatter's memorization surface?

The candidate token is 1,285 columns and 1,152 of them (89.6%) are mean-pooled atom
embeddings: the fragment's own atoms, its complement, and its severed bonds. They are the
only per-candidate input that scales with molecular identity, and the cc atom mask alone
identifies the fragment uniquely (1.00 distinct fragments per descriptor bucket, measured in
``config/massshatter/ma_mi_merged/candidate_token_2026_09_22/e2_cc_atom_drop01.yml``). The
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

Validation status. The mechanics are verified end to end on a 128-molecule toy model
(``config/debug/debug_tiny_mass_shatter_cut_features.yml``, hidden 64, 60 epochs, the same
``inchikey_s`` split key production uses). That run is far too small to say anything about
the production model, but it does show the readout discriminates: corrupting the cc pools
left the train/validation gap unchanged (gap_closed -0.015) while corrupting the boundary
block closed 76-99% of it. A production checkpoint is required for the real answer.

Results and their interpretation:
``docs/findings/masshatter_head_input_asymmetry_2026_09_22.md`` (section 7.1b).

Usage:
    python scripts/debug/debug_pooled_embedding_memorization.py \
        --model_ckpt ckpts/RUN/last.ckpt \
        --custom_fp config/massshatter/ma_mi_merged/.../c4a_base_s1.yml \
        --batch_cutoff 20 \
        --output_fp pooled_memorization.csv
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch as th
from lightning import seed_everything
from tqdm import tqdm

from fragnnet.runner import init_dataloader, load_config
from fragnnet.utils.script_utils import (
    apply_inference_config,
    get_ckpt_path,
    init_model_from_ckpt,
    setup_device,
)
from fragnnet.utils.setup_utils import get_dataset_cls

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Positional layout of CandidateIntensityTransformerHead.forward, which MassShatterModel
# calls with nine positional arguments. Asserted against the live signature at install time.
FRAGMENT_ARG = 0
COMPLEMENT_ARG = 1
BOUNDARY_ARG = 2
BATCH_INDEX_ARG = 7
NUM_POSITIONAL_ARGS = 9

SCOPE_POSITIONS: dict[str, tuple[int, ...]] = {
    "cc": (FRAGMENT_ARG, COMPLEMENT_ARG),
    "boundary": (BOUNDARY_ARG,),
    "all": (FRAGMENT_ARG, COMPLEMENT_ARG, BOUNDARY_ARG),
}
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
        return SCOPE_POSITIONS[self.scope]


def build_conditions(scopes: list[str], modes: list[str]) -> list[Condition]:
    """Return the uncorrupted reference followed by every requested scope/mode pair."""
    unknown_scopes = [scope for scope in scopes if scope not in SCOPE_POSITIONS]
    if unknown_scopes:
        raise ValueError(f"Unknown corruption scopes: {unknown_scopes}")
    unknown_modes = [mode for mode in modes if mode not in MODES]
    if unknown_modes:
        raise ValueError(f"Unknown corruption modes: {unknown_modes}")
    conditions = [Condition("none", None, None)]
    conditions.extend(
        Condition(f"{scope}_{mode}", scope, mode) for scope in scopes for mode in modes
    )
    return conditions


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

    def __init__(self, head: th.nn.Module, seed: int) -> None:
        self.head = head
        self.seed = seed
        self._original_forward = head.forward
        self.condition = Condition("none", None, None)
        self.batch_index = 0
        self._installed = False

    def install(self) -> None:
        self._assert_argument_layout()
        self.head.forward = self._forward  # type: ignore[method-assign]
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

    def restore(self) -> None:
        if self._installed:
            self.head.forward = self._original_forward  # type: ignore[method-assign]
            self._installed = False

    def __enter__(self) -> PooledEmbeddingCorruptor:
        self.install()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.restore()

    def _forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.condition.is_identity:
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
        patched = list(args)
        for position in self.condition.positions:
            patched[position] = corrupt_block(
                patched[position],
                batch_indices,
                batch_size,
                self.condition.mode,
                generator,
            )
        return self._original_forward(*patched, **kwargs)


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


def evaluate_split(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    corruptor: PooledEmbeddingCorruptor,
    conditions: list[Condition],
    device: th.device,
    split: str,
    metrics: tuple[str, ...],
    batch_cutoff: int,
) -> dict[str, dict[str, float]]:
    """Evaluate every corruption condition on one split, paired batch by batch."""
    accumulators = {condition.label: WeightedMetricAccumulator(metrics) for condition in conditions}
    skipped = 0

    with th.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(dataloader, desc=f"pooled-corruption {split}", unit="batch")
        ):
            if 0 <= batch_cutoff <= batch_index:
                break
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            corruptor.batch_index = batch_index
            for condition in conditions:
                corruptor.condition = condition
                try:
                    batch_results = model._common_step(batch, split=split, log=False)
                except ValueError as error:
                    skipped += 1
                    logger.warning(
                        "Skipping batch %d under %s: %s", batch_index, condition.label, error
                    )
                    continue
                accumulators[condition.label].update(batch_results)
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
    parser.add_argument("--scopes", nargs="+", default=["cc", "boundary", "all"])
    parser.add_argument("--modes", nargs="+", default=list(MODES))
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
    if config_d["model_type"] != "mass_shatter":
        raise ValueError(f"Expected model_type=mass_shatter, got {config_d['model_type']}")
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

    rows: list[dict[str, Any]] = []
    with PooledEmbeddingCorruptor(model.model.intensity_head, config_d["seed"]) as corruptor:
        for split in args.splits:
            # Reseed per split so the batch sampler and any stochastic dataset step repeat.
            seed_everything(config_d["seed"], workers=True)
            dataset = dataset_cls(split=split, **config_d)
            dataloader = init_dataloader(dataset, config_d)
            results = evaluate_split(
                model=model,
                dataloader=dataloader,
                corruptor=corruptor,
                conditions=conditions,
                device=device,
                split=split,
                metrics=metrics,
                batch_cutoff=args.batch_cutoff,
            )
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
        "Read: gap_closed > 0 and train_drop > val_drop means the pooled block carried "
        "memorized signal. Equal drops mean it is a generalizing feature."
    )


if __name__ == "__main__":
    main()
