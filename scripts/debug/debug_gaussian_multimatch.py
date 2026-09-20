"""Count how often a true peak matches more than one predicted formula.

The Gaussian mass-error term in the loss only differs from ``peak_marginal`` when a
single true peak has two or more predicted formulas inside its tolerance window,
because predicted m/z are exact theoretical formula masses (``spec_mzs =
frag_formula_peak_mzs``) and carry no gradient. With one candidate per window both
distributions reduce to the same per-peak log-probability.

Predicted m/z do not depend on model weights, so a randomly-initialised model
reproduces the real ``pred_mzs`` exactly and no checkpoint is needed.

Reports, over matched true peaks:
  - the fraction with >= 2 candidates, raw and weighted by true intensity
  - the spread of Gaussian log-weights within multi-match windows, which bounds how
    far Gaussian reweighting can move the loss away from peak_marginal

Example:
    python scripts/debug/debug_gaussian_multimatch.py \
        -c config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml -n 20
"""

import argparse
import math
from collections import Counter

import numpy as np
import torch as th

from fragnnet.runner import init_dataloader, init_dataset, load_config
from fragnnet.utils.setup_utils import get_model_cls
from fragnnet.utils.spec_utils import calculate_match_mzs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template_fp", type=str, default="config")
    parser.add_argument("-c", "--custom_fp", type=str, required=True)
    parser.add_argument("-n", "--num_batches", type=int, default=20)
    parser.add_argument("-b", "--eval_batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if th.cuda.is_available() else "cpu")
    args = parser.parse_args()

    override = [
        f"eval_batch_size={args.eval_batch_size}",
        "num_workers=0",
        "share_memory=False",
    ]
    config_d = load_config(args.template_fp, args.custom_fp, override)

    # Build the model before the dataset, matching init_run: init_dataset mutates the
    # shared nested config dicts, which changes the featuriser dims the model derives.
    device = th.device(args.device)
    model = get_model_cls(config_d["model_type"])(**config_d)
    model.eval()
    model.to(device)

    val_ds = init_dataset(config_d, splits=["val"])[0]
    val_dl = init_dataloader(val_ds, config_d)

    tolerance = model.tolerance
    relative = model.relative
    tolerance_min_mz = model.tolerance_min_mz
    # The loss routes a true peak to OOS unless a prediction lands inside
    # oos_tolerance_multiple * tolerance, so match on that same window.
    window = config_d["oos_tolerance_multiple"] * tolerance
    print(
        f"> matching window: {config_d['oos_tolerance_multiple']} x {tolerance} "
        f"({'relative' if relative else 'absolute'}, min_mz={tolerance_min_mz})"
    )

    count_hist: Counter = Counter()
    n_matched = 0
    n_multi = 0
    int_matched = 0.0
    int_multi = 0.0
    logw_spreads: list[float] = []

    with th.no_grad():
        for batch_idx, batch in enumerate(val_dl):
            if batch_idx >= args.num_batches:
                break
            batch = model.transfer_batch_to_device(batch, device, 0)

            true_mzs, true_logprobs, true_batch_idxs = model.preproc_spec(
                batch["spec_mzs"],
                batch["spec_ints"],
                batch["spec_batch_idxs"],
                train=True,
                pred=False,
                log_in=False,
                log_out=True,
            )
            pred_d = model.forward(**batch)
            pred_mzs, pred_logprobs, pred_batch_idxs = model.preproc_spec(
                pred_d["pred_mzs"],
                pred_d["pred_logprobs"],
                pred_d["pred_batch_idxs"],
                train=True,
                pred=True,
                log_in=True,
                log_out=True,
            )

            batch_size = int(true_batch_idxs.max().item() + 1)
            for b in range(batch_size):
                b_true_mzs = true_mzs[true_batch_idxs == b]
                b_pred_mzs = pred_mzs[pred_batch_idxs == b]
                if b_true_mzs.numel() == 0 or b_pred_mzs.numel() == 0:
                    continue
                b_true_ints = true_logprobs[true_batch_idxs == b].exp()

                match_mask = th.as_tensor(
                    calculate_match_mzs(
                        b_true_mzs,
                        b_pred_mzs,
                        tolerance=window,
                        relative=relative,
                        tolerance_min_mz=tolerance_min_mz,
                    ),
                    device=b_true_mzs.device,
                )
                counts = match_mask.sum(dim=1)
                matched = counts >= 1
                multi = counts >= 2

                for c in counts[matched].tolist():
                    count_hist[int(c)] += 1
                n_matched += int(matched.sum().item())
                n_multi += int(multi.sum().item())
                int_matched += float(b_true_ints[matched].sum().item())
                int_multi += float(b_true_ints[multi].sum().item())

                # Within a multi-match window the Gaussian weights candidate j by
                # -(true - pred_j)^2 / (2 sigma_j^2). peak_marginal weights all
                # candidates equally, so the spread of these log-weights is exactly
                # what the two distributions disagree about.
                for t in th.nonzero(multi, as_tuple=False).flatten().tolist():
                    cand = th.nonzero(match_mask[t], as_tuple=False).flatten()
                    cand_mzs = b_pred_mzs[cand]
                    if relative:
                        sigma = th.clamp(cand_mzs, min=tolerance_min_mz) * tolerance
                    else:
                        sigma = tolerance * th.ones_like(cand_mzs)
                    logw = -((b_true_mzs[t] - cand_mzs) ** 2) / (2 * sigma**2)
                    logw_spreads.append(float((logw.max() - logw.min()).item()))

    if n_matched == 0:
        raise RuntimeError("No matched true peaks found; check config and tolerance settings.")

    print(f"\n> matched true peaks:      {n_matched}")
    print(f"> multi-match (>=2 cands): {n_multi} ({100.0 * n_multi / n_matched:.3f}%)")
    print(f"> multi-match, intensity-weighted: {100.0 * int_multi / int_matched:.3f}%")
    print("\n> candidates per matched true peak:")
    for c in sorted(count_hist):
        print(f"    {c:>3}: {count_hist[c]:>8} ({100.0 * count_hist[c] / n_matched:.3f}%)")

    if logw_spreads:
        spreads = np.array(logw_spreads)
        print("\n> Gaussian log-weight spread within multi-match windows (nats):")
        for q in (50, 90, 99):
            print(f"    p{q}: {np.percentile(spreads, q):.4f}")
        print(f"    max: {spreads.max():.4f}")
        print(
            "\n  A spread of s nats means the Gaussian favours one candidate over another\n"
            f"  by up to exp(s) = {math.exp(min(spreads.max(), 20)):.2f}x relative to peak_marginal."
        )
    else:
        print("\n> No multi-match windows found: gaussian and peak_marginal are identical here.")


if __name__ == "__main__":
    main()
