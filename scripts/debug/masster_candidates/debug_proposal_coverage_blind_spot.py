"""What does MASSter's proposal coverage loss reward, exactly?

``_proposal_coverage_loss`` sums ``-w_p * log(coverage_p)`` over the observed peaks that at
least one proposed fragment reaches, where ``w_p`` is the peak's observed intensity and
``coverage_p`` is the total probability of the distinct fragments whose m/z fall inside the
tolerance. Peaks no fragment reaches are dropped by a ``th.where`` on the covered mask, not
penalized.

Because ``coverage_p <= 1``, every term is nonnegative, so the loss is minimized at 0 -- and 0
is attained two ways: every covered peak covered with probability 1, or **no peak covered at
all**. This probe makes that concrete on a two-peak toy spectrum and also measures the
weighting consequence, which is the less obvious half: since the covered peaks' intensities
are the weights, the loss is exactly proportional to the covered intensity fraction, so a
spectrum whose candidate set misses most of its signal contributes a proportionally smaller
gradient than one whose candidates already work.

No model, checkpoint or dataset is involved; this calls the production loss function directly
with hand-built tensors. Results are recorded in
``docs/findings/masster/masster_proposal_loss_audit_2026_09_22.md`` section 3.
"""

from __future__ import annotations

import logging

import torch as th

from fragnnet.models.masster.masster_pl import _proposal_coverage_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 0.01 Da absolute, so the toy m/z values below are unambiguous. The production path uses a
# relative tolerance from ``inst_type_loss_tol``; the blind spot is independent of both.
TOLERANCE_DA = 0.01


def _case(
    name: str,
    true_mzs: list[float],
    true_ints: list[float],
    candidate_mzs: list[float],
    candidate_fragment_idxs: list[int],
    fragment_probs: list[float],
) -> tuple[float, float]:
    """Evaluate one toy spectrum and return its loss and total gradient magnitude.

    Args:
        name: Label for the logged row.
        true_mzs: Observed peak m/z values of the single spectrum.
        true_ints: Observed intensities, which are the per-peak loss weights.
        candidate_mzs: Proposed channel m/z values.
        candidate_fragment_idxs: Fragment index behind each proposed channel.
        fragment_probs: Probability of each distinct proposed fragment.

    Returns:
        The scalar loss and the summed absolute gradient with respect to the fragment logits.
    """
    fragment_logprobs = th.tensor(fragment_probs).log().requires_grad_(True)
    loss = _proposal_coverage_loss(
        true_mzs=th.tensor(true_mzs),
        true_logprobs=th.tensor(true_ints).log(),
        true_batch_idxs=th.zeros(len(true_mzs), dtype=th.long),
        candidate_mzs=th.tensor(candidate_mzs),
        candidate_fragment_idxs=th.tensor(candidate_fragment_idxs),
        fragment_logprobs=fragment_logprobs,
        fragment_batch_idxs=th.zeros(len(fragment_probs), dtype=th.long),
        batch_size=1,
        tolerance=TOLERANCE_DA,
        relative=False,
        tolerance_min_mz=0.0,
    )
    loss.sum().backward()
    grad_magnitude = float(fragment_logprobs.grad.abs().sum())
    logger.info("%-46s loss=%9.6f  sum|grad|=%.3f", name, float(loss), grad_magnitude)
    return float(loss), grad_magnitude


def main() -> None:
    """Run the six toy spectra and state what the pattern means."""
    two_peaks = [100.0, 200.0]
    equal = [0.5, 0.5]

    logger.info("Coverage cases, two equal peaks at 100 and 200 Da:")
    _case("A  full coverage, P=0.5 each", two_peaks, equal, two_peaks, [0, 1], [0.5, 0.5])
    _case("B  half coverage, all probability on the hit", two_peaks, equal, [100.0], [0], [1.0])
    _case("C  ZERO coverage (sole candidate at 300)", two_peaks, equal, [300.0], [0], [1.0])
    _case("E  one fragment covers both peaks, P=1", two_peaks, equal, two_peaks, [0, 0], [1.0])

    # Same proposal as case A, with intensity moved onto a peak no candidate can reach.
    logger.info("Weighting cases, same proposal as A, intensity moved to an unreachable peak:")
    three_peaks = [100.0, 200.0, 300.0]
    proposal = (two_peaks, [0, 1], [0.5, 0.5])
    _case("F  20% of intensity unreachable", three_peaks, [0.4, 0.4, 0.2], *proposal)
    _case("G  60% of intensity unreachable", three_peaks, [0.2, 0.2, 0.6], *proposal)

    logger.info(
        "Read: case C is the global minimum -- a candidate set that reaches nothing is a "
        "zero-loss, zero-gradient training example, indistinguishable by loss value from "
        "case E. F and G show the loss scales exactly with the covered intensity fraction "
        "(0.693 -> 0.555 -> 0.277 at 100%, 80%, 40% covered), so the objective down-weights "
        "the molecules whose coverage is worst."
    )


if __name__ == "__main__":
    main()
