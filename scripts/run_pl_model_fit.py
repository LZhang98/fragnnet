import argparse
import logging
import os

from fragnnet.runner import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--template_fp",
        type=str,
        default="config",
        help="path to template config file",
    )
    parser.add_argument("-c", "--custom_fp", type=str, required=False)
    parser.add_argument(
        "-w",
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("-j", "--job_id", type=str, required=False)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. model.lr=1e-4 train.batch_size=64",
    )

    args = parser.parse_args()

    # Must be set before the CUDA allocator initialises (first tensor.cuda() call).
    # setdefault lets a SLURM-level export override this if needed.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    model = run_training(
        args.template_fp, args.custom_fp, args.wandb_mode, args.job_id, args.override
    )
