"""Check PyTorch CUDA availability and basic GPU health on a node.

This script is intended for quick HPC node screening before scheduling
expensive jobs. It prints key diagnostics and returns a non-zero exit code
when required GPU conditions are not met.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI options.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Return non-zero exit code when no CUDA GPU is available.",
    )
    parser.add_argument(
        "--min-gpu-count",
        type=int,
        default=1,
        help="Minimum CUDA device count when --require-gpu is enabled. Defaults to 1.",
    )
    return parser


def run_check(require_gpu: bool, min_gpu_count: int) -> int:
    """Run GPU availability checks.

    Args:
        require_gpu: Whether to fail if CUDA GPU is not available.
        min_gpu_count: Minimum number of required CUDA devices.

    Returns:
        Process exit code.
    """
    if min_gpu_count < 1:
        raise ValueError("--min-gpu-count must be >= 1")

    try:
        import torch
    except ImportError:
        LOGGER.exception("Failed to import torch")
        if require_gpu:
            return 2
        return 0

    host = socket.gethostname()
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    LOGGER.info("host: %s", host)
    LOGGER.info("torch version: %s", torch_version)
    LOGGER.info("cuda available: %s", cuda_available)
    LOGGER.info("cuda device count: %d", device_count)

    for idx in range(device_count):
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        total_memory_gb = props.total_memory / (1024**3)
        capability = f"{props.major}.{props.minor}"
        LOGGER.info(
            "gpu[%d]: %s | cc=%s | mem=%.1fGB",
            idx,
            name,
            capability,
            total_memory_gb,
        )

    if require_gpu and (not cuda_available or device_count < min_gpu_count):
        LOGGER.error(
            "GPU requirement failed (required >= %d, found %d).",
            min_gpu_count,
            device_count,
        )
        return 1

    return 0


def main() -> int:
    """CLI entrypoint.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()
    return run_check(require_gpu=args.require_gpu, min_gpu_count=args.min_gpu_count)


if __name__ == "__main__":
    sys.exit(main())
