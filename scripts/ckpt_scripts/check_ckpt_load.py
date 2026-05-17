"""Check that FraGNNet checkpoints load cleanly.

For each checkpoint:
  1. Reads hyper_parameters and rebuilds the model.
  2. Loads the state dict (strict=True, falls back to strict=False).
  3. Reports missing / unexpected keys and non-finite parameters.

Usage:
    # All *.ckpt under saved_ckpts/
    python scripts/debug/check_ckpt_load.py

    # Explicit paths or directories
    python scripts/debug/check_ckpt_load.py saved_ckpts/model.ckpt saved_ckpts/deepmet/

    # Recurse into a directory
    python scripts/debug/check_ckpt_load.py --dir saved_ckpts/

    # Dry-run (skip model build / state-dict load)
    python scripts/debug/check_ckpt_load.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch as th

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CKPT_DIR = _REPO_ROOT / "saved_ckpts"
_FRAGNNET_TYPES = {"frag_gnn", "fraggnet"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_raw(path: Path) -> dict:
    from fragnnet.utils.nn_utils import decompile_jit_ckpt

    ckpt = th.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("hyper_parameters", {}).get("compile", False):
        log.info("  decompiling compiled checkpoint …")
        ckpt = decompile_jit_ckpt(ckpt)
    return ckpt


def _check_one(path: Path, dry_run: bool) -> bool:
    """Return True if the checkpoint passes all checks."""
    ok = True
    log.info(f"\n{'='*60}")
    log.info(f"  {path.name}")
    log.info(f"  {path}")
    log.info(f"{'='*60}")

    # --- load raw ckpt ---
    try:
        ckpt = _load_raw(path)
    except Exception as e:
        log.error(f"  FAIL  could not load file: {e}")
        return False

    # --- basic structure ---
    for key in ("state_dict", "hyper_parameters"):
        if key not in ckpt:
            log.error(f"  FAIL  missing key in checkpoint: {key}")
            ok = False

    if not ok:
        return False

    hparams = ckpt["hyper_parameters"]
    model_type = hparams.get("model_type", "<unknown>")
    log.info(f"  model_type : {model_type}")
    log.info(f"  epoch      : {ckpt.get('epoch', '?')}")
    log.info(f"  step       : {ckpt.get('global_step', '?')}")
    log.info(f"  state_dict : {len(ckpt['state_dict'])} keys")

    if dry_run:
        log.info("  [dry-run] skipping model build and state-dict load")
        return True

    if model_type not in _FRAGNNET_TYPES:
        log.warning(f"  SKIP  not a FraGNNet checkpoint (model_type={model_type})")
        return True

    # --- build model ---
    from fragnnet.utils.pl_utils import get_pl_hparams
    from fragnnet.utils.setup_utils import get_model_cls

    try:
        config_d = get_pl_hparams(ckpt)
        config_d["compile"] = False
        model_cls = get_model_cls(model_type)
        model = model_cls(**config_d)
    except Exception as e:
        log.error(f"  FAIL  could not build model from hparams: {e}")
        return False

    log.info(f"  model built: {model_cls.__name__}")

    # --- load state dict ---
    missing: list[str] = []
    unexpected: list[str] = []
    try:
        result = model.load_state_dict(ckpt["state_dict"], strict=True)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        log.info("  state_dict : loaded with strict=True")
    except RuntimeError as e:
        log.warning(f"  strict=True failed, retrying with strict=False: {e}")
        result = model.load_state_dict(ckpt["state_dict"], strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        log.info("  state_dict : loaded with strict=False")

    if missing:
        log.error(f"  FAIL  missing keys ({len(missing)}):")
        for k in missing:
            log.error(f"    - {k}")
        ok = False

    if unexpected:
        log.error(f"  FAIL  unexpected keys ({len(unexpected)}):")
        for k in unexpected:
            log.error(f"    - {k}")
        ok = False

    # --- finite-parameter check ---
    bad_params = [
        name for name, p in model.named_parameters() if not th.isfinite(p).all()
    ]
    if bad_params:
        log.error(f"  FAIL  non-finite parameters ({len(bad_params)}):")
        for name in bad_params:
            log.error(f"    - {name}")
        ok = False

    if ok:
        log.info("  PASS")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _collect(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []

    # explicit positional paths
    for p in args.paths:
        p = Path(p)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.ckpt")))
        else:
            paths.append(p)

    # --dir flag
    if args.dir:
        d = Path(args.dir)
        paths.extend(sorted(d.rglob("*.ckpt")))

    # default: saved_ckpts/
    if not paths:
        if _DEFAULT_CKPT_DIR.exists():
            paths = sorted(_DEFAULT_CKPT_DIR.rglob("*.ckpt"))
        else:
            log.error(f"No paths given and default directory not found: {_DEFAULT_CKPT_DIR}")
            sys.exit(1)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="*", help="checkpoint files or directories")
    parser.add_argument("--dir", metavar="DIR", help="recursively scan a directory")
    parser.add_argument("--dry-run", action="store_true", help="only check file structure")
    args = parser.parse_args()

    ckpt_paths = _collect(args)
    if not ckpt_paths:
        log.error("no checkpoint files found")
        sys.exit(1)

    log.info(f"Checking {len(ckpt_paths)} checkpoint(s) …")

    results: dict[Path, bool] = {}
    for p in ckpt_paths:
        results[p] = _check_one(p, dry_run=args.dry_run)

    # summary
    passed = [p for p, ok in results.items() if ok]
    failed = [p for p, ok in results.items() if not ok]

    log.info(f"\n{'='*60}")
    log.info(f"SUMMARY  {len(passed)} passed  /  {len(failed)} failed  /  {len(results)} total")
    if failed:
        log.info("Failed:")
        for p in failed:
            log.info(f"  FAIL  {p}")
        sys.exit(1)


if __name__ == "__main__":
    main()
