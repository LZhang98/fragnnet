"""Compiler discovery, nvcc builds and library loading for MASSter's native CUDA kernels.

Status: PRODUCTION, used by `cuda_mask_ranking` and `cuda_topology` for the CUDA backend.
The cache filename hashes the `.cu` source, so editing a kernel, even a comment in it,
forces one recompile per cache directory.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from threading import Lock

import torch as th

logger = logging.getLogger(__name__)


def default_cuda_cache_dir(stage: str) -> str:
    """Return a stable per-user /tmp cache without requiring a CUDA device.

    Hash the home directory so different cluster users do not share a directory
    created with another user's ownership, while paths stay free of usernames.
    """
    user_identity = hashlib.sha256(str(Path.home()).encode()).hexdigest()[:16]
    return f"/tmp/fragnnet_cuda_{stage}_{user_identity}"


@lru_cache(maxsize=16)
def _compiler_release(home: str) -> tuple[int, int]:
    compiler = Path(home) / "bin" / "nvcc"
    if not compiler.is_file():
        raise RuntimeError(f"CUDA compiler not found at {compiler}")
    result = subprocess.run(
        [str(compiler), "--version"], check=True, capture_output=True, text=True, timeout=30
    )
    version = re.search(r"release\s+(\d+)\.(\d+)", result.stdout)
    if version is None:
        raise RuntimeError(f"Cannot identify CUDA compiler version at {compiler}")
    return int(version[1]), int(version[2])


def _validated_home(home: str, required_major: int | None) -> str:
    resolved = str(Path(home).expanduser().resolve())
    major, minor = _compiler_release(resolved)
    if required_major is not None and major != required_major:
        raise RuntimeError(
            f"CUDA compiler {major}.{minor} at {resolved} does not match PyTorch's CUDA "
            f"major version {required_major}; choose a matching cuda_home or CUDA_HOME"
        )
    return resolved


@lru_cache(maxsize=16)
def _discover_home(
    explicit: str | None, environment: str | None, path_compiler: str | None, torch_cuda: str | None
) -> str:
    required_major = None if torch_cuda is None else int(torch_cuda.split(".")[0])
    # Explicit choices remain visible: an invalid argument/environment value must
    # fail instead of silently selecting a different toolkit.
    if explicit is not None:
        return _validated_home(explicit, required_major)
    if environment:
        return _validated_home(environment, required_major)
    failures = []
    if path_compiler is not None:
        try:
            return _validated_home(str(Path(path_compiler).resolve().parent.parent), required_major)
        except RuntimeError as error:
            failures.append(str(error))
    if torch_cuda is not None:
        # A workstation can expose an older nvcc first on PATH while keeping the
        # matching toolkit installed. Prefer that exact installed release next.
        for home in (
            f"/usr/local/cuda-{torch_cuda}",
            f"/opt/cuda-{torch_cuda}",
            "/usr/local/cuda",
        ):
            try:
                resolved = _validated_home(home, required_major)
            except RuntimeError as error:
                failures.append(str(error))
                continue
            logger.info("Using installed CUDA toolkit %s instead of PATH nvcc", resolved)
            return resolved
    details = "; ".join(failures)
    raise RuntimeError(
        "No compatible installed CUDA toolkit found; provide cuda_home or CUDA_HOME"
        + (f". {details}" if details else "")
    )


def resolve_cuda_home(explicit: str | None = None) -> str:
    """Resolve an existing toolkit without installing or changing the environment.

    Priority is explicit argument, CUDA_HOME, then a compatible nvcc on PATH.
    If PATH is absent or exposes a different CUDA major version, try conventional
    installed locations matching PyTorch. Explicit incompatible choices fail.
    Compiler probing is cached; discovery remains sensitive to environment changes.
    """
    return _discover_home(
        explicit, os.environ.get("CUDA_HOME"), shutil.which("nvcc"), th.version.cuda
    )


def build_cuda_library(
    source: Path,
    stage: str,
    lock: Lock,
    cache_dir: str,
    cuda_home: str,
    flags: Sequence[str],
) -> ctypes.CDLL:
    """Compile `source` lazily into the /tmp cache and load it; callers declare the ABI.

    The cache filename hashes the source, compiler and the repr of `flags` (the full nvcc
    flag sequence), so identical inputs reuse one binary. `lock` is the calling stage's own
    build lock.
    """
    compiler = Path(cuda_home) / "bin" / "nvcc"
    if not compiler.is_file():
        raise RuntimeError(f"CUDA {stage} requires an existing nvcc compiler: {compiler}")
    cache = Path(cache_dir).resolve()
    if not cache.is_relative_to(Path("/tmp").resolve()):
        raise ValueError(f"CUDA {stage} build cache must be inside /tmp")
    version = subprocess.run(
        [str(compiler), "--version"], check=True, capture_output=True, timeout=30
    ).stdout
    identity = hashlib.sha256(
        source.read_bytes() + str(compiler.resolve()).encode() + version + repr(flags).encode()
    ).hexdigest()[:20]
    binary = cache / f"masster_cuda_{stage}_{identity}.so"
    with lock:
        cache.mkdir(parents=True, exist_ok=True)
        if not binary.is_file():
            temporary = binary.with_suffix(f".{os.getpid()}.tmp.so")
            command = [str(compiler), *flags, str(source), "-o", str(temporary)]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
                temporary.replace(binary)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"CUDA {stage} compilation failed:\n{error.stderr}") from error
            finally:
                temporary.unlink(missing_ok=True)
    return ctypes.CDLL(str(binary))
