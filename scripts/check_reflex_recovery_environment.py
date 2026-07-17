#!/usr/bin/env python3
"""Report reflex-recovery runtime prerequisites without exposing credentials."""

from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unavailable"


def main() -> int:
    """Print runtime status and return nonzero when core imports fail."""
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    print(f"Git commit: {_git_commit()}")
    try:
        import jax
        import mujoco

        print(f"JAX: {jax.__version__}")
        print(f"JAX backend: {jax.default_backend()}")
        print(f"JAX devices: {jax.devices()}")
        print(f"MuJoCo: {mujoco.__version__}")
        try:
            from mujoco import mjx

            print(f"MuJoCo MJX: available ({mjx.__name__})")
        except Exception as error:
            print(f"MuJoCo MJX: unavailable ({error})")
    except Exception as error:
        print(f"Core import failure: {error}")
        return 1
    memory = shutil.disk_usage(Path.cwd())
    print(f"Workspace disk free GiB: {memory.free / 2**30:.1f}")
    cache_root = Path(os.environ.get("MUSCLEMIMIC_HOME", Path.home() / ".musclemimic"))
    print(f"MuscleMimic cache root: {cache_root}")
    print(f"Hugging Face authentication appears available: {bool(os.environ.get('HF_TOKEN') or (Path.home() / '.cache/huggingface/token').exists())}")
    print(f"W&B authentication appears available: {bool(os.environ.get('WANDB_API_KEY') or (Path.home() / '.netrc').exists())}")
    print(f"Warp Python package: {importlib.util.find_spec('warp') is not None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
