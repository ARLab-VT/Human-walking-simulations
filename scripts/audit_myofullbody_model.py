#!/usr/bin/env python3
"""Audit a compiled MyoFullBody-compatible MJCF model."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

from musclemimic.research.reflex_recovery.model_audit import audit_model


def main() -> int:
    """Load the exact XML selected by the caller and export audit artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_xml", type=Path, help="Exact compiled-model source XML")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reflex_recovery/model_audit"))
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model_xml.resolve()))
    summary = audit_model(model, args.output_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
