#!/usr/bin/env python3
"""Run one benchmark episode in a fresh process to release MuJoCo memory."""
from __future__ import annotations

import argparse
from pathlib import Path

from runtime import run_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--generation-seed", type=int, required=True)
    parser.add_argument("--physics-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_episode(args.suite, args.spec, args.gold, args.generation_seed,
                args.physics_seed, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
