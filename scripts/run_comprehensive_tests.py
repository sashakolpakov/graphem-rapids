#!/usr/bin/env python3
"""Run the canonical CPU contract and, when requested, GPU tests."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Execute the selected test contract and return its process status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-gpu", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    arguments = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "pytest", "-q"]
    if arguments.parallel:
        command.extend(["-n", "auto"])
    if arguments.coverage:
        command.extend(
            [
                "--cov=graphem_rapids",
                "--cov-report=term-missing",
                "--cov-fail-under=80",
            ]
        )
    if not arguments.include_gpu:
        command.extend(["-m", "not gpu"])
    completed = subprocess.run(command, cwd=repository, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
