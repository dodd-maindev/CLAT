"""Verification script to test a trained checkpoint on a specific fold split."""

import argparse
import json
import subprocess
import sys


class CheckpointVerifier:
    """Class responsible for validating checkpoints on specific test folds."""

    def __init__(self, checkpoint_path: str, fold_idx: int) -> None:
        self.checkpoint_path = checkpoint_path
        self.fold_idx = fold_idx

    def verify(self) -> int:
        """Executes Lightning test subcommand for the given fold and checkpoint."""
        print(f"\n{'='*70}")
        print(f"VERIFYING CHECKPOINT: {self.checkpoint_path}")
        print(f"TEST FOLD INDEX:     {self.fold_idx}")
        print(f"{'='*70}\n")

        cmd = [
            sys.executable,
            "src/main.py",
            "test",
            "--config",
            "configs/milvt_olap.yaml",
            "--data",
            "configs/data/DDR.yaml",
            f"--data.fold_num={self.fold_idx}",
            f"--ckpt_path={self.checkpoint_path}",
        ]
        return subprocess.run(cmd).returncode


def main() -> None:
    """Entry point for checkpoint verification."""
    parser = argparse.ArgumentParser(
        description="Verify model checkpoint on a fold test split."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index to test against (0-9).",
    )
    args = parser.parse_args()
    verifier = CheckpointVerifier(args.ckpt, args.fold)
    sys.exit(verifier.verify())


if __name__ == "__main__":
    main()
