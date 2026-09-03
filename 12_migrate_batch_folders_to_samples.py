"""Flatten old MP_Data/sign/batch/sample folders into MP_Data/sign/sample."""

from __future__ import annotations

import argparse

from sign_language_dataset import flatten_legacy_batches
from sign_memory import rebuild_memory_from_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Move legacy batch samples into direct sample folders.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned moves without changing files.")
    return parser.parse_args()


def main():
    args = parse_args()
    moves = flatten_legacy_batches(dry_run=args.dry_run)

    if not moves:
        print("No legacy batch samples found.")
    else:
        for move in moves:
            print(f"{move.source} -> {move.destination}")
        action = "Would move" if args.dry_run else "Moved"
        print(f"{action} {len(moves)} sample folders.")

    if not args.dry_run:
        memory = rebuild_memory_from_dataset()
        print(f"Rebuilt memory.json for {len(memory.get('examples', []))} examples.")


if __name__ == "__main__":
    main()
