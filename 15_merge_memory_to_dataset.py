"""Copy immediate-memory examples into MP_Data for later neural training."""

from __future__ import annotations

import argparse

from sign_ai.memory.memory_manager import merge_memory_to_dataset
from sign_language_common import update_actions_from_folders


def parse_args():
    parser = argparse.ArgumentParser(description="Merge memory examples into MP_Data.")
    parser.add_argument("--labels", nargs="*", help="Only merge these labels. Defaults to all labels.")
    parser.add_argument("--exclude-corrections", action="store_true")
    parser.add_argument("--force", action="store_true", help="Copy examples even if they were already merged before.")
    return parser.parse_args()


def main():
    args = parse_args()
    written = merge_memory_to_dataset(
        labels=args.labels,
        include_corrections=not args.exclude_corrections,
        force=args.force,
    )
    update_actions_from_folders()
    if not written:
        print("No memory examples were merged.")
        return
    print("Merged memory examples into MP_Data:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
