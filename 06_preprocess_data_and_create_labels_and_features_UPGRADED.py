"""Build X/y arrays from MP_Data and skip broken sample folders."""

import argparse

from sign_ai.config import FEATURE_MODE, LANDMARK_ONLY, LANDMARK_PLUS_VELOCITY, SEQUENCE_LENGTH
from sign_language_common import PROCESSED_DATA_FILE
from sign_language_dataset import build_and_save_processed_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Build processed_data.npz from MP_Data.")
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--feature-mode", choices=[LANDMARK_ONLY, LANDMARK_PLUS_VELOCITY], default=FEATURE_MODE)
    parser.add_argument("--include-legacy-batches", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_and_save_processed_dataset(
        include_legacy_batches=args.include_legacy_batches,
        target_length=args.sequence_length,
        normalize=True,
        feature_mode=args.feature_mode,
    )
    print(f"Saved: {PROCESSED_DATA_FILE.name}")
    print(f"X shape: {result.X.shape}")
    print(f"y shape: {result.y.shape}")
    print(f"Feature mode: {result.feature_mode}")
    print("Complete samples used for training:")
    for action in result.actions:
        print(f"  - {action}: {result.counts[action]}")
    if result.skipped:
        print("Skipped incomplete/broken samples:")
        for item in result.skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
