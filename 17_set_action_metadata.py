"""Set optional static/dynamic metadata for a sign label."""

from __future__ import annotations

import argparse

from sign_ai.actions_metadata import set_action_type


def parse_args():
    parser = argparse.ArgumentParser(description="Set action metadata.")
    parser.add_argument("label")
    parser.add_argument("type", choices=["static", "dynamic"])
    return parser.parse_args()


def main():
    args = parse_args()
    set_action_type(args.label, args.type)
    print(f"Saved metadata: {args.label} -> {args.type}")


if __name__ == "__main__":
    main()
