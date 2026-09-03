"""Switch action.keras back to a saved model version."""

from __future__ import annotations

import argparse

from sign_ai.training.model_manager import activate_model, load_registry


def parse_args():
    parser = argparse.ArgumentParser(description="Switch the active compatibility model.")
    parser.add_argument("version", nargs="?", help="Example: model_v003")
    parser.add_argument("--list", action="store_true", help="List known model versions.")
    return parser.parse_args()


def main():
    args = parse_args()
    registry = load_registry()
    if args.list or not args.version:
        print("Model registry:")
        if not registry["models"]:
            print("  No versioned models found.")
            return
        for version_id, info in sorted(registry["models"].items()):
            active = " ACTIVE" if registry.get("active_model") == version_id else ""
            best = " BEST" if registry.get("best_model") == version_id else ""
            print(f"  - {version_id}{active}{best}: {info.get('model_path')}")
        return

    activate_model(args.version)
    print(f"Activated {args.version}.")


if __name__ == "__main__":
    main()
