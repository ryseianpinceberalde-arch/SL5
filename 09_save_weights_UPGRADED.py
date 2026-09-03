"""Save a compatibility copy of the trained model."""

from sign_language_common import LEGACY_MODEL_PATH, MODEL_PATH, load_model_safely


def main():
    model = load_model_safely()
    model.save(MODEL_PATH)
    model.save(LEGACY_MODEL_PATH)
    print(f"Saved preferred model: {MODEL_PATH.name}")
    print(f"Saved legacy compatibility model: {LEGACY_MODEL_PATH.name}")


if __name__ == "__main__":
    main()
