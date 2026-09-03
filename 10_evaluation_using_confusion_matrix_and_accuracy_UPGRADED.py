"""Evaluate the trained model with confusion matrices and accuracy."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix

from sign_ai.config import TRAINING_RESULTS_DIR
from sign_ai.training.evaluation import save_evaluation_artifacts
from sign_language_common import TEST_DATA_FILE, load_model_actions, load_model_safely


def main():
    if not TEST_DATA_FILE.exists():
        raise FileNotFoundError("Run 07_build_and_train_lstm_neural_network_UPGRADED.py first.")

    data = np.load(TEST_DATA_FILE, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    actions = [str(action) for action in data["actions"]] if "actions" in data else load_model_actions()
    model = load_model_safely()
    if model.output_shape[-1] != len(actions):
        raise RuntimeError("Model output count does not match the saved test split labels.")

    yhat = model.predict(X_test)
    ytrue_labels = np.argmax(y_test, axis=1)
    yhat_labels = np.argmax(yhat, axis=1)

    print("Actions:", actions)
    print("Multilabel confusion matrix:")
    print(multilabel_confusion_matrix(ytrue_labels, yhat_labels))
    print("Accuracy:", accuracy_score(ytrue_labels, yhat_labels))
    print("Classification report:")
    print(classification_report(ytrue_labels, yhat_labels, labels=list(range(len(actions))), target_names=actions, zero_division=0))

    output_dir = TRAINING_RESULTS_DIR / "manual_evaluation"
    save_evaluation_artifacts(
        model,
        actions,
        output_dir,
        X_test,
        y_test,
        X_test,
        y_test,
        X_test,
        y_test,
    )
    print(f"Saved evaluation artifacts: {output_dir}")


if __name__ == "__main__":
    main()
