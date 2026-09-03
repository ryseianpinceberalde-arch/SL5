"""Make predictions on the saved test split."""

import numpy as np

from sign_language_common import TEST_DATA_FILE, load_model_actions, load_model_safely


TOP_N = 3


def top_predictions(prediction, actions, top_n=TOP_N):
    top_indices = np.argsort(prediction)[::-1][: min(top_n, len(actions))]
    return [
        (actions[int(index)], float(prediction[int(index)]))
        for index in top_indices
    ]


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

    predictions = model.predict(X_test)
    for index in range(min(5, len(X_test))):
        actual = actions[int(np.argmax(y_test[index]))]
        top_3 = top_predictions(predictions[index], actions)
        top_text = ", ".join(
            f"{label}={confidence * 100:.2f}%"
            for label, confidence in top_3
        )
        print(f"Sample {index}: actual={actual} top_3=[{top_text}]")


if __name__ == "__main__":
    main()
