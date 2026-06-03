from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COUNTRY_NAMES = {
    "au": "Australia",
    "ca": "Canada",
    "de": "Germany",
    "gb": "United Kingdom",
    "us": "United States",
}


def nearest_centroid_as_linear_model(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> dict:
    if mean is None:
        mean = np.zeros(features.shape[1], dtype=float)
    if scale is None:
        scale = np.ones(features.shape[1], dtype=float)
    scale = np.where(scale == 0, 1, scale)
    standardized = (features - mean) / scale

    coefficients = []
    intercepts = []
    total_count = len(labels)
    for class_index in range(len(label_names)):
        class_features = standardized[labels == class_index]
        centroid = class_features.mean(axis=0)
        prior = np.log(len(class_features) / total_count)
        coefficients.append(2 * centroid)
        intercepts.append(-float(np.dot(centroid, centroid)) + float(prior))

    return {
        "labels": label_names,
        "scaler": {
            "mean": mean.round(8).tolist(),
            "scale": scale.round(8).tolist(),
        },
        "coefficients": np.array(coefficients).round(8).tolist(),
        "intercepts": np.array(intercepts).round(8).tolist(),
    }


def predict(model: dict, features: np.ndarray) -> np.ndarray:
    mean = np.array(model["scaler"]["mean"], dtype=float)
    scale = np.array(model["scaler"]["scale"], dtype=float)
    coefficients = np.array(model["coefficients"], dtype=float)
    intercepts = np.array(model["intercepts"], dtype=float)
    standardized = (features - mean) / scale
    logits = standardized @ coefficients.T + intercepts
    return logits.argmax(axis=1)


def export_model(model: dict, output_path: Path, metadata: dict) -> None:
    model["metadata"] = metadata
    output_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")


def train_relu_network(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_units: int = 96,
    epochs: int = 16,
    learning_rate: float = 0.18,
    batch_size: int = 256,
) -> tuple[dict, float]:
    rng = np.random.default_rng(406)
    labels = [str(index) for index in range(10)]
    w1 = (rng.standard_normal((x_train.shape[1], hidden_units)) * np.sqrt(2 / x_train.shape[1])).astype(
        np.float32
    )
    b1 = np.zeros(hidden_units, dtype=np.float32)
    w2 = (rng.standard_normal((hidden_units, 10)) * np.sqrt(2 / hidden_units)).astype(np.float32)
    b2 = np.zeros(10, dtype=np.float32)
    targets = np.eye(10, dtype=np.float32)[y_train]

    for _ in range(epochs):
        indices = rng.permutation(len(x_train))
        for start in range(0, len(x_train), batch_size):
            batch_indices = indices[start : start + batch_size]
            xb = x_train[batch_indices]
            yb = targets[batch_indices]
            z1 = xb @ w1 + b1
            a1 = np.maximum(z1, 0)
            logits = a1 @ w2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)

            dlogits = (probabilities - yb) / len(xb)
            dw2 = a1.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dz1 = (dlogits @ w2.T) * (z1 > 0)
            dw1 = xb.T @ dz1
            db1 = dz1.sum(axis=0)

            w1 -= learning_rate * dw1
            b1 -= learning_rate * db1
            w2 -= learning_rate * dw2
            b2 -= learning_rate * db2

    test_logits = np.maximum(x_test @ w1 + b1, 0) @ w2 + b2
    accuracy = float((test_logits.argmax(axis=1) == y_test).mean())
    model = {
        "labels": labels,
        "scaler": {
            "mean": np.zeros(x_train.shape[1], dtype=float).tolist(),
            "scale": np.ones(x_train.shape[1], dtype=float).tolist(),
        },
        "layers": [
            {
                "weights": w1.T.round(6).tolist(),
                "biases": b1.round(6).tolist(),
                "activation": "relu",
            },
            {
                "weights": w2.T.round(6).tolist(),
                "biases": b2.round(6).tolist(),
                "activation": "linear",
            },
        ],
    }
    return model, accuracy


def train_digit_model(train_csv: Path, test_csv: Path, output_dir: Path) -> None:
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    x_train = train.iloc[:, 1:].to_numpy(dtype=float) / 255.0
    y_train = train.iloc[:, 0].to_numpy(dtype=int)
    x_test = test.iloc[:, 1:].to_numpy(dtype=float) / 255.0
    y_test = test.iloc[:, 0].to_numpy(dtype=int)

    model, accuracy = train_relu_network(x_train, y_train, x_test, y_test)
    export_model(
        model,
        output_dir / "digit-model.json",
        {
            "name": "MNIST one-hidden-layer ReLU classifier",
            "source": str(train_csv),
            "testAccuracy": round(accuracy, 4),
            "featureCount": 784,
            "inputShape": [28, 28],
            "hiddenUnits": 96,
        },
    )


def train_ufo_model(csv_path: Path, output_dir: Path) -> None:
    data = pd.read_csv(csv_path)
    ufo = pd.DataFrame(
        {
            "seconds": pd.to_numeric(data["duration (seconds)"], errors="coerce"),
            "latitude": pd.to_numeric(data["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(data["longitude"], errors="coerce"),
            "country": data["country"].astype(str).str.lower().str.strip(),
        }
    )
    ufo = ufo.dropna()
    ufo = ufo[(ufo["seconds"] >= 1) & (ufo["seconds"] <= 60)]
    ufo = ufo[ufo["country"].isin(COUNTRY_NAMES)]

    classes = sorted(ufo["country"].unique())
    class_to_index = {label: index for index, label in enumerate(classes)}
    label_names = [COUNTRY_NAMES[label] for label in classes]
    features = ufo[["seconds", "latitude", "longitude"]].to_numpy(dtype=float)
    labels = ufo["country"].map(class_to_index).to_numpy(dtype=int)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)

    rng = np.random.default_rng(406)
    indices = np.arange(len(features))
    rng.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_indices = indices[:split]
    test_indices = indices[split:]

    model = nearest_centroid_as_linear_model(
        features[train_indices],
        labels[train_indices],
        label_names,
        mean=mean,
        scale=scale,
    )
    accuracy = float((predict(model, features[test_indices]) == labels[test_indices]).mean())
    export_model(
        model,
        output_dir / "ufo-model.json",
        {
            "name": "UFO sighting country nearest-centroid classifier",
            "source": str(csv_path),
            "testAccuracy": round(accuracy, 4),
            "featureNames": ["Duration seconds", "Latitude", "Longitude"],
            "sampleCount": int(len(ufo)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-train-csv", required=True, type=Path)
    parser.add_argument("--mnist-test-csv", required=True, type=Path)
    parser.add_argument("--ufo-csv", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_digit_model(args.mnist_train_csv, args.mnist_test_csv, args.output_dir)
    train_ufo_model(args.ufo_csv, args.output_dir)


if __name__ == "__main__":
    main()
