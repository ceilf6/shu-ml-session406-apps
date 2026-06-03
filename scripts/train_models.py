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
    output_path.write_text(json.dumps(model, separators=(",", ":")) + "\n", encoding="utf-8")


def train_relu_network(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_layers: tuple[int, ...] = (256, 128),
    epochs: int = 28,
    learning_rate: float = 0.001,
    batch_size: int = 256,
) -> tuple[dict, float]:
    rng = np.random.default_rng(406)
    labels = [str(index) for index in range(10)]
    layer_sizes = (x_train.shape[1], *hidden_layers, 10)
    weights = [
        (rng.standard_normal((layer_sizes[index], layer_sizes[index + 1])) * np.sqrt(2 / layer_sizes[index])).astype(
            np.float32
        )
        for index in range(len(layer_sizes) - 1)
    ]
    biases = [np.zeros(size, dtype=np.float32) for size in layer_sizes[1:]]
    weight_moments = [np.zeros_like(weight) for weight in weights]
    weight_velocities = [np.zeros_like(weight) for weight in weights]
    bias_moments = [np.zeros_like(bias) for bias in biases]
    bias_velocities = [np.zeros_like(bias) for bias in biases]
    targets = np.eye(10, dtype=np.float32)[y_train]
    step = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    for _ in range(epochs):
        indices = rng.permutation(len(x_train))
        for start in range(0, len(x_train), batch_size):
            step += 1
            batch_indices = indices[start : start + batch_size]
            xb = x_train[batch_indices]
            yb = targets[batch_indices]
            activations = [xb]
            pre_activations = []
            for layer_index, (weight, bias) in enumerate(zip(weights, biases)):
                z = activations[-1] @ weight + bias
                pre_activations.append(z)
                activations.append(np.maximum(z, 0) if layer_index < len(weights) - 1 else z)

            logits = activations[-1]
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)

            gradient = (probabilities - yb) / len(xb)
            gradient_weights = []
            gradient_biases = []
            for layer_index in range(len(weights) - 1, -1, -1):
                gradient_weights.insert(0, activations[layer_index].T @ gradient)
                gradient_biases.insert(0, gradient.sum(axis=0))
                if layer_index > 0:
                    gradient = (gradient @ weights[layer_index].T) * (pre_activations[layer_index - 1] > 0)

            for layer_index in range(len(weights)):
                weight_moments[layer_index] = beta1 * weight_moments[layer_index] + (1 - beta1) * gradient_weights[
                    layer_index
                ]
                weight_velocities[layer_index] = beta2 * weight_velocities[layer_index] + (
                    1 - beta2
                ) * np.square(gradient_weights[layer_index])
                bias_moments[layer_index] = beta1 * bias_moments[layer_index] + (1 - beta1) * gradient_biases[
                    layer_index
                ]
                bias_velocities[layer_index] = beta2 * bias_velocities[layer_index] + (1 - beta2) * np.square(
                    gradient_biases[layer_index]
                )

                corrected_weight_moment = weight_moments[layer_index] / (1 - beta1**step)
                corrected_weight_velocity = weight_velocities[layer_index] / (1 - beta2**step)
                corrected_bias_moment = bias_moments[layer_index] / (1 - beta1**step)
                corrected_bias_velocity = bias_velocities[layer_index] / (1 - beta2**step)
                weights[layer_index] -= learning_rate * corrected_weight_moment / (
                    np.sqrt(corrected_weight_velocity) + epsilon
                )
                biases[layer_index] -= learning_rate * corrected_bias_moment / (
                    np.sqrt(corrected_bias_velocity) + epsilon
                )

    test_activation = x_test
    for layer_index, (weight, bias) in enumerate(zip(weights, biases)):
        test_activation = test_activation @ weight + bias
        if layer_index < len(weights) - 1:
            test_activation = np.maximum(test_activation, 0)
    test_logits = test_activation
    accuracy = float((test_logits.argmax(axis=1) == y_test).mean())
    model = {
        "labels": labels,
        "scaler": {
            "mean": np.zeros(x_train.shape[1], dtype=float).tolist(),
            "scale": np.ones(x_train.shape[1], dtype=float).tolist(),
        },
        "layers": [
            {
                "weights": weight.T.round(5).tolist(),
                "biases": bias.round(5).tolist(),
                "activation": "relu" if index < len(weights) - 1 else "linear",
            }
            for index, (weight, bias) in enumerate(zip(weights, biases))
        ],
    }
    return model, accuracy


def load_digit_data(
    train_csv: Path,
    test_csv: Path,
    full_npz: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if full_npz is not None:
        full = np.load(full_npz)
        x_train = full["x_train"].reshape(-1, 28 * 28).astype(np.float32) / 255.0
        y_train = full["y_train"].astype(int)
        x_test = full["x_test"].reshape(-1, 28 * 28).astype(np.float32) / 255.0
        y_test = full["y_test"].astype(int)
        return x_train, y_train, x_test, y_test, "Keras full MNIST dataset (60k train / 10k test)"

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    x_train = train.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0
    y_train = train.iloc[:, 0].to_numpy(dtype=int)
    x_test = test.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0
    y_test = test.iloc[:, 0].to_numpy(dtype=int)
    return x_train, y_train, x_test, y_test, "course MNIST CSV subset"


def train_digit_model(train_csv: Path, test_csv: Path, output_dir: Path, full_npz: Path | None = None) -> None:
    x_train, y_train, x_test, y_test, source = load_digit_data(train_csv, test_csv, full_npz)

    model, accuracy = train_relu_network(x_train, y_train, x_test, y_test)
    export_model(
        model,
        output_dir / "digit-model.json",
        {
            "name": "MNIST two-hidden-layer ReLU classifier",
            "source": source,
            "testAccuracy": round(accuracy, 4),
            "trainSamples": int(len(x_train)),
            "testSamples": int(len(x_test)),
            "featureCount": 784,
            "inputShape": [28, 28],
            "hiddenLayers": [256, 128],
            "preprocessing": "Canvas strokes are cropped, scaled to a 20x20 box, and centered in a 28x28 MNIST frame.",
        },
    )
    print(f"digit accuracy={accuracy:.4f} train={len(x_train)} test={len(x_test)} source={source}")


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
    country_profiles = []
    for country_code in classes:
        rows = ufo[ufo["country"] == country_code]
        country_profiles.append(
            {
                "label": COUNTRY_NAMES[country_code],
                "code": country_code.upper(),
                "count": int(len(rows)),
                "share": round(float(len(rows) / len(ufo)), 4),
                "centroid": {
                    "seconds": round(float(rows["seconds"].median()), 2),
                    "latitude": round(float(rows["latitude"].mean()), 4),
                    "longitude": round(float(rows["longitude"].mean()), 4),
                },
                "range": {
                    "latitude": [
                        round(float(rows["latitude"].quantile(0.1)), 4),
                        round(float(rows["latitude"].quantile(0.9)), 4),
                    ],
                    "longitude": [
                        round(float(rows["longitude"].quantile(0.1)), 4),
                        round(float(rows["longitude"].quantile(0.9)), 4),
                    ],
                },
            }
        )
    export_model(
        model,
        output_dir / "ufo-model.json",
        {
            "name": "UFO sighting country nearest-centroid classifier",
            "source": "course UFO sightings CSV",
            "testAccuracy": round(accuracy, 4),
            "featureNames": ["Duration seconds", "Latitude", "Longitude"],
            "sampleCount": int(len(ufo)),
            "countryProfiles": country_profiles,
        },
    )
    print(f"ufo accuracy={accuracy:.4f} samples={len(ufo)} source=course UFO sightings CSV")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-train-csv", required=True, type=Path)
    parser.add_argument("--mnist-test-csv", required=True, type=Path)
    parser.add_argument("--mnist-full-npz", type=Path)
    parser.add_argument("--ufo-csv", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_digit_model(args.mnist_train_csv, args.mnist_test_csv, args.output_dir, args.mnist_full_npz)
    train_ufo_model(args.ufo_csv, args.output_dir)


if __name__ == "__main__":
    main()
