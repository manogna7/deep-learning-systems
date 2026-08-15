"""A fully connected binary image classifier implemented with NumPy."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "cifar-2class.npz"


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Compute sigmoid probabilities without overflowing for large logits."""
    probabilities = np.empty_like(logits, dtype=np.result_type(logits, np.float64))
    positive = logits >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    return probabilities


class SigmoidCrossEntropy:
    """Mean binary cross-entropy operating directly on logits."""

    def __init__(self) -> None:
        self.labels: np.ndarray | None = None
        self.probabilities: np.ndarray | None = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        if logits.shape != labels.shape:
            raise ValueError(
                f"logits and labels must have the same shape; got {logits.shape} and {labels.shape}"
            )
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError("labels must contain only 0 and 1")

        self.labels = labels
        self.probabilities = sigmoid(logits)
        losses = np.logaddexp(0.0, logits) - labels * logits
        return float(np.mean(losses))

    def backward(self) -> np.ndarray:
        if self.labels is None or self.probabilities is None:
            raise RuntimeError("forward must be called before backward")
        return (self.probabilities - self.labels) / self.labels.size


class ReLU:
    """Rectified linear activation."""

    def __init__(self) -> None:
        self.input: np.ndarray | None = None

    def forward(self, input_values: np.ndarray) -> np.ndarray:
        self.input = input_values
        return np.maximum(0, input_values)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise RuntimeError("forward must be called before backward")
        return gradient * (self.input > 0)

    def step(
        self, step_size: float, momentum: float = 0.0, weight_decay: float = 0.0
    ) -> None:
        del step_size, momentum, weight_decay


class LinearLayer:
    """Affine transformation with momentum buffers for optimization."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("layer dimensions must be positive")

        generator = rng if rng is not None else np.random.default_rng()
        self.weights = generator.standard_normal((input_dim, output_dim)) * np.sqrt(
            2.0 / input_dim
        )
        self.bias = np.zeros((1, output_dim))
        self.input: np.ndarray | None = None
        self.grad_weights: np.ndarray | None = None
        self.grad_bias: np.ndarray | None = None
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_bias = np.zeros_like(self.bias)

    def forward(self, input_values: np.ndarray) -> np.ndarray:
        if input_values.ndim != 2 or input_values.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"expected input shape (batch, {self.weights.shape[0]}); got {input_values.shape}"
            )
        self.input = input_values
        return input_values @ self.weights + self.bias

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise RuntimeError("forward must be called before backward")
        if gradient.shape != (self.input.shape[0], self.weights.shape[1]):
            raise ValueError("gradient shape does not match the layer output")

        self.grad_weights = self.input.T @ gradient
        self.grad_bias = np.sum(gradient, axis=0, keepdims=True)
        return gradient @ self.weights.T

    def step(
        self,
        step_size: float,
        momentum: float = 0.8,
        weight_decay: float = 0.0,
    ) -> None:
        if self.grad_weights is None or self.grad_bias is None:
            raise RuntimeError("backward must be called before step")

        self.velocity_weights = momentum * self.velocity_weights - step_size * (
            self.grad_weights + weight_decay * self.weights
        )
        self.velocity_bias = momentum * self.velocity_bias - step_size * self.grad_bias
        self.weights += self.velocity_weights
        self.bias += self.velocity_bias


class FeedForwardNeuralNetwork:
    """A configurable stack of fully connected layers and ReLU activations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        generator = rng if rng is not None else np.random.default_rng()
        if num_layers == 1:
            self.layers: list[LinearLayer | ReLU] = [
                LinearLayer(input_dim, output_dim, generator)
            ]
        else:
            self.layers = [LinearLayer(input_dim, hidden_dim, generator), ReLU()]
            for _ in range(num_layers - 2):
                self.layers.extend(
                    [LinearLayer(hidden_dim, hidden_dim, generator), ReLU()]
                )
            self.layers.append(LinearLayer(hidden_dim, output_dim, generator))

    def forward(self, input_values: np.ndarray) -> np.ndarray:
        output = input_values
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, gradient: np.ndarray) -> None:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def step(self, step_size: float, momentum: float, weight_decay: float) -> None:
        for layer in self.layers:
            layer.step(step_size, momentum, weight_decay)


def binary_predictions(logits: np.ndarray) -> np.ndarray:
    """Convert logits to labels using the sigmoid decision boundary."""
    return (logits >= 0.0).astype(np.uint8)


def batches(num_examples: int, batch_size: int) -> Iterator[slice]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, num_examples, batch_size):
        yield slice(start, min(start + batch_size, num_examples))


def evaluate(
    model: FeedForwardNeuralNetwork,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> tuple[float, float]:
    """Return example-weighted loss and accuracy for a dataset."""
    if len(features) == 0 or len(features) != len(labels):
        raise ValueError("features and labels must be non-empty and have equal lengths")

    total_loss = 0.0
    total_correct = 0
    loss_fn = SigmoidCrossEntropy()

    for batch_slice in batches(len(features), batch_size):
        batch_features = features[batch_slice]
        batch_labels = labels[batch_slice]
        logits = model.forward(batch_features)
        total_loss += loss_fn.forward(logits, batch_labels) * len(batch_features)
        total_correct += int(np.sum(binary_predictions(logits) == batch_labels))

    return total_loss / len(features), total_correct / len(features)


def load_dataset(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the non-executable NumPy dataset archive."""
    required_keys = ("train_data", "train_labels", "test_data", "test_labels")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required_keys) - set(archive.files)
        if missing:
            raise ValueError(f"dataset is missing arrays: {', '.join(sorted(missing))}")
        train_data, train_labels, test_data, test_labels = (
            np.array(archive[key]) for key in required_keys
        )

    for split_name, features, labels in (
        ("training", train_data, train_labels),
        ("test", test_data, test_labels),
    ):
        if features.ndim != 2 or labels.shape != (len(features), 1):
            raise ValueError(f"invalid {split_name} split shapes")
        if not np.all(np.isfinite(features)):
            raise ValueError(f"{split_name} features contain non-finite values")
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError(f"{split_name} labels must contain only 0 and 1")

    if train_data.shape[1] != test_data.shape[1]:
        raise ValueError("training and test feature dimensions must match")

    return (
        train_data.astype(np.float32),
        train_labels.astype(np.uint8),
        test_data.astype(np.float32),
        test_labels.astype(np.uint8),
    )


def normalize_features(
    train_data: np.ndarray, test_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize both splits using training-only statistics."""
    mean = np.mean(train_data, axis=0)
    standard_deviation = np.std(train_data, axis=0)
    safe_standard_deviation = np.where(standard_deviation == 0, 1, standard_deviation)
    return (
        (train_data - mean) / safe_standard_deviation,
        (test_data - mean) / safe_standard_deviation,
    )


def train(
    model: FeedForwardNeuralNetwork,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    step_size: float,
    momentum: float,
    weight_decay: float,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """Optimize the network and return batch and epoch metrics."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    history: dict[str, list[float]] = {
        "train_batch_loss": [],
        "train_batch_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    loss_fn = SigmoidCrossEntropy()

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(train_data))
        epoch_loss = 0.0
        epoch_correct = 0

        for batch_slice in batches(len(train_data), batch_size):
            batch_indices = indices[batch_slice]
            batch_features = train_data[batch_indices]
            batch_labels = train_labels[batch_indices]

            logits = model.forward(batch_features)
            loss = loss_fn.forward(logits, batch_labels)
            model.backward(loss_fn.backward())
            model.step(step_size, momentum, weight_decay)

            batch_count = len(batch_features)
            batch_correct = int(np.sum(binary_predictions(logits) == batch_labels))
            epoch_loss += loss * batch_count
            epoch_correct += batch_correct
            history["train_batch_loss"].append(loss)
            history["train_batch_accuracy"].append(batch_correct / batch_count)

        test_loss, test_accuracy = evaluate(model, test_data, test_labels, batch_size)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        LOGGER.info(
            "Epoch %d/%d | loss %.4f | train accuracy %.2f%% | test accuracy %.2f%%",
            epoch,
            epochs,
            epoch_loss / len(train_data),
            100 * epoch_correct / len(train_data),
            100 * test_accuracy,
        )

    return history


def plot_history(
    history: dict[str, list[float]],
    batches_per_epoch: int,
    output_path: Path | None,
    show: bool,
) -> None:
    """Plot batch-level training metrics and epoch-level test metrics."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epoch_positions = [
        (epoch + 1) * batches_per_epoch - 1
        for epoch in range(len(history["test_loss"]))
    ]
    figure, loss_axis = plt.subplots(figsize=(12, 7))
    loss_axis.plot(
        history["train_batch_loss"], color="tab:red", alpha=0.3, label="Train loss"
    )
    loss_axis.plot(
        epoch_positions, history["test_loss"], color="tab:red", label="Test loss"
    )
    loss_axis.set_xlabel("Optimization step")
    loss_axis.set_ylabel("Binary cross-entropy", color="tab:red")
    loss_axis.tick_params(axis="y", labelcolor="tab:red")

    accuracy_axis = loss_axis.twinx()
    accuracy_axis.plot(
        history["train_batch_accuracy"],
        color="tab:blue",
        alpha=0.3,
        label="Train accuracy",
    )
    accuracy_axis.plot(
        epoch_positions,
        history["test_accuracy"],
        color="tab:blue",
        label="Test accuracy",
    )
    accuracy_axis.set_ylabel("Accuracy", color="tab:blue")
    accuracy_axis.tick_params(axis="y", labelcolor="tab:blue")
    accuracy_axis.set_ylim(-0.01, 1.01)

    loss_axis.legend(loc="center left")
    accuracy_axis.legend(loc="center right")
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160)
        LOGGER.info("Saved training curves to %s", output_path)
    if show:
        plt.show()
    plt.close(figure)


def display_example(image: np.ndarray) -> None:
    """Display one flattened channel-first CIFAR image."""
    import matplotlib.pyplot as plt

    if image.shape != (3072,):
        raise ValueError("a flattened CIFAR image must contain 3,072 values")
    red, green, blue = (channel.reshape(32, 32) for channel in np.split(image, 3))
    plt.imshow(np.stack([red, green, blue], axis=2).astype(np.uint8))
    plt.axis("off")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a NumPy feed-forward network on a binary CIFAR dataset."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def positive_prefix(
    features: np.ndarray, labels: np.ndarray, limit: int | None, option: str
) -> tuple[np.ndarray, np.ndarray]:
    if limit is None:
        return features, labels
    if limit <= 0:
        raise ValueError(f"{option} must be positive")
    return features[:limit], labels[:limit]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    train_data, train_labels, test_data, test_labels = load_dataset(args.data)
    train_data, train_labels = positive_prefix(
        train_data, train_labels, args.max_train_samples, "--max-train-samples"
    )
    test_data, test_labels = positive_prefix(
        test_data, test_labels, args.max_test_samples, "--max-test-samples"
    )
    train_data, test_data = normalize_features(train_data, test_data)

    rng = np.random.default_rng(args.seed)
    model = FeedForwardNeuralNetwork(
        train_data.shape[1],
        1,
        args.hidden_width,
        args.layers,
        rng,
    )
    history = train(
        model,
        train_data,
        train_labels,
        test_data,
        test_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        step_size=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        rng=rng,
    )

    final_loss, final_accuracy = evaluate(
        model, test_data, test_labels, args.batch_size
    )
    LOGGER.info(
        "Final test loss %.4f | final test accuracy %.2f%%",
        final_loss,
        final_accuracy * 100,
    )
    if args.plot is not None or args.show:
        plot_history(
            history,
            sum(1 for _ in batches(len(train_data), args.batch_size)),
            args.plot,
            args.show,
        )


if __name__ == "__main__":
    main()
