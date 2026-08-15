import tempfile
import unittest
from pathlib import Path

import numpy as np

from feedforward import (
    FeedForwardNeuralNetwork,
    LinearLayer,
    ReLU,
    SigmoidCrossEntropy,
    batches,
    evaluate,
    load_dataset,
    train,
)


class SigmoidCrossEntropyTests(unittest.TestCase):
    def test_forward_is_stable_for_extreme_logits(self) -> None:
        loss = SigmoidCrossEntropy().forward(
            np.array([[1000.0], [-1000.0]]), np.array([[1.0], [0.0]])
        )
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(loss, 0.0, places=12)

    def test_backward_matches_mean_loss_numerical_gradient(self) -> None:
        logits = np.array([[0.3], [-1.1], [2.0]])
        labels = np.array([[1.0], [0.0], [1.0]])
        loss_fn = SigmoidCrossEntropy()
        loss_fn.forward(logits, labels)
        analytic = loss_fn.backward()
        numerical = np.zeros_like(logits)
        epsilon = 1e-6

        for index in np.ndindex(logits.shape):
            higher = logits.copy()
            lower = logits.copy()
            higher[index] += epsilon
            lower[index] -= epsilon
            numerical[index] = (
                SigmoidCrossEntropy().forward(higher, labels)
                - SigmoidCrossEntropy().forward(lower, labels)
            ) / (2 * epsilon)

        np.testing.assert_allclose(analytic, numerical, atol=1e-9)


class LayerTests(unittest.TestCase):
    def test_relu_masks_nonpositive_inputs(self) -> None:
        layer = ReLU()
        output = layer.forward(np.array([[-2.0, 0.0, 3.0]]))
        gradient = layer.backward(np.ones((1, 3)))
        np.testing.assert_array_equal(output, [[0.0, 0.0, 3.0]])
        np.testing.assert_array_equal(gradient, [[0.0, 0.0, 1.0]])

    def test_linear_weight_gradient_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(7)
        layer = LinearLayer(3, 2, rng)
        inputs = rng.normal(size=(4, 3))
        upstream = rng.normal(size=(4, 2))
        layer.forward(inputs)
        layer.backward(upstream)
        analytic = layer.grad_weights.copy()
        numerical = np.zeros_like(layer.weights)
        epsilon = 1e-6

        for index in np.ndindex(layer.weights.shape):
            original = layer.weights[index]
            layer.weights[index] = original + epsilon
            higher = np.sum(layer.forward(inputs) * upstream)
            layer.weights[index] = original - epsilon
            lower = np.sum(layer.forward(inputs) * upstream)
            layer.weights[index] = original
            numerical[index] = (higher - lower) / (2 * epsilon)

        np.testing.assert_allclose(analytic, numerical, atol=1e-9)


class EvaluationTests(unittest.TestCase):
    def test_logit_threshold_is_zero(self) -> None:
        class FixedModel:
            def forward(self, features: np.ndarray) -> np.ndarray:
                return features

        features = np.array([[0.2], [-0.2]])
        labels = np.array([[1], [0]])
        _, accuracy = evaluate(FixedModel(), features, labels, batch_size=1)
        self.assertEqual(accuracy, 1.0)

    def test_evaluation_weights_partial_batches_by_example_count(self) -> None:
        class FixedModel:
            def forward(self, features: np.ndarray) -> np.ndarray:
                return features

        features = np.array([[1.0], [1.0], [-1.0]])
        labels = np.array([[1], [1], [1]])
        _, accuracy = evaluate(FixedModel(), features, labels, batch_size=2)
        self.assertAlmostEqual(accuracy, 2 / 3)

    def test_batch_iterator_keeps_the_remainder(self) -> None:
        self.assertEqual(
            [(item.start, item.stop) for item in batches(5, 2)],
            [(0, 2), (2, 4), (4, 5)],
        )


class DatasetTests(unittest.TestCase):
    def test_loader_reads_non_pickle_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            np.savez_compressed(
                path,
                train_data=np.ones((2, 3), dtype=np.uint8),
                train_labels=np.array([[0], [1]], dtype=np.uint8),
                test_data=np.ones((2, 3), dtype=np.uint8),
                test_labels=np.array([[1], [0]], dtype=np.uint8),
            )
            arrays = load_dataset(path)

        self.assertEqual(arrays[0].dtype, np.float32)
        self.assertEqual(arrays[1].dtype, np.uint8)


class TrainingTests(unittest.TestCase):
    def test_small_network_learns_linearly_separable_data(self) -> None:
        rng = np.random.default_rng(11)
        features = rng.normal(size=(80, 2))
        labels = (features[:, :1] + features[:, 1:] > 0).astype(np.uint8)
        model = FeedForwardNeuralNetwork(2, 1, 8, 2, rng)
        initial_loss, _ = evaluate(model, features, labels, batch_size=16)
        train(
            model,
            features,
            labels,
            features,
            labels,
            epochs=30,
            batch_size=16,
            step_size=0.05,
            momentum=0.0,
            weight_decay=0.0,
            rng=rng,
        )
        final_loss, final_accuracy = evaluate(model, features, labels, batch_size=16)
        self.assertLess(final_loss, initial_loss)
        self.assertGreater(final_accuracy, 0.9)


if __name__ == "__main__":
    unittest.main()
