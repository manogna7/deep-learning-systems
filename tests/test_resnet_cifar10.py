import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from resnet_cifar10 import (
    BasicBlock,
    ResNet14,
    build_optimizer,
    evaluate,
    seed_everything,
    train_one_epoch,
)


class ArchitectureTests(unittest.TestCase):
    def test_projection_shortcut_changes_channels_and_resolution(self) -> None:
        block = BasicBlock(16, 32, stride=2).eval()
        with torch.no_grad():
            output = block(torch.randn(2, 16, 32, 32))
        self.assertEqual(output.shape, (2, 32, 16, 16))

    def test_resnet_produces_ten_class_logits(self) -> None:
        model = ResNet14().eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(logits.shape, (2, 10))


class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(17)
        features = torch.randn(16, 3, 32, 32)
        labels = torch.arange(16) % 10
        self.loader = DataLoader(TensorDataset(features, labels), batch_size=8)
        self.device = torch.device("cpu")
        self.criterion = nn.CrossEntropyLoss()

    def test_training_step_updates_parameters_and_returns_metrics(self) -> None:
        model = ResNet14().to(self.device)
        optimizer = build_optimizer("sgd", model, 0.01, 0.0)
        initial = model.conv1.weight.detach().clone()
        metrics = train_one_epoch(
            model, self.loader, self.criterion, optimizer, self.device
        )
        self.assertFalse(torch.equal(initial, model.conv1.weight))
        self.assertGreater(metrics.loss, 0.0)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)

    def test_evaluation_is_repeatable(self) -> None:
        model = ResNet14().to(self.device)
        first = evaluate(model, self.loader, self.criterion, self.device)
        second = evaluate(model, self.loader, self.criterion, self.device)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
