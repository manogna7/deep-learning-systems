"""Train a compact residual network on CIFAR-10."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float


class BasicBlock(nn.Module):
    """Two convolutions with an identity or projected residual shortcut."""

    def __init__(
        self, input_channels: int, output_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.shortcut: nn.Module
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = torch.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return torch.relu(outputs + residual)


class ResNet14(nn.Module):
    """A 14-layer residual network sized for 32-by-32 images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.current_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, block_count=2, stride=1)
        self.stage2 = self._make_stage(32, block_count=2, stride=2)
        self.stage3 = self._make_stage(64, block_count=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
        self._initialize_parameters()

    def _make_stage(
        self, output_channels: int, block_count: int, stride: int
    ) -> nn.Sequential:
        blocks = [BasicBlock(self.current_channels, output_channels, stride)]
        self.current_channels = output_channels
        blocks.extend(
            BasicBlock(self.current_channels, output_channels)
            for _ in range(block_count - 1)
        )
        return nn.Sequential(*blocks)

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.relu(self.bn1(self.conv1(inputs)))
        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.pool(outputs)
        return self.classifier(torch.flatten(outputs, 1))


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch and request deterministic kernels when available."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_data_loaders(
    data_directory: Path,
    *,
    batch_size: int,
    workers: int,
    validation_fraction: float,
    seed: int,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build augmented training and deterministic validation/test loaders."""
    from torchvision import datasets, transforms

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    mean = (0.4914, 0.4822, 0.4465)
    standard_deviation = (0.2470, 0.2435, 0.2616)
    training_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, standard_deviation)]
    )

    augmented_training_data = datasets.CIFAR10(
        root=data_directory,
        train=True,
        transform=training_transform,
        download=True,
    )
    evaluation_training_data = datasets.CIFAR10(
        root=data_directory,
        train=True,
        transform=evaluation_transform,
        download=True,
    )
    test_data = datasets.CIFAR10(
        root=data_directory,
        train=False,
        transform=evaluation_transform,
        download=True,
    )

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(
        len(augmented_training_data), generator=generator
    ).tolist()
    validation_count = int(len(shuffled_indices) * validation_fraction)
    validation_indices = shuffled_indices[:validation_count]
    training_indices = shuffled_indices[validation_count:]

    if max_train_samples is not None:
        training_indices = training_indices[:max_train_samples]
    if max_validation_samples is not None:
        validation_indices = validation_indices[:max_validation_samples]
    test_indices = list(range(len(test_data)))
    if max_test_samples is not None:
        test_indices = test_indices[:max_test_samples]

    if not training_indices or not validation_indices or not test_indices:
        raise ValueError("every data split must contain at least one example")

    loader_arguments = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        Subset(augmented_training_data, training_indices),
        shuffle=True,
        generator=generator,
        **loader_arguments,
    )
    validation_loader = DataLoader(
        Subset(evaluation_training_data, validation_indices),
        shuffle=False,
        **loader_arguments,
    )
    test_loader = DataLoader(
        Subset(test_data, test_indices), shuffle=False, **loader_arguments
    )
    return train_loader, validation_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Metrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = len(inputs)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    return Metrics(total_loss / total_examples, total_correct / total_examples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        batch_size = len(inputs)
        total_loss += criterion(logits, labels).item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    return Metrics(total_loss / total_examples, total_correct / total_examples)


def build_optimizer(
    name: str, model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {name}")


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epochs: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    writer: SummaryWriter | None = None,
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor], float]:
    """Train against a validation split and retain the best model state."""
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_metrics = evaluate(model, validation_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_metrics.loss)
        history["train_accuracy"].append(train_metrics.accuracy)
        history["validation_loss"].append(validation_metrics.loss)
        history["validation_accuracy"].append(validation_metrics.accuracy)
        if writer is not None:
            writer.add_scalars(
                "loss",
                {"train": train_metrics.loss, "validation": validation_metrics.loss},
                epoch,
            )
            writer.add_scalars(
                "accuracy",
                {
                    "train": train_metrics.accuracy,
                    "validation": validation_metrics.accuracy,
                },
                epoch,
            )

        if validation_metrics.accuracy > best_accuracy:
            best_accuracy = validation_metrics.accuracy
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

        LOGGER.info(
            "Epoch %d/%d | train loss %.4f | validation loss %.4f | "
            "train accuracy %.2f%% | validation accuracy %.2f%%",
            epoch,
            epochs,
            train_metrics.loss,
            validation_metrics.loss,
            train_metrics.accuracy * 100,
            validation_metrics.accuracy * 100,
        )

    return history, best_state, best_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-14 on CIFAR-10.")
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "cifar10"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "resnet14-best.pt",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--cosine-schedule", action="store_true")
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError(
            "epochs and batch size must be positive; workers cannot be negative"
        )
    for limit in (
        args.max_train_samples,
        args.max_validation_samples,
        args.max_test_samples,
    ):
        if limit is not None and limit <= 0:
            raise ValueError("sample limits must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 0.001 if args.optimizer == "adam" else 0.1
    LOGGER.info("Using %s with %s optimizer", device, args.optimizer)

    train_loader, validation_loader, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        max_test_samples=args.max_test_samples,
    )
    model = ResNet14().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model, learning_rate, args.weight_decay)
    scheduler = None
    if args.cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    writer = None
    if args.tensorboard_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(args.tensorboard_dir)

    try:
        history, best_state, best_accuracy = fit(
            model,
            train_loader,
            validation_loader,
            criterion,
            optimizer,
            device=device,
            epochs=args.epochs,
            scheduler=scheduler,
            writer=writer,
        )
    finally:
        if writer is not None:
            writer.close()

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "optimizer": args.optimizer,
            "learning_rate": learning_rate,
            "validation_accuracy": best_accuracy,
            "test_accuracy": test_metrics.accuracy,
            "history": history,
        },
        args.checkpoint,
    )
    LOGGER.info(
        "Best validation accuracy %.2f%% | test loss %.4f | test accuracy %.2f%%",
        best_accuracy * 100,
        test_metrics.loss,
        test_metrics.accuracy * 100,
    )
    LOGGER.info("Saved checkpoint to %s", args.checkpoint)


if __name__ == "__main__":
    main()
