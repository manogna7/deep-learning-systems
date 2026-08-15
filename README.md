# Deep Learning Systems

Projects tracing a progression from the mathematics behind optimization and backpropagation to neural networks implemented from scratch, residual image classifiers in PyTorch, and real-time object detection.

## Learning progression

### Mathematical foundations

The written work established the mechanics used throughout the implementations:

- Gradients, Hessians, stationary points, and curvature-based classification
- Positive-definite matrices and quadratic forms
- Matrix calculus and the chain rule for neural-network layers
- High-dimensional geometry and the curse of dimensionality

### Neural network from scratch

`feedforward.py` implements a binary CIFAR classifier using NumPy only. It includes vectorized dense layers, ReLU activations, numerically stable sigmoid cross-entropy, manual backpropagation, He initialization, mini-batch training, momentum, L2 regularization, deterministic experiments, and example-weighted evaluation.

The accompanying two-class dataset contains 10,000 training images and 2,000 test images. Each image is represented by 3,072 flattened RGB values, and both splits are balanced.

### Residual image classification

`resnet_cifar10.py` implements a ResNet-14 classifier for CIFAR-10 with PyTorch. It separates stochastic training augmentation from deterministic evaluation preprocessing, creates independent optimizer experiments, uses adaptive pooling, tracks validation metrics, optionally writes TensorBoard events, and saves the best checkpoint.

The original experiment explored Adam and momentum-based SGD, residual shortcuts, batch normalization, data augmentation, GPU training, and TensorBoard monitoring.

### Architectures, optimization, and interpretation

The broader material connected implementation choices to modern deep-learning systems:

- Convolutional inductive bias, padding, stride, pooling, receptive fields, and residual learning
- Computational graphs, topological execution, reverse-mode automatic differentiation, and PyTorch autograd
- RMSProp, Adam, exponential moving averages, learning-rate schedules, dropout, ensembles, and batch/group normalization
- Feature visualization, input-gradient sensitivity, adversarial examples, network dissection, and mask-based explanations
- Encoder-decoder attention, queries/keys/values, scaled dot-product and multi-head attention, layer normalization, and positional encoding
- GPT-style language models, Vision Transformers, Swin Transformers, DETR, and Segment Anything
- Autoencoders, variational autoencoders, GANs, diffusion models, LPIPS, and Fréchet Inception Distance
- Point-cloud representations and graph neural networks with message passing, edge features, attention, GraphSAGE, and PinSAGE

### Real-time road-damage detection

The team project extended image classification to object detection with YOLOv8 and the RDD2022 dataset. The work covered COCO-to-YOLO annotation conversion, transfer learning, mixed-precision GPU training, Mosaic and MixUp augmentation, optimizer comparison, confusion-matrix analysis, precision-recall and F1 curves, mean average precision at multiple IoU thresholds, inference latency, false-positive analysis, and edge-deployment tradeoffs.

## Repository layout

| Path | Purpose |
| --- | --- |
| `feedforward.py` | NumPy neural-network layers, training loop, evaluation, and CLI |
| `resnet_cifar10.py` | PyTorch residual network, data pipeline, training, and checkpointing |
| `data/cifar-2class.npz` | Balanced two-class CIFAR data for the NumPy model |
| `tests/` | Numerical-gradient, behavior, architecture, and training-step tests |
| `learning-materials/` | Course-number-free lecture references covering the concepts summarized above |

## Run locally

Create an isolated environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Train the original four-layer fully connected model:

```powershell
python feedforward.py
```

Use smaller settings for a quick smoke test or vary the exposed hyperparameters for controlled experiments:

```powershell
python feedforward.py --epochs 1 --layers 2 --hidden-width 32 --max-train-samples 1024 --max-test-samples 512
```

Train ResNet-14 with a fresh model for each optimizer configuration:

```powershell
python resnet_cifar10.py --optimizer adam --epochs 30
python resnet_cifar10.py --optimizer sgd --epochs 30 --learning-rate 0.1 --cosine-schedule
```

CIFAR-10 is downloaded into the ignored `data/cifar10` directory. Add `--tensorboard-dir artifacts/tensorboard` to record metrics, and run `python resnet_cifar10.py --help` or `python feedforward.py --help` for every option.
