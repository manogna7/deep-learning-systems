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

The team project extended image classification to object detection with YOLOv8 and the RDD2022 dataset. `road_damage_detection.py` now provides the complete executable workflow: COCO-to-YOLO annotation conversion, deterministic data splitting, image and label validation, exact-duplicate leakage detection, transfer learning, mixed-precision GPU training, optimizer experiments, test-set evaluation, structured inference output, and deployment export.

Three named profiles preserve the original SGD and AdamW experiments while allowing every hyperparameter to be overridden. Evaluation produces confusion matrices, precision-recall and F1 curves, and mAP at multiple IoU thresholds through Ultralytics. Prediction saves annotated media, YOLO labels with confidence values, and JSON Lines records for downstream systems.

## Repository layout

| Path | Purpose |
| --- | --- |
| `feedforward.py` | NumPy neural-network layers, training loop, evaluation, and CLI |
| `resnet_cifar10.py` | PyTorch residual network, data pipeline, training, and checkpointing |
| `road_damage_detection.py` | Road-damage dataset preparation, validation, YOLO training, evaluation, inference, and export CLI |
| `data/cifar-2class.npz` | Balanced two-class CIFAR data for the NumPy model |
| `tests/` | Numerical-gradient, behavior, architecture, data-integrity, and training-step tests |
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

## Road-damage detection workflow

The large RDD2022 images are intentionally not versioned. Convert a COCO annotation export into the standard YOLO directory layout; the command refuses to overwrite an existing destination:

```powershell
python road_damage_detection.py prepare `
  --annotations C:\datasets\rdd2022\annotations.json `
  --images-dir C:\datasets\rdd2022\images `
  --output-dir data\road-damage `
  --enhance-contrast --median-filter-size 3
```

Contrast enhancement and median denoising are optional, deterministic preprocessing steps from the original project; omit both flags to preserve source pixels exactly. The generated manifest records the chosen preprocessing, annotation checksum, split seed, source filenames, class distribution, skipped images, and rejected boxes. Re-run the preflight audit at any time:

```powershell
python road_damage_detection.py validate-data --dataset-root data\road-damage
python road_damage_detection.py profiles
```

Train the original experiment profiles with pretrained YOLOv8 weights. Use `--model yolov8l.pt` when the larger variant and available GPU memory are appropriate:

```powershell
python road_damage_detection.py train --profile sgd-30 --device 0
python road_damage_detection.py train --profile adamw-40 --device 0
python road_damage_detection.py train --profile sgd-50 --model yolov8l.pt --device 0
```

Evaluate the selected checkpoint only after model selection, then run inference or export it for a deployment runtime:

```powershell
python road_damage_detection.py evaluate `
  --weights artifacts\road-damage\runs\sgd-30\weights\best.pt `
  --split test --device 0

python road_damage_detection.py predict `
  --weights artifacts\road-damage\runs\sgd-30\weights\best.pt `
  --source C:\datasets\road-images --device 0

python road_damage_detection.py export `
  --weights artifacts\road-damage\runs\sgd-30\weights\best.pt `
  --format onnx --dynamic
```

Training checkpoints, diagnostics, predictions, converted data, and exported models are ignored by Git because they are reproducible build artifacts. Exact metrics can vary with the source dataset revision, accelerator, driver stack, and dependency versions even when deterministic execution is enabled.
