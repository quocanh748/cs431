# Efficient ViT with Depth-Wise Convolution

This repository is a reproduction-oriented fork based on [https://github.com/ZTX-100/Efficient_ViT_with_DW](https://github.com/ZTX-100/Efficient_ViT_with_DW). It keeps the depth-wise convolution idea for Vision Transformer-style models and adapts the code to the current workspace and datasets used for reproduction.

## Overview

The codebase focuses on image classification with lightweight ViT variants that inject depth-wise convolution for better local feature modeling. The main entrypoint is [`main.py`](main.py), which handles distributed training, validation, evaluation-only runs, throughput measurement, checkpointing, and resume/pretrained loading.

Supported model families in this fork are:

- ViT
- ViT-S
- CaiT-XXS
- Swin-Tiny

Supported datasets in the current code are:

- CIFAR-10
- CIFAR-100
- STL-10
- Tiny-ImageNet
- Flowers102
- Caltech101

## Repository Layout

- [`main.py`](main.py): training and evaluation entrypoint
- [`config.py`](config.py): YAML and CLI config handling
- [`models/`](models): model definitions and factory
- [`data/`](data): dataset loading, transforms, samplers, and caching helpers
- [`configs/`](configs): ready-to-run experiment configs
- [`kernels/window_process/`](kernels/window_process): optional fused window process kernel for Swin
- [`figures/`](figures): architecture illustrations used by the original paper

## Requirements

The project depends on the PyTorch ecosystem plus the libraries used by the upstream Swin Transformer codebase.

Typical packages include:

- `torch`
- `torchvision`
- `timm`
- `yacs`
- `fvcore`
- `pyyaml`

Optional acceleration features:

- `apex` for fused layer norm
- the fused window process kernel under `kernels/window_process/` for Swin acceleration

If you are installing from scratch, follow the PyTorch install that matches your CUDA setup first, then install the Python dependencies used by the repo.

## Data Preparation

The current loader behavior is implemented in [`data/build.py`](data/build.py).

- CIFAR-10, CIFAR-100, STL-10, Flowers102, and Caltech101 can be downloaded automatically by `torchvision` when `--data-path` points to a writable location.
- Tiny-ImageNet expects a folder layout with `train/` and `val/` under the data path.
- For STL-10 on Kaggle, use a writable path such as `/kaggle/working` if you want automatic download.
- The code also contains support for ImageNet-style folder layouts and zipped ImageNet-style datasets, but the current reproduction configs are centered on the small-dataset classification setups above.

## Training

The code is written for distributed execution. Even on a single GPU, run it through `torchrun` or an equivalent distributed launcher.

Example for ViT on CIFAR-10:

```bash
torchrun --nproc_per_node=1 main.py \
  --cfg configs/vit/vit_tiny_16_224_cifar10.yaml \
  --data-path /path/to/cifar10 \
  --batch-size 128 \
  --output output \
  --tag vit_cifar10
```

Other ready-to-use config files are available under `configs/`:

- `configs/vit/`
- `configs/cait/`
- `configs/swin/`

You can override YAML options from the command line with `--opts`, for example:

```bash
torchrun --nproc_per_node=4 main.py \
  --cfg configs/vit/vit_tiny_16_224_cifar100.yaml \
  --data-path /path/to/cifar100 \
  --batch-size 128 \
  --opts TRAIN.EPOCHS 300 TRAIN.WEIGHT_DECAY 0.05
```

## Evaluation and Resume

- Use `--resume /path/to/checkpoint.pth` to continue training from a checkpoint.
- Use `--pretrained /path/to/checkpoint.pth` to load weights before evaluation or fine-tuning.
- Add `--eval` to run evaluation only after loading a checkpoint or pretrained weights.
- Add `--throughput` to measure throughput only.

The default output path is `output/<model_name>/<tag>` unless you override `--output` and `--tag`.

## Notes

- The configs in this fork are tuned for the current reproduction workflow and small-dataset experiments rather than the full original paper scope.
- Some data-loading behavior has been adjusted for local and Kaggle-style paths.
- The original implementation ideas come from the upstream repository and the referenced Swin Transformer and `vit-pytorch` codebases.

## Citation

If you use this repository, please cite the original paper:

```bibtex
@article{zhang2025depth,
  title={Depth-wise convolutions in vision transformers for efficient training on small datasets},
  author={Zhang, Tianxiao and Xu, Wenju and Luo, Bo and Wang, Guanghui},
  journal={Neurocomputing},
  volume={617},
  pages={128998},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgements

This fork is based on [Efficient_ViT_with_DW](https://github.com/ZTX-100/Efficient_ViT_with_DW) and reuses ideas and code patterns from the upstream Swin Transformer and `vit-pytorch` projects.

